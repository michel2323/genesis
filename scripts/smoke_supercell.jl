# Bring-up ladder for Breeze.jl/examples/splitting_supercell.jl on oneAPI.
#
# Same physical setup as the example (profiles, Kessler, WENO(order=9), Float32, halo 5,
# TetensFormula) on a 32x32x16 grid, no plotting.
#
# Environment variables:
#   ARCH  = cpu | oneapi                      (default cpu)
#   RUNG  = 1..6 | all | comma list "1,3,4"   (default all)
#   NSTEPS = number of time steps for rungs 1,3,4 (default 10; rung 1 uses min(NSTEPS,3))
#   KARGS = off | onfail | report | skip      (default onfail; only meaningful on oneapi)
#       onfail: when a `launch!` throws, print the per-argument byte table of that kernel
#       report: print the table of every launch whose arguments exceed KARGS_LIMIT (then launch)
#       skip:   like report, but do not launch kernels over the limit (lets the run proceed,
#               with wrong numbers, to find every oversized kernel)
#       all:    print the table for every distinct kernel (whatever its size), and launch
#   KARGS_LIMIT = byte limit (default 2048)
#   CONSTRUCT_ONLY = 1   rungs 3,4: build model and set!, do not time step
#   OUTDIR = where rung 6 writes final fields (default scripts/output)
#
# Run from the repository root with:  julia +1.12 --project=. scripts/smoke_supercell.jl

using Breeze
using Breeze: DCMIP2016KesslerMicrophysics, TetensFormula
using Breeze.Thermodynamics: hydrostatic_density, hydrostatic_temperature,
                             pressure_balanced_density
using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: znodes
using JLD2
using Printf
using Logging

# log to stdout so that log lines interleave with println output in redirected logs
global_logger(ConsoleLogger(stdout))

const ARCH_NAME = lowercase(get(ENV, "ARCH", "cpu"))
const ONEAPI = ARCH_NAME == "oneapi"
ONEAPI && @eval using oneAPI
import KernelAbstractions as KA

const NSTEPS = parse(Int, get(ENV, "NSTEPS", "10"))
const KARGS_MODE = lowercase(get(ENV, "KARGS", "onfail"))
const KARGS_LIMIT = parse(Int, get(ENV, "KARGS_LIMIT", "2048"))
const CONSTRUCT_ONLY = get(ENV, "CONSTRUCT_ONLY", "0") == "1"
const OUTDIR = get(ENV, "OUTDIR", joinpath(@__DIR__, "output"))

rung_spec = lowercase(get(ENV, "RUNG", "all"))
const RUNGS = rung_spec == "all" ? collect(1:6) : parse.(Int, split(rung_spec, ","))

arch() = ONEAPI ? GPU(oneAPIBackend()) : CPU()

sync() = ONEAPI ? oneAPI.synchronize() : nothing

#####
##### Kernel-argument size measurement (oneAPI only)
#####

if ONEAPI
    @eval begin
        import oneAPI.oneAPIKernels: oneAPIBackend
        import Oceananigans.Utils as OU
        import Oceananigans.Architectures as OA

        # Bytes oneAPI passes for one kernel parameter of type `T` (see oneAPI.call_arguments:
        # ghost/const types are dropped, non-isbits ones become a `Ptr{Any}`).
        function param_bytes(T)
            (oneAPI.isghosttype(T) || Core.Compiler.isconstType(T)) && return 0
            return isbitstype(T) ? sizeof(T) : sizeof(Ptr{Any})
        end

        # Reproduce KA 0.10's launch preparation to get the real parameter types:
        # (KernelState, ctx, converted args...)
        function kernel_param_table(loop!, worksize, device_args)
            ndrange = OU.launch_ndrange(worksize)
            obj = KA.sizes_only(loop!)
            skip, nd, wgs, iterspace = KA.launch_setup(obj, ndrange, nothing)
            launch = KA.select_launch(obj, wgs, iterspace)
            subgroups = KA.kernel_subgroups(loop!)
            ctx = KA.mkcontext(obj, nd, iterspace, launch; subgroups)
            backend = KA.backend(loop!)
            _, tt = KA.prepare_arguments(backend, loop!.f, ctx, device_args)
            rows = Tuple{String, Int, String}[]
            push!(rows, ("<KernelState>", param_bytes(oneAPI.KernelState), "oneAPI.KernelState"))
            push!(rows, ("<kernel function>", param_bytes(typeof(loop!.f)), string(typeof(loop!.f))))
            for (i, T) in enumerate(tt.parameters)
                name = i == 1 ? "<KA ctx>" : "arg $(i - 1)"
                push!(rows, (name, param_bytes(T), string(T)))
            end
            return rows
        end

        const reported_kernels = Set{String}()

        function print_param_table(kname, rows; header="")
            total = sum(r[2] for r in rows)
            println("\n==== kernel-argument sizes: $kname $header")
            println(@sprintf("     total = %d bytes (limit %d) %s", total, KARGS_LIMIT,
                             total > KARGS_LIMIT ? "OVER LIMIT" : "ok"))
            for (name, bytes, T) in sort(rows; by = r -> -r[2])
                Tshort = length(T) > 140 ? first(T, 140) * "…" : T
                println(@sprintf("   %6d  %-18s %s", bytes, name, Tshort))
            end
            # finer breakdown for big tuple / NamedTuple / struct arguments
            return total
        end

        function kernel_name(loop!)
            f = loop!.f
            return string(nameof(typeof(f)))
        end

        # Per-field breakdown of a large argument, one level deep
        function breakdown(T; depth=1, indent="        ", minbytes=64)
            (isbitstype(T) && fieldcount(T) > 0) || return
            for (fname, FT) in zip(fieldnames(T), fieldtypes(T))
                b = param_bytes(FT)
                b < minbytes && continue
                s = string(FT); s = length(s) > 110 ? first(s, 110) * "…" : s
                println(@sprintf("%s%6d  .%s :: %s", indent, b, fname, s))
                depth > 1 && breakdown(FT; depth=depth-1, indent=indent * "    ", minbytes)
            end
        end

        function measure_and_report(loop!, worksize, device_args, loc; force=false)
            kname = kernel_name(loop!)
            rows = try
                kernel_param_table(loop!, worksize, device_args)
            catch err
                println("[kargs] could not measure $kname: ", sprint(showerror, err))
                return 0
            end
            total = sum(r[2] for r in rows)
            key = kname * "@" * string(hash(map(r -> r[3], rows)))
            if (force || total > KARGS_LIMIT || KARGS_MODE == "all") && !(key in reported_kernels)
                push!(reported_kernels, key)
                print_param_table(kname, rows; header=loc)
                for (name, bytes, _) in rows
                    if bytes >= 256 && startswith(name, "arg")
                        i = parse(Int, split(name)[2])
                        println("     breakdown of $name ($bytes B):")
                        breakdown(typeof(KA.argconvert(loop!, device_args[i])); depth=2)
                    end
                end
            end
            return total
        end

        function OU._launch!(arch::OA.GPU{<:oneAPIBackend}, grid, workspec, kernel!, kernel_args::Vararg{Any, N};
                             exclude_periphery = false, reduced_dimensions = ()) where N
            workspec = OU.possibly_load_active_cells_map(grid, workspec, exclude_periphery)
            location = Oceananigans.instantiated_location(first(kernel_args))
            loop!, worksize = OU.configure_kernel(arch, grid, workspec, kernel!, Val(exclude_periphery);
                                                  location, reduced_dimensions)
            length(worksize) > 0 || return nothing
            device_args = OA.convert_to_device(arch, kernel_args)

            if KARGS_MODE in ("report", "skip", "all")
                total = measure_and_report(loop!, worksize, device_args, "(pre-launch)")
                if KARGS_MODE == "skip" && total > KARGS_LIMIT
                    println("[kargs] SKIPPED launch of $(kernel_name(loop!)) ($total B)")
                    return nothing
                end
            end

            if KARGS_MODE == "off"
                loop!(device_args...; ndrange = OU.launch_ndrange(worksize))
            else
                try
                    loop!(device_args...; ndrange = OU.launch_ndrange(worksize))
                catch err
                    println("[kargs] launch of $(kernel_name(loop!)) threw $(typeof(err)); measuring its arguments")
                    measure_and_report(loop!, worksize, device_args, "(failed launch)"; force=true)
                    rethrow()
                end
            end
            return nothing
        end
    end
end

#####
##### Physical setup (from the example)
#####

Oceananigans.defaults.FloatType = Float32

Nx, Ny, Nz = 32, 32, 16
Lx, Ly, Lz = 168kilometers, 168kilometers, 20kilometers

constants = ThermodynamicConstants(saturation_vapor_pressure = TetensFormula())

p₀  = 100000
pˢᵗ = 100000
θ₀     = 300
θᵖ     = 343
zᵖ     = 12000
Tᵖ     = 213
qᵛ_max = 0.014
zˢ = 5kilometers
uˢ = 30
uᶜ = 15
g   = constants.gravitational_acceleration
cᵖᵈ = constants.dry_air.heat_capacity

function θ_background(z)
    θᵗ = θ₀ + (θᵖ - θ₀) * (z / zᵖ)^(5/4)
    θˢ = θᵖ * exp(g / (cᵖᵈ * Tᵖ) * (z - zᵖ))
    return (z ≤ zᵖ) * θᵗ + (z > zᵖ) * θˢ
end

function qᵛ_bg(z)
    ℋ = (1 - 3/4 * (z / zᵖ)^(5/4)) * (z ≤ zᵖ) + 1/4 * (z > zᵖ)
    T = hydrostatic_temperature(z, p₀, θ_background, pˢᵗ, constants)
    ρ = hydrostatic_density(z, p₀, θ_background, pˢᵗ, constants)
    qᵛ⁺ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
    return min(ℋ * qᵛ⁺, qᵛ_max)
end

function u_background(z)
    uˡ = uˢ * (z / zˢ) - uᶜ
    uᵗ = (-4/5 + 3 * (z / zˢ) - 5/4 * (z / zˢ)^2) * uˢ - uᶜ
    uᵘ = uˢ - uᶜ
    return (z < (zˢ - 1000)) * uˡ +
           (abs(z - zˢ) ≤ 1000) * uᵗ +
           (z > (zˢ + 1000)) * uᵘ
end

Δθ  = 3
rᵇʰ = 10kilometers
rᵇᵛ = 1500
zᵇ  = 1500
xᵇ  = Lx / 2
yᵇ  = Ly / 2

function θᵢ(x, y, z)
    θ̄ = θ_background(z)
    r = sqrt((x - xᵇ)^2 + (y - yᵇ)^2)
    R = sqrt((r / rᵇʰ)^2 + ((z - zᵇ) / rᵇᵛ)^2)
    θ′ = ifelse(R < 1, Δθ * cos(π * R / 2)^2, 0.0)
    return θ̄ + θ′
end

uᵢ(x, y, z) = u_background(z)

#####
##### Rung machinery
#####

const RESULTS = Tuple{Int, String, Symbol, Float64, String}[]  # rung, label, status, seconds, note

function rung(f, n, label)
    println("\n", "="^100)
    println("RUNG $n [$ARCH_NAME]: $label")
    println("="^100)
    flush(stdout)
    t0 = time_ns()
    try
        note = f()
        sync()
        dt = 1e-9 * (time_ns() - t0)
        push!(RESULTS, (n, label, :passed, dt, string(something(note, ""))))
        @info @sprintf("RUNG %d PASSED in %.1f s", n, dt)
        return true
    catch err
        dt = 1e-9 * (time_ns() - t0)
        msg = sprint(showerror, err)
        push!(RESULTS, (n, label, :failed, dt, first(split(msg, '\n'))))
        println("RUNG $n FAILED after $(round(dt, digits=1)) s:")
        showerror(stdout, err, catch_backtrace())
        println()
        return false
    finally
        flush(stdout)
    end
end

function make_grid()
    return RectilinearGrid(arch(),
                           size = (Nx, Ny, Nz),
                           x = (0, Lx), y = (0, Ly), z = (0, Lz),
                           halo = (5, 5, 5),
                           topology = (Periodic, Periodic, Bounded))
end

# Lazily built shared state
const STATE = Dict{Symbol, Any}()

function initial_fields!()
    haskey(STATE, :grid) && return STATE
    grid = make_grid()
    qᵛ_column = Field{Nothing, Nothing, Center}(grid)
    set!(qᵛ_column, qᵛ_bg)
    θ_background_column = Field{Nothing, Nothing, Center}(grid)
    set!(θ_background_column, θ_background)
    θ_initial_field = CenterField(grid)
    set!(θ_initial_field, θᵢ)
    STATE[:grid] = grid
    STATE[:qᵛ_column] = qᵛ_column
    STATE[:θ_background_column] = θ_background_column
    STATE[:θ_initial_field] = θ_initial_field
    return STATE
end

microphysics() = DCMIP2016KesslerMicrophysics()
advection() = WENO(order=9)

function build_anelastic()
    S = initial_fields!()
    grid = S[:grid]
    reference_state = ReferenceState(grid, constants; base_pressure = p₀, potential_temperature = θ₀)
    dynamics = AnelasticDynamics(reference_state)
    model = AtmosphereModel(grid; dynamics, microphysics = microphysics(), advection = advection(),
                            thermodynamic_constants = constants)
    return model
end

function set_anelastic!(model)
    S = initial_fields!()
    set!(model, θ=S[:θ_initial_field], qᵛ=S[:qᵛ_column], u=uᵢ)
    return model
end

function build_compressible()
    S = initial_fields!()
    grid = S[:grid]
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                    base_pressure = p₀,
                                    standard_pressure = pˢᵗ,
                                    reference_potential_temperature = θ_background,
                                    reference_vapor_mass_fraction = qᵛ_bg)
    model = AtmosphereModel(grid; dynamics, microphysics = microphysics(), advection = advection(),
                            thermodynamic_constants = constants)
    return model
end

function set_compressible!(model)
    S = initial_fields!()
    grid = S[:grid]
    ρ_initial = CenterField(grid)
    set!(ρ_initial, pressure_balanced_density(model.dynamics.reference_state.density,
                                              S[:θ_background_column], S[:θ_initial_field]))
    set!(model, θ=S[:θ_initial_field], qᵛ=S[:qᵛ_column], u=uᵢ, ρ=ρ_initial)
    return model
end

function step_model!(model, n; Δt=2)
    for i in 1:n
        time_step!(model, Δt)
        if i == 1
            sync()
            @info "  first time step done"
        end
    end
    sync()
    return model
end

function model_summary(model)
    u, v, w = model.velocities
    return @sprintf("iter=%d t=%.1f max|u|=%.4f max|w|=%.4g", model.clock.iteration, model.clock.time,
                    maximum(abs, u), maximum(abs, w))
end

#####
##### Rungs
#####

function rung1()
    grid = make_grid()
    model = NonhydrostaticModel(grid; advection = WENO(order=9), tracers = :b, buoyancy = BuoyancyTracer())
    @info "  pressure solver: $(summary(model.pressure_solver))"
    set!(model, u = (x, y, z) -> u_background(z), b = (x, y, z) -> 1e-2 * exp(-((x - xᵇ)^2 + (y - yᵇ)^2) / rᵇʰ^2))
    n = min(NSTEPS, 3)
    for _ in 1:n
        time_step!(model, 1)
    end
    sync()
    any(isnan, interior(model.velocities.w)) && error("NaN in w after $n steps")
    return @sprintf("%d steps, max|w|=%.4g", n, maximum(abs, model.velocities.w))
end

function rung2()
    S = initial_fields!()
    grid = S[:grid]
    reference_state = ReferenceState(grid, constants; base_pressure = p₀, potential_temperature = θ₀)
    S[:reference_state] = reference_state
    return @sprintf("θ_init ∈ (%.2f, %.2f), max qᵛ = %.4g, ρ₀ range (%.4f, %.4f)",
                    minimum(S[:θ_initial_field]), maximum(S[:θ_initial_field]), maximum(S[:qᵛ_column]),
                    minimum(reference_state.density), maximum(reference_state.density))
end

function rung_model(build, setf, key)
    model = build()
    @info "  constructed $(key) model"
    STATE[key] = model
    setf(model)
    sync()
    @info "  set! done: $(model_summary(model))"
    CONSTRUCT_ONLY && return "constructed + set! only: " * model_summary(model)
    step_model!(model, NSTEPS)
    s = model_summary(model)
    any(isnan, interior(model.velocities.w)) && error("NaN after $NSTEPS steps: $s")
    return s
end

rung3() = rung_model(build_compressible, set_compressible!, :compressible)
rung4() = rung_model(build_anelastic, set_anelastic!, :anelastic)

function rung5()
    S = initial_fields!()
    grid = S[:grid]
    z_centers = znodes(grid, Center())
    k_5km    = searchsortedfirst(z_centers, 5000)
    j_center = Ny ÷ 2 + 1
    notes = String[]
    for (label, build, setf) in (("anelastic", build_anelastic, set_anelastic!),
                                 ("compressible", build_compressible, set_compressible!))
        model = setf(build())
        θ  = liquid_ice_potential_temperature(model)
        θ′ = Field(θ - deepcopy(θ))
        qᶜˡ = model.microphysical_fields.qᶜˡ
        qʳ  = model.microphysical_fields.qʳ
        qᵛ  = model.microphysical_fields.qᵛ
        u, v, w = model.velocities

        simulation = Simulation(model; Δt=2, stop_iteration=6)
        conjure_time_step_wizard!(simulation, cfl=0.7)
        Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

        function progress(sim)
            compute!(θ′)
            @info @sprintf("  [%s] iter %d t %s Δt %s max|u| %.3f max w %.3g min w %.3g θ′ (%.3g, %.3g) max qᵛ %.3e qᶜˡ %.3e qʳ %.3e",
                           label, iteration(sim), prettytime(sim), prettytime(sim.Δt),
                           maximum(abs, u), maximum(w), minimum(w), minimum(θ′), maximum(θ′),
                           maximum(qᵛ), maximum(qᶜˡ), maximum(qʳ))
        end
        add_callback!(simulation, progress, IterationInterval(2))

        slice_outputs = (wxy   = view(w,   :, :, k_5km),
                         qᶜˡxy = view(qᶜˡ, :, :, k_5km),
                         qʳxy  = view(qʳ,  :, :, k_5km),
                         wxz   = view(w,   :, j_center, :),
                         θ′xz  = view(θ′,  :, j_center, :),
                         qᶜˡxz = view(qᶜˡ, :, j_center, :),
                         qʳxz  = view(qʳ,  :, j_center, :))
        mkpath(OUTDIR)
        filename = joinpath(OUTDIR, "smoke_$(ARCH_NAME)_$(label)_slices.jld2")
        simulation.output_writers[:slices] = JLD2Writer(model, slice_outputs; filename,
                                                        schedule=IterationInterval(2), overwrite_files=true)
        run!(simulation)
        ts = FieldTimeSeries(filename, "wxy")
        push!(notes, "$label: $(iteration(simulation)) iters, $(length(ts.times)) slices saved")
    end
    return join(notes, "; ")
end

function rung6()
    mkpath(OUTDIR)
    notes = String[]
    for key in (:compressible, :anelastic)
        haskey(STATE, key) || (push!(notes, "$key: no model from rung $(key == :compressible ? 3 : 4)"); continue)
        model = STATE[key]
        fields = Oceananigans.prognostic_fields(model)
        data = Dict(string(name) => Array(interior(f)) for (name, f) in pairs(fields))
        file = joinpath(OUTDIR, "final_$(key)_$(ARCH_NAME).jld2")
        jldsave(file; fields = data, iteration = model.clock.iteration, time = model.clock.time)
        push!(notes, "$key -> $file")
        other = joinpath(OUTDIR, "final_$(key)_$(ONEAPI ? "cpu" : "oneapi").jld2")
        if isfile(other)
            ref = load(other, "fields")
            println("\n  compare $key: $ARCH_NAME vs $(ONEAPI ? "cpu" : "oneapi")  (iterations $(model.clock.iteration) vs $(load(other, "iteration")))")
            for name in sort(collect(keys(data)))
                haskey(ref, name) || continue
                a, b = Float64.(data[name]), Float64.(ref[name])
                d = maximum(abs, a .- b)
                s = max(maximum(abs, b), eps())
                println(@sprintf("    %-6s max|Δ| = %.3e   max|Δ|/max|ref| = %.3e   max|ref| = %.3e", name, d, d / s, s))
            end
        end
    end
    return join(notes, "; ")
end

const RUNG_TABLE = Dict(
    1 => ("Oceananigans NonhydrostaticModel, FFT pressure solver", rung1),
    2 => ("Breeze grid, reference state, initial-condition fields", rung2),
    3 => ("Compressible (split-explicit) model: construct, set!, $(NSTEPS) steps", rung3),
    4 => ("Anelastic model: construct, set!, $(NSTEPS) steps", rung4),
    5 => ("Diagnostics/output Simulation (both dynamics)", rung5),
    6 => ("Save final fields of rungs 3/4 and compare across archs", rung6),
)

@info "ARCH=$ARCH_NAME RUNG=$RUNGS NSTEPS=$NSTEPS KARGS=$KARGS_MODE CONSTRUCT_ONLY=$CONSTRUCT_ONLY"
if ONEAPI
    @info "device: $(oneAPI.properties(oneAPI.device()).name)"
end

for n in RUNGS
    label, f = RUNG_TABLE[n]
    rung(f, n, label)
end

println("\n", "="^100)
println("SUMMARY [$ARCH_NAME]")
for (n, label, status, dt, note) in RESULTS
    println(@sprintf("  rung %d  %-7s %8.1f s  %s | %s", n, status, dt, label, note))
end
