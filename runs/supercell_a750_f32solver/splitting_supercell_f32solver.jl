# # Splitting supercell — anelastic vs compressible
#
# This example simulates the development of a splitting supercell thunderstorm with
# both Breeze dynamical cores side-by-side, following the idealized test case
# described by [KlempEtAl2015](@citet) and the DCMIP2016 supercell intercomparison
# by [Zarzycki2019](@citet). This benchmark evaluates the model's ability to capture
# deep moist convection with warm-rain microphysics and strong updrafts.
#
# For microphysics we use the Kessler scheme, which includes prognostic cloud water
# and rain water with autoconversion, accretion, rain evaporation, and sedimentation
# processes. This is the same scheme used in the DCMIP2016 supercell intercomparison
# [Zarzycki2019](@cite).
#
# We run two simulations from identical initial conditions: one with the anelastic
# solver and one with the fully compressible solver (split-explicit acoustic
# substepping). At the end we compare horizontal (`xy` at ``z \approx 5 \, {\rm km}``)
# and vertical (`xz` at ``y = L_y/2``) slices side-by-side, and plot maximum
# vertical-velocity time series for both runs on the same axes.
#
# ## Physical setup
#
# The simulation initializes a conditionally unstable atmosphere with a warm bubble
# perturbation that triggers deep convection. The environment includes:
# - A realistic tropospheric potential temperature profile with a tropopause at 12 km
# - Relative humidity that decreases with height, with the resulting water vapor mixing ratio capped at
#   0.014 kg/kg "to approximate a well-mixed boundary layer in the lowest kilometer" ([KlempEtAl2015](@citet)).
# - Wind shear in the lower 5 km to promote storm rotation and supercell development
#
# ### Potential temperature profile
#
# The background potential temperature follows a piecewise profile
# (Equation 14 in [KlempEtAl2015](@citet)):
#
# ```math
# θ(z) = \begin{cases}
#     θ_0 + (θ_{\rm tr} - θ_0) \left(\dfrac{z}{z_{\rm tr}}\right)^{5/4} & z \leq z_{\rm tr} \\
#     θ_{\rm tr} \exp\left[\dfrac{g}{c_p^d T_{\rm tr}} (z - z_{\rm tr})\right] & z > z_{\rm tr}
# \end{cases}
# ```
#
# where ``θ_0 = 300 \, {\rm K}`` is the surface potential temperature,
# ``θ_{\rm tr} = 343 \, {\rm K}`` is the tropopause potential temperature,
# ``z_{\rm tr} = 12 \, {\rm km}`` is the tropopause height, and
# ``T_{\rm tr} = 213 \, {\rm K}`` is the tropopause temperature.
#
# ### Warm bubble perturbation
#
# A localized warm bubble triggers convection (Equations 17–18 in [KlempEtAl2015](@citet)):
#
# ```math
# θ'(x, y, z) = \begin{cases}
#     Δθ \cos^2\left(π R / 2 \right) & R < 1 \\
#     0 & R \geq 1
# \end{cases}
# ```
#
# where ``R = \sqrt{(r/r_h)^2 + [(z-z_c)/r_z]^2}`` is the normalized radius,
# ``r = \sqrt{(x-x_c)^2 + (y-y_c)^2}`` is the horizontal distance from the bubble center,
# ``Δθ = 3 \, {\rm K}`` is the perturbation amplitude, ``r_h = 10 \, {\rm km}`` is the
# horizontal radius, and ``r_z = 1.5 \, {\rm km}`` is the vertical radius.
#
# ### Wind shear profile
#
# The zonal wind increases linearly with height up to the shear layer ``z_s = 5 \, {\rm km}``,
# with a smooth transition zone, providing the environmental shear necessary for supercell
# development and mesocyclone formation (Equations 15-16 in [KlempEtAl2015](@citet)).

using Breeze
using Breeze: DCMIP2016KesslerMicrophysics, TetensFormula
using Breeze.Thermodynamics: hydrostatic_density, hydrostatic_temperature,
                             pressure_balanced_density
using Oceananigans: Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: znodes

using CairoMakie
using oneAPI
using Printf

# ## Domain and grid
#
# The domain is 168 km × 168 km × 20 km with 168 × 168 × 40 grid points, giving
# 1 km horizontal resolution and 500 m vertical resolution. The grid uses periodic
# lateral boundary conditions and bounded top/bottom boundaries.

Oceananigans.defaults.FloatType = Float32

Nx, Ny, Nz = 168, 168, 40
Lx, Ly, Lz = 168kilometers, 168kilometers, 20kilometers

grid = RectilinearGrid(GPU(oneAPIBackend()),
                       size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (0, Lz),
                       halo = (5, 5, 5),
                       topology = (Periodic, Periodic, Bounded))

# ## Background profiles
#
# Thermodynamic constants and the surface/standard pressures shared by both runs:

constants = ThermodynamicConstants(saturation_vapor_pressure = TetensFormula())

p₀  = 100000
pˢᵗ = 100000

# Stratification parameters define the troposphere–stratosphere transition:

θ₀     = 300       # K - surface potential temperature
θᵖ     = 343       # K - tropopause potential temperature
zᵖ     = 12000     # m - tropopause height
Tᵖ     = 213       # K - tropopause temperature
qᵛ_max = 0.014     # kg/kg - cap on water vapor mixing ratio from Klemp et al. (2015)
nothing #hide

# Wind shear parameters:

zˢ = 5kilometers  # m - shear layer height
uˢ = 30           # m/s - maximum shear wind speed
uᶜ = 15           # m/s - storm motion (Galilean translation speed)
nothing #hide

# Thermodynamic constants used inside the profile functions:

g   = constants.gravitational_acceleration
cᵖᵈ = constants.dry_air.heat_capacity
nothing #hide

# Background potential temperature profile (Equation 14 in [KlempEtAl2015](@citet)):

function θ_background(z)
    θᵗ = θ₀ + (θᵖ - θ₀) * (z / zᵖ)^(5/4)
    θˢ = θᵖ * exp(g / (cᵖᵈ * Tᵖ) * (z - zᵖ))
    return (z ≤ zᵖ) * θᵗ + (z > zᵖ) * θˢ
end

# Relative humidity profile (Equations 11–12 by [KlempEtAl2015](@citet)) combined with
# the water vapor cap ``qᵛ_{max}``. The local temperature and density are obtained by
# numerically integrating the hydrostatic balance with the actual ``θ(z)`` profile:

function qᵛ_bg(z)
    ℋ = (1 - 3/4 * (z / zᵖ)^(5/4)) * (z ≤ zᵖ) + 1/4 * (z > zᵖ)
    T = hydrostatic_temperature(z, p₀, θ_background, pˢᵗ, constants)
    ρ = hydrostatic_density(z, p₀, θ_background, pˢᵗ, constants)
    qᵛ⁺ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
    return min(ℋ * qᵛ⁺, qᵛ_max)
end

# Zonal wind profile with linear shear below ``zˢ`` and smooth transition (Equations 15-16):

function u_background(z)
    uˡ = uˢ * (z / zˢ) - uᶜ
    uᵗ = (-4/5 + 3 * (z / zˢ) - 5/4 * (z / zˢ)^2) * uˢ - uᶜ
    uᵘ = uˢ - uᶜ
    return (z < (zˢ - 1000)) * uˡ +
           (abs(z - zˢ) ≤ 1000) * uᵗ +
           (z > (zˢ + 1000)) * uᵘ
end

# ## Warm bubble perturbation
#
# The warm bubble parameters following Equations 17–18 in [KlempEtAl2015](@citet):

Δθ  = 3              # K - perturbation amplitude
rᵇʰ = 10kilometers   # m - bubble horizontal radius
rᵇᵛ = 1500           # m - bubble vertical radius
zᵇ  = 1500           # m - bubble center height
xᵇ  = Lx / 2         # m - bubble center x-coordinate
yᵇ  = Ly / 2         # m - bubble center y-coordinate
nothing #hide

# The total initial potential temperature combines the background profile with the
# cosine-squared warm bubble perturbation:

function θᵢ(x, y, z)
    θ̄ = θ_background(z)
    r = sqrt((x - xᵇ)^2 + (y - yᵇ)^2)
    R = sqrt((r / rᵇʰ)^2 + ((z - zᵇ) / rᵇᵛ)^2)
    θ′ = ifelse(R < 1, Δθ * cos(π * R / 2)^2, 0.0)
    return θ̄ + θ′
end

uᵢ(x, y, z) = u_background(z)

# ## Initial-condition fields
#
# We evaluate the background ``qᵛ`` and the bubble-augmented ``θ`` once on column /
# 3D fields and reuse them for both the anelastic and compressible runs. The
# hydrostatic integration in `qᵛ_bg` then runs only once per vertical level rather
# than once per horizontal grid point.

qᵛ_column = Field{Nothing, Nothing, Center}(grid)
set!(qᵛ_column, qᵛ_bg)

θ_background_column = Field{Nothing, Nothing, Center}(grid)
set!(θ_background_column, θ_background)

θ_initial_field = CenterField(grid)
set!(θ_initial_field, θᵢ)

# ## Visualization of initial conditions and warm bubble perturbation
#
# We visualize the background potential temperature, water vapor mixing ratio, and
# wind shear profiles that define the environmental stratification:

θ_profile  = set!(Field{Nothing, Nothing, Center}(grid), z -> θ_background(z))
qᵛ_profile = set!(Field{Nothing, Nothing, Center}(grid), 1000 * qᵛ_column) # convert kg/kg -> g/kg
u_profile  = set!(Field{Nothing, Nothing, Center}(grid), z -> u_background(z))

fig = Figure(size=(1000, 400), fontsize=14)

axθ = Axis(fig[1, 1], xlabel="θ (K)", ylabel="z (km)", title="Potential temperature")
lines!(axθ, θ_profile, linewidth=2, color=:magenta)
hlines!(axθ, [zᵖ / 1000], color=:gray, linestyle=:dash)

axqᵛ = Axis(fig[1, 2], xlabel="qᵛ (g/kg)", ylabel="z (km)", title="Water vapor mixing ratio")
lines!(axqᵛ, qᵛ_profile, linewidth=2, color=:dodgerblue)
hlines!(axqᵛ, [zᵖ / 1000], color=:gray, linestyle=:dash)

axu = Axis(fig[1, 3], xlabel="u (m/s)", ylabel="z (km)", title="Wind profile")
lines!(axu, u_profile, linewidth=2, color=:orangered)
hlines!(axu, [zˢ / 1000], color=:gray, linestyle=:dash)
vlines!(axu, [0], color=:black, linestyle=:dot)

save("supercell_initial_conditions.png", fig) #src
fig

# Visualize the warm bubble perturbation on a vertical slice through the domain center:

θ′_slice = set!(Field{Center, Nothing, Center}(grid), (x, z) -> θᵢ(x, yᵇ, z) - θ_background(z))

fig = Figure(size=(700, 400), fontsize=14)
ax = Axis(fig[1, 1], xlabel="x (km)", ylabel="z (km)",
          title="Warm bubble perturbation θ′")

hm = heatmap!(ax, θ′_slice, colormap=:thermal, colorrange=(0, Δθ))
Colorbar(fig[1, 2], hm, label="θ′ (K)")

save("supercell_warm_bubble.png", fig) #src
fig

# ## Reference state — anelastic
#
# Breeze dynamics subtract a hydrostatically-balanced reference column from the
# prognostic state so that the time-tendency variables carry only the deviation
# from rest. For the **anelastic** core, density is a fixed background ``\bar ρ(z)``
# rather than a prognostic variable, and we construct the reference from a single
# surface potential temperature `θ₀`. Anelastic dynamics are insensitive to the
# exact reference stratification as long as buoyancy perturbations remain small
# compared with it.

reference_state_anelastic = ReferenceState(grid, constants;
                                           base_pressure = p₀,
                                           potential_temperature = θ₀)

# ## Dynamics — anelastic
#
# `AnelasticDynamics` integrates an incompressible-with-stratification equation set
# that filters acoustic waves by construction through a pressure-Poisson solve at
# every RK substep:

dynamics_anelastic = AnelasticDynamics(reference_state_anelastic; pressure_solver_float_type=Float32)

# ## Microphysics
#
# Kessler warm-rain microphysics carries prognostic cloud water `qᶜˡ` and rain
# water `qʳ` alongside vapor `qᵛ`, with autoconversion, accretion, rain
# evaporation, and sedimentation:

microphysics = DCMIP2016KesslerMicrophysics()

# ## Advection
#
# We use WENO advection at order 9. [KlempEtAl2015](@citet) note that supercell
# intensity and structure are highly sensitive to numerical diffusion; high-order
# WENO keeps it low without adding an explicit diffusion operator.

advection = WENO(order=9)

# ## Building the anelastic model
#
# `AtmosphereModel` ties together the grid, dynamics, microphysics, advection, and
# thermodynamic constants. The same constructor signature works for both dynamical
# cores — only the `dynamics` argument changes.

model_anelastic = AtmosphereModel(grid; dynamics = dynamics_anelastic,
                                  microphysics, advection,
                                  thermodynamic_constants = constants)

# ## Initializing the anelastic model
#
# `set!` accepts pointwise functions (`uᵢ`) and pre-built fields
# (`θ_initial_field`, `qᵛ_column`). It calls `update_state!` internally so a single
# invocation refreshes all auxiliary diagnostics.

set!(model_anelastic, θ=θ_initial_field, qᵛ=qᵛ_column, u=uᵢ)

# ## Slice indices for output
#
# We save horizontal slices at ``z \approx 5 \, {\rm km}`` (mid-troposphere, where
# the rotating updraft is well-developed) and vertical slices through the bubble
# center at ``y = L_y/2``. Both runs use the same indices.

z_centers = znodes(grid, Center())
const k_5km    = searchsortedfirst(z_centers, 5000)
const j_center = Ny ÷ 2 + 1
@info "Saving xy at z = $(z_centers[k_5km]) m (k = $k_5km); xz at y = $(Ly/2) m (j = $j_center)"

# ## Simulation driver
#
# `run_simulation` accepts an already-built `AtmosphereModel`, runs it for 2 hours
# with a CFL-controlled time-step wizard, periodically writes horizontal (`xy` at
# ``z \approx 5 \, {\rm km}``) and vertical (`xz` at ``y = L_y/2``) slices, and
# collects the maximum vertical-velocity time series. We use this same driver for
# both the anelastic and compressible runs.

function run_simulation(model, label)
    @info "=== Running case: $label ==="

    θ  = liquid_ice_potential_temperature(model)
    θ′ = Field(θ - deepcopy(θ))  ## lazy field, recomputed each output

    qᶜˡ = model.microphysical_fields.qᶜˡ
    qʳ  = model.microphysical_fields.qʳ
    qᵛ  = model.microphysical_fields.qᵛ
    u, v, w = model.velocities

    simulation = Simulation(model; Δt=2, stop_time=2hours)
    conjure_time_step_wizard!(simulation, cfl=0.7)
    Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

    wall_clock = Ref(time_ns())
    function progress(sim)
        elapsed = 1e-9 * (time_ns() - wall_clock[])
        compute!(θ′)
        msg = @sprintf("[%s] Iter: %d, t: %s, Δt: %s, wall: %s, max|u|: %.2f, max w: %.2f, min w: %.2f, extrema(θ'): (%.2f, %.2f)",
                       label, iteration(sim), prettytime(sim), prettytime(sim.Δt), prettytime(elapsed),
                       maximum(abs, u), maximum(w), minimum(w),
                       minimum(θ′), maximum(θ′))
        msg *= @sprintf(", max(qᵛ): %.2e, max(qᶜˡ): %.2e, max(qʳ): %.2e",
                        maximum(qᵛ), maximum(qᶜˡ), maximum(qʳ))
        @info msg
        return nothing
    end
    add_callback!(simulation, progress, IterationInterval(100))

    max_w_ts    = Float64[]
    max_w_times = Float64[]
    function collect_max_w(sim)
        push!(max_w_times, time(sim))
        push!(max_w_ts, maximum(w))
        return nothing
    end
    add_callback!(simulation, collect_max_w, TimeInterval(1minutes))

    slice_outputs = (
        wxy   = view(w,   :, :, k_5km),
        qᶜˡxy = view(qᶜˡ, :, :, k_5km),
        qʳxy  = view(qʳ,  :, :, k_5km),
        wxz   = view(w,   :, j_center, :),
        θ′xz  = view(θ′,  :, j_center, :),
        qᶜˡxz = view(qᶜˡ, :, j_center, :),
        qʳxz  = view(qʳ,  :, j_center, :),
    )

    slices_filename = "splitting_supercell_$(label)_slices.jld2"
    simulation.output_writers[:slices] = JLD2Writer(model, slice_outputs;
        filename=slices_filename, schedule=TimeInterval(2minutes), overwrite_files=true)

    run!(simulation)
    wall_seconds = simulation.run_wall_time
    @info @sprintf("[%s] DONE. wall time = %.1f s (%.2f min) over %d iterations",
                   label, wall_seconds, wall_seconds / 60, iteration(simulation))

    return (; label, wall_seconds, slices_filename, max_w_ts, max_w_times,
            iterations = iteration(simulation))
end

# Run the anelastic simulation:

results = Dict{String, Any}()
results["anelastic"] = run_simulation(model_anelastic, "anelastic")

r = results["anelastic"]
@printf("%-14s  %12.1f  %12.2f  %12d\n", "anelastic_f32", r.wall_seconds, r.wall_seconds / 60, r.iterations)
using JLD2
jldsave("max_w_anelastic_f32solver.jld2"; times = r.max_w_times, max_w = r.max_w_ts)
