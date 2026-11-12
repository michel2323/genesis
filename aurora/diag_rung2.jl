# Step-by-step replay of smoke ladder rung 2 (grid, set! of the initial-condition fields,
# ReferenceState) on oneAPI, flushing after every step to locate a crash.
# Run from the repo root:  julia --project=. aurora/diag_rung2.jl
using Breeze
using Breeze: TetensFormula
using Breeze.Thermodynamics: hydrostatic_density, hydrostatic_temperature
using Oceananigans
using Oceananigans.Units
using oneAPI

step(msg) = (println(">>> ", msg); flush(stdout); flush(stderr))

Oceananigans.defaults.FloatType = Float32
Nx, Ny, Nz = 32, 32, 16
Lx, Ly, Lz = 168kilometers, 168kilometers, 20kilometers
constants = ThermodynamicConstants(saturation_vapor_pressure = TetensFormula())
p₀ = 100000; pˢᵗ = 100000; θ₀ = 300; θᵖ = 343; zᵖ = 12000; Tᵖ = 213; qᵛ_max = 0.014
g = constants.gravitational_acceleration
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
Δθ = 3; rᵇʰ = 10kilometers; rᵇᵛ = 1500; zᵇ = 1500; xᵇ = Lx / 2; yᵇ = Ly / 2
function θᵢ(x, y, z)
    θ̄ = θ_background(z)
    r = sqrt((x - xᵇ)^2 + (y - yᵇ)^2)
    R = sqrt((r / rᵇʰ)^2 + ((z - zᵇ) / rᵇᵛ)^2)
    θ′ = ifelse(R < 1, Δθ * cos(π * R / 2)^2, 0.0)
    return θ̄ + θ′
end

for archname in (:cpu, :oneapi)
    arch = archname == :cpu ? CPU() : GPU(oneAPIBackend())
    step("[$archname] grid")
    grid = RectilinearGrid(arch, size = (Nx, Ny, Nz), x = (0, Lx), y = (0, Ly), z = (0, Lz),
                           halo = (5, 5, 5), topology = (Periodic, Periodic, Bounded))
    step("[$archname] Field{Nothing,Nothing,Center}")
    qᵛ_column = Field{Nothing, Nothing, Center}(grid)
    step("[$archname] set!(qᵛ_column, qᵛ_bg)")
    set!(qᵛ_column, qᵛ_bg)
    step("[$archname]   max qᵛ = $(maximum(qᵛ_column))")
    θ_background_column = Field{Nothing, Nothing, Center}(grid)
    step("[$archname] set!(θ_background_column, θ_background)")
    set!(θ_background_column, θ_background)
    step("[$archname]   max θ_bg = $(maximum(θ_background_column))")
    θ_initial_field = CenterField(grid)
    step("[$archname] set!(θ_initial_field, θᵢ)")
    set!(θ_initial_field, θᵢ)
    step("[$archname]   θ_init ∈ ($(minimum(θ_initial_field)), $(maximum(θ_initial_field)))")
    step("[$archname] ReferenceState")
    rs = ReferenceState(grid, constants; base_pressure = p₀, potential_temperature = θ₀)
    step("[$archname]   ρ₀ ∈ ($(minimum(rs.density)), $(maximum(rs.density)))")
end
step("done")
