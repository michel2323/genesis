# Minimal reproducer (no Oceananigans, no KernelAbstractions): a small kernel that does
# Float64 arithmetic on Float32 data gives wrong results on an Arc A750 (FP64 emulated via
# OverrideDefaultFP64Settings=1 IGC_EnableDPEmulation=1) when compiled with a required
# sub-group size of 32. SIMD8/16 are correct, and so is the default (IGC picks SIMD8).
# The KernelInterface backend of oneAPI.jl compiles every KA kernel with sub_group_size=32.
using oneAPI

# 5-point upwind stencil with Float64 coefficients (as Oceananigans' UpwindBiased/WENO{Float64})
@inline left(ψ, i) = @inbounds 0.033333333333333326 * ψ[i-3] - 0.21666666666666667 * ψ[i-2] +
                               0.7833333333333333 * ψ[i-1] + 0.45 * ψ[i] - 0.04999999999999998 * ψ[i+1]
@inline flux(u, i) = @inbounds ((u[i] + u[i+1]) / 2f0) * left(u, i+1)
@inline g(u, i) = flux(u, i) - flux(u, i-1)

function kern(out, u)
    i = get_global_id() + 4
    @inbounds out[i] = Float32(g(u, i))
    return
end

u = Float32[1f-2 * sin(2π * (i - 4.5) / 16) for i in 1:24]
host = [(5 <= i <= 20) ? Float32(g(u, i)) : 0f0 for i in eachindex(u)]

for sg in (nothing, 8, 16, 32)
    d = oneArray(zero(u)); du = oneArray(u)
    k = sg === nothing ? (@oneapi launch=false kern(d, du)) :
                         (@oneapi launch=false sub_group_size=sg kern(d, du))
    k(d, du; items=16)
    p = oneAPI.oneL0.properties(k.fun)
    println("sub_group_size=", rpad(something(sg, "default"), 8),
            " compiled SIMD", p.maxSubgroupSize, "  spill=", p.spillMemSize, " B/thread",
            "  max|gpu-cpu| = ", maximum(abs.(Array(d) .- host)), "  (max|cpu| = ", maximum(abs, host), ")")
end
