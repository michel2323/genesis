# PVC variant of repro/repro_simd32_fp64.jl (that file is left unchanged). PVC supports only
# sub-group sizes 16 and 32, so the original aborts at sub_group_size=8; this copy tries
# default/8/16/32 and skips the sizes the device does not support.
# Run from the repo root:  julia --project=repro aurora/repro_simd32_fp64_pvc.jl
using oneAPI

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
    try
        d = oneArray(zero(u)); du = oneArray(u)
        k = sg === nothing ? (@oneapi launch=false kern(d, du)) :
                             (@oneapi launch=false sub_group_size=sg kern(d, du))
        k(d, du; items=16)
        p = oneAPI.oneL0.properties(k.fun)
        err = maximum(abs.(Array(d) .- host))
        println("sub_group_size=", rpad(something(sg, "default"), 8),
                " compiled SIMD", p.maxSubgroupSize, "  spill=", p.spillMemSize, " B/thread",
                "  max|gpu-cpu| = ", err, "  (max|cpu| = ", maximum(abs, host), ")  ",
                err <= 1f-6 * maximum(abs, host) ? "CORRECT" : "WRONG")
    catch e
        println("sub_group_size=", rpad(something(sg, "default"), 8), " SKIPPED: ",
                first(split(sprint(showerror, e), '\n')))
    end
end
