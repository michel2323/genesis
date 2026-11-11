# The SIMD32/emulated-FP64 reproducer (repro_simd32_fp64.jl) through the default
# KernelAbstractions path, i.e. whatever sub-group size oneAPI.jl's KA backend picks.
using oneAPI, KernelAbstractions
import KernelAbstractions as KA
import KernelInterface as KI

@inline left(ψ, i) = @inbounds 0.033333333333333326 * ψ[i-3] - 0.21666666666666667 * ψ[i-2] +
                               0.7833333333333333 * ψ[i-1] + 0.45 * ψ[i] - 0.04999999999999998 * ψ[i+1]
@inline flux(u, i) = @inbounds ((u[i] + u[i+1]) / 2f0) * left(u, i+1)
@inline g(u, i) = flux(u, i) - flux(u, i-1)

@kernel function kern!(out, u)
    i = @index(Global) + 4
    @inbounds out[i] = Float32(g(u, i))
end

u = Float32[1f-2 * sin(2π * (i - 4.5) / 16) for i in 1:24]
host = [(5 <= i <= 20) ? Float32(g(u, i)) : 0f0 for i in eachindex(u)]

backend = oneAPIBackend()
println("KI.sub_group_size(backend) = ", KI.sub_group_size(backend))
d = oneArray(zero(u)); du = oneArray(u)
kern!(backend, 16)(d, du; ndrange = 16)
KA.synchronize(backend)
err = maximum(abs.(Array(d) .- host))
println("default KA path: max|gpu-cpu| = ", err, "  (max|cpu| = ", maximum(abs, host), ")")
println(err <= 1f-6 * maximum(abs, host) * 16 ? "CORRECT" : "WRONG")
