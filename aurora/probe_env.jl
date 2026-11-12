# Driver/library probe for Aurora. Run from the repo root:  julia --project=. aurora/probe_env.jl
using oneAPI, Libdl

println("---- oneAPI.versioninfo() ----")
oneAPI.versioninfo()
@show oneAPI.oneL0.LTS[]
@show oneAPI.device()
@show length(oneAPI.devices())

println("---- JLL library paths (as configured) ----")
for (mod, syms) in (("NEO_jll", (:libze_intel_gpu, :libigdrcl)),
                    ("libigc_jll", (:libigc, :libigdfcl)),
                    ("gmmlib_jll", (:libigdgmm,)),
                    ("oneAPI_Level_Zero_Loader_jll", (:libze_loader,)))
    try
        idx = findfirst(k -> k.name == mod, collect(keys(Base.loaded_modules)))
        idx === nothing && error("not loaded by oneAPI")
        m = Base.loaded_modules[collect(keys(Base.loaded_modules))[idx]]
        for s in syms
            p = try
                getfield(m, Symbol(s, :_path))
            catch
                isdefined(m, s) ? getfield(m, s) : "undefined"
            end
            println(rpad("$mod.$s", 48), p)
        end
    catch e
        println(rpad(mod, 48), "not loadable: ", sprint(showerror, e))
    end
end

println("---- Libdl.dllist() filtered (libraries actually loaded) ----")
for l in Libdl.dllist()
    occursin(r"ze_|libze|igc|igdgmm|igdrcl|igdfcl|opencl-clang|ocloc|ze_loader"i, basename(l)) &&
        println("  ", realpath(l) == l ? l : "$l -> $(realpath(l))")
end

println("---- trivial kernels vs CPU ----")
function k!(c, x, y)
    i = get_global_id()
    @inbounds c[i] = x[i] * y[i] + 2 * x[i]
    return
end
function check(T)
    n = 1 << 16
    a = rand(T, n); b = rand(T, n)
    da = oneArray(a); db = oneArray(b)
    # broadcast
    dc = da .* db .+ sin.(da)
    ok1 = Array(dc) ≈ (a .* b .+ sin.(a))
    # hand-written kernel (top-level k!, so the kernel does not capture the non-isbits `T`)
    dk = similar(da)
    @oneapi items=256 groups=cld(n, 256) k!(dk, da, db)
    ok2 = Array(dk) ≈ (a .* b .+ T(2) .* a)
    s = sum(da)
    ok3 = isapprox(s, sum(a); rtol = sqrt(eps(T)) * 10)
    println("$T: broadcast=$(ok1 ? "OK" : "WRONG")  kernel=$(ok2 ? "OK" : "WRONG")  sum=$(ok3 ? "OK" : "WRONG") ($s vs $(sum(a)))")
end
for T in (Float32, Float64)
    try
        check(T)
    catch e
        println("$T: ERROR ", sprint(showerror, e))
    end
end
