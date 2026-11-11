# Total argument size > 2048 B, but every single argument < 2048 B (Breeze-like state split
# over several kernel arguments), launched through KernelAbstractions. Compared exactly to CPU.
using oneAPI, KernelAbstractions, Random, Printf

const SALT = parse(Int, get(ENV, "SALT", "1"))
println("# setting IGC_OverrideOCLMaxParamSize=$(get(ENV, "IGC_OverrideOCLMaxParamSize", "unset")) SALT=$SALT")

struct Mixed
    a::Int8; b::Float64; c::Int16; d::Float32; e::UInt8; f::Int64; g::Int32
end

@generated function nsum(nt::NamedTuple{names}, i, j, k) where names
    ex = :(zero(UInt32))
    for (m, nm) in enumerate(names)
        ex = :($ex + UInt32($m) * @inbounds(nt.$nm[i, j, k]))
    end
    ex
end
@inline scalar_mix(p, x::Mixed) =
    reinterpret(UInt32, p.dt) ⊻ (p.n % UInt32) ⊻ (p.m % UInt32) ⊻ ((p.m >> 32) % UInt32) ⊻
    (x.a % UInt32) ⊻ (reinterpret(UInt64, x.b) % UInt32) ⊻ ((reinterpret(UInt64, x.b) >> 32) % UInt32) ⊻
    (x.c % UInt32) * UInt32(3) ⊻ reinterpret(UInt32, x.d) * UInt32(5) ⊻ (x.e % UInt32) * UInt32(7) ⊻
    (x.f % UInt32) ⊻ ((x.f >> 32) % UInt32) * UInt32(11) ⊻ (x.g % UInt32) * UInt32(13)

@kernel function k_split!(tendencies, velocities, tracers, params, mixed, aux, sentinel, ::Val{S}) where S
    i, j, k = @index(Global, NTuple)
    (i == 1 && j == 1 && k == 1) && (@inbounds sentinel[1] = S)
    a = nsum(tracers, i, j, k); b = nsum(velocities, i, j, k); c = nsum(aux, i, j, k)
    sc = scalar_mix(params, mixed)
    @inbounds begin
        tendencies.Gu[i, j, k] = a + sc
        tendencies.Gv[i, j, k] = b ⊻ sc
        tendencies.Gw[i, j, k] = c * UInt32(7) + a
        last = aux[end][i, j, k]
        aux[end][i, j, k] = last + a + b + c
    end
end

function cpu!(tendencies, velocities, tracers, params, mixed, aux)
    nsumc(nt) = sum(UInt32(m) .* x for (m, x) in enumerate(values(nt)))
    a = nsumc(tracers); b = nsumc(velocities); c = nsumc(aux)
    sc = scalar_mix(params, mixed)
    tendencies.Gu .= a .+ sc; tendencies.Gv .= b .⊻ sc; tendencies.Gw .= c .* UInt32(7) .+ a
    aux[end] .= aux[end] .+ a .+ b .+ c
end

function run(ntr, naux, dims; R = 4)
    rng = Xoshiro(SALT + 100ntr + naux)
    arr() = oneArray(rand(rng, UInt32(0):UInt32(1 << 20), dims))
    states = map(1:R) do _
        (; tendencies = (; Gu = arr(), Gv = arr(), Gw = arr()),
           velocities = (; u = arr(), v = arr(), w = arr()),
           tracers = NamedTuple{ntuple(m -> Symbol(:c, m), ntr)}(ntuple(_ -> arr(), ntr)),
           params = (; dt = rand(rng, Float32), n = rand(rng, Int32), m = rand(rng, Int64)),
           mixed = Mixed(rand(rng, Int8), rand(rng), rand(rng, Int16), rand(rng, Float32), rand(rng, UInt8), rand(rng, Int64), rand(rng, Int32)),
           aux = NamedTuple{ntuple(m -> Symbol(:x, m), naux)}(ntuple(_ -> arr(), naux)))
    end
    hosts = [map(x -> x isa NamedTuple ? map(y -> y isa oneArray ? Array(y) : y, x) : x, s) for s in states]
    sentinel = oneArray(zeros(Int, 1))
    sizes = map(x -> sizeof(typeof(oneAPI.kernel_convert(x))), states[1])
    total = sum(sizes) + 48 + 24
    kern! = k_split!(oneAPIBackend())
    try
        for s in states   # no synchronisation between launches
            kern!(s.tendencies, s.velocities, s.tracers, s.params, s.mixed, s.aux, sentinel, Val(SALT); ndrange = dims)
        end
        oneAPI.synchronize()
    catch e
        println("SPLIT ntr=$ntr naux=$naux COMPILE_FAIL ", first(sprint(showerror, e), 120)); return
    end
    ok = Array(sentinel)[1] == SALT
    for h in hosts
        cpu!(h.tendencies, h.velocities, h.tracers, h.params, h.mixed, h.aux)
    end
    for (s, h) in zip(states, hosts)
        ok &= all(Array(s.tendencies[f]) == h.tendencies[f] for f in (:Gu, :Gv, :Gw)) && Array(s.aux[end]) == h.aux[end]
    end
    println(@sprintf("SPLIT ntr=%-3d naux=%-3d dims=%-12s largest_arg=%-5d julia_total=%-5d %s sentinel_ok=%s",
                     ntr, naux, string(dims), maximum(sizes), total, ok ? "CORRECT" : "WRONG", Array(sentinel)[1] == SALT))
end

for (ntr, naux) in ((20, 20), (30, 30), (36, 36))      # each NamedTuple 960 / 1440 / 1728 B
    run(ntr, naux, (64, 64, 16))
    run(ntr, naux, (3, 2, 1))
    flush(stdout)
end
println("# done")
