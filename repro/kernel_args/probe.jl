# Probe whether IGC_OverrideOCLMaxParamSize yields correct kernels with >2048 B of arguments.
# Run in a fresh process per setting:  SALT=<int> IGC_OverrideOCLMaxParamSize=<n> julia probe.jl
# Every kernel carries the salt as a compile-time constant (written to a sentinel), so its
# SPIR-V differs between runs and no driver-side binary cache can serve an old build.

using oneAPI, KernelAbstractions, Random, Printf
const KA = KernelAbstractions

const SALT = parse(Int, get(ENV, "SALT", "1"))
const SETTING = get(ENV, "IGC_OverrideOCLMaxParamSize", "unset")
const TARGETS = [1900, 2100, 2500, 2900, 3500, 4000]   # intended total argument bytes
const R = 4                                             # repeats per case

println("# setting IGC_OverrideOCLMaxParamSize=$SETTING SALT=$SALT NEO_CACHE_PERSISTENT=$(get(ENV,"NEO_CACHE_PERSISTENT","default"))")
println("# device: ", oneAPI.properties(oneAPI.device()).name)

argbytes(args...) = sum(a -> sizeof(typeof(oneAPI.kernel_convert(a))), args; init=0)
function compiler_size(e)
    m = match(r"Total arguments size: (\d+)", sprint(showerror, e))
    m === nothing ? "?" : m[1]
end
short(e) = first(replace(sprint(showerror, e), '\n' => ' '), 160)
result(case, target, bytes, status; extra="") =
    println(@sprintf("RESULT %-28s target=%-5d juliabytes=%-5d %s %s", case, target, bytes, status, extra))

function sentinel_ok(s)
    v = Array(s)[1]
    v == SALT || println("  !! sentinel $(v) != SALT $SALT")
    v == SALT
end

#############################################################################################
# Case 1: one large NTuple of distinct pseudo-random values, every element read
#############################################################################################

function k_tuple(::Val{S}, sentinel, out, hsh, t::NTuple{n,T}) where {S,n,T}
    i = get_global_id(1)
    i == 1 && (@inbounds sentinel[1] = S)
    @inbounds if i <= length(out)
        out[i] = t[mod1(i, n)]
        # position-weighted hash of the full tuple, mixing in i so every work item differs
        h = UInt64(i)
        for k in 1:n
            h = h * 0x9E3779B97F4A7C15 + reinterpret_u(t[k]) * UInt64(k)
        end
        hsh[i] = h
    end
    return
end
reinterpret_u(x::UInt64) = x
reinterpret_u(x::Float32) = UInt64(reinterpret(UInt32, x))

function cpu_tuple(N, t::NTuple{n}) where n
    out = [t[mod1(i, n)] for i in 1:N]
    hsh = map(1:N) do i
        h = UInt64(i)
        for k in 1:n
            h = h * 0x9E3779B97F4A7C15 + reinterpret_u(t[k]) * UInt64(k)
        end
        h
    end
    out, hsh
end

function describe_wrong(name, got, exp)
    bad = findall(got .!= exp)
    isempty(bad) && return ""
    s = "  !! $name: $(length(bad))/$(length(exp)) wrong, first=$(first(bad)) last=$(last(bad))"
    for b in bad[1:min(4, end)]
        s *= " [$b got=$(got[b]) exp=$(exp[b])]"
    end
    s
end

function run_case1(T, target)
    payload = target - 3 * 24          # three 1-D device arrays (24 B each)
    n = payload ÷ sizeof(T)
    N = 1 << 16                        # 65536 work items -> many work-groups / HW threads
    rng = Xoshiro(SALT * 1000 + target)
    case = "c1_tuple_$(T)_n$n"
    sentinel = oneArray(zeros(Int, 1))
    ts = [ntuple(_ -> T === Float32 ? rand(rng, Float32) * 1f3 - 5f2 : rand(rng, UInt64), n) for _ in 1:R]
    push!(ts, ts[1])                   # identical repeat for nondeterminism check
    outs = [oneArray(zeros(T, N)) for _ in ts]
    hshs = [oneArray(zeros(UInt64, N)) for _ in ts]
    bytes = argbytes(sentinel, outs[1], hshs[1], ts[1])
    local kern
    try
        kern = @oneapi launch=false k_tuple(Val(SALT), sentinel, outs[1], hshs[1], ts[1])
    catch e
        result(case, target, bytes, "COMPILE_FAIL", extra="compiler_total=$(compiler_size(e)) :: $(short(e))")
        return
    end
    # all launches back to back without synchronisation
    for r in eachindex(ts)
        kern(Val(SALT), sentinel, outs[r], hshs[r], ts[r]; groups=N ÷ 256, items=256)
    end
    oneAPI.synchronize()
    ok = sentinel_ok(sentinel)
    msgs = String[]
    for r in eachindex(ts)
        eo, eh = cpu_tuple(N, ts[r])
        go, gh = Array(outs[r]), Array(hshs[r])
        m1 = describe_wrong("rep$r elements", go, eo); m2 = describe_wrong("rep$r hash", gh, eh)
        isempty(m1) || push!(msgs, m1); isempty(m2) || push!(msgs, m2)
        ok &= isempty(m1) && isempty(m2)
    end
    nondet = Array(outs[1]) != Array(outs[end]) || Array(hshs[1]) != Array(hshs[end])
    result(case, target, bytes, ok ? "CORRECT" : "WRONG", extra="repeats=$(length(ts)) nondeterministic=$nondet")
    foreach(println, msgs)
end

#############################################################################################
# Case 2: tuple of K distinct oneArrays, every array read, the last ones written
#############################################################################################

@generated function wsum(arrs::NTuple{K,Any}, i, j, k, ::Val{M}) where {K,M}
    ex = :(zero(UInt32))
    for m in 1:M
        ex = :($ex + UInt32($m) * @inbounds(arrs[$m][i, j, k]))
    end
    ex
end

function k_arrays(::Val{S}, sentinel, arrs::NTuple{K,Any}) where {S,K}
    i = get_global_id(1); j = get_global_id(2); k = get_global_id(3)
    (i == 1 && j == 1 && k == 1) && (@inbounds sentinel[1] = S)
    @inbounds if i <= size(arrs[1], 1) && j <= size(arrs[1], 2) && k <= size(arrs[1], 3)
        s = wsum(arrs, i, j, k, Val(K - 3))
        last = arrs[K][i, j, k]
        arrs[K-2][i, j, k] = s
        arrs[K-1][i, j, k] = s ⊻ arrs[1][i, j, k]
        arrs[K][i, j, k] = last * UInt32(3) + s
    end
    return
end

function cpu_arrays!(arrs)
    K = length(arrs)
    s = sum(UInt32(m) .* arrs[m] for m in 1:K-3)
    last = copy(arrs[K])
    arrs[K-2] .= s
    arrs[K-1] .= s .⊻ arrs[1]
    arrs[K] .= last .* UInt32(3) .+ s
end

function run_case2(target, dims, label)
    per = sizeof(typeof(oneAPI.kernel_convert(oneArray(zeros(UInt32, 1, 1, 1)))))
    K = (target - 24) ÷ per
    case = "c2_arrays_$(label)_K$K"
    rng = Xoshiro(SALT * 7 + target)
    host = [rand(rng, UInt32(0):UInt32(1 << 20), dims) for _ in 1:K]
    dev = [oneArray(h) for h in host]
    sentinel = oneArray(zeros(Int, 1))
    bytes = argbytes(sentinel, Tuple(dev))
    groups = cld.(dims, (8, 8, 4)); items = min.(dims, (8, 8, 4))
    groups = cld.(dims, items)
    local kern
    try
        kern = @oneapi launch=false k_arrays(Val(SALT), sentinel, Tuple(dev))
    catch e
        result(case, target, bytes, "COMPILE_FAIL", extra="compiler_total=$(compiler_size(e)) :: $(short(e))")
        return
    end
    # R launches, each with the arrays rotated (different pointer values per slot), no sync
    for r in 0:R-1
        kern(Val(SALT), sentinel, Tuple(circshift(dev, r)); groups, items)
    end
    oneAPI.synchronize()
    for r in 0:R-1
        rot = circshift(host, r)        # views the same Array objects
        cpu_arrays!(rot)
    end
    ok = sentinel_ok(sentinel)
    msgs = String[]
    for m in 1:K
        g = Array(dev[m])
        if g != host[m]
            ok = false
            bad = findall(g .!= host[m])
            push!(msgs, "  !! array $m: $(length(bad)) wrong, first=$(Tuple(first(bad)))")
        end
    end
    result(case, target, bytes, ok ? "CORRECT" : "WRONG", extra="ndrange=$dims repeats=$R")
    foreach(println, msgs[1:min(10, end)])
end

#############################################################################################
# Case 3: Breeze-like nested NamedTuples + scalars + mixed-alignment struct, through KA
#############################################################################################

struct Mixed
    a::Int8
    b::Float64
    c::Int16
    d::Float32
    e::UInt8
    f::Int64
    g::Int32
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

@kernel function k_breeze!(st, sentinel, ::Val{S}) where S
    i, j, k = @index(Global, NTuple)
    (i == 1 && j == 1 && k == 1) && (@inbounds sentinel[1] = S)
    a = nsum(st.fields.tracers, i, j, k)
    b = nsum(st.fields.velocities, i, j, k)
    c = nsum(st.aux, i, j, k)
    sc = scalar_mix(st.params, st.mixed)
    @inbounds begin
        st.tendencies.Gu[i, j, k] = a + sc
        st.tendencies.Gv[i, j, k] = b ⊻ sc
        st.tendencies.Gw[i, j, k] = c * UInt32(7) + a
        last = st.aux[end][i, j, k]
        st.aux[end][i, j, k] = last + a + b + c
    end
end

function cpu_breeze!(st)
    nsumc(nt) = sum(UInt32(m) .* x for (m, x) in enumerate(values(nt)))
    a = nsumc(st.fields.tracers); b = nsumc(st.fields.velocities); c = nsumc(st.aux)
    sc = scalar_mix(st.params, st.mixed)
    st.tendencies.Gu .= a .+ sc
    st.tendencies.Gv .= b .⊻ sc
    st.tendencies.Gw .= c .* UInt32(7) .+ a
    st.aux[end] .= st.aux[end] .+ a .+ b .+ c
end

function make_state(rng, dims, ntr, naux, AT)
    arr() = AT(rand(rng, UInt32(0):UInt32(1 << 20), dims))
    tracers = NamedTuple{ntuple(m -> Symbol(:c, m), ntr)}(ntuple(_ -> arr(), ntr))
    aux = NamedTuple{ntuple(m -> Symbol(:x, m), naux)}(ntuple(_ -> arr(), naux))
    (; fields = (; velocities = (; u = arr(), v = arr(), w = arr()), tracers),
       tendencies = (; Gu = arr(), Gv = arr(), Gw = arr()),
       params = (; dt = rand(rng, Float32), n = rand(rng, Int32), m = rand(rng, Int64)),
       mixed = Mixed(rand(rng, Int8), rand(rng), rand(rng, Int16), rand(rng, Float32), rand(rng, UInt8), rand(rng, Int64), rand(rng, Int32)),
       aux)
end

to_host(st) = (; fields = (; velocities = map(Array, st.fields.velocities), tracers = map(Array, st.fields.tracers)),
                 tendencies = map(Array, st.tendencies), params = st.params, mixed = st.mixed, aux = map(Array, st.aux))

function run_case3(target, dims, label)
    per = sizeof(typeof(oneAPI.kernel_convert(oneArray(zeros(UInt32, 1, 1, 1)))))
    fixed = 6 * per + sizeof(Mixed) + 16 + 24 + 64   # vel+tend arrays, struct, params, sentinel, KA ctx (approx)
    narr = max(2, (target - fixed) ÷ per)
    ntr = narr ÷ 2; naux = narr - ntr
    case = "c3_breeze_$(label)_tr$(ntr)_aux$(naux)"
    rng = Xoshiro(SALT * 13 + target)
    backend = oneAPIBackend()
    sentinel = oneArray(zeros(Int, 1))
    sts = [make_state(rng, dims, ntr, naux, oneArray) for _ in 1:R]   # distinct values per repeat
    hosts = [to_host(s) for s in sts]
    bytes = argbytes(sts[1], sentinel)
    kern! = k_breeze!(backend)
    try
        for r in 1:R
            kern!(sts[r], sentinel, Val(SALT); ndrange = dims)
        end
        # also chain two launches on the same state (in-order dependency)
        kern!(sts[1], sentinel, Val(SALT); ndrange = dims)
        oneAPI.synchronize()
    catch e
        result(case, target, bytes, "COMPILE_FAIL", extra="compiler_total=$(compiler_size(e)) :: $(short(e))")
        return
    end
    for r in 1:R
        cpu_breeze!(hosts[r])
    end
    cpu_breeze!(hosts[1])
    ok = sentinel_ok(sentinel)
    msgs = String[]
    for r in 1:R
        g = to_host(sts[r]); h = hosts[r]
        for (nm, ga, ha) in (("Gu", g.tendencies.Gu, h.tendencies.Gu), ("Gv", g.tendencies.Gv, h.tendencies.Gv),
                             ("Gw", g.tendencies.Gw, h.tendencies.Gw), ("auxlast", g.aux[end], h.aux[end]))
            if ga != ha
                ok = false
                bad = findall(ga .!= ha)
                push!(msgs, "  !! rep$r $nm: $(length(bad)) wrong, first=$(Tuple(first(bad))) got=$(ga[first(bad)]) exp=$(ha[first(bad)])")
            end
        end
    end
    result(case, target, bytes, ok ? "CORRECT" : "WRONG", extra="ndrange=$dims repeats=$(R+1)")
    foreach(println, msgs[1:min(10, end)])
end

#############################################################################################

for target in TARGETS
    for f in (() -> run_case1(UInt64, target),
              () -> run_case1(Float32, target),
              () -> run_case2(target, (64, 64, 16), "big"),
              () -> run_case2(target, (2, 3, 1), "tiny"),
              () -> run_case3(target, (64, 64, 16), "big"),
              () -> run_case3(target, (3, 2, 1), "tiny"))
        try
            f()
        catch e
            println("RESULT ERROR target=$target :: ", short(e))
        end
        flush(stdout)
    end
end
println("# done")
