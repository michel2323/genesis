# Characterise which argument bytes a kernel receives under IGC_OverrideOCLMaxParamSize.
# Each kernel echoes its by-value arguments word by word into an output matrix, one column
# per work item, so the received bytes can be compared with what the host passed.
using oneAPI, Random, Printf

const SALT = parse(Int, get(ENV, "SALT", "1"))
println("# setting IGC_OverrideOCLMaxParamSize=$(get(ENV, "IGC_OverrideOCLMaxParamSize", "unset")) SALT=$SALT")
const G = 512   # work items (2 work-groups of 256)

# dynamic indexing (tuple spilled to private memory)
function k_echo_dyn(::Val{S}, out, pre::NTuple{p,UInt64}, t::NTuple{n,UInt64}) where {S,p,n}
    g = get_global_id(1)
    @inbounds begin
        for w in 1:p; out[w, g] = pre[w]; end
        for w in 1:n; out[p+w, g] = t[w] ⊻ UInt64(S - S); end
    end
    return
end

# fully unrolled constant indexing
@generated function k_echo_static(::Val{S}, out, pre::NTuple{p,UInt64}, t::NTuple{n,UInt64}) where {S,p,n}
    body = [:(@inbounds out[$w, g] = pre[$w]) for w in 1:p]
    append!(body, [:(@inbounds out[$(p + w), g] = t[$w] ⊻ UInt64($S - $S)) for w in 1:n])
    quote
        g = get_global_id(1)
        $(body...)
        return
    end
end

function echo(kernel, label, p, n)
    rng = Xoshiro(SALT + 31p + n)
    pre = ntuple(_ -> rand(rng, UInt64) | 1, p)   # never zero
    t = ntuple(_ -> rand(rng, UInt64) | 1, n)
    out = oneArray(zeros(UInt64, p + n, G))
    total = 24 + 8 * (p + n)       # Julia bytes (output array 24 B); IGC reports this + 24
    try
        @oneapi groups=G ÷ 256 items=256 kernel(Val(SALT), out, pre, t)
        oneAPI.synchronize()
    catch e
        println(@sprintf("ECHO %-6s p=%-3d n=%-4d total=%-5d COMPILE_FAIL %s", label, p, n, total,
                         something(match(r"Total arguments size: \d+, limit: \d+", sprint(showerror, e)), (match="?",)).match))
        return
    end
    got = Array(out)
    exp = [pre..., t...]
    cols_equal = all(c -> got[:, c] == got[:, 1], 1:G)
    bad = findall(got[:, 1] .!= exp)
    if isempty(bad)
        println(@sprintf("ECHO %-6s p=%-3d n=%-4d total=%-5d CORRECT items_agree=%s", label, p, n, total, cols_equal))
    else
        w = first(bad)
        # byte offset of the first bad word within all by-value args (output array first)
        off = 24 + 8 * (w - 1)
        zeros_ = count(i -> got[i, 1] == 0, bad)
        contiguous = bad == collect(w:p+n)
        # is the bad tail a shifted copy of earlier data?
        shifted = [s for s in 1:w-1 if all(i -> i - s >= 1 && got[i, 1] == exp[i-s], bad)]
        println(@sprintf("ECHO %-6s p=%-3d n=%-4d total=%-5d WRONG first_bad_word=%d (arg byte offset %d) nbad=%d zeros=%d tail_contiguous=%s shifted_copy_of=%s items_agree=%s",
                         label, p, n, total, w, off, length(bad), zeros_, contiguous, shifted, cols_equal))
    end
end

for (p, totals) in ((0, (2056, 2080, 2104, 2112, 2120, 2128, 2136, 2152, 2200, 2304, 2504, 2904, 3504, 4000)),
                    (32, (2104, 2128, 2152, 2504)),
                    (128, (2104, 2128, 2152, 2504)))
    for total in totals
        n = (total - 24) ÷ 8 - p
        echo(k_echo_dyn, "dyn", p, n)
        echo(k_echo_static, "static", p, n)
        flush(stdout)
    end
end

# tuple of K 3-D device arrays (40 B each): echo dims and maxsize, then the first element
function k_arr_meta(::Val{S}, out, arrs::NTuple{K,Any}) where {S,K}
    g = get_global_id(1)
    @inbounds for m in 1:K
        a = arrs[m]
        out[4m-3, g] = size(a, 1) + S - S
        out[4m-2, g] = size(a, 2)
        out[4m-1, g] = size(a, 3)
        out[4m,   g] = getfield(a, :maxsize)
    end
    return
end
function k_arr_read(::Val{S}, out, arrs::NTuple{K,Any}) where {S,K}
    g = get_global_id(1)
    @inbounds for m in 1:K
        out[m, g] = arrs[m][1, 1, 1] + S - S
    end
    return
end

for K in (44, 48, 50, 51, 52, 53, 56, 60, 72)
    arrs = Tuple(oneArray(fill(UInt32(1000 + m), 1 + m % 3, 2, 1 + m % 5)) for m in 1:K)
    total = 24 + 40K
    exp_meta = reduce(vcat, [[1 + m % 3, 2, 1 + m % 5, (1 + m % 3) * 2 * (1 + m % 5) * 4] for m in 1:K])
    meta = oneArray(zeros(Int, 4K, G))
    try
        @oneapi groups=G ÷ 256 items=256 k_arr_meta(Val(SALT), meta, arrs)
        oneAPI.synchronize()
    catch e
        println("ARRAYS K=$K total=$total COMPILE_FAIL"); continue
    end
    got = Array(meta)[:, 1]
    bad = findall(got .!= exp_meta)
    s = isempty(bad) ? "meta CORRECT" :
        "meta WRONG first_bad_word=$(first(bad)) (array $(cld(first(bad), 4)), field $(("dims1", "dims2", "dims3", "maxsize")[(first(bad) - 1) % 4 + 1]), arg byte offset $(24 + 40 * (cld(first(bad), 4) - 1) + (0, 8, 16, 32)[(first(bad) - 1) % 4 + 1])) nbad=$(length(bad)) zeros=$(count(i -> got[i] == 0, bad))"
    rd = oneArray(zeros(UInt32, K, G))
    @oneapi groups=G ÷ 256 items=256 k_arr_read(Val(SALT), rd, arrs)
    oneAPI.synchronize()
    gr = Array(rd)[:, 1]
    badr = findall(gr .!= UInt32.(1000 .+ (1:K)))
    s2 = isempty(badr) ? "reads CORRECT" : "reads WRONG arrays=$(badr) got=$(gr[badr[1:min(3, end)]])"
    println("ARRAYS K=$K total=$total $s $s2")
    flush(stdout)
end
println("# done")
