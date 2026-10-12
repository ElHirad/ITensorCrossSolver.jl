"""
Result of TT-cross: the `mps`, per-sweep sampled relative `errors`, completed
`sweeps`, unique callback `evaluations`, and a sampled `converged` flag.
Sampled convergence is not a rigorous global error bound.
"""
struct CrossResult
    mps::MPS
    errors::Vector{Float64}
    sweeps::Int
    evaluations::Int
    converged::Bool
end

# Saturating products avoid enumerating or overflowing exponentially large grids.
function _capped_product(dims, cap)
    p = 1
    for n in dims
        p > cap ÷ n && return cap
        p *= n
    end
    return min(p, cap)
end

function _random_tuples(rng, dims, count; existing=Tuple[])
    target = _capped_product(dims, count)
    samples = unique(Tuple.(existing))
    length(samples) > target && resize!(samples, target)
    seen = Set(samples)
    # Enumerate only small sample spaces, avoiding rejection near saturation.
    total = _capped_product(dims, max(4target, 1))
    if total <= 2target
        candidates = vec([Tuple(i) for i in CartesianIndices(Tuple(dims))])
        shuffle!(rng, candidates)
        for i in candidates
            length(samples) == target && break
            i in seen || (push!(samples, i); push!(seen, i))
        end
    else
        while length(samples) < target
            i = Tuple(rand(rng, 1:n) for n in dims)
            i in seen || (push!(samples, i); push!(seen, i))
        end
    end
    return samples
end

function _svd_rank(s, tolerance, cap)
    scale = maximum(s)
    iszero(scale) && return 1
    relative = s ./ scale
    r = min(length(s), cap)
    # Discard the smallest singular values subject to a relative Frobenius bound.
    # Work in normalized units: squaring raw values can underflow or overflow.
    threshold = tolerance^2 * sum(abs2, relative)
    tail = sum(abs2, @view relative[(r + 1):end])
    while r > 1 && tail + abs2(relative[r]) <= threshold
        tail += abs2(relative[r])
        r -= 1
    end
    return r
end

function _cross_sweep(f, dims, right, caps, tolerance, adaptive, oversampling)
    cores = Array[]
    prefixes = Tuple[()]
    extended = Vector{Vector{Tuple}}()
    for k in 1:(length(dims) - 1)
        fiber = evaluate_fiber(f, dims, k, prefixes, right[k])
        matrix = reshape(fiber, length(prefixes) * dims[k], :)
        F = svd(matrix)
        cap = min(caps[k], length(F.S))
        # Even in fixed-budget mode remove numerically null singular directions.
        localtol = adaptive ? tolerance / sqrt(length(dims)-1) : 100eps(eltype(F.S))
        r = _svd_rank(F.S, localtol, cap)
        U = F.U[:, 1:r]
        pivots = deim(U)
        G = U / U[pivots, :]
        push!(cores, reshape(G, length(prefixes), dims[k], r))
        extra = min(oversampling, size(U, 1)-r)
        samples = gappy_pod(U, extra, pivots)
        # Julia reshape uses the left bond as the fastest-varying row index.
        expand(p) = (prefixes[mod1(p, length(prefixes))]..., cld(p, length(prefixes)))
        push!(extended, Tuple[expand(p) for p in samples])
        prefixes = Tuple[expand(p) for p in pivots]
    end
    push!(cores, evaluate_fiber(f, dims, length(dims), prefixes, [()]))
    return cores, extended
end

"""
    tt_cross(f, dims; sites, ranks=2, maxdim=32, tolerance=1e-8,
             maxsweeps=20, oversampling=2, adaptive=true,
             validation_samples=64, rng=MersenneTwister(0))

Alternating SVD/DEIM tensor-train cross interpolation of `f(index_tuple)`.
Only sampled entries are evaluated. Uses Gappy-POD enrichment, cached evaluations,
and independent random validation entries (exhaustive only on very small tensors).
`ranks` is an initial scalar or per-bond rank budget; adaptive mode grows budgets
when validation fails and truncates sampled singular spectra. With `adaptive=false`
the supplied rank budgets remain fixed, apart from numerical rank deficiency.

The returned MPS always uses the original site order, for odd and even sweep counts.
Inspect `result.converged` and `result.errors[end]`; a rank/sweep limit can prevent
convergence. Random samples can miss isolated spikes; this is a smooth/low-rank
approximation method, not a global error certificate. Pass your own RNG for control.
"""
function tt_cross(
    f,
    dims;
    sites=nothing,
    ranks=2,
    maxdim::Integer=32,
    tolerance::Real=1e-8,
    maxsweeps::Integer=20,
    oversampling::Integer=2,
    adaptive::Bool=true,
    validation_samples::Integer=64,
    rng::AbstractRNG=MersenneTwister(0),
)
    dims = Int.(collect(dims))
    !isempty(dims) && all(>(0), dims) || throw(ArgumentError("invalid mode dimensions"))
    maxdim > 0 && maxsweeps > 0 && validation_samples > 0 ||
        throw(ArgumentError("rank, sweep and sample limits must be positive"))
    isfinite(tolerance) && tolerance > 0 ||
        throw(ArgumentError("tolerance must be finite and positive"))
    oversampling >= 0 || throw(ArgumentError("oversampling must be nonnegative"))
    d = length(dims)
    sites =
        isnothing(sites) ? [Index(n, "Site,n=$k") for (k, n) in enumerate(dims)] : collect(sites)
    length(sites) == d && dim.(sites) == dims || throw(DimensionMismatch("site dimensions"))
    caps = ranks isa Integer ? fill(Int(ranks), d-1) : Int.(collect(ranks))
    length(caps) == d-1 && all(>(0), caps) || throw(ArgumentError("invalid rank budgets"))
    ranks isa Integer && ranks <= 0 && throw(ArgumentError("rank must be positive"))
    caps = min.(caps, maxdim)
    cache = Dict{Tuple,Number}()
    function sample(idx)
        key = Tuple(idx)
        return get!(cache, key) do
            value = f(key)
            value isa Number && isfinite(value) ||
                throw(ArgumentError("callback must return finite scalars"))
            return float(value)
        end
    end
    # Keep scalar types concrete in the fibers, despite a heterogeneous cache.
    T = typeof(sample(Tuple(ones(Int, d))))
    typed_sample(idx) = convert(T, sample(idx))
    validation = _random_tuples(rng, dims, validation_samples)
    truth = typed_sample.(validation)
    if d == 1
        c = [reshape([typed_sample((i,)) for i in 1:dims[1]], 1, dims[1], 1)]
        return CrossResult(cores_to_mps(c; sites), [0.0], 1, length(cache), true)
    end
    right = [_random_tuples(rng, dims[(k + 1):end], caps[k]+oversampling) for k in 1:(d - 1)]
    errors = Float64[]
    cores = Array[]
    current_dims = copy(dims)
    success = false
    for sweep in 1:maxsweeps
        backward = iseven(sweep)
        oriented(idx) = typed_sample(backward ? reverse(idx) : idx)
        cores, extended = _cross_sweep(
            oriented, current_dims, right, caps, tolerance, adaptive, oversampling
        )
        original = backward ? reverse_cores(cores) : cores
        prediction = [_evaluate(original, idx) for idx in validation]
        denominator = norm(truth)
        err = iszero(denominator) ? norm(prediction - truth) : norm(prediction - truth)/denominator
        push!(errors, err)
        # At least one pair of opposite sweeps unless the caller asks for one.
        if err <= tolerance && (sweep >= 2 || maxsweeps == 1)
            cores = original
            success = true
            break
        end
        if sweep == maxsweeps
            cores = original
            break
        end
        if adaptive && err > tolerance && iseven(sweep)
            caps = min.(maxdim, max.(caps .+ 1, 2 .* caps))
        end
        current_dims = reverse(current_dims)
        caps = reverse(caps)
        right = [Tuple[reverse(t) for t in extended[d - k]] for k in 1:(d - 1)]
        for k in 1:(d - 1)
            right[k] = _random_tuples(
                rng, current_dims[(k + 1):end], caps[k]+oversampling; existing=right[k]
            )
        end
    end
    return CrossResult(cores_to_mps(cores; sites), errors, length(errors), length(cache), success)
end

"""TT-cross of an existing array; tensor axes correspond directly to MPS sites."""
tt_cross_tensor(A::AbstractArray; kwargs...) = tt_cross(i -> A[i...], size(A); kwargs...)
