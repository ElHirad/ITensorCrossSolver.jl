function _mpo_from_cores(cores, sites)
    n = length(sites)
    links = [Index(size(cores[k], 4), "Link,l=$k") for k in 1:(n - 1)]
    tensors = ITensor[]
    for k in 1:n
        is = Index[]
        k > 1 && push!(is, links[k - 1])
        append!(is, [prime(sites[k]), sites[k]])
        k < n && push!(is, links[k])
        push!(tensors, ITensor(reshape(cores[k], Tuple(dim.(is))), is...))
    end
    return MPO(tensors)
end

function _mpo_cores(A::MPO)
    return map(1:length(A)) do k
        s = siteind(A, k; plev=0)
        is = Index[]
        k > 1 && push!(is, linkind(A, k-1))
        append!(is, [prime(s), s])
        k < length(A) && push!(is, linkind(A, k))
        l = k == 1 ? 1 : dim(first(is))
        r = k == length(A) ? 1 : dim(last(is))
        return reshape(Array(A[k], is...), l, dim(s), dim(s), r)
    end
end

# Binary ripple-carry automaton for (S u)[x] = u[x+1]. Each bond has size 2.
function _shift_mpo(sites, periodic)
    n = length(sites)
    cores = Array{Float64,4}[]
    for k in 1:n
        W = zeros(2, 2, 2, 2)
        for output in 0:1, carry_right in 0:1
            input = (output + carry_right) % 2
            carry_left = (output + carry_right) ÷ 2
            W[carry_left + 1, output + 1, input + 1, carry_right + 1] = 1
        end
        if k == 1
            W = periodic ? sum(W; dims=1) : W[1:1, :, :, :]
        end
        k == n && (W = W[:, :, :, 2:2])
        push!(cores, W)
    end
    return _mpo_from_cores(cores, sites)
end

"""
    finite_difference_mpo(sites, dx; order=2, boundary=:periodic, direction=:forward)

Second-order centered first/second spatial derivative on a binary quantics grid.
No dense grid matrix is formed. `:dirichlet` (or `:homogeneous`) uses zero values
outside the interior-point grid, whose spacing is `(b-a)/(2^n+1)`; periodic grids
omit the duplicate endpoint and have spacing `(b-a)/2^n`.
With `direction=:reverse`, reverse both site order and bit order of the state.
"""
function finite_difference_mpo(
    sites, dx::Real; order::Integer=2, boundary::Symbol=:periodic, direction::Symbol=:forward
)
    !isempty(sites) && all(s -> dim(s) == 2, sites) || throw(ArgumentError("binary sites required"))
    isfinite(dx) && dx > 0 || throw(ArgumentError("spacing must be finite and positive"))
    order in (1, 2) || throw(ArgumentError("derivative order must be 1 or 2"))
    boundary in (:periodic, :dirichlet, :homogeneous) ||
        throw(ArgumentError("unsupported boundary"))
    direction in (:forward, :reverse) || throw(ArgumentError("unsupported direction"))
    S = _shift_mpo(sites, boundary == :periodic)
    St = _mpo_from_cores([permutedims(c, (1, 3, 2, 4)) for c in _mpo_cores(S)], sites)
    identity = _mpo_from_cores(
        [reshape(Matrix{Float64}(I, 2, 2), 1, 2, 2, 1) for _ in sites], sites
    )
    # Exact addition followed by a roundoff-level compression keeps MPO bonds small.
    A = order == 1 ? (S - St) / (2dx) : (S + St - 2identity) / dx^2
    length(A) > 1 && truncate!(A; cutoff=1e-28)
    if direction == :reverse
        cs = [permutedims(c, (4, 2, 3, 1)) for c in reverse(_mpo_cores(A))]
        A = _mpo_from_cores(cs, reverse(sites))
    end
    return A
end

"""
    multidim_derivative_mpo(sites, bits_per_dim, axis, dx; kwargs...)

Embed a derivative along 1-based physical `axis` into a grouped-bit multidimensional
grid. `bits_per_dim` is a vector or tuple of bit counts; identity operators pad other axes.
`direction=:reverse` reverses the entire network, including the padding.
"""
function multidim_derivative_mpo(
    sites, bits_per_dim, axis::Integer, dx::Real; direction::Symbol=:forward, kwargs...
)
    all(n -> n isa Integer && n > 0, bits_per_dim) && sum(bits_per_dim; init=0) == length(sites) ||
        throw(DimensionMismatch("bit counts must be positive and sum to the number of sites"))
    all(s -> dim(s) == 2, sites) || throw(ArgumentError("binary sites required"))
    1 <= axis <= length(bits_per_dim) || throw(ArgumentError("invalid physical axis"))
    direction in (:forward, :reverse) || throw(ArgumentError("unsupported direction"))
    a = sum(bits_per_dim[1:(axis - 1)]; init=0) + 1
    b = a + bits_per_dim[axis] - 1
    localop = finite_difference_mpo(sites[a:b], dx; kwargs...)
    cores = [reshape(Matrix{Float64}(I, 2, 2), 1, 2, 2, 1) for _ in sites]
    cores[a:b] = _mpo_cores(localop)
    if direction == :reverse
        cores = [permutedims(c, (4, 2, 3, 1)) for c in reverse(cores)]
        sites = reverse(sites)
    end
    return _mpo_from_cores(cores, sites)
end

"""Convert an MPS to the MPO of pointwise multiplication by its entries."""
function diagonal_mpo(psi::MPS)
    cores = map(mps_to_cores(psi)) do c
        l, n, r = size(c)
        W = zeros(eltype(c), l, n, n, r)
        for i in 1:n
            W[:, i, i, :] = c[:, i, :]
        end
        return W
    end
    return _mpo_from_cores(cores, siteinds(psi))
end

"""
    hadamard(u, v; method=:direct, cutoff=1e-12, maxdim=typemax(Int), leg=:lower)

Pointwise MPS product without conjugation. For an MPO `v`, `leg=:lower` gives
`v * diag(u)` and `leg=:upper` gives `diag(u) * v`. MPO sites are vectorized
locally, never as a full matrix. `:direct`, `:zipup`, and `:fit` use ITensorMPS
naive contraction, zip-up contraction, and variational fitting respectively.
Additional keywords (e.g. `nsweeps`) are forwarded to the contraction algorithm.
Variational fitting defaults to two sweeps and `outputlevel=0`; `nsweeps` must
be a positive integer when supplied.
Scalars multiply directly and `nothing` propagates, as in the Python helper.
"""
function hadamard(
    u::MPS,
    v::MPS;
    method::Symbol=:direct,
    cutoff::Real=1e-12,
    maxdim::Integer=typemax(Int),
    leg::Symbol=:lower,
    kwargs...,
)
    method in (:direct, :zipup, :fit) || throw(ArgumentError("unsupported product method"))
    isfinite(cutoff) && cutoff >= 0 && maxdim > 0 ||
        throw(ArgumentError("invalid truncation parameters"))
    leg in (:lower, :upper) || throw(ArgumentError("invalid operator leg"))
    dim.(siteinds(u)) == dim.(siteinds(v)) || throw(DimensionMismatch("site dimensions"))
    algorithm_kwargs =
        method == :fit ? merge((; nsweeps=2, outputlevel=0), (; kwargs...)) : (; kwargs...)
    if method == :fit
        sweeps = algorithm_kwargs.nsweeps
        sweeps isa Integer && sweeps > 0 ||
            throw(ArgumentError("nsweeps must be a positive integer"))
    end
    aligned = cores_to_mps(mps_to_cores(v); sites=siteinds(u))
    alg = method == :direct ? "naive" : String(method)
    # A one-site product needs no decomposition or variational sweep.
    if length(u) == 1
        c = mps_to_cores(u)[1] .* mps_to_cores(aligned)[1]
        return cores_to_mps([c]; sites=siteinds(u))
    end
    return apply(diagonal_mpo(u), aligned; alg, cutoff, maxdim, algorithm_kwargs...)
end

function hadamard(u::MPS, A::MPO; leg::Symbol=:lower, kwargs...)
    leg in (:lower, :upper) || throw(ArgumentError("invalid operator leg"))
    ac = _mpo_cores(A)
    uc = mps_to_cores(u)
    length(ac) == length(uc) && all(k -> size(ac[k], 2) == size(uc[k], 2), eachindex(ac)) ||
        throw(DimensionMismatch("site dimensions"))
    vs = [Index(size(c, 2)^2, "Site,vectorized,n=$k") for (k, c) in enumerate(ac)]
    av = cores_to_mps([reshape(c, size(c, 1), size(c, 2)^2, size(c, 4)) for c in ac]; sites=vs)
    weights = map(uc) do c
        l, n, r = size(c)
        W = zeros(eltype(c), l, n, n, r)
        for i in 1:n, j in 1:n
            W[:, i, j, :] = c[:, leg == :lower ? j : i, :]
        end
        return reshape(W, l, n*n, r)
    end
    product = hadamard(cores_to_mps(weights; sites=vs), av; kwargs...)
    cs = mps_to_cores(product)
    sites = [siteind(A, k; plev=0) for k in 1:length(A)]
    return _mpo_from_cores(
        [
            reshape(c, size(c, 1), dim(sites[k]), dim(sites[k]), size(c, 3)) for
            (k, c) in enumerate(cs)
        ],
        sites,
    )
end
hadamard(::Nothing, v; kwargs...) = nothing
hadamard(u, ::Nothing; kwargs...) = nothing
hadamard(::Nothing, ::Nothing; kwargs...) = nothing
hadamard(u::Number, v; kwargs...) = u * v
hadamard(u, v::Number; kwargs...) = u * v
hadamard(u::Number, v::Number; kwargs...) = u * v
hadamard(::Nothing, ::Number; kwargs...) = nothing
hadamard(::Number, ::Nothing; kwargs...) = nothing
