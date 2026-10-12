"""Create `n` binary ITensor site indices, most significant bit first."""
function quantics_sites(n::Integer)
    n > 0 || throw(ArgumentError("number of bits must be positive"))
    return [Index(2, "Site,Qubit,n=$k") for k in 1:n]
end

"""Encode a 1-based grid index as 1-based binary site values, MSB first."""
function quantics_index(i::Integer, n::Integer)
    0 < n < 8sizeof(Int)-1 || throw(ArgumentError("unsupported bit count"))
    1 <= i <= 2^n || throw(BoundsError(1:(2 ^ n), i))
    return [Int(((i - 1) >> (n - k)) & 1) + 1 for k in 1:n]
end

"""Decode 1-based binary site values, most significant bit first."""
function grid_index(bits)
    0 < length(bits) < 8sizeof(Int)-1 || throw(ArgumentError("unsupported bit count"))
    all(b -> b in (1, 2), bits) || throw(ArgumentError("binary indices must be 1 or 2"))
    i = 0
    for b in bits
        i = 2i + b - 1
    end
    return i + 1
end

"""
    quantics_function(f, grids)

Wrap `f(x₁, x₂, …)` as an index-tuple callback for TT-cross. Each grid must
have power-of-two length. Coordinate bits are grouped by dimension, MSB first.
"""
function quantics_function(f, grids)
    ns = Int[]
    for grid in grids
        n = length(grid)
        n >= 2 && ispow2(n) || throw(ArgumentError("grid lengths must be powers of two ≥ 2"))
        push!(ns, trailing_zeros(n))
    end
    isempty(ns) && throw(ArgumentError("at least one grid is required"))
    function sampled(idx)
        length(idx) == sum(ns) || throw(DimensionMismatch("incorrect number of sites"))
        offset = 0
        xs = map(eachindex(ns)) do k
            i = grid_index(idx[(offset + 1):(offset + ns[k])])
            offset += ns[k]
            return grids[k][i]
        end
        return f(xs...)
    end
    return sampled
end

function _check_cores(cores)
    isempty(cores) && throw(ArgumentError("at least one core is required"))
    all(c -> ndims(c) == 3 && all(>(0), size(c)), cores) ||
        throw(ArgumentError("cores must have shape (left, physical, right)"))
    size(first(cores), 1) == size(last(cores), 3) == 1 ||
        throw(DimensionMismatch("boundary bond dimensions must be one"))
    all(k -> size(cores[k], 3) == size(cores[k + 1], 1), 1:(length(cores) - 1)) ||
        throw(DimensionMismatch("adjacent bond dimensions do not match"))
    return nothing
end

"""Convert `(left, physical, right)` TT cores to an ITensorMPS `MPS`."""
function cores_to_mps(cores; sites=[Index(size(c, 2), "Site,n=$k") for (k, c) in enumerate(cores)])
    _check_cores(cores)
    n = length(cores)
    length(sites) == n && all(k -> dim(sites[k]) == size(cores[k], 2), 1:n) ||
        throw(DimensionMismatch("site dimensions do not match cores"))
    length(unique(sites)) == n || throw(ArgumentError("site indices must be distinct"))
    links = [Index(size(cores[k], 3), "Link,l=$k") for k in 1:(n - 1)]
    tensors = ITensor[]
    for k in 1:n
        is = Index[]
        k > 1 && push!(is, links[k - 1])
        push!(is, sites[k])
        k < n && push!(is, links[k])
        push!(tensors, ITensor(reshape(copy(cores[k]), Tuple(dim.(is))), is...))
    end
    return MPS(tensors)
end

"""Extract TT cores in `(left, physical, right)` order, including unit end bonds."""
function mps_to_cores(psi::MPS)
    n = length(psi)
    n > 0 || throw(ArgumentError("empty MPS"))
    return map(1:n) do k
        s = siteind(psi, k)
        is = Index[]
        k > 1 && push!(is, linkind(psi, k-1))
        push!(is, s)
        k < n && push!(is, linkind(psi, k))
        l = k == 1 ? 1 : dim(first(is))
        r = k == n ? 1 : dim(last(is))
        return reshape(Array(psi[k], is...), l, dim(s), r)
    end
end

"""Reverse site order and swap left/right core bonds, without conjugation."""
function reverse_cores(cores)
    _check_cores(cores)
    return [permutedims(c, (3, 2, 1)) for c in reverse(cores)]
end

function _evaluate(cores, idx)
    v = ones(eltype(first(cores)), 1, 1)
    for k in eachindex(cores)
        v = v * @view(cores[k][:, idx[k], :])
    end
    return only(v)
end

"""Evaluate one entry of an MPS or core list without reconstructing its tensor."""
function evaluate(cores::AbstractVector, idx)
    _check_cores(cores)
    length(idx) == length(cores) || throw(DimensionMismatch("incorrect index length"))
    all(k -> 1 <= idx[k] <= size(cores[k], 2), eachindex(cores)) || throw(BoundsError(cores, idx))
    return _evaluate(cores, idx)
end
evaluate(psi::MPS, idx) = evaluate(mps_to_cores(psi), idx)

"""
    evaluate_fiber(f, dims, mode, left, right)

Sample `f((left..., i, right...))` for all site values at `mode`. `left` and
`right` are lists of index tuples; use `[()]` at a boundary. Returns a tensor
of size `(length(left), dims[mode], length(right))`.
"""
function evaluate_fiber(f, dims, mode::Integer, left, right)
    1 <= mode <= length(dims) || throw(ArgumentError("invalid mode"))
    !isempty(left) && !isempty(right) || throw(ArgumentError("empty sample set"))
    all(>(0), dims) || throw(ArgumentError("mode dimensions must be positive"))
    all(x -> length(x) == mode-1, left) || throw(DimensionMismatch("invalid prefix"))
    all(x -> length(x) == length(dims)-mode, right) || throw(DimensionMismatch("invalid suffix"))
    for (samples, sizes) in ((left, dims[1:(mode - 1)]), (right, dims[(mode + 1):end]))
        all(t -> all(k -> 1 <= t[k] <= sizes[k], eachindex(sizes)), samples) ||
            throw(ArgumentError("fiber indices out of bounds"))
    end
    return [f((l..., i, r...)) for l in left, i in 1:dims[mode], r in right]
end
function evaluate_fiber(psi::MPS, mode::Integer, left, right)
    c = mps_to_cores(psi)
    return evaluate_fiber(idx -> _evaluate(c, idx), size.(c, 2), mode, left, right)
end

"""Reconstruct a small MPS as a Julia array with axes in site order."""
dense_tensor(psi::MPS) = Array(prod(psi), siteinds(psi)...)

"""Reconstruct in lexicographic/MSB-first grid order (last site varies fastest)."""
function dense_vector(psi::MPS)
    A = dense_tensor(psi)
    return vec(permutedims(A, reverse(1:ndims(A))))
end

"""Return the internal TT bond dimensions."""
bond_dimensions(psi::MPS) = [dim(linkind(psi, k)) for k in 1:(length(psi) - 1)]

"""Reconstruct a small MPO as a matrix in the same grid order as `dense_vector`."""
function dense_operator(A::MPO)
    sites = [siteind(A, k; plev=0) for k in 1:length(A)]
    tensor = Array(prod(A), prime.(reverse(sites))..., reverse(sites)...)
    n = prod(dim.(sites))
    return reshape(tensor, n, n)
end
