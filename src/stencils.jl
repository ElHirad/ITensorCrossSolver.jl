function _stencil_grid(bits_per_dim, spacings, boundary)
    bits = Tuple(bits_per_dim)
    !isempty(bits) && all(n -> n isa Integer && 0 < n < 8sizeof(Int)-1, bits) ||
        throw(ArgumentError("each physical axis needs a positive, supported bit count"))
    dx = Tuple(spacings)
    length(dx) == length(bits) || throw(DimensionMismatch("one spacing per physical axis required"))
    all(h -> h isa Real && isfinite(h) && h > 0, dx) ||
        throw(ArgumentError("spacings must be finite and positive"))
    boundaries = boundary isa Symbol ? ntuple(_ -> boundary, length(bits)) : Tuple(boundary)
    length(boundaries) == length(bits) ||
        throw(DimensionMismatch("one boundary per physical axis required"))
    all(b -> b in (:periodic, :dirichlet, :homogeneous), boundaries) ||
        throw(ArgumentError("unsupported boundary"))
    offsets = cumsum([0; collect(Int, bits)])
    return Int.(bits), dx, boundaries, offsets
end

"""
    finite_difference_samples(u, bits_per_dim, spacings; boundary=:periodic, cache=true)

Approach 1 (pure TT-CUR-DEIM): return callbacks `value(idx)`, `first(idx,axis)`,
and `second(idx,axis)` for an MPS and its centered second-order finite differences.
Only the requested entries and physical-grid stencil neighbors are contracted;
no derivative MPO, derivative MPS, or dense tensor is constructed.

Bits are grouped by physical axis, most significant first. Carry/borrow and
periodic wrapping affect only the selected axis. `boundary` is a symbol or one
symbol per axis; `:dirichlet`/`:homogeneous` use zero values outside an interior
grid, as in `multidim_derivative_mpo`. Spacings must match that grid convention.

The callbacks share a snapshot of `u`. With `cache=true`, each distinct center
or auxiliary entry is contracted at most once per evaluator. The cache lives
only as long as these callbacks, and uses memory proportional to the sampled
entries (potentially the entire grid on a small problem). Set `cache=false` to
trade repeated contractions for lower memory use. Create a new evaluator for
each RK stage; `rk4_step`/`solve_pde` already build their RHS stage by stage.
"""
function finite_difference_samples(
    u::MPS, bits_per_dim, spacings; boundary=:periodic, cache::Bool=true
)
    bits, dx, boundaries, offsets = _stencil_grid(bits_per_dim, spacings, boundary)
    sum(bits) == length(u) && all(s -> dim(s) == 2, siteinds(u)) ||
        throw(DimensionMismatch("bit counts must match a binary MPS"))
    cores = mps_to_cores(u)
    # Include the index length in the key type; avoid abstract Number-valued caches.
    T = promote_type(eltype.(cores)...)
    entries = Dict{NTuple{length(u),Int},T}()
    function checked_index(idx)
        length(idx) == offsets[end] || throw(DimensionMismatch("incorrect index length"))
        all(i -> i isa Integer && i in (1, 2), idx) ||
            throw(ArgumentError("binary indices must be 1 or 2"))
        return Tuple(Int.(idx))
    end
    function entry(idx)
        cache || return _evaluate(cores, idx)
        return get!(entries, idx) do
            return _evaluate(cores, idx)
        end
    end
    function neighbor(idx, axis, shift)
        a, b = offsets[axis]+1, offsets[axis + 1]
        coordinate = grid_index(idx[a:b]) + shift
        n = 1 << bits[axis]
        if !(1 <= coordinate <= n)
            boundaries[axis] == :periodic || return zero(T)
            coordinate = mod1(coordinate, n)
        end
        # Encode this physical coordinate without carrying into an adjacent axis.
        adjacent = ntuple(length(idx)) do k
            return a <= k <= b ? Int(((coordinate-1) >> (b-k)) & 1)+1 : idx[k]
        end
        return entry(adjacent)
    end
    function derivative(idx, axis, order)
        axis isa Integer && 1 <= axis <= length(bits) ||
            throw(ArgumentError("invalid physical axis"))
        key = checked_index(idx)
        plus, minus = neighbor(key, axis, 1), neighbor(key, axis, -1)
        return order == 1 ? (plus-minus)/(2dx[axis]) : (plus-2entry(key)+minus)/dx[axis]^2
    end
    return (;
        value=idx -> entry(checked_index(idx)),
        first=(idx, axis) -> derivative(idx, axis, 1),
        second=(idx, axis) -> derivative(idx, axis, 2),
    )
end
