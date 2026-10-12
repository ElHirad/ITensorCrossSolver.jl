function _check_operator_sites(A::MPO, u::MPS)
    length(A) == length(u) && all(k -> siteind(A, k; plev=0) == siteind(u, k), 1:length(u)) ||
        throw(DimensionMismatch("operator and state must share the same site indices"))
    return nothing
end

"""
    burgers_rhs(D1, D2; viscosity, cutoff=1e-14, maxdim=64)

Build an RHS factory for `u_t = -u*u_x + viscosity*u_xx`. Derivatives are applied
as MPOs once per RK stage; nonlinear products are evaluated only at cross samples.
The returned `rhs(u,t)` is an index-tuple callback, usable by `approximate_rhs`.
Centered spatial differences require a resolved, smooth viscous solution.
"""
function burgers_rhs(D1::MPO, D2::MPO; viscosity::Real, cutoff::Real=1e-14, maxdim::Integer=64)
    isfinite(viscosity) && viscosity >= 0 ||
        throw(ArgumentError("viscosity must be finite and nonnegative"))
    isfinite(cutoff) && cutoff >= 0 && maxdim > 0 ||
        throw(ArgumentError("invalid truncation parameters"))
    function rhs(u::MPS, t)
        _check_operator_sites(D1, u)
        _check_operator_sites(D2, u)
        c = mps_to_cores(u)
        cx = mps_to_cores(apply(D1, u; alg="naive", cutoff, maxdim))
        cxx = mps_to_cores(apply(D2, u; alg="naive", cutoff, maxdim))
        return idx -> -_evaluate(c, idx) * _evaluate(cx, idx) + viscosity * _evaluate(cxx, idx)
    end
    return rhs
end

"""
    burgers_rhs(nbits, dx; viscosity, boundary=:periodic, cache=true)

Pure TT-CUR-DEIM Burgers RHS using physical-neighbor finite differences, with
no derivative MPOs. `nbits` is the number of binary sites in the 1D grid.
The spatial stencil and nonlinear term match the MPO-based `burgers_rhs`.
"""
function burgers_rhs(
    nbits::Integer, dx::Real; viscosity::Real, boundary=:periodic, cache::Bool=true
)
    isfinite(viscosity) && viscosity >= 0 ||
        throw(ArgumentError("viscosity must be finite and nonnegative"))
    _stencil_grid((nbits,), (dx,), boundary)
    function rhs(u::MPS, t)
        samples = finite_difference_samples(u, (nbits,), (dx,); boundary, cache)
        return idx -> -samples.value(idx)*samples.first(idx, 1) + viscosity*samples.second(idx, 1)
    end
    return rhs
end

"""
    schrodinger_rhs(laplacians, potential; epsilon=1e-4, cutoff=1e-14, maxdim=64)

RHS factory for `ψ_t = (im/2)Σ D₂ψ - im*V*ψ - im*epsilon*abs2(ψ)*ψ`, matching
the Python example. `potential` is a scalar, an MPS, or a callback `(idx,t)`.
Use `multidim_derivative_mpo` for the per-axis second derivatives.
"""
function schrodinger_rhs(
    laplacians::AbstractVector{<:MPO},
    potential;
    epsilon::Real=1e-4,
    cutoff::Real=1e-14,
    maxdim::Integer=64,
)
    !isempty(laplacians) || throw(ArgumentError("at least one derivative is required"))
    isfinite(epsilon) || throw(ArgumentError("epsilon must be finite"))
    isfinite(cutoff) && cutoff >= 0 && maxdim > 0 ||
        throw(ArgumentError("invalid truncation parameters"))
    pc = potential isa MPS ? mps_to_cores(potential) : nothing
    function rhs(u::MPS, t)
        for A in laplacians
            _check_operator_sites(A, u)
        end
        if potential isa MPS
            dim.(siteinds(potential)) == dim.(siteinds(u)) ||
                throw(DimensionMismatch("potential dimensions"))
        end
        c = mps_to_cores(u)
        derivatives = [mps_to_cores(apply(A, u; alg="naive", cutoff, maxdim)) for A in laplacians]
        function sampled(idx)
            value = _evaluate(c, idx)
            V = if potential isa Number
                potential
            elseif isnothing(pc)
                potential(idx, t)
            else
                _evaluate(pc, idx)
            end
            lap = sum(_evaluate(dc, idx) for dc in derivatives)
            return (im/2)*lap - im*V*value - im*epsilon*abs2(value)*value
        end
        return sampled
    end
    return rhs
end

"""
    schrodinger_rhs(bits_per_dim, spacings, potential; epsilon=1e-4, boundary=:dirichlet, cache=true)

Pure TT-CUR-DEIM version of the nonlinear Schrödinger RHS. Physical-neighbor
stencils replace derivative MPO application; the potential and nonlinear term
have the same conventions as the MPO overload. Supports real or complex MPSs.
"""
function schrodinger_rhs(
    bits_per_dim, spacings, potential; epsilon::Real=1e-4, boundary=:dirichlet, cache::Bool=true
)
    isfinite(epsilon) || throw(ArgumentError("epsilon must be finite"))
    bits, dx, boundaries, _ = _stencil_grid(bits_per_dim, spacings, boundary)
    pc = potential isa MPS ? mps_to_cores(potential) : nothing
    function rhs(u::MPS, t)
        samples = finite_difference_samples(u, bits, dx; boundary=boundaries, cache)
        if potential isa MPS
            dim.(siteinds(potential)) == dim.(siteinds(u)) ||
                throw(DimensionMismatch("potential dimensions"))
        end
        function sampled(idx)
            value = samples.value(idx)
            V = if potential isa Number
                potential
            elseif isnothing(pc)
                potential(idx, t)
            else
                _evaluate(pc, idx)
            end
            lap = sum(samples.second(idx, axis) for axis in eachindex(bits))
            return (im/2)*lap-im*V*value-im*epsilon*abs2(value)*value
        end
        return sampled
    end
    return rhs
end

function _cross_state(f, u; cross_kwargs, strict)
    result = tt_cross(f, dim.(siteinds(u)); sites=siteinds(u), cross_kwargs...)
    if strict && !result.converged
        error(
            "TT-cross failed to converge: sampled relative error $(result.errors[end]). Increase maxdim/maxsweeps or relax tolerance.",
        )
    end
    return result
end

"""Cross-approximate `rhs(u,t)`, returning a `CrossResult`; by default reject failed convergence."""
function approximate_rhs(rhs, u::MPS, t::Real; cross_kwargs::NamedTuple=(;), strict::Bool=true)
    isfinite(t) || throw(ArgumentError("time must be finite"))
    return _cross_state(rhs(u, t), u; cross_kwargs, strict)
end

"""Evaluate a fiber of `u + dt*(k₁ + 2k₂ + 2k₃ + k₄)/6` for arbitrary stage ranks."""
function rk4_fiber(u::MPS, stages, dt::Real, mode::Integer, left, right)
    length(stages) == 4 || throw(ArgumentError("four RK stages are required"))
    isfinite(dt) || throw(ArgumentError("dt must be finite"))
    dims = dim.(siteinds(u))
    all(s -> dim.(siteinds(s)) == dims, stages) || throw(DimensionMismatch("stage dimensions"))
    cs = mps_to_cores(u)
    ks = mps_to_cores.(stages)
    f(idx) =
        _evaluate(cs, idx) + dt/6 * sum(w*_evaluate(k, idx) for (w, k) in zip((1, 2, 2, 1), ks))
    return evaluate_fiber(f, dims, mode, left, right)
end

function _cross_combination(states, weights; cross_kwargs, strict)
    cores = mps_to_cores.(states)
    f(idx) = sum(w*_evaluate(c, idx) for (w, c) in zip(weights, cores))
    return _cross_state(f, first(states); cross_kwargs, strict).mps
end

"""
    rk4_step(rhs, u, t, dt; cross_kwargs=(;), strict=true)

One classical explicit RK4 step, with TT-cross compression of every RHS, stage
state, and final combination. `rhs(u,t)` must return an entry callback. No dense
tensor is built. `strict=true` throws if any cross approximation fails validation.
This controls cross error, not temporal error or explicit-method stability.
"""
function rk4_step(rhs, u::MPS, t::Real, dt::Real; cross_kwargs::NamedTuple=(;), strict::Bool=true)
    isfinite(t) && isfinite(dt) && dt > 0 ||
        throw(ArgumentError("finite time and positive dt required"))
    k1 = approximate_rhs(rhs, u, t; cross_kwargs, strict).mps
    u2 = _cross_combination((u, k1), (1, dt/2); cross_kwargs, strict)
    k2 = approximate_rhs(rhs, u2, t+dt/2; cross_kwargs, strict).mps
    u3 = _cross_combination((u, k2), (1, dt/2); cross_kwargs, strict)
    k3 = approximate_rhs(rhs, u3, t+dt/2; cross_kwargs, strict).mps
    u4 = _cross_combination((u, k3), (1, dt); cross_kwargs, strict)
    k4 = approximate_rhs(rhs, u4, t+dt; cross_kwargs, strict).mps
    return _cross_combination(
        (u, k1, k2, k3, k4), (1, dt/6, dt/3, dt/3, dt/6); cross_kwargs, strict
    )
end

"""Saved `times` and independent MPS `states` from `solve_pde`, including both endpoints."""
struct PDESolution
    times::Vector{Float64}
    states::Vector{MPS}
end

"""
    solve_pde(rhs, initial, (t0, tf); dt, saveevery=1, cross_kwargs=(;), strict=true)

Advance with fixed-step RK4, shortening the final step to land on `tf` exactly.
Saves the initial state, every `saveevery` steps, and the final state. Choose `dt`
to satisfy spatial CFL/diffusion restrictions; there is no adaptive time controller.
"""
function solve_pde(
    rhs,
    initial::MPS,
    tspan;
    dt::Real,
    saveevery::Integer=1,
    cross_kwargs::NamedTuple=(;),
    strict::Bool=true,
)
    length(tspan) == 2 || throw(ArgumentError("tspan must have two endpoints"))
    t0, tf = Float64.(tspan)
    isfinite(t0) && isfinite(tf) && tf >= t0 || throw(ArgumentError("invalid time interval"))
    isfinite(dt) && dt > 0 && saveevery > 0 ||
        throw(ArgumentError("dt and saveevery must be positive"))
    times = [t0]
    states = MPS[deepcopy(initial)]
    u = deepcopy(initial)
    t = t0
    step = 0
    while t < tf
        next_t = min(tf, t0 + (step+1)*Float64(dt))
        h = next_t-t
        t + h > t || throw(ArgumentError("dt is too small to advance time"))
        u = rk4_step(rhs, u, t, h; cross_kwargs, strict)
        t = next_t
        step += 1
        if step % saveevery == 0 || t == tf
            push!(times, t)
            push!(states, deepcopy(u))
        end
    end
    return PDESolution(times, states)
end
