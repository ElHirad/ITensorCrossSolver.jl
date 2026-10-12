using ITensorCrossSolver
using ITensorMPS
using LinearAlgebra

"""Smooth, nonseparable periodic initial data, bounded between 0.1 and 0.5."""
adr2d_initial(x, y) = 0.3 + 0.12sin(x)*cos(y) + 0.08cos(2x-y)

"""
Build a sampled RHS for
    u_t = -vx*u_x - vy*u_y + Dx*u_xx + Dy*u_yy + r*u*(1-u).

`operators` contains the MPOs `x`, `y`, `xx`, and `yy`. Apply them once per
stage, then evaluate the nonlinear reaction only at the cross sample points.
This example uses the package's public MPS/core and PDE interfaces.
"""
function adr2d_cross_rhs(
    operators;
    velocity=(1.0, -0.5),
    diffusivity=(0.05, 0.025),
    reaction_rate=1.0,
    cutoff=1e-22,
    maxdim=64,
)
    function rhs(u, t)
        c = mps_to_cores(u)
        derivatives = map(A -> mps_to_cores(apply(A, u; alg="naive", cutoff, maxdim)), operators)
        function sampled(idx)
            value = evaluate(c, idx)
            return -velocity[1]*evaluate(derivatives.x, idx) -
                   velocity[2]*evaluate(derivatives.y, idx) +
                   diffusivity[1]*evaluate(derivatives.xx, idx) +
                   diffusivity[2]*evaluate(derivatives.yy, idx) +
                   reaction_rate*value*(1-value)
        end
        return sampled
    end
    return rhs
end

"""Approach 1 RHS: evaluate physical-neighbor stencils only at cross samples."""
function adr2d_cross_rhs(
    bits_per_dim,
    spacings;
    velocity=(1.0, -0.5),
    diffusivity=(0.05, 0.025),
    reaction_rate=1.0,
    cache=true,
)
    function rhs(u, t)
        samples = finite_difference_samples(u, bits_per_dim, spacings; boundary=:periodic, cache)
        function sampled(idx)
            value = samples.value(idx)
            return -velocity[1]*samples.first(idx, 1) - velocity[2]*samples.first(idx, 2) +
                   diffusivity[1]*samples.second(idx, 1) +
                   diffusivity[2]*samples.second(idx, 2) +
                   reaction_rate*value*(1-value)
        end
        return sampled
    end
    return rhs
end

"""
Independent dense/DNS spatial RHS on a periodic grid, with `U[i,j] = u(xᵢ,yⱼ)`.
Uses the same centered second-order stencils as the MPOs, evaluated explicitly
at every grid point. No MPO reconstruction or tensor compression is used here.
"""
function adr2d_dns_rhs(
    U, dx, dy; velocity=(1.0, -0.5), diffusivity=(0.05, 0.025), reaction_rate=1.0
)
    nx, ny = size(U)
    result = similar(U, promote_type(eltype(U), Float64))
    for j in 1:ny, i in 1:nx
        ip, im = mod1(i+1, nx), mod1(i-1, nx)
        jp, jm = mod1(j+1, ny), mod1(j-1, ny)
        u = U[i, j]
        ux = (U[ip, j]-U[im, j])/(2dx)
        uy = (U[i, jp]-U[i, jm])/(2dy)
        uxx = (U[ip, j]-2u+U[im, j])/dx^2
        uyy = (U[i, jp]-2u+U[i, jm])/dy^2
        result[i, j] =
            -velocity[1]*ux - velocity[2]*uy +
            diffusivity[1]*uxx +
            diffusivity[2]*uyy +
            reaction_rate*u*(1-u)
    end
    return result
end

"""One independent, uncompressed classical RK4 step for the DNS field."""
function adr2d_dns_step(U, h, dx, dy; kwargs...)
    isfinite(h) && h > 0 || throw(ArgumentError("step size must be finite and positive"))
    k1 = adr2d_dns_rhs(U, dx, dy; kwargs...)
    k2 = adr2d_dns_rhs(U+h*k1/2, dx, dy; kwargs...)
    k3 = adr2d_dns_rhs(U+h*k2/2, dx, dy; kwargs...)
    k4 = adr2d_dns_rhs(U+h*k3, dx, dy; kwargs...)
    return U+h*(k1+2k2+2k3+k4)/6
end

"""
Report the quadrature-weighted discrete L2 norm, sqrt(dx*dy*sum(abs2, field)),
for each solution and their difference. Also return relative L2 error and the
unweighted Euclidean error. A nonzero error against a zero DNS field has Inf
relative error; two zero fields have zero relative error.
"""
function adr2d_l2_metrics(numerical, dns, dx, dy)
    size(numerical) == size(dns) || throw(DimensionMismatch("solution grids must match"))
    isfinite(dx) && dx > 0 && isfinite(dy) && dy > 0 ||
        throw(ArgumentError("grid spacings must be finite and positive"))
    weight = sqrt(dx*dy)
    euclidean_l2_error = norm(numerical-dns)
    l2_error = weight*euclidean_l2_error
    tt_l2 = weight*norm(numerical)
    dns_l2 = weight*norm(dns)
    relative_l2_error = iszero(dns_l2) ? (iszero(l2_error) ? 0.0 : Inf) : l2_error/dns_l2
    return (; tt_l2, dns_l2, l2_error, relative_l2_error, euclidean_l2_error)
end

"""
    adr2d_example(; nbits=(5,5), final_time=0.2, dt=0.01, ...)

Solve periodic 2D advection–diffusion–logistic-reaction with TT-cross and DNS.
The two methods use the same prescribed initial data, grid, centered spatial
stencils, RK4 coefficients, and time steps (including the shortened final step).
The DNS uses dense arrays and no compression. Thus their discrepancy measures
the accumulated TT/MPO approximation and roundoff on this discretization, not
the error relative to the continuum PDE.

`nbits` may be a scalar for a square grid or a two-tuple for a rectangular grid.
`cross_kwargs` overrides the TT-cross settings; `operator_cutoff` and
`operator_maxdim` control derivative-MPO application. All cross convergence
checks remain strict. Returns both final fields, the TT solution history,
error metrics at every step, and parameters. Optional CSV files contain final
fields and the complete L2-error history.

Choose `derivative_method=:pure` for the paper's Approach 1 (sampled physical
neighbors), or `:hybrid` for Approach 2 (the default derivative MPOs). Both use
the identical RK4 stage-compression algorithm. `sample_cache` controls reuse
of auxiliary state entries in the pure method; MPO settings affect only hybrid.
"""
function adr2d_example(;
    nbits=(5, 5),
    final_time=0.2,
    dt=0.01,
    velocity=(1.0, -0.5),
    diffusivity=(0.05, 0.025),
    reaction_rate=1.0,
    cross_kwargs=(;),
    operator_cutoff=1e-22,
    operator_maxdim=64,
    derivative_method=:hybrid,
    sample_cache=true,
    write_output=true,
    output_dir=joinpath(@__DIR__, "output"),
)
    derivative_method in (:pure, :hybrid) ||
        throw(ArgumentError("derivative_method must be :pure or :hybrid"))
    bits = nbits isa Integer ? (nbits, nbits) : Tuple(nbits)
    length(bits) == 2 && all(n -> n isa Integer && 1 <= n < 8sizeof(Int)-1, bits) ||
        throw(ArgumentError("nbits must contain two positive bit counts"))
    length(velocity) == length(diffusivity) == 2 ||
        throw(ArgumentError("two velocity and diffusion components required"))
    all(isfinite, velocity) && all(d -> isfinite(d) && d >= 0, diffusivity) ||
        throw(ArgumentError("velocity must be finite and diffusivity nonnegative"))
    isfinite(reaction_rate) && reaction_rate >= 0 ||
        throw(ArgumentError("reaction rate must be finite and nonnegative"))
    isfinite(final_time) && final_time >= 0 && isfinite(dt) && dt > 0 ||
        throw(ArgumentError("finite nonnegative final time and positive dt required"))
    isfinite(operator_cutoff) &&
    operator_cutoff >= 0 &&
    operator_maxdim isa Integer &&
    operator_maxdim > 0 || throw(ArgumentError("invalid operator truncation settings"))
    nx, ny = 2 .^ bits
    dx, dy = 2pi/nx, 2pi/ny
    # Conservative example diagnostic, not an adaptive stability controller.
    rate_bound =
        abs(velocity[1])/dx +
        abs(velocity[2])/dy +
        4diffusivity[1]/dx^2 +
        4diffusivity[2]/dy^2 +
        reaction_rate
    min(dt, final_time)*rate_bound <= 2 ||
        throw(ArgumentError("reduce dt for this grid and PDE coefficients"))
    x, y = collect(0:(nx - 1)) .* dx, collect(0:(ny - 1)) .* dy
    initial_field = [adr2d_initial(xi, yj) for xi in x, yj in y]
    sites = quantics_sites(sum(bits))
    cross = merge(
        (;
            ranks=4,
            maxdim=32,
            tolerance=1e-10,
            maxsweeps=24,
            oversampling=3,
            validation_samples=128,
        ),
        cross_kwargs,
    )
    initial = tt_cross(
        quantics_function(adr2d_initial, (x, y)), fill(2, sum(bits)); sites, cross...
    )
    initial.converged || error("initial condition TT-cross did not converge")
    rhs = if derivative_method == :hybrid
        operators = (
            x=multidim_derivative_mpo(sites, bits, 1, dx; order=1, boundary=:periodic),
            y=multidim_derivative_mpo(sites, bits, 2, dy; order=1, boundary=:periodic),
            xx=multidim_derivative_mpo(sites, bits, 1, dx; order=2, boundary=:periodic),
            yy=multidim_derivative_mpo(sites, bits, 2, dy; order=2, boundary=:periodic),
        )
        adr2d_cross_rhs(
            operators;
            velocity,
            diffusivity,
            reaction_rate,
            cutoff=operator_cutoff,
            maxdim=operator_maxdim,
        )
    else
        adr2d_cross_rhs(bits, (dx, dy); velocity, diffusivity, reaction_rate, cache=sample_cache)
    end
    # saveevery=1 is essential: the DNS takes exactly these RK4 steps.
    solution = solve_pde(rhs, initial.mps, (0.0, final_time); dt, saveevery=1, cross_kwargs=cross)
    # dense_vector groups x bits first and varies y fastest; Julia arrays vary
    # their first axis fastest. Transpose this reshape to obtain U[x,y].
    to_field(psi) = permutedims(reshape(dense_vector(psi), ny, nx), (2, 1))
    dns = copy(initial_field)
    numerical = to_field(solution.states[1])
    history = NamedTuple[]
    for k in eachindex(solution.times)
        if k > 1
            h = solution.times[k]-solution.times[k - 1]
            dns = adr2d_dns_step(dns, h, dx, dy; velocity, diffusivity, reaction_rate)
            numerical = to_field(solution.states[k])
        end
        metrics = adr2d_l2_metrics(numerical, dns, dx, dy)
        push!(
            history,
            (;
                time=solution.times[k],
                metrics...,
                maxbond=maximum(bond_dimensions(solution.states[k])),
            ),
        )
    end
    metrics = adr2d_l2_metrics(numerical, dns, dx, dy)
    if write_output
        mkpath(output_dir)
        open(joinpath(output_dir, "adr_2d_fields.csv"), "w") do io
            println(io, "x,y,initial,tt,dns,error")
            for i in 1:nx, j in 1:ny
                println(
                    io,
                    join(
                        (
                            x[i],
                            y[j],
                            initial_field[i, j],
                            numerical[i, j],
                            dns[i, j],
                            numerical[i, j]-dns[i, j],
                        ),
                        ',',
                    ),
                )
            end
        end
        open(joinpath(output_dir, "adr_2d_errors.csv"), "w") do io
            println(io, join(string.(keys(first(history))), ','))
            for row in history
                println(io, join(values(row), ','))
            end
        end
        println("Wrote ADR fields and L2-error history to ", output_dir)
    end
    println(
        "2D ADR ($derivative_method): grid=$(nx)×$(ny), t=$(last(solution.times)), RK4 steps=$(length(history)-1)",
    )
    println("  TT L2_h=$(metrics.tt_l2); DNS L2_h=$(metrics.dns_l2)")
    println("  Error L2_h=$(metrics.l2_error); relative L2 error=$(metrics.relative_l2_error)")
    println(
        "  Euclidean error=$(metrics.euclidean_l2_error); final max bond=$(last(history).maxbond)"
    )
    parameters = (;
        nbits=bits,
        dx,
        dy,
        dt,
        final_time,
        velocity,
        diffusivity,
        reaction_rate,
        cross,
        operator_cutoff,
        operator_maxdim,
        derivative_method,
        sample_cache,
    )
    return (; solution, x, y, initial_field, numerical, dns, history, metrics, parameters)
end

if abspath(PROGRAM_FILE) == @__FILE__
    BLAS.set_num_threads(1)
    adr2d_example()
end
