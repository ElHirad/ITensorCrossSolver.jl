using ITensorCrossSolver
using LinearAlgebra

"""Periodic Cole–Hopf solution of u_t + u*u_x = ν*u_xx on [0, 2π)."""
function burgers_exact(x, t; viscosity=0.1, amplitude=0.5)
    a = amplitude*exp(-viscosity*t)
    return 2viscosity*a*sin(x)/(1+a*cos(x))
end

"""
Run periodic viscous Burgers in quantics MPS form. The smooth Cole–Hopf initial
condition supplies an exact solution for validation. Return the solution and
error; write a CSV with `write_output=true` (no plotting packages required).
`derivative_method=:pure` selects sampled neighbor stencils (Approach 1);
the default `:hybrid` selects derivative MPOs (Approach 2), with identical RK4.
"""
function burgers_example(;
    nbits=6,
    viscosity=0.1,
    amplitude=0.5,
    final_time=0.1,
    dt=0.01,
    write_output=true,
    derivative_method=:hybrid,
)
    derivative_method in (:pure, :hybrid) ||
        throw(ArgumentError("derivative_method must be :pure or :hybrid"))
    viscosity > 0 || throw(ArgumentError("viscosity must be positive"))
    abs(amplitude) < 1 || throw(ArgumentError("Cole–Hopf amplitude must have magnitude < 1"))
    n = 2^nbits
    dx = 2pi/n
    x = collect(0:(n - 1)) .* dx # no duplicated periodic endpoint
    sites = quantics_sites(nbits)
    exact(x, t) = burgers_exact(x, t; viscosity, amplitude)
    cross = (; ranks=4, maxdim=24, tolerance=1e-10, maxsweeps=20, oversampling=2)
    initial = tt_cross(quantics_function(x -> exact(x, 0.0), (x,)), fill(2, nbits); sites, cross...)
    initial.converged || error("initial condition cross approximation did not converge")
    # Conservative diagnostic for the explicit RK4 step used in this example.
    umax = maximum(abs, exact.(x, 0.0))
    dt * (umax/dx + 4viscosity/dx^2) <= 2 ||
        throw(ArgumentError("reduce dt for this grid/viscosity (explicit stability restriction)"))
    rhs = if derivative_method == :hybrid
        D1 = finite_difference_mpo(sites, dx; order=1, boundary=:periodic)
        D2 = finite_difference_mpo(sites, dx; order=2, boundary=:periodic)
        burgers_rhs(D1, D2; viscosity, cutoff=1e-20, maxdim=32)
    else
        burgers_rhs(nbits, dx; viscosity)
    end
    solution = solve_pde(rhs, initial.mps, (0.0, final_time); dt, cross_kwargs=cross)
    numerical = real.(dense_vector(solution.states[end]))
    reference = exact.(x, final_time)
    reference_norm = norm(reference)
    relative_error = if iszero(reference_norm)
        norm(numerical-reference)
    else
        norm(numerical-reference)/reference_norm
    end
    ranks = bond_dimensions(solution.states[end])
    if write_output
        directory = joinpath(@__DIR__, "output")
        mkpath(directory)
        path = joinpath(directory, "burgers_1d.csv")
        open(path, "w") do io
            println(io, "x,initial,numerical,exact,error")
            for k in eachindex(x)
                println(
                    io,
                    join(
                        (
                            x[k],
                            exact(x[k], 0.0),
                            numerical[k],
                            reference[k],
                            numerical[k]-reference[k],
                        ),
                        ',',
                    ),
                )
            end
        end
        println("Wrote ", path)
    end
    println(
        "Burgers: N=$n, t=$(solution.times[end]), relative L2 error=$relative_error, ranks=$ranks"
    )
    return (; solution, x, numerical, reference, error=relative_error, ranks)
end

if abspath(PROGRAM_FILE) == @__FILE__
    BLAS.set_num_threads(1)
    burgers_example()
end
