using ITensorCrossSolver
using LinearAlgebra

"""Small, configurable version of the Python script's nonlinear Schrödinger example."""
function schrodinger_example(; nbits=3, final_time=0.002, dt=0.001, derivative_method=:hybrid)
    derivative_method in (:pure, :hybrid) ||
        throw(ArgumentError("derivative_method must be :pure or :hybrid"))
    n = 2^nbits
    dx = pi/(n+1) # interior unknowns; zero Dirichlet data at 0 and π
    x = collect(1:n) .* dx
    sites = quantics_sites(2nbits)
    cross = (; ranks=4, maxdim=24, maxsweeps=20, tolerance=1e-9)
    ic(x, y) = complex(0.3994sin(x)*sin(2y))
    w(x) = 1+exp(cos(x)^2)+0.75*(1+exp(sin(x)^2))
    initial = tt_cross(quantics_function(ic, (x, x)), fill(2, 2nbits); sites, cross...)
    potential = tt_cross(
        quantics_function((x, y)->w(x)+w(y), (x, x)), fill(2, 2nbits); sites, cross...
    )
    initial.converged && potential.converged || error("initial data cross approximation failed")
    rhs = if derivative_method == :hybrid
        operators = [
            multidim_derivative_mpo(sites, [nbits, nbits], axis, dx; order=2, boundary=:dirichlet) for axis in 1:2
        ]
        schrodinger_rhs(operators, potential.mps; epsilon=1e-4, cutoff=1e-20, maxdim=32)
    else
        schrodinger_rhs((nbits, nbits), (dx, dx), potential.mps; epsilon=1e-4, boundary=:dirichlet)
    end
    solution = solve_pde(rhs, initial.mps, (0.0, final_time); dt, cross_kwargs=cross)
    # MPS norm is enough for a probability-mass diagnostic; no dense tensor needed.
    mass0 = dx^2*norm(first(solution.states))^2
    mass1 = dx^2*norm(last(solution.states))^2
    println("Schrödinger: t=$(last(solution.times)), relative mass drift=$(abs(mass1/mass0-1))")
    return (; solution, mass0, mass1)
end

if abspath(PROGRAM_FILE) == @__FILE__
    BLAS.set_num_threads(1)
    schrodinger_example()
end
