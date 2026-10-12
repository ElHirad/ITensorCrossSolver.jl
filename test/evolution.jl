function dense_rk4(f, u, t, h)
    k1 = f(u, t)
    k2 = f(u+h*k1/2, t+h/2)
    k3 = f(u+h*k2/2, t+h/2)
    k4 = f(u+h*k3, t+h)
    return u+h*(k1+2k2+2k3+k4)/6
end

@testset "PDE RHS and RK4" begin
    sites = quantics_sites(3)
    n, dx = 8, 2pi/8
    x = collect(0:(n - 1)) .* dx
    u = tt_cross(i -> 0.2+sin(x[grid_index(i)]), fill(2, 3); sites, ranks=4, tolerance=1e-12).mps
    values = dense_vector(u)
    kw = (; ranks=4, maxdim=8, tolerance=1e-11, maxsweeps=6)
    D1 = finite_difference_mpo(sites, dx; order=1)
    D2 = finite_difference_mpo(sites, dx; order=2)
    d1 = reference_derivative(n, dx, 1, true)
    d2 = reference_derivative(n, dx, 2, true)
    rhs = burgers_rhs(D1, D2; viscosity=0.1, cutoff=1e-26)
    dense_rhs(v, t) = -v .* (d1*v)+0.1d2*v
    samples = rhs(u, 0.0)
    @test [samples(Tuple(quantics_index(i, 3))) for i in 1:n] ≈ dense_rhs(values, 0) atol=1e-11
    result = approximate_rhs(rhs, u, 0.0; cross_kwargs=kw)
    @test result.converged
    @test dense_vector(result.mps) ≈ dense_rhs(values, 0) atol=1e-10
    next = rk4_step(rhs, u, 0.2, 0.01; cross_kwargs=kw)
    @test dense_vector(next) ≈ dense_rk4(dense_rhs, values, 0.2, 0.01) atol=1e-10
    # Unequal stage ranks must work at first, middle and last fibers.
    stages = [
        tt_cross(i -> j+sin(j*x[grid_index(i)]), fill(2, 3); sites, ranks=4, tolerance=1e-12).mps
        for j in 1:4
    ]
    expected = values + 0.01/6 * sum(w*dense_vector(s) for (w, s) in zip((1, 2, 2, 1), stages))
    for mode in 1:3
        left = [Tuple(ones(Int, mode-1))]
        right = [Tuple(ones(Int, 3-mode))]
        fiber = rk4_fiber(u, stages, 0.01, mode, left, right)
        @test vec(fiber) ≈ [expected[grid_index((left[1]..., i, right[1]...))] for i in 1:2]
    end
    # Original complex nonlinear Schrödinger equation, including time-dependent V.
    psi =
        tt_cross(i -> (1+0.3im)*values[grid_index(i)], fill(2, 3); sites, ranks=4, tolerance=1e-12).mps
    for V in (0.4, u, (idx, t)->0.2t+cos(x[grid_index(idx)]))
        nls = schrodinger_rhs([D2], V; epsilon=0.3, cutoff=1e-26)
        potential(t) =
            if V isa Number
                fill(V, n)
            elseif V isa MPS
                values
            else
                0.2t .+ cos.(x)
            end
        dense_nls(v, t) = (im/2)*d2*v-im*potential(t) .* v-im*0.3abs2.(v) .* v
        out = rk4_step(nls, psi, 0.1, 0.001; cross_kwargs=kw)
        @test dense_vector(out) ≈ dense_rk4(dense_nls, dense_vector(psi), 0.1, 0.001) atol=1e-10
    end
    # Exact polynomial-in-time forcing detects incorrect nonautonomous stage times.
    forcing(state, t) = idx -> complex(t^3)
    zero = tt_cross(_ -> 0im, fill(2, 3); sites).mps
    sol = solve_pde(forcing, zero, (0.0, 0.105); dt=0.04, saveevery=2, cross_kwargs=kw)
    @test sol.times ≈ [0, 0.08, 0.105]
    @test dense_vector(sol.states[end]) ≈ fill(0.105^4/4, n) atol=1e-12
    @test dense_vector(sol.states[1]) == zeros(n)
    still = solve_pde(forcing, zero, (0.2, 0.2); dt=0.1)
    @test still.times == [0.2]
    @test length(still.states) == 1
    exact_steps = solve_pde(forcing, zero, (0.0, 0.1); dt=0.01, cross_kwargs=kw)
    @test length(exact_steps.states) == 11
    # Fourth-order temporal convergence for u'=u, independent of spatial errors.
    exp_rhs(state, t) =
        let c=mps_to_cores(state)
            idx -> evaluate(c, idx)
        end
    e1 = norm(
        dense_vector(solve_pde(exp_rhs, u, (0.0, 0.4); dt=0.2, cross_kwargs=kw).states[end])-exp(
            0.4
        )*values,
    )
    e2 = norm(
        dense_vector(solve_pde(exp_rhs, u, (0.0, 0.4); dt=0.1, cross_kwargs=kw).states[end])-exp(
            0.4
        )*values,
    )
    @test 12 < e1/e2 < 20
    @test dense_vector(u) ≈ values
    @test_throws ArgumentError burgers_rhs(D1, D2; viscosity=-1)
    @test_throws ArgumentError schrodinger_rhs(MPO[], 0.0)
    @test_throws DimensionMismatch rhs(cores_to_mps(mps_to_cores(u)), 0.0)
    @test_throws ArgumentError rk4_step(rhs, u, 0.0, 0.0)
    @test_throws ArgumentError rk4_fiber(u, stages[1:3], 0.1, 1, [()], [()])
    @test_throws ArgumentError solve_pde(rhs, u, (1.0, 0.0); dt=0.1)
    @test_throws ArgumentError solve_pde(rhs, u, (0.0, 1.0); dt=0.0)
    bad = (; ranks=1, adaptive=false, maxsweeps=1, tolerance=1e-14)
    @test_throws ErrorException approximate_rhs(rhs, u, 0.0; cross_kwargs=bad)
    @test !approximate_rhs(rhs, u, 0.0; cross_kwargs=bad, strict=false).converged
    small_sites = quantics_sites(1)
    small_state = tt_cross(i->Float64(i[1]), [2]; sites=small_sites).mps
    small_rhs = burgers_rhs(
        finite_difference_mpo(small_sites, 1.0; order=1),
        finite_difference_mpo(small_sites, 1.0);
        viscosity=0.1,
    )
    @test dense_vector(approximate_rhs(small_rhs, small_state, 0.0).mps) ≈ [0.2, -0.2]
end
