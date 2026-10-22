@testset "MSB-first ordering across the public API" begin
    @test quantics_index(3, 4) == [1, 1, 2, 1]
    @test quantics_index(9, 4) == [2, 1, 1, 1]
    @test grid_index((1, 1, 2, 1)) == 3
    @test_throws ArgumentError grid_index((1.0, 2.0))

    @testset "Dense conversion and unfoldings" begin
        for dims in ((5,), (2, 2, 2, 2), (2, 3, 4), (1, 2, 1, 3))
            v = [complex(i, -i^2) for i in 1:prod(dims)]
            A = msb_tensor(v, dims)
            @test size(A) == dims
            @test msb_vector(A) == v
            for I in CartesianIndices(A)
                flat = 1 + sum((I[k]-1)*prod(dims[(k + 1):end]) for k in eachindex(dims))
                @test A[I] == v[flat]
            end
            for bond in 1:(length(dims) - 1)
                B = msb_unfolding(A, bond)
                @test size(B) == (prod(dims[1:bond]), prod(dims[(bond + 1):end]))
                @test [B[i, j] for i in axes(B, 1) for j in axes(B, 2)] == v
            end
            flat = msb_vector(A)
            flat[1] = -100im
            @test A[begin] == v[1]
            A[begin] = 200im
            @test v[1] == 1-im
        end
        @test msb_tensor(1:6, (2, 3)) == [1 2 3; 4 5 6]
        @test msb_unfolding(msb_tensor(1:16, (2, 2, 2, 2)), 2) ==
            [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]
        @test_throws ArgumentError msb_tensor([1], ())
        @test_throws ArgumentError msb_tensor([1], (0,))
        @test_throws ArgumentError msb_tensor([1], (1.0,))
        @test_throws DimensionMismatch msb_tensor(1:6, (2, 2))
        @test_throws DimensionMismatch msb_tensor(1:6, (typemax(Int), 2))
        @test_throws ArgumentError msb_vector(zeros(0, 2))
        @test_throws ArgumentError msb_vector(fill(1.0))
        @test_throws ArgumentError msb_unfolding(ones(2, 2), 0)
        @test_throws ArgumentError msb_unfolding(ones(2, 2), 2)
    end

    @testset "MPS input, output, fibers, and sweep directions" begin
        rng = MersenneTwister(19)
        for dims in ((5,), (2, 3, 2), (1, 2, 1, 3)), maxsweeps in (1, 2)
            v = randn(rng, ComplexF64, prod(dims))
            result = tt_cross_tensor(v, dims; ranks=8, maxdim=8, tolerance=1e-12, maxsweeps)
            @test result.converged
            @test result.sweeps == maxsweeps || length(dims) == 1
            @test dense_vector(result.mps) ≈ v atol=2e-12
            @test msb_vector(dense_tensor(result.mps)) ≈ v atol=2e-12
            @test dense_tensor(result.mps, dims) ≈ msb_tensor(v, dims) atol=2e-12
        end
        x = collect(0:15) * (1/15)
        y = @. exp(x)-x^4
        result = tt_cross_tensor(y, fill(2, 4))
        @test result.converged
        @test bond_dimensions(result.mps) == [2, 4, 2]
        @test norm(dense_vector(result.mps)-y)/norm(y) < 1e-12
        @test_throws DimensionMismatch dense_tensor(result.mps, (2, 4))
        f = quantics_function(z -> exp(z)-z^4, (x,))
        for i in eachindex(y)
            idx = quantics_index(i, 4)
            @test evaluate(result.mps, idx) ≈ y[i] atol=2e-12
            @test f(idx) == y[i]
        end
        # Nontrivial prefixes/suffixes exercise all three fiber axes.
        left, right = [(1,), (2,)], [(1, 1), (2, 1)]
        fiber = evaluate_fiber(result.mps, 2, left, right)
        @test fiber ≈ [y[grid_index((l..., i, r...))] for l in left, i in 1:2, r in right]
        stages = [j*result.mps for j in 1:4]
        @test rk4_fiber(result.mps, stages, 0.01, 2, left, right) ≈ 1.025fiber
    end

    @testset "Rectangular complex grid, derivatives, and products" begin
        bits = (2, 3)
        # Asymmetric markers distinguish axes, bit reversal, and conjugation.
        U = [complex(0.03i+0.004j^2, 0.01i*j) for i in 1:4, j in 1:8]
        v = [U[i, j] for i in 1:4 for j in 1:8]
        sites = quantics_sites(5)
        psi = tt_cross_tensor(v, fill(2, 5); sites, ranks=8, tolerance=1e-12).mps
        @test dense_tensor(psi, (4, 8)) ≈ U atol=2e-12
        @test dense_operator(diagonal_mpo(psi)) ≈ Diagonal(v) atol=2e-12
        @test dense_vector(hadamard(psi, psi; cutoff=1e-25)) ≈ v .^ 2 atol=2e-12
        for boundary in (:periodic, :dirichlet)
            samples = finite_difference_samples(psi, bits, (0.2, 0.3); boundary)
            for axis in 1:2, order in 1:2
                h = (0.2, 0.3)[axis]
                D = reference_derivative(2^bits[axis], h, order, boundary==:periodic)
                expected = axis == 1 ? D*U : U*transpose(D)
                op = multidim_derivative_mpo(sites, bits, axis, h; order, boundary)
                @test dense_tensor(apply(op, psi; cutoff=1e-25), (4, 8)) ≈ expected atol=2e-10
                callback = order == 1 ? samples.first : samples.second
                actual = [
                    callback((quantics_index(i, 2)..., quantics_index(j, 3)...), axis) for
                    i in 1:4, j in 1:8
                ]
                @test actual ≈ expected atol=2e-10
            end
        end
        @test dense_vector(psi) ≈ v atol=2e-12
    end

    @testset "Vector input through pure and hybrid RK4" begin
        x = collect(0:15) * (2pi/16)
        v = @. 0.2 + 0.1sin(x) + 0.03cos(2x)
        sites = quantics_sites(4)
        psi = tt_cross_tensor(v, fill(2, 4); sites, tolerance=1e-12).mps
        h, nu, dt = 2pi/16, 0.1, 0.001
        D1 = reference_derivative(16, h, 1, true)
        D2 = reference_derivative(16, h, 2, true)
        rhs_dense(u, t) = -u .* (D1*u)+nu*(D2*u)
        expected = dense_rk4(rhs_dense, v, 0.0, dt)
        pure = burgers_rhs(4, h; viscosity=nu)
        hybrid = burgers_rhs(
            finite_difference_mpo(sites, h; order=1),
            finite_difference_mpo(sites, h; order=2);
            viscosity=nu,
            cutoff=1e-26,
        )
        for rhs in (pure, hybrid)
            sol = solve_pde(
                rhs, psi, (0.0, dt); dt, cross_kwargs=(; ranks=8, maxdim=16, tolerance=1e-12)
            )
            @test dense_vector(sol.states[end]) ≈ expected atol=2e-11
            @test dense_vector(sol.states[1]) ≈ v atol=2e-12
            @test siteinds(sol.states[end]) == sites
        end
    end
end
