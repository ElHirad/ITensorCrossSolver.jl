@testset "Pure TT sampled finite differences" begin
    rng = MersenneTwister(81)
    @testset "Every entry, carry/borrow, and axis-local boundaries" begin
        # Random data distinguishes every neighbor, including carry chains and
        # physical-axis boundaries. Include complex states and two-point grids.
        for bits in ((1,), (3,), (2, 3), (1, 2, 1)), T in (Float64, ComplexF64)
            sites = quantics_sites(sum(bits))
            tensor = randn(rng, T, ntuple(_->2, sum(bits)))
            u = tt_cross_tensor(tensor; sites, ranks=8, maxdim=16, tolerance=1e-12).mps
            vector = dense_vector(u)
            spacings = ntuple(k -> 0.2+0.1k, length(bits))
            for boundary in (:periodic, :dirichlet, :homogeneous), cache in (true, false)
                samples = finite_difference_samples(u, bits, spacings; boundary, cache)
                @test [samples.value(quantics_index(i, sum(bits))) for i in eachindex(vector)] ≈ vector atol=2e-12
                for axis in eachindex(bits), order in (1, 2)
                    # Independent dense tensor-product stencil (no MPO oracle).
                    factors = [
                        if k == axis
                            reference_derivative(
                                2^bits[k], spacings[k], order, boundary==:periodic
                            )
                        else
                            Matrix{Float64}(I, 2^bits[k], 2^bits[k])
                        end for k in eachindex(bits)
                    ]
                    expected = reduce(kron, factors)*vector
                    callback = order == 1 ? samples.first : samples.second
                    actual = [
                        callback(quantics_index(i, sum(bits)), axis) for i in eachindex(vector)
                    ]
                    @test actual ≈ expected atol=2e-10
                end
            end
            @test dense_tensor(u) ≈ tensor atol=2e-12
        end
    end

    @testset "Mixed boundaries, snapshot, and many total bits" begin
        bits = (2, 3)
        u = tt_cross(i -> Float64(grid_index(i)), fill(2, 5); ranks=4, tolerance=1e-12).mps
        samples = finite_difference_samples(u, bits, (0.2, 0.3); boundary=(:periodic, :dirichlet))
        old = dense_vector(u)
        u[1] *= 2
        @test samples.value((1, 1, 1, 1, 1)) ≈ old[1]
        @test samples.first((1, 1, 1, 1, 1), 1) ≈ (old[9]-old[25])/0.4
        @test samples.second((1, 1, 1, 1, 1), 2) ≈ (old[2]-2old[1])/0.3^2
        # Total bits may exceed machine-word size: coordinates are decoded per
        # physical axis, never by flattening the entire tensor's index to Int.
        big = cores_to_mps([reshape([1.0, 1.0], 1, 2, 1) for _ in 1:66])
        large = finite_difference_samples(big, (22, 22, 22), (0.1, 0.2, 0.3))
        @test large.value(ntuple(_->2, 66)) == 1.0
        @test large.first(ntuple(_->2, 66), 2) == 0.0
        @test large.second(ntuple(_->1, 66), 3) == 0.0
        mixed = cores_to_mps([reshape([1.0, 2.0], 1, 2, 1), reshape([1+im, 2-im], 1, 2, 1)])
        complex_samples = finite_difference_samples(mixed, (2,), (0.5,); boundary=:dirichlet)
        @test complex_samples.value((2, 1)) == 2+2im
        @test complex_samples.second((1, 1), 1) == -12im
    end

    @testset "Invalid stencil inputs" begin
        u = tt_cross(_->1.0, fill(2, 3)).mps
        for (bits, dx, boundary, exception) in (
            ((), (), :periodic, ArgumentError),
            ((0, 3), (1.0, 1.0), :periodic, ArgumentError),
            ((3.0,), (1.0,), :periodic, ArgumentError),
            ((63,), (1.0,), :periodic, ArgumentError),
            ((3,), (0.0,), :periodic, ArgumentError),
            ((3,), (Inf,), :periodic, ArgumentError),
            ((3,), (1.0, 2.0), :periodic, DimensionMismatch),
            ((3,), (1.0,), :bad, ArgumentError),
            ((3,), (1.0,), (:periodic, :dirichlet), DimensionMismatch),
            ((2,), (1.0,), :periodic, DimensionMismatch),
        )
            @test_throws exception finite_difference_samples(u, bits, dx; boundary)
        end
        nonbinary = cores_to_mps([reshape(ones(3), 1, 3, 1)])
        @test_throws DimensionMismatch finite_difference_samples(nonbinary, (1,), (1.0,))
        s = finite_difference_samples(u, (3,), (1.0,))
        @test_throws DimensionMismatch s.value((1, 1))
        @test_throws ArgumentError s.value((1, 1, 3))
        @test_throws ArgumentError s.value((1, 1, 1.0))
        @test_throws ArgumentError s.first((1, 1, 1), 0)
        @test_throws ArgumentError s.second((1, 1, 1), 2)
    end
end

@testset "Pure PDE right-hand sides" begin
    sites = quantics_sites(3)
    dx = 0.3
    u = tt_cross(i -> 0.2+0.1sin(grid_index(i)), fill(2, 3); sites, ranks=4, tolerance=1e-12).mps
    for boundary in (:periodic, :dirichlet), cache in (true, false)
        pure = burgers_rhs(3, dx; viscosity=0.2, boundary, cache)
        hybrid = burgers_rhs(
            finite_difference_mpo(sites, dx; order=1, boundary),
            finite_difference_mpo(sites, dx; order=2, boundary);
            viscosity=0.2,
            cutoff=1e-26,
        )
        p, h = pure(u, 0.0), hybrid(u, 0.0)
        @test [p(quantics_index(i, 3)) for i in 1:8] ≈ [h(quantics_index(i, 3)) for i in 1:8] atol=1e-12
    end
    bits = (2, 1)
    psi = (1+0.5im)*u
    for boundary in (:periodic, :dirichlet), V in (0.4, u, (idx, t)->0.2t+grid_index(idx)/8)
        pure = schrodinger_rhs(bits, (dx, 2dx), V; epsilon=0.3, boundary)
        ops = [
            multidim_derivative_mpo(sites, bits, axis, axis*dx; order=2, boundary) for axis in 1:2
        ]
        hybrid = schrodinger_rhs(ops, V; epsilon=0.3, cutoff=1e-26)
        p, h = pure(psi, 0.1), hybrid(psi, 0.1)
        @test [p(quantics_index(i, 3)) for i in 1:8] ≈ [h(quantics_index(i, 3)) for i in 1:8] atol=2e-12
    end
    @test_throws ArgumentError burgers_rhs(3, dx; viscosity=-1.0)
    @test_throws ArgumentError burgers_rhs(0, dx; viscosity=0.1)
    @test_throws ArgumentError schrodinger_rhs(bits, (dx, dx), 0.0; epsilon=NaN)
    wrong = tt_cross(_->1.0, fill(2, 2)).mps
    @test_throws DimensionMismatch schrodinger_rhs(bits, (dx, dx), wrong)(u, 0.0)
end
