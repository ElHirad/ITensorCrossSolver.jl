@testset "Alternating TT cross" begin
    rng = MersenneTwister(3)
    for T in (Float64, ComplexF64)
        # Nonuniform mode sizes make sweep orientation/indexing errors visible.
        A = randn(rng, T, 2, 3, 4, 2)
        for sweeps in (1, 2, 3, 4)
            result = tt_cross_tensor(
                A; ranks=12, maxdim=12, adaptive=false, maxsweeps=sweeps, tolerance=1e-12
            )
            @test result.converged
            @test dense_tensor(result.mps) ≈ A atol=1e-11
            @test result.evaluations <= length(A)
            @test result.sweeps <= sweeps
        end
    end
    A = randn(rng, 2, 3, 3, 2)
    r = tt_cross_tensor(A; ranks=1, maxdim=8, maxsweeps=14, tolerance=1e-11)
    @test r.converged
    @test maximum(bond_dimensions(r.mps)) > 1
    @test dense_tensor(r.mps) ≈ A atol=1e-10
    low = tt_cross_tensor(A; ranks=1, adaptive=false, maxsweeps=3, tolerance=1e-12)
    @test !low.converged
    @test low.errors[end] > 1e-3
    @test bond_dimensions(low.mps) == [1, 1, 1]
    for value in (0.0, 2.0, 0im, 1+2im)
        result = tt_cross(_ -> value, fill(2, 6); maxsweeps=5)
        @test result.converged
        @test dense_vector(result.mps) ≈ fill(value, 64) atol=5e-14
        @test bond_dimensions(result.mps) == ones(Int, 5)
    end
    @test dense_vector(tt_cross(i -> i[1]^2, [5]).mps) ≈ [1, 4, 9, 16, 25]
    # A large tensor with an analytic rank-2 formula. Count all callback calls.
    count = Ref(0)
    f(i) = (count[] += 1; prod(1+0.01k*x for (k, x) in enumerate(i)) + 0.3)
    result = tt_cross(f, fill(2, 18); ranks=2, tolerance=1e-10, validation_samples=80)
    @test result.converged
    @test result.evaluations == count[]
    @test count[] < 2^18 ÷ 20
    for _ in 1:30
        idx = Tuple(rand(rng, 1:2, 18))
        exact = prod(1+0.01k*x for (k, x) in enumerate(idx)) + 0.3
        @test evaluate(result.mps, idx) ≈ exact rtol=1e-10
    end
    @test_throws ArgumentError tt_cross(identity, [])
    @test_throws ArgumentError tt_cross(identity, [0, 2])
    @test_throws ArgumentError tt_cross(identity, [2, 2]; maxdim=0)
    @test_throws ArgumentError tt_cross(identity, [2, 2]; maxsweeps=0)
    @test_throws ArgumentError tt_cross(identity, [2, 2]; tolerance=0)
    @test_throws ArgumentError tt_cross(identity, [2, 2]; ranks=[1, 2])
    @test_throws ArgumentError tt_cross(identity, [2, 2]; oversampling=-1)
    @test_throws ArgumentError tt_cross(_ -> NaN, [2, 2])
    @test_throws DimensionMismatch tt_cross(identity, [2, 3]; sites=quantics_sites(2))
    # Rank selection must be invariant to an overall change of amplitude.
    for scale in (1e-200, 1.0, 1e200), adaptive in (false, true)
        A = scale .* [1.0 0.2; 0.3 2.0]
        r = tt_cross_tensor(A; ranks=2, adaptive, tolerance=1e-12)
        @test r.converged
        @test bond_dimensions(r.mps) == [2]
        @test dense_tensor(r.mps) ./ scale ≈ A ./ scale rtol=1e-12
    end
    for T in (Float32, ComplexF32)
        A = randn(rng, T, 2, 3, 2)
        r = tt_cross_tensor(A; ranks=4, tolerance=1e-5)
        @test r.converged
        @test dense_tensor(r.mps) ≈ A rtol=1e-5
    end
    singleton = reshape(collect(1.0:6.0), 1, 2, 1, 3)
    @test dense_tensor(tt_cross_tensor(singleton; ranks=4).mps) ≈ singleton
end
