@testset "DEIM and Gappy POD" begin
    U = [1.0 0; 0 1; 0.5 0.25; 0.1 -0.2]
    @test deim(U) == [1, 2]
    @test deim(U, 1) == [1]
    rng = MersenneTwister(8)
    for T in (Float64, ComplexF64)
        Q = Matrix(qr(randn(rng, T, 20, 4)).Q)[:, 1:4]
        p = deim(Q)
        @test length(unique(p)) == 4
        x = randn(rng, T, 4)
        @test Q * (Q[p, :] \ (Q * x)[p]) ≈ Q*x
        g = gappy_pod(Q, 5, p)
        @test g[1:4] == p
        @test length(unique(g)) == 9
        @test minimum(svdvals(Q[g, :])) >= minimum(svdvals(Q[p, :])) - 1e-14
        @test gappy_pod(Q, 0, p) == p
        @test length(gappy_pod(Q, 16, p)) == 20
    end
    @test gappy_pod(reshape([3.0, 2, 1], 3, 1), 1) == [1, 2]
    @test sort(gappy_pod([1.0 0; 0 1; 1 1], 1, [1, 2])) == [1, 2, 3]
    @test_throws ArgumentError deim(zeros(3, 2))
    @test_throws ArgumentError deim(ones(3, 2))
    @test_throws ArgumentError deim(U, 3)
    @test_throws ArgumentError deim(fill(NaN, 3, 1))
    @test_throws ArgumentError gappy_pod(U, 3)
    @test_throws ArgumentError gappy_pod(U, -1)
    @test_throws ArgumentError gappy_pod(U, 0, [1, 1])
    @test_throws ArgumentError gappy_pod(U, 0, [1, 5])
    @test_throws ArgumentError gappy_pod(U, 0, [1])
    # Numerically dependent single-precision columns must not pass a Float64 test.
    A32 = randn(MersenneTwister(31), Float32, 8, 2)
    @test_throws ArgumentError deim(hcat(A32, A32[:, 1]+2.0f0*A32[:, 2]))
    for T in (Float32, ComplexF32)
        Q = Matrix(qr(randn(rng, T, 12, 3)).Q)[:, 1:3]
        p = deim(Q)
        @test Q * (Q[p, :] \ (Q * ones(T, 3))[p]) ≈ Q*ones(T, 3) rtol=1e-5
    end
    for scale in (1e-200, 1.0, 1e200)
        @test gappy_pod(scale .* reshape([1.0, 3, 2], 3, 1), 1, [3]) == [3, 2]
        basis = [1.0 0; 0 2; 1 1; 0.5 -0.7]
        @test gappy_pod(scale .* basis, 2, [1, 2]) == gappy_pod(basis, 2, [1, 2])
    end
end
