@testset "Tensor conversions, indices, and fibers" begin
    rng = MersenneTwister(10)
    cores = [
        randn(rng, ComplexF64, 1, 2, 2),
        randn(rng, ComplexF64, 2, 3, 2),
        randn(rng, ComplexF64, 2, 2, 1),
    ]
    psi = cores_to_mps(cores)
    @test mps_to_cores(psi) ≈ cores
    @test bond_dimensions(psi) == [2, 2]
    A = [
        only(cores[1][:, i, :]*cores[2][:, j, :]*cores[3][:, k, :]) for i in 1:2, j in 1:3, k in 1:2
    ]
    @test dense_tensor(psi) ≈ A
    @test dense_vector(psi) ≈ [A[i, j, k] for i in 1:2 for j in 1:3 for k in 1:2]
    @test evaluate(psi, (2, 3, 1)) ≈ A[2, 3, 1]
    @test evaluate(cores, (1, 2, 2)) ≈ A[1, 2, 2]
    reversed = cores_to_mps(reverse_cores(cores); sites=reverse(siteinds(psi)))
    @test dense_tensor(reversed) ≈ permutedims(A, (3, 2, 1))
    @test reverse_cores(reverse_cores(cores)) ≈ cores
    for k in 1:3
        left = k == 1 ? [()] : [Tuple(ones(Int, k-1)), Tuple(size(A, j) for j in 1:(k - 1))]
        right = k == 3 ? [()] : [Tuple(ones(Int, 3-k)), Tuple(size(A, j) for j in (k + 1):3)]
        ref = [A[l..., i, r...] for l in left, i in 1:size(A, k), r in right]
        @test evaluate_fiber(psi, k, left, right) ≈ ref
        @test evaluate_fiber(idx -> A[idx...], size(A), k, left, right) ≈ ref
    end
    single = cores_to_mps([reshape([1.0, 2, 3], 1, 3, 1)])
    @test dense_vector(single) == [1, 2, 3]
    @test mps_to_cores(single)[1] == reshape([1, 2, 3], 1, 3, 1)
    @test isempty(bond_dimensions(single))
    for n in 1:8, i in 1:(2 ^ n)
        @test grid_index(quantics_index(i, n)) == i
    end
    @test quantics_index(6, 3) == [2, 1, 2]
    @test dim.(quantics_sites(4)) == fill(2, 4)
    f = quantics_function((x, y) -> x+2y, ([0.0, 1, 2, 3], [10.0, 20]))
    @test f((2, 1, 2)) == 42
    @test_throws ArgumentError quantics_sites(0)
    @test_throws ArgumentError grid_index([0, 1])
    @test_throws BoundsError quantics_index(9, 3)
    @test_throws ArgumentError quantics_function(identity, ([1, 2, 3],))
    @test_throws DimensionMismatch f((1, 2))
    @test_throws ArgumentError cores_to_mps([])
    @test_throws DimensionMismatch cores_to_mps([ones(2, 2, 1)])
    @test_throws DimensionMismatch cores_to_mps([ones(1, 2, 2), ones(3, 2, 1)])
    @test_throws DimensionMismatch cores_to_mps(cores; sites=quantics_sites(3))
    @test_throws DimensionMismatch evaluate(psi, (1, 2))
    @test_throws BoundsError evaluate(psi, (1, 4, 1))
    @test_throws ArgumentError evaluate_fiber(psi, 0, [()], [()])
    @test_throws ArgumentError evaluate_fiber(psi, 2, [(3,)], [(1,)])
    copied = mps_to_cores(psi)
    copied[1] .*= 2
    @test dense_tensor(psi) ≈ A
end
