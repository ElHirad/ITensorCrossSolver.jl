# Independently assemble the stencil by direct row/column indexing.
function reference_derivative(n, dx, order, periodic)
    A = zeros(n, n)
    for i in 1:n
        for (delta, coefficient) in (
            order == 1 ? ((-1, -1/(2dx)), (1, 1/(2dx))) : ((-1, 1/dx^2), (0, -2/dx^2), (1, 1/dx^2))
        )
            j = i + delta
            if periodic
                A[i, mod1(j, n)] += coefficient
            elseif 1 <= j <= n
                A[i, j] += coefficient
            end
        end
    end
    return A
end

@testset "Finite difference MPOs" begin
    for nbits in 1:4, order in (1, 2), boundary in (:periodic, :dirichlet, :homogeneous)
        sites = quantics_sites(nbits)
        dx = 0.2
        A = finite_difference_mpo(sites, dx; order, boundary)
        ref = reference_derivative(2^nbits, dx, order, boundary == :periodic)
        @test dense_operator(A) ≈ ref atol=2e-11
        @test [siteind(A, k; plev=0) for k in 1:length(A)] == sites
    end
    sites = quantics_sites(5)
    for axis in 1:2
        A = multidim_derivative_mpo(sites, [2, 3], axis, 0.1; order=2, boundary=:periodic)
        D4 = reference_derivative(4, 0.1, 2, true)
        D8 = reference_derivative(8, 0.1, 2, true)
        ref = axis == 1 ? kron(D4, Matrix(I, 8, 8)) : kron(Matrix(I, 4, 4), D8)
        @test dense_operator(A) ≈ ref atol=1e-10
        tuple_op = multidim_derivative_mpo(sites, (2, 3), axis, 0.1; order=2, boundary=:periodic)
        @test dense_operator(tuple_op) ≈ ref atol=1e-10
        @test [siteind(A, k; plev=0) for k in 1:length(A)] == sites
    end
    # Analytic Fourier eigenvalues independently verify signs and grid ordering.
    n = 64
    sites = quantics_sites(6)
    x = 2pi .* (0:(n - 1)) ./ n
    psi = tt_cross(i -> sin(x[grid_index(i)]), fill(2, 6); sites, tolerance=1e-12).mps
    D1 = finite_difference_mpo(sites, 2pi/n; order=1)
    D2 = finite_difference_mpo(sites, 2pi/n; order=2)
    @test dense_vector(apply(D1, psi; cutoff=1e-25)) ≈ sin(2pi/n)/(2pi/n) .* cos.(x) atol=1e-10
    @test dense_vector(apply(D2, psi; cutoff=1e-25)) ≈ (-4sin(pi/n)^2/(2pi/n)^2) .* sin.(x) atol=1e-10
    # Construction scales with bit count, not 2^(2n).
    large = finite_difference_mpo(quantics_sites(24), 1.0)
    @test length(large) == 24
    @test maximum(dim(linkind(large, k)) for k in 1:23) <= 5
    @test_throws ArgumentError finite_difference_mpo(sites, 0.0)
    @test_throws ArgumentError finite_difference_mpo(sites, 0.1; order=3)
    @test_throws ArgumentError finite_difference_mpo(sites, 0.1; boundary=:bad)
    @test_throws ArgumentError finite_difference_mpo(sites, 0.1; direction=:bad)
    @test_throws ArgumentError finite_difference_mpo(sites, 0.1; direction=:reverse)
    @test_throws ArgumentError multidim_derivative_mpo(
        quantics_sites(5), (2, 3), 1, 0.1; direction=:reverse
    )
    @test_throws ArgumentError finite_difference_mpo([Index(3)], 0.1)
    @test_throws DimensionMismatch multidim_derivative_mpo(sites, [2, 3], 1, 0.1)
    @test_throws ArgumentError multidim_derivative_mpo(sites, [3, 3], 3, 0.1)
    @test_throws DimensionMismatch multidim_derivative_mpo(sites, (3.0, 3.0), 1, 0.1)
    @test_throws DimensionMismatch multidim_derivative_mpo(sites, (), 1, 0.1)
    middle = multidim_derivative_mpo(
        quantics_sites(4), [1, 2, 1], 2, 0.2; order=1, boundary=:dirichlet
    )
    @test dense_operator(middle) ≈
        kron(Matrix(I, 2, 2), reference_derivative(4, 0.2, 1, false), Matrix(I, 2, 2)) atol=1e-11
end

@testset "Hadamard MPS and MPO products" begin
    rng = MersenneTwister(14)
    sites = quantics_sites(3)
    a = randn(rng, ComplexF64, 2, 2, 2)
    b = randn(rng, ComplexF64, 2, 2, 2)
    u = tt_cross_tensor(a; sites, ranks=4, tolerance=1e-12).mps
    v = tt_cross_tensor(b; ranks=4, tolerance=1e-12).mps
    du, dv = dense_vector(u), dense_vector(v)
    @test dense_operator(diagonal_mpo(u)) ≈ Diagonal(du)
    A = finite_difference_mpo(sites, 0.3; order=1)
    mat = dense_operator(A)
    for method in (:direct, :zipup, :fit)
        kw = method == :fit ? (; nsweeps=3, outputlevel=0) : (;)
        result = hadamard(u, v; method, cutoff=1e-24, maxdim=16, kw...)
        @test dense_vector(result) ≈ du .* dv atol=1e-10
        @test siteinds(result) == siteinds(u)
        for leg in (:lower, :upper)
            product = hadamard(u, A; leg, method, cutoff=1e-24, maxdim=32, kw...)
            expected = leg == :lower ? mat*Diagonal(du) : Diagonal(du)*mat
            @test dense_operator(product) ≈ expected atol=1e-9
        end
    end
    @test dense_tensor(u) ≈ a  # helpers do not mutate inputs
    @test dense_tensor(v) ≈ b
    @test dense_vector(hadamard(2, u)) ≈ 2du
    @test dense_vector(hadamard(u, 2im)) ≈ 2im*du
    @test hadamard(2, 3) == 6
    @test isnothing(hadamard(nothing, u))
    @test isnothing(hadamard(u, nothing))
    @test isnothing(hadamard(nothing, nothing))
    @test isnothing(hadamard(nothing, 2))
    @test isnothing(hadamard(2, nothing))
    one = cores_to_mps([reshape([1.0, 2, 3], 1, 3, 1)])
    @test dense_vector(hadamard(one, one)) == [1, 4, 9]
    @test_throws ArgumentError hadamard(u, v; method=:bad)
    @test_throws ArgumentError hadamard(u, A; leg=:bad)
    @test_throws ArgumentError hadamard(u, v; cutoff=-1)
    @test_throws DimensionMismatch hadamard(u, one)
    @test dense_vector(hadamard(u, v; method=:fit, cutoff=1e-24)) ≈ du .* dv atol=1e-10
    @test dense_operator(hadamard(u, A; method=:fit, cutoff=1e-24)) ≈ mat*Diagonal(du) atol=1e-9
    @test_throws ArgumentError hadamard(u, v; method=:fit, nsweeps=0)
    @test_throws ArgumentError hadamard(u, v; method=:fit, nsweeps=-1)
    @test_throws ArgumentError hadamard(u, v; method=:fit, nsweeps=1.5)
    zero_state = tt_cross(_->0.0, fill(2, 3); sites).mps
    for method in (:direct, :zipup, :fit)
        @test norm(dense_vector(hadamard(zero_state, u; method))) < 1e-14
        @test maximum(bond_dimensions(hadamard(u, v; method, maxdim=1))) == 1
    end
    @test dense_operator(A) ≈ mat
end
