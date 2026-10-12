include("../examples/advection_diffusion_reaction_2d.jl")

@testset "2D advection–diffusion–reaction versus DNS" begin
    @testset "Initial data and independent Fourier stencil check" begin
        nx, ny = 8, 16
        dx, dy = 2pi/nx, 2pi/ny
        x, y = (0:(nx - 1)) .* dx, (0:(ny - 1)) .* dy
        @test adr2d_initial(0.0, 0.0) ≈ 0.38
        @test adr2d_initial(0.37, -0.28) ≈ adr2d_initial(0.37+2pi, -0.28+2pi)
        initial = [adr2d_initial(xi, yj) for xi in x, yj in y]
        @test all(u -> 0.1-eps() <= u <= 0.5+eps(), initial)
        # The centered finite-difference symbols differ from the continuum
        # wavenumbers. This checks the DNS independently of both MPOs and loops.
        kx, ky = 1, 2
        phase = [kx*xi+ky*yj for xi in x, yj in y]
        U = 0.3 .+ 0.1cos.(phase)
        ux = -0.1sin(kx*dx)/dx .* sin.(phase)
        uy = -0.1sin(ky*dy)/dy .* sin.(phase)
        uxx = -0.4sin(kx*dx/2)^2/dx^2 .* cos.(phase)
        uyy = -0.4sin(ky*dy/2)^2/dy^2 .* cos.(phase)
        v, D, r = (0.7, -0.3), (0.06, 0.02), 1.2
        expected = -v[1]*ux-v[2]*uy+D[1]*uxx+D[2]*uyy+r*U .* (1 .- U)
        before = copy(U)
        @test adr2d_dns_rhs(U, dx, dy; velocity=v, diffusivity=D, reaction_rate=r) ≈ expected atol=2e-15
        @test U == before
    end

    @testset "Sampled and compressed RHS on an unequal-axis grid" begin
        bits = (3, 2)
        nx, ny = 2 .^ bits
        dx, dy = 2pi/nx, 2pi/ny
        x, y = (0:(nx - 1)) .* dx, (0:(ny - 1)) .* dy
        sites = quantics_sites(sum(bits))
        kw = (; ranks=8, maxdim=16, tolerance=1e-12, maxsweeps=12, validation_samples=64)
        initial = tt_cross(
            quantics_function(adr2d_initial, (x, y)), fill(2, sum(bits)); sites, kw...
        )
        @test initial.converged
        U = [adr2d_initial(xi, yj) for xi in x, yj in y]
        to_field(psi) = permutedims(reshape(dense_vector(psi), ny, nx), (2, 1))
        @test to_field(initial.mps) ≈ U atol=2e-13
        operators = (
            x=multidim_derivative_mpo(sites, bits, 1, dx; order=1),
            y=multidim_derivative_mpo(sites, bits, 2, dy; order=1),
            xx=multidim_derivative_mpo(sites, bits, 1, dx; order=2),
            yy=multidim_derivative_mpo(sites, bits, 2, dy; order=2),
        )
        rhs = adr2d_cross_rhs(operators; cutoff=1e-26)
        callback = rhs(initial.mps, 0.0)
        sampled = [
            callback((quantics_index(i, bits[1])..., quantics_index(j, bits[2])...)) for
            i in 1:nx, j in 1:ny
        ]
        expected = adr2d_dns_rhs(U, dx, dy)
        @test sampled ≈ expected atol=2e-12
        compressed = approximate_rhs(rhs, initial.mps, 0.0; cross_kwargs=kw)
        @test compressed.converged
        @test to_field(compressed.mps) ≈ expected atol=2e-12
        stepped = rk4_step(rhs, initial.mps, 0.0, 0.01; cross_kwargs=kw)
        @test to_field(stepped) ≈ adr2d_dns_step(U, 0.01, dx, dy) atol=2e-12
        @test to_field(initial.mps) ≈ U atol=2e-13
    end

    @testset "DNS RK4 nonlinear temporal convergence" begin
        # Constant fields reduce exactly to logistic growth, regardless of the
        # nonzero advection and diffusion coefficients. The analytic solution
        # verifies the dense time stepper without reusing an RK4 implementation.
        start = fill(0.2, 4, 8)
        exact = 1/(1+4exp(-0.8))
        errors = map((0.2, 0.1)) do h
            U = copy(start)
            for _ in 1:round(Int, 0.8 / h)
                U = adr2d_dns_step(U, h, 2pi/4, 2pi/8)
            end
            @test maximum(U)-minimum(U) == 0
            abs(U[1, 1]-exact)
        end
        @test 12 < errors[1]/errors[2] < 20
        @test start == fill(0.2, 4, 8)
        @test_throws ArgumentError adr2d_dns_step(start, 0.0, 1.0, 1.0)
        @test_throws ArgumentError adr2d_dns_step(start, NaN, 1.0, 1.0)
    end

    @testset "L2 normalization" begin
        metrics = adr2d_l2_metrics(fill(3.0, 2, 4), fill(2.0, 2, 4), 0.5, 0.25)
        @test metrics.tt_l2 ≈ 3.0
        @test metrics.dns_l2 ≈ 2.0
        @test metrics.l2_error ≈ 1.0
        @test metrics.relative_l2_error ≈ 0.5
        @test metrics.euclidean_l2_error ≈ sqrt(8.0)
        @test adr2d_l2_metrics(zeros(2, 4), zeros(2, 4), 1.0, 1.0).relative_l2_error == 0.0
        @test adr2d_l2_metrics(ones(2, 4), zeros(2, 4), 1.0, 1.0).relative_l2_error == Inf
        @test_throws DimensionMismatch adr2d_l2_metrics(zeros(2, 4), zeros(4, 2), 1.0, 1.0)
        @test_throws ArgumentError adr2d_l2_metrics(zeros(2, 4), zeros(2, 4), 0.0, 1.0)
    end

    @testset "Full solve, shortened final step, and output" begin
        # The default grid has more entries than the cross validation sample
        # count. Validate the actual documented run over its entire dense grid.
        default = adr2d_example(; write_output=false)
        @test size(default.numerical) == (32, 32)
        @test length(default.history) == 21
        @test default.solution.times[end] == 0.2
        @test default.history[1].relative_l2_error < 1e-12
        @test maximum(row.relative_l2_error for row in default.history) < 1e-8
        @test default.metrics.l2_error < 1e-8
        pure = adr2d_example(; derivative_method=:pure, write_output=false)
        @test maximum(row.relative_l2_error for row in pure.history) < 1e-8
        @test norm(pure.numerical-default.numerical)/norm(default.numerical) < 1e-8
        @test pure.solution.times == default.solution.times
        square = adr2d_example(; nbits=3, final_time=0.035, dt=0.01, write_output=false)
        @test square.solution.times ≈ [0.0, 0.01, 0.02, 0.03, 0.035]
        @test getproperty.(square.history, :time) == square.solution.times
        @test maximum(row.relative_l2_error for row in square.history) < 5e-8
        @test square.metrics.relative_l2_error < 5e-8
        @test norm(square.numerical-square.initial_field)/norm(square.initial_field) > 0.01
        @test all(isfinite, square.numerical)
        @test all(u -> 0 < u < 1, square.numerical)
        @test square.metrics.l2_error ≈
            sqrt(square.parameters.dx*square.parameters.dy) * norm(square.numerical-square.dns)
        @test square.metrics.relative_l2_error ≈ norm(square.numerical-square.dns)/norm(square.dns)
        # Repeat the reference integration using exactly the recorded TT times.
        reference = copy(square.initial_field)
        for h in diff(square.solution.times)
            reference = adr2d_dns_step(reference, h, square.parameters.dx, square.parameters.dy)
        end
        @test square.dns == reference
        mktempdir() do output_dir
            rectangular = adr2d_example(; nbits=(2, 3), final_time=0.02, dt=0.01, output_dir)
            @test size(rectangular.numerical) == (4, 8)
            @test rectangular.metrics.relative_l2_error < 5e-8
            rows = readlines(joinpath(output_dir, "adr_2d_fields.csv"))
            @test rows[1] == "x,y,initial,tt,dns,error"
            @test length(rows) == 1+4*8
            # Check an off-diagonal grid point to catch transposed output axes.
            i, j = 2, 3
            row = parse.(Float64, split(rows[1 + (i - 1) * 8 + j], ','))
            @test row ≈ [
                rectangular.x[i],
                rectangular.y[j],
                rectangular.initial_field[i, j],
                rectangular.numerical[i, j],
                rectangular.dns[i, j],
                rectangular.numerical[i, j]-rectangular.dns[i, j],
            ]
            error_rows = readlines(joinpath(output_dir, "adr_2d_errors.csv"))
            @test length(error_rows) == 1+length(rectangular.history)
            @test split(error_rows[1], ',') == string.(collect(keys(first(rectangular.history))))
            @test parse.(Float64, split(error_rows[end], ',')) ≈
                collect(values(last(rectangular.history)))
        end
        still = adr2d_example(; nbits=(2, 2), final_time=0.0, write_output=false)
        @test length(still.history) == 1
        @test still.dns == still.initial_field
        @test still.metrics.relative_l2_error < 1e-12
    end

    @testset "Invalid setup and failed initial cross" begin
        for kw in (
            (; nbits=(2, 0)),
            (; nbits=(2,)),
            (; nbits=(2.5, 3)),
            (; velocity=(1.0,)),
            (; velocity=(NaN, 1.0)),
            (; diffusivity=(-0.1, 0.1)),
            (; reaction_rate=-1.0),
            (; dt=0.0),
            (; final_time=-1.0),
            (; operator_cutoff=-1.0),
            (; operator_maxdim=0),
            (; operator_maxdim=1.5),
            (; nbits=5, dt=10.0, final_time=10.0),
        )
            @test_throws ArgumentError adr2d_example(; write_output=false, kw...)
        end
        @test_throws ErrorException adr2d_example(;
            nbits=3,
            write_output=false,
            cross_kwargs=(; ranks=1, maxdim=1, adaptive=false, maxsweeps=1, tolerance=1e-13),
        )
        @test_throws ArgumentError adr2d_example(; derivative_method=:bad)
    end
end
