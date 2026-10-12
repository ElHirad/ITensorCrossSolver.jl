include("../examples/burgers_1d.jl")
include("../examples/schrodinger_2d.jl")

@testset "Runnable PDE examples and spatial convergence" begin
    coarse = burgers_example(; nbits=4, final_time=0.05, dt=0.01, write_output=false)
    fine = burgers_example(; nbits=5, final_time=0.05, dt=0.01, write_output=false)
    @test fine.error < 3e-4
    @test 3 < coarse.error/fine.error < 5
    @test abs(sum(fine.numerical)) < 1e-8
    @test fine.solution.times[end] == 0.05
    @test norm(fine.numerical) <= norm(burgers_exact.(fine.x, 0.0)) + 1e-10
    nls = schrodinger_example(; nbits=2, final_time=0.002, dt=0.001)
    @test abs(nls.mass1/nls.mass0-1) < 1e-7
    @test length(nls.solution.states) == 3
    pure_burgers = burgers_example(;
        nbits=5, final_time=0.05, dt=0.01, write_output=false, derivative_method=:pure
    )
    @test pure_burgers.numerical ≈ fine.numerical atol=1e-9
    pure_nls = schrodinger_example(; nbits=2, final_time=0.002, dt=0.001, derivative_method=:pure)
    @test dense_vector(last(pure_nls.solution.states)) ≈ dense_vector(last(nls.solution.states)) atol=1e-10
    @test_throws ArgumentError burgers_example(; derivative_method=:bad)
    @test_throws ArgumentError schrodinger_example(; derivative_method=:bad)
    zero = burgers_example(; nbits=3, amplitude=0.0, final_time=0.0, write_output=false)
    @test zero.error == 0.0
    @test_throws ArgumentError burgers_example(; amplitude=1.0, write_output=false)
    @test_throws ArgumentError burgers_example(; viscosity=-0.1, write_output=false)
    @test_throws ArgumentError burgers_example(; nbits=4, dt=100.0, write_output=false)
end
