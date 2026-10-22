using Test
using LinearAlgebra
using Random
using ITensors
using ITensorMPS
using ITensorCrossSolver

BLAS.set_num_threads(1)

@testset "ITensorCrossSolver" begin
    @testset "Method definitions" begin
        @test isempty(Test.detect_ambiguities(ITensorCrossSolver; recursive=true))
        @test isempty(Test.detect_unbound_args(ITensorCrossSolver; recursive=true))
    end
    include("sampling.jl")
    include("tensors.jl")
    include("cross.jl")
    include("operators.jl")
    include("stencils.jl")
    include("evolution.jl")
    include("ordering.jl")
    include("examples.jl")
    include("advection_diffusion_reaction_2d.jl")
end
