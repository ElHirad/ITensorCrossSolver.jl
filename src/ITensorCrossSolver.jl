"""
Tensor-train cross interpolation, quantics finite differences, and explicit PDE
time integration built on ITensors.jl and ITensorMPS.jl.
"""
module ITensorCrossSolver

using LinearAlgebra
using Random
using ITensors
using ITensorMPS

export deim,
    gappy_pod,
    CrossResult,
    tt_cross,
    tt_cross_tensor,
    cores_to_mps,
    mps_to_cores,
    reverse_cores,
    evaluate,
    evaluate_fiber,
    dense_tensor,
    dense_vector,
    dense_operator,
    bond_dimensions,
    quantics_sites,
    quantics_index,
    grid_index,
    quantics_function,
    msb_tensor,
    msb_vector,
    msb_unfolding,
    finite_difference_mpo,
    multidim_derivative_mpo,
    finite_difference_samples,
    hadamard,
    diagonal_mpo,
    rk4_fiber,
    burgers_rhs,
    schrodinger_rhs,
    approximate_rhs,
    rk4_step,
    solve_pde,
    PDESolution

include("sampling.jl")
include("tensors.jl")
include("cross.jl")
include("operators.jl")
include("stencils.jl")
include("evolution.jl")

end
