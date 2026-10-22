# ITensorCrossSolver.jl

[![Julia tests](https://github.com/ElHirad/ITensorCrossSolver.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/ElHirad/ITensorCrossSolver.jl/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Tensor-train cross interpolation and explicit PDE integration in Julia, built
on [ITensors.jl](https://docs.itensor.org/ITensors/stable/) and
[ITensorMPS.jl](https://docs.itensor.org/ITensorMPS/stable/).

- Adaptive TT cross interpolation using SVD, DEIM, and Gappy-POD sampling.
- Binary quantics grids, MPS/core conversion, and point/fiber evaluation.
- Two derivative approaches: sampled finite-difference neighbors (**pure**) and
  derivative MPO application (**hybrid**).
- Classical RK4 with cross compression, real or complex states, periodic or
  zero-Dirichlet boundaries, and configurable tensor ranks and tolerances.
- Tested examples for viscous Burgers, nonlinear Schrödinger, and 2D nonlinear
  advection–diffusion–reaction (ADR), including an independent dense DNS comparison.

## Install

Requires **Julia 1.10 or later**. Install directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/ElHirad/ITensorCrossSolver.jl", rev="v0.2.0")
using ITensorCrossSolver
```

The package is not registered in Julia's General registry; use the URL above
instead of `Pkg.add("ITensorCrossSolver")`. No Python installation is required.

Version 0.2 adds explicit MSB array conversions and removes reversed derivative
networks: `direction=:reverse` now raises an error. Use MSB-first sites for all
spatial operators. See the [changelog](CHANGELOG.md) for migration details.

## Quick start

```julia
using ITensorCrossSolver

sites = quantics_sites(6)
grid = collect(0:63) .* (2pi / 64)
initial = tt_cross(quantics_function(x -> 0.1sin(x), (grid,)), fill(2, 6);
                   sites, tolerance=1e-10)
@assert initial.converged

# Approach 1: sample physical neighbors directly, without derivative MPOs.
rhs = burgers_rhs(6, 2pi / 64; viscosity=0.1)
solution = solve_pde(rhs, initial.mps, (0.0, 0.1); dt=0.01,
                     cross_kwargs=(; tolerance=1e-9, maxdim=24))
values = dense_vector(last(solution.states))
```

For an already sampled vector, pass the site dimensions explicitly:

```julia
x = collect(0:15) / 15
y = @. exp(x) - x^4
result = tt_cross_tensor(y, fill(2, 4))
values = dense_vector(result.mps)  # same physical grid order as y
```

All grid APIs use **MSB-first sites**, with the last site varying fastest.
Use `msb_tensor(y, dims)` and `msb_vector(A)` for dense conversions, and
`msb_unfolding(A, bond)` for dense DEIM diagnostics. Plain Julia `reshape`
and `vec` use a different storage order. For multidimensional physical fields,
`dense_tensor(psi, (nx, ny))` returns entries indexed as `field[ix, iy]`.

For hybrid derivatives, construct `finite_difference_mpo` operators and pass
them to `burgers_rhs(D1, D2; viscosity=0.1)`. See the
[manual](docs/manual.md) for both approaches, indexing conventions, numerical
limitations, and measured accuracy/runtime comparisons. The
[API guide](docs/api.md) lists every exported function; Julia's `?function_name`
help provides the corresponding docstrings.

## Examples and benchmarks

Clone the repository to run its scripts:

```sh
git clone https://github.com/ElHirad/ITensorCrossSolver.jl.git
cd ITensorCrossSolver.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. examples/burgers_1d.jl
julia --project=. examples/schrodinger_2d.jl
julia --project=. examples/advection_diffusion_reaction_2d.jl
julia --project=. examples/deim_indices_1d.jl
julia --project=. benchmarks/derivative_methods.jl
```

All three examples accept `derivative_method=:pure` or `:hybrid` (the default):

```julia
include("examples/advection_diffusion_reaction_2d.jl")
result = adr2d_example(; derivative_method=:pure, write_output=false)
println(result.metrics.relative_l2_error)
```

The ADR example uses the same grid, centered differences, and RK4 steps for TT
and dense DNS. The benchmark warms both methods and reports repeated solve
times at matched accuracy; its results depend on the problem and hardware.
Generated CSVs are saved under `examples/output/` or `benchmarks/output/`.

## Tests and development

```julia
using Pkg
Pkg.test("ITensorCrossSolver")
```

In a checkout, run `julia --project=. -e 'using Pkg; Pkg.test()'`.
CI covers Julia 1.10 and the latest stable Julia on Linux, macOS, and Windows.
See [CONTRIBUTING.md](CONTRIBUTING.md) for formatting and strict numerical checks.

Cross convergence checks use sampled entries and do not certify global error.
RK4 requires a stable, user-selected time step. Validate accuracy independently
for each application; the examples demonstrate suitable checks.

## License and attribution

Released under the [MIT license](LICENSE). The numerical method and its relation
to the supplied research algorithm are described in the [manual](docs/manual.md).
Use [CITATION.cff](CITATION.cff) to cite this software. Changes are recorded in
[CHANGELOG.md](CHANGELOG.md).
