# Public API

[Installation](../README.md) · [Numerical manual](manual.md)

Load the package with `using ITensorCrossSolver`. Every public function has a
Julia docstring, accessible with `?name` in the REPL.

## Cross interpolation and sampling

| Function or type | Purpose |
| --- | --- |
| `deim(U, r)` | Select distinct DEIM interpolation rows from a basis. |
| `gappy_pod(U, oversampling, indices)` | Enrich interpolation rows with Gappy-POD samples. |
| `tt_cross(f, dims; kwargs...)` | Approximate an entry callback as an MPS with adaptive cross interpolation. |
| `tt_cross_tensor(A; kwargs...)` | Interpolate an array whose axes are already in MSB-first site order. |
| `tt_cross_tensor(values, dims; kwargs...)` | Interpolate a physical-order vector with explicit MSB-first site dimensions. |
| `CrossResult` | MPS, sampled errors, sweep count, evaluation count, and convergence flag. |

Important `tt_cross` keywords are `sites`, `ranks`, `maxdim`, `tolerance`,
`maxsweeps`, `oversampling`, `adaptive`, `validation_samples`, and `rng`.
Check `result.converged`; the reported sampled error is not a global certificate.

## Quantics indexing and MPS utilities

| Function | Purpose |
| --- | --- |
| `quantics_sites(n)` | Create `n` binary physical indices. |
| `quantics_index(i, n)` | Encode a 1-based grid index as MSB-first binary site values. |
| `grid_index(bits)` | Decode 1-based binary site values to a grid index. |
| `quantics_function(f, grids)` | Wrap a coordinate function for grouped-bit cross sampling. |
| `msb_tensor(values, dims)` | Convert a physical-order vector to a tensor with the last axis fastest. |
| `msb_vector(A)` | Flatten site or physical-grid axes in MSB-first order. |
| `msb_unfolding(A, bond)` | Dense prefix/suffix matrix with MSB-first rows and columns. |
| `cores_to_mps(cores; sites)` | Convert `(left, physical, right)` cores to an MPS. |
| `mps_to_cores(psi)` | Extract core arrays including unit boundary bonds. |
| `reverse_cores(cores)` | Low-level network permutation used by internal backward sweeps; restore order before grid operations. |
| `evaluate(psi_or_cores, idx)` | Contract one entry. |
| `evaluate_fiber(f, dims, mode, left, right)` | Evaluate a selected fiber of a callback. |
| `evaluate_fiber(psi, mode, left, right)` | Evaluate a selected fiber of an MPS. |
| `bond_dimensions(psi)` | Return internal bond dimensions. |
| `dense_tensor(psi)` | Reconstruct in site-axis order for small diagnostics. |
| `dense_tensor(psi, dims)` | Reconstruct with grouped physical-grid axes, e.g. `(nx,ny)`. |
| `dense_vector(psi)` | Reconstruct in physical grid order, last site fastest. |
| `dense_operator(A)` | Reconstruct an MPO with the same grid ordering. |

Each physical dimension occupies a consecutive MSB-first group of sites.
The last physical axis varies fastest in flattened grid data. `dense_vector`
equals `msb_vector(dense_tensor(...))`; Julia's `vec` uses a different order.
`tt_cross_tensor(A)` preserves entries exactly as supplied and cannot infer
whether the caller used a plain Julia `reshape`. Use the vector overload or
`msb_tensor` to encode sampled vectors without reversing their bits.

## Derivatives and products

| Function | Purpose |
| --- | --- |
| `finite_difference_mpo(sites, dx; order, boundary)` | Centered first/second derivative on one binary grid. |
| `multidim_derivative_mpo(sites, bits, axis, dx; kwargs...)` | Embed a derivative along one physical axis. |
| `finite_difference_samples(u, bits, spacings; boundary, cache)` | Return `value(idx)`, `first(idx,axis)`, and `second(idx,axis)` callbacks without derivative MPOs. |
| `diagonal_mpo(psi)` | Construct the pointwise multiplication MPO. |
| `hadamard(u, v; method, cutoff, maxdim)` | Pointwise MPS multiplication. |
| `hadamard(u, A; leg, kwargs...)` | Multiply an MPO's input/output entries by an MPS. |

Boundaries are `:periodic`, `:dirichlet`, or the alias `:homogeneous`. Periodic
grids exclude their repeated endpoint; Dirichlet grids contain interior points.
MPO derivative functions reject `direction=:reverse`; their legacy `direction`
keyword accepts only `:forward`. Both derivative methods use the same MSB-first
grouped-bit ordering. The sampled method accepts per-axis boundary symbols.

Hadamard methods are `:direct`, `:zipup`, and `:fit`; `:fit` defaults to two
sweeps. For an MPO, `leg=:lower` computes `A * diag(u)` and `:upper` computes
`diag(u) * A`. Scalars and `nothing` are supported.

## PDE integration

| Function or type | Purpose |
| --- | --- |
| `burgers_rhs(D1, D2; viscosity, cutoff, maxdim)` | Hybrid viscous Burgers RHS. |
| `burgers_rhs(nbits, dx; viscosity, boundary, cache)` | Pure sampled-stencil Burgers RHS. |
| `schrodinger_rhs(laplacians, potential; epsilon, cutoff, maxdim)` | Hybrid nonlinear Schrödinger RHS. |
| `schrodinger_rhs(bits, spacings, potential; epsilon, boundary, cache)` | Pure nonlinear Schrödinger RHS. |
| `approximate_rhs(rhs, u, t; cross_kwargs, strict)` | Cross-approximate a stage RHS; returns `CrossResult`. |
| `rk4_fiber(u, stages, dt, mode, left, right)` | Evaluate a fiber of a four-stage RK4 combination. |
| `rk4_step(rhs, u, t, dt; cross_kwargs, strict)` | Advance one classical RK4 step. |
| `solve_pde(rhs, initial, tspan; dt, saveevery, cross_kwargs, strict)` | Integrate with fixed steps and a shortened final step. |
| `PDESolution` | Saved `times` and independent MPS `states`. |

A custom `rhs(u, t)` returns an entry callback `f(idx)`. Prepare the derivative
MPSs or shared stencil evaluator once per stage inside this factory. Pass cross
options as a named tuple. `strict=true` rejects failed sampled convergence, but
does not control spatial/time-discretization error or explicit-method stability.

The three example scripts and benchmark driver have their own documented
functions; these are demonstrations, not exported package API.
