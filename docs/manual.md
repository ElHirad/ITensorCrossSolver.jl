# ITensorCrossSolver manual

[Installation and quick start](../README.md) · [Public API](api.md)

All example and benchmark commands below run from the repository root.

The Burgers example solves

```math
u_t + u u_x = \nu u_{xx},\qquad x\in[0,2\pi),\quad \nu=0.1
```

with periodic boundaries. It uses the Cole–Hopf solution

```math
u(x,t)=\frac{2\nu a e^{-\nu t}\sin x}{1+a e^{-\nu t}\cos x},\qquad a=0.5,
```

both as its initial condition and as an independent accuracy reference. The
default run uses 64 points, `dt=0.01`, and final time `0.1`. It prints relative
L2 error and bond dimensions, and saves `examples/output/burgers_1d.csv` with
the initial, numerical, and exact solutions. Include the file and call
`burgers_example(; nbits=7, dt=0.002)` to change parameters.

The Schrödinger example reproduces the equation and separable initial condition
in the Python file with consistent interior-point Dirichlet grids. It reports
probability-mass drift using MPS norms. Although the Python file is named “4D”,
its active configuration uses two physical dimensions; this example does too.
Derivative MPOs and cross callbacks support other numbers of dimensions.

## 2D advection–diffusion–reaction and DNS comparison

`examples/advection_diffusion_reaction_2d.jl` solves

```math
u_t + u_x - \tfrac12 u_y = 0.05u_{xx} + 0.025u_{yy} + u(1-u),
\qquad (x,y)\in[0,2\pi)^2,
```

with periodic boundaries and smooth nonseparable initial data

```math
u(x,y,0) = 0.3 + 0.12\sin x\cos y + 0.08\cos(2x-y).
```

The default grid is **32 × 32**, with `dt=0.01` and final time `0.2` (20 steps).
The TT calculation uses `multidim_derivative_mpo`, `quantics_function`,
`tt_cross`, `mps_to_cores`, `evaluate`, and `solve_pde`. It applies derivative
MPOs once per RK4 stage and samples the nonlinear reaction through the cross
callback. Cross tolerance is `1e-10`, maximum cross bond dimension is 32, and
MPO-application cutoff is `1e-22` with maximum bond dimension 64. Failed cross
convergence is an error.

The DNS is an independent dense-array implementation of the **same centered
second-order spatial differences and classical RK4 algorithm**. It starts from
the same prescribed initial field and uses every recorded TT time step,
including a shortened final step when needed. It evaluates periodic stencils
directly, without MPOs or tensor compression. Consequently, the comparison
measures TT/MPO approximation and roundoff on a shared discretization; it does
not measure error against the continuum PDE.

Norms use the periodic-grid quadrature weight:

```math
\|v\|_{L^2_h}=\sqrt{\Delta x\,\Delta y\sum_{i,j}|v_{ij}|^2},
\qquad E_{\rm rel}=\frac{\|u_{\rm TT}-u_{\rm DNS}\|_{L^2_h}}
{\|u_{\rm DNS}\|_{L^2_h}}.
```

Measured default run on Julia 1.13.1:

| Quantity at `t=0.2` | Value |
| --- | ---: |
| TT solution L2 norm | 2.21672470201794 |
| DNS solution L2 norm | 2.21672470201307 |
| Absolute L2 error | 5.32708e-10 |
| Relative L2 error | 2.40313e-10 |
| Unweighted Euclidean error | 2.71306e-9 |
| Final maximum TT bond dimension | 16 |

The example prints these metrics and saves `examples/output/adr_2d_fields.csv`
with coordinates, initial field, both final fields, and their difference.
`examples/output/adr_2d_errors.csv` contains norms, errors, and maximum bond
dimensions at every step, including the initial condition. Small numerical
differences across platforms or BLAS implementations are expected.

Include the example to customize the coefficients, rectangular grid, or cross
settings and access all results directly:

```julia
include("examples/advection_diffusion_reaction_2d.jl")
result = adr2d_example(; nbits=(4,5), dt=0.005, final_time=0.1,
    velocity=(0.7,-0.3), diffusivity=(0.06,0.02), reaction_rate=1.2,
    cross_kwargs=(; tolerance=1e-11), write_output=false)
println(result.metrics)
```

`nbits=5` is shorthand for `(5,5)`. Returned fields have shape `(nx,ny)` with
`field[i,j] = u(x[i],y[j])`. The example checks a conservative step-size bound;
reduce `dt` when refining the grid or increasing coefficients. Dense fields and
all TT snapshots are retained for this comparison, so this diagnostic example
is intended for moderate grid sizes.

## Cross interpolation

```julia
using ITensorCrossSolver, Random

sites = quantics_sites(12)
f(idx) = exp(-0.001 * (grid_index(idx)-1)) + 0.1
result = tt_cross(f, fill(2, 12); sites, ranks=2, maxdim=16,
                  tolerance=1e-10, rng=MersenneTwister(42))
@assert result.converged
psi = result.mps
value = evaluate(psi, quantics_index(123, 12))
println((result.evaluations, result.errors, bond_dimensions(psi)))
```

`tt_cross_tensor(A; kwargs...)` provides the same algorithm for an existing
array. Cross interpolation evaluates small fibers, takes local SVDs, chooses
DEIM interpolation rows, and alternates sweep direction. Gappy-POD supplies
additional candidate samples. With `adaptive=true`, sampled singular-value
tails can reduce ranks and failed validation grows rank budgets every two
sweeps. `adaptive=false` fixes the supplied budgets, removing only numerically
null singular directions. `ranks` accepts a scalar or one budget per bond.

Results always use the original site order. `CrossResult` exposes:

| Field | Meaning |
| --- | --- |
| `mps` | ITensorMPS state |
| `errors` | Relative errors on the validation entries after each sweep |
| `sweeps` | Number of completed directional sweeps |
| `evaluations` | Unique callback evaluations, including validation |
| `converged` | Whether the sampled tolerance was met |

Callbacks receive a tuple of **1-based** site indices and must return finite
scalars of a consistent real or complex type. Evaluations are cached within a
cross call, so the callback must be deterministic during that call. The RNG
defaults to a fresh `MersenneTwister(0)` for reproducibility and does not change
Julia's global random state.

Validation uses random entries (all entries for tensors smaller than the
validation sample count). It is a sampled diagnostic, not a rigorous global
error estimate: isolated spikes or unsampled features can be missed. Increase
validation samples and independently check application-specific observables.
No full tensor is assembled by cross interpolation or PDE stepping. Small
problems can nevertheless have all their entries sampled. Evaluation caches
use memory proportional to the unique samples in one cross call.

## Index and boundary conventions

- TT cores have shape `(left bond, physical index, right bond)`, including
  unit boundary bonds. `cores_to_mps` and `mps_to_cores` preserve that order.
- Binary site values `1,2` represent bits `0,1`. The first site is the most
  significant bit. `quantics_index` and `grid_index` convert between a 1-based
  physical grid index and site indices.
- `quantics_function(f, grids)` wraps a physical-coordinate function
  `f(x,y,...)`; each grid has a power-of-two length and coordinate bits are
  grouped by dimension. For example, all x bits precede all y bits.
- `dense_tensor(psi)` returns axes in site order. Julia's first axis varies
  fastest in memory; `dense_vector(psi)` explicitly returns physical grid order
  with the last site varying fastest. Use `dense_vector`, not
  `vec(dense_tensor(...))`, for quantics grid plots. `dense_operator` uses that
  same grid order. These dense utilities are intended for small diagnostics.
- Periodic grids omit the repeated endpoint: `dx=(b-a)/N`, `x=a+(0:N-1)*dx`.
- `:dirichlet` / `:homogeneous` grids contain interior unknowns only:
  `dx=(b-a)/(N+1)`, `x=a+(1:N)*dx`; values at the two external boundary nodes
  are zero. Boundary values are not evolving MPS entries.

`finite_difference_mpo(sites, dx; order=1|2, boundary=:periodic)` constructs
centered second-order derivatives using rank-two binary shift automata. It
does not construct an `N × N` matrix. For multiple physical coordinates use
`multidim_derivative_mpo(sites, [nx_bits,ny_bits], axis, dx; ...)` with a 1-based
axis. `direction=:reverse` reverses the complete MPO site order; apply it to
a state created from `reverse_cores` with reversed site indices.

## Two derivative approaches

Section 2.6 of *A Cross Method for Solving Quantized Tensor Differential Equations*
(Ghahremani, Alipanah, Givi, and Babaee; supplied manuscript, PDF pages 17–19)
describes two ways to obtain derivatives at cross-selected entries:

- **Approach 1, pure TT-CUR-DEIM:** contract the state at the selected entry and
  its finite-difference neighbors. Convert each physical coordinate between its
  grouped binary indices and grid index; carry/borrow and boundary wrapping stay
  within that physical axis. No derivative MPO or derivative MPS is formed.
- **Approach 2, hybrid:** apply each derivative MPO to the current MPS once per
  RK stage, then sample the resulting derivative MPSs. The existing implementation
  uses ITensorMPS's `naive` contraction with controlled truncation.

Both versions retain the package's **same classical RK4 integrator**, including
cross approximation of stage derivatives, stage states, and the final update.
This deliberately isolates the derivative evaluation strategy. It does not
implement the paper's additional hybrid stage-elimination or prior-step index
reuse strategies, so its timings are not a reproduction of the paper's full
integrator comparison.

All three examples accept `derivative_method=:pure` or `:hybrid` (the default):

```julia
include("examples/advection_diffusion_reaction_2d.jl")
pure = adr2d_example(; derivative_method=:pure, write_output=false)
hybrid = adr2d_example(; derivative_method=:hybrid, write_output=false)
```

For custom equations, the new public interface returns shared callbacks:

```julia
# u is a binary MPS with 5 x-bits followed by 6 y-bits.
samples = finite_difference_samples(u, (5,6), (2pi/32,2pi/64);
                                    boundary=:periodic, cache=true)
u_at_point = samples.value(idx)
ux = samples.first(idx, 1)
uyy = samples.second(idx, 2)
```

The stencils are the same centered, second-order differences used by the MPOs.
`boundary` accepts a single symbol or per-axis symbols, including `:dirichlet`
and its alias `:homogeneous` for zero exterior values on an interior grid.
Real and complex states, unequal axis sizes, and total bit counts larger than
one machine word are supported; each individual axis must fit the existing
`grid_index` convention. The callbacks retain a snapshot of the state.

Each stage's callbacks share a cache of sampled state values, including stencil
neighbors. Repeated neighboring entries are contracted only once, and the cache
is released with the stage callback. Its storage grows with the unique sampled
entries and may reach the entire grid on small problems. Set `cache=false`
(`sample_cache=false` in the ADR example) to disable this reuse. A new evaluator
must be built whenever the stage state changes; the supplied RHS factories do so.

The package also provides pure versions of the existing PDE factories:

```julia
rhs_burgers = burgers_rhs(6, 2pi/64; viscosity=0.1, boundary=:periodic)
rhs_nls = schrodinger_rhs((4,4), (pi/17,pi/17), potential;
                          epsilon=1e-4, boundary=:dirichlet)
```

### Reproducible runtime comparison

```sh
julia --project=. benchmarks/derivative_methods.jl
# Optional: seven repetitions on 16², 32², and 64² grids
julia --project=. benchmarks/derivative_methods.jl 7 4,5,6
```

The benchmark runs the nonlinear periodic ADR example above through `t=0.2`
with `dt=0.01`, identical initial MPSs, cross settings, and snapshot frequency.
It warms both complete timed code paths, interleaves method order, runs a full
GC before each observation, and reports the median of five repetitions. BLAS
uses one thread. Every measured solve must satisfy a relative L2 error below
`1e-8` against the independent same-grid DNS.

Solve timings include all RK4 stages, cross interpolation, derivative evaluation,
and per-stage caches. They exclude JIT warm-up, initial interpolation, one-time
MPO construction, DNS, dense reconstruction, and file I/O. Warm RHS/operator
setup is measured separately. Allocation figures are cumulative bytes allocated
per solve, **not peak memory**. The benchmark writes raw repetitions, summaries,
and environment/settings to `benchmarks/output/`.

Measured on October 12, 2026, with **Julia 1.13.1, Apple M2, one Julia thread,
and one BLAS thread**, using five warmed repetitions per method and grid:

| Grid | Pure median (s) | Hybrid median (s) | Pure / hybrid time | Pure relative L2 error | Hybrid relative L2 error |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 × 16 | 1.036 | 1.191 | 0.870 | 1.87246e-10 | 1.87246e-10 |
| 32 × 32 | 2.204 | 2.150 | 1.025 | 2.40315e-10 | 2.40313e-10 |
| 64 × 64 | 7.721 | 5.454 | 1.416 | 2.75290e-10 | 2.75122e-10 |

Both methods finish with maximum bond dimensions 11, 16, and 19, respectively.
Their pairwise relative L2 differences are `6.25e-15`, `4.60e-14`, and `1.46e-12`.
At 64 × 64, pure timings ranged from 7.654–7.853 s and hybrid from 5.406–5.482 s.
One-time warm hybrid operator setup took 4.2–6.5 ms and is excluded from these
solve times. See `benchmarks/output/derivative_timings.csv` for every observation,
`derivative_summary.csv` for medians/ranges/allocations, and `environment.txt`
for settings and hardware.

**Recommendation for this implementation and ADR workload:** retain hybrid as
the default. It uses about **29% less time on 64 × 64** at matched accuracy.
The methods are close on 32 × 32; pure is faster on 16 × 16, where it uses about
13% less time. Pure also allocated fewer cumulative bytes in these runs, but
peak memory was not measured. This is a workload-dependent crossover, not a
universal ranking: grid size, tensor ranks, stencil width, cached-neighbor reuse,
MPO contraction method, and tolerances can change the result. Benchmark the
actual application before choosing a backend for a substantially different PDE.

## PDE interface

```julia
sites = quantics_sites(6)
dx = 2pi / 64
grid = collect(0:63) .* dx
initial = tt_cross(quantics_function(x -> 0.1sin(x), (grid,)),
                   fill(2,6); sites, tolerance=1e-10).mps
D1 = finite_difference_mpo(sites, dx; order=1)
D2 = finite_difference_mpo(sites, dx; order=2)
rhs = burgers_rhs(D1, D2; viscosity=0.1)
solution = solve_pde(rhs, initial, (0.0,0.1); dt=0.01,
                     cross_kwargs=(; tolerance=1e-9, maxdim=24))
values = dense_vector(last(solution.states))
```

For a custom PDE, provide `rhs(u::MPS,t)`, returning a callback `f(idx)` for an
entry of the RHS. Construct derivative MPSs or a shared sampled-stencil evaluator
once when constructing that callback.
`approximate_rhs` returns a `CrossResult`; `rk4_step` returns an MPS.
`solve_pde` returns `PDESolution(times, states)`, including the initial and final
states. `saveevery` controls snapshot frequency. The last step is shortened
to reach the requested final time, and a zero-length interval takes no steps.

`schrodinger_rhs(laplacians, potential; epsilon)` implements

```math
\psi_t = \frac{i}{2}\sum_j D_{2,j}\psi-i V\psi-i\epsilon|\psi|^2\psi.
```

`potential` can be a scalar, an MPS, or a callback `(idx,t)`. The potential is
the total potential; build a sum of per-coordinate potentials in its callback.
MPOs and the initial state must share the same ITensor site indices.

Each RK4 stage and the final update are compressed by cross interpolation.
PDE functions default to `strict=true`, which throws on a failed cross
convergence check. Explicitly passing `strict=false` permits an unconverged
approximation. Spatial, temporal, MPO-truncation, and cross errors are separate:
`cross_kwargs.tolerance` controls only sampled cross error, and the RHS factory's
`cutoff`/`maxdim` control MPO application. ITensor's cutoff measures discarded
singular-value weight and is not the same as a relative cross tolerance.

RK4 uses a user-selected fixed time step and has no automatic error controller.
The Burgers example checks a conservative advection/diffusion step restriction.
The centered Burgers discretization is intended for resolved smooth viscous
solutions; it is not a shock-capturing discretization. Runtime grows with ranks,
and a general high-rank solution may not be compressible.

## Mapping from the Python file

| Python function or block | Julia equivalent |
| --- | --- |
| `mpo_multidim_acc2_diff` | `multidim_derivative_mpo`, `finite_difference_mpo` |
| `myDEIM` | `deim` |
| `GappyPOD` | `gappy_pod` |
| `Eval_IC` | `quantics_function` with `evaluate_fiber` |
| `TT_DEIM_func` | `tt_cross` |
| `TT_DEIM_tensor` | `tt_cross_tensor` |
| `Eval_RHS` | `schrodinger_rhs(...)(u,t)`, sampled with `evaluate_fiber` |
| `Approximate_RHS` | `approximate_rhs` |
| `Fiber_eval` | `rk4_fiber` |
| `core_to_mps` | `cores_to_mps` |
| `mps_to_core` | `mps_to_cores` |
| `mpsHadamardProd` | `hadamard`, `diagonal_mpo` |
| time loop and rank updates | `rk4_step`, `solve_pde`, adaptive `tt_cross` |
| forward/reverse core manipulation | `reverse_cores` |
| dense contractions for plotting | `dense_tensor`, `dense_vector` |

The port removes global state and flattened triangular index bookkeeping.
It uses solves instead of explicit inverses, prevents duplicate pivots, handles
rank-one Gappy-POD and complex absolute squares, and does not swap the meaning
of forward/reverse arrays after an odd number of sweeps. The original ad hoc
upper/lower rank thresholds are replaced by singular-tail truncation plus
sampled-error-driven rank growth. The original `split` storage option is
unnecessary: the Julia derivative interface explicitly uses grouped quantics
bits, and can accept different bit counts per physical dimension.

`hadamard` accepts `method=:direct`, `:zipup`, or `:fit`. These select ITensorMPS
naive contraction, zip-up contraction, or variational fitting. `:fit` defaults to
two sweeps and `outputlevel=0`; override these with options such as `nsweeps=3`.
The sweep count must be a positive integer. For an MPO factor,
`leg=:lower` computes `A*diag(u)` and `leg=:upper` computes `diag(u)*A`.
Products never conjugate automatically. Scalars multiply normally and `nothing`
propagates. Input states and operators are not mutated.

## Tests

Tests exercise every exported function and all three Hadamard algorithms,
including both MPO legs. They compare cross decompositions with independent
random dense tensors, first/second derivative MPOs with directly assembled
stencils, complex nonlinear Schrödinger and Burgers RK4 steps with dense
references, fourth-order temporal convergence, and second-order Burgers spatial
convergence against the exact Cole–Hopf solution. Other cases include unequal
mode sizes, odd/even sweeps, one-site/zero/rank-one inputs, rank growth, convergence
failure, invalid arguments, input immutability, multidimensional padding and
reversal, and a 24-bit derivative operator without dense construction.

The 2D ADR tests independently check finite-difference Fourier symbols, nonlinear
logistic RK4 convergence against its exact spatially uniform solution, sampled
and compressed RHS values, and full TT/DNS agreement on square and rectangular
grids. They also check shortened final steps, L2 normalization, CSV axis ordering,
invalid inputs, and strict rejection of failed initial cross interpolation.

Pure-derivative tests check every grid entry against independent dense stencils
for real and complex states, first and second derivatives, two-point and unequal
axis sizes, and periodic/Dirichlet/mixed boundaries. They exercise carry/borrow,
cache on/off, snapshots, mixed scalar types, and a 66-site state without dense
reconstruction. Burgers, Schrödinger, and ADR integration tests compare both
derivative approaches, including the default 32 × 32 ADR case against DNS.

Regression tests also check scale-invariant rank selection and Gappy-POD
sampling at amplitudes from `1e-200` to `1e200`, Float32/ComplexF32 inputs,
default variational-fitting options, zero products, bond limits, singleton
modes, one-site PDE states, and method ambiguities. Rank selection and Gappy-POD
normalize their spectra/bases before squaring to prevent amplitude-dependent
underflow or overflow.

CI is configured to test Julia 1.10 on Linux and the latest stable Julia 1.x
release on Linux, macOS, and Windows. The latest-version jobs also run all three default examples.

Validated on **Julia 1.13.1**, macOS Apple Silicon, with ITensors 0.9.32 and
ITensorMPS 0.4.2 on October 12, 2026:

- **1,183 tests passed**, with array-bounds checks enabled and deprecations treated
  as errors; no method ambiguities or unbound type parameters were found.
- **541/541 instrumented source lines exercised** (100% measured line coverage).
  Line coverage does not imply exhaustive branch or numerical-input coverage.
- Default 64-point Burgers example: relative L2 error `6.49e-5`, maximum bond
  dimension 8, compared with the exact Cole–Hopf solution.
- Default Schrödinger example: relative probability-mass drift `7.61e-13`.
- Default 32 × 32 ADR example: absolute L2 error `5.33e-10`, relative L2 error
  `2.40e-10` versus the same-grid DNS; maximum bond dimension 16. Both derivative
  methods satisfy this accuracy; their relative difference is `4.60e-14`.

Reproduce the strict test run with:

```sh
julia --project=. -e 'using Pkg; Pkg.test(; coverage="user", julia_args=["--check-bounds=yes", "--depwarn=error"])'
```

Dependency lockfiles and generated solver outputs are local artifacts and are
excluded from the published package. Resolve dependencies with `Pkg.instantiate()`
in a checkout, or use the URL installation instructions in the README.
