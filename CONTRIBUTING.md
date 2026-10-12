# Contributing

Open an issue with a minimal reproducible example for bugs or numerical
questions. Include Julia and dependency versions, grid and boundary conventions,
time step, cross tolerances/rank limits, and an independent reference when possible.

## Local setup

```sh
git clone https://github.com/ElHirad/ITensorCrossSolver.jl.git
cd ITensorCrossSolver.jl
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

The package supports Julia 1.10 and later. Do not commit a local `Manifest.toml`,
generated CSV outputs, coverage files, or downloaded research manuscripts.

## Numerical changes

Add tests against independent stencils, dense calculations, or analytic
solutions. Check real and complex types, boundary behavior, indexing, and input
immutability where relevant. Keep tests deterministic; timing assertions do not
belong in the regression suite. Run the strict suite before submitting:

```sh
julia --project=. -e 'using Pkg; Pkg.test(; julia_args=["--check-bounds=yes", "--depwarn=error"])'
```

Use `benchmarks/derivative_methods.jl` for performance comparisons. Keep grid,
accuracy settings, thread counts, and time integration identical, and report
warm-up, repetitions, timing scope, and accuracy alongside runtime.

## Formatting

The repository pins JuliaFormatter 2.14.0 with the checked-in `.JuliaFormatter.toml`.
The formatter has a separate environment and is not a runtime dependency:

```sh
julia --project=tools -e 'using Pkg; Pkg.instantiate()'
julia --project=tools tools/format.jl
julia --project=tools tools/format.jl --check
```

Keep public behavior and docstrings consistent with `docs/api.md` and
`docs/manual.md`. Contributions are distributed under the repository's MIT license.
