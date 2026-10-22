# Changelog

## 0.2.0 — 2026-10-22

- Standardize cross-local row packing and small-grid enumeration on MSB-first order.
- Add `msb_tensor`, `msb_vector`, `msb_unfolding`, vector input with explicit
  dimensions for `tt_cross_tensor`, and grouped-grid output for `dense_tensor`.
- Use the shared grid conversions in the ADR example and derivative benchmark.
- Reject reversed derivative networks (`direction=:reverse`); spatial operators
  now support only MSB-first sites. Internal backward cross sweeps remain supported.
- Reject noninteger binary site values and add asymmetric ordering regressions
  covering conversion, interpolation, derivatives, products, and RK4.
- Add the 16-point DEIM tracing example. Fixed-seed pivot sequences can change
  because local row packing and sample enumeration are now consistent.

## 0.1.0 — 2026-10-12

Initial public release.

- Adaptive tensor-train cross interpolation with DEIM and Gappy-POD enrichment.
- Quantics indexing, MPS/core conversion, and sampled/dense diagnostic utilities.
- Centered finite-difference MPOs and pure sampled-neighbor derivatives.
- MPS/MPO Hadamard products and classical RK4 PDE integration.
- Burgers, nonlinear Schrödinger, and nonlinear 2D ADR examples.
- Independent same-grid DNS validation and repeated derivative benchmarks.
- Tests, installation documentation, and cross-platform Julia CI.
