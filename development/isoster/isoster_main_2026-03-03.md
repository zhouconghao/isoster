---
date: 2026-03-03
repo: isoster
branch: main
tags:
  - journal
  - documentation
  - roadmap
  - autoprof
  - optimization
  - phase-26
  - lazy-gradient
---

## Progress

- Aligned `algorithm.md`, `configuration-reference.md`, `spec.md`, and `user-guide.md` with current codebase behaviors.
- Conducted theoretical analysis of `autoprof` and validated `isoster` as a Newton-Raphson optimizer.
- Identified missing gradient uncertainty term in geometric error propagation.
- Rewrote `docs/future.md` with a strategic roadmap (Lazy Gradient, Noise Floor, Vectorized Forced Photometry, WLS).
- **Implemented Lazy Gradient Evaluation** (Modified Newton Method) in `isoster/fitting.py`.
- Added `use_lazy_gradient` to `IsosterConfig`.
- Optimized fitting loop to reuse radial gradient unless convergence stalls (3 non-improving iterations).
- Created `benchmarks/lazy_gradient/run_benchmark.py` with high-quality QA (2D residuals + 5-panel profiles).
- Verified **~45% reduction in sampling calls** and **1.4x–2.8x speedup** over normal mode.
- Updated `docs/todo.md` (Phase 26) and consolidated document improvements.

## Lessons Learned

- `isoster` stability in LSB regions is limited by the dependency on the radial gradient's inverse; freezing the Jacobian (Lazy Gradient) is a safe optimization when geometry shifts are minute.
- Geometric error bars in the outer disk are currently underestimated due to the omission of the $\sigma^2_g / g^4$ term.
- Instrumenting core functions via `unittest.mock.patch` effectively verifies complexity reductions.
- Forced photometry can be radicalized from $O(N_{sma})$ to $O(1)$ grid projection using `binned_statistic`.

## Key Issues

- Implement remaining Phase 26 items: corrected error propagation math and `sigma_bg` noise floor.
- Investigate performance impact of `binned_statistic` vs. `map_coordinates` for whole-image projection.
- Monitor for geometry drift in highly asymmetric cases with Jacobian freezing.
