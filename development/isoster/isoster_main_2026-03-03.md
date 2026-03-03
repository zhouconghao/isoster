---
date: 2026-03-03
repo: isoster
branch: main
tags:
  - journal
  - documentation
  - roadmap
  - autoprof
---

## Progress

- Aligned `algorithm.md`, `configuration-reference.md`, `spec.md`, and `user-guide.md` with the current codebase implementation.
- Removed non-existent `forced=True` and `forced_sma` parameters from documentation.
- Updated all "Forced Mode" references to use the `template` argument in `fit_image()`.
- Corrected `stop_code=1` logic description to `actual_points < total_points * (1.0 - fflag)`.
- Synchronized `compute_gradient` normalization description with the actual implementation ( `/ delta_r`).
- Conducted a theoretical deep-dive into the `autoprof` algorithm, validating the Jedrzejewski step as a Newton-Raphson optimizer.
- Identified a critical omission in geometric error propagation (missing gradient uncertainty term).
- Created a synthesized roadmap in `docs/archive/review/autoprof-7-synthesis-and-paths.md` and `docs/archive/review/isoster-efficiency-first-principles.md`.
- Rewrote `docs/future.md` with a high-impact strategy: Lazy Gradient Evaluation, Explicit Noise Floor, Vectorized Template Photometry, and WLS via Variance Maps.
- Developed a concrete implementation plan for Phase 26 (Efficiency & Accuracy) in `docs/plan/phase-26-efficiency-accuracy.md`.
- Deleted the obsolete `docs/stop-codes.md` file.
- Merged all changes from `doc-alignment-2026-03-03` into `main`.

## Lessons Learned

- `isoster`'s speed comes from its analytic Newton-Raphson step, but its stability in LSB regions is limited by the dependency on the radial gradient's inverse.
- Reusing the previously converged isophote for gradient estimation (Lazy Gradient) is a safer and faster optimization than profile-based estimation.
- Geometric error bars in the outer disk are currently underestimated due to the omission of the $\sigma^2_g / g^4$ term.
- Forced photometry can be radicalized from $O(N_{sma})$ interpolations to $O(1)$ grid projection using `binned_statistic`.

## Key Issues

- Implement Phase 26: Lazy Gradient Evaluation, `sigma_bg` noise floor, and corrected error propagation math.
- Investigate performance impact of `binned_statistic` vs. `map_coordinates` for Idea A (Whole-Image Grid Projection).
- Monitor for geometry drift or stalling when Jacobian freezing is active.
