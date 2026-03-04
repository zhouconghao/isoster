---
date: 2026-03-04
repo: isoster
branch: main
tags:
  - journal
  - phase-27
  - noise-floor
  - error-propagation
  - autoprof
  - mockgal
---

## Progress

- **Implemented Phase 27**: Explicit noise floor (`sigma_bg`) and corrected geometric error propagation.
- Added `sigma_bg` and `use_corrected_errors` to `IsosterConfig`.
- Modified `fit_isophote` to use `max(rms, sigma_bg/sqrt(N))` as a hard lower bound for convergence.
- Overhauled `compute_parameter_errors` to include the radial gradient uncertainty term ($\sigma_g$) in $\epsilon$ and $PA$ variance.
- Created and installed the **`mockgal` repository-level skill** for realistic multi-component Sersic rendering via `libprofit`.
- Generated $z=0.2$ HSC-Wide mock images for **NGC 1453** and **NGC 3585** to test the LSB regime.
- Created `benchmarks/noise_floor/run_benchmark.py` comparing isoster (Baseline/Fix), photutils, and AutoProf.
- Overhauled QA figures with **3 image panels** (Data, Baseline Res, Fix Res) and **5 profile panels** (SB, SB Diff, Centroids, b/a, PA).
- Verified that `sigma_bg` logic prevents overfitting in outskirts and increases geometric error bars by ~6x–12x to realistic levels.
- **Performance**: Confirmed `sigma_bg` adds no overhead; `isoster` is ~30x faster than photutils and ~100x faster than AutoProf on large mocks.
- Updated `docs/todo.md`, `docs/algorithm.md`, and `docs/configuration-reference.md`.
- Merged `feat/noise-floor-and-error-fix` into `main` via merge-commit.

## Lessons Learned

- Analytic Newton-Raphson stability in LSB regions depends on bounding the Jacobian scaling; `sigma_bg` provides a physical floor that prevents chasing noise.
- Omission of the gradient uncertainty term ($\sigma_g/g^2$) causes standard residual-based methods to report "false confidence" in the outer disk.
- SB-difference plots ($\Delta I/I$) are far more sensitive to fitting quality than log-intensity plots in the noise-dominated regime.
- AutoProf's global optimization is robust but significantly slower (~100x) than `isoster`'s vectorized local updates.

## Key Issues

- Roadmap Item 3: Vectorized Template Photometry (grid-projection extraction using `binned_statistic`).
- Evaluate systematic impacts of `sigma_bg` under-estimation in real data.
