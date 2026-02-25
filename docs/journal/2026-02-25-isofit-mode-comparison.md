# ISOFIT Mode Comparison — Session 8 (2026-02-25)

## Goal

Add an `isofit_mode` config parameter to compare the original Ciambur 2015 ISOFIT algorithm (post-hoc simultaneous harmonics) against isoster's current in-loop variant, and run a quantitative comparison on LegacySurvey galaxies.

## What Was Done

### 1. Config: `isofit_mode` parameter
- Added `isofit_mode: str = 'in_loop' | 'original'` to `IsosterConfig`
- Only meaningful when `simultaneous_harmonics=True`

### 2. Fitting: "original" mode in `fit_isophote()`
- `isofit_mode='original'`: uses 5-param inside the loop (identical to default), then fits all higher-order harmonics simultaneously post-hoc after convergence
- Saves `best_angles`, `best_intens`, `best_gradient` at best-geometry snapshot for accurate post-hoc fitting
- Three post-hoc exit points updated (convergence, geometry convergence, max-iter)

### 3. Unit tests (3 new, 182 total)
- `test_isofit_original_mode_geometry_matches_default`: confirms geometry, RMS, niter, stop_code are bitwise identical to default 5-param
- `test_isofit_original_mode_stores_harmonics`: verifies post-hoc harmonics are nonzero
- `test_isofit_original_differs_from_in_loop`: both modes produce valid, different results

### 4. Comparison script
- `examples/compare_isofit_modes.py` — runs ESO243-49 and NGC3610 with 3 configs
- Generates QA figures with 2D residual maps + 6-panel 1D profiles

## Key Findings

### Convergence confirms implementation correctness
- **Baseline and original have identical iteration counts and geometry** (both use 5-param in loop)
- In-loop mode has slightly different convergence due to lower RMS from full-model subtraction

### ESO243-49 (edge-on S0)
| Mode | Converged | Mean Iter |
|------|-----------|-----------|
| Baseline | 60/60 | 11.1 |
| ISOFIT in-loop | 58/60 | 12.8 |
| ISOFIT original | 60/60 | 11.1 |

- In-loop loses 2 isophotes — the lower ISOFIT RMS makes convergence criterion harder to satisfy relative to the rescaled threshold
- 2D residual patterns visually identical across modes

### NGC3610 (boxy elliptical)
| Mode | Converged | Mean Iter | Mid |res|% | Mid rms% |
|------|-----------|-----------|---------|---------|
| Baseline | 60/60 | 10.9 | 6.87 | 13.68 |
| ISOFIT in-loop | 60/60 | 11.3 | 6.72 | 13.34 |
| ISOFIT original | 60/60 | 10.9 | 6.79 | 13.62 |

- In-loop gives marginally better mid-region residuals
- Differences in a4/b4 visible at outer radii

### Interpretation
The structured residual patterns in both galaxies are **not caused by the ISOFIT algorithm variant**. All three modes produce nearly identical 2D residuals. The residual structure likely comes from:
1. Real morphological features not captured by low-order harmonics
2. Masking artifacts (both galaxies have significant masked regions)
3. Model reconstruction interpolation effects

## Branch Status
- **Branch**: `feat/isofit-mode-comparison` (uncommitted changes)
- **Base**: `main` at `7643184`
- **Files modified**: `isoster/config.py`, `isoster/fitting.py`, `tests/unit/test_fitting.py`
- **Files created**: `examples/compare_isofit_modes.py`
- **Output figures**: `outputs/isofit_mode_comparison/{eso243-49,ngc3610}/`

## Next Steps
- Decide whether to commit and merge or iterate further
- Consider whether the structured residuals warrant investigation of higher harmonic orders ([3,4,5,6,7]) or other improvements
- Possible: compare against photutils.isophote results for ground truth
