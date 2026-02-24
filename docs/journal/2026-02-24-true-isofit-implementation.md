# True ISOFIT Implementation — 2026-02-24

## Summary

Implemented true ISOFIT (Ciambur 2015) simultaneous harmonic fitting within the
`fit_isophote()` iteration loop. When `simultaneous_harmonics=True`, higher-order
harmonics are now fitted jointly with geometry harmonics (orders 1-2) using an
extended design matrix, rather than being computed post-hoc after convergence.

## Key Changes

### New Functions in `fitting.py`
- `build_isofit_design_matrix(angles, orders)` — extended design matrix
- `fit_all_harmonics(angles, intens, orders)` — simultaneous lstsq fit
- `evaluate_harmonic_model(angles, coeffs, orders)` — full ISOFIT model evaluation

### Modified `fit_isophote()` Loop
- Per-iteration `use_isofit_this_iter` flag after sigma clipping
- Dual-path fitting: `fit_all_harmonics()` vs `fit_first_and_second_harmonics()`
- ISOFIT RMS excludes higher-order signal (cleaner convergence criterion)
- 5x5 covariance sub-matrix for geometry error estimation in ISOFIT mode
- Harmonics stored at best-geometry update during iteration
- Post-convergence guards: skip post-hoc if ISOFIT already stored harmonics
- Fallback with RuntimeWarning when `n_points < 1 + 2*(2 + len(orders))`

## Design Decisions
1. **No numba for ISOFIT path** — variable-width matrices negate JIT benefits
2. **Per-iteration fallback** — decided after sigma clipping, not pre-loop
3. **Harmonics stored during iteration** — at best-geometry update, not only at convergence
4. **Geometry updates unchanged** — `A1, B1, A2, B2 = coeffs[1:5]` in both paths

## Test Results

### Unit Tests (Phase 1-2): 11 new tests
- 6 ISOFIT helper tests (design matrix, coefficient recovery, RMS comparison)
- 5 behavior tests (default path unchanged, convergence, fallback, mixed mode)

### Integration Tests (Phase 3): 4 new tests
- **Boxy Sersic recovery**: median |a4,b4| = 0.043 vs injected 0.04
- **ISOFIT vs post-hoc**: b4 correlation = 1.000, identical median magnitudes
- **M51 regression**: no convergence degradation (55.6% vs 66.7%)
- **EA + ISOFIT**: 100% convergence on eps=0.6 mock, 0.3% intensity accuracy

### Full Suite
175/175 tests pass, zero regressions.

## Remaining
- Phase 5: Performance benchmark (`benchmarks/bench_isofit_overhead.py`)
