# Phase 25: Benchmark isoster vs AutoProf

## Context

Before public release, isoster needs comprehensive benchmarks against both
photutils (done) and AutoProf — a second independent isophote fitting method
using FFT-based global optimization with radial regularization.

AutoProf is an "automatic pipeline" with built-in background, PSF, and center
determination.  For **fair benchmarking**, these automatic steps must be
bypassed — we provide the same fixed initial values to both isoster and AutoProf.

## Branch

`feat/bench-vs-autoprof` (off `main`)

## Deliverables

| ID | File | Purpose |
|----|------|---------|
| P25-001 | `docs/plan/phase-25.md` | This plan |
| P25-002 | `docs/todo.md` | Updated with Phase 25 section |
| P25-003 | `docs/lessons-autoprof.md` | AutoProf-specific lessons |
| P25-004 | `benchmarks/utils/autoprof_adapter.py` | AutoProf wrapper + profile parser |
| P25-005 | `benchmarks/performance/bench_vs_autoprof.py` | Main benchmark script |

## Fair Comparison Strategy

Both tools receive **identical** inputs:

1. Same FITS image (no preprocessing)
2. Same fixed center (x0, y0)
3. Same initial ellipticity and PA
4. Same background level and noise estimate
5. AutoProf's automatic background/PSF/center steps bypassed via `ap_set_*` parameters

## Galaxy Registry

| Galaxy | pixscale | zeropoint | center | eps | pa (rad) | Source |
|--------|----------|-----------|--------|-----|----------|--------|
| IC3370_mock2 | 0.168 | 27.0 | (566, 566) | 0.239 | -0.489 | FITS header + IC3370 config |
| eso243-49 | 0.25 | 22.5 | auto-detect | TBD | TBD | CD1_1 header; zp assumed |
| ngc3610 | 1.0 | 22.5 | auto-detect | TBD | TBD | CD1_1 header; zp assumed |

## AutoProf Bypass Parameters

| AutoProf Parameter | Purpose | Value to set |
|-------------------|---------|-------------|
| `ap_set_center` | Fix center | `{'x': x0, 'y': y0}` |
| `ap_set_background` | Fix background level | measured value |
| `ap_set_background_noise` | Fix noise estimate | measured sigma |
| `ap_isoinit_pa_set` | Fix initial PA (degrees) | astro convention |
| `ap_isoinit_ellip_set` | Fix initial ellipticity | same eps |
| `ap_doplot` | Disable diagnostic plots | `False` |
| `ap_fluxunits` | Output in intensity | `'intensity'` |

## Key Risks

1. **AutoProf failure on edge-on galaxies** — eso243-49 may be challenging.
2. **PA convention** — wrapping needs careful validation.
3. **AutoProf log pollution** — always set `loggername` to redirect into output dir.
4. **Missing zeropoints** — eso243-49 and ngc3610 use 22.5 (LegacySurvey default).

## Verification

1. `uv run python benchmarks/performance/bench_vs_autoprof.py --quick --plots` completes
2. Output files exist: `summary.json`, `summary.csv`, `comparison_IC3370_mock2.png`
3. Profile plots show reasonable agreement
4. Timing numbers are plausible
