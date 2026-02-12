# Test and Benchmark Improvement Plan

## Objective
Improve test meaningfulness and benchmark diagnostic quality for ISOSTER by:
- hardening assertions to prevent false-pass behavior
- increasing public API and real-data coverage
- standardizing quantitative criteria and artifacts
- preparing a higher-fidelity mock-generation pathway for future benchmark/test expansion

## Scope and Boundaries
- This plan covers `tests/`, `benchmarks/`, test/benchmark documentation, and related output standards.
- This plan does not change core fitting algorithms directly unless required by testability/benchmark instrumentation.

## User-Defined Directives (Persisted)
1. Treat `examples/data/m51/M51.fits` as the canonical basic real-data test dataset.
2. Rename the basic real-data test to `m51_test`.
3. For future mock generation, use `/Users/mac/Dropbox/work/project/otters/isophote_test/mockgal.py` (libprofit-based) for PSF/noise-capable mock images.
4. For noiseless single-Sersic validation without PSF, use analytic 1-D Sersic truth with accurate `b_n` evaluation (not low-accuracy approximations).
5. Keep tests and benchmarks quantitative using explicit statistics from 1-D deviations and 2-D residuals.

## Workstreams

### Workstream A: Test Reliability Hardening
- Replace conditional metric checks that currently allow silent pass when no valid points exist.
- Add explicit minimum valid-point assertions before computing profile metrics.
- Convert diagnostic-only tests into pass/fail tests with threshold assertions.
- Ensure real-data tests fail clearly when expected input is missing vs silently skipping in normal usage where appropriate.

### Workstream B: Public API Coverage Completion
- Add direct tests for:
  - `isoster.fit_isophote`
  - `isoster.isophote_results_to_astropy_tables`
  - `isoster.plot_qa_summary` (file artifact generation + basic structure checks)
  - CLI entry (`isoster/cli.py`) with smoke-level integration validation
- Add regression tests for deprecated surface (`build_ellipse_model`) to guard compatibility behavior.

### Workstream C: Real-Data Baseline Coverage
- Rename and normalize M51 basic real-data test entrypoint to `m51_test`.
- Add a reproducible artifact bundle for M51 test runs:
  - profile summary JSON
  - stop-code distribution
  - QA figure with overplotted isophotes and residual map
- Define strict but realistic M51 quality gates based on measured baselines.

### Workstream D: Mock Truth Pipeline (Current + Future)
- Near-term:
  - Standardize analytic Sersic truth generation to use accurate `b_n` computation (for example, `scipy.special.gammaincinv`).
  - Align 1-D truth evaluation across tests/benchmarks to avoid mixed approximations.
- Future:
  - Add an adapter workflow to call `mockgal.py` from the `isophote_test` repo for high-fidelity mocks with:
    - PSF convolution
    - sky/background noise via `sblimit`
    - paired noiseless truth images for aperture-flux validation
  - Store recipe metadata (input config + seed + environment) with each generated mock set.

### Workstream E: Benchmark Diagnostics and Output Standardization
- Standardize all benchmark/profiling outputs under `outputs/benchmarks_*`.
- Ensure each benchmark writes machine-readable summaries (JSON/CSV) by default.
- Ensure each benchmark can emit QA plots by default for failing/borderline cases.
- Persist profiler artifacts (`.prof` + parsed hotspot summary) rather than console-only output.
- Add environment metadata to benchmark outputs:
  - git SHA
  - Python/NumPy/SciPy/Numba versions
  - CPU info and relevant env vars

## Quantitative Criteria Framework

## 1-D Profile Criteria
- Primary metric: relative intensity residual
  - `delta_I = (I_isoster - I_truth) / I_truth`
- Report at minimum:
  - median `|delta_I|`
  - max `|delta_I|`
  - valid-point count used for metrics
- Radial windows:
  - noiseless, no-PSF single-Sersic: `[max(3 px, 0.5 Re), 8 Re]`
  - noisy mocks: `[max(3 px, 0.5 Re), 5 Re]`
  - with PSF: ignore `<= 2 * psf_fwhm`

## 2-D Residual Criteria
- Primary metrics:
  - fractional residual: `100 * (model - data) / data`
  - fractional absolute residual: `100 * |model - data| / data`
  - chi-square map for noisy data: `(model - data)^2 / sigma^2`
- Report per radial band:
  - `<0.5 Re`
  - `0.5-4 Re`
  - `4-8 Re`
- Statistics per band:
  - median fractional residual
  - median fractional absolute residual
  - max absolute fractional residual
  - integrated chi-square (when sigma is defined)

## Threshold Definition Policy
- Do not invent new numeric thresholds.
- Use one baseline-calibration phase to measure current distributions, then lock thresholds from measured statistics and record them in this document.
- Any threshold update requires:
  - before/after benchmark evidence
  - documented rationale in `docs/journal/`

## Locked Thresholds (2026-02-11)

Source files:
- Baseline metrics: `outputs/tests_integration/baseline_metrics/phase4_profile_baseline_metrics.json`
- Locked threshold record: `benchmarks/baselines/phase4_profile_thresholds_2026-02-11.json`

Locked values (directly measured, no synthetic margin):

| Case | Metric Window (SMA) | Min Valid Points | Max \|ΔI\| Threshold | Median \|ΔI\| Threshold |
|---|---|---:|---:|---:|
| `sersic_n4_noiseless` | `[10.0, 160.0]` | 19 | 0.002098 | 0.000096 |
| `sersic_n1_high_eps_noise` | `[12.5, 125.0]` | 15 | 0.114202 | 0.003759 |
| `sersic_n4_extreme_eps_noise` | `[10.0, 100.0]` | 15 | 0.009876 | 0.002971 |

## Artifact Contract
Every non-trivial validation/benchmark run should produce:
1. machine-readable metrics (`.json`)
2. QA figure(s) (`.png` or `.pdf`)
3. run metadata (`git SHA`, environment versions, input parameters)

Recommended output layout:
- `outputs/tests_integration/<run_name>/...`
- `outputs/tests_validation/<run_name>/...`
- `outputs/tests_real_data/<run_name>/...`
- `outputs/benchmarks_performance/<run_name>/...`
- `outputs/benchmarks_profiling/<run_name>/...`

## Execution Phases

### Phase 0: Baseline Measurement and Threshold Locking
- Collect baseline metrics from current suites.
- Confirm metric reproducibility and valid-point counts.
- Propose threshold values from measured distributions.

### Phase 1: Test Hardening and API Coverage
- Remove false-pass patterns.
- Add missing API and CLI tests.
- Rename/normalize M51 basic test to `m51_test`.

### Phase 2: Benchmark and Profiling Output Upgrades
- Standardize outputs for all benchmarks.
- Add default JSON + figure + metadata outputs.
- Save profiler data files and parsed summaries.

### Phase 3: High-Fidelity Mock Pipeline Integration
- Add reproducible bridge to `mockgal.py` workflows.
- Add PSF/noise-aware benchmark scenarios with paired noiseless truth validation.

## Risks and Mitigations
- Risk: threshold instability across machines.
  - Mitigation: lock thresholds using robust statistics and include environment metadata.
- Risk: external mock generator path dependency.
  - Mitigation: make integration optional with clear skip reason and documented setup.
- Risk: artifact volume growth.
  - Mitigation: keep default quick mode small; retain full outputs for scheduled/deep runs.

## Deliverables
- Updated tests with strict meaningful assertions.
- Expanded API/CLI test coverage.
- Renamed and stabilized M51 baseline real-data test (`m51_test`).
- Unified benchmark/profiling outputs under `outputs/`.
- Quantitative criteria documented and enforced.
- Future-ready mock generation workflow design using libprofit-based `mockgal.py`.

## Post-Phase4 Follow-up (2026-02-12)

- Numba diagnostics were upgraded in `benchmarks/performance/bench_numba_speedup.py`:
  - configurable `--n-runs`
  - configurable `--scale-factor`
  - first-run vs steady-state speedup diagnostics
  - per-case variability and slowdown flags
- `mockgal` adapter gained science-ready presets with override support:
  - `single_noiseless_truth`
  - `single_psf_noise_sblimit`
  - `models_config_batch`
- Preset discovery and execution workflows are now explicit in `benchmarks/README.md`.
- Added case-specific drill workflow for flagged scenarios in `benchmarks/profiling/profile_numba_flagged_cases.py`:
  - consumes benchmark JSON results from scale1/scale2 runs
  - auto-targets slowdown/high-variability cases
  - prioritizes `n1_medium_eps07__scale2`
  - persists per-case `.prof` + timing diagnostics and aggregate summary JSON
- Added reusable `models_config_batch` template files:
  - `examples/mockgal/models_config_batch/galaxies.yaml`
  - `examples/mockgal/models_config_batch/image_config.yaml`
