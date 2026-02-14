# Active Plan and Review

## Completed Summary

Phases 1-6 are complete and archived, including docs reorganization, uv workflow adoption, test/benchmark hardening, numba diagnostics, and Huang2013 workflow stabilization. The active focus is now Phase 7 documentation integrity and evidence-based code review follow-through. Detailed completed history has been moved out of this file to keep this checklist operational. Future updates should keep this file short, actionable, and tied to verifiable artifacts.

## History Archive

- Detailed Phase 1-6 progress summary: `docs/journal/2026-02-13-phase1-6-progress-summary.md`
- Phase 7 kickoff plan note: `docs/journal/2026-02-13-phase7-doc-audit-and-code-review-plan.md`

## Phase 7 Checklist

| Item | Status | Notes |
|---|---|---|
| 1. Build claim-vs-code matrix for algorithm/spec/stop-code docs | [x] | verified against `driver/fitting/sampling/config/model/cog` |
| 2. Update `docs/spec.md` and `docs/algorithm.md` for correctness | [x] | removed stale claims and aligned implementation behavior |
| 3. Consolidate stop-code docs into canonical location | [x] | canonical location set to `docs/user-guide.md` |
| 4. Refactor `docs/todo.md` to active-focused and archive completed history | [x] | summary header + archive link added |
| 5. Maintain `docs/future.md` as long-term roadmap | [x] | trimmed to realistic long-term items and removed implemented work |
| 6. Run careful code review and record findings | [x] | short-term findings below; long-term in `docs/future.md` |
| 7. Run final link/reference sweep | [x] | active docs/navigation updated; no stale markdown links remain |

## Short-Term Findings (Actionable)

### P1. Central regularization is effectively inactive in regular `fit_image` flow

Evidence:

- Regularization returns `0.0` if `previous_geom is None` in `isoster/fitting.py:32`.
- Driver calls to `fit_isophote` do not pass `previous_geometry` in `isoster/driver.py:126`, `isoster/driver.py:147`, `isoster/driver.py:175`.

Action:

- Pass prior accepted geometry into `fit_isophote(..., previous_geometry=...)` in outward/inward loops and validate with a targeted regression test.

### P1. Stop code `2` is documented but not emitted by current core fitter

Evidence:

- `fit_isophote` emits only `0`, `1`, `3`, `-1` in `isoster/fitting.py:651`, `isoster/fitting.py:664`, `isoster/fitting.py:705`, `isoster/fitting.py:711`, `isoster/fitting.py:757`.
- `extract_forced_photometry` emits `0` or `3` in `isoster/fitting.py:101`, `isoster/fitting.py:125`.
- `fit_central_pixel` emits `0` or `-1` in `isoster/driver.py:35`.

Action:

- Decide whether to reintroduce explicit code `2` emission in `isoster` or remove it from compatibility descriptions in code/docs.

### P1. Inward pass can start from a failed first isophote

Evidence:

- Inward loop starts when `minsma < sma0` without checking first-isophote quality in `isoster/driver.py:157`.
- It seeds with `current_iso = first_iso` in `isoster/driver.py:158`.

Action:

- Gate inward growth on acceptable initial stop codes, mirroring outward pass quality gating.

### P2. `linear_growth=True` gradient normalization likely uses inconsistent radius delta

Evidence:

- Linear mode sets `gradient_sma = sma + step` in `isoster/fitting.py:464`.
- Gradient still divides by `sma * step` in `isoster/fitting.py:480` and `isoster/fitting.py:485`.

Action:

- Benchmark and verify intended finite-difference definition for linear mode, then align denominator and tests.

### P2. Model reconstruction currently trusts all `sma > 0` rows

Evidence:

- `build_isoster_model` keeps rows by `iso['sma'] > 0` only in `isoster/model.py:80`.
- Interpolation then consumes raw `intens` values in `isoster/model.py:104` and `isoster/model.py:116`.

Action:

- Add optional quality filter (`stop_code` and finite-intensity checks) before interpolation.

## Review Notes

- Long-term upgrades and deferred research items are tracked in `docs/future.md`.
- Stop-code canonical user docs now live in `docs/user-guide.md`.

## Phase 8 Kickoff (Code Quality Improvement)

- Closeout summary and clean-window starter prompt: `docs/journal/2026-02-14-phase7-closeout-and-phase8-kickoff.md`

## Phase 8 Checklist

| Item | Status | Notes |
|---|---|---|
| 1. Wire `previous_geometry` through regular outward/inward `fit_image` calls | [x] | `isoster/driver.py` now passes `previous_geometry=current_iso` for both growth directions. |
| 2. Prevent inward growth when first isophote is not acceptable | [x] | Inward startup now gates on same acceptable stop codes as outward startup in `isoster/driver.py`. |
| 3. Resolve stop-code `2` policy mismatch | [x] | Active code/docs now use emitted stop-code set (`0`, `1`, `3`, `-1`); compatibility acceptance of `2` removed from growth gating and active docs. |
| 4. Add/adjust targeted tests and run focused verification + collect-only | [x] | Added driver-flow regressions in `tests/unit/test_driver.py`; updated API stop-code assertion in `tests/unit/test_public_api.py`; verification commands/results recorded below. |
| 5. Update `docs/spec.md`, `docs/algorithm.md`, `docs/user-guide.md` only where behavior changed | [x] | Updated all three docs for `previous_geometry` propagation, inward gating, and stop-code policy. |
| 6. Update `docs/todo.md` progress and review notes with concrete evidence | [x] | This checklist + review section captures touched files and test evidence. |

## Phase 8 Review Notes

### Code Changes

- `isoster/driver.py`: introduced shared acceptable stop-code gate (`0`, `1`), wired `previous_geometry` in outward/inward `fit_isophote` calls, and gated inward startup on first-isophote acceptability.
- `isoster/config.py`: removed stop-code `2` from active config docstring stop-code list.
- `tests/unit/test_driver.py`: added targeted regression tests for previous-geometry propagation, inward-start gating, and non-acceptance of stop-code `2`.
- `tests/unit/test_public_api.py`: updated public stop-code assertion to active emitted set.

### Behavior Docs Updated

- `docs/spec.md`: removed stop-code `2` compatibility statement; added current regularization/inward-gating behavior notes.
- `docs/algorithm.md`: updated pipeline and caveat notes for inward gating and active regularization wiring; removed stop-code `2` compatibility note.
- `docs/user-guide.md`: removed stop-code `2` row from canonical table and documented active growth acceptability policy.

### Test Evidence

- Focused regression run:
  - Command: `uv run pytest tests/unit/test_driver.py tests/unit/test_public_api.py -q`
  - Result: `21 passed in 2.33s`
- Collection sanity check:
  - Command: `uv run pytest --collect-only -q`
  - Result: `110/114 tests collected (4 deselected) in 1.12s`

## Phase 9 Plan (Benchmark + mockgal QA)

| Item | Status | Notes |
|---|---|---|
| 1. Persist benchmark constraints from latest review | [x] | Added to `CLAUDE.md` (force `mockgal.py` libprofit backend; 2-D metric caveat as system-level diagnostic). |
| 2. Rename legacy output folder for EA harmonics artifacts | [x] | Renamed `outputs/figures` -> `outputs/examples_ea_harmonics`. |
| 3. Update benchmark strategy doc with reviewed gaps and next workstreams | [x] | Added `Session Update (2026-02-14)` section in `docs/test-benchmark-improvement-plan.md`. |
| 4. Enforce `--engine libprofit` in `mockgal_adapter` presets and preflight checks | [x] | Updated `benchmarks/baselines/mockgal_adapter.py` with forced `libprofit`, strict `--engine` validation, preflight `profit-cli` resolution (including `/Users/mac/Dropbox/work/project/otters/isophote_test/libprofit/build`), and metadata provenance fields. |
| 5. Unify Sersic truth generation to accurate `b_n` across benchmark/test generators | [x] | Replaced remaining approximation usage in benchmark/test generators with exact helper-backed `compute_bn()` (`gammaincinv`). |
| 6. Add baseline-locked benchmark gate for efficiency + quantitative QA metrics | [x] | Added lock/gate workflow via `benchmarks/baselines/lock_efficiency_thresholds.py` and `benchmarks/baselines/run_benchmark_gate.py`; gate reports efficiency + 1-D locked metrics and 2-D system-level diagnostics with explicit caveat. |

## Phase 9 Review Notes

### Item 4 (mockgal libprofit enforcement)

- Preset defaults now force `engine=libprofit` for:
  - `single_noiseless_truth`
  - `single_psf_noise_sblimit`
  - `models_config_batch`
- Adapter now rejects any non-libprofit override from `--preset-value engine=...` or pass-through `--engine ...`.
- Added explicit libprofit preflight (`profit-cli` must resolve) and persisted metadata fields:
  - `engine`
  - `profit_cli_path`
  - `preflight_passed`
  - `preflight_message`
  - `mockgal_git_sha`

### Item 5 (exact `b_n` helper migration)

- Updated shared test fixture helper to exact `gammaincinv` implementation:
  - `tests/fixtures/sersic_factory.py`
- Removed remaining benchmark/test approximation-based `b_n` formulas by switching to helper imports in:
  - `benchmarks/performance/bench_efficiency.py`
  - `benchmarks/performance/bench_numba_speedup.py`
  - `benchmarks/profiling/profile_hotpaths.py`
  - `benchmarks/profiling/profile_numba_flagged_cases.py`
  - `tests/integration/test_sersic_accuracy.py`
  - `tests/integration/test_numba_validation.py`
  - `tests/validation/test_model_residuals.py`
  - `tests/validation/test_photutils_comparison.py`

### Item 6 (baseline-locked gate + artifacts)

- Added new scripts:
  - `benchmarks/baselines/lock_efficiency_thresholds.py`
  - `benchmarks/baselines/run_benchmark_gate.py`
- Added lock files:
  - `benchmarks/baselines/efficiency_thresholds_2026-02-14.json`
  - `benchmarks/baselines/efficiency_thresholds_quick_2026-02-14.json`
- Gate output includes:
  - efficiency locked checks
  - 1-D profile locked checks (primary QA metric)
  - 2-D residual quantitative system-level diagnostics with caveat carried in JSON/console output

### Verification Commands and Evidence

- `ruff check benchmarks/baselines/mockgal_adapter.py benchmarks/baselines/lock_efficiency_thresholds.py benchmarks/baselines/run_benchmark_gate.py benchmarks/performance/bench_efficiency.py benchmarks/performance/bench_numba_speedup.py benchmarks/profiling/profile_hotpaths.py benchmarks/profiling/profile_numba_flagged_cases.py tests/fixtures/sersic_factory.py tests/integration/test_sersic_accuracy.py tests/integration/test_numba_validation.py tests/validation/test_model_residuals.py tests/validation/test_photutils_comparison.py`
  - Result: `All checks passed!`
- `.venv/bin/python benchmarks/performance/bench_efficiency.py --quick --n-runs 1`
  - Result: completed, artifact paths below.
- `.venv/bin/python benchmarks/performance/bench_numba_speedup.py --n-runs 2 --scale-factor 1.0`
  - Result: completed, validation passed for all cases (`✓`), mean speedup `1.06x`, steady-state mean speedup `1.10x`.
- `.venv/bin/python benchmarks/performance/bench_vs_photutils.py --quick`
  - Result: `2/2` cases passed, mean speedup `20.0x`.
- `.venv/bin/python benchmarks/baselines/lock_efficiency_thresholds.py`
  - Result: wrote `benchmarks/baselines/efficiency_thresholds_2026-02-14.json`.
- `.venv/bin/python benchmarks/baselines/lock_efficiency_thresholds.py --input outputs/benchmarks_performance/bench_efficiency/efficiency_benchmark_results.json --output benchmarks/baselines/efficiency_thresholds_quick_2026-02-14.json`
  - Result: wrote quick-mode lock file from measured quick baseline.
- `.venv/bin/python benchmarks/baselines/run_benchmark_gate.py --quick --n-runs 1 --efficiency-lock benchmarks/baselines/efficiency_thresholds_quick_2026-02-14.json`
  - Result: `Gate pass: True`; 2-D caveat printed and persisted.
- `.venv/bin/python benchmarks/baselines/mockgal_adapter.py --preset single_noiseless_truth --dry-run --output outputs/benchmarks_performance/mockgal_adapter_dry_run`
  - Result: `status: dry_run`, preflight passed, command prepared.
- `.venv/bin/python benchmarks/baselines/mockgal_adapter.py --preset single_noiseless_truth --dry-run --mockgal-args "--engine astropy" --output outputs/benchmarks_performance/mockgal_adapter_invalid_engine`
  - Result: `failed_validation` with explicit non-libprofit rejection.

### Key Output Paths

- `outputs/benchmarks_performance/bench_efficiency/efficiency_benchmark_results.json`
- `outputs/benchmarks_performance/bench_efficiency/efficiency_benchmark_results.csv`
- `outputs/benchmarks_performance/bench_efficiency/efficiency_benchmark_summary.png`
- `outputs/benchmarks_performance/bench_numba_speedup/numba_benchmark_results.json`
- `outputs/benchmarks_performance/bench_numba_speedup/numba_benchmark_results.csv`
- `outputs/benchmarks_performance/bench_numba_speedup/numba_speedup.png`
- `outputs/benchmarks_performance/bench_vs_photutils/benchmark_results.json`
- `outputs/benchmarks_performance/bench_vs_photutils/benchmark_results.csv`
- `outputs/benchmarks_performance/benchmark_gate/benchmark_gate_report.json`
- `outputs/benchmarks_performance/benchmark_gate/benchmark_gate_efficiency.csv`
- `outputs/benchmarks_performance/benchmark_gate/benchmark_gate_profile_1d.csv`
- `outputs/benchmarks_performance/benchmark_gate/benchmark_gate_profile_2d_system.csv`
- `outputs/benchmarks_performance/mockgal_adapter_dry_run/mockgal_adapter_run.json`
- `outputs/benchmarks_performance/mockgal_adapter_invalid_engine/mockgal_adapter_run.json`
