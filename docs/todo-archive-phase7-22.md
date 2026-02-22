# Archived Phase 7-22 Checklists and Review Notes

Archived from `docs/todo.md` on 2026-02-22 to keep the active todo file concise.
Also includes Phases 13-15 and 22 which were at the top of the original file.

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

## Phase 10 Plan (QA Figure Integration + Next-Batch Readiness)

| Item | Status | Notes |
|---|---|---|
| 1. Integrate updated QA figure script into benchmark gate pipeline | [x] | Hook into `benchmarks/baselines/run_benchmark_gate.py` so each Phase-4 case emits QA figures in the same run. |
| 2. Persist QA artifact paths in gate JSON | [x] | Add per-case `qa_figure_path` (and optional extras) into `benchmark_gate_report.json`. |
| 3. Keep gating policy explicit (1-D primary, 2-D system-level caveat) | [x] | Maintain current pass/fail semantics: locked efficiency + locked 1-D decide pass; 2-D remains quantitative report with caveat. |
| 4. Run quick smoke and then full (non-quick) lock refresh run | [x] | Quick run first for pipeline validation, then full baseline lock+gate using matching case set before next production batch. |
| 5. Update docs with evidence and exact output paths | [x] | Record commands/results and output paths in `docs/todo.md` review section after full run. |

## Phase 10 Parallel Small Tasks

| Item | Status | Notes |
|---|---|---|
| A. Stabilize benchmark plotting cache defaults for headless runs | [x] | Added writable `XDG_CACHE_HOME` + `MPLCONFIGDIR` defaults (under `outputs/tmp`) before matplotlib imports in benchmark scripts. |
| B. Add command examples for full lock refresh + full gate | [x] | Added reproducible command sequence to benchmarks/README.md (full efficiency/profile lock refresh + full gate). |
| C. Integrate built-in QA generation in benchmark gate | [x] | Replaced scaffold with built-in Huang2013 method-QA generation plus per-case artifact paths in report JSON. |

### Phase 10 Early Evidence (Parallel Tasks A-C)

- `ruff check benchmarks/performance/bench_efficiency.py benchmarks/performance/bench_numba_speedup.py benchmarks/performance/bench_vs_photutils.py`
  - Result: `All checks passed!`
- `.venv/bin/python benchmarks/performance/bench_efficiency.py --quick --n-runs 1`
  - Result: completed with updated cache-env defaults; artifacts refreshed under `outputs/benchmarks_performance/bench_efficiency/`.
- Updated `benchmarks/README.md` with full lock-refresh + full gate command sequence:
  - `uv run python benchmarks/performance/bench_efficiency.py --n-runs 3 --output outputs/benchmarks_performance/bench_efficiency_full_refresh`
  - `uv run python benchmarks/baselines/lock_efficiency_thresholds.py --input outputs/benchmarks_performance/bench_efficiency_full_refresh/efficiency_benchmark_results.json --output benchmarks/baselines/efficiency_thresholds_full_refresh.json`
  - `uv run python benchmarks/baselines/collect_phase4_profile_baseline.py --output outputs/tests_integration/baseline_metrics_full_refresh`
  - `uv run python benchmarks/baselines/lock_phase4_thresholds.py --input outputs/tests_integration/baseline_metrics_full_refresh/phase4_profile_baseline_metrics.json --output benchmarks/baselines/phase4_profile_thresholds_full_refresh.json`
  - `uv run python benchmarks/baselines/run_benchmark_gate.py --n-runs 3 --efficiency-lock benchmarks/baselines/efficiency_thresholds_full_refresh.json --profile-lock benchmarks/baselines/phase4_profile_thresholds_full_refresh.json --require-all-locked-cases`
- QA gate integration smoke:
  - `ruff check benchmarks/baselines/run_benchmark_gate.py`
  - `.venv/bin/python benchmarks/baselines/run_benchmark_gate.py --quick --n-runs 1 --efficiency-lock benchmarks/baselines/efficiency_thresholds_quick_2026-02-14.json`
  - Result: gate still passes; report now includes generated per-case `qa_artifacts` (`qa_figure_path`, `profile_fits_path`) and `qa_generation` metadata under `outputs/benchmarks_performance/benchmark_gate/benchmark_gate_report.json`.

## Clean Context Handoff

- Snapshot file: `docs/journal/2026-02-14-phase10-qa-figure-clean-context-handoff.md`
- Includes:
  - current branch/dirty-state summary,
  - exact resume commands,
  - copy-paste first prompt for next session.

## Phase 10 Task D (IC2597 QA Figure Style Pass)

| Item | Status | Notes |
|---|---|---|
| 1. Rework method/comparison QA left-panel layout to eliminate misalignment/overlap | [x] | Updated to explicit 3x2 left grids (`panel + colorbar` per row) in `examples/huang2013/run_huang2013_real_mock_demo.py`. |
| 2. Enforce LaTeX-style typography and optimize overall aspect ratio | [x] | Added shared `configure_qa_plot_style()` (Computer-Modern serif + LaTeX when available) and reduced figure widths (`14.2`/`14.6`). |
| 3. Improve image/model rendering (viridis + low-SB-friendly scaling) | [x] | Added `make_arcsinh_display()` and applied viridis/arcsinh scaling to original and model panels. |
| 4. Switch right-side 1-D profiles to black defaults for single-method QA | [x] | Method QA now uses monochrome stop-code markers with open/filled differentiation (`plot_profile_by_stop_code(..., monochrome=True)`). |
| 5. Regenerate IC2597 QA artifacts and capture verification evidence | [x] | Rebuilt with afterburner into workspace outputs; visual checks confirm alignment/title/label fixes. |

### Phase 10 Task D Review Notes

- Updated file:
  - `examples/huang2013/run_huang2013_real_mock_demo.py`
- New/updated plotting helpers:
  - `configure_qa_plot_style()`
  - `latex_safe_text()`
  - `make_arcsinh_display()`
- Key behavior updates:
  - Left image/model/residual rows now stay column-aligned because each row has a dedicated colorbar column.
  - Figure title no longer collides with panel content after top margin + figure-size retuning.
  - Image and model displays use viridis and robust arcsinh scaling tuned for low surface-brightness visibility.
  - Single-method 1-D profiles default to black markers/lines with marker-face and marker-shape differentiation.

### Phase 10 Task D Verification Evidence

- `uv run ruff check examples/huang2013/run_huang2013_real_mock_demo.py`
  - Result: `All checks passed!`
- `uv run python examples/huang2013/run_huang2013_qa_afterburner.py --galaxy IC2597 --mock-id 1 --method both --config-tag baseline --output-dir outputs/huang2013_ic2597_qa_style --photutils-profile-fits /Users/mac/work/hsc/huang2013/IC2597/IC2597_mock1_photutils_baseline_profile.fits --isoster-profile-fits /Users/mac/work/hsc/huang2013/IC2597/IC2597_mock1_isoster_baseline_profile.fits --photutils-run-json /Users/mac/work/hsc/huang2013/IC2597/IC2597_mock1_photutils_baseline_run.json --isoster-run-json /Users/mac/work/hsc/huang2013/IC2597/IC2597_mock1_isoster_baseline_run.json`
  - Result: completed; QA manifest and figures regenerated.

### Phase 10 Task D Output Paths

- `outputs/huang2013_ic2597_qa_style/IC2597_mock1_photutils_baseline_qa.png`
- `outputs/huang2013_ic2597_qa_style/IC2597_mock1_isoster_baseline_qa.png`
- `outputs/huang2013_ic2597_qa_style/IC2597_mock1_compare_baseline_qa.png`
- `outputs/huang2013_ic2597_qa_style/IC2597_mock1_qa_manifest.json`

### Phase 10 Task D Follow-up (Template Propagation + Error Bars)

- Updated `build_comparison_qa_figure()` to follow the visual template used in `build_method_qa_figure()` (layout proportions, annotation style, typography scale, monochrome plotting language).
- Enforced shared absolute display scaling between data and model in `build_method_qa_figure()` by deriving arcsinh parameters from data and applying them directly to the model panel.
- Added centroid error bars (`x0_err`, `y0_err`) for `dx`/`dy` profiles and axis-ratio error bars (`ellip_err` or `eps_err`) by default.
- Verification:
  - `uv run ruff check examples/huang2013/run_huang2013_real_mock_demo.py` passed.
  - IC2597 afterburner rerun completed and regenerated QA figures under `outputs/huang2013_ic2597_qa_style/`.

### Clean Context Handoff (v2)

- Snapshot file: `docs/journal/2026-02-14-phase10-clean-context-handoff-v2.md`
- Includes an up-to-date copy-paste first prompt for the next clean-context session.

## Phase 10 Completion Update (QA Gate Integration)

| Item | Status | Notes |
|---|---|---|
| 1. Integrate updated QA figure script into benchmark gate pipeline | [x] | `benchmarks/baselines/run_benchmark_gate.py` now generates per-case method QA figures directly via `build_method_qa_figure()` (Huang2013 basic-QA style). |
| 2. Persist QA artifact paths in gate JSON | [x] | Each `profile_2d_system_level.cases[*]` row now includes `qa_artifacts` with `qa_figure_path` and `profile_fits_path`. |
| 3. Keep gating policy explicit (1-D primary, 2-D system-level caveat) | [x] | Pass/fail remains efficiency + 1-D only; 2-D remains report-only with caveat in console + JSON. |
| 4. Run quick smoke and then full (non-quick) lock refresh run | [x] | Completed quick gate and full lock-refresh/full gate sequence (full gate run currently fails strict timing lock due runtime jitter; details below). |
| 5. Update docs with evidence and exact output paths | [x] | Commands, outcomes, and artifact paths are recorded below and in `docs/journal/2026-02-14-phase10-qa-gate-integration-closeout.md`. |

### Phase 10 Completion Review Notes

#### Code Changes

- `benchmarks/baselines/run_benchmark_gate.py`
  - replaced placeholder QA hook scaffold with built-in QA generation.
  - integrated finalized Huang2013 method-QA renderer (`build_method_qa_figure`) for each Phase-4 case.
  - persisted per-case `qa_artifacts` metadata in gate JSON.
  - retained gate semantics: efficiency lock + 1-D lock decide pass/fail; 2-D is report-only diagnostics.
  - added `--qa-overlay-step` and `--qa-dpi` CLI controls; `--qa-output-subdir` retained.
- `benchmarks/README.md`
  - replaced stale "QA hook lands later" note with active command examples for built-in QA generation.

#### Verification Commands and Results

- `ruff check benchmarks/baselines/run_benchmark_gate.py`
  - Result: `All checks passed!`
- `uv run python benchmarks/baselines/run_benchmark_gate.py --quick --n-runs 1 --efficiency-lock benchmarks/baselines/efficiency_thresholds_quick_2026-02-14.json --output outputs/benchmarks_performance/benchmark_gate_phase10_quick_qa`
  - Result: `Gate pass: True`
  - Result: per-case QA artifacts generated under `outputs/benchmarks_performance/benchmark_gate_phase10_quick_qa/qa_figures/`.
- `uv run python benchmarks/performance/bench_efficiency.py --n-runs 3 --output outputs/benchmarks_performance/bench_efficiency_full_refresh_phase10_qa`
  - Result: completed; full refresh efficiency JSON/CSV/plots generated.
- `uv run python benchmarks/baselines/lock_efficiency_thresholds.py --input outputs/benchmarks_performance/bench_efficiency_full_refresh_phase10_qa/efficiency_benchmark_results.json --output benchmarks/baselines/efficiency_thresholds_full_refresh_phase10_qa.json`
  - Result: completed; new full-refresh efficiency lock file written.
- `uv run python benchmarks/baselines/collect_phase4_profile_baseline.py --output outputs/tests_integration/baseline_metrics_full_refresh_phase10_qa`
  - Result: completed; baseline metrics JSON written.
- `uv run python benchmarks/baselines/lock_phase4_thresholds.py --input outputs/tests_integration/baseline_metrics_full_refresh_phase10_qa/phase4_profile_baseline_metrics.json --output benchmarks/baselines/phase4_profile_thresholds_full_refresh_phase10_qa.json`
  - Result: completed; new full-refresh profile lock file written.
- `uv run python benchmarks/baselines/run_benchmark_gate.py --n-runs 3 --efficiency-lock benchmarks/baselines/efficiency_thresholds_full_refresh_phase10_qa.json --profile-lock benchmarks/baselines/phase4_profile_thresholds_full_refresh_phase10_qa.json --require-all-locked-cases --output outputs/benchmarks_performance/benchmark_gate_full_refresh_phase10_qa`
  - Result: `Gate pass: False`
  - Result: failure is efficiency-only due strict runtime lock jitter (example: `n1_medium_eps07` current `0.056894s` vs threshold `0.056213s`), while 1-D profile lock checks pass.
- `uv run python benchmarks/baselines/run_benchmark_gate.py --n-runs 3 --efficiency-lock benchmarks/baselines/efficiency_thresholds_full_refresh_phase10_qa.json --profile-lock benchmarks/baselines/phase4_profile_thresholds_full_refresh_phase10_qa.json --require-all-locked-cases --output outputs/benchmarks_performance/benchmark_gate_full_refresh_phase10_qa_rerun`
  - Result: `Gate pass: False` (same class of strict timing lock jitter; still produced QA artifacts and full report payload).

#### Key Output Paths (Phase 10 Completion)

- Quick gate (pass):
  - `outputs/benchmarks_performance/benchmark_gate_phase10_quick_qa/benchmark_gate_report.json`
  - `outputs/benchmarks_performance/benchmark_gate_phase10_quick_qa/benchmark_gate_efficiency.csv`
  - `outputs/benchmarks_performance/benchmark_gate_phase10_quick_qa/benchmark_gate_profile_1d.csv`
  - `outputs/benchmarks_performance/benchmark_gate_phase10_quick_qa/benchmark_gate_profile_2d_system.csv`
  - `outputs/benchmarks_performance/benchmark_gate_phase10_quick_qa/qa_figures/sersic_n4_noiseless_isoster_phase4_qa.png`
  - `outputs/benchmarks_performance/benchmark_gate_phase10_quick_qa/qa_figures/sersic_n1_high_eps_noise_isoster_phase4_qa.png`
  - `outputs/benchmarks_performance/benchmark_gate_phase10_quick_qa/qa_figures/sersic_n4_extreme_eps_noise_isoster_phase4_qa.png`
- Full refresh locks:
  - `outputs/benchmarks_performance/bench_efficiency_full_refresh_phase10_qa/efficiency_benchmark_results.json`
  - `benchmarks/baselines/efficiency_thresholds_full_refresh_phase10_qa.json`
  - `outputs/tests_integration/baseline_metrics_full_refresh_phase10_qa/phase4_profile_baseline_metrics.json`
  - `benchmarks/baselines/phase4_profile_thresholds_full_refresh_phase10_qa.json`
- Full gate runs:
  - `outputs/benchmarks_performance/benchmark_gate_full_refresh_phase10_qa/benchmark_gate_report.json`
  - `outputs/benchmarks_performance/benchmark_gate_full_refresh_phase10_qa/qa_figures/sersic_n4_noiseless_isoster_phase4_qa.png`
  - `outputs/benchmarks_performance/benchmark_gate_full_refresh_phase10_qa/qa_figures/sersic_n1_high_eps_noise_isoster_phase4_qa.png`
  - `outputs/benchmarks_performance/benchmark_gate_full_refresh_phase10_qa/qa_figures/sersic_n4_extreme_eps_noise_isoster_phase4_qa.png`
  - `outputs/benchmarks_performance/benchmark_gate_full_refresh_phase10_qa_rerun/benchmark_gate_report.json`

### Phase 10 Post-Fix: Efficiency Jitter Policy (Gate Robustness)

- Updated `benchmarks/baselines/run_benchmark_gate.py` efficiency evaluation to include measured runtime jitter tolerance:
  - `adjusted_mean_time_threshold = locked_threshold + max(jitter_floor, jitter_sigma * current_std_time)`
  - default CLI policy: `jitter_sigma=3.0`, `jitter_floor=0.002s`
- New CLI flags:
  - `--efficiency-time-jitter-sigma`
  - `--efficiency-time-jitter-floor-seconds`
- Gate report now records jitter policy in `gate_policy` and per-case columns in efficiency rows:
  - `mean_time_threshold_adjusted`
  - `std_time_current`
  - `time_jitter_tolerance_seconds`

#### Post-Fix Verification

- `ruff check benchmarks/baselines/run_benchmark_gate.py`
  - Result: `All checks passed!`
- `uv run python benchmarks/baselines/run_benchmark_gate.py --quick --n-runs 1 --efficiency-lock benchmarks/baselines/efficiency_thresholds_quick_2026-02-14.json --output outputs/benchmarks_performance/benchmark_gate_phase10_quick_qa_jitter`
  - Result: `Gate pass: True`
- `uv run python benchmarks/baselines/run_benchmark_gate.py --n-runs 3 --efficiency-lock benchmarks/baselines/efficiency_thresholds_full_refresh_phase10_qa.json --profile-lock benchmarks/baselines/phase4_profile_thresholds_full_refresh_phase10_qa.json --require-all-locked-cases --output outputs/benchmarks_performance/benchmark_gate_full_refresh_phase10_qa_jitter_seq`
  - Result: `Gate pass: True`

#### Post-Fix Output Paths

- `outputs/benchmarks_performance/benchmark_gate_phase10_quick_qa_jitter/benchmark_gate_report.json`
- `outputs/benchmarks_performance/benchmark_gate_full_refresh_phase10_qa_jitter_seq/benchmark_gate_report.json`
- `outputs/benchmarks_performance/benchmark_gate_full_refresh_phase10_qa_jitter_seq/benchmark_gate_efficiency.csv`

### Phase 10 Post-Fix 2: Gate Defaults File (No-Flag Workflow)

- Added repository defaults file: `benchmarks/baselines/benchmark_gate_defaults.json`.
- `run_benchmark_gate.py` now loads defaults from `--gate-defaults` (default path above) for:
  - `efficiency_time_jitter_sigma`
  - `efficiency_time_jitter_floor_seconds`
  - `qa_overlay_step`
  - `qa_dpi`
  - `qa_output_subdir`
- CLI overrides still take priority over defaults-file values.

#### Post-Fix 2 Verification

- `ruff check benchmarks/baselines/run_benchmark_gate.py`
  - Result: `All checks passed!`
- `uv run python benchmarks/baselines/run_benchmark_gate.py --quick --n-runs 1 --efficiency-lock benchmarks/baselines/efficiency_thresholds_quick_2026-02-14.json --output outputs/benchmarks_performance/benchmark_gate_phase10_quick_qa_config_defaults`
  - Result: `Gate pass: True` (using defaults-file policy, no jitter flags passed).
- `uv run python benchmarks/baselines/run_benchmark_gate.py --n-runs 3 --efficiency-lock benchmarks/baselines/efficiency_thresholds_full_refresh_phase10_qa.json --profile-lock benchmarks/baselines/phase4_profile_thresholds_full_refresh_phase10_qa.json --require-all-locked-cases --output outputs/benchmarks_performance/benchmark_gate_full_refresh_phase10_qa_config_defaults_seq`
  - Result: `Gate pass: True` (sequential full run, defaults-file policy).

## Phase 11 Plan (Huang2013 Campaign Fault Tolerance)

| Item | Status | Notes |
|---|---|---|
| 1. Make extraction stage resilient to per-method failures | [x] | `run_huang2013_profile_extraction.py` now catches per-method errors, continues, and records `status` per method in `*_profiles_manifest.json`. |
| 2. Make QA afterburner resilient to missing/invalid method artifacts | [x] | `run_huang2013_qa_afterburner.py` now skips missing profile FITS and records `method_skips` / `method_failures` in QA manifest. |
| 3. Add all-galaxy campaign runner with aggregate summary | [x] | Added `examples/huang2013/run_huang2013_campaign.py` with end-of-run JSON/Markdown summary and per-method fail counts. |
| 4. Document full-sample run command | [x] | Added full-campaign section and command in `examples/huang2013/README.md`. |
| 5. Run lint/smoke verification for changed scripts | [x] | `uv run ruff check` on 3 scripts, `--help` on all 3 CLIs, and campaign smoke with missing-input root completed. |

### Phase 11 Review Notes

- Campaign-level summary outputs:
  - `huang2013_campaign_summary.json`
  - `huang2013_campaign_summary.md`
- Summary includes:
  - input-missing case count,
  - extraction/QA invocation failures,
  - method-level success/failure/unknown counters,
  - per-case manifest/log paths.

## Phase 12 Plan (Huang2013 Campaign Observability + Resume)

| Item | Status | Notes |
|---|---|---|
| 1. QA should skip method QA/comparison when extraction status is failed | [x] | Afterburner now reads extraction manifest status and skips methods with non-success status. |
| 2. Add campaign `--verbose` and per-method start/end/error telemetry | [x] | Campaign and extraction/QA scripts now emit explicit START/END/SKIP logs per stage/method. |
| 3. Add campaign `--save-log` with method-specific log files | [x] | Campaign writes JSON logs to `<PREFIX>_photutils.log`, `<PREFIX>_isoster.log`, and `<PREFIX>_qa.log`. |
| 4. Add campaign timeout option (`--max-runtime-seconds`, default 900) | [x] | Timeout now marks stage as `timeout`, records logs, and continues to next stage/case. |
| 5. Add campaign resume (`--continue-from`) | [x] | Added `--continue-from` and `--continue-from-case` (inclusive resume). |
| 6. Add update semantics (`--update`) for batch + single-image extraction | [x] | Campaign and extraction now skip reusable outputs by default and rerun only with `--update`. |
| 7. Update docs and run smoke verification | [x] | README updated; ruff + py_compile + CLI help checks + timeout/resume/update smoke completed. |

### Phase 12 Review Notes

- Implemented QA extraction-status handshake:
  - `run_huang2013_qa_afterburner.py` skips methods with extraction status not equal to `success` and therefore avoids invalid method/comparison QA generation.
- Implemented campaign observability controls:
  - `--verbose`, `--save-log`, `--max-runtime-seconds`, `--continue-from`, `--continue-from-case`, `--update`.
- Implemented default skip + explicit update rerun:
  - campaign skips reusable method/QA artifacts unless `--update`.
  - single-image extraction skips reusable method artifacts unless `--update`.
- Timeout robustness fix:
  - normalized timeout stdout/stderr payloads to text before JSON log serialization.

#### Phase 12 Verification Commands

- `uv run ruff check examples/huang2013/run_huang2013_profile_extraction.py examples/huang2013/run_huang2013_qa_afterburner.py examples/huang2013/run_huang2013_campaign.py`
- `uv run python -m py_compile examples/huang2013/run_huang2013_profile_extraction.py examples/huang2013/run_huang2013_qa_afterburner.py examples/huang2013/run_huang2013_campaign.py`
- `uv run python examples/huang2013/run_huang2013_campaign.py --help`
- `uv run python examples/huang2013/run_huang2013_profile_extraction.py --help`
- `uv run python examples/huang2013/run_huang2013_qa_afterburner.py --help`
- Dry-run resume test:
  - `python examples/huang2013/run_huang2013_campaign.py --huang-root outputs/tmp_phase12 --galaxies GA GB --mock-ids 1 --continue-from GB --dry-run --summary-dir outputs/tmp_phase12/summary_dry`
- Timeout smoke test:
  - `python examples/huang2013/run_huang2013_campaign.py --huang-root outputs/tmp_phase12 --galaxies GA --mock-ids 1 --method both --max-runtime-seconds 1 --save-log --verbose --summary-dir outputs/tmp_phase12/summary_timeout`
- Update semantics smoke test:
  - first run: `python examples/huang2013/run_huang2013_campaign.py --huang-root outputs/tmp_phase12 --galaxies GB --mock-ids 1 --method isoster --summary-dir outputs/tmp_phase12/summary_update_first`
  - second run (skip existing): `python examples/huang2013/run_huang2013_campaign.py --huang-root outputs/tmp_phase12 --galaxies GB --mock-ids 1 --method isoster --summary-dir outputs/tmp_phase12/summary_update_second`
  - third run (`--update`): `python examples/huang2013/run_huang2013_campaign.py --huang-root outputs/tmp_phase12 --galaxies GB --mock-ids 1 --method isoster --update --summary-dir outputs/tmp_phase12/summary_update_third`

## Clean Context Handoff (Phase 12)

- Snapshot handoff file:
  - `docs/journal/2026-02-15-phase12-clean-context-handoff.md`
- Includes branch/commit state, completed scope, verification evidence, and exact resume commands.

## Phase 13 Plan (Huang2013 Retry Policy Alignment)

| Item | Status | Notes |
|---|---|---|
| 1. Apply unified 5-attempt retry ladder to both photutils and isoster extraction | [x] | Implemented in `run_huang2013_profile_extraction.py`: `sma0 += 2.0` and `astep += 0.02` per attempt; `maxsma` unchanged. |
| 2. Persist retry metadata and attempts-used in run payloads | [x] | Success/failure run JSON now includes `attempt_count`, `max_attempts`, and expanded `fit_retry_log` attempt metadata. |
| 3. Add unit tests with mocked failed attempts for both methods | [x] | Added retry progression and max-attempt exhaustion tests in `tests/unit/test_huang2013_campaign_fault_tolerance.py`. |
| 4. Run targeted Huang2013 fault-tolerance tests | [x] | `uv run pytest tests/unit/test_huang2013_campaign_fault_tolerance.py -q` -> `11 passed in 1.34s`. |
| 5. Run requested ESO185-G054 campaign verification command | [x] | Completed with summary at `outputs/huang2013_campaign_eso185_g054/huang2013_campaign_summary.json`. |

### Phase 13 Review Notes

- Extraction retry behavior is now consistent for both methods and retries on:
  - fit exceptions,
  - negative-error validation failures,
  - insufficient profile table (`isophote_count < 3`).
- `fit_retry_log` entries now include per-attempt config values (`sma0`, `astep`, `maxsma`) and failure reasons.
- Campaign verification outcome for ESO185-G054:
  - photutils: `success=3`, `failed=1` (mock1 exhausted all 5 attempts with `ValueError: cannot convert float NaN to integer`)
  - isoster: `success=4`, `failed=0`

#### Phase 13 Verification Commands

- `uv run ruff check examples/huang2013/run_huang2013_profile_extraction.py tests/unit/test_huang2013_campaign_fault_tolerance.py`
- `uv run pytest tests/unit/test_huang2013_campaign_fault_tolerance.py -q`
- `uv run python examples/huang2013/run_huang2013_campaign.py --huang-root /Users/mac/work/hsc/huang2013 --galaxies ESO185-G054 --mock-ids 1 2 3 4 --method both --config-tag baseline --update --verbose --save-log --max-runtime-seconds 900 --summary-dir outputs/huang2013_campaign_eso185_g054`

## Phase 14 Plan (QA Figure Readability Tweaks)

| Item | Status | Notes |
|---|---|---|
| 1. Remove overlapping comparison-panel title | [x] | Removed right-panel title for relative SB difference in comparison QA figure. |
| 2. Fix SB y-limits to use finite valid profile values only | [x] | Method/comparison SB ranges now use finite SB values (ignore NaN/Inf and error bars), with margin. |
| 3. Improve non-zero stop-code visual separation in method QA | [x] | Added dark per-stop colors for monochrome mode and enforced open markers for `stop != 0`. |
| 4. Adjust comparison centroid legend placement/readability | [x] | Legend now uses `loc=\"best\"` and slightly larger fontsize. |
| 5. Replace fixed axis-ratio y-limits with data-driven limits | [x] | Method/comparison axis-ratio panels now use finite data range + margin, clipped to `[0, 1]`. |
| 6. Regenerate ESO185-G054 QA without rerunning extraction | [x] | Rebuilt QA via `run_huang2013_qa_afterburner.py` for mock1-4 only. |

### Phase 14 Review Notes

- Updated file:
  - `examples/huang2013/run_huang2013_real_mock_demo.py`
- Auxiliary robustness fix:
  - `examples/huang2013/run_huang2013_qa_afterburner.py` now tolerates non-dict `runtime` payloads in failed run JSON.
- QA regeneration explicitly used afterburner only; extraction was not invoked.
- Sandbox required escalated permissions to write regenerated artifacts under `/Users/mac/work/hsc/huang2013/...`.

#### Phase 14 Verification Commands

- `uv run ruff check examples/huang2013/run_huang2013_real_mock_demo.py examples/huang2013/run_huang2013_qa_afterburner.py`
- `.venv/bin/python examples/huang2013/run_huang2013_qa_afterburner.py --huang-root /Users/mac/work/hsc/huang2013 --galaxy ESO185-G054 --mock-id 1 --method both --config-tag baseline --output-dir /Users/mac/work/hsc/huang2013/ESO185-G054 --profiles-manifest /Users/mac/work/hsc/huang2013/ESO185-G054/ESO185-G054_mock1_profiles_manifest.json --verbose`
- `.venv/bin/python examples/huang2013/run_huang2013_qa_afterburner.py --huang-root /Users/mac/work/hsc/huang2013 --galaxy ESO185-G054 --mock-id 2 --method both --config-tag baseline --output-dir /Users/mac/work/hsc/huang2013/ESO185-G054 --profiles-manifest /Users/mac/work/hsc/huang2013/ESO185-G054/ESO185-G054_mock2_profiles_manifest.json --verbose`
- `.venv/bin/python examples/huang2013/run_huang2013_qa_afterburner.py --huang-root /Users/mac/work/hsc/huang2013 --galaxy ESO185-G054 --mock-id 3 --method both --config-tag baseline --output-dir /Users/mac/work/hsc/huang2013/ESO185-G054 --profiles-manifest /Users/mac/work/hsc/huang2013/ESO185-G054/ESO185-G054_mock3_profiles_manifest.json --verbose`
- `.venv/bin/python examples/huang2013/run_huang2013_qa_afterburner.py --huang-root /Users/mac/work/hsc/huang2013 --galaxy ESO185-G054 --mock-id 4 --method both --config-tag baseline --output-dir /Users/mac/work/hsc/huang2013/ESO185-G054 --profiles-manifest /Users/mac/work/hsc/huang2013/ESO185-G054/ESO185-G054_mock4_profiles_manifest.json --verbose`
- `.venv/bin/python examples/huang2013/run_huang2013_qa_afterburner.py --huang-root /Users/mac/work/hsc/huang2013 --galaxy ESO185-G054 --mock-id 1 --method both --config-tag baseline --output-dir /Users/mac/work/hsc/huang2013/ESO185-G054 --profiles-manifest /Users/mac/work/hsc/huang2013/ESO185-G054/ESO185-G054_mock1_profiles_manifest.json --ignore-extraction-status --verbose`

## Phase 14 Follow-Up Plan (Outlier-Limit + Legend Order)

| Item | Status | Notes |
|---|---|---|
| 1. Visually inspect regenerated QA PNGs and identify exact remaining readability issues | [x] | Remaining issues were outlier-dominated y-limits in comparison `dI/I`, centroid, and PA panels, plus non-intuitive stop-code legend ordering. |
| 2. Apply minimal plotting edits in shared QA renderer | [x] | Added robust percentile y-limits for comparison `dI/I`, centroid, PA; sorted stop-code legend with `stop=0` first. |
| 3. Re-run afterburner only for affected mocks | [x] | Regenerated mock2-4 (comparison + method figures) without rerunning extraction. |

### Phase 14 Follow-Up Review Notes

- Updated file:
  - `examples/huang2013/run_huang2013_real_mock_demo.py`
- Rerun scope:
  - `ESO185-G054` mock2-4 only (mock1 unchanged by these specific tweaks).
- Visual result:
  - Comparison panels now keep central trends readable under outer outliers; method legends now list `stop=0` first.

#### Phase 14 Follow-Up Verification Commands

- `uv run ruff check examples/huang2013/run_huang2013_real_mock_demo.py examples/huang2013/run_huang2013_qa_afterburner.py`
- `MPLCONFIGDIR=/tmp/mplconfig uv run python examples/huang2013/run_huang2013_qa_afterburner.py --huang-root /Users/mac/work/hsc/huang2013 --galaxy ESO185-G054 --mock-id 2 --method both --config-tag baseline --output-dir /Users/mac/work/hsc/huang2013/ESO185-G054 --profiles-manifest /Users/mac/work/hsc/huang2013/ESO185-G054/ESO185-G054_mock2_profiles_manifest.json --verbose`
- `MPLCONFIGDIR=/tmp/mplconfig uv run python examples/huang2013/run_huang2013_qa_afterburner.py --huang-root /Users/mac/work/hsc/huang2013 --galaxy ESO185-G054 --mock-id 3 --method both --config-tag baseline --output-dir /Users/mac/work/hsc/huang2013/ESO185-G054 --profiles-manifest /Users/mac/work/hsc/huang2013/ESO185-G054/ESO185-G054_mock3_profiles_manifest.json --verbose`
- `MPLCONFIGDIR=/tmp/mplconfig uv run python examples/huang2013/run_huang2013_qa_afterburner.py --huang-root /Users/mac/work/hsc/huang2013 --galaxy ESO185-G054 --mock-id 4 --method both --config-tag baseline --output-dir /Users/mac/work/hsc/huang2013/ESO185-G054 --profiles-manifest /Users/mac/work/hsc/huang2013/ESO185-G054/ESO185-G054_mock4_profiles_manifest.json --verbose`

## Phase 14 Follow-Up 2 Plan (X-Range Margin + Stop 4/5 Contrast)

| Item | Status | Notes |
|---|---|---|
| 1. Add right-edge x-axis margin for shared 1-D profile panels | [x] | Added helper and applied to both method/comparison QA shared x-limits. |
| 2. Improve `stop=4/5` style contrast against black/circle defaults | [x] | Added explicit marker/color mappings for stop 4 and 5 in standard and monochrome styles. |
| 3. Re-run full ESO185-G054 QA afterburner | [x] | Regenerated mock1-4 (method + comparison), all stages successful. |

### Phase 14 Follow-Up 2 Review Notes

- Updated file:
  - `examples/huang2013/run_huang2013_real_mock_demo.py`
- Visual checks:
  - right edge now has slight x-margin on all 1-D panels,
  - `stop=4` now uses brown `P`,
  - `stop=5` now uses teal `v`.

#### Phase 14 Follow-Up 2 Verification Commands

- `uv run ruff check examples/huang2013/run_huang2013_real_mock_demo.py examples/huang2013/run_huang2013_qa_afterburner.py`
- `MPLCONFIGDIR=/tmp/mplconfig uv run python examples/huang2013/run_huang2013_qa_afterburner.py --huang-root /Users/mac/work/hsc/huang2013 --galaxy ESO185-G054 --mock-id 1 --method both --config-tag baseline --output-dir /Users/mac/work/hsc/huang2013/ESO185-G054 --profiles-manifest /Users/mac/work/hsc/huang2013/ESO185-G054/ESO185-G054_mock1_profiles_manifest.json --ignore-extraction-status --verbose`
- `MPLCONFIGDIR=/tmp/mplconfig uv run python examples/huang2013/run_huang2013_qa_afterburner.py --huang-root /Users/mac/work/hsc/huang2013 --galaxy ESO185-G054 --mock-id 2 --method both --config-tag baseline --output-dir /Users/mac/work/hsc/huang2013/ESO185-G054 --profiles-manifest /Users/mac/work/hsc/huang2013/ESO185-G054/ESO185-G054_mock2_profiles_manifest.json --ignore-extraction-status --verbose`
- `MPLCONFIGDIR=/tmp/mplconfig uv run python examples/huang2013/run_huang2013_qa_afterburner.py --huang-root /Users/mac/work/hsc/huang2013 --galaxy ESO185-G054 --mock-id 3 --method both --config-tag baseline --output-dir /Users/mac/work/hsc/huang2013/ESO185-G054 --profiles-manifest /Users/mac/work/hsc/huang2013/ESO185-G054/ESO185-G054_mock3_profiles_manifest.json --ignore-extraction-status --verbose`
- `MPLCONFIGDIR=/tmp/mplconfig uv run python examples/huang2013/run_huang2013_qa_afterburner.py --huang-root /Users/mac/work/hsc/huang2013 --galaxy ESO185-G054 --mock-id 4 --method both --config-tag baseline --output-dir /Users/mac/work/hsc/huang2013/ESO185-G054 --profiles-manifest /Users/mac/work/hsc/huang2013/ESO185-G054/ESO185-G054_mock4_profiles_manifest.json --ignore-extraction-status --verbose`

## Phase 15 Plan (Method-Specific 2-D Model Builder)

| Item | Status | Notes |
|---|---|---|
| 1. Audit current photutils model path and options | [x] | Confirmed photutils fit uses `Ellipse.fit_image(...)`, but 2-D model path previously used `isoster.build_isoster_model(...)` for both methods. |
| 2. Switch to method-specific model generation | [x] | Added photutils-native reconstruction via `photutils.isophote.build_ellipse_model` while keeping `isoster.build_isoster_model` for isoster profiles. |
| 3. Regenerate only ESO185-G054 mock3 photutils QA into local outputs | [x] | Created `outputs/huang2013_mock3_photutils_model_pass1/ESO185-G054_mock3_photutils_baseline_qa.png` via afterburner. |

### Phase 15 Verification Commands

- `uv run ruff check examples/huang2013/run_huang2013_real_mock_demo.py examples/huang2013/run_huang2013_qa_afterburner.py`
- `MPLCONFIGDIR=/tmp/mplconfig uv run python examples/huang2013/run_huang2013_qa_afterburner.py --galaxy ESO185-G054 --mock-id 3 --input-fits /Users/mac/work/hsc/huang2013/ESO185-G054/ESO185-G054_mock3.fits --method photutils --config-tag baseline --output-dir outputs/huang2013_mock3_photutils_model_pass1 --photutils-profile-fits /Users/mac/work/hsc/huang2013/ESO185-G054/ESO185-G054_mock3_photutils_baseline_profile.fits --photutils-run-json /Users/mac/work/hsc/huang2013/ESO185-G054/ESO185-G054_mock3_photutils_baseline_run.json --skip-comparison --ignore-extraction-status --verbose`

## Phase 16 Plan (isoster CoG Point Completeness in QA)

| Item | Status | Notes |
|---|---|---|
| 1. Diagnose missing CoG points in ESO185-G054 mock3 isoster QA | [x] | Found `method_cog_flux` tied to `tflux_e`, which has NaNs on 16 rows despite finite SB/intensity; isoster `cog` column is finite for all rows. |
| 2. Ensure CoG plotting uses best available method CoG source with fallback | [x] | Added `harmonize_method_cog_columns()` with source priority: `method_cog_flux -> cog -> tflux_e -> true_cog_flux`; recomputes `cog_rel_diff`. |
| 3. Apply harmonization in both extraction prep and afterburner read path | [x] | Integrated in `prepare_profile_table()` and after `Table.read(...)` in afterburner/main profile-load path. |
| 4. Regenerate only ESO185-G054 mock3 isoster QA in local outputs | [x] | Generated `outputs/huang2013_mock3_isoster_cog_fix/ESO185-G054_mock3_isoster_baseline_qa.png`; CoG finite coverage restored from 42/58 to 58/58. |

### Phase 16 Verification Commands

- `uv run ruff check examples/huang2013/run_huang2013_real_mock_demo.py examples/huang2013/run_huang2013_qa_afterburner.py`
- `MPLCONFIGDIR=/tmp/mplconfig uv run python examples/huang2013/run_huang2013_qa_afterburner.py --galaxy ESO185-G054 --mock-id 3 --input-fits /Users/mac/work/hsc/huang2013/ESO185-G054/ESO185-G054_mock3.fits --method isoster --config-tag baseline --output-dir outputs/huang2013_mock3_isoster_cog_fix --isoster-profile-fits /Users/mac/work/hsc/huang2013/ESO185-G054/ESO185-G054_mock3_isoster_baseline_profile.fits --isoster-run-json /Users/mac/work/hsc/huang2013/ESO185-G054/ESO185-G054_mock3_isoster_baseline_run.json --skip-comparison --ignore-extraction-status --verbose`

## Phase 17 Plan (mock3 CoG NaN Root Cause + Canonical Source Policy)

| Item | Status | Notes |
|---|---|---|
| 1. Trace `tflux_e` NaN generation path for `ESO185-G054_mock3` isoster rows | [x] | Reproduced on current code; affected `stop_code=0` rows have `niter == maxit`, so convergence block is not entered and `full_photometry` is never executed, leaving default `tflux_e=np.nan`. |
| 2. Set canonical QA CoG source policy for isoster tables | [x] | `harmonize_method_cog_columns(..., method_name=\"isoster\")` now prioritizes `cog` before `tflux_e`/fallback sources to keep one-to-one CoG completeness with finite isoster rows. |
| 3. Add regression test for CoG completeness contract | [x] | Added unit test that enforces finite `method_cog_flux` when `cog` is finite and `tflux_e` is sparse/NaN for isoster harmonization. |

### Phase 17 Review Notes

- Updated files:
  - `examples/huang2013/run_huang2013_real_mock_demo.py`
  - `examples/huang2013/run_huang2013_qa_afterburner.py`
  - `tests/unit/test_huang2013_campaign_fault_tolerance.py`
- CoG policy:
  - isoster priority: `cog -> method_cog_flux -> tflux_e -> true_cog_flux`
  - photutils priority: `tflux_e -> method_cog_flux -> cog -> true_cog_flux`

### Phase 17 Verification Commands

- `uv run pytest tests/unit/test_huang2013_campaign_fault_tolerance.py -q`
- `uv run ruff check examples/huang2013/run_huang2013_real_mock_demo.py examples/huang2013/run_huang2013_qa_afterburner.py`

## Phase 18 Plan (Max-Iteration Stop Code Parity)

| Item | Status | Notes |
|---|---|---|
| 1. Emit explicit max-iteration stop code in core fitter | [x] | `fit_isophote` now emits `stop_code=2` when `maxit` is reached without convergence. |
| 2. Keep stop-code policy internally consistent | [x] | Driver acceptable stop codes updated to `{0,1,2}` and API/config/docs/tests aligned with emitted set. |
| 3. Preserve aperture photometry on `stop_code=2` rows | [x] | Full photometry is now attached for max-iteration fallback using best geometry. |
| 4. Add targeted regression for max-iteration labeling | [x] | Added unit test verifying `stop_code=2` and finite `tflux_e` on forced maxit exhaustion. |
| 5. Re-run ESO185-G054 mock3 isoster extraction in `outputs/` | [x] | Generated new artifacts in `outputs/huang2013_mock3_isoster_stopcode2/` with stop-code summary `{-1:15, 0:42, 2:8}` and zero non-finite `tflux_e` among stop codes `0/2`. |

### Phase 18 Verification Commands

- `uv run pytest tests/unit/test_driver.py tests/unit/test_public_api.py tests/unit/test_fitting.py -q`
- `MPLCONFIGDIR=/tmp/mplconfig uv run python examples/huang2013/run_huang2013_real_mock_demo.py --galaxy ESO185-G054 --mock-id 3 --input-fits /Users/mac/work/hsc/huang2013/ESO185-G054/ESO185-G054_mock3.fits --method isoster --config-tag stopcode2-pass --output-dir outputs/huang2013_mock3_isoster_stopcode2 --skip-comparison`

## Phase 19 Plan (Broader Regression + Integration Stop-Code Drift)

| Item | Status | Notes |
|---|---|---|
| 1. Run broad regression command exactly as requested | [x] | `uv run pytest tests/ -q` now passes after adding `tests/__init__.py` for stable intra-test imports. |
| 2. Review integration tests for stop-code drift | [x] | Updated integration assumptions to treat `{0,1,2}` as usable where selection logic required it. |
| 3. Stabilize template-integration tests against fitter convergence variability | [x] | Template tests now build deterministic template isophotes via forced extraction, then validate template-forced behavior. |

### Phase 19 Verification Commands

- `uv run pytest tests/ -q`
- `uv run pytest tests/integration/test_edge_cases.py tests/integration/test_template_forced.py -q`

## Phase 20 Plan (Unify CoG Recipe Across isoster and photutils)

| Item | Status | Notes |
|---|---|---|
| 1. Document exact isoster CoG recipe and apply it consistently | [x] | Added `ensure_isoster_style_cog_columns()` to compute CoG via `isoster.cog.compute_cog` from `{sma, eps, intens, x0, y0}` geometry/intensity columns. |
| 2. Remove photutils-first `tflux_e` preference in QA harmonization | [x] | `harmonize_method_cog_columns()` now prioritizes `cog` for both `isoster` and `photutils`; `tflux_e` is fallback only. |
| 3. Add regression test for photutils CoG harmonization policy | [x] | Added unit test proving photutils `method_cog_flux` is sourced from computed `cog` even when `tflux_e` is present. |

### Phase 20 Verification Commands

- `uv run pytest tests/unit/test_huang2013_campaign_fault_tolerance.py -q`

## Phase 21 Plan (Retry maxsma 5% Decay on Failure)

| Item | Status | Notes |
|---|---|---|
| 1. Change Huang2013 retry config builder to shrink `maxsma` on each failed attempt | [x] | `build_retry_attempt_config()` now applies `maxsma *= 0.95 ** retry_offset` while keeping existing `sma0`/`astep` increments. |
| 2. Update retry-policy regression tests | [x] | Unit expectations now validate decaying `maxsma` sequence instead of fixed `maxsma` across attempts. |
| 3. Reproduce and validate `ESO185-G054_mock1` photutils behavior | [x] | Confirmed default maxsma failure regime and recovery when retries reduce `maxsma`; sweep artifacts saved under `outputs/huang2013_mock1_photutils_maxsma_scan/`. |

### Phase 21 Verification Commands

- `uv run pytest tests/unit/test_huang2013_campaign_fault_tolerance.py -q`
- `MPLCONFIGDIR=/tmp/mplconfig uv run python examples/huang2013/run_huang2013_profile_extraction.py --galaxy ESO185-G054 --mock-id 1 --huang-root /Users/mac/work/hsc/huang2013 --output-dir outputs/huang2013_mock1_photutils_retry_decay --method photutils --config-tag retry-decay --verbose --save-log --update`
## Phase 13 Plan (Huang2013 Campaign Reorganization)

Plan file: `docs/todo.md`

| Item | Status | Notes |
|---|---|---|
| 1. Propose campaign module boundaries and manifest contract updates | [x] | Added to `docs/spec.md` under Huang2013 workflow section. |
| 2. Add phased reorg checklist and checkpoints | [x] | This section is the active execution checklist for the reorg slice. |
| 3. Implement first refactor slice: shared contract helper + campaign/QA wiring | [x] | Added `examples/huang2013/huang2013_campaign_contract.py`; campaign and QA scripts now use canonical helper paths/loaders. |
| 4. Add/adjust targeted regression tests for contract slice | [x] | Added helper-contract tests in `tests/unit/test_huang2013_campaign_fault_tolerance.py`. |
| 5. Run targeted verification commands and record evidence | [x] | Verified with targeted unit test and one-case campaign execution (details below). |

### Phase 13 Review Notes (In Progress)

- Goal: reorganize workflow boundaries while preserving current extraction/QA behavior and manifest compatibility.
- Scope of this slice:
  - no manifest filename changes
  - no schema-breaking field changes
  - no extraction retry/CoG behavior changes
- Verification evidence:
  - `uv run pytest tests/unit/test_huang2013_campaign_fault_tolerance.py -q`
    - Result: `15 passed in 1.26s`
  - `MPLCONFIGDIR=/tmp/mplconfig uv run python examples/huang2013/run_huang2013_campaign.py --huang-root /Users/mac/work/hsc/huang2013 --output-root /Users/mac/work/hsc/huang2013 --galaxies ESO185-G054 --method both --config-tag baseline --limit 1 --verbose --save-log --update`
    - Result: extraction success for both methods and QA success for `ESO185-G054_mock1`.
    - Summary artifacts: `outputs/huang2013_campaign/huang2013_campaign_summary.json`, `outputs/huang2013_campaign/huang2013_campaign_summary.md`

## Phase 14 Plan (Huang2013 Cleanup Utility)

Plan file: `docs/todo.md`

| Item | Status | Notes |
|---|---|---|
| 1. Add cleanup script under `examples/huang2013` with three scopes | [x] | Added `clean_huang2013_outputs.py` for single-test, single-galaxy, and all-galaxies cleanup modes. |
| 2. Preserve mock input FITS and galaxy mosaic PNG exactly as requested | [x] | Preserves `<GALAXY>_<TEST>.fits` and `<GALAXY>_mosaic.png`; removes generated files only. |
| 3. Add dry-run safety and summary output | [x] | Added `--dry-run`, `--verbose`, and per-run summary counters. |
| 4. Document usage in Huang2013 README | [x] | Added `Cleanup Utility` section with command examples. |
| 5. Run synthetic verification (non-destructive to real data) | [x] | Verified all three modes in temp root; preserve rules held and generated artifacts were removed as expected. |

### Phase 14 Review Notes (In Progress)

- Goal: prepare for major campaign runs by quickly clearing stale generated artifacts without touching mock image inputs.
- Verification evidence:
  - `uv run python -m py_compile examples/huang2013/clean_huang2013_outputs.py`
    - Result: success.
  - Synthetic dataset validation under a temporary root:
    - single-test dry-run and apply (`--galaxy ESO185-G054 --test-name mock1`)
    - single-galaxy all-tests apply (`--galaxy ESO185-G054`)
    - all-galaxies apply (`--all-galaxies`)
    - Result: removed generated files while preserving `<GALAXY>_<TEST>.fits` and `<GALAXY>_mosaic.png`.
  - Follow-up compatibility update:
    - cleanup now supports both legacy flat outputs and nested `<GALAXY>/mock<ID>/` outputs, including empty test-directory removal.

## Phase 15 Plan (Per-Test Output Folder Layout)

Plan file: `docs/todo.md`

| Item | Status | Notes |
|---|---|---|
| 1. Change default output layout to `<GALAXY>/<TEST>/` for campaign/extraction/QA scripts | [x] | Updated path resolution to `mock<ID>` subfolders while keeping FITS inputs in `<GALAXY>/`. |
| 2. Ensure per-test output folder creation happens before execution and skips when existing | [x] | Added create-or-skip telemetry in campaign/extraction/QA (`mkdir` stage messages). |
| 3. Keep manifest and artifact naming compatibility inside new folder layout | [x] | Prefix/file names unchanged; only parent output directory changed. |
| 4. Update docs for new default layout and usage expectations | [x] | Updated `docs/spec.md` and `examples/huang2013/README.md`. |
| 5. Run targeted verification + one real ESO185-G054 mock1 isoster run | [x] | Verified with unit tests, real extraction run, and QA afterburner run using default `<GALAXY>/<TEST>/` output path. |

### Phase 15 Review Notes (In Progress)

- Goal: isolate each test run outputs under `<GALAXY>/<TEST>/` to simplify pre-run cleanup and result management.
- Verification evidence:
  - `uv run pytest tests/unit/test_huang2013_campaign_fault_tolerance.py -q`
    - Result: `15 passed in 2.01s`.
  - Real extraction run:
    - `MPLCONFIGDIR=/tmp/mplconfig uv run python examples/huang2013/run_huang2013_profile_extraction.py --huang-root /Users/mac/work/hsc/huang2013 --galaxy ESO185-G054 --mock-id 1 --method isoster --config-tag baseline --verbose --save-log --update`
    - Result: output directory created at `/Users/mac/work/hsc/huang2013/ESO185-G054/mock1` and manifest written there.
  - Real QA run:
    - `MPLCONFIGDIR=/tmp/mplconfig uv run python examples/huang2013/run_huang2013_qa_afterburner.py --huang-root /Users/mac/work/hsc/huang2013 --galaxy ESO185-G054 --mock-id 1 --method isoster --config-tag baseline --verbose --skip-comparison`
    - Result: output directory reuse logged as skip-existing; QA manifest written to `/Users/mac/work/hsc/huang2013/ESO185-G054/mock1/`.
  - Real campaign run (single case smoke):
    - `MPLCONFIGDIR=/tmp/mplconfig uv run python examples/huang2013/run_huang2013_campaign.py --huang-root /Users/mac/work/hsc/huang2013 --output-root /Users/mac/work/hsc/huang2013 --galaxies ESO185-G054 --mock-ids 1 --method isoster --config-tag baseline --limit 1 --verbose --save-log --update`
    - Result: campaign `mkdir` stage logged as skip-existing for `/Users/mac/work/hsc/huang2013/ESO185-G054/mock1`, then extraction + QA succeeded.

## Phase 22 Plan (Huang2013 20-Galaxy Follow-Ups)

Plan file: `docs/todo.md`

| Item | Status | Notes |
|---|---|---|
| 1. Review dirty worktree items and confirm setup branch | [x] | Reviewed tracked `pycache` modifications and untracked journal prompt files; branch `huang2013-small-issues-2026-02-21` created for follow-up work. |
| 2. Extract first-20 run outcome summary and failure cluster | [x] | `80/80` cases processed; `photutils` has `3` timeouts (`IC2006`, `IC4797`, `IC4889`, all `mock1`). |
| 3. Decide worktree hygiene policy for tracked `*.pyc` and session prompt files | [ ] | Pending decision: quick restore-only vs durable untrack policy for pycache; keep/commit vs local-ignore for new journal prompts. |
| 4. Implement selected small workflow fixes from the first-20 run | [x] | Added explicit failed/timeout case aggregation and reporting in campaign summary JSON/Markdown and terminal output (`run_huang2013_campaign.py`). |
| 5. Re-run targeted verification and capture evidence | [x] | Added regression tests and ran targeted lint/test verification (details below). |

### Phase 22 Review Notes (In Progress)

- Current run summary source: `outputs/huang2013_campaign/huang2013_campaign_summary.json`.
- Method counters:
  - `isoster`: `80` success, `0` timeout/failure.
  - `photutils`: `74` success, `3` timeout (counted as failed in method counters).
- QA warning footprint (from per-case QA manifests): `21` `artifact_missing` warnings and `10` `isophote_count_mismatch` warnings.
- Worktree hygiene root cause:
  - `.gitignore` ignores `__pycache__` and `*.pyc`, but multiple `*.pyc` files are already tracked in git history, so runtime regenerates them as modified files.
- First small issue implemented:
  - campaign output now emits direct case labels for:
    - extraction failed cases per method,
    - extraction timeout cases per method,
    - QA failed/timeout cases.
  - same information is persisted in `huang2013_campaign_summary.json` and rendered in `huang2013_campaign_summary.md`.
- Verification evidence:
  - `uv run ruff check examples/huang2013/run_huang2013_campaign.py tests/unit/test_huang2013_campaign_fault_tolerance.py`
    - Result: pass.
  - `uv run pytest tests/unit/test_huang2013_campaign_fault_tolerance.py -q`
    - Result: `19 passed`.
- Second small issue implemented (isoster 2-D model gaps):
  - Added isoster model-input sanitization in `run_huang2013_real_mock_demo.py` before calling `isoster.build_isoster_model(...)`.
  - Filter now removes non-finite required rows (`sma`, `intens`, `eps`, `pa`, `x0`, `y0`) and de-duplicates repeated SMA rows.
  - Added post-build finite guard to replace any residual non-finite model pixels with `0.0` plus runtime warning.
  - Added targeted regressions:
    - `test_extract_isoster_model_rows_filters_invalid_rows`
    - `test_build_model_image_isoster_replaces_nonfinite_with_zero`
  - Re-generated QA (no extraction rerun) for `IC2006_mock1` isoster:
    - `MPLCONFIGDIR=/tmp/mplconfig uv run python examples/huang2013/run_huang2013_qa_afterburner.py --huang-root /Users/mac/work/hsc/huang2013 --galaxy IC2006 --mock-id 1 --method isoster --config-tag baseline --output-dir /Users/mac/work/hsc/huang2013/IC2006/mock1 --verbose`
    - Result: method QA success; updated figure at `/Users/mac/work/hsc/huang2013/IC2006/mock1/IC2006_mock1_isoster_baseline_qa.png`.
- Third small issue implemented (campaign failed-case reporting + log collision):
  - `run_huang2013_campaign.py` now stores per-method `counted_status` and `manifest_status` in case stage payloads.
  - Failed/timeout case lists now use counted status so manifest-level extraction failures are included even when subprocess exit code was success.
  - `--save-log` campaign stage logs now use dedicated JSON filenames to avoid overwrite collision with extraction method logs:
    - `<PREFIX>_<METHOD>_campaign-stage.json`
    - `<PREFIX>_qa_campaign-stage.json`
  - Added regression:
    - `test_collect_problem_cases_honors_counted_status_for_manifest_failures`
  - Verification:
    - `uv run ruff check examples/huang2013/run_huang2013_campaign.py tests/unit/test_huang2013_campaign_fault_tolerance.py examples/huang2013/README.md`
    - `uv run pytest tests/unit/test_huang2013_campaign_fault_tolerance.py -q`
    - Result: `20 passed`.

