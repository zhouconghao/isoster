# Phase 10 Closeout: Benchmark Gate QA Integration (2026-02-14)

## Scope Completed

1. Integrated built-in QA generation into `benchmarks/baselines/run_benchmark_gate.py`.
2. Enforced persisted QA style rule by using Huang2013 basic-QA method figure builder (`build_method_qa_figure`).
3. Stored per-case QA artifact paths in gate JSON (`qa_artifacts.qa_figure_path` + `profile_fits_path`).
4. Preserved gate semantics: efficiency + 1-D lock checks decide pass/fail; 2-D remains report-only diagnostics with caveat.
5. Ran quick smoke and full lock-refresh/full gate verification sequence.

## Files Updated

- `benchmarks/baselines/run_benchmark_gate.py`
- `benchmarks/README.md`
- `docs/todo.md`
- `benchmarks/baselines/efficiency_thresholds_full_refresh_phase10_qa.json`
- `benchmarks/baselines/phase4_profile_thresholds_full_refresh_phase10_qa.json`

## Verification Summary

- Quick gate run:
  - Command: `uv run python benchmarks/baselines/run_benchmark_gate.py --quick --n-runs 1 --efficiency-lock benchmarks/baselines/efficiency_thresholds_quick_2026-02-14.json --output outputs/benchmarks_performance/benchmark_gate_phase10_quick_qa`
  - Result: `Gate pass: True`
  - QA artifacts: generated for all 3 Phase-4 cases.

- Full lock-refresh:
  - `uv run python benchmarks/performance/bench_efficiency.py --n-runs 3 --output outputs/benchmarks_performance/bench_efficiency_full_refresh_phase10_qa`
  - `uv run python benchmarks/baselines/lock_efficiency_thresholds.py --input outputs/benchmarks_performance/bench_efficiency_full_refresh_phase10_qa/efficiency_benchmark_results.json --output benchmarks/baselines/efficiency_thresholds_full_refresh_phase10_qa.json`
  - `uv run python benchmarks/baselines/collect_phase4_profile_baseline.py --output outputs/tests_integration/baseline_metrics_full_refresh_phase10_qa`
  - `uv run python benchmarks/baselines/lock_phase4_thresholds.py --input outputs/tests_integration/baseline_metrics_full_refresh_phase10_qa/phase4_profile_baseline_metrics.json --output benchmarks/baselines/phase4_profile_thresholds_full_refresh_phase10_qa.json`

- Full gate runs:
  - Command: `uv run python benchmarks/baselines/run_benchmark_gate.py --n-runs 3 --efficiency-lock benchmarks/baselines/efficiency_thresholds_full_refresh_phase10_qa.json --profile-lock benchmarks/baselines/phase4_profile_thresholds_full_refresh_phase10_qa.json --require-all-locked-cases --output outputs/benchmarks_performance/benchmark_gate_full_refresh_phase10_qa`
  - Result: `Gate pass: False`
  - Primary failure mode: strict efficiency timing lock jitter (1-D lock checks passed).
  - Rerun (`benchmark_gate_full_refresh_phase10_qa_rerun`) remained `Gate pass: False` with same class of efficiency-only timing drift.

## Key Artifacts

- Quick pass report:
  - `outputs/benchmarks_performance/benchmark_gate_phase10_quick_qa/benchmark_gate_report.json`
- Full refresh lock files:
  - `benchmarks/baselines/efficiency_thresholds_full_refresh_phase10_qa.json`
  - `benchmarks/baselines/phase4_profile_thresholds_full_refresh_phase10_qa.json`
- Full gate reports:
  - `outputs/benchmarks_performance/benchmark_gate_full_refresh_phase10_qa/benchmark_gate_report.json`
  - `outputs/benchmarks_performance/benchmark_gate_full_refresh_phase10_qa_rerun/benchmark_gate_report.json`
- Example QA artifacts (full gate):
  - `outputs/benchmarks_performance/benchmark_gate_full_refresh_phase10_qa/qa_figures/sersic_n4_noiseless_isoster_phase4_qa.png`
  - `outputs/benchmarks_performance/benchmark_gate_full_refresh_phase10_qa/qa_figures/sersic_n1_high_eps_noise_isoster_phase4_qa.png`
  - `outputs/benchmarks_performance/benchmark_gate_full_refresh_phase10_qa/qa_figures/sersic_n4_extreme_eps_noise_isoster_phase4_qa.png`

## Resume Commands

```bash
uv run ruff check benchmarks/baselines/run_benchmark_gate.py
uv run python benchmarks/baselines/run_benchmark_gate.py --quick --n-runs 1 --efficiency-lock benchmarks/baselines/efficiency_thresholds_quick_2026-02-14.json --output outputs/benchmarks_performance/benchmark_gate_phase10_quick_qa
uv run python benchmarks/baselines/run_benchmark_gate.py --n-runs 3 --efficiency-lock benchmarks/baselines/efficiency_thresholds_full_refresh_phase10_qa.json --profile-lock benchmarks/baselines/phase4_profile_thresholds_full_refresh_phase10_qa.json --require-all-locked-cases --output outputs/benchmarks_performance/benchmark_gate_full_refresh_phase10_qa
```

## Post-Closeout Update: Efficiency Timing Jitter Handling

After initial integration, full gate failures were traced to strict efficiency timing comparisons that did not account for measured per-case runtime variance.

### Policy Update

- Gate now applies measured jitter tolerance per efficiency case:
  - `adjusted_threshold = locked_threshold + max(0.002s, 3.0 * current_std_time)`
- Configurable via:
  - `--efficiency-time-jitter-sigma`
  - `--efficiency-time-jitter-floor-seconds`
- Policy parameters are written into `benchmark_gate_report.json` under `gate_policy`.

### Verification

- Quick gate with jitter policy:
  - `outputs/benchmarks_performance/benchmark_gate_phase10_quick_qa_jitter/benchmark_gate_report.json`
  - Result: `Gate pass: True`
- Full gate with jitter policy (sequential run):
  - `outputs/benchmarks_performance/benchmark_gate_full_refresh_phase10_qa_jitter_seq/benchmark_gate_report.json`
  - Result: `Gate pass: True`

### Operational Note

- Do not run multiple benchmark gate commands in parallel when collecting timing evidence; concurrent runs cause resource contention and invalidate runtime comparisons.

## Post-Closeout Update: Gate Defaults File

To avoid repeated CLI flags, gate runtime/QA defaults now come from:

- `benchmarks/baselines/benchmark_gate_defaults.json`

The gate script supports optional override path:

- `--gate-defaults <path>`

CLI-provided values still override file values.
