# Benchmarks

This folder contains performance and comparison benchmarks.

## Scope

- `performance/`: runtime and throughput comparisons (including method-vs-method).
- `profiling/`: hotspot and profiler-oriented diagnostics.
- `baselines/`: baseline data used for benchmarking context.
- `utils/`: shared benchmark helpers.

## Goals

- Quantify speed and scaling behavior.
- Compare ISOSTER against reference methods under controlled scenarios.
- Generate reproducible machine-readable outputs (JSON/CSV) and optional figures.

## Reproducible Commands

```bash
# Efficiency benchmark
uv run python benchmarks/performance/bench_efficiency.py

# Numba speedup benchmark
uv run python benchmarks/performance/bench_numba_speedup.py

# Numba diagnostic sweep (more stable timing + scaled workload)
uv run python benchmarks/performance/bench_numba_speedup.py --n-runs 4 --scale-factor 1.0
uv run python benchmarks/performance/bench_numba_speedup.py --n-runs 3 --scale-factor 2.0 --output outputs/benchmarks_performance/bench_numba_speedup_scale2

# Flagged-case numba profiling drills (defaults target scale1+scale2 JSON outputs)
uv run python benchmarks/profiling/profile_numba_flagged_cases.py --timing-runs 3 --profile-runs 1

# Method comparison benchmark: isoster vs photutils (quick mode)
uv run python benchmarks/performance/bench_vs_photutils.py --quick

# Method comparison benchmark: isoster vs AutoProf (quick mode — IC3370 only)
uv run python benchmarks/performance/bench_vs_autoprof.py --quick --plots

# AutoProf full benchmark (all galaxies)
uv run python benchmarks/performance/bench_vs_autoprof.py --plots

# AutoProf benchmark — isoster only (skip AutoProf, replot from cache)
uv run python benchmarks/performance/bench_vs_autoprof.py --skip-autoprof --plots

# AutoProf benchmark — specific galaxies
uv run python benchmarks/performance/bench_vs_autoprof.py --galaxies IC3370_mock2 ngc3610 --plots

# Profiling artifacts (.prof + summaries)
uv run python benchmarks/profiling/profile_hotpaths.py --multi 1 --top-n 20
uv run python benchmarks/profiling/profile_isophote.py --repetitions 1 --top-n 20

# Method comparison with explicit output root
uv run python benchmarks/performance/bench_vs_photutils.py --quick --output outputs/benchmarks_performance/manual_run

# Phase 4 profile baseline metrics
uv run python benchmarks/baselines/collect_phase4_profile_baseline.py

# Lock threshold file from baseline metrics
uv run python benchmarks/baselines/lock_phase4_thresholds.py

# Lock efficiency thresholds from measured efficiency baselines
uv run python benchmarks/baselines/lock_efficiency_thresholds.py

# Run combined baseline-locked benchmark gate (efficiency + 1-D + 2-D system diagnostics)
# Defaults are loaded from benchmarks/baselines/benchmark_gate_defaults.json
# Efficiency timing gate default: adjusted_threshold = locked_threshold + max(0.002s, 3.0 * current_std_time)
uv run python benchmarks/baselines/run_benchmark_gate.py --quick --n-runs 1 --efficiency-lock benchmarks/baselines/efficiency_thresholds_quick_2026-02-14.json

# Override defaults file (optional)
uv run python benchmarks/baselines/run_benchmark_gate.py --quick --n-runs 1 --gate-defaults benchmarks/baselines/benchmark_gate_defaults.json --efficiency-lock benchmarks/baselines/efficiency_thresholds_quick_2026-02-14.json

# Full lock-refresh + full gate workflow before large batches
uv run python benchmarks/performance/bench_efficiency.py --n-runs 3 --output outputs/benchmarks_performance/bench_efficiency_full_refresh
uv run python benchmarks/baselines/lock_efficiency_thresholds.py --input outputs/benchmarks_performance/bench_efficiency_full_refresh/efficiency_benchmark_results.json --output benchmarks/baselines/efficiency_thresholds_full_refresh.json
uv run python benchmarks/baselines/collect_phase4_profile_baseline.py --output outputs/tests_integration/baseline_metrics_full_refresh
uv run python benchmarks/baselines/lock_phase4_thresholds.py --input outputs/tests_integration/baseline_metrics_full_refresh/phase4_profile_baseline_metrics.json --output benchmarks/baselines/phase4_profile_thresholds_full_refresh.json
# Run this full gate command sequentially (do not run concurrent benchmark commands).
uv run python benchmarks/baselines/run_benchmark_gate.py --n-runs 3 --efficiency-lock benchmarks/baselines/efficiency_thresholds_full_refresh.json --profile-lock benchmarks/baselines/phase4_profile_thresholds_full_refresh.json --require-all-locked-cases

# Gate now generates built-in Huang2013 basic-QA artifacts per Phase-4 case
uv run python benchmarks/baselines/run_benchmark_gate.py --quick --n-runs 1 --efficiency-lock benchmarks/baselines/efficiency_thresholds_quick_2026-02-14.json --qa-output-subdir qa_figures --qa-overlay-step 10 --qa-dpi 180

# Optional external mockgal adapter (safe dry-run)
uv run python benchmarks/baselines/mockgal_adapter.py --dry-run

# List and use science-ready mockgal presets
uv run python benchmarks/baselines/mockgal_adapter.py --list-presets
uv run python benchmarks/baselines/mockgal_adapter.py --preset single_psf_noise_sblimit --dry-run
uv run python benchmarks/baselines/mockgal_adapter.py --preset models_config_batch --dry-run
uv run python benchmarks/baselines/mockgal_adapter.py --preset models_config_batch --preset-value models_file=my_models.yaml --preset-value config_file=my_config.yaml --preset-value workers=2 --dry-run

# Scaffold reusable models_config_batch template files into a new outputs/ run directory
uv run python benchmarks/baselines/scaffold_models_config_batch_templates.py
```

## Output Policy

Benchmark outputs should be saved under `outputs/` with source-specific folder names.
You can override the output root with `ISOSTER_OUTPUT_ROOT`.

Recommended naming:
- `outputs/benchmarks_performance/<run_id>/...`
- `outputs/benchmarks_profiling/<run_id>/...`

Performance and profiling scripts should emit:
- machine-readable summaries (`.json`, `.csv`)
- plot artifacts (`.png`/`.pdf`) for summary or failing/borderline cases
- run metadata (git SHA, package versions, platform info)
- profiler artifacts (`.prof`) for profiling scripts
