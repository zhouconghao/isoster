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

# Method comparison benchmark (quick mode)
uv run python benchmarks/performance/bench_vs_photutils.py --quick

# Profiling artifacts (.prof + summaries)
uv run python benchmarks/profiling/profile_hotpaths.py --multi 1 --top-n 20
uv run python benchmarks/profiling/profile_isophote.py --repetitions 1 --top-n 20

# Method comparison with explicit output root
uv run python benchmarks/performance/bench_vs_photutils.py --quick --output outputs/benchmarks_performance/manual_run

# Phase 4 profile baseline metrics
uv run python benchmarks/baselines/collect_phase4_profile_baseline.py

# Lock threshold file from baseline metrics
uv run python benchmarks/baselines/lock_phase4_thresholds.py

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
