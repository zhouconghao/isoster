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

# Method comparison benchmark (quick mode)
uv run python benchmarks/performance/bench_vs_photutils.py --quick

# Method comparison with explicit output root
uv run python benchmarks/performance/bench_vs_photutils.py --quick --output outputs/benchmarks_performance/manual_run
```

## Output Policy

Benchmark outputs should be saved under `outputs/` with source-specific folder names.
You can override the output root with `ISOSTER_OUTPUT_ROOT`.

Recommended naming:
- `outputs/benchmarks_performance/<run_id>/...`
- `outputs/benchmarks_profiling/<run_id>/...`
