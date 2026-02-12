# 2026-02-12 Phase 5 Follow-up: Numba Diagnostics + mockgal Presets

## Summary

- Reviewed existing benchmark/profiling artifacts under:
  - `outputs/benchmarks_performance/`
  - `outputs/benchmarks_profiling/`
- Implemented deeper numba speedup diagnostics in:
  - `benchmarks/performance/bench_numba_speedup.py`
- Added science-ready preset support in:
  - `benchmarks/baselines/mockgal_adapter.py`
- Updated docs:
  - `benchmarks/README.md`
  - `docs/todo.md`
  - `docs/lessons.md`
  - `docs/spec.md`
  - `docs/test-benchmark-improvement-plan.md`

## Numba Diagnostic Enhancements

`bench_numba_speedup.py` now supports:
- `--n-runs` for stability control
- `--scale-factor` for workload scaling tests
- mean and steady-state speedup
- first-run overhead and coefficient-of-variation diagnostics
- slowdown and high-variability flags in JSON summary

## Key Verification Runs

```bash
UV_CACHE_DIR=.uv-cache uv run python benchmarks/performance/bench_numba_speedup.py --n-runs 4 --scale-factor 1.0
UV_CACHE_DIR=.uv-cache uv run python benchmarks/performance/bench_numba_speedup.py --n-runs 3 --scale-factor 2.0 --output outputs/benchmarks_performance/bench_numba_speedup_scale2
```

### Observed Summaries

- Scale 1.0:
  - mean speedup: `1.32x`
  - steady-state mean speedup: `1.32x`
  - slowdown cases: `0`
- Scale 2.0:
  - mean speedup: `1.22x`
  - steady-state mean speedup: `1.25x`
  - slowdown cases (steady-state): `1`
  - high variability cases: `3`

## mockgal Preset Enhancements

Added presets:
- `single_noiseless_truth`
- `single_psf_noise_sblimit`
- `models_config_batch`

Validation commands:

```bash
UV_CACHE_DIR=.uv-cache uv run python benchmarks/baselines/mockgal_adapter.py --list-presets
UV_CACHE_DIR=.uv-cache uv run python benchmarks/baselines/mockgal_adapter.py --preset single_psf_noise_sblimit --dry-run
UV_CACHE_DIR=.uv-cache uv run python benchmarks/baselines/mockgal_adapter.py --preset models_config_batch --preset-value models_file=my_models.yaml --preset-value config_file=my_config.yaml --preset-value workers=2 --dry-run --output outputs/benchmarks_performance/mockgal_adapter_models_example
```

## Continuation Prompt

Continue from `docs/test-benchmark-improvement-plan-20260211` with Phase 5 follow-up in progress. Use `outputs/benchmarks_performance/bench_numba_speedup/numba_benchmark_results.json` and `outputs/benchmarks_performance/bench_numba_speedup_scale2/numba_benchmark_results.json` to design and implement case-specific numba profiling drills for flagged/high-variability scenarios (especially `n1_medium_eps07__scale2`), then add reusable preset template files for `models_config_batch` (example `galaxies.yaml` + `image_config.yaml`) under `outputs/` or `examples/` as appropriate.
