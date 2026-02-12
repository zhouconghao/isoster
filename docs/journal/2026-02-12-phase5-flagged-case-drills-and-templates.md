# 2026-02-12 Phase 5 Follow-up: Flagged Case Drills and Preset Templates

## Summary

- Implemented case-specific numba profiling drills in:
  - `benchmarks/profiling/profile_numba_flagged_cases.py`
- Added reusable `models_config_batch` template files:
  - `examples/mockgal/models_config_batch/galaxies.yaml`
  - `examples/mockgal/models_config_batch/image_config.yaml`
- Updated `mockgal_adapter` preset defaults to use tracked template paths.

## Drill Workflow

The new drill script reads measured benchmark JSON artifacts and selects flagged cases from:

- `summary.high_variability_cases`
- `summary.slowdown_cases_mean`
- `summary.slowdown_cases_steady_state`

Default source files:

- `outputs/benchmarks_performance/bench_numba_speedup/numba_benchmark_results.json`
- `outputs/benchmarks_performance/bench_numba_speedup_scale2/numba_benchmark_results.json`

Default priority case:

- `n1_medium_eps07__scale2`

Per-case artifacts:

- `case_profile.prof`
- `top_cumulative.txt`
- `top_tottime.txt`
- `case_drill_summary.json`

Aggregate artifact:

- `numba_case_drill_summary.json`

## Verification Runs

```bash
/bin/zsh -lc 'UV_CACHE_DIR=.uv-cache uv run python benchmarks/profiling/profile_numba_flagged_cases.py --timing-runs 2 --profile-runs 1 --top-n 8 --output outputs/benchmarks_profiling/profile_numba_flagged_cases_smoke'
/bin/zsh -lc 'UV_CACHE_DIR=.uv-cache uv run python benchmarks/baselines/mockgal_adapter.py --preset models_config_batch --dry-run --output outputs/benchmarks_performance/mockgal_adapter_template_example'
```

Observed outcomes:

- Drill run selected and completed 3 flagged cases:
  - `n1_medium_eps07__scale2`
  - `n1_large_eps06__scale2`
  - `n1_medium_eps04__scale2`
- `mockgal_adapter` dry-run command resolved template defaults to:
  - `examples/mockgal/models_config_batch/galaxies.yaml`
  - `examples/mockgal/models_config_batch/image_config.yaml`

## Notes

- In this environment, `uv` command execution needed elevated mode because sandbox mode triggered a `system-configuration` panic.
