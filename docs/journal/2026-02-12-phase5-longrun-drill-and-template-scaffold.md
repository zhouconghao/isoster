# 2026-02-12 Phase 5 Follow-up Extension: Long-Run Drill and Template Scaffold

## Summary

Executed the two approved follow-ups:

1. Longer flagged-case numba drill run with explicit variability delta analysis.
2. Added and validated a helper script to scaffold `models_config_batch` templates into a new output run directory.

## Long-Run Drill Execution

Command:

```bash
/bin/zsh -lc 'UV_CACHE_DIR=.uv-cache uv run python benchmarks/profiling/profile_numba_flagged_cases.py --timing-runs 8 --profile-runs 2 --top-n 20 --output outputs/benchmarks_profiling/profile_numba_flagged_cases_longrun'
```

Selected cases (auto from flagged list + focus ordering):

- `n1_medium_eps07__scale2`
- `n1_large_eps06__scale2`
- `n1_medium_eps04__scale2`

Primary outputs:

- `outputs/benchmarks_profiling/profile_numba_flagged_cases_longrun/numba_case_drill_summary.json`
- `outputs/benchmarks_profiling/profile_numba_flagged_cases_longrun/variability_delta_summary.json`

### Variability Delta Highlights

From `variability_delta_summary.json`:

- `n1_medium_eps07__scale2`:
  - source numba CV: `0.16849`
  - drill steady-state CV: `0.00167`
  - interpretation: large reduction in observed runtime variability in drill conditions.
- `n1_large_eps06__scale2`:
  - source numba CV: `0.00453`
  - drill steady-state CV: `0.01578`
- `n1_medium_eps04__scale2`:
  - source numba CV: `0.00027`
  - drill steady-state CV: `0.11441`
  - note: this run included an outlier timing sample (`~0.1129 s`) in the 8-sample sequence.

## Template Scaffold Helper

Added script:

- `benchmarks/baselines/scaffold_models_config_batch_templates.py`

Purpose:

- Copy tracked templates from:
  - `examples/mockgal/models_config_batch/galaxies.yaml`
  - `examples/mockgal/models_config_batch/image_config.yaml`
- Into a new output run directory under `outputs/`.
- Persist machine-readable manifest with environment + copied/skipped files.

Verification command:

```bash
/bin/zsh -lc 'UV_CACHE_DIR=.uv-cache uv run python benchmarks/baselines/scaffold_models_config_batch_templates.py --output outputs/benchmarks_performance/mockgal_models_config_batch_templates_run1'
```

Verified output:

- `outputs/benchmarks_performance/mockgal_models_config_batch_templates_run1/galaxies.yaml`
- `outputs/benchmarks_performance/mockgal_models_config_batch_templates_run1/image_config.yaml`
- `outputs/benchmarks_performance/mockgal_models_config_batch_templates_run1/template_scaffold_manifest.json`

## Note

In this environment, `uv` command execution required elevated mode due sandbox `system-configuration` panic behavior.
