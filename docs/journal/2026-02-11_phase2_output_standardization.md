# 2026-02-11: Phase 2 Output Standardization

## Scope

Standardize generated artifact paths under `outputs/` for tests, benchmarks, and examples.

## What Changed

- Added shared helper: `isoster/output_paths.py`.
  - `resolve_output_directory()`
  - `ISOSTER_OUTPUT_ROOT` support via `OUTPUT_ROOT_ENV_VAR`
- Migrated active writers:
  - `tests/integration/test_sersic_accuracy.py`
  - `tests/integration/test_numba_validation.py`
  - `tests/validation/test_model_residuals.py`
  - `tests/real_data/test_ea_harmonics_comparison.py`
  - `benchmarks/performance/bench_vs_photutils.py`
  - `benchmarks/performance/bench_numba_speedup.py`
  - `examples/basic_usage.py`
  - `examples/huang2013/test_huang2013_mocks.py` (default output dir)
- Updated reproducibility docs:
  - `tests/README.md`
  - `benchmarks/README.md`
  - `examples/README.md`
  - `docs/todo.md` phase 2 checklist and review
  - `docs/lessons.md`
- Fixed benchmark script import issue:
  - `benchmarks/performance/bench_vs_photutils.py` now imports from `benchmarks.utils.sersic_model`
- Removed legacy `tests/qa_outputs/` directory.

## Verification

- `pytest --collect-only -q` passed.
- Targeted tests passed:
  - `tests/integration/test_sersic_accuracy.py::test_sersic_n4_noiseless`
  - `tests/integration/test_numba_validation.py::test_numba_qa_figure`
  - `tests/validation/test_model_residuals.py::test_model_building`
- Benchmark CLI checks passed:
  - `python benchmarks/performance/bench_vs_photutils.py --help`
  - `python benchmarks/performance/bench_numba_speedup.py --help`
- Example execution check passed:
  - `python examples/basic_usage.py` generated
    `outputs/examples_basic_usage/basic_usage_example.png`

## Remaining

- Historical `qa/` reference figure workflows can be archived or migrated in a later cleanup pass.
- `mkdocs` CLI is unavailable in this shell; docs rendering verification remains pending.
