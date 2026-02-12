# 2026-02-11 Phase 4 Step 1 Implementation

## Scope Completed

1. Baseline metric collection started with a reproducible script:
   - `benchmarks/baselines/collect_phase4_profile_baseline.py`
2. False-pass hardening completed for current integration targets:
   - `tests/integration/test_sersic_accuracy.py`
   - `tests/integration/test_numba_validation.py`
3. Missing API/CLI test coverage added:
   - `tests/unit/test_public_api.py`
   - `tests/integration/test_cli.py`
4. Canonical M51 basic real-data test normalized:
   - `tests/real_data/test_m51.py::TestM51::test_m51_test`

## Baseline Artifact

- Output path:
  - `outputs/tests_integration/baseline_metrics/phase4_profile_baseline_metrics.json`
- Measured cases:
  - `sersic_n4_noiseless`: valid=19, median|ΔI|=0.000096, max|ΔI|=0.002098
  - `sersic_n1_high_eps_noise`: valid=15, median|ΔI|=0.003759, max|ΔI|=0.114202
  - `sersic_n4_extreme_eps_noise`: valid=15, median|ΔI|=0.002971, max|ΔI|=0.009876

## Verification Commands and Results

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/unit/test_public_api.py tests/integration/test_cli.py -q
# 5 passed

UV_CACHE_DIR=.uv-cache uv run pytest tests/integration/test_sersic_accuracy.py tests/integration/test_numba_validation.py -q
# 27 passed

UV_CACHE_DIR=.uv-cache uv run pytest tests/real_data/test_m51.py --collect-only -q -m real_data
# collected tests/real_data/test_m51.py::TestM51::test_m51_test
```

## Remaining Phase 4 Work

1. Lock thresholds from collected baseline distributions and document rationale.
2. Standardize benchmark/profiling metadata outputs and `.prof` artifact persistence.
3. Implement optional `mockgal.py` adapter workflow for future high-fidelity mocks.
