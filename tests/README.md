# Tests

This folder contains automated correctness tests for ISOSTER development.

## Scope

- `unit/`: low-level module behavior and API contracts.
- `integration/`: cross-module behavior on synthetic data.
- `validation/`: method-level validation against expectations/reference behavior.
- `real_data/`: real-galaxy tests, marked `real_data` and excluded by default.
- `fixtures/`: reusable synthetic data factories.

## Non-Scope

- Performance benchmarking belongs in `benchmarks/`.
- End-to-end scientific workflows belong in `examples/`.

## Reproducible Commands

```bash
# Fast local default (excludes real_data by marker policy)
uv run pytest tests/ -q

# Unit only
uv run pytest tests/unit -q

# Integration only
uv run pytest tests/integration -q

# Validation only
uv run pytest tests/validation -q

# Real data tests (explicit)
uv run pytest tests/real_data -m real_data -v -s

# Focused Phase 4 regression checks
uv run pytest tests/unit/test_public_api.py tests/integration/test_cli.py -q
uv run pytest tests/integration/test_sersic_accuracy.py tests/integration/test_numba_validation.py -q

# Collection sanity check
uv run pytest --collect-only -q
```

## Output Policy

Tests that generate artifacts should write under `outputs/`, not under `tests/`.
You can override the output root with `ISOSTER_OUTPUT_ROOT`.

Recommended naming:
- `outputs/tests_unit/<run_id>/...`
- `outputs/tests_integration/<run_id>/...`
- `outputs/tests_validation/<run_id>/...`
- `outputs/tests_real_data/<run_id>/...`
