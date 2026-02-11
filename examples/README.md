# Examples

This folder contains reproducible workflow examples on mock or real-like data.

## Scope

- `basic_usage.py`: minimal API walkthrough.
- `curve_of_growth.py`: curve-of-growth oriented workflow.
- `huang2013/`: realistic mock-galaxy workflow and QA generation.
- `data/`: example input data used by scripts.

## Non-Scope

- Regression/unit checks belong in `tests/`.
- Performance benchmarking belongs in `benchmarks/`.

## Reproducible Commands

```bash
# Basic usage example
uv run python examples/basic_usage.py

# Curve-of-growth example
uv run python examples/curve_of_growth.py

# Huang2013 mock workflow
uv run python examples/huang2013/test_huang2013_mocks.py --help
uv run python examples/huang2013/test_huang2013_mocks.py --output-dir outputs/examples_huang2013/manual_run
```

## Output Policy

Example outputs should be saved under `outputs/`.
You can override the output root with `ISOSTER_OUTPUT_ROOT`.

Recommended naming:
- `outputs/examples_basic_usage/<run_id>/...`
- `outputs/examples_curve_of_growth/<run_id>/...`
- `outputs/examples_huang2013/<run_id>/...`
