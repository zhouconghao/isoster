# Examples

This folder contains reproducible workflow examples on synthetic, survey, and
external mock datasets.

## Layout

- `example_basic_usage/`
  - Minimal synthetic end-to-end walkthrough.
  - Run with `uv run python examples/example_basic_usage/basic_usage.py`.
  - Default output: `outputs/example_basic_usage/`.
- `example_cog/`
  - Curve-of-growth validation on a synthetic Sersic image.
  - Run with `uv run python examples/example_cog/example_curve_of_growth.py`.
  - Default output: `outputs/example_cog/`.
- `example_ls_highorder_harmonic/`
  - Real-galaxy Legacy Survey harmonic-fitting example on `eso243-49` and `ngc3610`.
  - Main entrypoint: `uv run python examples/example_ls_highorder_harmonic/run_example.py --galaxy eso243-49 --band-index 1`.
  - Default main output: `outputs/example_ls_highorder_harmonic/`.
  - Additional exploration scripts exist in this folder, but do not run them by default.
- `example_huang2013/`
  - Two-stage Huang2013 external mock workflow: profile extraction, QA afterburner, campaign runner, and cleanup helper.
  - Main products are intentionally written outside `outputs/` under the external Huang2013 root because each case generates many files.
  - Campaign summaries still default to `outputs/huang2013_campaign/`.

## Shared Data

- Example FITS inputs tracked in this repo now live under the repository-level `data/` folder, not `examples/data/`.
- Legacy Survey scripts expect files like `data/eso243-49.fits` and `data/ngc3610.fits`.

## Output Policy

- Default rule: each example folder writes to `outputs/<example-folder>/`.
- Exception: `example_huang2013` writes case-level artifacts under `--huang-root/<GALAXY>/mock<ID>/` and only campaign summaries under `outputs/` by default.
- Set `ISOSTER_OUTPUT_ROOT` to override the shared `outputs/` root for examples that use the standard output helper.

## Non-Scope

- Regression and unit checks belong in `tests/`.
- Performance benchmarks belong in `benchmarks/`.
