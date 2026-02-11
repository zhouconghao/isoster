# ISOSTER Technical Specification

## Purpose

ISOSTER is a Python library for accelerated elliptical isophote fitting on galaxy images, targeting scientific consistency with `photutils.isophote` while improving runtime through vectorized sampling and optimized fitting workflows.

## Primary Goals

- Fit radial isophote profiles with stable geometry estimation.
- Support template-based forced photometry for multiband analysis.
- Provide stop-code based quality control for profile interpretation.
- Export fit products for downstream analysis and model reconstruction.

## Public Interfaces

- `isoster.fit_image(image, mask=None, config=None, template_isophotes=None)`
- `isoster.fit_isophote(...)`
- `isoster.isophote_results_to_fits(...)`
- `isoster.isophote_results_from_fits(...)`
- `isoster.isophote_results_to_astropy_tables(...)`
- `isoster.build_isoster_model(...)`
- CLI entry point: `isoster` (`isoster/cli.py`)

## Module Architecture

- `isoster/optimize.py`: facade for core fitting API.
- `isoster/driver.py`: image-level orchestration (central, outward, inward growth).
- `isoster/fitting.py`: single-isophote fitting, harmonics, clipping, error terms.
- `isoster/sampling.py`: ellipse coordinate sampling and extraction.
- `isoster/model.py`: 2-D model reconstruction from fitted profiles.
- `isoster/config.py`: pydantic configuration schema and validation.
- `isoster/utils.py`: serialization and table/FITS conversion utilities.
- `isoster/numba_kernels.py`: optional accelerated kernels.

## Data Flow

1. Validate/construct config.
2. Choose mode:
   - normal fitting
   - fixed-geometry forced mode
   - template-based forced mode
3. Fit central point and radial isophotes.
4. Attach quality flags (`stop_code`) and optional derived measurements.
5. Return `{'isophotes': [...], 'config': IsosterConfig}`.

## Quality and Verification

- Unit/integration coverage in `tests/`.
- Real-data checks in `tests/real_data/` marked `real_data`.
- Performance comparisons in `benchmarks/`.
- Reproducible scientific workflows in `examples/`.

## Output Policy

Generated artifacts (plots, benchmark JSON, QA figures, temporary results) should be written under `outputs/` using deterministic folder naming by source and scenario.

## Documentation Policy

- Stable docs live in `docs/` root.
- Historical work products live in `docs/archive/`.
- Naming convention: lowercase kebab-case markdown files.
