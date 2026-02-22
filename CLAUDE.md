# CLAUDE.md

This file provides guidance to coding agents when working with code in this repository.

## Project Overview

ISOSTER (ISOphote on STERoid) is an accelerated Python library for elliptical isophote fitting in galaxy images. It provides 10-15x faster performance compared to `photutils.isophote` using vectorized path-based sampling via scipy's `map_coordinates`.

## Non-negotiable Rules for developing

- Always create a new branch for new features and new development. Do not merge back into the main branch unless I approve it.
- It is essential to provide informative and concise docstrings and inline comments.
- Warn the users when the context window has <30% left. Remind the users to save the conversation and start fresh. Also propose ways to compact the conversation and save the current progress in files.
- Keep record of development progress, important lessons, and critical decisions in markdown files in `docs/journal/`.
- Keep `docs/spec.md` updated for architecture and interface changes.
- Keep active execution checklist and end-of-phase review in `docs/todo.md`.
- Use lowercase kebab-case for markdown file names (for example, `stop-codes.md`).
- Generated artifacts must be written under `outputs/` and not mixed into source folders.
- Use `uv` as the default tool for dependency management and environment execution.

## Testing and Benchmark Directives (2026-02-11)

- The canonical basic real-data dataset is `examples/data/m51/M51.fits`; the basic real-data test should be named `m51_test`.
- For Huang2013 workflows, use a fixed default initial SMA of `6.0` pixels (`sma0`) instead of deriving it from `RE_PX1`.
- For future high-fidelity mock generation, use `/Users/mac/Dropbox/work/project/otters/isophote_test/mockgal.py` (libprofit-based) when PSF convolution and realistic background-noise controls are required.
- For `mockgal.py` benchmark/test workflows, force `--engine libprofit` and do not rely on astropy fallback rendering.
- For noiseless single-Sersic validation without PSF convolution, compare against an analytic 1-D Sersic truth using accurate `b_n` evaluation (for example, `scipy.special.gammaincinv`) rather than low-accuracy approximations.
- Tests and benchmarks must be quantitative, with explicit statistics from 1-D profile deviations and 2-D residual diagnostics.
- Treat 2-D residual metrics as system-level diagnostics (profile extraction + model reconstruction combined), not as an isolated extraction-only metric.

## Context and Memory Preservation

When a task is long or the context window is becoming constrained:

1. Write a concise status snapshot to `docs/journal/` (what was done, what remains, blockers).
2. Update `docs/todo.md` checklist status and review notes.
3. Update `docs/spec.md` if architecture or interfaces changed.
4. Update `docs/lessons.md` when a correction yields a reusable lesson.
5. Before ending, include exact file paths and commands needed to resume.

## Build and Test Commands

```bash
# Sync project environment (core + development + docs tooling)
uv sync --extra dev --extra docs

# Run all tests (main package)
uv run pytest tests/

# Run a single test file
uv run pytest tests/unit/test_fitting.py

# Run reference implementation tests (photutils-compatible)
uv run pytest reference/tests/

# Run with verbose output
uv run pytest tests/ -v

# CLI usage
uv run isoster image.fits --output isophotes.fits --config config.yaml

# Build docs
uv run mkdocs serve
```

## uv Workflow Rules

Follow these rules for all Python environment and dependency work in this repository:

1. Use `uv` as the single workflow for environment and dependency management.
2. Do not use `pip`, `poetry`, or `conda` commands for project dependency changes.
3. Install/sync environment with:
   - `uv sync --extra dev --extra docs`
4. Run project commands through the managed environment:
   - `uv run pytest ...`
   - `uv run python ...`
   - `uv run isoster ...`
   - `uv run mkdocs ...`
5. When adding/removing dependencies, update `pyproject.toml` and then run:
   - `uv lock`
   - `uv sync --extra dev --extra docs`
6. Keep `uv.lock` committed and up to date with dependency changes.
7. For tools that are not compatible with all Python versions in `requires-python`,
   use dependency markers in `pyproject.toml` (for example `python_version >= '3.9'`).
8. Minimum verification after dependency changes:
   - `uv run pytest --collect-only -q`
   - `uv run mkdocs --version`

## Architecture

### Core Pipeline Flow

1. **driver.py** (`fit_image`) - Main entry point orchestrating the fitting:
   - Fits central pixel, then initial isophote at `sma0`
   - Iteratively grows outward (increasing SMA) then inward
   - Supports template-based forced photometry via `template_isophotes` parameter for multiband analysis
   - Returns dict with `{'isophotes': [...], 'config': IsosterConfig}`

2. **sampling.py** (`extract_isophote_data`) - Vectorized ellipse sampling:
   - Returns `IsophoteData` namedtuple with `(angles, phi, intens, radii)`
   - Two sampling modes: uniform in position angle φ (traditional) or eccentric anomaly ψ (Ciambur 2015, better for high ellipticity)

3. **fitting.py** (`fit_isophote`) - Single isophote harmonic fitting:
   - Fits 1st and 2nd harmonics to intensity profile: `I(φ) = Ī + A₁sin(φ) + B₁cos(φ) + A₂sin(2φ) + B₂cos(2φ)`
   - Iteratively updates geometry (x0, y0, eps, pa) based on harmonic coefficients
   - Convergence when `max_harmonic_amplitude < conver * rms`

4. **config.py** (`IsosterConfig`) - Pydantic configuration model with all parameters

5. **model.py** (`build_isoster_model`) - Reconstructs 2D image from isophote profiles using radial interpolation, with optional higher-order harmonic deviations

### Key Concepts

- **Eccentric Anomaly (EA) mode**: When `use_eccentric_anomaly=True`, samples uniformly in ψ for uniform arc-length coverage on high-ellipticity ellipses. Harmonics fitted in ψ-space, geometry updates in φ-space. Recommended for ε > 0.3.

- **Forced mode**: When `forced=True`, extracts photometry at predefined SMA values without geometry fitting (uses single fixed geometry).

- **Template-based forced mode**: When `template_isophotes` is provided to `fit_image()`, applies variable geometry (one per SMA) from the template to new image. Enables consistent multiband photometry by using geometry from one band (e.g., g-band) for other bands (r, i, z).

- **Simultaneous harmonics fitting**: When `simultaneous_harmonics=True`, uses ISOFIT-style simultaneous fitting of higher-order harmonics (orders specified by `harmonic_orders=[3, 4, ...]`) that accounts for cross-correlations.

- **Permissive geometry mode**: When `permissive_geometry=True`, enables photutils-style "best effort" geometry updates that continue even from failed fits, preventing cascading failures in challenging data.

- **Central regularization**: When `use_central_regularization=True`, applies geometry regularization in low S/N central regions (SMA < `central_reg_sma_threshold`) to stabilize fitting by penalizing large geometry changes.

- **Convergence scaling**: `convergence_scaling='sector_area'` (default) scales the convergence threshold by the approximate sector area, which grows with SMA. This matches photutils behavior and eliminates most stop=2 failures at outer isophotes. Use `'none'` for legacy (SMA-independent) behavior.

- **Geometry damping**: `geometry_damping` (0-1, default 0.7) scales geometry corrections to prevent oscillations. The default 0.7 was validated across 20 Huang2013 galaxies, eliminating nearly all stop=2 failures when combined with `sector_area` scaling. Use `1.0` for legacy (undamped) behavior.

- **Geometry convergence**: When `geometry_convergence=True`, declares convergence when geometry parameters stabilize for `geometry_stable_iters` consecutive iterations (within `geometry_tolerance`), even if the harmonic criterion is not met.

- **Stop codes**: 0=converged, 1=too many flagged pixels, 2=max iterations reached, 3=too few points, -1=gradient error

### Directory Layout

- `isoster/` - Main package
- `reference/` - Photutils-compatible reference implementation (excluded from install)
- `tests/` - Unit/integration/validation/real-data tests
- `reference/tests/` - Tests for reference implementation
- `benchmarks/` - Performance and profiling scripts
- `examples/` - Reproducible workflow examples
- `outputs/` - Generated artifacts (gitignored)
- `docs/` - Stable docs + archived historical notes (`docs/index.md`)

## Public API

### Basic Usage

```python
import isoster

# Single-band fitting
results = isoster.fit_image(image, mask, config)

# Save results to FITS
isoster.isophote_results_to_fits(results, "output.fits")

# Load results from FITS
results = isoster.isophote_results_from_fits("output.fits")

# Build 2D model (with optional harmonics)
model = isoster.build_isoster_model(
    image.shape,
    results['isophotes'],
    use_harmonics=True,      # Include harmonic deviations (default: True)
    harmonic_orders=[3, 4]   # Which harmonics to use (default: [3, 4])
)

# Convert to Astropy table
table = isoster.isophote_results_to_astropy_tables(results)

# QA visualization
isoster.plot_qa_summary(...)
```

### Multiband Analysis with Template-Based Forced Photometry

```python
import isoster

# Fit reference band (e.g., g-band) normally
results_g = isoster.fit_image(image_g, mask_g, config)
isoster.isophote_results_to_fits(results_g, "galaxy_g.fits")

# Apply g-band geometry to other bands using template
results_r = isoster.fit_image(
    image_r, mask_r, config,
    template_isophotes=results_g['isophotes']
)

results_i = isoster.fit_image(
    image_i, mask_i, config,
    template_isophotes=results_g['isophotes']
)

# OR load template from saved FITS
template = isoster.isophote_results_from_fits('galaxy_g.fits')
results_r = isoster.fit_image(
    image_r, mask_r, config,
    template_isophotes=template['isophotes']
)
```

## Important Configuration Parameters

### Advanced Fitting Options

```python
from isoster.config import IsosterConfig

config = IsosterConfig(
    # Basic geometry
    sma0=10.0,           # Initial semi-major axis
    x0=None, y0=None,    # Center (None = auto-detect)
    eps=0.2, pa=0.0,     # Initial ellipticity and position angle

    # Sampling mode
    use_eccentric_anomaly=True,  # Recommended for eps > 0.3

    # Higher-order harmonics
    simultaneous_harmonics=False,  # Enable ISOFIT-style simultaneous fitting
    harmonic_orders=[3, 4],        # Harmonic orders to fit

    # Geometry behavior
    permissive_geometry=False,     # Enable photutils-style "best effort" updates

    # Central region stabilization
    use_central_regularization=False,  # Enable geometry regularization
    central_reg_sma_threshold=5.0,     # SMA threshold (pixels)
    central_reg_strength=1.0,          # Regularization strength (0-10)
    central_reg_weights={              # Per-parameter weights
        'eps': 1.0,
        'pa': 1.0,
        'center': 1.0
    },

    # Convergence criteria
    maxgerr=0.5,         # Max gradient error (use 1.0-1.2 for eps > 0.6)
    conver=0.05,         # Harmonic convergence threshold
    convergence_scaling='sector_area',  # Scale threshold with SMA (default, matches photutils)
    geometry_damping=0.7,               # Damping factor for geometry updates (default 0.7; 1.0 = no damping)
    geometry_convergence=False,         # Enable geometry-stability convergence
    geometry_tolerance=0.01,            # Tolerance for geometry convergence
    geometry_stable_iters=3,            # Consecutive stable iterations required

    # Photometry
    full_photometry=False,   # Enable flux integration metrics (default: False)
    compute_cog=False,       # Enable curve-of-growth photometry (default: False)
    integrator='mean',       # Integration method: 'mean', 'median', or 'adaptive'
    lsb_sma_threshold=None,  # SMA for switching to median in adaptive mode (pixels)
)
```

### Parameter Guidelines

- **Eccentric Anomaly (EA) mode**: Use `use_eccentric_anomaly=True` for ε > 0.3 to improve sampling uniformity
- **High ellipticity**: For ε > 0.6, use relaxed `maxgerr=1.0` or higher to prevent excessive failures
- **Challenging data**: Enable `permissive_geometry=True` to continue fitting through convergence issues
- **Low S/N centers**: Enable `use_central_regularization=True` with appropriate threshold and strength
- **Morphological features**: Enable `simultaneous_harmonics=True` to capture higher-order deviations
- **Convergence scaling**: Default `convergence_scaling='sector_area'` matches photutils behavior; use `'none'` to revert to legacy constant threshold
- **Geometry damping**: Default `geometry_damping=0.7` stabilizes most cases; use `0.5` for severely oscillating fits, or `1.0` for legacy undamped behavior
- **Geometry-based convergence**: Enable `geometry_convergence=True` as supplementary convergence criterion for challenging outer isophotes

## QA Figure Rules

When making QA figures to compare the `isoster` result with the truth or the `photutils.isophote` results, following the guidelines below:

- Show the original image with a few selective isophotes on top.
- If possible, reconstruct the 2-D model and subtract it from the image to highlight the residual pattern is informative.
- When showing 1-D profiles, using `SMA ** 0.25` as the X-axis as it compress the outer profile that typically has a shallow slope while not zooming in to the center too much
- When comparing with truth or different methods, use the relative 1-D residual or difference in the form of `(intensity_isoster - intensity_reference) / intensity_reference`.
- Arrange all the sub-plots for 1-D information, including the surface brightness, residual/difference, position angle, axis ratio, and centroid vertically, sharing the same X-axis to save space. Among them, 1-D surface brightness profile should occupy larger area than the rest.
- Using scatter plot with errorbar to show these 1-D profiles (`plt.scatter()`), not lines (`plt.plot`). Using dash line for the true model profiles.
- Intensity, position angle, axis ratio, and centroid results in the outskirt can have huge errorbars. When setting the Y-axis ranges, do not include the error bars.
- Normalize the position angle: sudden jump larger than 90 degrees often mean normalization issue.
- For `isoster` or `photutils.isophote` results, should visually separate the valid and problematic 1-D datapoints using the stop code.
- Treat the finalized IC2597 Huang2013 basic-QA style (`build_method_qa_figure` / `build_comparison_qa_figure`) as the default style/habit baseline for future QA figures unless a task explicitly requests a different style.

## Benchmark Tests Rules

### Mock Single-Sersic Model Tests

- The mock galaxy model shall be centered with the image array.
- The half-size of the mock image shall be at least 10 times of the effective radius of the mock model; if the mock model's effective radius is not very large, 15x would be even better.
- Pay attention to the oversampling ratio, high-Sersic index and high ellipticity often require higher oversampling in the central region.
- When comparing results between the truth or the reference profile:
  - Ignore the region smaller than 3 pixels when there is no PSF convolution because sampling the central region often has numerical issues; ignore the region `<= 2 * psf_fwhm` due to PSF convolution.
  - Ignore the region in the outskirt where problematic data points (using stop code) begin to appear or the intensity error bars become huge.
- Metrics to evaluate the results:
  - Median or maximum difference of a property between `0.5 * r_effective` (or 3 pixels, whichever is larger) to `8 * r_effective` is good for noiseless mocks; and to `5 * r_effective` is good for mocks with noise.
  - Using median or maximum absolute different is a more strict standard.
  - `isoster` can provide the curve of growth measurement, the relative difference of the curve of growth values at a few typical radius could be useful metrics.

### Build Ellipse Model Tests

- Key residual statistics:
  - Fractional residual level: `100.0 * (model - data) / data`;
  - Fractional absolute residual level: `100.0 * |model - data| / data`
  - Chi-Square statistics: `(model - data) ** 2.0 / (sigma ** 2)`
- Key metrics to evaluate the 2-D ellipse model:
  - 1. Statistics of the fractional residual level, e.g., median or maximum values, within different radial range. This works the best for noiseless mock.
  - 1. Statistics of the integrated values of the fractional absolute residual level within different radial range. This works the best for noiseless mock.
  - 1. Integrated Chi-square statistics within different radial range. This works the best for noisy-added mocks or real images.
- Radial ranges:
  - < 0.5 Re (effective radius)
  - 0.5-4 Re
  - 4-8 Re
