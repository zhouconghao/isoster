# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ISOSTER (ISOphote on STERoid) is an accelerated Python library for elliptical isophote fitting in galaxy images. It provides 10-15x faster performance compared to `photutils.isophote` using vectorized path-based sampling via scipy's `map_coordinates`.

## Non-negotiable Rules for developing
- Always create a new branch for new features and new development. Do not merge back into the main branch unless I approve it.
- It is essential to provide informative and concise docstrings and inline comments.

## Build and Test Commands

```bash
# Install locally
pip install .

# Run all tests (main package)
pytest tests/

# Run a single test file
pytest tests/test_fitting.py

# Run reference implementation tests (photutils-compatible)
pytest reference/tests/

# Run with verbose output
pytest tests/ -v

# CLI usage
isoster image.fits --output isophotes.fits --config config.yaml
```

## Architecture

### Core Pipeline Flow

1. **driver.py** (`fit_image`) - Main entry point orchestrating the fitting:
   - Fits central pixel, then initial isophote at `sma0`
   - Iteratively grows outward (increasing SMA) then inward
   - Returns dict with `{'isophotes': [...], 'config': IsosterConfig}`

2. **sampling.py** (`extract_isophote_data`) - Vectorized ellipse sampling:
   - Returns `IsophoteData` namedtuple with `(angles, phi, intens, radii)`
   - Two sampling modes: uniform in position angle φ (traditional) or eccentric anomaly ψ (Ciambur 2015, better for high ellipticity)

3. **fitting.py** (`fit_isophote`) - Single isophote harmonic fitting:
   - Fits 1st and 2nd harmonics to intensity profile: `I(φ) = Ī + A₁sin(φ) + B₁cos(φ) + A₂sin(2φ) + B₂cos(2φ)`
   - Iteratively updates geometry (x0, y0, eps, pa) based on harmonic coefficients
   - Convergence when `max_harmonic_amplitude < conver * rms`

4. **config.py** (`IsosterConfig`) - Pydantic configuration model with all parameters

5. **model.py** (`build_ellipse_model`) - Reconstructs 2D image from isophote profiles

### Key Concepts

- **Eccentric Anomaly (EA) mode**: When `use_eccentric_anomaly=True`, samples uniformly in ψ for uniform arc-length coverage on high-ellipticity ellipses. Harmonics fitted in ψ-space, geometry updates in φ-space.

- **Forced mode**: When `forced=True`, extracts photometry at predefined SMA values without geometry fitting.

- **Stop codes**: 0=converged, 1=too many flagged pixels, 2=minor issues, 3=too few points, -1=gradient error

### Directory Layout

- `isoster/` - Main package
- `reference/` - Photutils-compatible reference implementation (excluded from install)
- `tests/` - Unit tests for main package
- `reference/tests/` - Tests for reference implementation
- `examples/` - Benchmark and usage examples
- `benchmarks/` - Performance profiling scripts

## Public API

```python
import isoster

results = isoster.fit_image(image, mask, config)
model = isoster.build_ellipse_model(image.shape, results['isophotes'])
isoster.isophote_results_to_fits(results, "output.fits")
table = isoster.isophote_results_to_astropy_tables(results)
isoster.plot_qa_summary(...)  # QA visualization
```

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

## Benchmark Tests using mock Sersic model

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