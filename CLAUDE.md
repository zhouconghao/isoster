# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ISOSTER (ISOphote on STERoid) is an accelerated Python library for elliptical isophote fitting in galaxy images. It provides 10-15x faster performance compared to `photutils.isophote` using vectorized path-based sampling via scipy's `map_coordinates`.

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
