# ISOSTER: ISOphote fitting with Speedy Templated Extraction and Regression

ISOSTER is an accelerated Python library for elliptical isophote fitting in galaxy images. It provides a significant performance boost over standard implementations while maintaining scientific accuracy and compatibility.

## Key Features

- **High Performance**: 10-15x faster than `photutils.isophote` using vectorized path-based sampling.
- **Multiband Analysis**: Template-based forced photometry for consistent color profiles across wavelengths.
- **Advanced Harmonics**: Simultaneous higher-order harmonic fitting and 2D model reconstruction with shape deviations.
- **Modular Design**: Refactored into specialized modules for sampling, fitting, and model building.
- **Enhanced Photometry**: Includes local flux integration metrics (`tflux_e`, `tflux_c`, `npix_e`, `npix_c`).
- **Model Building**: Reconstruct 2D galaxy images from isophote profiles with optional harmonic deviations.
- **Photutils Compatibility**: Logical and algorithmic consistency with industry standards.
- **Function-based API**: Simple, stateless API for easier integration and testing.

## Installation

```bash
pip install .
```

## Basic Usage

### Python API

```python
import isoster
from astropy.io import fits

image = fits.getdata("galaxy.fits")
config = {
    'sma0': 10.0,
    'minsma': 0.0,
    'maxsma': 100.0,
    'full_photometry': True  # Enable flux integration metrics
}

# Run optimized fitting
results = isoster.fit_image(image, None, config)

# Save results to FITS
isoster.isophote_results_to_fits(results, "isophotes.fits")

# Build 2D model with harmonics
model = isoster.build_isoster_model(
    image.shape,
    results['isophotes'],
    use_harmonics=True,      # Include harmonic deviations (default)
    harmonic_orders=[3, 4]   # Higher-order shape features
)
fits.writeto("model.fits", model, overwrite=True)
```

### Multiband Analysis

```python
import isoster
from astropy.io import fits

# Fit reference band (e.g., g-band) with full geometry fitting
image_g = fits.getdata("galaxy_g.fits")
results_g = isoster.fit_image(image_g, None, config)
isoster.isophote_results_to_fits(results_g, "galaxy_g_isophotes.fits")

# Apply g-band geometry to r-band for consistent photometry
image_r = fits.getdata("galaxy_r.fits")
results_r = isoster.fit_image(
    image_r, None, config,
    template_isophotes=results_g['isophotes']
)

# OR load template from saved FITS
template = isoster.isophote_results_from_fits('galaxy_g_isophotes.fits')
image_i = fits.getdata("galaxy_i.fits")
results_i = isoster.fit_image(
    image_i, None, config,
    template_isophotes=template['isophotes']
)
```

## Important Considerations

### High-Ellipticity Fitting (ε > 0.6)

For highly elliptical galaxies (ellipticity ε = 1 - b/a > 0.6), the default `maxgerr=0.5` parameter may be too strict, causing excessive isophote failures even when photometry is accurate. This occurs because gradient errors are amplified in high-ellipticity geometries.

**Recommendation:** Use relaxed `maxgerr` values for high-ellipticity cases:

```python
from isoster.config import IsosterConfig

# For high ellipticity (eps > 0.6)
config = IsosterConfig(
    eps=0.7,
    pa=1.0,
    maxgerr=1.0,  # Relaxed from default 0.5
    use_eccentric_anomaly=True,  # Recommended for eps > 0.3
)

results = isoster.fit_image(image, mask, config)
```

**Guidelines:**
- **ε < 0.6**: Use default `maxgerr=0.5`
- **0.6 ≤ ε ≤ 0.7**: Use `maxgerr=1.0` (improves convergence from ~40% to ~85%)
- **ε > 0.7**: Use `maxgerr=1.2` or higher
- Always enable `use_eccentric_anomaly=True` for ε > 0.3

See `docs/stop-codes.md` for detailed information on stop codes and convergence criteria.

### Advanced Features

ISOSTER includes several advanced capabilities for challenging data:

**Permissive Geometry Mode**: For data with convergence issues, enable "best effort" geometry updates:
```python
config = IsosterConfig(
    permissive_geometry=True,  # Continue fitting through failures
)
```

**Central Region Regularization**: Stabilize fitting in low S/N central regions:
```python
config = IsosterConfig(
    use_central_regularization=True,
    central_reg_sma_threshold=5.0,  # Regularize below 5 pixels
    central_reg_strength=1.0,       # Moderate strength
)
```

**Simultaneous Higher-Order Harmonics**: Capture morphological features (boxy/disky isophotes):
```python
config = IsosterConfig(
    simultaneous_harmonics=True,
    harmonic_orders=[3, 4, 5, 6],  # Fit up to 6th order
    use_eccentric_anomaly=True,    # Recommended with harmonics
)
```

For complete documentation, see:

- `docs/README.md` for docs structure
- `docs/spec.md` for architecture and interfaces
- `docs/user-guide.md` for usage guidance
- `CLAUDE.md` for agent/development rules

## Test and Benchmark Entry Points

```bash
# Tests (default set)
pytest tests/ -q

# Real-data tests (explicit)
pytest tests/real_data -m real_data -v -s

# Benchmarks
python benchmarks/performance/bench_vs_photutils.py --quick
```

## Repository Structure

- `isoster/`:
    - `sampling.py`: Vectorized elliptical coordinate sampling.
    - `fitting.py`: Iterative harmonic fitting and error estimation.
    - `driver.py`: High-level image fitting orchestration.
    - `model.py`: 2D image reconstruction.
    - `plotting.py`: Comparison and analysis visualization.
- `tests/`: Unit/integration/validation tests (`tests/README.md`).
- `benchmarks/`: Performance and profiling benchmarks (`benchmarks/README.md`).
- `examples/`: Reproducible mock/realistic workflows (`examples/README.md`).
- `outputs/`: Generated artifacts (gitignored).
- `docs/`: Stable docs + archived historical notes (`docs/README.md`).

## Acknowledgments

ISOSTER began as an optimization of the `photutils.isophote` package. We thank the photutils contributors for their robust foundational algorithms.
