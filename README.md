# ISOSTER: ISOphote fitting with Speedy Templated Extraction and Regression

ISOSTER is an accelerated Python library for elliptical isophote fitting in galaxy images. It provides a significant performance boost over standard implementations while maintaining scientific accuracy and compatibility.

## Key Features

- **High Performance**: 10-15x faster than `photutils.isophote` using vectorized path-based sampling.
- **Modular Design**: Refactored into specialized modules for sampling, fitting, and model building.
- **Enhanced Photometry**: Includes local flux integration metrics (`tflux_e`, `tflux_c`, `npix_e`, `npix_c`).
- **Model Building**: Reconstruct 2D galaxy images from isophote profiles.
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

# Build 2D model
model = isoster.build_isoster_model(image.shape, results['isophotes'])
fits.writeto("model.fits", model, overwrite=True)
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

See `docs/STOP_CODES.md` for detailed information on stop codes and convergence criteria.

## Repository Structure

- `isoster/`:
    - `sampling.py`: Vectorized elliptical coordinate sampling.
    - `fitting.py`: Iterative harmonic fitting and error estimation.
    - `driver.py`: High-level image fitting loops.
    - `model.py`: 2D image reconstruction.
    - `plotting.py`: Comparison and analysis visualization.
- `examples/`: Comprehensive benchmarks and usage examples.
- `tests/`: Unit and regression tests.

## Acknowledgments

ISOSTER began as an optimization of the `photutils.isophote` package. We thank the photutils contributors for their robust foundational algorithms.
