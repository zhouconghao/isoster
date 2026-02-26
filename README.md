<p align="center">
  <img src="docs/isoster_logo.svg" alt="ISOSTER logo" width="320">
</p>

<h1 align="center">ISOSTER</h1>

<p align="center">
  <strong>ISOphote on STERoid</strong> — Accelerated elliptical isophote fitting for galaxy images
</p>

<p align="center">
  <a href="https://opensource.org/licenses/BSD-3-Clause"><img src="https://img.shields.io/badge/License-BSD_3--Clause-blue.svg" alt="License: BSD-3-Clause"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python 3.9+"></a>
</p>

---

ISOSTER is a Python library for elliptical isophote fitting that provides **10-15x faster performance** compared to `photutils.isophote`. It uses vectorized path-based sampling via scipy's `map_coordinates` while maintaining scientific accuracy and full compatibility with the photutils isophote analysis workflow.

## Installation

```bash
# Install from source
pip install .

# Or set up a development environment with uv
uv sync --extra dev --extra docs
```

## Quick Start

```python
import isoster
from astropy.io import fits

# Load a galaxy image
image = fits.getdata("galaxy.fits")

# Fit isophotes
config = {'sma0': 10.0, 'maxsma': 100.0}
results = isoster.fit_image(image, None, config)

# Save results to FITS
isoster.isophote_results_to_fits(results, "isophotes.fits")

# Build a 2D model with harmonic deviations
model = isoster.build_isoster_model(
    image.shape,
    results['isophotes'],
    use_harmonics=True,
    harmonic_orders=[3, 4],
)
```

### Multiband Analysis

Use template-based forced photometry for consistent color profiles across wavelengths:

```python
# Fit reference band (e.g., g-band) with full geometry fitting
results_g = isoster.fit_image(image_g, None, config)

# Apply g-band geometry to other bands
results_r = isoster.fit_image(image_r, None, config, template=results_g)
results_i = isoster.fit_image(image_i, None, config, template='galaxy_g.fits')
```

## Key Features

- **High performance**: 10-15x faster than `photutils.isophote` via vectorized sampling.
- **Template-based forced photometry**: Consistent multiband photometry using geometry from a reference band.
- **Eccentric anomaly sampling**: Uniform arc-length coverage for high-ellipticity galaxies (Ciambur 2015).
- **Simultaneous harmonics**: ISOFIT-style joint fitting of higher-order harmonics within the iteration loop.
- **2D model building**: Reconstruct galaxy images from isophote profiles with optional harmonic deviations.
- **Convergence controls**: Sector-area scaling, geometry damping, and geometry-stability convergence.
- **Photometry metrics**: Integrated flux, curve-of-growth, and adaptive integration modes.
- **Photutils compatibility**: Consistent algorithms and output format with industry standards.
- **Function-based API**: Simple, stateless interface for easy integration and testing.

## Documentation

- [User Guide](docs/user-guide.md) — usage guidance, public API, and stop-code reference
- [Configuration Reference](docs/configuration-reference.md) — all parameters and guidelines
- [Algorithm Notes](docs/algorithm.md) — fitting and sampling implementation details
- [Architecture Spec](docs/spec.md) — interfaces and design decisions
- [Future Roadmap](docs/future.md) — planned upgrades and research directions

## Repository Structure

```
isoster/          Core package (config, driver, sampling, fitting, model, plotting, cog)
tests/            Unit, integration, and validation tests
benchmarks/       Performance and profiling benchmarks
examples/         Reproducible workflow examples
docs/             Project documentation
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and pull request guidelines.

## Citation

If you use ISOSTER in your research, please cite using the metadata in [CITATION.cff](CITATION.cff).

## Acknowledgments

ISOSTER began as an optimization of the [`photutils.isophote`](https://photutils.readthedocs.io/) package. We thank the photutils contributors for their robust foundational algorithms.

## License

BSD-3-Clause. See [LICENSE](LICENSE) for details.
