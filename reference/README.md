# Reference Implementation Directory

This directory contains the **photutils-compatible reference implementation** and historical files used for benchmarking, testing, and understanding the algorithm evolution.

## Purpose

The reference implementation provides:

1. **Compatibility testing** - Compare results with `photutils.isophote`
2. **Algorithm validation** - Verify correctness of the optimized main package
3. **Benchmarking baseline** - Measure performance improvements
4. **Educational resource** - Understand algorithm details and trade-offs

## Directory Contents

### Active Reference Implementation

These files implement a photutils-compatible API using the same optimization techniques as the main package:

- **`ellipse.py`** - `Ellipse` class compatible with `photutils.isophote.Ellipse`
- **`fitter.py`** - Fitting orchestration and convergence logic
- **`geometry.py`** - Geometric parameter management (`EllipseGeometry`)
- **`sample.py`** - Elliptical path sampling methods
- **`model.py`** - 2D image model reconstruction
- **`harmonics.py`** - Harmonic decomposition (1st and 2nd order)
- **`integrator.py`** - Integration modes (mean, median, adaptive)
- **`isophote.py`** - `Isophote` data structure
- **`tests/`** - Test suite for reference implementation
- **`ellipse`** - Command-line tool (reference CLI)
- **`isofit`** - Alternative CLI entry point

### Historical Files

- **`optimize_backup.py`** - ⚠️ **HISTORICAL REFERENCE ONLY - DO NOT USE**
  - Original monolithic implementation (pre-refactoring, circa December 2023)
  - Contains extensive inline documentation comparing algorithms to photutils
  - Superseded by the modular implementation in `isoster/` (driver.py, fitting.py, sampling.py)
  - Kept for historical context and algorithm documentation
  - **Not maintained** - use main `isoster` package instead

- **`ellipse_model.pyx`** - Cython optimization experiments (unused)

## Usage

### Installing the Reference Implementation

The reference implementation is **excluded from the main package install** (see `pyproject.toml`). To use it:

```bash
# Add reference/ to PYTHONPATH
export PYTHONPATH=/path/to/isoster/reference:$PYTHONPATH

# Or import directly from the directory
python
>>> import sys
>>> sys.path.insert(0, '/path/to/isoster/reference')
>>> from ellipse import Ellipse
```

### Running Reference Tests

```bash
# From isoster root directory
pytest reference/tests/ -v
```

### Comparing with Main Package

```python
# Main package (optimized)
import isoster
results_main = isoster.fit_image(image, mask, config)

# Reference implementation (for validation)
from reference.ellipse import Ellipse
ellipse = Ellipse(image, x0=config.x0, y0=config.y0, sma=config.sma0)
ellipse.fit_image()
results_ref = ellipse.to_table()
```

## Implementation Differences

| Feature | Main Package (`isoster/`) | Reference (`reference/`) |
|---------|--------------------------|--------------------------|
| **API** | Functional (`fit_image()`) | Object-oriented (`Ellipse`) |
| **Structure** | Modular (driver, fitting, sampling) | Modular (ellipse, fitter, geometry) |
| **Dependencies** | Minimal (numpy, scipy, astropy) | Same + photutils (optional) |
| **Performance** | Optimized (10-15x speedup) | Same algorithms, slightly different architecture |
| **Use case** | Production, large-scale surveys | Testing, validation, compatibility |

## Key Design Principles

Both implementations follow the same core algorithm (Jedrzejewski 1987) with modern optimizations:

1. **Vectorized path sampling** - Use `scipy.ndimage.map_coordinates` instead of pixel-by-pixel iteration
2. **Stateless functions** - Avoid object creation overhead in tight loops
3. **Efficient mask handling** - Sample masks with nearest-neighbor interpolation
4. **Eccentric anomaly mode** - Uniform arc-length sampling for high-ellipticity galaxies

## Maintenance Status

- ✅ **Active files** - Maintained for compatibility testing and benchmarking
- ❌ **`optimize_backup.py`** - Historical only, not maintained
- ❌ **`ellipse_model.pyx`** - Experimental, not maintained

## Migration Guide

If you have code using the reference implementation:

```python
# OLD (reference)
from reference.ellipse import Ellipse
ell = Ellipse(image, sma=10)
ell.fit()
table = ell.to_table()

# NEW (main package - recommended)
import isoster
from isoster.config import IsosterConfig
config = IsosterConfig(sma0=10, maxsma=100)
results = isoster.fit_image(image, mask=None, config=config)
table = isoster.isophote_results_to_astropy_tables(results)
```

## See Also

- Main package documentation: `../README.md`
- Algorithm details: `../docs/algorithm.md`
- Architecture overview: `../docs/description.md`
- Code review: `../CODE_REVIEW.md`
