# data/

Shared FITS datasets used by tests, benchmarks, and examples.

## Files

### IC3370_mock2.fits

- **Source**: Synthetic mock from Huang et al. (2013), generated with the
  external `mockgal.py` workflow using the `libprofit` engine.
- **Galaxy type**: Elliptical (mock Sérsic model, NGC-type morphology)
- **Image size**: 256 × 256 pixels
- **Pixel scale**: 0.168 arcsec/px (HSC-like)
- **Notes**: Primary benchmark target. Used in `benchmarks/ic3370_exhausted/` (39-config sweep)
  and `benchmarks/performance/bench_vs_autoprof.py`.
- **Rights**: Synthetic data, no restrictions.

### eso243-49.fits

- **Source**: Legacy Survey (DESI Legacy Imaging Surveys, DR9)
- **Galaxy type**: Edge-on S0 galaxy
- **Image size**: 256 × 256 pixels, 3-band cube (g, r, z)
- **Pixel scale**: 0.25 arcsec/px
- **Notes**: Used in EA harmonics comparison tests and AutoProf benchmark.
- **Rights**: Public survey data (CC0 / public domain per Legacy Survey policy).

### ngc3610.fits

- **Source**: Legacy Survey (DESI Legacy Imaging Surveys, DR9)
- **Galaxy type**: Boxy-bulge elliptical galaxy
- **Image size**: 256 × 256 pixels, 3-band cube (g, r, i)
- **Pixel scale**: 1.0 arcsec/px
- **Notes**: Used in EA harmonics comparison tests and AutoProf benchmark.
  NGC 3610 exhibits strong a4/b4 boxiness signatures.
- **Rights**: Public survey data (CC0 / public domain per Legacy Survey policy).

### m51/M51.fits

- **Source**: HST archival (public archive)
- **Galaxy type**: Grand-design spiral galaxy (M51 / NGC 5194)
- **Image size**: varies (full mosaic)
- **Pixel scale**: HST ACS pixel scale (~0.05 arcsec/px)
- **Notes**: Canonical basic real-data test dataset.
  Referenced by `tests/real_data/test_m51.py`.
- **Rights**: HST public archive data, no restrictions.

## Usage

Load a file from any script or test using a path relative to the project root:

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent  # adjust depth as needed
data_path = PROJECT_ROOT / "data" / "IC3370_mock2.fits"
```

Or from a test file two levels deep (`tests/real_data/`):

```python
DATA_DIR = Path(__file__).parent.parent.parent / "data"
```

## Notes

- These files are git-tracked (FITS are git-ignored by default; check `.gitignore`
  if a file fails to appear after clone).
- External Huang2013 data (full 20-galaxy set) lives at
  `/Users/mac/work/hsc/huang2013/<GALAXY>/` and is **not** tracked in this repo.
