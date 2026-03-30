# ISOSTER User Guide

This guide covers installation, configuration, common workflows, and stop-code interpretation.

## Installation

Use `uv` for environment and dependency workflow in this repository.

```bash
uv sync --extra dev --extra docs
uv run isoster --help
```

## Basic Fit Workflow

```python
from astropy.io import fits
from isoster import fit_image, isophote_results_to_fits
from isoster.config import IsosterConfig

image = fits.getdata("data/m51/M51.fits")

config = IsosterConfig(
    sma0=10.0,
    minsma=0.0,
    maxsma=120.0,
    full_photometry=True,
)

results = fit_image(image, mask=None, config=config)
isophote_results_to_fits(results, "outputs/m51_isophotes.fits")
```

## Mode Selection

`fit_image` mode priority:

1. `template` provided -> template-based forced photometry.
2. Otherwise -> regular iterative fitting.

### Template-Based Forced Photometry

Forced photometry (multiband analysis) is performed by passing the `template` argument. It accepts a results dict, a FITS path, or a list of isophote dicts.

```python
from isoster import fit_image
from isoster.config import IsosterConfig

config = IsosterConfig()

# Fit reference band (geometry discovery)
results_g = fit_image(image_g, mask=mask_g, config=config)

# Apply g-band geometry to r-band (forced photometry)
results_r = fit_image(
    image_r,
    mask=mask_r,
    config=config,
    template=results_g,
)
```

To perform forced photometry with a fixed geometry at specific SMAs, provide a list of dicts to `template`:

```python
template = [
    {'sma': s, 'x0': 128.0, 'y0': 128.0, 'eps': 0.3, 'pa': 0.2}
    for s in [5, 10, 20, 40, 80]
]
results = fit_image(image, mask=mask, config=config, template=template)
```

### Using Variance Maps (WLS Fitting)

When a per-pixel variance map is available (e.g., from survey pipelines like DESI Legacy Survey), pass it to `fit_image` to enable Weighted Least Squares fitting:

```python
from astropy.io import fits
from isoster import fit_image
from isoster.config import IsosterConfig
import numpy as np

image = fits.getdata("galaxy-image-r.fits.fz")
invvar = fits.getdata("galaxy-invvar-r.fits.fz")

# Convert inverse variance to variance; mask zero-invvar pixels
variance_map = np.where(invvar > 0, 1.0 / invvar, 1e30)

config = IsosterConfig(sma0=10.0, maxsma=100.0)
results = fit_image(image, config=config, variance_map=variance_map)
```

WLS benefits:
- **Exact covariance**: error bars come from the variance map directly, not from fit residuals.
- **Automatic outlier down-weighting**: cosmic rays and hot pixels (high variance) get negligible weight.
- **Pure photon-noise gradient error**: cleanly separated from galaxy structure scatter (arms, dust, bars).
- **Byte-identical fallback**: when `variance_map=None` (default), the code path is identical to OLS.

Input requirements and safeguards:
- `variance_map` must have the same shape as `image` (raises `ValueError` otherwise).
- `NaN` and `inf` values are replaced with `1e30` (near-zero weight) and a `RuntimeWarning` is emitted.
- Non-positive values (zeros, negatives) are clamped to `1e-30` internally with a warning. Consider masking these pixels instead for cleaner results.
- The caller's array is never mutated (copy-on-write).
- For inverse-variance maps, convert with `variance = 1.0 / invvar` and handle zero-invvar pixels appropriately (mask them or set variance to a large value like `1e30`).

Compatibility:
- WLS works with all constraint modes: `fix_center`, `fix_pa`, `fix_eps`, and `simultaneous_harmonics` (isofit).
- WLS error bars are typically 1.2–2.1x larger than OLS for outer isophotes, reflecting realistic per-pixel noise rather than fit-residual scatter.

## Key Configuration Options

### Sampling and Stability

- `use_eccentric_anomaly=True`: harmonic fitting in `psi` with geometry updates in `phi`.
- `permissive_geometry=True`: allows geometry propagation through weaker gradient diagnostics.
- `maxgerr`: controls tolerance for gradient relative error checks.

### Harmonics

- `compute_deviations=True`: enables higher-order deviation outputs (`a{n}`, `b{n}`).
- `harmonic_orders=[3, 4, ...]`: harmonic orders to compute.
- `simultaneous_harmonics=True`: true ISOFIT (Ciambur 2015) — fits higher-order harmonics jointly with geometry harmonics inside the iteration loop via a single extended design matrix. Accounts for cross-correlations and produces cleaner RMS estimates. Falls back to 5-param fit when insufficient sample points for the full design matrix.

### First Isophote Robustness

When the first isophote at `sma0` fails, the entire fitting silently returns only the central pixel.
Two config options improve this:

- `max_retry_first_isophote` (default `0`, disabled): number of retry attempts with perturbed `sma0`
  and initial geometry (`eps`, `pa`). Each attempt tries a different combination to find an
  acceptable starting isophote. Set to `5` for robust batch processing.
- `first_isophote_fail_count` (default `3`): how many consecutive initial isophotes must all fail
  before a `FIRST_FEW_ISOPHOTE_FAILURE` warning is emitted.

When a failure is detected, the result dict contains:
- `result["first_isophote_failure"]` = `True`
- `result["first_isophote_retry_log"]` (list of attempt dicts, only when retries were attempted)

```python
config = IsosterConfig(sma0=10.0, max_retry_first_isophote=5)
result = fit_image(image, mask=None, config=config)

if result.get("first_isophote_failure"):
    print("First isophote fitting failed — check sma0 and initial geometry")
```

### Photometry Outputs

- `full_photometry=True`: adds `tflux_e`, `tflux_c`, `npix_e`, `npix_c`.
- `compute_cog=True`: adds CoG fields (`cog`, `cog_annulus`, `area_annulus`, crossing flags) in regular mode.

## Model Reconstruction

```python
from isoster import build_isoster_model

# Auto-detect harmonic orders and EA mode from isophote results
model = build_isoster_model(
    image.shape,
    results["isophotes"],
    use_harmonics=True,
)
```

`build_isoster_model` parameters for harmonic reconstruction:

- `harmonic_orders` (default `None`): When `None`, auto-detects which harmonic orders are present by scanning isophote dicts for `a{n}` keys (n >= 3). Pass an explicit list like `[3, 4]` to restrict to specific orders.
- `use_eccentric_anomaly` (default `None`): When `None`, auto-detects from the `use_eccentric_anomaly` flag stored in isophote dicts by `fit_isophote()`. When `True`, evaluates harmonics in eccentric anomaly (psi) space; when `False`, uses position angle (phi) space. Must match the angle space used during fitting for correct reconstruction.

## Serialization Helpers

### FITS

`isophote_results_to_fits` writes a 3-HDU FITS file:

- **HDU 0** — `PrimaryHDU`: empty, no data.
- **HDU 1** — `BinTableHDU` named `ISOPHOTES`: one row per fitted isophote, columns matching the isophote dict keys.
- **HDU 2** — `BinTableHDU` named `CONFIG`: two columns (`PARAM`, `VALUE`) with the full `IsosterConfig` serialized as JSON strings, one row per field.

This layout avoids `HIERARCH` warnings that occurred when config was written as FITS header keywords. Files written by older versions (config in header keywords, no `CONFIG` extension) are still readable — the reader detects the missing `CONFIG` HDU and falls back to header-keyword reconstruction automatically.

```python
from isoster import isophote_results_to_fits, isophote_results_from_fits

# Save
isophote_results_to_fits(results, "outputs/galaxy_isophotes.fits")

# Load
loaded = isophote_results_from_fits("outputs/galaxy_isophotes.fits")
```

### ASDF

ASDF (Advanced Scientific Data Format) is supported as an optional alternative. It preserves Python types natively and avoids FITS column-type limitations.

Install the optional dependency:

```bash
pip install 'isoster[asdf]'
# or with uv:
uv add 'isoster[asdf]'
```

Usage:

```python
from isoster import isophote_results_to_asdf, isophote_results_from_asdf

# Save to ASDF (requires: pip install 'asdf>=3.0')
isophote_results_to_asdf(results, 'galaxy.asdf')

# Load from ASDF
loaded = isophote_results_from_asdf('galaxy.asdf')
```

### Astropy table export

- `isophote_results_to_astropy_tables`: returns the isophote list as one or more `astropy.table.Table` objects for downstream analysis without writing to disk.

## Troubleshooting

- Too many `stop_code=1`: inspect masks and clipping settings (`fflag`, `sclip`, `nclip`).
- Many `stop_code=-1` outward: reduce `maxsma` or relax `maxgerr` after validation.
- Many `stop_code=3`: increase `minsma` and inspect masked/edge regions.
- Unstable high-ellipticity fits: enable `use_eccentric_anomaly=True`.

## Stop Codes (Canonical Reference)

This section is the canonical stop-code reference for the current `isoster` implementation.

### Code Table

| Code | Current Meaning | Primary Trigger in Core Code | Typical Action |
|---|---|---|---|
| `0` | Success | Converged fit, or successful forced extraction | Keep |
| `1` | Too many flagged samples | `actual_points < total_points * (1.0 - fflag)` | Inspect mask/clipping; treat cautiously |
| `2` | Max-iteration fallback | Reached `maxit` without convergence criterion | Keep with caution; geometry is best-so-far |
| `3` | Too few points | `< 6` valid points for harmonic fit | Discard this radius |
| `-1` | Gradient failure | Gradient checks fail (or zero gradient) | Treat as boundary/failure |

### Notes by Path

- Regular mode (`fit_isophote`) emits `0`, `1`, `2`, `3`, `-1`.
- In regular `fit_image` growth, stop codes `0`, `1`, and `2` are treated as acceptable for outward/inward propagation.
- `fit_central_pixel` emits `0` for unmasked center and `-1` for masked center.
- Forced extraction (`extract_forced_photometry`) emits `0` or `3`.

### Gradient-Failure Detail (`-1`)

In outward fitting (`going_inwards=False`), gradient quality checks are applied with a two-strike rule:

- first exceedance sets an internal flag (`lexceed=True`),
- second exceedance sets stop code `-1`.

A direct `gradient == 0` also produces `-1`.

### Minimal Filtering Patterns

```python
good = [iso for iso in results["isophotes"] if iso["stop_code"] == 0]
usable = [iso for iso in results["isophotes"] if iso["stop_code"] in {0, 1, 2}]
failed = [iso for iso in results["isophotes"] if iso["stop_code"] < 0]
```

## Related Docs

- `docs/04-architecture.md`
- `docs/03-algorithm.md`
