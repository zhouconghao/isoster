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

1. `template_isophotes` provided -> template-based forced photometry.
2. `forced=True` -> fixed-geometry forced photometry at `forced_sma`.
3. Otherwise -> regular iterative fitting.

### Fixed-Geometry Forced Mode

```python
from isoster import fit_image
from isoster.config import IsosterConfig

config = IsosterConfig(
    forced=True,
    forced_sma=[5, 10, 20, 40, 80],
    x0=128.0,
    y0=128.0,
    eps=0.3,
    pa=0.2,
)

results = fit_image(image, mask=None, config=config)
```

### Template-Based Forced Photometry

```python
from isoster import fit_image
from isoster.config import IsosterConfig

config = IsosterConfig()

results_g = fit_image(image_g, mask=mask_g, config=config)
results_r = fit_image(
    image_r,
    mask=mask_r,
    config=config,
    template_isophotes=results_g["isophotes"],
)
```

## Key Configuration Options

### Sampling and Stability

- `use_eccentric_anomaly=True`: harmonic fitting in `psi` with geometry updates in `phi`.
- `permissive_geometry=True`: allows geometry propagation through weaker gradient diagnostics.
- `maxgerr`: controls tolerance for gradient relative error checks.

### Harmonics

- `compute_deviations=True`: enables higher-order deviation outputs (`a{n}`, `b{n}`).
- `harmonic_orders=[3, 4, ...]`: harmonic orders to compute.
- `simultaneous_harmonics=True`: true ISOFIT (Ciambur 2015) — fits higher-order harmonics jointly with geometry harmonics inside the iteration loop via a single extended design matrix. Accounts for cross-correlations and produces cleaner RMS estimates. Falls back to 5-param fit when insufficient sample points for the full design matrix.

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

- FITS save/load: `isophote_results_to_fits`, `isophote_results_from_fits`
- Astropy table export: `isophote_results_to_astropy_tables`

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
| `1` | Too many flagged samples | `actual_points < total_points * fflag` | Inspect mask/clipping; treat cautiously |
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

- `docs/spec.md`
- `docs/algorithm.md`
- `docs/todo.md`
- `docs/future.md`
