# Multi-Band Isoster (Experimental)

> **Status: experimental, Stage-1.** API and output schema are subject to
> change. No CLI integration. ASDF I/O, ISOFIT, LSB auto-lock, and outer-
> center regularization are not supported in Stage 1. See
> `docs/agent/plan-2026-04-29-multiband-feasibility.md` for the locked
> design record.

## What it does

`isoster.multiband.fit_image_multiband` fits elliptical isophotes
simultaneously on multiple aligned, same-pixel-grid images of the same
target (e.g. HSC g/r/i/z/y coadds). It produces a **single shared
geometry per SMA** with **per-band intensities and per-band harmonic
deviations**. This replaces the traditional forced-photometry workflow
("fit one band, apply the geometry to others") with a joint fit where
every band contributes to the geometry.

The joint design matrix per ellipse, B bands, N kept samples:

```
[ 1_g 0    0    sin(φ) cos(φ) sin(2φ) cos(2φ) ] [I0_g]    [intens_g]
[ 0   1_r  0    sin(φ) cos(φ) sin(2φ) cos(2φ) ] [I0_r]  = [intens_r]
[ 0   0    1_i  sin(φ) cos(φ) sin(2φ) cos(2φ) ] [I0_i]    [intens_i]
                                              [A1   ]
                                              [B1   ]
                                              [A2   ]
                                              [B2   ]
```

Free parameters: `(5 + B)` per ellipse (per-band background `I0_b` plus
shared geometric harmonic coefficients). Solved once per iteration in
WLS or OLS mode. Per-band weights `w_b` enter as `√w_b` row scaling on
each band's block.

## Public API (placeholder — body filled when implementation lands)

```python
from isoster.multiband import fit_image_multiband, IsosterConfigMB

config = IsosterConfigMB(
    bands=["g", "r", "i", "z", "y"],
    reference_band="i",
    band_weights={"g": 1.0, "r": 1.0, "i": 1.0, "z": 1.0, "y": 1.0},
    harmonic_combination="joint",  # or "ref" for reference-band fallback
    sma0=10.0, maxsma=384.0, astep=0.1,
    debug=True, compute_deviations=True,
)
result = fit_image_multiband(
    images=[g_image, r_image, i_image, z_image, y_image],
    masks=object_mask,                    # single ndarray broadcast or list per band
    variance_maps=[g_var, r_var, ...],    # all-or-nothing
    config=config,
)
```

## Input contract (placeholder)

- `images`: `list[ndarray]` of length B, all of shape `(H, W)`.
- `masks`: `None`, single `(H, W)` boolean ndarray (broadcast to all
  bands), or `list[ndarray | None]` of length B. `None` per band means
  "no bad pixels in that band."
- `variance_maps`: all-or-nothing. Either `None` (full OLS), a single
  `(H, W)` ndarray (broadcast), or a `list[ndarray]` of length B.
- `bands`: list of strings, regex `^[A-Za-z][A-Za-z0-9_]*$`, no
  duplicates. Strings appear verbatim as column suffixes (`intens_g`,
  `intens_r`, ...).
- `reference_band`: string in `bands`. Used for diagnostics only; does
  not affect joint geometry.
- `band_weights`: dict (every band as key) or list (length B) of
  positive finite floats. Default uniform 1.0.

## Output schema (Schema 1)

`fit_image_multiband` returns a dict with `'isophotes'` (list of dicts,
one per SMA) and the multi-band-specific top-level keys:

```
result['bands']                : list[str]
result['multiband']            : True
result['harmonic_combination'] : 'joint' | 'ref'
result['reference_band']       : str
result['band_weights']         : dict[str, float]
result['variance_mode']        : 'wls' | 'ols'
```

Each isophote row carries shared columns and per-band-suffixed columns:

- Shared: `sma, x0, y0, eps, pa, x0_err, y0_err, eps_err, pa_err,
  stop_code, niter, rms, valid, use_eccentric_anomaly, ndata, nflag,
  tflux_e, tflux_c, npix_e, npix_c`.
- Per band `<b>`: `intens_<b>, intens_err_<b>, rms_<b>, a3_<b>,
  a3_err_<b>, b3_<b>, b3_err_<b>, a4_<b>, a4_err_<b>, b4_<b>,
  b4_err_<b>` (plus `grad_<b>, grad_error_<b>, grad_r_error_<b>` when
  `debug=True`).

The FITS writer (`isophote_results_to_fits`) uses the existing 3-HDU
layout (`PrimaryHDU`, `ISOPHOTES`, `CONFIG`), with the CONFIG HDU
recording multi-band parameters (`BANDS`, `REFERENCE_BAND`, `BAND_WEIGHTS`,
`HARMONIC_COMBINATION`, `VARIANCE_MODE`, `MULTIBAND`) alongside the usual
single-band fields.

**B=1 fallback:** when `len(bands) == 1`, `fit_image_multiband`
delegates to `fit_image` and returns the legacy single-band schema
unmodified, with an informational warning.

## Worked example (placeholder)

See `examples/example_asteris_denoised/run_isoster_multiband.py` for the
end-to-end Stage-1 demo: joint multi-band fit on the asteris denoised
HSC coadds of object 37484563299062823, with the existing i-band
single-band result loaded as a geometry-overlay reference.

## Caveats

- Inputs are assumed to be PSF-matched, or the user accepts PSF-mismatch
  artifacts on isophotes whose SMA is comparable to or smaller than the
  worst per-band PSF FWHM. No PSF handling in the driver.
- Sample-validity is shared across bands: a sample is dropped from the
  joint solve if any band's mask flags it, any band has NaN at that
  location, or any band's variance is non-positive. Edge cases (one
  problematic band dropping all bands' samples) are a known revisit
  item.
- Variance maps are all-or-nothing: either every band has one (full
  WLS) or no band has one (full OLS). Mixed mode is rejected.
- The driver runs no LSB auto-lock and no outer-center regularization
  in Stage 1. Run single-band isoster on the reference band if those
  features are needed.

## Related docs

- `docs/04-architecture.md` — multi-band module tree.
- `docs/agent/plan-2026-04-29-multiband-feasibility.md` — locked
  24-decision design record.
- `docs/02-configuration-reference.md` — single-band config (most
  multi-band fields share the same semantics).
