# Multi-Band Isoster (Experimental)

> **Status: experimental, Stage-1 shipped on `feat/multiband-feasibility`
> on 2026-04-30.** API and output schema are subject to change before
> the feature is merged to `main`. No CLI integration. ASDF I/O, ISOFIT,
> LSB auto-lock, and outer-center regularization are not supported in
> Stage 1. See `docs/agent/plan-2026-04-29-multiband-feasibility.md`
> for the locked design record (24 decisions captured from a structured
> interview before any code was written).

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

When ``IsosterConfigMB.fix_per_band_background_to_zero=True`` (D11
backport), the leading ``B`` per-band intercept columns are dropped
from the design matrix; the solve becomes a 4-column shared
``(A1, B1, A2, B2)`` system. Per-band ``intens_<b>`` is then reported
as the band's ring-mean intensity (IVW under WLS, simple mean under
OLS), and ``intens_err_<b>`` is the band's own SEM. Use this when
inputs have well-subtracted sky and the joint solver's free ``I0_b``
is being driven by sky residual rather than galaxy structure.
Mutually exclusive with ``harmonic_combination='ref'``.

## Performance

On the asteris denoised dataset (768×768 cutouts, 74 isophotes, all five
HSC bands), the joint multi-band fit runs in **~2× single-band wall
time end-to-end** (0.49 s for B=5 vs 0.25 s for single-band on i-band,
including FITS I/O). This is well within the Stage-1 quality bar of
≤2.5× and reflects two key optimizations: (1) the
``(N × (B+4))`` joint design matrix builder is numba-accelerated with
a NumPy fallback (``isoster/multiband/numba_kernels_mb.py``); (2) the
driver pre-resolves image / mask / variance arrays once per fit and
threads them through every per-iteration sampler call instead of
re-allocating per call. See decision D19 in the plan doc.

## Public API

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
  NaN/inf values are replaced with `1e30` (near-zero WLS weight);
  non-positive values are clamped to `1e-30` (near-infinite WLS weight)
  with a `RuntimeWarning` advising the user to mask those pixels
  instead. The sanitization mirrors single-band semantics; users who
  want bad pixels excluded from the fit must add them to `masks` —
  sanitization alone does not drop samples.
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

## Worked example

See `examples/example_asteris_denoised/run_isoster_multiband.py` for the
end-to-end Stage-1 demo: joint multi-band fit on the asteris denoised
HSC coadds of object 37484563299062823. The script loads all five HSC
bands of denoised cutouts, the existing object mask (built by
`build_object_mask.py` on the noisy cutout), and per-band uniform-
variance maps from the sigma-clipped sky RMS. It runs the joint
multi-band fit, writes a Schema-1 FITS result and the composite QA
PNG to `outputs/example_asteris_denoised/<id>/`, and prints a geometry
sanity check against the existing i-band single-band reference fit
(loaded from the same outputs directory).

Typical sanity-check output on object 37484563299062823:

```
geometry sanity vs i-band reference (median over valid rings):
  eps:  multi-band=0.123   i-band=0.118
  pa:   multi-band=139.65 deg   i-band=144.96 deg
```

The small offset between multi-band and i-band-only geometries is
expected: the joint fit pools harmonic-coefficient information across
all five bands, so the recovered `pa` is a band-weighted compromise
rather than the i-band-specific solution. The two are in family — a
real bias would manifest as a shift of several degrees or a larger
ellipticity discrepancy.

## Algorithm notes

### Joint design matrix (decision D2)

For one isophote with B bands and N kept samples, the joint solver
fits a single ``(B + 4)``-parameter least-squares system per iteration:

```
[ 1_g 0   0   sin(φ) cos(φ) sin(2φ) cos(2φ) ] [I0_g]    [intens_g]
[ 0   1_r 0   sin(φ) cos(φ) sin(2φ) cos(2φ) ] [I0_r]  = [intens_r]
[ 0   0   1_i sin(φ) cos(φ) sin(2φ) cos(2φ) ] [I0_i]    [intens_i]
                                            [A1   ]
                                            [B1   ]
                                            [A2   ]
                                            [B2   ]
```

Per-band weights `w_b` enter as `√w_b` row scaling on each band's
block; in WLS mode they compose with per-pixel inverse variance as
`w_b / variance_<b>(pixel)`. With B=1 and `w_b = 1` the joint solver
reduces to the existing single-band 5-parameter system bit-for-bit
(verified by `test_joint_solver_b1_matches_single_band_solver`).

### Combined gradient (decision D10)

The geometry-update math (Jedrzejewski 1987) requires a single radial
gradient. For multi-band the driver computes per-band gradients
separately and combines them with the same per-band weights:

```
gradient_joint = Σ_b w_b · grad_b / Σ_b w_b
σ²_joint       = Σ_b w_b² · σ_b² / (Σ_b w_b)²    (independent measurements)
```

Plugged into the standard geometry-update formulas; the gradient-error
gate (`maxgerr`) reads `σ_joint / |gradient_joint|`.

### Sample-validity rule (decision D9)

A sample on the ellipse is dropped from the joint solve if **any**
band's mask flags it, **any** band has NaN at that location, or
**any** band's variance is non-positive after sanitization. This
guarantees that every band's row block in the joint design matrix has
the same `N` samples, which the joint solve requires. Edge cases where
one bad band drops samples in all bands are a known revisit item.

### Sigma clipping (decision D9)

Each band is clipped independently against its own intensity
statistics; the surviving sample masks are AND-ed across bands and
applied uniformly. Reduces to single-band exactly when B=1.

## Testing

Multi-band tests live under `tests/multiband/`:

| Module | Cases | Coverage |
|---|---|---|
| `test_config_mb.py` | 27 | band-name regex, duplicate detection, reference-band membership, band_weights validation, integrator restriction, SMA/iteration consistency. |
| `test_sampling_mb.py` | 16 | B=1 numerical parity with single-band sampler, shared-validity (per-band masks, NaN, non-positive variance), all-masked degeneracy, mask broadcasting, variance all-or-nothing rejection, joint design matrix kernel parity. |
| `test_fitting_mb.py` | 12 | joint solver coefficient recovery, B=1 single-band parity, WLS exact covariance, band-weight scaling, per-band sigma clip + AND, fit_isophote_mb planted-galaxy recovery (within Stage-1 thresholds), too-few-points → stop_code=3, B=1 schema, ref-mode fallback, forced photometry, mixed-variance rejection. |
| `test_driver_mb.py` | 10 | B=1 → single-band delegation, B=2 end-to-end recovery, WLS variance-mode tagging, band_weights passthrough, ref-mode end-to-end, missing config, image / shape / variance-list mismatches, FIRST_FEW_ISOPHOTE_FAILURE. |
| `test_utils_mb.py` | 7 | per-band column presence, FITS round-trip, PrimaryHDU multi-band keywords, WLS round-trip, load_bands_from_hdus. |
| `test_plotting_mb.py` | 4 | composite QA renders without exception, with SB constants, missing-bands error, image-count mismatch. |

Run with:

```bash
uv run pytest tests/multiband/ -v
```

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
