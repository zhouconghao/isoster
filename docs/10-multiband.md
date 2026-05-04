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

### Per-band intercept mode

The ``IsosterConfigMB.fit_per_band_intens_jointly`` boolean controls
how each band's per-isophote intercept ``I0_b`` (reported as
``intens_<b>``) is computed. The default ``True`` is the **coupled /
joint** mode described above: ``B`` per-band intercept columns sit in
the ``(B*N, B + 4)`` joint design matrix, so the per-band intercept
and the shared harmonic deformations are co-fit. On full rings the
sin/cos columns are mathematically orthogonal to the constant column
over ``[0, 2π]``, so the joint coefficient is numerically equivalent
to the per-band inverse-variance-weighted ring mean. On partial rings
(masking, loose validity), orthogonality breaks and the joint solve
absorbs the harmonic-shape coupling between bands when reporting each
band's intercept — this is where the joint mode adds information
across bands relative to a per-band ring statistic.

Setting ``fit_per_band_intens_jointly=False`` switches to a
**decoupled / ring-mean** mode: the leading ``B`` per-band intercept
columns are dropped from the design matrix; the solve becomes a
4-column shared ``(A1, B1, A2, B2)`` system. Per-band ``intens_<b>``
is then reported as the band's ring-mean intensity (IVW under WLS,
simple mean under OLS), independent of the harmonic fit, and
``intens_err_<b>`` is the band's own SEM. Use this when you want the
per-band intercept to be a clean ring statistic decoupled from the
harmonic block — for example, when you have already done a high-
quality sky subtraction upstream. Mutually exclusive with
``harmonic_combination='ref'`` (ref-mode bypasses the joint solve).

> **Note on integrator scope (Stage-2):** the multi-band path
> currently always uses an inverse-variance-weighted mean (WLS) or
> simple mean (OLS) for ``intens_<b>``, in both modes. The
> ``integrator`` config field does **not** affect ``intens_<b>``
> reporting at converged isophotes — it only applies to per-band
> gradient computation and the forced-photometry fallback. Median-
> integrator support for ``intens_<b>`` will land when the single-
> band integrator features are backported (Stage-3).

Renamed in Section 6 cleanup from the deprecated
``fix_per_band_background_to_zero`` (with inverted polarity:
``fix_per_band_background_to_zero=True`` ↔
``fit_per_band_intens_jointly=False``). The old field name is
rejected at config construction with a clear error pointing at the
new name; FITS files written by the new code carry only the new
name. The multi-band path is pre-production so we take a clean
break with no auto-translation.

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
result['bands']                       : list[str]
result['multiband']                   : True
result['harmonic_combination']        : 'joint' | 'ref'
result['reference_band']              : str
result['band_weights']                : dict[str, float]
result['variance_mode']               : 'wls' | 'ols'
result['fit_per_band_intens_jointly'] : bool
result['loose_validity']              : bool
result['multiband_higher_harmonics']  : 'independent' | 'shared' |
                                        'simultaneous_in_loop' |
                                        'simultaneous_original'
result['harmonic_orders']             : list[int]    # default [3, 4]
result['harmonics_shared']            : bool          # True iff non-'independent'
```

Optional, present only when the caller post-processed with
``subtract_outermost_sky_offset``:

```
result['sky_offsets']  : dict[str, float]   # per-band offset that was subtracted
```

Each isophote row carries shared columns and per-band-suffixed columns:

- Shared: `sma, x0, y0, eps, pa, x0_err, y0_err, eps_err, pa_err,
  stop_code, niter, rms, valid, use_eccentric_anomaly, ndata, nflag,
  tflux_e, tflux_c, npix_e, npix_c`.
- Per band `<b>`: `intens_<b>, intens_err_<b>, rms_<b>,
  n_valid_<b>` (the last is the per-band surviving-sample count after
  sigma clipping; under shared validity it equals the shared `ndata`).
- Per band per order `<n>`: `a<n>_<b>, b<n>_<b>, a<n>_err_<b>,
  b<n>_err_<b>` for each `n` in ``harmonic_orders`` (default `[3, 4]`,
  extensible to `[3, 4, 5, 6]` etc.). Under non-``'independent'``
  ``multiband_higher_harmonics`` modes these per-band columns all
  carry the **identical shared value** across bands at every isophote
  (the joint refit / in-loop solve produces one shared coefficient per
  order). Per-band Bender normalization at plotting time scales the
  shared raw value by ``-1/(sma·|dI/da_b|)``, which still produces
  band-distinct curves because per-band gradients differ.
- Debug-only per band: `grad_<b>, grad_error_<b>, grad_r_error_<b>`
  when `debug=True`.

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

### Loose validity (D9 backport)

When ``IsosterConfigMB.loose_validity=True`` (default ``False``), the
shared-validity AND is relaxed: each band keeps its own surviving
samples and the joint design matrix becomes block-diagonal in the
per-band intercept columns. The shared geometric coefficients
``(A1, B1, A2, B2)`` are still constrained jointly. A band that falls
below ``loose_validity_min_per_band_count`` (absolute count, default 6)
or ``loose_validity_min_per_band_frac`` (fraction of attempted samples,
default 0.2) at a given isophote is dropped from the joint solve at
that isophote — its ``intens_<b>``, ``intens_err_<b>``, and harmonic
columns are set to NaN, and the surviving bands still constrain the
shared geometry. The whole isophote is marked ``stop_code=3`` only
when fewer than 2 bands survive.

A new per-isophote column ``n_valid_<b>`` reports each band's actual
surviving-sample count (after the per-band sigma clip). The QA figure
auto-adds a small panel showing ``n_valid_<b> / n_attempted`` per band
when ``loose_validity=True``.

The optional knob ``loose_validity_band_normalization`` controls how
per-band sample counts feed the joint solve and the combined gradient:

- ``"none"`` (default): each band's row block contributes
  proportionally to its own ``N_b``; ``w_b`` multiplies every row;
  combined gradient is ``Σ w_b · grad_b / Σ w_b`` over surviving
  bands. Bands with more kept samples dominate.
- ``"per_band_count"``: each band's row block is renormalized by
  ``√(1/N_b)`` so its total contribution equals ``w_b`` regardless of
  ``N_b``; combined gradient is weighted by ``(w_b · N_b)``. Restores
  the user's "this band matters this much" mental model when
  per-band masks differ. Requires ``loose_validity=True``.

Loose validity composes cleanly with both ``harmonic_combination='ref'``
and ``fit_per_band_intens_jointly=False``.

### Higher-order harmonics modes (Section 6, locked 2026-05-02)

The ``IsosterConfigMB.multiband_higher_harmonics`` enum controls how
higher-order (n ≥ 3) harmonic deviations ``(A_n, B_n)`` are fit. The
default ``'independent'`` reproduces Stage-1 behavior bit-for-bit; the
other three values share higher-order coefficients across bands in
different ways. ``IsosterConfigMB.harmonic_orders`` (default ``[3, 4]``,
extensible to ``[3, 4, 5, 6]`` etc.) selects which orders are included
in any of the three non-``'independent'`` modes.

| Value | Where higher orders are solved | Cross-band coupling | Refits |
|-------|-------------------------------|---------------------|--------|
| ``independent`` (default) | Post-hoc, per band, per order via the single-band ``compute_deviations`` solver | None — each band independent | a3_b, b3_b, a4_b, b4_b ... |
| ``shared`` | Post-hoc joint solve over residuals | Shared (A_n, B_n) for n ≥ 3 across bands | (A_n, B_n) for n ≥ 3 only |
| ``simultaneous_in_loop`` | In every iteration's joint solve | Shared (A_n, B_n) for ALL n | (A1, B1, A2, B2, A_n, B_n) every iteration |
| ``simultaneous_original`` | 5-param in loop; one wider post-hoc joint solve over all orders | Shared (A_n, B_n) for ALL n | One post-hoc solve refits all orders |

#### `independent` (default)

Stage-1 behavior: at convergence, the driver calls
``_attach_per_band_harmonics`` which runs an independent
``compute_deviations`` per band per order. Each band's a_n / b_n
reflects its own ring's intensity, gradient, and noise. Per-band
``a_n_<b>`` and ``b_n_<b>`` typically differ across bands.

#### `shared` (NEW DEVELOPMENT)

Replaces the per-band post-hoc step with **one joint refit** of
higher-order coefficients shared across bands. The geometry-loop
values for ``(A1, B1, A2, B2)`` and per-band ``I0_b`` are frozen.
Design matrix is the smallest possible: ``(B·N, 2·L)`` where
``L = len(harmonic_orders)``. Per-band ``a_n_<b>`` columns all carry
the same shared value at every isophote. Errors come from the joint
covariance; per-band ``a_n_err_<b>`` is the same across bands too.

Astrophysical motivation: a multi-band joint fit is meant to produce
a single 1-D profile of the galaxy across bands; if the higher-order
geometric deviation genuinely differed band-to-band you should be
running ``harmonic_combination='ref'`` plus forced photometry instead.

#### `simultaneous_in_loop` and `simultaneous_original` (RECOVERED FEATURE)

Multi-band lifts of the single-band ``IsosterConfig.simultaneous_harmonics``
/ ``isofit_mode`` features. Both extend the joint design matrix to
``(B·N, B + 4 + 2·L)`` with shared higher-order columns; the
difference is when the wider solve fires:

- ``simultaneous_in_loop``: every iteration's joint solve uses the
  wider matrix. Geometry update reads ``coeffs[B..B+3]`` exactly as
  before — coefficient layout is unchanged at those indices.
- ``simultaneous_original``: standard 5-param iteration loop, then ONE
  wider joint solve over all orders runs after convergence (Ciambur
  2015 original variant).

Both modes emit a ``UserWarning`` at config construction citing
single-band benchmark concerns; validate carefully on PGC006669 and
asteris before trusting.

#### Validators

- ``shared`` / ``simultaneous_*`` × ``harmonic_combination='ref'``:
  hard-error. Ref-mode bypasses the joint solver entirely.
- ``shared`` / ``simultaneous_*`` × ``fit_per_band_intens_jointly=False``:
  silently allowed. Compose cleanly with the ring-mean intercept mode.
- ``shared`` / ``simultaneous_*`` × ``loose_validity=True``: silently
  allowed. The jagged builders ``build_joint_design_matrix_jagged`` and
  ``build_joint_design_matrix_jagged_higher`` carry the higher-order
  columns through the loose-validity path.

#### Performance bar (Stage-1 D17 / Section 6.1 Q11)

All four modes must stay within ≤ 2.5× single-band wall time on the
asteris perf benchmark. Measured on the B=5 768² cutout (2026-05-03):

| Configuration | Median (s) | Ratio vs SB |
|---|---:|---:|
| singleband (i-band) | 0.209 | 1.00× |
| multiband independent | 0.293 | 1.41× |
| multiband shared | 0.279 | 1.34× |
| multiband simultaneous_in_loop | 0.317 | 1.52× |
| multiband simultaneous_original | 0.272 | **1.30×** |

All four PASS. ``simultaneous_original`` is the cheapest non-default
mode because one post-hoc joint solve replaces ``B × L`` separate
``compute_deviations`` calls.

### Per-band intercept mode revisited (Section 6.12)

The boolean ``IsosterConfigMB.fit_per_band_intens_jointly`` (default
``True``, renamed from the deprecated
``fix_per_band_background_to_zero=False`` in the Section 6.12 cleanup)
governs how each isophote's per-band intercept ``I0_b`` is computed.
See the [Per-band intercept mode](#per-band-intercept-mode) section
above for the full mechanism description.

The two modes are **numerically equivalent on full rings** because
sin(nφ) and cos(nφ) are orthogonal to the constant column over [0, 2π].
They diverge only on partial rings (loose validity, masked sectors),
where the joint solve absorbs harmonic-shape coupling between bands.
This is when the joint mode adds information across bands relative to
a per-band ring statistic.

Sky correction is **not** part of the fit. The optional post-process
``subtract_outermost_sky_offset(result, n_outer=N)`` lives in
``isoster.multiband.plotting_mb`` and operates only on already-fit
results; it never modifies the fit. The library never invokes it
implicitly. Demo scripts call it explicitly between FITS write and
QA plot generation; raw FITS files always preserve the unmodified
``intens_<b>``.

### Outer-region regularization — damping mode (Stage-3 Stage-B)

Backport of the single-band ``outer_reg_*`` family in **damping**
mode. Use ``use_outer_center_regularization=True`` to enable a soft
Tikhonov-style step shrink in the LSB regime: per-iteration geometry
updates ``δ_param`` are multiplied by ``(1 - α(sma))``, where ``α``
ramps from 0 below ``outer_reg_sma_onset`` to a saturation set by
``outer_reg_strength``. Saturated clipped jumps in PA, ellipticity,
and center are suppressed; the fit still walks to its data-preferred
geometry — there is no pull toward the reference. The full Tikhonov
solver mode (with reference pull) lands in Stage E.

Six fields, mirroring single-band semantics:

| Field | Default | Notes |
|---|---|---|
| ``use_outer_center_regularization`` | ``False`` | Master toggle. |
| ``outer_reg_sma_onset`` | ``50.0`` | Sigmoid midpoint (pixels). |
| ``outer_reg_strength`` | ``2.0`` | Saturation amplitude of the ramp. |
| ``outer_reg_weights`` | ``{center: 1, eps: 1, pa: 1}`` | Per-axis weights; default ``{1,1,1}`` damps all four geometry parameters and prevents the selector-asymmetry failure mode. |
| ``outer_reg_sma_width`` | ``None`` (auto = ``0.4 * onset``) | Sigmoid slope width. |
| ``outer_reg_ref_sma_factor`` | ``2.0`` | Inward isophotes within ``sma0 × factor`` feed the flux-weighted reference. |

The ``outer_reg_mode`` field is currently a one-value
``Literal["damping"]``. Stage E will widen it to
``Literal["damping", "solver"]`` and add the full Tikhonov
ref-pull. Users who set ``outer_reg_mode='solver'`` today get a
clear pydantic Literal validation error.

**Auto-enable.** With ``use_outer_center_regularization=True`` and
default ``geometry_convergence=False``, the validator emits a
``UserWarning`` and flips ``geometry_convergence=True`` automatically.
Without geometry convergence the fit runs to ``maxit`` on outer
isophotes with no change to recorded geometry — pure cost.
Suppress the warning by setting ``geometry_convergence=True``
explicitly.

**Sanity warnings.** The validator also emits warnings for:

- ``outer_reg_sma_onset < sma0`` (penalty fires from the first
  outward step; typical onset ~ ``sma0 * 3``).
- ``minsma >= sma0`` (no inward isophotes → reference falls back to
  the anchor's geometry).
- All ``outer_reg_weights`` zero (feature is inert).
- ``fix_<axis>=True`` with ``outer_reg_weights[<axis>]>0`` (axis is
  frozen so the penalty is identically zero).

**Reference geometry.** Built once per fit by
``_build_outer_reference_mb`` in the driver. Flux-weighted mean
(weights = ``intens_<reference_band>``) over inward isophotes with
``sma <= sma0 * outer_reg_ref_sma_factor`` plus the anchor; ``pa``
uses a circular mean on ``2*pa`` so axis-like angles around 0 and π
average sensibly. Falls back to the anchor's geometry if the inward
sweep is empty or all candidates fail finiteness checks.

**Asteris benchmark.**
``benchmarks/multiband/bench_outer_reg_damping_asteris.py`` compares
three configs (no outer-reg, center-only damper, all-axes damper) on
HSC g/r/i/z/y. Headline results (n_repeats=3, B=5, 768²):

| configuration | median (s) | ratio vs SB | outer eps MAD | outer pa MAD |
|---|---:|---:|---:|---:|
| singleband (i-band) | 0.333 | 1.00× | — | — |
| baseline (no outer-reg) | 0.425 | 1.28× | 1.17e-01 | 9.19e-02 |
| damping center-only | 0.436 | 1.31× | 5.39e-02 | 0.00 |
| damping all axes (default) | 0.388 | **1.16×** | 9.93e-06 | 0.00 |

All PASS the ≤ 2.5× single-band bar. Damping-vs-baseline overhead
is **0.91×** — the all-axes damper is faster than baseline because
the auto-enabled ``geometry_convergence`` lets damped fits exit
earlier than the harmonic-only criterion does in the LSB regime.

> **Real-data caveat for the all-axes default (2026-05-04).** The
> all-axes ``{center: 1, eps: 1, pa: 1}`` default mirrors single-band
> and looks excellent on aggregate scatter metrics (eps MAD ≈ 1e-5
> on asteris, pa MAD = 0 on both targets). But the QA mosaics in
> ``outputs/benchmark_multiband/outer_reg_damping_{asteris,pgc}/``
> show that on galaxies with **genuine outer-disc geometry
> evolution** the all-axes damper is too aggressive: ``alpha → 1``
> for eps and pa above the onset, pinning the outer-isophote shape
> to the inner reference and ignoring real structural change. On
> PGC006669 the outer disc inherits the bar's PA / ellipticity; on
> asteris the isophotal shape freezes before the LSB envelope
> rounds off. The ``pa MAD = 0`` is the geometric signature of
> "PA frozen," not "PA tracked." A bias metric vs a free-fit
> reference (not just scatter) is needed to distinguish a healthy
> damper from over-pinning.
>
> The implementation is correct and matches single-band semantics.
> The defaults — copied verbatim from single-band — were tuned on
> HSC BCG fits where the outer envelope is genuinely round and not
> evolving. For galaxies with bars, structural transitions, or
> evolving disc geometry, prefer ``outer_reg_weights={center: 1,
> eps: 0, pa: 0}`` (center-only) or lower ``outer_reg_strength``.
> A multi-band-specific re-default and a strength sweep on PGC are
> deferred to a future session; the journal entry
> ``docs/agent/journal/2026-05-04_stage_b_outer_reg_damping.md``
> records the open questions.

## Testing

Multi-band tests live under `tests/multiband/` (152 total, all green):

| Module | Cases | Coverage |
|---|---|---|
| `test_config_mb.py` | 60 | band-name regex, duplicate detection, reference-band membership, band_weights validation, integrator restriction, SMA/iteration consistency, loose-validity field defaults + normalization compatibility, multiband_higher_harmonics enum (4 values + parametrized variants), harmonic_orders validator (empty / <3 / duplicates / unique-sort), simultaneous_* experimental warning, shared-mode no-warn, ring-mean intercept × ref-mode incompatibility, ring-mean intercept × loose-validity composition. |
| `test_sampling_mb.py` | 19 | B=1 numerical parity with single-band sampler, shared-validity (per-band masks, NaN), variance sanitization (NaN/inf → 1e30, non-positive clamped + warning), all-masked degeneracy, mask broadcasting, variance all-or-nothing rejection, joint design matrix kernel parity. |
| `test_fitting_mb.py` | 23 | joint solver coefficient recovery, B=1 single-band parity, WLS exact covariance, band-weight scaling, per-band sigma clip + AND, fit_isophote_mb planted-galaxy recovery, too-few-points → stop_code=3, B=1 schema, ref-mode fallback, forced photometry, mixed-variance rejection, ring-mean intercept mode (fit_per_band_intens_jointly=False), ref-mode error scaling, loose-validity band drop / n_valid columns / normalization / combinations with ref + ring-mean intercept, legacy field-name rejection. |
| `test_higher_harmonics.py` | 20 | All four enum values; per-band-equality of shared higher orders; planted-m=4 recovery (shared + simultaneous_*); loose-validity × simultaneous (jagged kernel); simultaneous_original ≈ shared within tolerance; experimental UserWarning; FITS round-trip; harmonic_orders=[3,4,5,6] writes 16 expected per-band columns; D16 normalization separates curves under sharing; direct solver-level fit_simultaneous_joint{,_loose} planted-recovery. |
| `test_driver_mb.py` | 17 | B=1 → single-band delegation (incl. variance/mask unwrap), B=2 end-to-end recovery, WLS variance-mode tagging, band_weights passthrough, ref-mode end-to-end, missing config, image / shape / variance-sequence-or-tuple / mask-list / non-sequence mismatches, FIRST_FEW_ISOPHOTE_FAILURE. |
| `test_utils_mb.py` | 8 | per-band column presence, FITS round-trip, PrimaryHDU multi-band keywords, WLS round-trip, loose-validity n_valid round-trip, load_bands_from_hdus. |
| `test_plotting_mb.py` | 5 | composite QA renders without exception, with SB constants, missing-bands error, image-count mismatch, loose-validity n_valid panel rendered. |

Run with:

```bash
uv run pytest tests/multiband/ -v
```

## Caveats

- Inputs are assumed to be PSF-matched, or the user accepts PSF-mismatch
  artifacts on isophotes whose SMA is comparable to or smaller than the
  worst per-band PSF FWHM. No PSF handling in the driver.
- Sample-validity is shared across bands by default: a sample is
  dropped from the joint solve if any band's mask flags it, any band
  has NaN at that location, or any band's variance is non-positive.
  Set ``IsosterConfigMB.loose_validity=True`` to relax this — see the
  "Loose validity (D9 backport)" subsection above for the per-band-drop
  semantics, the new ``n_valid_<b>`` column, and the optional
  ``loose_validity_band_normalization`` knob.
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
