# Configuration Reference

## Overview

`IsosterConfig` is a Pydantic model that defines all tunable parameters for the isophote
fitting algorithm. It contains 42 parameters organized into 12 functional groups covering
geometry initialization, fitting control, quality control, output options, and advanced
algorithm modes.

Source: `isoster/config.py` (`IsosterConfig`)

Pydantic validation enforces type constraints, value ranges, and cross-parameter
consistency (e.g., `minit <= maxit`, `forced=True` requires `forced_sma`,
`integrator='adaptive'` requires `lsb_sma_threshold`, all `harmonic_orders >= 3`).

---

## Parameter Reference

### 1. Geometry Initialization

Initial ellipse geometry. When `x0` or `y0` is `None`, the image center is used.

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `x0` | `None` | `Optional[float]` | Initial center x coordinate. `None` uses `image.shape[1] / 2`. |
| `y0` | `None` | `Optional[float]` | Initial center y coordinate. `None` uses `image.shape[0] / 2`. |
| `eps` | `0.2` | `float` (0 <= eps < 1) | Initial ellipticity. |
| `pa` | `0.0` | `float` | Initial position angle in radians. |

**Interaction notes:**

- These values seed the first isophote at `sma0`. Subsequent isophotes inherit geometry
  from the previous converged fit.
- In forced mode (`forced=True`), these values are used as the single fixed geometry for
  all SMA values.
- When `fix_center=True`, `x0` and `y0` remain constant throughout the run.
  Similarly for `fix_pa` and `fix_eps`.

---

### 2. SMA Control

Semi-major axis stepping and bounds.

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `sma0` | `10.0` | `float` (> 0) | Starting semi-major axis length in pixels. |
| `minsma` | `0.0` | `float` (>= 0) | Minimum SMA to fit. Set > 0 to skip central region. |
| `maxsma` | `None` | `Optional[float]` (> 0) | Maximum SMA. `None` uses `max(image_height, image_width) / 2`. |
| `astep` | `0.1` | `float` (> 0) | Step size for SMA growth. |
| `linear_growth` | `False` | `bool` | If `True`, `sma_next = sma + astep`. If `False`, `sma_next = sma * (1 + astep)`. |

**Interaction notes:**

- The driver fits at `sma0` first, then grows outward to `maxsma`, then inward from
  `sma0` to `max(minsma, 0.5)`.
- When `minsma <= 0.0`, a central pixel (SMA=0) result is prepended automatically.
- Geometric growth (default) gives denser sampling near the center and sparser sampling
  at large radii. Linear growth gives uniform spacing.
- Validated: `maxsma > minsma` when both are specified.

---

### 3. Fitting Control

Iteration limits and convergence criteria.

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `maxit` | `50` | `int` (> 0) | Maximum iterations per isophote. |
| `minit` | `10` | `int` (> 0) | Minimum iterations before convergence check. |
| `conver` | `0.05` | `float` (> 0) | Convergence threshold: `max_harmonic_amplitude / rms`. |
| `convergence_scaling` | `'sector_area'` | `str` | Scale convergence threshold with SMA. Options: `'none'`, `'sector_area'`, `'sqrt_sma'`. |

**Interaction notes:**

- Convergence is declared when the largest harmonic amplitude drops below
  `conver * rms * scale_factor`, where `scale_factor` depends on `convergence_scaling`.
- `'sector_area'` (default) matches photutils behavior and eliminates most stop=2
  failures at outer isophotes. The scale factor grows with SMA.
- `'none'` uses a constant threshold (legacy behavior).
- `'sqrt_sma'` multiplies by `sqrt(sma)`.
- `minit` iterations always run before the convergence criterion is checked.
- Validated: `minit <= maxit`.

---

### 4. Geometry Update Control

Controls how geometry corrections are applied during iteration.

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `geometry_damping` | `0.7` | `float` (0 < d <= 1) | Damping factor for geometry corrections. Each correction is multiplied by this value. |
| `geometry_update_mode` | `'largest'` | `str` | `'largest'`: update only the parameter with the largest harmonic amplitude (coordinate descent). `'simultaneous'`: update all four parameters each iteration. |
| `geometry_convergence` | `False` | `bool` | Enable secondary convergence based on geometry stability. |
| `geometry_tolerance` | `0.01` | `float` (> 0) | Threshold for geometry convergence metric. |
| `geometry_stable_iters` | `3` | `int` (>= 2) | Consecutive stable iterations required for geometry convergence. |
| `permissive_geometry` | `False` | `bool` | Enable photutils-style "best effort" geometry updates that continue even from failed fits. |

**Interaction notes:**

- `geometry_damping=0.7` is validated across 20 Huang2013 galaxies. Use `0.5` with
  `geometry_update_mode='simultaneous'` for stability. Use `1.0` for undamped legacy
  behavior.
- `'largest'` mode (default) matches isofit/photutils coordinate-descent behavior.
  `'simultaneous'` updates x0, y0, PA, and eps every iteration, typically converging
  in fewer iterations but requiring lower damping.
- Geometry convergence declares success when
  `max(|delta_eps|, |delta_pa/pi|, |delta_x0/sma|, |delta_y0/sma|) < geometry_tolerance`
  for `geometry_stable_iters` consecutive iterations, even if the harmonic criterion is
  not met. This is a supplementary criterion useful for challenging outer isophotes.
- `permissive_geometry=True` always updates geometry from the latest fit, even when the
  fit failed. This prevents cascading failures where one bad isophote poisons all
  subsequent geometry propagation. `None` gradient errors are treated as acceptable.
- Both `geometry_update_mode` options respect `fix_center`, `fix_pa`, and `fix_eps`.

---

### 5. Quality Control

Sigma clipping and data quality thresholds.

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `sclip` | `3.0` | `float` (> 0) | Symmetric sigma clipping threshold. |
| `nclip` | `0` | `int` (>= 0) | Number of sigma clipping iterations. `0` disables clipping. |
| `sclip_low` | `None` | `Optional[float]` | Lower (negative) sigma clipping threshold. Overrides `sclip` on the low side when set. |
| `sclip_high` | `None` | `Optional[float]` | Upper (positive) sigma clipping threshold. Overrides `sclip` on the high side when set. |
| `fflag` | `0.5` | `float` (0-1) | Maximum fraction of flagged data points (masked + sigma-clipped). Exceeding this triggers stop_code=1. |
| `maxgerr` | `0.5` | `float` (> 0) | Maximum relative error in the local radial gradient. Exceeding this triggers stop_code=-1. |

**Interaction notes:**

- `fflag` only counts sigma-clipped points as flagged. Mask-excluded points are removed
  during sampling before the harmonic fit.
- For high-ellipticity galaxies (eps > 0.6), relax `maxgerr` to 1.0-1.2 to prevent
  excessive gradient-error failures.
- `sclip_low` and `sclip_high` enable asymmetric clipping, useful when bright sources
  contaminate one tail of the intensity distribution.
- Sigma clipping is applied every iteration within `fit_isophote` and once in forced
  photometry extraction.

---

### 6. Constraints

Fix individual geometry parameters during iterative fitting.

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `fix_center` | `False` | `bool` | Fix center coordinates (x0, y0) during fitting. |
| `fix_pa` | `False` | `bool` | Fix position angle during fitting. |
| `fix_eps` | `False` | `bool` | Fix ellipticity during fitting. |

**Interaction notes:**

- Fixed parameters retain their initial values (from `x0`/`y0`/`eps`/`pa` config or
  inherited from the previous isophote) throughout all iterations.
- When all three are `True`, fitting becomes pure photometry extraction with
  predetermined geometry (similar to forced mode but still iterates for convergence).
- Constraints are respected by both `'largest'` and `'simultaneous'`
  `geometry_update_mode`.
- When `compute_cog=True`, fixing all three affects crossing detection logic in the
  curve-of-growth module.

---

### 7. Eccentric Anomaly

Sampling mode for high-ellipticity isophotes.

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `use_eccentric_anomaly` | `False` | `bool` | Use eccentric anomaly (psi) for uniform arc-length ellipse sampling. |

**Interaction notes:**

- When `True`, sampling is uniform in eccentric anomaly psi, which provides uniform
  arc-length coverage on the ellipse. Recommended for eps > 0.3.
- Harmonics are fitted in psi-space; geometry updates are always performed in phi-space
  (position angle) to maintain correct geometric interpretation.
- Affects both regular fitting and forced/template-forced photometry extraction.
- `build_isoster_model()` auto-detects whether eccentric anomaly was used from the
  isophote dicts (presence of `'use_eccentric_anomaly'` key). Can be overridden
  explicitly.

---

### 8. Higher-Order Harmonics / ISOFIT

Controls for simultaneous harmonic fitting following Ciambur (2015).

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `simultaneous_harmonics` | `False` | `bool` | Enable true ISOFIT simultaneous harmonic fitting. |
| `harmonic_orders` | `[3, 4]` | `List[int]` | Harmonic orders to fit. All values must be >= 3. |
| `isofit_mode` | `'in_loop'` | `str` | ISOFIT algorithm variant: `'in_loop'` or `'original'`. |

**Interaction notes:**

- When `simultaneous_harmonics=False` (default), higher-order harmonics (a3, b3, a4, b4,
  etc.) are fitted post-hoc after geometry convergence. Orders 1-2 are always used
  internally for geometry updates.
- When `simultaneous_harmonics=True`, higher-order harmonics are fitted jointly with
  geometry harmonics inside the iteration loop. This accounts for cross-correlations and
  yields cleaner RMS estimates.
- `isofit_mode='in_loop'` fits all orders simultaneously inside the loop (more
  aggressive). `isofit_mode='original'` matches Ciambur (2015): 5-parameter fit inside
  the loop for geometry, then simultaneous post-hoc fit for higher orders. `isofit_mode`
  is only meaningful when `simultaneous_harmonics=True`.
- Falls back to 5-parameter fit when the number of sample points is less than
  `2 * len(harmonic_orders) + 5` (the minimum for the extended design matrix).
  A `RuntimeWarning` is emitted on first fallback per isophote.
- ISOFIT coefficients are stored as `[I_0, A_1, B_1, A_2, B_2, A_n1, B_n1, ...]` where
  the k-th higher order occupies indices `5+2k` and `5+2k+1`.
- Validated: all entries in `harmonic_orders` must be >= 3.
- ISOFIT overhead is approximately 25-35% compared to the default path.

---

### 9. Central Regularization

Geometry regularization for low-S/N central regions.

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `use_central_regularization` | `False` | `bool` | Enable geometry regularization at low SMA. |
| `central_reg_sma_threshold` | `5.0` | `float` (> 0) | SMA threshold in pixels. Regularization strength decays as `exp(-(sma/threshold)^2)`. |
| `central_reg_strength` | `1.0` | `float` (>= 0) | Maximum regularization strength at SMA=0. Range: 0 (none) to 10 (strong). |
| `central_reg_weights` | `{'eps': 1.0, 'pa': 1.0, 'center': 1.0}` | `dict` | Relative weights for regularization penalties on ellipticity, PA, and center. |

**Interaction notes:**

- Regularization penalizes large geometry changes relative to the previous isophote,
  stabilizing fits in noisy central regions.
- The penalty decays to effectively zero beyond approximately 3x the threshold SMA.
- Per-parameter weights allow selective regularization (e.g., fix center strongly but
  allow PA to vary by setting `{'eps': 1.0, 'pa': 0.1, 'center': 5.0}`).
- Requires a previous isophote for comparison; has no effect on the first isophote
  at `sma0`.

---

### 10. Output and Features

Toggle optional output quantities.

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `compute_errors` | `True` | `bool` | Calculate parameter errors (x0_err, y0_err, eps_err, pa_err). |
| `compute_deviations` | `True` | `bool` | Calculate higher-order harmonic deviations (a3, b3, a4, b4, etc.) post-hoc. |
| `full_photometry` | `False` | `bool` | Calculate aperture flux integration metrics (tflux_e, tflux_c, npix_e, npix_c). |
| `compute_cog` | `False` | `bool` | Calculate curve-of-growth photometry (cog, cog_annulus, area_annulus, flag_cross, flag_negative_area). |
| `debug` | `False` | `bool` | Include debug fields (ndata, nflag, grad, grad_error, grad_r_error) and enable verbose output. |

**Interaction notes:**

- `compute_deviations` controls the post-hoc harmonic deviation calculation. This is
  independent of `simultaneous_harmonics`, which controls joint fitting inside the
  iteration loop.
- When `simultaneous_harmonics=True`, deviations for the specified `harmonic_orders` are
  computed as part of the joint fit. `compute_deviations=True` additionally computes
  deviations post-hoc (which may differ slightly from the simultaneous values).
- See the "Photometry Outputs" section below for details on `full_photometry` vs
  `compute_cog`.

---

### 11. Integration Mode

Controls how the representative intensity is derived from sampled points.

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `integrator` | `'mean'` | `str` | Integration method: `'mean'`, `'median'`, or `'adaptive'`. |
| `lsb_sma_threshold` | `None` | `Optional[float]` (> 0) | SMA threshold in pixels for switching to median in adaptive mode. |

**Interaction notes:**

- `'mean'` reports the harmonic-fit DC term (the fitted mean `y0_fit`, not `np.mean`).
- `'median'` reports `np.median(intens)` of the sigma-clipped samples.
- `'adaptive'` uses `'mean'` for SMA <= threshold, `'median'` for SMA > threshold.
  Inner high-S/N regions benefit from harmonic fit precision; outer low-S/N regions
  benefit from median robustness.
- Validated: `lsb_sma_threshold` must be provided when `integrator='adaptive'`.
- See the "Integration Modes" section below for architectural details.

---

### 12. Forced Mode

Extract photometry without iterative geometry fitting.

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `forced` | `False` | `bool` | Enable pure forced photometry mode using a single fixed geometry. |
| `forced_sma` | `None` | `Optional[List[float]]` | List of SMA values to extract. Required when `forced=True`. |

**Interaction notes:**

- When `forced=True`, the geometry from `x0`, `y0`, `eps`, `pa` is applied at every
  SMA in `forced_sma` without any iterative fitting.
- Validated: `forced_sma` must be provided and non-empty when `forced=True`.
- For variable geometry across SMA values, use `template_isophotes` instead (see next
  section).
- `template_isophotes` takes priority over `forced=True` when both are set.

---

## Forced Photometry Modes

Isoster provides two forced photometry pathways, both accessible through `fit_image()`.

### Single Fixed Geometry (`forced=True`)

Uses a single geometry (x0, y0, eps, pa) from the config for all SMA values. The SMA
list is explicitly provided via `forced_sma`. This mode is a simple photometry
extraction and does not iterate.

### Template-Based Variable Geometry (`template_isophotes`)

Uses per-SMA geometry from a previous isoster run. Each isophote in the template
provides its own x0, y0, eps, and pa, enabling variable geometry along the radial
profile. This is the IRAF `ellipse` forced-mode equivalent for multiband analysis.

```python
# Fit reference band
results_g = isoster.fit_image(image_g, mask_g, config)

# Apply g-band geometry to r-band
results_r = isoster.fit_image(image_r, mask_r, config,
                              template_isophotes=results_g['isophotes'])
```

### Comparison

| Aspect | `forced=True` | `template_isophotes` |
|--------|---------------|----------------------|
| Geometry source | Config scalars (x0, y0, eps, pa) | Per-SMA dicts from prior fit |
| Center | Single fixed value | Varies per SMA |
| Ellipticity / PA | Single fixed value | Varies per SMA |
| SMA list | Explicit via `forced_sma` | Inherited from template |
| Primary use case | Quick extraction at known geometry | Multiband color profiles |

### Config parameters silently dropped in forced modes

Both forced modes bypass the iterative fitting loop entirely. The following config
parameters have no effect in either forced mode:

- `sclip_low`, `sclip_high` (only symmetric `sclip` is used)
- `fflag` (no flagging check)
- `full_photometry` (aperture photometry not computed)
- `compute_errors` (all errors set to 0.0)
- `compute_deviations` (all deviations set to 0.0)
- `compute_cog` (curve-of-growth not computed in forced extraction; driver-level CoG
  is still computed post-sweep if `compute_cog=True` and `forced=True` in normal forced
  mode)

---

## Photometry Outputs

### `full_photometry` -- Aperture Photometry

Direct pixel summation from the 2D image within elliptical and circular apertures.
Computed per-isophote via `compute_aperture_photometry()` in `fitting.py`.

Output fields added to each isophote dict:

| Field | Description |
|-------|-------------|
| `tflux_e` | Total flux within the elliptical aperture |
| `tflux_c` | Total flux within the circular aperture (radius = SMA) |
| `npix_e` | Number of valid pixels in the elliptical aperture |
| `npix_c` | Number of valid pixels in the circular aperture |

### `compute_cog` -- Curve-of-Growth Photometry

Model-based cumulative flux profile derived from the fitted 1D intensity profile using
trapezoidal annular integration. Computed post-sweep in `driver.py` via `cog.py`. This
method never touches raw pixels; it integrates the analytic annular flux from the fitted
intensity values and ellipse geometry.

Output fields added to each isophote dict:

| Field | Description |
|-------|-------------|
| `cog` | Cumulative flux enclosed within the isophote |
| `cog_annulus` | Annular flux contribution from this isophote |
| `area_annulus` | Geometric area of the annular region |
| `flag_cross` | Boolean flag for isophote crossing detection |
| `flag_negative_area` | Boolean flag for negative annular area |

### When to Use Each

- Use `full_photometry=True` for ground-truth aperture flux measured directly from the
  image. Suitable for total magnitude estimation and aperture correction work.
- Use `compute_cog=True` for model-derived cumulative flux profiles. Suitable for smooth
  growth curves and enclosed-flux diagnostics where pixel noise should be suppressed.

---

## Integration Modes

### Sub-pixel Interpolation

All sampling modes use bilinear sub-pixel interpolation internally via
`scipy.ndimage.map_coordinates` with `order=1`. This is not configurable.

### Mean Integrator (`integrator='mean'`)

Reports the harmonic-fit DC term (`y0_fit`) as the representative intensity. This is the
fitted mean from the 5-parameter (or extended ISOFIT) least-squares harmonic model, not
a simple `np.mean` of the sampled intensities. The harmonic fit naturally downweights
outliers through the least-squares objective.

### Median Integrator (`integrator='median'`)

Reports `np.median(intens)` of the sigma-clipped intensity samples. More robust to
contamination from bright foreground objects or cosmic rays, but discards the harmonic
model information.

### Adaptive Integrator (`integrator='adaptive'`)

Switches between mean and median based on SMA:

- SMA <= `lsb_sma_threshold`: uses mean (harmonic-fit DC term)
- SMA > `lsb_sma_threshold`: uses median

Rationale: inner high-S/N isophotes benefit from the precision of the harmonic fit, while
outer low-S/N isophotes benefit from the robustness of the median against noise spikes.

### Comparison with photutils

The photutils `BILINEAR` integrator uses area-weighted sector integration, which is
architecturally different from isoster's point-sampling path. Both use bilinear
interpolation at the sub-pixel level, but the integration geometry differs. Isoster
samples discrete points along the ellipse; photutils integrates over sectors spanning
the annular width.

---

## Known Limitations

The following limitations are documented based on the current implementation state.
Cross-references point to `docs/future.md` for planned improvements.

1. **Forced mode drops most output fields.** Both `forced=True` and `template_isophotes`
   produce isophote dicts with zero-valued errors, zero deviations, and no aperture
   photometry. Only intensity, RMS, and geometry fields are meaningful.
   (See future.md: "Typed isophote result schema".)

2. **No stop-code filtering in model builder.** `build_isoster_model()` accepts all
   rows with `sma > 0` regardless of stop code or NaN content. Users must pre-filter
   isophotes before model reconstruction for best results.
   (See future.md: "Model builder robustness against low-quality rows".)

3. **CLI exposes only a subset of config.** Advanced options (eccentric anomaly, ISOFIT
   harmonics, CoG, permissive geometry) are not available via CLI flags. Use the Python
   API or a YAML config file.
   (See future.md: "CLI option parity with advanced config features".)

4. **Serial forced/template execution.** Forced and template-based forced modes process
   each SMA sequentially. No parallel execution path exists.
   (See future.md: "Parallel forced/template execution path".)

5. **Central regularization requires previous isophote.** The regularization penalty
   compares against the previous isophote's geometry. It has no effect on the first
   isophote at `sma0`, which may still exhibit instability in low-S/N centers.

6. **Gradient estimator is tightly coupled.** The current gradient computation uses a
   fixed two-radius finite-difference scheme. Alternative estimators are not configurable.
   (See future.md: "Gradient estimator redesign".)

7. **Fixed sampling density formula.** Sampling density follows
   `max(64, int(2 * pi * sma))` and does not adapt to ellipticity, S/N, or local
   curvature.
   (See future.md: "Adaptive sampling density with error controls".)
