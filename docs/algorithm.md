# Algorithm and Implementation Notes

This document describes the behavior of the current code path in `isoster/sampling.py`, `isoster/fitting.py`, and `isoster/driver.py`.

## Pipeline Overview

Regular mode in `fit_image` (`isoster/driver.py`) runs:

1. Central pixel estimate (`fit_central_pixel`).
2. First free-geometry isophote at `sma0` (`fit_isophote`).
3. Outward growth until `maxsma`.
4. Optional inward growth until `max(minsma, 0.5)`.
5. Optional CoG attachment (`compute_cog`) in regular mode.

Mode selection priority is:

1. `template_isophotes` provided.
2. `config.forced=True`.
3. Regular mode.

## Sampling and Angle Spaces

`extract_isophote_data` uses `map_coordinates` to sample intensities along ellipse coordinates computed by `compute_ellipse_coords`.

Sampling density:

- `n_samples = max(64, int(2*pi*sma))`

Angle semantics:

- Regular mode (`use_eccentric_anomaly=False`): harmonics use `phi` (position angle).
- Eccentric-anomaly mode (`use_eccentric_anomaly=True`): harmonics use `psi`; geometry updates still use `phi`.

The returned structure is `IsophoteData(angles, phi, intens, radii)` where `angles` is the harmonic-fit angle basis.

## Per-Isophote Fitting Loop

`fit_isophote` iterates up to `maxit`:

1. Sample data for current geometry.
2. Sigma-clip sampled profile (`sigma_clip`).
3. Early quality exits:
   - stop code `1` when `actual_points < total_points * fflag`.
   - stop code `3` when `len(intens) < 6`.
4. Fit first/second harmonics by least squares.
5. Compute radial gradient using offset SMA sampling.
6. Apply gradient quality checks (outward mode only):
   - uses `maxgerr`, sign checks, and a two-strike `lexceed` rule before stop code `-1`.
7. Compute dominant harmonic amplitude and geometry correction target.
8. Track the best geometry by minimal `effective_amp = abs(max_amp) + regularization_penalty`.
9. Converge when `abs(max_amp) < conver * rms` and `i >= minit`.

On convergence, optional blocks are attached:

- parameter errors (`compute_errors=True`)
- higher-order harmonics (`compute_deviations=True`, sequential or simultaneous)
- aperture metrics (`full_photometry=True`)

## Geometry Update Mapping

The dominant first/second harmonic term selects the parameter update:

- max index `0` (`A1`) -> center shift (minor-axis direction component).
- max index `1` (`B1`) -> center shift (major-axis direction component).
- max index `2` (`A2`) -> position-angle update.
- max index `3` (`B2`) -> ellipticity update with bounds/wrap handling.

Constraint flags (`fix_center`, `fix_pa`, `fix_eps`) zero the corresponding harmonic terms in the dominance selection step.

## Integrators

Supported config values:

- `mean`
- `median`
- `adaptive` (switches to `median` when `sma > lsb_sma_threshold`)

`adaptive` requires `lsb_sma_threshold` by config validation.

## Forced and Template Modes

### Forced (`config.forced=True`)

- Extracts per-SMA photometry using a single fixed geometry `(x0, y0, eps, pa)`.
- Returns stop code `0` for successful extraction, `3` for empty sampled data.

### Template-based forced (`template_isophotes`)

- Sorts template by `sma`.
- Reuses each template isophote geometry at its own SMA.
- Handles `sma == 0` via `fit_central_pixel`.

## Stop Codes

Current `isoster` fitting emits:

- `0`: success
- `1`: too many flagged points
- `3`: too few points
- `-1`: gradient-related failure

`2` is reserved/compatibility-only in current core code and is not emitted by `fit_isophote`.

Canonical user-facing reference: `docs/user-guide.md`.

## Current Caveats

- Central regularization depends on `previous_geometry`, but regular `fit_image` calls currently do not pass it.
- In `compute_gradient`, both linear and multiplicative growth use `/ (sma * astep)` normalization; linear-growth users should treat this behavior as implementation-specific.
- `build_isoster_model` currently filters only on `sma > 0`, not by stop-code quality.
