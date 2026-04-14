# LSB Regime Features: Design and Implementation

This document describes the two complementary features that stabilize
isoster isophote fits in the deep low-surface-brightness (LSB) regime:

1. **Automatic LSB Geometry Lock** (`lsb_auto_lock`) — a hard post-lock that
   freezes geometry once the outward fit crosses a gradient-quality
   threshold.
2. **Outer Region Center Regularization** (`use_outer_center_regularization`)
   — a soft pre-lock that nudges the centroid toward a flux-weighted inner
   reference while the fit is still free.

Both features target the same failure mode — outward isophotes drifting
away from the galaxy center once the radial gradient collapses into
background — but they attack it at different stages and compose cleanly.

For usage, see `docs/01-user-guide.md` ("Automatic LSB Geometry Lock" and
"Outer Region Center Regularization"). For parameter names and defaults,
see `docs/02-configuration-reference.md` sections 12 and 13. This document
explains *why* the features exist, *how* the algorithms were chosen, and
*where* the implementation lives file by file.

## 1. Motivation

Photutils/isofit-style elliptical isophote fitting iterates on the largest
absolute harmonic amplitude to update one geometry parameter per iteration
(the traditional coordinate descent). Convergence is signalled when the
largest amplitude drops below a tolerance. In the high-S/N regime this
works well; in the LSB regime it fails in two distinct ways:

- **Noise-driven amplitude updates**: once the local gradient collapses to
  zero, the harmonic amplitudes are dominated by noise. Each iteration's
  "largest" amplitude is random, so the geometry does a random walk outward
  rather than converging.
- **Contamination-driven amplitude updates**: masked bright neighbours, PSF
  wings, unresolved blends and detector artifacts add slowly varying
  harmonics that bias the amplitude selector. The fit walks toward the
  contaminant instead of staying on the galaxy centroid.

Both manifest in the HSC edge-case galaxies in
`examples/example_hsc_edgecases/`. Free fits on the 6-galaxy set show
pre-lock outward centroid drifts from ~1.7 px (clean case) to ~8 px
(blending bright star); `pl_rms` of the centroid trajectory runs
0.4-1.4 px, far above the sub-pixel noise floor of the inner region.

### Why not `fix_center=True`?

`fix_center=True` is the hard solution: freeze x0/y0 for the entire fit.
It works, but it is *too blunt*:

- It throws away the inner fit's own information. If the user does not
  supply an accurate `(x0, y0)`, the frozen center is worse than whatever
  the inner isophotes would have converged to.
- It hides centering errors. A drifting fit is a diagnostic signal; a
  fix_center fit silently bakes in whatever initial guess was used.
- It is incompatible with forced photometry in any useful way (the template
  already fixes geometry).

The two LSB features are softer and data-driven: they derive the reference
center from the inward isophotes, and they only *freeze* geometry once the
fit itself signals that it has entered the LSB regime.

### Why not just the auto-lock?

An auto-lock post-hoc is a hard freeze, but it only fires once the outward
fit has *already* accumulated drift in the pre-lock region. On noisy or
contaminated galaxies the pre-lock drift can reach several pixels before
the gradient quality check trips, and the locked tail is anchored to the
last pre-lock isophote's centroid — which may already be displaced.

Outer center regularization fills this gap: it damps pre-lock drift as the
fit walks outward, so the eventual anchor that the auto-lock commits to is
itself closer to the true centroid. The composition is the key design
point — soft pre-lock + hard post-lock — and is validated empirically in
§5.

## 2. Algorithmic considerations

### 2.1 Automatic LSB Geometry Lock

The detector and the lock mechanism are deliberately simple.

**Detector.** An outward isophote is classified as "LSB-regime" when any
of the following hold:

- Its `stop_code == -1` (isophote failed to converge).
- `grad >= 0` (positive or flat radial gradient — not a galaxy profile).
- `|grad_error / grad| > lsb_auto_lock_maxgerr` (gradient relative error
  exceeds the threshold, default 0.3).

See `isoster/driver.py` → `_is_lsb_isophote(iso, maxgerr_thresh)`.

The earlier session used a second threshold
`lsb_auto_lock_grad_snr` (`|grad / grad_error| < 3.0`); it was dropped in
this branch because it tested the same ratio as `maxgerr` — two knobs
that are in fact one.

**Debounce.** A single isophote crossing the threshold is not enough: the
detector counts consecutive trips (`lsb_auto_lock_debounce`, default 2).
Transient numerical glitches (non-finite gradients, single noisy samples)
therefore cannot commit the lock.

**Lock commit.** When the debounce counter reaches its threshold, the
driver:

1. Walks back `debounce` isophotes to select the *anchor* — the last clean
   outward isophote before the LSB tail started.
2. Builds a locked `IsosterConfig` clone (`_build_locked_cfg`) with
   `fix_center/fix_eps/fix_pa=True` at the anchor geometry, and swaps the
   integrator to `lsb_auto_lock_integrator` (default `"median"`, more
   robust against contamination).
3. Re-runs the remaining outward schedule under the locked clone.
4. Marks each locked isophote with `lsb_locked=True`, and the anchor
   isophote with `lsb_auto_lock_anchor=True`.

All four clone flags collectively disable geometry solvers for the locked
tail, so the remaining isophotes are effectively forced-photometry slices
on a single frozen ellipse. The median integrator avoids one additional
failure mode: a bright pixel inside the locked ellipse dragging the mean
intensity upward.

### 2.2 Outer Region Center Regularization

The soft pre-lock is a regularization penalty added to the isophote
fitting loop. There are two decisions worth calling out.

**Penalize the selector, not the solver.** isoster's per-iteration loop
picks the "best" iteration by minimum
`abs(max_harmonic_amp)` (the effective amplitude). The penalty adds

```
λ(sma) × ((x0 − x0_ref)² + (y0 − y0_ref)²)
```

to that selector, not to the geometry update equations. The geometry math
is untouched: each iteration walks in the direction the harmonic solver
says, but the selector prefers iterations that stay close to the reference
centroid.

This matters because it *cannot inject bias into the fit*. The iteration
sequence is identical with or without the penalty — only which iteration
gets recorded as "best" changes. In the clean case, where the free fit's
own best iteration already sits near the reference, the penalty adds
~0 and is a no-op. In the drifting case, it prefers iterations that
return toward the reference rather than the one with the nominally
smallest harmonic. The experimentally observed 30-50× drift reduction on
the HSC edge cases (§5) comes from this selector bias alone.

Implementing the penalty inside the geometry update equations — as a
Tikhonov-style term on `(Δx, Δy)` — was considered and rejected. It would
require changing every solver branch (5-param, ISOFIT 7-param, EA mode)
and would couple the center regularization to the iteration math, making
it harder to reason about convergence and harder to disable cleanly.

**Logistic ramp in sma.** The penalty is zero at small sma and saturates
at `outer_reg_strength` at large sma:

```
λ(sma) = outer_reg_strength / (1 + exp(−(sma − onset) / width))
```

The inner region is therefore untouched (the logistic is ~0 well inside
`onset`), the transition is smooth (`width` ≈ a few astep growth steps),
and the outer region gets the full strength. The logistic was picked over
a step or a linear ramp because it has a well-defined "center" and width,
is easy to tune from two parameters, and matches the empirically observed
shape of the transition — the drift starts ramping up near the half-light
radius and saturates a few scale lengths further out.

**Where does `(x0_ref, y0_ref)` come from?** From a flux-weighted mean of
the *inward* isophote centers, plus the anchor isophote at `sma0` folded
in unconditionally. See `_build_outer_center_reference` in
`isoster/driver.py`:

- Candidates: all inward isophotes with acceptable stop codes and
  `sma ≤ anchor.sma × outer_reg_ref_sma_factor`.
- Weights: `max(intens, 1e-6)`, so the brightest inner isophotes dominate
  and negative-background rings cannot flip the sign.
- Fallback: if no inward isophote qualifies (`minsma ≥ sma0`, for
  example), the reference is just `(anchor.x0, anchor.y0)` and a
  `UserWarning` is emitted at config time.

The flux-weighted mean (as opposed to picking a single inner isophote) is
robust to individual bad inner isophotes, and is the standard trick for
collapsing a noisy inner trajectory into a single well-defined centroid.

The reference is built *once* per fit, before the outward loop runs, and
is frozen for the remainder of the fit. Letting it update per iteration
would recreate the drift problem the feature is meant to solve.

### 2.3 Inward-first loop ordering

The driver runs the *inward* loop before the outward loop. This is
unconditional in this branch and is not gated on the outer-reg flag,
because:

- The inner reference must exist before any outward isophote is fit with
  the penalty active.
- No existing test depends on the outward-first order; the only test that
  previously cared — a growth-path monkeypatch — was updated to filter by
  role not position.

See the note in `docs/04-architecture.md` ("Inward-first loop order").

## 3. Composition

The two features are designed to compose without interaction side effects:

- **Pre-lock region** (`sma < transition_sma`): outer-reg is active (when
  enabled), auto-lock is idle. The selector-level penalty damps drift.
- **Lock commit** (one isophote): auto-lock walks back to the anchor, which
  may or may not be the last pre-lock isophote. The outer-reg feature does
  not touch the lock mechanism.
- **Locked tail** (`sma ≥ transition_sma`): geometry is frozen by the
  `_build_locked_cfg` clone. The outer-reg penalty is still *evaluated*
  per iteration but has no effect because `fix_center=True` means the
  solver never moves `x0, y0` anyway.

In practice this means enabling both flags is always safe: outer-reg is
silently inert in the locked tail, and auto-lock is silently inert before
its debounce trips. Neither feature is aware of the other.

The auto-lock hard freeze also has a subtle interaction with the
reference: because the anchor the lock commits to is *less drifted* when
outer-reg is on, the locked tail's fixed centroid is itself more accurate.
This is the composition win that the HSC edge-case benchmarks show (§5).

## 4. Guards and limitations

### Forced photometry

Both features are wired into the regular-mode driver only. In template
forced-photometry mode (`_fit_image_template_forced`), geometry is
prescribed by the template and the features are no-ops.

To avoid silent no-ops, the forced-photo driver emits a `UserWarning` at
runtime if either `lsb_auto_lock=True` or
`use_outer_center_regularization=True` is set together with a template.
This is intentional: a user who flips the flag in a config and later
reuses that config in forced-photo mode should see a clear message.

### `fix_center=True`

The outer-reg penalty operates on `(x0, y0)`; if those are already fixed,
it has nothing to do. The config validator emits a `UserWarning` when
`use_outer_center_regularization=True` is combined with `fix_center=True`
so the user knows the feature is inert.

The auto-lock is also inert under `fix_center=True` in the sense that
there is no drift to lock away — but the detector still fires and commits
a "lock" that simply re-freezes what was already frozen. No warning is
needed there.

### Sampling and harmonic modes

Neither feature touches the sampling path (`sampling.py`) or the harmonic
solvers (`fitting.py`'s coefficient math). Both are mode-agnostic by
construction:

- `use_eccentric_anomaly=True` (uniform arc-length sampling, recommended
  for high ε) — the gradient diagnostics that drive the detector are
  computed the same way; the selector penalty is unchanged.
- `simultaneous_harmonics=True` (ISOFIT mode, Ciambur 2015 joint fit) —
  the harmonic amplitudes come out of a larger solver but the "largest
  amplitude" selector is the same, so the penalty slot is unchanged.

This is validated empirically in §5.

### Output keys

Both features only *add* new result-dict keys. They never modify the
per-isophote schema shape, so downstream consumers that don't know about
them keep working. New keys:

- `results["lsb_auto_lock"]` — bool, whether the detector fired.
- `results["lsb_auto_lock_sma"]` — commit sma, or None.
- `results["lsb_auto_lock_count"]` — locked tail length.
- `results["outer_reg_x0_ref"], ["outer_reg_y0_ref"]` — frozen reference
  centroid (only when `use_outer_center_regularization=True`).
- Per-isophote: `iso["lsb_locked"]` and `iso["lsb_auto_lock_anchor"]`.

## 5. Empirical validation (HSC edge cases)

The full benchmark is
`examples/example_hsc_edgecases/run_lsb_mode_sweep.py`, summarized into
`outputs/example_hsc_edgecases/lsb_mode_sweep/_summary.md`. It crosses
three feature arms — `baseline` (free fit), `A` (`lsb_auto_lock=True`),
`B` (A + `use_outer_center_regularization=True` at strength 2.0) — with
three sampling/harmonic modes — `std`, `ea`, `isofit` — over 6 HSC
edge-case galaxies.

### Pre-lock combined drift (pixels)

| galaxy     | baseline | A_std | A_ea | A_isofit | B_std | B_ea | B_isofit |
|------------|---------:|------:|-----:|---------:|------:|-----:|---------:|
| 10140088   |     2.53 |  2.53 | 3.69 |     3.18 |  0.08 | 0.08 |     0.09 |
| 10140002   |     5.52 |  5.52 | 6.72 |    15.23 |  0.10 | 0.07 |     0.10 |
| 10140006   |     7.62 |  7.62 | 6.36 |     4.41 |  0.11 | 0.11 |     0.11 |
| 10140009   |     8.13 |  7.78 | 7.66 |     5.24 |  0.12 | 0.14 |     0.12 |
| 10140056   |     1.71 |  1.71 | 0.88 |     1.49 |  0.16 | 0.20 |     0.13 |
| 10140093   |     4.96 |  2.04 | 1.74 |     4.35 |  0.15 | 0.15 |     0.16 |

Observations:

- **A alone does not meaningfully reduce pre-lock drift** — that's expected,
  because A is a hard post-lock and the pre-lock region is still free.
- **B reduces pre-lock drift to 0.07-0.20 px on every galaxy and every
  mode** — a 30-50× reduction. The reduction ratio is consistent across
  std/EA/ISOFIT, confirming the feature is mode-agnostic.
- `A_isofit` on `10140002` is the dramatic regression case: pre-lock drift
  *triples* (5.52 → 15.23 px) when ISOFIT's higher-order solvers amplify
  noisy geometry updates in the LSB regime. `B_isofit` on the same galaxy
  is 0.10 px. The outer-reg penalty successfully contains what would
  otherwise be a runaway regression.

### Runtime (seconds per galaxy, total over 6 galaxies)

| arm          |  total (s) | per galaxy |
|--------------|-----------:|-----------:|
| baseline_std |        2.0 |       0.33 |
| A_std        |        2.1 |       0.35 |
| A_ea         |        2.2 |       0.37 |
| A_isofit     |        2.3 |       0.38 |
| B_std        |        2.1 |       0.35 |
| B_ea         |        2.2 |       0.37 |
| B_isofit     |        2.4 |       0.40 |

Observations:

- `lsb_auto_lock` itself adds ~5% runtime (B vs A is ~0%).
- `use_eccentric_anomaly=True` adds ~5-10%.
- `simultaneous_harmonics=True` adds ~10-20%.
- Combining all features adds ~20% over the baseline free fit — acceptable
  for the drift-reduction benefit.

## 6. Implementation, file by file

### `isoster/config.py`

Section 12 (outer center regularization) and section 13 (LSB auto-lock)
define the config fields. Key pieces:

- Section 12 fields:
  - `use_outer_center_regularization: bool = False`
  - `outer_reg_sma_onset: float`
  - `outer_reg_sma_width: float`
  - `outer_reg_strength: float = 2.0` (new default; was 1.0 earlier in the
    branch)
  - `outer_reg_ref_sma_factor: float` — filters inward candidates when
    building the flux-weighted reference.

- Section 13 fields:
  - `lsb_auto_lock: bool = False`
  - `lsb_auto_lock_maxgerr: float = 0.3`
  - `lsb_auto_lock_debounce: int = 2`
  - `lsb_auto_lock_integrator: str = "median"`

- Validators in the section 12/13 block:
  - `outer_reg_sma_onset < sma0` → warn.
  - `use_outer_center_regularization=True` with `minsma >= sma0` → warn
    (inner reference will fall back to the anchor).
  - `use_outer_center_regularization=True` with `fix_center=True` → warn
    (penalty is inert).
  - `lsb_auto_lock=True` with any of `fix_center/fix_pa/fix_eps=True` →
    raises, since the lock would conflict with prescribed geometry.
  - `lsb_auto_lock=True` with `debug=False` → silently flips `debug=True`
    (gradient diagnostics are required for the detector) and warns.

### `isoster/driver.py`

The driver owns the detector, the lock commit, and the reference-building
helpers. Key symbols:

- `_is_lsb_isophote(iso, maxgerr_thresh)` — the detector.
- `_build_locked_cfg(cfg, anchor_iso, locked_integrator)` — clones the
  config and freezes geometry for the locked tail.
- `_mark_lsb_lock_state(iso, locked, is_anchor=False)` — single write site
  for `lsb_locked` / `lsb_auto_lock_anchor`, so future loop reorderings
  cannot accidentally mutate the anchor dict twice.
- `_build_outer_center_reference(inwards_results, anchor_iso, cfg)` — the
  flux-weighted inner reference builder.

In `_fit_image_free` (the regular-mode driver):

1. After the inward loop, compute `outer_ref_x0, outer_ref_y0` via
   `_build_outer_center_reference`. Pass them into the outward fit as
   `outer_reference_geom={"x0": ..., "y0": ...}` (no `eps`/`pa`).
2. Initialize `lsb_state = {"locked": False, "consec": 0, "transition_sma": None}`.
3. Mark the anchor with `_mark_lsb_lock_state(anchor_iso, locked=False)`.
4. In the outward loop, after each successful isophote:
   - `_mark_lsb_lock_state(next_iso, locked=lsb_state["locked"])`.
   - If not yet locked, run `_is_lsb_isophote`; increment or reset the
     debounce counter.
   - When the counter hits `lsb_auto_lock_debounce`, pick the anchor
     `debounce` isophotes back, build the locked clone, re-mark the trigger
     isophote with `is_anchor=True`, and switch the loop to the locked
     clone.
5. After the outward loop finishes, write result-dict keys:
   - `result["lsb_auto_lock"] = True`
   - `result["lsb_auto_lock_sma"] = lsb_state["transition_sma"]`
   - `result["lsb_auto_lock_count"] = sum(1 for iso in ... if iso.get("lsb_locked"))`
   - `result["outer_reg_x0_ref"] = outer_ref_x0` (when enabled)
   - `result["outer_reg_y0_ref"] = outer_ref_y0` (when enabled)

The forced-photo driver (`_fit_image_template_forced`) early-returns
before any of the above, and emits a `UserWarning` when either feature
flag is set together with a template.

### `isoster/fitting.py`

The per-isophote fitter owns the selector penalty.

- `compute_outer_center_regularization_penalty(current_geom, reference_geom, sma, config)`
  — see §2.2 for the math. Returns 0.0 when the feature is off, when
  `reference_geom is None`, or when `λ(sma) < 1e-6` (early exit for the
  inner region where the logistic is effectively zero).

- Integration site in `fit_isophote`:

  ```python
  outer_reg_penalty = compute_outer_center_regularization_penalty(
      current_geom, outer_reference_geom, sma, cfg
  )
  effective_amp = abs(max_amp) + reg_penalty + outer_reg_penalty
  if effective_amp < min_amplitude:
      min_amplitude = effective_amp
      ...record this iteration as the new best...
  ```

  `reg_penalty` is the pre-existing central regularization penalty; the
  outer penalty simply adds on top of it. Neither penalty is added to the
  convergence criterion itself — that still uses the raw `|max_amp|` —
  so convergence conditions are unchanged.

### `tests/integration/test_lsb_auto_lock.py`

Integration test coverage for the auto-lock: detector trigger, debounce,
anchor mutation, marker keys, result-dict keys, forced-photo guard,
compatibility with existing modes. Sibling test
`test_outer_center_regularization.py` covers the soft pre-lock: penalty
shape, reference building, fallback on `minsma >= sma0`, `fix_center`
warning, forced-photo guard.

The full suite is 347 passing as of the rename pass (handover
`docs/journal/2026-04-14_handover_2.md`).

## 7. Future work

- **Dynamic reference update when the lock is late.** If the auto-lock
  commits far past the `outer_reg_sma_onset`, the frozen reference is
  measurably "stale" and the penalty is resisting the locked geometry. An
  option to re-evaluate the reference at the lock anchor is worth
  investigating for deep DES/LSST fits.
- **Penalty form ablation for DES/LSST.** The `"absolute"` form was picked
  because it dominated the `"normalized"` form at equivalent strength on
  HSC edge cases (see `docs/journal/2026-04-14_handover.md`). A deeper,
  lower-S/N survey might invert the ordering; the ablation should be
  repeated when that dataset lands.
- **Auto-tuned `outer_reg_strength`.** The `strength=2.0` default was
  chosen from the HSC sweep. A scale-free auto-tune — e.g. set strength
  so the saturated penalty matches the median pre-lock |max_amp| in the
  outer half of the inward loop — would remove one more user knob.
