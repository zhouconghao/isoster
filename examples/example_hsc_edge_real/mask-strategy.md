# Two-pass detection-based mask strategy

`build_custom_masks.py` builds per-galaxy object masks for the three real
HSC edge cases in `data/`. The HDF5 bitplane mask is intentionally
**ignored** — the mask is rebuilt from scratch by running
`photutils.segmentation` twice on the i-band science image with two
different parameter sets, then radially blending the results around the
BCG center.

All three galaxies have their BCG peak within ~2 px of the frame
center, so the detection anchor is a smoothed brightest-pixel location
inside an 80 px box centered on the frame center (see
`smoothed_peak()`). The segment containing that anchor is always
treated as the target and excluded from both masks **before** dilation.

## Pass 1 — aggressive outer mask

Purpose: catch everything far from the BCG — faint field galaxies,
scattered stars, and the wide wings of bright stars. Allowed to be
generous because it only affects `r > r_inner`.

| parameter | default | role |
|---|---|---|
| `box_size` | 128 | `Background2D` mesh: large box keeps the BCG envelope out of the background model |
| `filter_size` | 3 | median smoothing of the background mesh |
| `detect_fwhm` | 3.5 | Gaussian kernel FWHM for detection convolution |
| `nsigma` | 1.5 | low threshold — catches faint sources and bright-star wings |
| `npixels` | 8 | minimum connected pixels per segment |
| `deblend_nlevels` | 32 | fine deblend ladder — splits merged field sources |
| `deblend_contrast` | 0.001 | very aggressive deblend contrast |
| `dilate_fwhm` | 10.0 | Gaussian dilation FWHM (kept tight so sources don't bridge) |
| `dilate_threshold` | 0.02 | conservative dilation cutoff |
| `min_peak` | `None` | no peak-flux filter — anything detected is masked |

## Pass 2 — careful inner mask

Purpose: keep the BCG core and its own substructure intact, but still
mask genuine bright contaminants (companions, foreground stars) that
land close to the core. Must not eat the galaxy's own features.

| parameter | default | role |
|---|---|---|
| `box_size` | 48 | smaller background box to follow the galaxy envelope |
| `filter_size` | 3 | same median smoothing |
| `detect_fwhm` | 2.5 | slightly smaller detection kernel |
| `nsigma` | 2.5 | higher threshold so BCG substructure doesn't trigger detection |
| `npixels` | 6 | minimum connected pixels per segment |
| `deblend_nlevels` | 32 | fine deblend ladder |
| `deblend_contrast` | 0.02 | gentler deblend than the outer pass |
| `dilate_fwhm` | 5.0 | small dilation, keeps masks close to each source |
| `dilate_threshold` | 0.02 | same threshold |
| `min_peak` | 6.0 | peak-flux floor (image units) — only bright contaminants survive |

The `min_peak` filter is what makes the careful pass "careful": any
non-target segment whose brightest pixel (in the background-subtracted
image) is below this value is silently dropped before dilation, so
faint residuals from the galaxy's own envelope never end up in the
mask.

## Blending

After both passes we have two boolean masks of the same shape. The
final mask is built by radial blending around the anchor:

```
r < r_inner           → careful[r < r_inner]
r_inner ≤ r ≤ r_outer → careful[band] OR aggressive[band]   # union
r > r_outer           → aggressive[r > r_outer]
```

`r_inner` / `r_outer` are per-galaxy (see below). Everything inside
`r_inner` sees only the careful mask, so BCG substructure is preserved.
Everything outside `r_outer` sees only the aggressive mask, so bright
star halos are captured with the generous dilation. The transition
band (`r_inner ≤ r ≤ r_outer`) takes the union, which gives a smooth
handoff without sharp step artifacts.

## Per-galaxy overrides

Only three knobs are tuned per galaxy: the blend radii, the careful
`min_peak`, and (for the halo-star galaxy) the aggressive dilation.

| galaxy | description | `r_inner` / `r_outer` | careful `min_peak` | aggressive `dilate_fwhm` |
|---|---|---|---|---|
| `37498869835124888` | cluster BCG with multiple bright companions | 140 / 240 | 4.0 | 10.0 (default) |
| `42177291811318246` | BCG with bright NW companion + bright edge star | 140 / 240 | 5.0 | 14.0 |
| `42310032070569600` | BCG with bright halo star + blended bright source | 150 / 260 | 8.0 | 20.0 |

`37498869835124888` uses the lowest `min_peak` because the cluster
field is dense with modest-brightness members that should still be
masked near the core. `42310032070569600` uses the largest aggressive
dilation because the dominant contaminant is a bright saturated star
whose halo extends well beyond its detected segment boundary.

## Final coverage (i-band, blended)

| galaxy | aggressive | careful | blended |
|---|---|---|---|
| `37498869835124888` | 38.9% | 9.7% | **36.8%** |
| `42177291811318246` | 32.0% | 3.4% | **29.9%** |
| `42310032070569600` | 42.4% | 3.5% | **40.8%** |

## Outputs

Written to `data/{obj_id}/`:

- `{obj_id}_mask_custom.fits` — band-agnostic blended mask.
- `{obj_id}_HSC_{G,R,I}_mask_custom.fits` — per-band copies of the
  same blended mask (detection is i-band only; the downstream
  step1/step2/lsb_auto_lock runners expect per-band paths).

Written to `examples/example_hsc_edge_real/`:

- `{obj_id}_mask_compare.png` — six-panel QA figure (i-band + rings,
  aggressive, careful, blended, 400 px central zoom, and an
  aggressive-vs-careful diff map) for visual review.

## Dependencies

`photutils.segmentation.deblend_sources` requires `scikit-image`.
It is declared in the `dev` extra of `pyproject.toml`; install with
`uv sync --extra dev --extra docs`.
