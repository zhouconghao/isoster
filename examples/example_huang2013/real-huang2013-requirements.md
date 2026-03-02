# Huang2013 Real-Mock Test Requirements

## Objective

Build a reproducible, scalable workflow to compare `photutils.isophote` and `isoster` on externally generated Huang2013 mock FITS images. Start with one demo: `IC2597_mock1.fits` (noiseless), then scale to all galaxies and all mock sets.

## Dataset Context

- External root directory: `/Users/mac/work/hsc/huang2013`
- Per-galaxy folder structure: `${HUANG_ROOT}/${GALAXY}/`
- Mock files per galaxy:
  - `${GALAXY}_mock1.fits`: `z=0.05`, noiseless, highest-resolution truth reference
  - `${GALAXY}_mock2.fits`: `z=0.05`, noisy
  - `${GALAXY}_mock3.fits`: `z=0.20`, noisy
  - `${GALAXY}_mock4.fits`: `z=0.50`, noisy
- Script location requirement: all scripts under `examples/example_huang2013/`
- Artifact location requirement: all generated run artifacts saved in the external galaxy folder, not in this repository.

## Non-Negotiable Workflow Requirements

1. `photutils.isophote` and `isoster` runs must be independent and reproducible.
2. Pipeline stages must be split:
   - profile extraction stage (no QA plotting)
   - QA afterburner stage (reads saved profiles and generates figures/reports)
3. Result files must preserve enough information to reproduce:
   - 1-D profiles
   - reconstructed 2-D model images
   - key run configurations and runtime metadata
4. Naming must use `GALAXY_MOCKID` prefix and informative suffixes.
5. Runtime profiling must always be performed and persisted.
6. Outputs must include saved results, QA figures, and a brief run report.

## Artifact Contract

For each method (`photutils`, `isoster`), use stem:

- `${PREFIX}_${METHOD}_${CONFIG_TAG}`
- where `${PREFIX} = ${GALAXY}_mock${MOCK_ID}`

Required outputs:

- `${STEM}_profile.fits`: primary machine-readable 1-D profile product
- `${STEM}_profile.ecsv`: readable profile table mirror
- `${STEM}_run.json`: full run/config/runtime metadata
- `${STEM}_runtime-profile.txt`: cProfile top-call summary
- `${STEM}_qa.png`: per-method QA figure

Joint outputs (when both methods are available):

- `${PREFIX}_compare_${PHOTUTILS_TAG}_vs_${ISOSTER_TAG}_qa.png`
- `${PREFIX}_report.md`: concise run summary and key metrics
- `${PREFIX}_manifest.json`: file index and run linkage metadata

## QA Figure Requirements

Per-method QA figure must include:

1. Original image with selected fitted isophote overlays.
2. Reconstructed 2-D model image (generated in-memory, not saved as FITS):
   - display using arcsinh scaling and `viridis`.
3. 2-D residual map `(model - data) / data`:
   - use diverging colormap that highlights small differences.
4. Compact 1-D panel stack sharing one x-axis (no gaps):
   - surface brightness (`zp=27.0` mag; with error bars)
   - centroid profile
   - axis-ratio profile
   - normalized PA profile (unwrap to avoid artificial jumps)
   - curve-of-growth profile
5. Problematic stop-code points must be visually distinct from valid points.
6. X-axis must be `kpc^0.25`, derived from standard cosmology, and exclude the first pixel (`sma <= 1`).
7. Figure title must include runtime statistics.
8. Y-axis limits must be set using profile values, not inflated by extreme error bars.

Comparison QA figure must include:

1. Comparable layout to per-method QA summary.
2. Direct `photutils` vs `isoster` profile comparisons.
3. Explicit relative surface-brightness difference panel.

## True Curve-of-Growth Requirement

For each method result:

1. Use the final fitted ellipse geometry at each SMA.
2. Perform high-precision elliptical aperture photometry on the noiseless image.
3. Save resulting true CoG alongside fitted profile rows where possible.
4. Persist CoG method settings (engine, subpixel factor, assumptions) in run metadata.

## Implementation Notes for Current Demo

- Demo target: `IC2597_mock1.fits`
- Default parameter decisions for this demo:
  - Pixel scale source: FITS header (`PIXSCALE`)
  - isoster sampling mode: `use_eccentric_anomaly=False`
  - True CoG aperture setting: `subpixels=9`
- Must be designed so only run arguments change when scaling to:
  - all galaxies
  - all four mock sets
  - multiple `isoster` configuration variants
