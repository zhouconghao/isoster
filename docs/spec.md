# ISOSTER Technical Specification

## Purpose

ISOSTER is a Python library for elliptical isophote fitting on 2D images, with a function-first API and vectorized sampling/fitting internals.

## Public Interfaces

- `isoster.fit_image(image, mask=None, config=None, template_isophotes=None)`
- `isoster.fit_isophote(...)`
- `isoster.isophote_results_to_fits(...)`
- `isoster.isophote_results_from_fits(...)`
- `isoster.isophote_results_to_astropy_tables(...)`
- `isoster.build_isoster_model(...)`
- CLI entry point: `isoster` (`isoster/cli.py`)

## Core Modules

- `isoster/driver.py`: image-level orchestration and mode routing.
- `isoster/fitting.py`: per-isophote fitting loop, gradient checks, geometry updates.
- `isoster/sampling.py`: vectorized ellipse sampling via `scipy.ndimage.map_coordinates`.
- `isoster/config.py`: `IsosterConfig` schema and validators.
- `isoster/model.py`: 2D model reconstruction by radial interpolation.
- `isoster/cog.py`: curve-of-growth computation and crossing flags.
- `isoster/utils.py`: serialization to/from FITS and Astropy tables.
- `isoster/optimize.py`: compatibility facade re-exporting driver/fitting APIs.

## Runtime Modes

`fit_image` selects exactly one mode in this priority order:

1. `template_isophotes` provided -> template-based forced photometry (`driver._fit_image_template_forced`).
2. `config.forced=True` -> fixed-geometry forced photometry (`fitting.extract_forced_photometry`) using `forced_sma`.
3. Otherwise -> regular iterative fitting (central pixel, outward growth, optional inward growth).

## Regular Fitting Contract

For each SMA in regular mode:

1. Sample ellipse points (`sampling.extract_isophote_data`) with `n_samples = max(64, int(2*pi*sma))`.
2. Apply sigma clipping to sampled `(angle, intensity)` pairs.
3. Fit first/second harmonics (`I0`, `A1`, `B1`, `A2`, `B2`).
4. Estimate radial gradient (`fitting.compute_gradient`).
5. Update geometry based on dominant harmonic coefficient.
6. Check convergence criterion: `abs(max_amp) < conver * rms` with iteration index check `i >= minit`.

## Stop Codes (Implemented Semantics)

Stop codes currently emitted by core `isoster` fitting paths are:

- `0`: converged / successful forced extraction.
- `1`: too many flagged/clipped samples for the current ellipse.
- `2`: reached `maxit` without convergence; best-so-far geometry fallback.
- `3`: too few points (`< 6`) for first/second harmonic fit.
- `-1`: gradient-related failure.

Canonical user-facing stop-code documentation lives in `docs/user-guide.md`.

## Output Contract

`fit_image` returns:

- `results['isophotes']`: list of dict rows, one per sampled/fitted SMA.
- `results['config']`: the resolved `IsosterConfig` object.

Each isophote row includes geometry/intensity fields and optional blocks depending on config:

- Harmonic deviations: `a{n}`, `b{n}`, `a{n}_err`, `b{n}_err` for requested `harmonic_orders`.
- Full aperture photometry (`full_photometry`): `tflux_e`, `tflux_c`, `npix_e`, `npix_c`.
- CoG (`compute_cog` in regular mode): `cog`, `cog_annulus`, `area_annulus`, `flag_cross`, `flag_negative_area`.

## Known Behavior Notes

- `template_isophotes` takes precedence over `forced=True`.
- `compute_cog` is only run in regular mode in current `fit_image`; forced/template branches return before CoG attachment.
- Regular mode passes `previous_geometry` during outward/inward growth, so central regularization can apply when enabled.
- Inward growth starts only when the first fitted isophote has an acceptable stop code (`0`, `1`, or `2`).

## Huang2013 Campaign Workflow

The external Huang2013 mock-comparison workflow under `examples/huang2013/` is a two-stage pipeline:

1. Profile extraction (`run_huang2013_profile_extraction.py`) with per-method status captured in `*_profiles_manifest.json` (success/failed without aborting the case).
2. QA afterburner (`run_huang2013_qa_afterburner.py`) that tolerates missing method products and emits `*_qa_manifest.json` with skip/failure metadata.

Extraction retries are method-local and shared across photutils/isoster with fixed policy:

- Maximum attempts: 5
- `sma0` increment: +2.0 pixels per retry attempt
- `astep` increment: +0.02 per retry attempt
- `maxsma`: multiplied by `0.95` per retry attempt (5% decay each attempt)
- Retry metadata (`fit_retry_log`, `attempt_count`, `max_attempts`) is persisted in per-method run JSON.

For full-sample execution, `run_huang2013_campaign.py` iterates galaxies/mock IDs, continues across case failures, and writes campaign-level summary JSON/Markdown with aggregate method failure counts.

Campaign controls include verbose stage telemetry (`--verbose`), per-stage logs (`--save-log`), per-stage timeout guard (`--max-runtime-seconds`, default 900), resume pointers (`--continue-from`, `--continue-from-case`), and skip-existing/rerun control (`--update`).

QA afterburner uses extraction-manifest method status as a guard and skips method QA/comparison for methods with non-success extraction status.

## Verification and Artifacts

- Tests: `tests/`
- Benchmarks/profiling: `benchmarks/`
- Reproducible examples: `examples/`
- Generated artifacts: `outputs/`

## Documentation Policy

- Stable docs live in `docs/` root.
- Historical records live under `docs/archive/` and `docs/journal/`.
- Use lowercase kebab-case markdown filenames.
