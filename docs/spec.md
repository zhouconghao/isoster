# ISOSTER Technical Specification

## Purpose

ISOSTER is a Python library for elliptical isophote fitting on 2D images, with a function-first API and vectorized sampling/fitting internals.

## Public Interfaces

- `isoster.fit_image(image, mask=None, config=None, template=None)`
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
- `isoster/model.py`: 2D model reconstruction by radial interpolation. `build_isoster_model()` supports `harmonic_orders=None` (auto-detect from isophote keys) and `use_eccentric_anomaly=None` (auto-detect from isophote dicts, or explicit `True`/`False` override) for correct harmonic evaluation in EA-mode fits.
- `isoster/cog.py`: curve-of-growth computation and crossing flags.
- `isoster/utils.py`: serialization to/from FITS and Astropy tables.
- `isoster/plotting.py`: QA visualization (`plot_qa_summary`, `plot_qa_summary_extended`, `plot_comparison_qa_figure`). Multi-method comparison figures support three auto-detected layout modes (solo, 1v1, 3-way), cross-method PA anchoring, errorbars, stop-code markers, and mask overlays. `build_method_profile()` standardizes isophote lists or array dicts into a common profile format. `METHOD_STYLES` provides default visual styles for isoster/photutils/autoprof.
- `isoster/cli.py`: command-line interface entry point.
- `isoster/numba_kernels.py`: optional Numba-accelerated kernels with NumPy fallback.
- `isoster/output_paths.py`: output directory and file path construction helpers.
- `isoster/optimize.py`: compatibility facade re-exporting driver/fitting APIs.

## Key Constants

- `ACCEPTABLE_STOP_CODES = {0, 1, 2}` (`driver.py`): stop codes considered acceptable for continued fitting (outward/inward growth). Used by `is_acceptable_stop_code()` to gate whether the next SMA step proceeds.

## Runtime Modes

`fit_image` selects exactly one mode in this priority order:

1. `template` provided -> template-based forced photometry (`_fit_image_template_forced`).
2. Otherwise -> regular iterative fitting (central pixel, outward growth, optional inward growth).

(`template_isophotes` is supported as a deprecated alias for `template`).

## Regular Fitting Contract

For each SMA in regular mode:

1. Sample ellipse points (`sampling.extract_isophote_data`) with `n_samples = max(64, int(2*pi*sma))`.
2. Apply sigma clipping to sampled `(angle, intensity)` pairs.
3. Fit harmonics:
   - **Default path** (`simultaneous_harmonics=False`): Fit 5-param model (`I0`, `A1`, `B1`, `A2`, `B2`) via `fit_first_and_second_harmonics()`. Higher-order harmonics fitted post-hoc after convergence.
   - **ISOFIT path** (`simultaneous_harmonics=True`): Fit all harmonics simultaneously via `fit_all_harmonics()` using an extended design matrix `[1, sin(θ), cos(θ), sin(2θ), cos(2θ), sin(n₁θ), cos(n₁θ), ...]`. Falls back to 5-param when `n_points < 1 + 2*(2 + len(orders))`. Geometry updates use `A1, B1, A2, B2 = coeffs[1:5]` identically in both paths.
4. Estimate radial gradient (`fitting.compute_gradient`).
5. Update geometry based on dominant harmonic coefficient.
6. Check convergence criterion: `abs(max_amp) < conver * rms` with iteration index check `i >= minit`.

## Stop Codes (Implemented Semantics)

Stop codes currently emitted by core `isoster` fitting paths are:

- `0`: converged / successful forced extraction.
- `1`: too many flagged/clipped samples (`actual_points < total_points * (1.0 - fflag)`).
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

- `template` takes precedence over regular fitting.
- `compute_cog` is only run in regular mode in current `fit_image`; forced/template branches return before CoG attachment.
- Regular mode passes `previous_geometry` during outward/inward growth, so central regularization can apply when enabled.
- Inward growth starts only when the first fitted isophote has an acceptable stop code (`0`, `1`, or `2`).

## Huang2013 Campaign Workflow

The external Huang2013 mock-comparison workflow under `examples/example_huang2013/` is a two-stage pipeline:

1. Profile extraction (`run_huang2013_profile_extraction.py`) with per-method status captured in `*_profiles_manifest.json` (success/failed without aborting the case).
2. QA afterburner (`run_huang2013_qa_afterburner.py`) that tolerates missing method products and emits `*_qa_manifest.json` with skip/failure metadata.

Extraction retries are method-local and shared across photutils/isoster with fixed policy:

- Maximum attempts: 5
- `sma0` increment: +2.0 pixels per retry attempt
- `astep` increment: +0.02 per retry attempt
- `maxsma`: multiplied by `0.95` per retry attempt (5% decay each attempt)
- Retry metadata (`fit_retry_log`, `attempt_count`, `max_attempts`) is persisted in per-method run JSON.

For full-sample execution, `run_huang2013_campaign.py` iterates galaxies/mock IDs, continues across case failures, and writes campaign-level summary JSON/Markdown with aggregate method failure counts plus explicit failed/timeout case labels (per method and QA stage).

Campaign controls include verbose stage telemetry (`--verbose`), per-stage logs (`--save-log`), per-stage timeout guard (`--max-runtime-seconds`, default 900), resume pointers (`--continue-from`, `--continue-from-case`), and skip-existing/rerun control (`--update`).

QA afterburner uses extraction-manifest method status as a guard and skips method QA/comparison for methods with non-success extraction status.

For QA/model reconstruction robustness, isoster 2-D model building in the Huang2013 workflow sanitizes profile rows before calling `isoster.build_isoster_model(...)`: rows with non-finite required fields are filtered, duplicate SMA rows are de-duplicated, and any residual non-finite model pixels are replaced with `0.0` with an explicit warning.

Default output layout is case-scoped:

- input FITS: `<huang-root>/<GALAXY>/<GALAXY>_mock<ID>.fits`
- generated artifacts: `<huang-root>/<GALAXY>/mock<ID>/...`

### Reorganization Boundaries (Planned, Compatibility-Preserving)

Target module boundaries for the campaign reorganization are:

1. CLI wrappers (stable user entry points):
   - `run_huang2013_campaign.py`
   - `run_huang2013_profile_extraction.py`
   - `run_huang2013_qa_afterburner.py`
2. Shared workflow contract module:
   - `huang2013_campaign_contract.py` for canonical case prefix, artifact/manifest path conventions, and manifest status parsing.
3. Future orchestration modules (next slices):
   - campaign case planner/executor and stage command builder split out of the CLI wrapper.

This boundary keeps extraction and QA behavior unchanged while reducing naming/path drift between stages.

### Manifest Compatibility Contract

Manifest compatibility is preserved with additive-only schema evolution:

- Filenames remain unchanged:
  - extraction: `*_profiles_manifest.json`
  - QA: `*_qa_manifest.json` (plus existing tag suffix variant)
- Existing stable fields remain unchanged:
  - extraction: `method_runs`, `run_summary`, `warnings`
  - QA: `method_outputs`, `method_skips`, `method_failures`, `comparison_qa`, `warnings`, `run_metadata`
- Cross-stage status contract remains:
  - QA/campaign decisions read `method_runs.<method>.status` from extraction manifest.
- New fields may be added, but existing field names/types above must remain backward compatible.

## Verification and Artifacts

- Tests: `tests/`
- Benchmarks/profiling: `benchmarks/`
- Reproducible examples: `examples/`
- Generated artifacts: `outputs/`

## Documentation Policy

- Stable docs live in `docs/` root.
- Historical records live under `docs/archive/` and `docs/journal/`.
- Use lowercase kebab-case markdown filenames.
