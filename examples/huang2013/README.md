# Huang2013 Real-Mock Workflow

This folder hosts the **current** Huang2013 workflow for external mock images stored in:

- `/Users/mac/work/hsc/huang2013/<GALAXY>/<GALAXY>_mock<ID>.fits`
- output artifacts are written to `/Users/mac/work/hsc/huang2013/<GALAXY>/mock<ID>/`

The workflow is split into two stages:

1. Profile extraction (fitting + reproducible profile products)
2. QA afterburner (figures + comparison + report from saved profiles)

## Table of Contents

- [Files](#files)
- [Test Setup](#test-setup)
- [Quick Start](#quick-start)
- [Full Campaign (All Galaxies x 4 Mocks)](#full-campaign-all-galaxies-x-4-mocks)
- [Long-Run Controls](#long-run-controls)
- [Cleanup Utility](#cleanup-utility)
- [Reproduce IC2597 QA Figures](#reproduce-ic2597-qa-figures)
- [Customize the Plots](#customize-the-plots)
- [Artifact Naming](#artifact-naming)

## Files

- `run_huang2013_profile_extraction.py`
  - Runs `photutils` and/or `isoster` independently.
  - Writes profile FITS/ECSV, per-method run JSON, runtime profile text, and extraction manifest.
- `run_huang2013_qa_afterburner.py`
  - Reads saved profile outputs.
  - Writes per-method QA, comparison QA, markdown report, and QA manifest.
- `run_huang2013_campaign.py`
  - Runs extraction + QA over all requested galaxies/mock IDs with fault tolerance.
  - Writes campaign summary JSON/Markdown with method-level failure statistics.
- `run_huang2013_real_mock_demo.py`
  - Shared helper implementation used by the two scripts above.
- `clean_huang2013_outputs.py`
  - Cleans generated artifacts while preserving mock FITS inputs and mosaic images.
- `real-huang2013-requirements.md`
  - Requirements memory for this campaign.

## Test Setup

Current baseline decisions:

- Pixel scale source: FITS header (`PIXSCALE`)
- `isoster` default: `use_eccentric_anomaly=False`
- True CoG aperture setting: `subpixels=9`
- Initial PA convention correction (Huang2013-specific): `PA_init = PA_header - 90 deg`
- Initial SMA default: fixed `6.0` pixels (then clamped to at least `3.0`)
- Output location: per-test folder `<GALAXY>/mock<ID>/` under `/Users/mac/work/hsc/huang2013`

## Quick Start

Run extraction (one-time per method/config):

```bash
python examples/huang2013/run_huang2013_profile_extraction.py \
  --galaxy IC2597 \
  --mock-id 1 \
  --method both \
  --config-tag baseline
```

Run QA afterburner (re-runnable):

```bash
python examples/huang2013/run_huang2013_qa_afterburner.py \
  --galaxy IC2597 \
  --mock-id 1 \
  --method both \
  --config-tag baseline \
  --output-dir /Users/mac/work/hsc/huang2013/IC2597/mock1
```

## Full Campaign (All Galaxies x 4 Mocks)

Run the fault-tolerant campaign runner across all discovered galaxy folders and mock IDs 1-4:

```bash
uv run python examples/huang2013/run_huang2013_campaign.py \
  --huang-root /Users/mac/work/hsc/huang2013 \
  --mock-ids 1 2 3 4 \
  --method both \
  --config-tag baseline \
  --summary-dir outputs/huang2013_campaign_full
```

Notes:

- Per-case fit failures do not stop the campaign.
- Missing method products are skipped in QA gracefully.
- Final totals are written to:
  - `outputs/huang2013_campaign_full/huang2013_campaign_summary.json`
  - `outputs/huang2013_campaign_full/huang2013_campaign_summary.md`

## Long-Run Controls

Recommended options for large campaigns:

- `--verbose`: prints stage/method start/end and error status.
- `--save-log`: writes per-stage logs (`<PREFIX>_photutils.log`, `<PREFIX>_isoster.log`, `<PREFIX>_qa.log`).
- `--max-runtime-seconds 900`: per-stage timeout guard (default 900 seconds).
- `--continue-from NGC4767`: resume by galaxy (inclusive).
- `--continue-from-case NGC4767_mock1`: resume by exact case (inclusive).
- `--update`: force rerun even when outputs already exist.

Example resume command:

```bash
uv run python examples/huang2013/run_huang2013_campaign.py \
  --huang-root /Users/mac/work/hsc/huang2013 \
  --mock-ids 1 2 3 4 \
  --method both \
  --config-tag baseline \
  --continue-from NGC4767 \
  --verbose --save-log \
  --max-runtime-seconds 900 \
  --summary-dir outputs/huang2013_campaign_full
```

Extraction/QA status handshake:

- `run_huang2013_qa_afterburner.py` now checks extraction status from `<PREFIX>_profiles_manifest.json`.
- If a method has extraction status `failed`, that method QA and two-method comparison QA are skipped automatically.

## Cleanup Utility

Clean a single test in one galaxy:

```bash
uv run python examples/huang2013/clean_huang2013_outputs.py \
  --huang-root /Users/mac/work/hsc/huang2013 \
  --galaxy ESO185-G054 \
  --test-name mock1
```

Clean all test outputs for one galaxy:

```bash
uv run python examples/huang2013/clean_huang2013_outputs.py \
  --huang-root /Users/mac/work/hsc/huang2013 \
  --galaxy ESO185-G054
```

Clean all galaxies:

```bash
uv run python examples/huang2013/clean_huang2013_outputs.py \
  --huang-root /Users/mac/work/hsc/huang2013 \
  --all-galaxies
```

Safety:

- Add `--dry-run` to preview file removals.
- Preserved files in each galaxy folder:
  - `<GALAXY>_<TEST>.fits`
  - `<GALAXY>_mosaic.png`
- Cleanup supports both legacy flat outputs and new `<GALAXY>/mock<ID>/` output folders.

## Reproduce IC2597 QA Figures

If profiles already exist in `~/work/hsc/huang2013/IC2597`, you only need the afterburner:

```bash
python examples/huang2013/run_huang2013_qa_afterburner.py \
  --galaxy IC2597 \
  --mock-id 1 \
  --method both \
  --config-tag baseline \
  --output-dir /Users/mac/work/hsc/huang2013/IC2597/mock1
```

Useful variants:

```bash
# Regenerate only photutils QA
python examples/huang2013/run_huang2013_qa_afterburner.py \
  --galaxy IC2597 --mock-id 1 --method photutils --config-tag baseline \
  --output-dir /Users/mac/work/hsc/huang2013/IC2597/mock1

# Skip comparison panel generation
python examples/huang2013/run_huang2013_qa_afterburner.py \
  --galaxy IC2597 --mock-id 1 --method both --config-tag baseline \
  --skip-comparison --output-dir /Users/mac/work/hsc/huang2013/IC2597/mock1
```

## Customize the Plots

Primary plotting functions are in:

- `examples/huang2013/run_huang2013_real_mock_demo.py`
  - `build_method_qa_figure(...)`
  - `build_comparison_qa_figure(...)`

After editing plot logic, rerun only `run_huang2013_qa_afterburner.py`; no refit is needed.

## Artifact Naming

Prefix: `<GALAXY>_mock<ID>`

Default artifact directory: `<GALAXY>/mock<ID>/`

Method products:

- `<PREFIX>_<METHOD>_<CONFIG_TAG>_profile.fits`
- `<PREFIX>_<METHOD>_<CONFIG_TAG>_profile.ecsv`
- `<PREFIX>_<METHOD>_<CONFIG_TAG>_run.json`
- `<PREFIX>_<METHOD>_<CONFIG_TAG>_runtime-profile.txt`
- `<PREFIX>_<METHOD>_<CONFIG_TAG>_qa.png`

Joint products:

- `<PREFIX>_compare_<TAG or TAG1_vs_TAG2>_qa.png`
- `<PREFIX>_report.md`
- `<PREFIX>_profiles_manifest.json`
- `<PREFIX>_qa_manifest.json`
- `<PREFIX>_photutils.log` (campaign `--save-log`)
- `<PREFIX>_isoster.log` (campaign `--save-log`)
- `<PREFIX>_qa.log` (campaign `--save-log`)
