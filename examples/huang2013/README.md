# Huang2013 Real-Mock Workflow

This folder hosts the **current** Huang2013 workflow for external mock images stored in:

- `/Users/mac/work/hsc/huang2013/<GALAXY>/<GALAXY>_mock<ID>.fits`

The workflow is split into two stages:

1. Profile extraction (fitting + reproducible profile products)
2. QA afterburner (figures + comparison + report from saved profiles)

## Table of Contents

- [Files](#files)
- [Test Setup](#test-setup)
- [Quick Start](#quick-start)
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
- `run_huang2013_real_mock_demo.py`
  - Shared helper implementation used by the two scripts above.
- `real-huang2013-requirements.md`
  - Requirements memory for this campaign.

## Test Setup

Current baseline decisions:

- Pixel scale source: FITS header (`PIXSCALE`)
- `isoster` default: `use_eccentric_anomaly=False`
- True CoG aperture setting: `subpixels=9`
- Output location: the target galaxy folder under `/Users/mac/work/hsc/huang2013`

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
  --output-dir /Users/mac/work/hsc/huang2013/IC2597
```

## Reproduce IC2597 QA Figures

If profiles already exist in `~/work/hsc/huang2013/IC2597`, you only need the afterburner:

```bash
python examples/huang2013/run_huang2013_qa_afterburner.py \
  --galaxy IC2597 \
  --mock-id 1 \
  --method both \
  --config-tag baseline \
  --output-dir /Users/mac/work/hsc/huang2013/IC2597
```

Useful variants:

```bash
# Regenerate only photutils QA
python examples/huang2013/run_huang2013_qa_afterburner.py \
  --galaxy IC2597 --mock-id 1 --method photutils --config-tag baseline \
  --output-dir /Users/mac/work/hsc/huang2013/IC2597

# Skip comparison panel generation
python examples/huang2013/run_huang2013_qa_afterburner.py \
  --galaxy IC2597 --mock-id 1 --method both --config-tag baseline \
  --skip-comparison --output-dir /Users/mac/work/hsc/huang2013/IC2597
```

## Customize the Plots

Primary plotting functions are in:

- `examples/huang2013/run_huang2013_real_mock_demo.py`
  - `build_method_qa_figure(...)`
  - `build_comparison_qa_figure(...)`

After editing plot logic, rerun only `run_huang2013_qa_afterburner.py`; no refit is needed.

## Artifact Naming

Prefix: `<GALAXY>_mock<ID>`

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
