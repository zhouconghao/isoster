# examples/ — Review Snapshot

> **Status**: Current inventory documented as of 2026-03-02.
> A full redesign is planned. This file captures the state before cleanup.

---

## Current Inventory

| Script | Lines | Purpose | Input | Status |
|--------|------:|---------|-------|--------|
| **Root-level** | | | | |
| `basic_usage.py` | 130 | Minimal `fit_image()` workflow on synthetic Sérsic | Synthetic (generated in-script) | Active — keep |
| `compare_isofit_modes.py` | 677 | Compare 3 EA/ISOFIT modes on LegacySurvey galaxies | `data/eso243-49.fits`, `data/ngc3610.fits` | Active — keep |
| `curve_of_growth.py` | 313 | Curve-of-growth photometry on synthetic Sérsic vs analytic truth | Synthetic (generated in-script) | Active — keep |
| `ngc3610_highorder_exploration.py` | 750 | High-order harmonics exploration on NGC 3610 | `data/ngc3610.fits` | Loose — archive |
| `ngc3610_mask_effect.py` | 495 | Masking effect on NGC 3610 isophotes | `data/ngc3610.fits` | Loose — archive |
| `ngc3610_sma0_effect.py` | 523 | Initial SMA effect on NGC 3610 isophotes | `data/ngc3610.fits` | Loose — archive |
| **huang2013/** | | | | |
| `huang2013_shared.py` | — | Shared helpers for Huang2013 campaign | External Huang2013 FITS | Canonical — keep |
| `run_huang2013_campaign.py` | — | Full Huang2013 multi-galaxy campaign runner | External Huang2013 FITS | Canonical — keep |
| `run_huang2013_profile_extraction.py` | — | Profile extraction sub-step | Campaign outputs | Canonical — keep |
| `run_huang2013_qa_afterburner.py` | — | QA figure generation for campaign | Campaign outputs | Canonical — keep |
| `run_huang2013_real_mock_demo.py` | — | Demo: real vs mock comparison | External Huang2013 FITS | Keep |
| `clean_huang2013_outputs.py` | — | Cleanup script for campaign outputs | Campaign outputs | Utility |
| `mockgal.py` | — | High-fidelity mock generation (libprofit) | Config YAML | Canonical — keep |
| `huang2013_campaign_contract.py` | — | Contract/protocol definitions for campaign | — | Keep |
| **real_galaxy_legacysurvey_highorder_harmonics/** | | | | |
| `run_example.py` | — | End-to-end high-order harmonics on LegacySurvey galaxies | `data/eso243-49.fits`, `data/ngc3610.fits` | Active — keep |
| `shared.py` | — | Shared galaxy metadata + plotting helpers | — | Active — keep |
| `masking.py` | — | Object masking for LegacySurvey images | — | Active — keep |
| **data/** | | | | |
| `data/` | — | **Moved to `data/` at project root** — see `data/README.md` | — | Relocated |
| **mockgal/** | | | | |
| `mockgal/models_config_batch/` | — | Batch template files for `mockgal_adapter.py` | — | Active |

---

## Cluster Analysis

Three natural clusters exist:

### Cluster 1 — Huang2013 Mock Workflow (`huang2013/`)

The most complete and scientifically significant workflow. Covers:
- Mock generation with PSF convolution and realistic noise (`mockgal.py`)
- Multi-galaxy campaign fitting (`run_huang2013_campaign.py`)
- Profile extraction and QA afterburner
- Real mock comparison

This workflow is the canonical production example and should be preserved and refactored
into a clean, documented user-facing tutorial.

**Data dependency**: external Huang2013 FITS at `/Users/mac/work/hsc/huang2013/<GALAXY>/`.
Not tracked in the repo.

### Cluster 2 — LegacySurvey High-Order Harmonics (`real_galaxy_legacysurvey_highorder_harmonics/`)

A structured example campaign covering:
- Multi-band fitting on ESO 243-49 (edge-on S0) and NGC 3610 (boxy elliptical)
- Six fitting conditions: PA/EA mode × [no harmonics / [3,4] / [3..10]]
- Object masking
- Comparative QA figures

**Data dependency**: `data/eso243-49.fits`, `data/ngc3610.fits`.

### Cluster 3 — Loose NGC 3610 Explorations (root-level)

Three loose scripts exploring NGC 3610:
- `ngc3610_highorder_exploration.py` — high-order harmonics
- `ngc3610_mask_effect.py` — masking effect
- `ngc3610_sma0_effect.py` — SMA0 sensitivity

These were exploratory development scripts and overlap significantly with
`real_galaxy_legacysurvey_highorder_harmonics/run_example.py`. They are candidates for archiving.

---

## Observations

1. `data/` subdirectory is now relocated to `data/` at project root (Phase A of housekeeping).
   Scripts that previously used `examples/data/` have been updated.

2. No root-level `__init__.py` — examples are not importable as a package, which is fine.

3. `basic_usage.py` and `curve_of_growth.py` use only synthetic data and have no FITS dependency —
   they are the most portable examples for new users.

4. `compare_isofit_modes.py` (677 lines) is a thorough comparative script but is not documented
   in the main README.

---

## Future Redesign Notes

The user has indicated a future redesign targeting three canonical workflows:

1. **Huang2013 mock workflow** — retain and refactor as the primary tutorial.
   Focus on science reproducibility and documentation quality.

2. **LegacySurvey SGA-2020** — new workflow using the Siena Galaxy Atlas 2020 catalog.
   Replaces ad-hoc LegacySurvey scripts.

3. **HSC massive galaxies** — new workflow for the HSC survey.

**Immediate cleanup** (before redesign):
- Archive `ngc3610_highorder_exploration.py`, `ngc3610_mask_effect.py`, `ngc3610_sma0_effect.py`
  to an `examples/archive/` folder.
- Retain `basic_usage.py`, `curve_of_growth.py`, `compare_isofit_modes.py` at root level.

**No code changes in the current housekeeping session** — this file documents intent only.
