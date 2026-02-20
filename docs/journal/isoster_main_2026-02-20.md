---
date: 2026-02-20
repo: isoster
branch: main
tags:
  - journal
  - huang2013
  - qa
---

## Progress

- Identified the `mock3` comparison QA crash root cause: `photutils` row with `intens < 0` produced negative `sb_err_mag`, which broke matplotlib `errorbar(yerr=...)`.
- Updated Huang2013 profile preparation to compute `sb_mag_arcsec2` and `sb_err_mag` only for `intens > 0`; non-positive intensity rows now keep both values as `NaN`.
- Added extraction-time validation for all error columns (`*_err`, `*_error`): any negative value now marks the method run as `failed` and writes warning/validation metadata into run JSON and manifests.
- Updated QA afterburner to carry extraction warnings from method run JSON into QA manifest/report warnings.
- Changed Huang2013 initial SMA behavior to fixed default `sma0 = 6.0` pixels (not `RE_PX1`-based fallback).
- Updated docs/rules to persist the fixed-6-pixel SMA rule (`CLAUDE.md`, `examples/huang2013/README.md`).
- Expanded regression tests in `tests/unit/test_huang2013_campaign_fault_tolerance.py`; targeted suite passed: `8 passed`.
- Re-ran `ESO185-G054` campaign in `--update` mode; all 4 mocks completed extraction and QA successfully for both methods, with 4 comparison QA figures generated.
- Confirmed all regenerated `ESO185-G054` run JSON files report `fit_config.sma0 = 6.0`.
- Committed and merged to `main` (fast-forward): commit `ae1c6ae`.

## Lessons Learned

- Surface-brightness quantities must be explicitly undefined when intensity is non-positive; propagating computed errors from invalid intensity causes downstream plotting failures.
- Error-sign validation should be centralized at extraction output boundaries so invalid uncertainties are caught before QA/report generation.
- Comparison QA plotting should apply finite/non-negative masks on both value and error arrays even when upstream validation exists.

## Key Issues

- `main` is ahead of `origin/main` by one commit (`ae1c6ae`); push is still pending.
- Two untracked handover notes remain outside the merge commit: `docs/journal/handover-2026-02-20-2223.md`, `docs/journal/next-session-prompt-2026-02-20-2223.md`.
