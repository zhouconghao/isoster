# Phase 11 Handoff: Huang2013 Campaign Fault Tolerance

## Scope

Prepared Huang2013 workflow for all-galaxy/all-mock batch execution with failure isolation and aggregate summaries.

## Changes

1. `examples/huang2013/run_huang2013_profile_extraction.py`
- Per-method execution is now wrapped in `try/except`.
- A failure in `photutils` or `isoster` no longer aborts the image-level extraction run.
- `*_profiles_manifest.json` now records per-method `status` and `run_summary` (`requested/successful/failed methods`).

2. `examples/huang2013/run_huang2013_qa_afterburner.py`
- Missing method profile FITS no longer raises immediate `FileNotFoundError`.
- Missing outputs are recorded in `method_skips`.
- QA-render exceptions are recorded in `method_failures`.
- QA manifests now include `method_summary` and continue even when only one method is available.

3. `examples/huang2013/run_huang2013_campaign.py` (new)
- Runs extraction + QA for all discovered galaxies and requested mock IDs.
- Continues across per-case failures and missing inputs.
- Writes campaign-wide summary files:
  - JSON: `huang2013_campaign_summary.json`
  - Markdown: `huang2013_campaign_summary.md`
- Tracks final counts for method outcomes (success/failed/unknown), extraction/QA invocation failures, missing inputs, and comparison QA generation.

4. `examples/huang2013/README.md`
- Added full-campaign execution section and command for all galaxies x 4 mocks.

## Resume Command (full sample)

```bash
uv run python examples/huang2013/run_huang2013_campaign.py \
  --huang-root /Users/mac/work/hsc/huang2013 \
  --mock-ids 1 2 3 4 \
  --method both \
  --config-tag baseline \
  --summary-dir outputs/huang2013_campaign_full
```
