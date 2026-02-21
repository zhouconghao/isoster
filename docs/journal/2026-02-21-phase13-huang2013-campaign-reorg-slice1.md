# Phase 13 Slice 1: Huang2013 Campaign Reorganization

## Objective

Reorganize Huang2013 campaign workflow boundaries while preserving extraction/QA behavior and manifest compatibility.

## Completed in This Slice

- Added shared contract helper: `examples/huang2013/huang2013_campaign_contract.py`
  - canonical case prefix builder
  - canonical method artifact path builder
  - canonical extraction/QA manifest path builders
  - robust JSON dict reader
  - extraction manifest method-status reader
- Updated campaign orchestration to use shared helper paths:
  - `examples/huang2013/run_huang2013_campaign.py`
- Updated QA afterburner to use shared helper paths/status loader:
  - `examples/huang2013/run_huang2013_qa_afterburner.py`
- Added targeted regression tests for helper contract:
  - `tests/unit/test_huang2013_campaign_fault_tolerance.py`
- Updated reorg documentation and checkpoints:
  - `docs/spec.md`
  - `docs/todo.md`

## Compatibility Notes

- No filename changes for:
  - `*_profiles_manifest.json`
  - `*_qa_manifest.json`
- No schema-breaking changes in extraction/QA manifests.
- No changes to retry policy or CoG harmonization behavior.

## Verification

- `uv run pytest tests/unit/test_huang2013_campaign_fault_tolerance.py -q`
  - `15 passed in 1.26s`
- `MPLCONFIGDIR=/tmp/mplconfig uv run python examples/huang2013/run_huang2013_campaign.py --huang-root /Users/mac/work/hsc/huang2013 --output-root /Users/mac/work/hsc/huang2013 --galaxies ESO185-G054 --method both --config-tag baseline --limit 1 --verbose --save-log --update`
  - photutils extraction success
  - isoster extraction success
  - QA stage success
  - summary artifacts under `outputs/huang2013_campaign/`

## Next Slice Candidates

1. Split campaign command construction into dedicated helper functions/module.
2. Move case planning (`continue-from`, `limit`, case expansion) into a reusable planner helper.
3. Add manifest schema validation tests for additive-only evolution rules.
