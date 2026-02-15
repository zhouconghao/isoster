# Phase 12 Implementation: Huang2013 Campaign Observability + Resume

## Branch

- `feat/huang2013-campaign-observability-resume`

## Implemented Changes

1. `examples/huang2013/run_huang2013_campaign.py`
- Added CLI controls:
  - `--verbose`
  - `--save-log`
  - `--max-runtime-seconds` (default `900`)
  - `--continue-from`
  - `--continue-from-case`
  - `--update`
- Extraction now runs per method when `--method both` to isolate stage-level failures/timeouts.
- Added timeout handling per stage with continuation across remaining methods/cases.
- Added per-stage logs with method-specific filenames:
  - `<PREFIX>_photutils.log`
  - `<PREFIX>_isoster.log`
  - `<PREFIX>_qa.log`
- Added default skip-existing behavior for reusable outputs; rerun only with `--update`.
- Added resume filtering by galaxy or exact case.

2. `examples/huang2013/run_huang2013_profile_extraction.py`
- Added CLI controls:
  - `--verbose`
  - `--save-log`
  - `--update`
- Added skip-existing logic (default) for reusable method outputs.
- Added per-method start/end telemetry.
- Added manifest merge behavior so per-method invocations do not erase prior method statuses in `*_profiles_manifest.json`.

3. `examples/huang2013/run_huang2013_qa_afterburner.py`
- Added extraction-status guard:
  - reads `*_profiles_manifest.json` (or `--profiles-manifest`),
  - skips methods with extraction status not equal to `success`,
  - thus avoids running invalid method QA and two-method comparison when one method failed.
- Added CLI controls:
  - `--profiles-manifest`
  - `--ignore-extraction-status`
  - `--verbose`

4. Documentation updates
- `examples/huang2013/README.md` updated with long-run controls and resume example.
- `docs/spec.md` updated with new campaign controls and extraction-status QA guard.
- `docs/todo.md` phase checklist and verification evidence updated.
- `docs/lessons.md` updated with reusable lessons from this fix set.

## Verification Summary

- Lint: `uv run ruff check ...` passed for all 3 updated scripts.
- Syntax: `uv run python -m py_compile ...` passed.
- CLI help checks passed for all 3 scripts.
- Smoke tests passed for:
  - timeout handling,
  - resume (`--continue-from`, `--continue-from-case`),
  - skip-existing and `--update`,
  - extraction-failed method auto-skip in QA afterburner.

## Example Resume Command

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
