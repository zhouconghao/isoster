# Phase 12 Clean-Context Handoff (Huang2013 Campaign)

## Snapshot

- Date: 2026-02-15
- Active branch: `feat/huang2013-campaign-fault-tolerance`
- Head commit: `6058746` (`feat(huang2013): add campaign observability, resume, and update controls`)
- Git status: clean working tree

## What Was Completed

1. Campaign observability and control features in `examples/huang2013/run_huang2013_campaign.py`:
- `--verbose`
- `--save-log`
- `--max-runtime-seconds` (default `900`)
- `--continue-from`
- `--continue-from-case`
- `--update`
- Per-method extraction stages when `--method both`.
- Stage-level timeout handling with continuation.

2. Extraction script update-mode and telemetry in `examples/huang2013/run_huang2013_profile_extraction.py`:
- `--verbose`
- `--save-log`
- `--update`
- Default skip-existing logic for reusable outputs.
- Manifest merge safety for per-method reruns.

3. QA extraction-status guard in `examples/huang2013/run_huang2013_qa_afterburner.py`:
- Reads extraction status from `*_profiles_manifest.json` (or `--profiles-manifest`).
- Skips method QA if extraction status is not `success`.
- Two-method comparison QA only runs when both methods are available/successful.

4. Docs and tracking updates:
- `examples/huang2013/README.md`
- `docs/spec.md`
- `docs/todo.md`
- `docs/lessons.md`
- `docs/journal/2026-02-15-huang2013-campaign-observability-plan.md`
- `docs/journal/2026-02-15-phase12-observability-implementation.md`

## Verification Already Run

- `uv run ruff check examples/huang2013/run_huang2013_profile_extraction.py examples/huang2013/run_huang2013_qa_afterburner.py examples/huang2013/run_huang2013_campaign.py`
- `uv run python -m py_compile examples/huang2013/run_huang2013_profile_extraction.py examples/huang2013/run_huang2013_qa_afterburner.py examples/huang2013/run_huang2013_campaign.py`
- CLI help checks for all three scripts.
- Smoke tests for timeout/resume/update and extraction-failed-method QA skip.

## Resume Commands

### Full campaign run (all galaxies x 4 mocks)

```bash
uv run python examples/huang2013/run_huang2013_campaign.py \
  --huang-root /Users/mac/work/hsc/huang2013 \
  --mock-ids 1 2 3 4 \
  --method both \
  --config-tag baseline \
  --verbose --save-log \
  --max-runtime-seconds 900 \
  --summary-dir outputs/huang2013_campaign_full
```

### Resume from a known galaxy

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

### Force rerun completed outputs

```bash
uv run python examples/huang2013/run_huang2013_campaign.py \
  --huang-root /Users/mac/work/hsc/huang2013 \
  --mock-ids 1 2 3 4 \
  --method both \
  --config-tag baseline \
  --continue-from NGC4767 \
  --update \
  --verbose --save-log \
  --max-runtime-seconds 900 \
  --summary-dir outputs/huang2013_campaign_full
```

## Key Output Artifacts

- Campaign summary:
  - `outputs/huang2013_campaign_full/huang2013_campaign_summary.json`
  - `outputs/huang2013_campaign_full/huang2013_campaign_summary.md`
- Per-case logs (when `--save-log`):
  - `<GALAXY>_mock<ID>_photutils.log`
  - `<GALAXY>_mock<ID>_isoster.log`
  - `<GALAXY>_mock<ID>_qa.log`

## Suggested Immediate Next Check

After running/resuming campaign, inspect these quickly:

1. `outputs/huang2013_campaign_full/huang2013_campaign_summary.json`
2. Any timeout/failure logs under `/Users/mac/work/hsc/huang2013/<GALAXY>/` for stuck cases.
