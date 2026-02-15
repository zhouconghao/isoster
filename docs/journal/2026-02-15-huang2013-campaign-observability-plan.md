# Huang2013 Campaign Observability + Resume Plan

## Problem Summary

On large cases (for example `NGC4767_mock1` at `4000x4000`), campaign execution can appear stuck with no live progress visibility. Current logs are post-hoc and coarse, and there is no per-stage timeout or resume pointer.

## Goals

1. Add live visibility (`--verbose`) for stage/method start/end and errors.
2. Add persistent per-stage logs (`--save-log`) with method-specific filenames.
3. Add per-stage timeout control (`--max-runtime-seconds`, default `900`).
4. Add resumable execution (`--continue-from` at least galaxy-level).
5. Add skip-by-default behavior with explicit `--update` override for reruns.
6. Ensure QA stage respects extraction failures so failed methods do not trigger method QA/comparison work.

## Implementation Plan

### A. QA extraction-status guard (correctness)

- Update `run_huang2013_qa_afterburner.py` to read extraction manifest (`<prefix>_profiles_manifest.json` by default).
- If extraction status for a method is not `success`, mark method as skipped (`reason=extraction_status_<status>`) and do not run method QA rendering for that method.
- Keep comparison QA disabled unless both methods are available and successful.

### B. Campaign runner observability and controls

- Refactor campaign extraction stage to run one subprocess per method when `--method both`.
- Add options to `run_huang2013_campaign.py`:
  - `--verbose`
  - `--save-log`
  - `--max-runtime-seconds` (default `900`)
  - `--continue-from`
  - `--update`
- Implement streaming subprocess runner (`Popen`) with:
  - timeout handling,
  - optional live stdout/stderr emission,
  - optional log persistence.
- Emit per-case/per-stage status in summary JSON (`success`, `failed`, `timeout`, `skipped_existing`).

### C. Single-image update mode support

- Add `--update` to `run_huang2013_profile_extraction.py`.
- By default, skip method execution when output artifacts already exist and run JSON indicates prior success.
- With `--update`, rerun and overwrite.

### D. Documentation and verification

- Update `examples/huang2013/README.md` with new options and resume examples.
- Update `docs/todo.md` with checklist and verification evidence.
- Run validation:
  - `ruff check` for modified scripts,
  - `--help` checks,
  - smoke runs for `--continue-from`, `--update`, and timeout path.

## Acceptance Criteria

- During campaign run, operator can identify active stage/method in real time.
- A hung method process is terminated at timeout and campaign continues.
- Resume from `--continue-from NGC4767` works without restarting earlier galaxies.
- Default run skips completed outputs; `--update` forces rerun.
- QA script does not run method QA/comparison for methods marked failed in extraction manifest.
