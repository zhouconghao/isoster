# Phase 14: Huang2013 Cleanup Utility

## Objective

Add a simple, safe cleanup script to remove generated Huang2013 workflow outputs while preserving mock input images.

## Implementation

- Added: `examples/huang2013/clean_huang2013_outputs.py`
- Supported scopes:
  1. Single test in one galaxy: `--galaxy <GALAXY> --test-name <TEST_NAME>`
  2. All tests in one galaxy: `--galaxy <GALAXY>`
  3. All galaxies: `--all-galaxies`
- Safety and observability:
  - `--dry-run` preview mode
  - `--verbose` per-file logging
  - summary counters (candidates/removed/preserved)

## Preserve Rules

Preserved files in each galaxy folder:

- `<GALAXY>_<TEST>.fits`
- `<GALAXY>_mosaic.png`

All other files matching generated Huang2013 naming patterns in the selected scope are removable.

## Verification

- `uv run python -m py_compile examples/huang2013/clean_huang2013_outputs.py` succeeded.
- Synthetic temp-root validation confirmed:
  - single-test cleanup removes only that test's generated artifacts,
  - galaxy cleanup removes all generated artifacts for that galaxy,
  - all-galaxies cleanup applies across folders,
  - preserved files remained untouched in all modes.
- Follow-up update for new output layout:
  - cleanup now handles nested `<GALAXY>/mock<ID>/` folders (in addition to legacy flat files),
  - empty test folders are removed after cleanup when possible.

## Documentation

- Updated `examples/huang2013/README.md` with cleanup usage examples.
- Updated `docs/todo.md` Phase 14 checklist and verification notes.
