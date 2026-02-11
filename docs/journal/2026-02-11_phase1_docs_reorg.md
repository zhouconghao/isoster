# 2026-02-11: Phase 1 Documentation Reorganization

## Scope

Executed phase 1 plan for documentation and structure cleanup.

## What Changed

- Created branch: `chore/docs-structure-phase1`.
- Preserved legacy planning memory: `docs/archive/legacy_todo_photutils_parity.md`.
- Reorganized docs:
  - moved `docs/development/` to `docs/archive/development/`
  - moved `docs/review/` to `docs/archive/review/`
- Renamed markdown files to lowercase kebab-case where applicable.
- Added canonical docs:
  - `docs/README.md`
  - `docs/spec.md`
  - `docs/lessons.md`
- Updated docs home page: `docs/index.md`.
- Added folder-level TOC READMEs:
  - `tests/README.md`
  - `benchmarks/README.md`
  - `examples/README.md`
- Updated:
  - `README.md` (links, run commands, repo structure, multiband snippet fix)
  - `CLAUDE.md` (naming/output/context-memory rules, test command fixes)
  - `mkdocs.yml` (removed missing API pages)
  - `isoster/config.py` doc link to stop-codes
  - `docs/stop-codes.md` stale review link

## Verification

- `pytest --collect-only -q` completed successfully.
- Result: 102 tests collected, 4 deselected.

## Remaining Work (Phase 2)

- Standardize all artifact output paths under `outputs/`.
- Replace `tests/qa_outputs` and hardcoded mixed output paths.
- Add shared output helper and migrate scripts/tests to it.
