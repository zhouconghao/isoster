# Phase 1 Plan: Documentation and Structure Reorganization

## Program Status

Architecture review program (Phases 1-3) is completed as of 2026-02-11.

## Scope
This plan covers the first implementation phase:
- documentation architecture cleanup
- naming consistency for markdown files
- folder-level README tables of contents for `tests`, `benchmarks`, and `examples`
- alignment updates in `README.md` and `CLAUDE.md`

## Checklist
| Item | Status | Notes |
|---|---|---|
| 1. Preserve existing planning memory and legacy docs in archive | [x] | `docs/archive/legacy_todo_photutils_parity.md` created |
| 2. Define canonical docs structure and naming convention | [x] | Convention set to lowercase kebab-case markdown names |
| 3. Reorganize `docs/` with a new `docs/README.md` and stable sections | [x] | Stable docs promoted; legacy material moved to `docs/archive/` |
| 4. Add `README.md` files to `tests/`, `benchmarks/`, and `examples/` | [x] | Added folder-level scope, reproducible commands, and output policy |
| 5. Update top-level `README.md` links and testing guidance | [x] | Updated links and added run entry points |
| 6. Update `CLAUDE.md` with architecture/file naming rules and memory-preservation workflow | [x] | Added explicit context and memory-preservation section |
| 7. Fix `mkdocs.yml` navigation to existing files | [x] | Removed missing API pages and linked current docs |
| 8. Verify by checking links/paths and running a fast sanity check | [x] | `pytest --collect-only -q` passed (102 collected, 4 deselected) |
| 9. Write phase review summary in this file | [x] | Added below |

## Review

### Completed

- Preserved legacy plan memory in `docs/archive/legacy_todo_photutils_parity.md`.
- Reorganized docs into stable root docs and historical archive:
  - moved `docs/development/` -> `docs/archive/development/`
  - moved `docs/review/` -> `docs/archive/review/`
- Normalized markdown naming to lowercase kebab-case for active docs and archive files.
- Added `docs/README.md` as docs table of contents and maintenance guide.
- Added `docs/spec.md` to serve as the architecture specification baseline.
- Added `docs/lessons.md` for persistent development lessons.
- Added folder-level READMEs for `tests/`, `benchmarks/`, and `examples/`.
- Updated `README.md` links and test/benchmark entry-point commands.
- Updated `CLAUDE.md` with:
  - docs naming convention
  - output policy
  - context and memory-preservation workflow
  - corrected test command examples
- Updated `mkdocs.yml` to remove missing API pages and point to existing docs.

### Remaining for Phase 2

- Validate MkDocs rendering in an environment with `mkdocs` installed (CLI unavailable in current shell).

## Phase 2 Plan: Output Path Standardization

### Scope
- consolidate generated artifacts under `outputs/`
- remove hardcoded mixed output paths from tests/benchmarks/examples
- keep reproducibility via explicit overrides

### Checklist
| Item | Status | Notes |
|---|---|---|
| 1. Add shared output-path utility | [x] | Added `isoster/output_paths.py` with `resolve_output_directory()` |
| 2. Migrate integration/validation/real-data test artifacts to standardized output folders | [x] | Updated `tests/integration/*`, `tests/validation/*`, `tests/real_data/*` |
| 3. Migrate benchmark outputs to standardized folders | [x] | Updated `bench_vs_photutils.py` and `bench_numba_speedup.py` |
| 4. Migrate example outputs to standardized folders | [x] | Updated `examples/basic_usage.py` and Huang2013 default output path |
| 5. Update folder README guidance for reproducible output roots | [x] | Added `ISOSTER_OUTPUT_ROOT` note in tests/benchmarks/examples READMEs |
| 6. Verify with collection and targeted execution | [x] | `pytest --collect-only -q` + targeted tests and script checks passed |
| 7. Record phase review and residual risks | [x] | Added below |

### Review

- Added a reusable path helper in `isoster/output_paths.py`.
- Replaced `tests/qa_outputs` and `outputs/figures` writers in active code paths with standardized `outputs/<category>/<run>` layout.
- Removed legacy `tests/qa_outputs/` directory to prevent further artifact drift.
- Updated benchmark scripts to support explicit `--output` override and standardized defaults.
- Fixed a benchmark import-path issue in `bench_vs_photutils.py` so CLI invocation now works.

Residual risks:
- `qa/` reference materials remain as historical content and still describe legacy generation workflows.
- `mkdocs` CLI is unavailable in current shell, so docs rendering was not validated here.

## Phase 3 Plan: Adopt uv Environment Management

### Scope
- adopt `uv` for dependency locking and environment sync
- align dependency metadata with actual runtime/test/docs usage
- install project dependencies with docs tooling (`mkdocs`) locally

### Checklist
| Item | Status | Notes |
|---|---|---|
| 1. Audit imports vs declared dependencies | [x] | Audited runtime/test/benchmark/docs imports and dependency gaps |
| 2. Update `pyproject.toml` dependency metadata for uv workflows | [x] | Added `pydantic` runtime dependency and `dev`/`docs` extras with Python markers |
| 3. Update docs (`README.md`, `CLAUDE.md`) with uv-first commands | [x] | Updated install/test/benchmark/docs command examples to uv-first |
| 4. Generate/update lockfile with `uv lock` | [x] | Generated `uv.lock` |
| 5. Install dependencies via `uv sync` including docs tooling | [x] | Installed core + dev + docs extras (including mkdocs) into `.venv` |
| 6. Verify with uv-based commands | [x] | `uv run pytest --collect-only -q` passed; `mkdocs --version` confirmed from `.venv` |
| 7. Record review and lessons | [x] | Updated this file, `docs/lessons.md`, and journal entry |

### Review

- `uv` is practical for this repo and now configured end-to-end:
  - dependency metadata in `pyproject.toml`
  - lockfile in `uv.lock`
  - synced environment in `.venv`
- Runtime dependency gap fixed: `pydantic` is now declared in core dependencies.
- Added optional extras:
  - `dev`: pytest, photutils, numba, ruff, pre-commit
  - `docs`: mkdocs, mkdocs-material, pymdown-extensions
- Added Python markers for some dev tools to keep resolution compatible with `requires-python >=3.8`.
- Updated project docs to uv-first command paths.

Notes:
- In this environment, `uv` commands required elevated execution due sandbox/system-configuration constraints.
