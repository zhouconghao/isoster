# 2026-02-11: Phase 3 uv Adoption

## Scope

Adopt `uv` as the primary dependency and environment manager for this repository.

## Changes

- Updated dependency metadata in `pyproject.toml`:
  - Added missing runtime dependency: `pydantic>=2.0`
  - Added extras:
    - `dev`: `pytest`, `photutils`, `numba`, `ruff`, `pre-commit`
    - `docs`: `mkdocs`, `mkdocs-material`, `pymdown-extensions`
  - Added Python markers for dev tools that require newer Python versions.
- Updated command documentation to uv-first workflows:
  - `README.md`
  - `CLAUDE.md`
  - `tests/README.md`
  - `benchmarks/README.md`
  - `examples/README.md`
- Generated lockfile: `uv.lock`.
- Synced environment: `.venv` created with core + dev + docs extras.

## Installation Outcome

Installed packages include:
- runtime: numpy, scipy, astropy, matplotlib, pyyaml, pydantic
- development: pytest, photutils, numba, ruff, pre-commit
- docs: mkdocs, mkdocs-material, pymdown-extensions

## Verification

- `UV_CACHE_DIR=.uv-cache uv run pytest --collect-only -q` passed.
- `.venv/bin/mkdocs --version` returned `mkdocs 1.6.1`.

## Environment Note

In this execution environment, `uv` commands needed elevated execution and a local cache override (`UV_CACHE_DIR=.uv-cache`) due sandbox restrictions.
