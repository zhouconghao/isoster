# CLAUDE.md

This file provides guidance to coding agents when working with code in this repository.

## Project Overview

ISOSTER (ISOphote on STERoid) is an accelerated Python library for elliptical isophote fitting in galaxy images. It provides 10-15x faster performance compared to `photutils.isophote` using vectorized path-based sampling via scipy's `map_coordinates`.

## Non-negotiable Rules for developing

- Always create a new branch for new features and new development. Do not merge back into the main branch unless I approve it.
- It is essential to provide informative and concise docstrings and inline comments.
- Warn the users when the context window has <30% left. Remind the users to save the conversation and start fresh. Also propose ways to compact the conversation and save the current progress in files.
- Keep record of development progress, important lessons, and critical decisions in markdown files in `docs/journal/`.
- Keep `docs/spec.md` updated for architecture and interface changes.
- Keep active execution checklist and end-of-phase review in `docs/todo.md`.
- Use lowercase kebab-case for markdown file names (for example, `stop-codes.md`).
- Generated artifacts must be written under `outputs/` and not mixed into source folders.
- Use `uv` as the default tool for dependency management and environment execution.

## Context and Memory Preservation

When a task is long or the context window is becoming constrained:

1. Write a concise status snapshot to `docs/journal/` (what was done, what remains, blockers).
2. Update `docs/todo.md` checklist status and review notes.
3. Update `docs/spec.md` if architecture or interfaces changed.
4. Update `docs/lessons.md` when a correction yields a reusable lesson.
5. Before ending, include exact file paths and commands needed to resume.

## Build and Test Commands

```bash
# Sync project environment (core + development + docs tooling)
uv sync --extra dev --extra docs

# Run all tests (main package)
uv run pytest tests/

# Run a single test file
uv run pytest tests/unit/test_fitting.py

# Run reference implementation tests (photutils-compatible)
uv run pytest reference/tests/

# Run with verbose output
uv run pytest tests/ -v

# CLI usage
uv run isoster image.fits --output isophotes.fits --config config.yaml

# Build docs
uv run mkdocs serve
```

## uv Workflow Rules

Follow these rules for all Python environment and dependency work in this repository:

1. Use `uv` as the single workflow for environment and dependency management.
2. Do not use `pip`, `poetry`, or `conda` commands for project dependency changes.
3. Install/sync environment with:
   - `uv sync --extra dev --extra docs`
4. Run project commands through the managed environment:
   - `uv run pytest ...`
   - `uv run python ...`
   - `uv run isoster ...`
   - `uv run mkdocs ...`
5. When adding/removing dependencies, update `pyproject.toml` and then run:
   - `uv lock`
   - `uv sync --extra dev --extra docs`
6. Keep `uv.lock` committed and up to date with dependency changes.
7. For tools that are not compatible with all Python versions in `requires-python`,
   use dependency markers in `pyproject.toml` (for example `python_version >= '3.9'`).
8. Minimum verification after dependency changes:
   - `uv run pytest --collect-only -q`
   - `uv run mkdocs --version`

## Documentation Index

| Document | Description |
|----------|-------------|
| `docs/spec.md` | Architecture, interfaces, and design decisions |
| `docs/user-guide.md` | Usage guidance, stop-code reference, public API |
| `docs/algorithm.md` | Fitting and sampling implementation notes |
| `docs/configuration-reference.md` | All configuration parameters and guidelines |
| `docs/testing.md` | Testing and benchmark directives |
| `docs/qa-figures.md` | QA figure layout and style conventions |
| `docs/future.md` | Long-term upgrades and research roadmap |
| `docs/todo.md` | Active execution checklist and review notes |
| `docs/lessons.md` | Development lessons and process guardrails |

## Testing and Benchmark Directives

See `docs/testing.md`.

## QA Figure Rules

See `docs/qa-figures.md`.

## Architecture and Key Concepts

See `docs/spec.md` and `docs/algorithm.md`.

## Public API and Configuration

See `docs/user-guide.md` and `docs/configuration-reference.md`.

## Mock Generation

For high-fidelity mock generation with PSF convolution and realistic noise, use `examples/huang2013/mockgal.py` (libprofit-based). Force `--engine libprofit` and do not rely on astropy fallback rendering.
