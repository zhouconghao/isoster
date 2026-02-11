# Docs Structure

This folder separates stable documentation from historical working notes.

## Stable Documents

- `index.md`: MkDocs home page and doc entry point.
- `spec.md`: canonical technical architecture and design decisions.
- `user-guide.md`: user-facing usage guide.
- `algorithm.md`: algorithm internals and equations.
- `stop-codes.md`: stop code reference and diagnostics.
- `advanced-optimization.md`: focused optimization notes still relevant to current code.
- `todo.md`: active execution plan with checklist and review section.
- `lessons.md`: development lessons to avoid repeated mistakes.

## Historical Documents

- `archive/development/`: historical optimization plans, implementation notes, and one-off analyses.
- `archive/review/`: historical code review and comparison artifacts.
- `archive/*.md`: archived legacy root-level docs preserved for traceability.
- `journal/`: chronological project journal notes.

## Naming Convention

- Use lowercase kebab-case for markdown files (for example: `stop-codes.md`).
- Reserve uppercase file names only when required by external conventions.

## Maintenance Rules

- Add new stable documentation in this folder root when it reflects current behavior.
- Move obsolete or superseded documentation to `archive/` instead of deleting it.
- Update links in `README.md`, `mkdocs.yml`, and `CLAUDE.md` when files move.
