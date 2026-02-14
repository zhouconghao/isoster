# ISOSTER Documentation

Use this page as the entry point and map for project documentation.

## Core Documents

- `README.md` (repo root): quick start and public overview
- `docs/spec.md`: architecture, interfaces, and design decisions
- `docs/user-guide.md`: practical usage guidance and canonical stop-code reference
- `docs/algorithm.md`: fitting and sampling implementation notes
- `docs/future.md`: long-term upgrades and optimization roadmap
- `docs/todo.md`: active implementation plan and review checklist
- `docs/lessons.md`: recurring lessons and process guardrails

## Docs Structure

This folder separates stable documentation from historical working notes.

### Stable Documents

- `index.md`: MkDocs home page and documentation map.
- `spec.md`: canonical technical architecture and design decisions.
- `user-guide.md`: user-facing usage guide.
- `algorithm.md`: implementation-level fitting and sampling notes.
- `stop-codes.md`: compatibility redirect to canonical stop-code section in `user-guide.md`.
- `future.md`: long-term upgrades and research roadmap.
- `todo.md`: active execution checklist and review notes.
- `lessons.md`: development lessons to avoid repeated mistakes.

### Historical Documents

- `archive/development/`: historical optimization plans, implementation notes, and one-off analyses.
- `archive/review/`: historical code review and comparison artifacts.
- `archive/*.md`: archived legacy root-level docs preserved for traceability.
- `journal/`: chronological project journal notes.

## Maintenance Rules

- Add new stable documentation in this folder root when it reflects current behavior.
- Move obsolete or superseded documentation to `archive/` instead of deleting it.
- Update links in `README.md`, `mkdocs.yml`, and `CLAUDE.md` when files move.
