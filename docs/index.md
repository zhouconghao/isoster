# ISOSTER Documentation

Use this page as the entry point and map for project documentation.

## Core Documents

- `README.md` (repo root): quick start and public overview
- `docs/spec.md`: architecture, interfaces, and design decisions
- `docs/user-guide.md`: practical usage guidance and canonical stop-code reference
- `docs/configuration-reference.md`: all configuration parameters and guidelines
- `docs/algorithm.md`: fitting and sampling implementation notes
- `docs/testing.md`: testing and benchmark directives
- `docs/qa-figures.md`: QA figure layout and style conventions
- `docs/future.md`: long-term upgrades and optimization roadmap

## Docs Structure

This folder separates stable documentation from historical working notes.

### Stable Documents

- `index.md`: MkDocs home page and documentation map.
- `spec.md`: canonical technical architecture and design decisions.
- `user-guide.md`: user-facing usage guide.
- `configuration-reference.md`: configuration parameters and guidelines.
- `algorithm.md`: implementation-level fitting and sampling notes.
- `testing.md`: testing and benchmark directives.
- `qa-figures.md`: QA figure layout and style conventions.
- `stop-codes.md`: compatibility redirect to canonical stop-code section in `user-guide.md`.
- `future.md`: long-term upgrades and research roadmap.

### Internal Documents (not tracked in git)

- `todo.md`: active execution checklist and review notes.
- `lessons.md`: development lessons to avoid repeated mistakes.
- `journal/`: chronological project journal notes.
- `review/`: code review artifacts.
- `plans/`: design and implementation plans.

## Maintenance Rules

- Add new stable documentation in this folder root when it reflects current behavior.
- Move obsolete or superseded documentation to `archive/` instead of deleting it.
- Update links in `README.md`, `mkdocs.yml`, and `CLAUDE.md` when files move.
