# ISOSTER Documentation

Use this page as the entry point and map for project documentation.

## Public Documents

- `README.md` (repo root): quick start and public overview
- `docs/SPEC.md`: compatibility entry point for agent workflows; points to the canonical technical specification
- `docs/01-user-guide.md`: practical usage guidance, QA/comparison entry points, and canonical stop-code reference
- `docs/02-configuration-reference.md`: all configuration parameters and guidelines
- `docs/03-algorithm.md`: fitting and sampling implementation notes
- `docs/04-architecture.md`: architecture, interfaces, and design decisions
- `docs/05-testing.md`: testing and benchmark directives
- `docs/06-qa-functions.md`: QA plotting functions, generation standard, usage, options, and examples
- `docs/07-lsb-features.md`: design and implementation of the LSB auto-lock and outer-region center regularization features
- `docs/08-outer-regularization.md`: publication-grade reference for the outer-region Tikhonov regularization (math, algorithm, config, benchmarks)
- `docs/09-exhausted-benchmark.md`: exhausted benchmark campaign framework reference (YAML schema, arms, output layout, model-evaluation standard, scoring, adapter recipe)
- `docs/10-multiband.md`: experimental multi-band interface, CLI, I/O, and benchmark notes

## Agent-Internal Documents

Agent-internal docs live in `docs/agent/` and are not tracked in git:

- `docs/agent/todo.md`: active execution checklist and review notes.
- `docs/agent/lessons.md`: development lessons to avoid repeated mistakes.
- `docs/agent/future.md`: long-term upgrades and optimization roadmap.
- `docs/agent/qa-figures.md`: QA figure layout and style conventions.
- `docs/agent/journal/`: chronological project journal notes.
- `docs/agent/archive/`: obsolete or superseded documentation.

## Maintenance Rules

- Public docs use numbered-index filenames (`NN-name.md`) and are served by mkdocs.
- Agent-internal docs go in `docs/agent/` and are gitignored.
- Update links in `README.md`, `mkdocs.yml`, and `CLAUDE.md` when files move.
