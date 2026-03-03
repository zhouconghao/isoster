---
date: 2026-03-02
repo: isoster
branch: doc-alignment-2026-03-03
tags:
  - journal
  - snapshot
  - handover
  - huang2013
---

## Progress

- Merged the example reorganization housekeeping into `main` as merge commit `fd71977` after feature commit `0d661d5`.
- Verified the updated examples with `uv sync --extra dev --extra docs`, a basic-usage smoke run, and a LegacySurvey smoke run.
- Left a separate docs-only worktree on `doc-alignment-2026-03-03` with uncommitted changes in the core docs.

## Lessons Learned

- Command shells needed explicit `source ~/.zshrc` to see `/Library/TeX/texbin` on `PATH`.
- The example reorganization required both code-path fixes and historical command/doc cleanup.
- Start the Huang2013 follow-up from a clean branch after resolving the current docs-only worktree state.

## Key Issues

- Uncommitted docs changes remain in `docs/algorithm.md`, `docs/configuration-reference.md`, `docs/index.md`, `docs/spec.md`, `docs/user-guide.md`, and deletion of `docs/stop-codes.md`.
- Untracked paths remain: `development/`, `docs/plan/`, `site/`.
- Next session should update the Huang2013 example, but only after deciding what to do with the current `doc-alignment-2026-03-03` worktree changes.
