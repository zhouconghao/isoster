---
date: 2026-03-03
repo: isoster
branch: doc-alignment-2026-03-03
tags:
  - journal
  - isoster
  - examples
  - docs
---

## Progress

- Read the current repo guidance and example-related docs before editing: `CLAUDE.md`, `README.md`, `docs/spec.md`, `docs/user-guide.md`, `docs/configuration-reference.md`, and `examples/README.md`.
- Created a feature branch `chore/examples-housekeeping` for the example housekeeping work.
- Updated example scripts to match the reorganized folder layout and the repo-level `data/` directory.
- Standardized default example outputs to `outputs/<example-folder>/` for `example_basic_usage`, `example_cog`, and `example_ls_highorder_harmonic`.
- Kept `example_huang2013` as the exception: case outputs remain under the external Huang2013 root and campaign summaries remain under `outputs/`.
- Rewrote `examples/README.md` to organize examples by subfolder, describe each example, document run commands, and document default outputs.
- Updated `examples/example_ls_highorder_harmonic/README.md` to document the main script, exploratory scripts, and default output locations.
- Updated `examples/example_huang2013/README.md` and `examples/example_huang2013/real-huang2013-requirements.md` to use the new folder path.
- Removed the stale local file `examples/example_huang2013/mockgal.py` after confirming there was no live example-script dependency on it.
- Updated supporting tracked docs that still pointed to removed example paths: `CLAUDE.md`, `docs/spec.md`, `docs/testing.md`, `data/README.md`, and `tests/README.md`.
- Cleaned obsolete example-path strings in internal markdown notes under `docs/journal/` and `docs/todo.md`.
- Fixed example-script lint issues reported by Ruff and reran lint successfully.
- Committed the feature work as `0d661d5` with message `chore: update examples for folder reorganization`.
- Merged `chore/examples-housekeeping` back into `main` with merge commit `fd71977`.

## Lessons Learned

- The shell used for commands did not pick up the updated `~/.zshrc` automatically; verification commands needed an explicit `source ~/.zshrc` to expose `/Library/TeX/texbin` on `PATH`.
- Example smoke tests were blocked by environment setup rather than code-path errors until `uv sync --extra dev --extra docs` installed `photutils`.
- `example_basic_usage` required LaTeX on `PATH` because the plotting configuration enables `text.usetex` when LaTeX is available.
- The examples reorganization affected both runtime code and historical inline command examples; doc-only cleanup was not enough.
- Null-delimited file pipelines are safer than plain shell word splitting for bulk markdown rewrites.

## Key Issues

- `main` was left ahead of `origin/main` by 2 commits after the feature commit and merge; push is still pending.
- Unrelated untracked paths were left untouched: `docs/plan/` and `site/`.
- The current workspace branch is now `doc-alignment-2026-03-03`; confirm whether follow-up work should continue there or return to `main`.
