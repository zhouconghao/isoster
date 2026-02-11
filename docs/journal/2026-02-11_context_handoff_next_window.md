# Context Handoff for New Window

Use this file to resume work in a fresh context window.

## Current Repository State

- Branch: `main`
- Architecture review and uv migration: complete
- Lockfile: `uv.lock`
- Local environment: `.venv` synced with core + dev + docs extras

## Mandatory Workflow (Agent)

1. Read first:
   - `CLAUDE.md`
   - `docs/spec.md`
   - `docs/todo.md`
   - `docs/lessons.md`
   - latest entries in `docs/journal/`
2. Use `uv` for all Python commands.
3. Keep outputs under `outputs/` only.
4. Update journal/todo/spec as changes are made.

## Fast Resume Commands

```bash
# Ensure environment is synced
uv sync --extra dev --extra docs

# Sanity checks
uv run pytest --collect-only -q
uv run mkdocs --version

# Optional docs preview
uv run mkdocs serve
```

## Suggested Next Work Item

Start the real-data testing phase and define a concrete execution matrix (datasets, quality metrics, output naming, and reproducible run commands).

## Notes for Context Compression

If context gets large again, record progress snapshots in `docs/journal/` and keep `docs/todo.md` status up to date before switching windows.
