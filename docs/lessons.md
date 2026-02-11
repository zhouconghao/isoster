# Lessons Learned

## 2026-02-11

- Keep stable documentation separate from historical development notes.
- Use one naming convention for markdown files: lowercase kebab-case.
- Define folder responsibilities explicitly (`tests`, `benchmarks`, `examples`, `outputs`) to prevent drift.
- Keep `docs/todo.md` as an active execution checklist with a review section for each phase.
- Use a shared output-path helper to avoid path drift across tests, benchmarks, and examples.
- Keep output-root override centralized with `ISOSTER_OUTPUT_ROOT` for reproducibility across environments.
