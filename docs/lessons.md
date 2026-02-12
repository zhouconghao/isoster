# Lessons Learned

## 2026-02-11

- Keep stable documentation separate from historical development notes.
- Use one naming convention for markdown files: lowercase kebab-case.
- Define folder responsibilities explicitly (`tests`, `benchmarks`, `examples`, `outputs`) to prevent drift.
- Keep `docs/todo.md` as an active execution checklist with a review section for each phase.
- Use a shared output-path helper to avoid path drift across tests, benchmarks, and examples.
- Keep output-root override centralized with `ISOSTER_OUTPUT_ROOT` for reproducibility across environments.
- For `uv` adoption, lock and sync should be part of the migration deliverable (`uv.lock` + `.venv` sync verification).
- When supporting older Python in `requires-python`, apply markers on dev-only tools that require newer Python.
- For noiseless single-Sersic truth comparisons, prefer accurate `b_n` computation (for example, `gammaincinv`) over coarse linear approximations.
- Keep test/benchmark acceptance criteria quantitative and evidence-based; define thresholds from measured baseline runs.
- Do not gate metric assertions behind `if valid.sum() > 0`; assert minimum valid-point counts first so regressions cannot pass silently.
- Benchmark scripts should emit JSON + CSV + metadata by default and place outputs under `outputs/benchmarks_performance/...`.
- Profiling scripts should persist `.prof` files and parsed hotspot summaries under `outputs/benchmarks_profiling/...`, not only print to console.
- For numba benchmarking, report both mean and steady-state speedup plus variability; first-run timing can bias conclusions.
- After detecting flagged variability/slowdown cases, run case-specific profiling drills that replay those exact benchmark cases and persist per-case `.prof` artifacts.
- When evaluating drill variability deltas, compare against both source numba CV and source no-numba CV to avoid mis-attributing runtime noise.
- For optional external generators, prefer preset-driven adapters with explicit override keys so workflows remain reproducible and auditable.
- Keep reusable preset input templates in `examples/` and point preset defaults to those tracked files.
