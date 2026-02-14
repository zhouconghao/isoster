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
- For iterative science QA workflows, split fitting/extraction and QA rendering into separate scripts so QA can be rerun independently as an afterburner without recomputing profiles.
- Profiling wrappers used in retry loops must disable cProfile in a `finally` block; otherwise failed attempts can leave an active profiler and break subsequent attempts.
- For photutils runs on large external mock frames, use a header-driven `maxsma` default (for example derived from `RE_PX*`) plus progressive fallback retries to avoid NaN-coordinate failures at extreme radii.

## 2026-02-14

- Keep stop-code policy synchronized across driver gating, config/doc tables, and API assertions to avoid compatibility drift.
- Features that depend on prior state (for example central regularization) require explicit state plumbing in driver loops, not only per-isophote support in core fitting.

- For `mockgal.py` benchmark/test workflows, explicitly force `--engine libprofit`; do not depend on fallback backend auto-selection.
- Interpret 2-D residual metrics as system-level checks (extraction + model reconstruction together), not extraction-only performance indicators.
- For externally generated mock benchmarks, enforce backend selection at adapter validation time and reject conflicting pass-through overrides before process launch.
- Keep separate efficiency lock files for quick smoke and full suites; lock from measured artifacts of the matching run shape to avoid false gate failures from case-set mismatch.
- When `uv run` is unstable in the execution environment, use the project interpreter (`.venv/bin/python`) and record that fallback in evidence logs.
- In LaTeX-rendered matplotlib figures, any literal `%` in titles/labels must be escaped (or preprocessed) to avoid silent text truncation.
- For stacked image panels with mixed colorbar usage, reserve a dedicated colorbar column per row to prevent panel-width drift and axis misalignment.
- For side-by-side data/model QA panels, derive display transform parameters from the data once and reuse them for the model to preserve absolute-value comparability.
- When QA panels split centroid behavior into `dx` and `dy`, include `x0_err`/`y0_err` error bars by default; for axis ratio, use `ellip_err` fallback to `eps_err` when available.
- Strict efficiency lock thresholds copied directly from a single timing run can be brittle across immediate reruns; expect small runtime jitter and record this explicitly in gate evidence when interpreting failures.
- Efficiency timing gates should include measured per-case jitter tolerance (for example, based on `std_time`) to avoid false failures from millisecond-level runtime noise.
- Do not run concurrent benchmark gate commands when collecting timing evidence; resource contention can inflate runtimes and produce invalid comparisons.
- For this repository's current gate stability, a more tolerant default (`jitter_sigma=3.0`, `jitter_floor=0.002s`) is a practical baseline for avoiding false efficiency failures while preserving regression detection.
- For recurring benchmark-gate knobs, keep a versioned project defaults JSON and let CLI options override it; this reduces command noise and keeps policy reproducible.
