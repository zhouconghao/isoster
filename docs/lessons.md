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
- For large campaign runs across many inputs, run extraction and QA stages so that per-method/per-case failures are recorded in manifests instead of raising; batch orchestration should continue and emit aggregate failure statistics at the end.
- For QA afterburner pipelines, use extraction-manifest method status as the source of truth; do not attempt method/comparison QA for methods marked failed, even if stale profile files exist.
- When serializing timeout results from `subprocess.TimeoutExpired`, normalize `stdout`/`stderr` values to text first because they may be bytes and can break JSON log writing.
- For long-running batch jobs, default to skip-existing behavior with explicit `--update` reruns to avoid wasting compute after partial completion.

## 2026-02-20

- For Huang2013 mock headers, apply a fixed `PA - 90 deg` initialization correction in the Huang2013 helper layer to align with image-space PA used by fitting libraries.
- For initialization from model headers, `RE_PX1` can be too small at high redshift/noisy mocks; keep a conservative fallback default (`6 px`) and a minimum clamp (`>= 3 px`) to avoid degenerate startup ellipses.
- For Huang2013 extraction retries, keep retry policy explicit and symmetric across methods: fixed attempt cap, deterministic `sma0`/`astep` increments, unchanged `maxsma`, and persisted per-attempt metadata for post-run diagnosis.
- For Huang2013 QA profile panels, compute y-axis limits from finite profile values (not error bars), and use data-driven axis-ratio limits with margins instead of fixed `[0, 1]` spans.
- For stop-code readability in dense QA panels, keep nominal `stop=0` filled but render `stop!=0` with open markers and distinct dark colors to avoid ambiguity in monochrome-like styling.
- QA afterburners that consume extraction run JSON should tolerate `runtime: null` payloads from failed runs; runtime parsing must be type-checked before `.get(...)`.
- For comparison QA panels, apply robust percentile y-limits for `dI/I`, centroid, and PA panels so a handful of outer outliers do not flatten the main trend; keep stop-code ordering with `stop=0` first in legends for quick scanning.
- Keep explicit style mappings for non-core photutils stop codes (for example `stop=4`, `stop=5`) to avoid fallback black/open-circle ambiguity in monochrome QA plots.
- For shared 1-D QA x-axes, add a small right-edge margin beyond the last valid isophote point to prevent visual crowding against the panel boundary.
