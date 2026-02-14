# 2026-02-14 Phase 10 QA Figure Clean-Context Handoff

## Current State

- Active branch: `phase10-qa-prep-p1`
- Working tree: dirty (not committed yet)
- Files currently modified in this branch:
  - `benchmarks/README.md`
  - `benchmarks/baselines/run_benchmark_gate.py`
  - `benchmarks/performance/bench_efficiency.py`
  - `benchmarks/performance/bench_numba_speedup.py`
  - `benchmarks/performance/bench_vs_photutils.py`
  - `docs/todo.md`

## What Is Completed

1. Phase 9 is complete and merged to `main`.
2. Phase 10 planning/tracking is recorded in `docs/todo.md`.
3. Parallel prep tasks completed:
   - Plot/cache env stabilization in benchmark scripts (`XDG_CACHE_HOME`, `MPLCONFIGDIR` defaults under `outputs/tmp`).
   - Full lock-refresh + full gate command sequence added in `benchmarks/README.md`.
   - QA-hook scaffold added to `run_benchmark_gate.py` (no external QA execution yet).
4. QA-hook scaffold verification completed:
   - Gate still passes in quick mode.
   - `benchmark_gate_report.json` now includes:
     - top-level `qa_hook_scaffold`
     - per-case `qa_artifacts.qa_figure_path` placeholder fields

## What Is Next (Primary)

Integrate the real QA figure generation into `benchmarks/baselines/run_benchmark_gate.py` via the scaffold, while keeping gate semantics unchanged:

- pass/fail driven by locked efficiency + locked 1-D
- 2-D metrics remain quantitative system-level diagnostics

## Resume Commands

```bash
# 1) Confirm branch and pending edits
git status --short --branch

# 2) Read active plan and handoff
sed -n '200,360p' docs/todo.md
sed -n '1,260p' docs/journal/2026-02-14-phase10-qa-figure-clean-context-handoff.md

# 3) Inspect current scaffold implementation
sed -n '1,260p' benchmarks/baselines/run_benchmark_gate.py
sed -n '260,620p' benchmarks/baselines/run_benchmark_gate.py

# 4) Validate current baseline behavior before wiring QA execution
ruff check benchmarks/baselines/run_benchmark_gate.py
.venv/bin/python benchmarks/baselines/run_benchmark_gate.py --quick --n-runs 1 --efficiency-lock benchmarks/baselines/efficiency_thresholds_quick_2026-02-14.json

# 5) After QA integration, rerun gate smoke and collect evidence
ruff check benchmarks/baselines/run_benchmark_gate.py
.venv/bin/python benchmarks/baselines/run_benchmark_gate.py --quick --n-runs 1 --efficiency-lock benchmarks/baselines/efficiency_thresholds_quick_2026-02-14.json --qa-script <PATH_TO_QA_SCRIPT>
```

## First Prompt for Next Session

Use this as the first prompt in a fresh session:

```text
Continue Phase 10 from docs/todo.md on branch phase10-qa-prep-p1.

Goal: wire the real QA figure generation into benchmarks/baselines/run_benchmark_gate.py using the existing QA-hook scaffold (`--qa-script`, `--qa-output-subdir`) while keeping gate semantics unchanged:
- locked efficiency + locked 1-D remain pass/fail criteria,
- 2-D remains system-level quantitative diagnostics with caveat.

Requirements:
1) Execute the QA script per Phase-4 case and write figures under outputs/benchmarks_performance/benchmark_gate/<qa_output_subdir>/.
2) Persist per-case QA artifact paths into benchmark_gate_report.json.
3) Keep behavior robust if QA script is missing/fails (report status clearly, do not silently pass).
4) Run quick gate smoke with QA enabled and update docs/todo.md evidence + output paths.
5) Use the installed skills for plotting/figure standards:
   - ~/.claude/skills/matplotlib/SKILL.md
   - ~/.claude/skills/scientific-visualization/SKILL.md

Do not merge to main in this session.
```

## Notes

- If `uv run` is unstable in this environment, use `.venv/bin/python` and record the fallback in evidence logs.
- Keep generated artifacts under `outputs/` only.
