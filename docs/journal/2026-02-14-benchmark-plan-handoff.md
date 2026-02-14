# 2026-02-14 Benchmark Plan Handoff

## What Was Done

- Reviewed active benchmark scripts and threshold policy:
  - `benchmarks/performance/bench_efficiency.py`
  - `benchmarks/performance/bench_numba_speedup.py`
  - `benchmarks/performance/bench_vs_photutils.py`
  - `benchmarks/baselines/collect_phase4_profile_baseline.py`
  - `benchmarks/baselines/phase4_profile_thresholds_2026-02-11.json`
- Reviewed external mock generator behavior in:
  - `/Users/mac/Dropbox/work/project/otters/isophote_test/mockgal.py`
- Persisted two user-corrected constraints into `CLAUDE.md`:
  1. force `mockgal.py` benchmark/test workflows to use `--engine libprofit`
  2. treat 2-D residual metrics as system-level diagnostics (extraction + model generation)
- Updated planning docs:
  - `docs/test-benchmark-improvement-plan.md` (new 2026-02-14 session update section)
  - `docs/todo.md` (new Phase 9 checklist)
- Renamed old output folder:
  - `outputs/figures` -> `outputs/examples_ea_harmonics`

## Key Decisions

- Keep threshold policy evidence-based: baseline measure first, then lock.
- Separate metric interpretation:
  - 1-D metrics are primary for extraction algorithm accuracy.
  - 2-D metrics are system-level and may fail due to model reconstruction path.
- mockgal integration must be explicit about backend (`libprofit`) to avoid backend drift.

## Remaining Work (Next Session)

1. Implement `mockgal_adapter` libprofit enforcement and backend preflight checks.
2. Unify remaining approximate-`b_n` generators to exact `gammaincinv`-based truth helpers.
3. Add benchmark collector/gate that combines:
   - efficiency regression metrics
   - baseline-locked 1-D QA metrics
   - 2-D system-level diagnostics
4. Run quick benchmark smoke then lock/update baseline files from measured outputs only.

## Resume Commands

```bash
# confirm current branch and pending edits
git status --short --branch

# inspect updated planning docs
sed -n '1,260p' docs/todo.md
sed -n '1,360p' docs/test-benchmark-improvement-plan.md

# inspect mockgal adapter before implementing constraints
sed -n '1,320p' benchmarks/baselines/mockgal_adapter.py

# quick benchmark smoke (after implementation)
uv run python benchmarks/performance/bench_efficiency.py --quick --n-runs 1
uv run python benchmarks/performance/bench_numba_speedup.py --n-runs 2 --scale-factor 1.0
uv run python benchmarks/performance/bench_vs_photutils.py --quick
```

## Suggested Next-Session Prompt

Continue Phase 9 from `docs/todo.md` and implement items 4-6 in order:
1) enforce `--engine libprofit` in `benchmarks/baselines/mockgal_adapter.py` with explicit preflight failure when libprofit is unavailable,
2) replace remaining approximate Sersic `b_n` benchmark/test truth calculations with exact `gammaincinv`-based helpers,
3) add a baseline-locked benchmark gate that reports efficiency metrics and quantitative QA metrics (1-D primary, 2-D system-level caveat preserved),
then run quick benchmark smoke commands and update `docs/todo.md` + `docs/lessons.md` with evidence and exact output paths.
