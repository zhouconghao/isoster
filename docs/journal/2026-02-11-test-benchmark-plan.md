# 2026-02-11 Test and Benchmark Planning Snapshot

## Summary
- Created a detailed roadmap for test and benchmark upgrades in `docs/test-benchmark-improvement-plan.md`.
- Added a new Phase 4 checklist in `docs/todo.md` for execution tracking.
- Persisted user-defined directives in `CLAUDE.md`:
  - canonical M51 dataset path and basic test naming (`m51_test`)
  - future libprofit-based mock generation source (`mockgal.py` in `isophote_test` repo)
  - accurate `b_n` requirement for noiseless single-Sersic truth comparisons
  - quantitative criteria requirement for tests/benchmarks
- Updated `docs/spec.md` and `docs/lessons.md` to align with metric-driven validation policy.

## Next Execution Steps
1. Run Phase 0 baseline measurements and lock thresholds from data.
2. Implement test hardening changes (remove false-pass patterns, add minimum valid-point assertions).
3. Add missing public API and CLI tests.
4. Standardize benchmark/profiling outputs and metadata artifacts.
