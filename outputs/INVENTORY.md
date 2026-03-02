# outputs/ Inventory

> Last updated: 2026-03-02
> `outputs/` is **not** git-tracked (see `.gitignore`). This file **is** tracked as a reference.

---

## Naming Rules

- Prefix must be **singular**: `benchmark_`, `test_`, `example_`
- Never use plural prefix (`benchmarks_`, `tests_`, `examples_`)
- New runs append ISO date suffix if multiple runs exist:
  `benchmark_vs_photutils_2026-03-01/`
- Subdirectory per benchmark script under each prefix, e.g.:
  `benchmark_performance/bench_vs_autoprof/`

---

## Active — benchmark_

| Folder | Script | Approx Size | Notes |
|--------|--------|------------:|-------|
| `benchmark_ic3370_exhausted/` | `benchmarks/exhausted/run_benchmark.py` | 119 MB | 39-config IC3370 sweep (canonical run); report.md present |
| `benchmark_performance/` | `benchmarks/performance/` (multiple scripts) | 31 MB | bench_vs_autoprof, bench_vs_photutils, bench_efficiency, bench_numba, gate runs |
| `benchmark_profiling/` | `benchmarks/profiling/` (multiple scripts) | 828 KB | cProfile hotspots, numba flagged cases |

---

## Active — tests_

| Folder | Script | Approx Size | Notes |
|--------|--------|------------:|-------|
| `tests_integration/` | `tests/integration/` | 5.9 MB | Integration test artifacts + baseline metrics |
| `tests_validation/` | `tests/validation/` | 264 KB | Photutils comparison artifacts |

---

## Cleanup Log

| Date | Action |
|------|--------|
| 2026-03-02 | Renamed `benchmarks_performance/` → `benchmark_performance/` |
| 2026-03-02 | Renamed `benchmarks_profiling/` → `benchmark_profiling/` |
| 2026-03-02 | Deleted empty `benchmark_results/` |
| 2026-03-02 | Deleted empty `tests_real_data/` |
| 2026-03-02 | Deleted `tmp/` (6.2 MB matplotlib/xdg cache) |
| 2026-03-02 | Deleted `tmp_eso185_g054_validation/` (24 KB temp prefix) |
| 2026-03-02 | Deleted loose root files: `convergence_diagnostic.png`, `huang2013_test_fixes.txt`, `qa_m51_test.png` |
| 2026-03-02 | Deleted all examples_, isofit_mode_comparison/, legacysurvey_*, huang2013_*, ngc* folders |
