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

## Active — examples_

| Folder | Script | Approx Size | Notes |
|--------|--------|------------:|-------|
| `examples_basic_usage/` | `examples/basic_usage.py` | 100 KB | Basic usage QA figures |
| `examples_ea_harmonics/` | `examples/compare_isofit_modes.py` | 2.5 MB | EA harmonics mode comparison QA |

---

## Active — other

| Folder | Script / Source | Approx Size | Notes |
|--------|----------------|------------:|-------|
| `isofit_mode_comparison/` | `examples/compare_isofit_modes.py` | 5.4 MB | ISOFIT mode comparison run |
| `legacysurvey_highorder_harmonics/` | `examples/real_galaxy_legacysurvey_highorder_harmonics/` | 22 MB | Active example campaign outputs |
| `legacysurvey_true_isofit/` | Related scripts | 9.6 MB | Active |

---

## Flagged for Review

These folders are not deleted but need owner review before clearing.

| Folder | Approx Size | Notes |
|--------|------------:|-------|
| `huang2013_campaign_full/` | 54 MB | Likely canonical campaign run; confirm before deleting others |
| `huang2013_campaign/` | 10 MB | May be redundant with `_full/`; confirm |
| `huang2013_campaign_eso185_g054/` | 100 KB | Single-galaxy sub-run |
| `huang2013_test/` | — | Old test run |
| `huang2013_test_SUMMARY.md` | — | Loose file; should move into a subdir or delete |
| `huang2013_ic2597_qa_style/` | — | Exploratory QA run |
| `huang2013_ic3370_ea_test/` | — | EA mode test on IC3370 |
| `huang2013_initial_geometry_checks/` | — | Early-stage geometry diagnostics |
| `huang2013_mock1_photutils_maxsma_scan/` | — | Old photutils scan |
| `huang2013_mock1_photutils_retry_decay/` | — | Old photutils retry test |
| `huang2013_mock3_isoster_cog_fix/` | — | Bug-fix verification run |
| `huang2013_mock3_isoster_stopcode2/` | — | Stop-code 2 investigation |
| `huang2013_mock3_photutils_model_pass1/` | — | Old photutils comparison |
| `huang2013_mock3_photutils_stopcode2/` | — | Old photutils stop-code run |
| `huang2013_mock3_stopcode2_compare/` | — | Old comparison run |
| `huang2013_convergence_benchmark/` | — | Obsolete; superseded by ic3370_exhausted |
| `ngc3610_highorder_exploration/` | 8.2 MB | Loose exploration; archive candidate |
| `ngc3610_sma0_effect/` | 4.5 MB | Loose exploration; archive candidate |
| `ngc1052_mock3_demo/` | 2.3 MB | Demo run; review before deleting |
| `ngc1209_convergence_stop2_test/` | 8.5 MB | Old NGC1209 convergence run; superseded |

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
