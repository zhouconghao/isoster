# Benchmarks

This folder contains performance and comparison benchmarks for isoster.

## Scope

- `performance/`: runtime and throughput comparisons (including method-vs-method).
- `profiling/`: hotspot and profiler-oriented diagnostics.
- `baselines/`: baseline data and CI regression gates.
- `ic3370_exhausted/`: 39-configuration exhaustive sweep on IC3370_mock2.
- `utils/`: shared benchmark helpers.

## Goals

- Quantify speed and scaling behavior.
- Compare isoster against reference methods under controlled scenarios.
- Generate reproducible machine-readable outputs (JSON/CSV) and optional figures.

---

## Script Census

| Script | Category | Motivation | Input Data | Output Artifacts | Status |
|--------|----------|-----------|-----------|-----------------|--------|
| `performance/bench_vs_photutils.py` | Performance | Speed + accuracy vs photutils | Synthetic Sérsic | CSV / JSON / figures | Active |
| `performance/bench_vs_autoprof.py` | Performance | Speed + accuracy vs AutoProf | IC3370, eso243-49, ngc3610 | Reports, figures | Active |
| `performance/bench_efficiency.py` | Performance | Standardized efficiency gate | Synthetic Sérsic | JSON timing | Active |
| `performance/bench_numba_speedup.py` | Performance | Numba JIT impact | Synthetic Sérsic | JSON timing | Active |
| `ic3370_exhausted/run_benchmark.py` | Config sweep | 39-config exhaustive sweep on IC3370 | `data/IC3370_mock2.fits` | Per-config FITS + figures | Active |
| `baselines/run_benchmark_gate.py` | CI gate | Regression gate (efficiency + QA) | Synthetic + Huang2013 | Pass/fail + artifacts | Active |
| `baselines/collect_phase4_profile_baseline.py` | Baseline | Collect Phase 4 profile metrics | Synthetic Sérsic | JSON | Active |
| `baselines/lock_phase4_thresholds.py` | Baseline | Lock Phase 4 profile thresholds | JSON from above | Thresholds JSON | Active |
| `baselines/lock_efficiency_thresholds.py` | Baseline | Lock efficiency thresholds | JSON from bench_efficiency | Thresholds JSON | Active |
| `baselines/mockgal_adapter.py` | Baseline | External mockgal adapter (libprofit) | Presets / config | Mock FITS images | Active |
| `baselines/scaffold_models_config_batch_templates.py` | Baseline | Scaffold batch template files | None | Template YAML/JSON | Active |
| `profiling/profile_hotpaths.py` | Profiling | cProfile hotspot analysis | Synthetic Sérsic | `.prof` + JSON | Active |
| `profiling/profile_isophote.py` | Profiling | Per-isophote profiling | Synthetic Sérsic | `.prof` + JSON | Active |
| `profiling/profile_numba_flagged_cases.py` | Profiling | Numba flagged-case timing | Synthetic Sérsic | JSON timing | Active |
| `convergence_diagnostic.py` | Convergence | Config sweep on single galaxy | NGC1209_mock2 | JSON + markdown | **Obsolete** |
| `huang2013_convergence_benchmark.py` | Convergence | 20-galaxy convergence sweep | Huang2013 (external) | JSON + figures | **Obsolete** |
| `ngc1209_convergence_benchmark.py` | Convergence | 10-config NGC1209 sweep | NGC1209_mock2 | Per-config FITS | **Obsolete** |
| `bench_isofit_overhead.py` | Performance | ISOFIT overhead point measurement | Synthetic Sérsic | Stdout | **Obsolete** |

Obsolete scripts are superseded by the `ic3370_exhausted/` sweep and are candidates for deletion.
See `FRAMEWORK.md` for guidance on adding new benchmarks.

---

## Reproducible Commands

```bash
# Efficiency benchmark
uv run python benchmarks/performance/bench_efficiency.py

# Numba speedup benchmark
uv run python benchmarks/performance/bench_numba_speedup.py

# Numba diagnostic sweep (more stable timing + scaled workload)
uv run python benchmarks/performance/bench_numba_speedup.py --n-runs 4 --scale-factor 1.0
uv run python benchmarks/performance/bench_numba_speedup.py --n-runs 3 --scale-factor 2.0 --output outputs/benchmark_performance/bench_numba_speedup_scale2

# Flagged-case numba profiling drills (defaults target scale1+scale2 JSON outputs)
uv run python benchmarks/profiling/profile_numba_flagged_cases.py --timing-runs 3 --profile-runs 1

# Method comparison benchmark: isoster vs photutils (quick mode)
uv run python benchmarks/performance/bench_vs_photutils.py --quick

# Method comparison benchmark: isoster vs AutoProf (quick mode — IC3370 only)
uv run python benchmarks/performance/bench_vs_autoprof.py --quick --plots

# AutoProf full benchmark (all galaxies)
uv run python benchmarks/performance/bench_vs_autoprof.py --plots

# AutoProf benchmark — isoster only (skip AutoProf, replot from cache)
uv run python benchmarks/performance/bench_vs_autoprof.py --skip-autoprof --plots

# AutoProf benchmark — specific galaxies
uv run python benchmarks/performance/bench_vs_autoprof.py --galaxies IC3370_mock2 ngc3610 --plots

# Profiling artifacts (.prof + summaries)
uv run python benchmarks/profiling/profile_hotpaths.py --multi 1 --top-n 20
uv run python benchmarks/profiling/profile_isophote.py --repetitions 1 --top-n 20

# Method comparison with explicit output root
uv run python benchmarks/performance/bench_vs_photutils.py --quick --output outputs/benchmark_performance/manual_run

# Phase 4 profile baseline metrics
uv run python benchmarks/baselines/collect_phase4_profile_baseline.py

# Lock threshold file from baseline metrics
uv run python benchmarks/baselines/lock_phase4_thresholds.py

# Lock efficiency thresholds from measured efficiency baselines
uv run python benchmarks/baselines/lock_efficiency_thresholds.py

# Run combined baseline-locked benchmark gate (efficiency + 1-D + 2-D system diagnostics)
# Defaults are loaded from benchmarks/baselines/benchmark_gate_defaults.json
uv run python benchmarks/baselines/run_benchmark_gate.py --quick --n-runs 1 --efficiency-lock benchmarks/baselines/efficiency_thresholds_quick_2026-02-14.json

# Override defaults file (optional)
uv run python benchmarks/baselines/run_benchmark_gate.py --quick --n-runs 1 --gate-defaults benchmarks/baselines/benchmark_gate_defaults.json --efficiency-lock benchmarks/baselines/efficiency_thresholds_quick_2026-02-14.json

# Full lock-refresh + full gate workflow before large batches
uv run python benchmarks/performance/bench_efficiency.py --n-runs 3 --output outputs/benchmark_performance/bench_efficiency_full_refresh
uv run python benchmarks/baselines/lock_efficiency_thresholds.py --input outputs/benchmark_performance/bench_efficiency_full_refresh/efficiency_benchmark_results.json --output benchmarks/baselines/efficiency_thresholds_full_refresh.json
uv run python benchmarks/baselines/collect_phase4_profile_baseline.py --output outputs/tests_integration/baseline_metrics_full_refresh
uv run python benchmarks/baselines/lock_phase4_thresholds.py --input outputs/tests_integration/baseline_metrics_full_refresh/phase4_profile_baseline_metrics.json --output benchmarks/baselines/phase4_profile_thresholds_full_refresh.json
# Run this full gate command sequentially (do not run concurrent benchmark commands).
uv run python benchmarks/baselines/run_benchmark_gate.py --n-runs 3 --efficiency-lock benchmarks/baselines/efficiency_thresholds_full_refresh.json --profile-lock benchmarks/baselines/phase4_profile_thresholds_full_refresh.json --require-all-locked-cases

# Gate now generates built-in Huang2013 basic-QA artifacts per Phase-4 case
uv run python benchmarks/baselines/run_benchmark_gate.py --quick --n-runs 1 --efficiency-lock benchmarks/baselines/efficiency_thresholds_quick_2026-02-14.json --qa-output-subdir qa_figures --qa-overlay-step 10 --qa-dpi 180

# Optional external mockgal adapter (safe dry-run)
uv run python benchmarks/baselines/mockgal_adapter.py --dry-run

# List and use science-ready mockgal presets
uv run python benchmarks/baselines/mockgal_adapter.py --list-presets
uv run python benchmarks/baselines/mockgal_adapter.py --preset single_psf_noise_sblimit --dry-run
uv run python benchmarks/baselines/mockgal_adapter.py --preset models_config_batch --dry-run
uv run python benchmarks/baselines/mockgal_adapter.py --preset models_config_batch --preset-value models_file=my_models.yaml --preset-value config_file=my_config.yaml --preset-value workers=2 --dry-run

# Scaffold reusable models_config_batch template files into a new outputs/ run directory
uv run python benchmarks/baselines/scaffold_models_config_batch_templates.py
```

---

## Output Policy

Benchmark outputs go under `outputs/` with source-specific folder names.
Override the output root with `ISOSTER_OUTPUT_ROOT`.

Naming rule: use **singular** prefix — `benchmark_`, not `benchmarks_`.

Recommended naming:
- `outputs/benchmark_performance/<run_id>/...`
- `outputs/benchmark_profiling/<run_id>/...`

Performance and profiling scripts should emit:
- machine-readable summaries (`.json`, `.csv`)
- plot artifacts (`.png`/`.pdf`) for summary or failing/borderline cases
- run metadata (git SHA, package versions, platform info)
- profiler artifacts (`.prof`) for profiling scripts

See `FRAMEWORK.md` for complete naming and content requirements.
