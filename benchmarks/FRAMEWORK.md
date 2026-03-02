# Benchmark Framework

Rules and conventions for adding new benchmarks to isoster.

---

## 1. Naming

**Script location**: `benchmarks/<category>/<verb>_<subject>.py`

Valid categories:

| Category | Purpose |
|----------|---------|
| `performance/` | Speed, throughput, method comparisons |
| `profiling/` | Hotspot analysis, `cProfile`, flame graphs |
| `baselines/` | Threshold locking, CI regression gates |
| `ic3370_exhausted/` | Config sweep on IC3370_mock2 |
| New subdirectory | Add when a new class of benchmark warrants its own folder |

**Output folder prefix**: always singular.
- Correct: `outputs/benchmark_performance/`, `outputs/benchmark_profiling/`
- Wrong: `outputs/benchmarks_performance/`, `outputs/benchmarks_profiling/`

---

## 2. Required Artifacts Per Benchmark Run

Every benchmark must emit:

| Artifact | Format | Notes |
|----------|--------|-------|
| `results.json` | JSON | Machine-readable summary — timings, metrics, config |
| `REPORT.md` | Markdown | Human-readable summary with key findings and interpretation |
| `figures/` | PNG, ≥150 DPI | QA figures: profiles, residuals, speedup bars |

Use `benchmarks/utils/run_metadata.py` to generate the environment block (git SHA, platform,
package versions) included in `results.json`.

---

## 3. Shared Utilities

| Module | Purpose |
|--------|---------|
| `benchmarks/utils/sersic_model.py` | Synthetic Sérsic image generation |
| `benchmarks/utils/run_metadata.py` | Environment metadata + JSON write |
| `benchmarks/utils/autoprof_adapter.py` | AutoProf subprocess adapter (needs `AUTOPROF_PYTHON` env var) |

Future planned: `benchmarks/utils/report.py` — shared markdown report builder.

---

## 4. Data

- **Shared FITS files**: `data/` at project root.
  - `data/IC3370_mock2.fits` — Huang2013 mock
  - `data/eso243-49.fits` — edge-on S0
  - `data/ngc3610.fits` — boxy-bulge elliptical
  - `data/m51/M51.fits` — M51 spiral
- **External Huang2013 data**: `/Users/mac/work/hsc/huang2013/<GALAXY>/` (not tracked in repo).
- **Synthetic data**: generate at runtime using `benchmarks/utils/sersic_model.py` or
  `benchmarks/baselines/mockgal_adapter.py`.

---

## 5. CLI Requirements

Every benchmark script must:

1. Accept `--help` and print a concise usage description.
2. Accept a `--quick` flag that runs a fast smoke test (single galaxy or single config) in
   under 60 seconds.
3. Accept `--output <dir>` to override the default output directory.

---

## 6. AutoProf Benchmark Notes

`bench_vs_autoprof.py` uses a subprocess-based adapter because AutoProf requires `numpy < 2`,
which conflicts with isoster's environment. The adapter spawns a separate Python interpreter.

Set the `AUTOPROF_PYTHON` environment variable to point to a compatible Python binary:

```bash
export AUTOPROF_PYTHON=/Users/mac/miniforge3/bin/python3
```

Default: `/Users/mac/miniforge3/bin/python3`.

---

## 7. Obsolete Scripts

The following scripts in `benchmarks/` are obsolete and candidates for deletion:

| Script | Reason |
|--------|--------|
| `convergence_diagnostic.py` | Superseded by `ic3370_exhausted/` config sweep |
| `huang2013_convergence_benchmark.py` | Superseded; Huang2013 convergence covered by `ic3370_exhausted/` |
| `ngc1209_convergence_benchmark.py` | NGC1209 was a transitional target; IC3370 is canonical |
| `bench_isofit_overhead.py` | Point measurement; overhead is now documented in reports |

Before deleting, confirm no outputs that depend on them need to be regenerated.

---

## 8. Adding a New Benchmark — Checklist

Before submitting a new benchmark script:

- [ ] Script lives under `benchmarks/<category>/`
- [ ] Script name follows `<verb>_<subject>.py`
- [ ] `--help`, `--quick`, `--output` flags implemented
- [ ] Emits `results.json` and `REPORT.md`
- [ ] Uses `benchmarks/utils/run_metadata.py` for environment block
- [ ] Output folder uses singular prefix (`benchmark_`, not `benchmarks_`)
- [ ] Data paths use `data/` at project root
- [ ] Script added to census table in `benchmarks/README.md`
