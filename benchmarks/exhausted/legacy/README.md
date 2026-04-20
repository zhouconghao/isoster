# benchmarks/exhausted/

Exhaustive isoster configuration sweep benchmark.

Tests every meaningful isoster configuration parameter (individually and in
combination) against a photutils baseline on a single galaxy FITS image.

---

## What it does

The sweep runs **39 configurations** (P00 + S00–S23 + C01–C12) defined in
`config_registry.py`:

- **P00** — photutils.isophote baseline (reference)
- **S00** — isoster baseline (all default parameters for the galaxy)
- **S01–S23** — single-parameter sweeps (convergence scaling, damping, EA mode, ISOFIT, harmonics, sclip, maxit, integrator, fix_center …)
- **C01–C12** — combination configs (e.g. EA + ISOFIT in_loop, permissive + damping, full Ciambur 2015)

For each configuration, the script collects:

- Convergence counts (stop codes 0 / 1 / 2 / 3 / -1)
- Wall time
- 1D profile accuracy (median |dI/I|, |dε|, |dPA|) per radial zone vs photutils
- 2D model fractional residuals (median and RMS)

Outputs: per-config `isophotes_<ID>.json`, QA PNGs, `timing_log.json`,
`summary_vs_P00.png`, `summary_vs_S00.png`, `report.md`.

---

## Files

| File | Purpose |
|------|---------|
| `run_benchmark.py` | Main orchestrator — loads image, runs sweep, writes artifacts |
| `config_registry.py` | All 39 configuration definitions + fitting parameter baseline |
| `config_registry.md` | Human-readable annotation of each config (optional reference) |

---

## Quick start

```bash
# Default: IC3370_mock2 with auto-detected geometry
uv run python benchmarks/exhausted/run_benchmark.py

# Smoke test (S00 + S08 only, ~2 min on IC3370)
uv run python benchmarks/exhausted/run_benchmark.py --quick

# Reproduce canonical IC3370 run with known geometry
uv run python benchmarks/exhausted/run_benchmark.py \
    --image data/IC3370_mock2.fits \
    --x0 566 --y0 566 --eps 0.239 --pa-deg -27.99 --sma0 6 --maxsma 283

# Run on another galaxy
uv run python benchmarks/exhausted/run_benchmark.py \
    --image data/ngc3610.fits --band-index 2 --galaxy-label ngc3610

# Run a subset of configs, skip QA figures
uv run python benchmarks/exhausted/run_benchmark.py \
    --configs S00,S08,C03 --skip-qa-figures

# Re-use cached photutils baseline, regenerate isoster configs only
uv run python benchmarks/exhausted/run_benchmark.py --skip-photutils
```

---

## CLI reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--image` | `data/IC3370_mock2.fits` | Path to a 2D (or 3D-cube) galaxy FITS file |
| `--band-index` | `0` | Plane index for 3D FITS cubes (0-based) |
| `--galaxy-label` | image stem | Short label for titles and output folder name |
| `--x0` | auto | Galaxy centre x-coordinate (pixels) |
| `--y0` | auto | Galaxy centre y-coordinate (pixels) |
| `--eps` | auto (`0.2`) | Initial ellipticity |
| `--pa-deg` | auto (`0.0`) | Initial position angle (degrees) |
| `--sma0` | auto | Starting semi-major axis (pixels) |
| `--maxsma` | auto | Maximum semi-major axis (pixels) |
| `--output-dir` | `outputs/benchmark_exhausted/<label>` | Output directory |
| `--configs` | all | Comma-separated config IDs to run |
| `--quick` | off | Smoke test: S00 + S08 only, no QA, no model |
| `--skip-photutils` | off | Reuse existing `P00/photutils_baseline.fits` |
| `--skip-qa-figures` | off | Skip per-config QA PNGs |
| `--skip-model` | off | Skip 2D model building and residuals |

---

## Geometry

All geometry arguments are optional. When omitted, the script auto-detects:

| Parameter | Auto value |
|-----------|-----------|
| `x0`, `y0` | Image centre |
| `eps` | 0.2 |
| `pa` | 0.0 rad |
| `sma0` | `max(5, min_halfsize × 0.04)` |
| `maxsma` | `min_halfsize × 0.90` |

For the canonical IC3370_mock2 benchmark, provide the known geometry explicitly
(`--x0 566 --y0 566 --eps 0.239 --pa-deg -27.99 --sma0 6 --maxsma 283`)
to reproduce the published results.

---

## Outputs

All outputs go under `outputs/benchmark_exhausted/<galaxy-label>/`:

```
<galaxy-label>/
├── P00/
│   ├── photutils_baseline.fits
│   └── qa_P00.png
├── S00/
│   ├── isophotes_S00.json
│   └── qa_S00.png
├── S01/ … C12/
│   └── isophotes_<ID>.json, qa_<ID>.png
├── timing_log.json          ← per-config timing + stop-code counts
├── summary_vs_P00.png       ← 4-panel comparison vs photutils
├── summary_vs_S00.png       ← 4-panel comparison vs isoster baseline
├── report.md                ← full metrics table
└── qa_gallery/              ← flat copy of all qa_*.png
```

---

## Configuration design

`config_registry.py` separates concerns:

- **`PARAMETER_BASELINE`** — fitting parameter defaults applied to every configuration
  (nclip, astep, maxit, convergence_scaling, damping, …).  No galaxy geometry.
- **`CONFIGURATIONS`** — list of `(config_id, description, overrides)` tuples.
  Each override dict modifies exactly the parameters being swept.
- **`PHOTUTILS_PARAMETER_CONFIG`** — photutils-specific fitting parameters.

Galaxy geometry (`x0`, `y0`, `eps`, `pa`, `sma0`, `maxsma`) is injected at
runtime from CLI arguments, keeping the registry galaxy-agnostic.

To add a new configuration: append a tuple to `CONFIGURATIONS` in
`config_registry.py` and add the config ID to `EXTENDED_HARMONIC_CONFIGS`
or `NEEDS_REFERENCE_GEOMETRY` if applicable.
