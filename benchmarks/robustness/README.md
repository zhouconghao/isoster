# Robustness Benchmark

Measures how isoster's outward/inward free fit responds to perturbations
of the user-supplied initial `sma0`, initial ellipticity `eps`, and
initial position angle `pa`. The fiducial center `(x0, y0)` is treated
as known user input and is **not** perturbed by the sweep.

This is a **benchmark experiment**, not a pass/fail test — the goal is
to characterize the capture radius of the fit, not to gate the code on
a threshold. Findings land in the per-tier `REPORT.md` files under
`outputs/benchmark_robustness/{tier}/` and the cross-tier
`outputs/benchmark_robustness/SUMMARY.md`, alongside the design doc at
`docs/agent/journal/2026-04-15_robustness_plan.md`.

## Layout

```
benchmarks/robustness/
  __init__.py
  datasets.py       # per-tier galaxy loaders + fiducial starting conditions
  metrics.py        # relative-intensity, angular-MAD, interpolated comparison
  persist.py        # per-row isophote FITS writer/reader
  run_sweep.py      # main entry — CLI + sweep driver + per-tier REPORT.md writer
  build_figures.py  # profile overlays + outlier QA figures from persisted FITS
  README.md         # this file
```

## Arms

| arm      | description                                                      |
|----------|------------------------------------------------------------------|
| `bare`   | free fit, `max_retry_first_isophote=0`, no LSB features          |
| `retry`  | `bare` + `max_retry_first_isophote=5`                            |
| `lsb`    | `retry` + `lsb_auto_lock=True` + outer-region center regularization |
| `ea_lsb` | `lsb` + `use_eccentric_anomaly=True` (uniform ellipse sampling)  |

The reference fit per galaxy uses `lsb` at the fiducial start — on
mocks this is truth-informed, on real galaxies it matches the shipped
LSB sweep's `B_std` arm. `ea_lsb` is the twin arm that adds
eccentric-anomaly sampling and is most useful for high-ε galaxies.

## Tiers

Easy → hard. The smoke path covers the first tier; the others are
wired up incrementally as the sweep matures.

| tier        | status       | galaxies                                   |
|-------------|--------------|--------------------------------------------|
| `mocks`     | implemented  | 4 synthetic Sersic profiles via `benchmarks.utils.sersic_model` |
| `huang2013` | implemented  | 4 libprofit mocks of `IC2597` (`mock1`..`mock4`), loaded from the external data root (`HUANG2013_DATA_ROOT` env var, default `~/.../isophote_test/output/huang2013/`); header-driven fiducial start via `examples/example_huang2013/huang2013_shared.infer_initial_geometry` and `run_huang2013_profile_extraction.infer_default_maxsma` |
| `highorder` | implemented  | `eso243-49`, `ngc3610` from `data/` at project root — reuses `examples/example_ls_highorder_harmonic/shared.py` loaders + `masking.make_object_mask`; masks cached under `outputs/benchmark_robustness/cache/highorder/` |
| `hsc`       | implemented  | 6 HSC edge-case galaxies (`10140002`, `10140006`, `10140009`, `10140056`, `10140088`, `10140093`) from `examples/example_hsc_edgecases/data/` with pre-packaged `HSC_I` image+variance+mask FITS; fiducial start mirrors `run_lsb_mode_sweep` (`sma0=10`, `eps=0.2`, `pa=0`) |

## Usage

```bash
# Sub-minute smoke test (one mock galaxy, bare arm, 3 sma0 factors)
uv run python benchmarks/robustness/run_sweep.py --quick

# Full 1-D sweep on the mocks tier (all 4 arms × 3 axes)
uv run python benchmarks/robustness/run_sweep.py --tiers mocks

# Restrict to specific axes or galaxies
uv run python benchmarks/robustness/run_sweep.py \
    --tiers mocks \
    --axes sma0 eps \
    --galaxies mock_disk_low_n

# Huang2013 tier — single mock, bare arm, sma0 axis (a few seconds)
uv run python benchmarks/robustness/run_sweep.py \
    --tiers huang2013 --galaxies IC2597_mock1 --arms bare --axes sma0

# Huang2013 tier — full sweep (4 IC2597 mocks × 4 arms × 3 axes)
uv run python benchmarks/robustness/run_sweep.py --tiers huang2013

# Highorder tier — single galaxy, one arm, sma0 axis only (a few seconds
# after the first-run mask cache is built)
uv run python benchmarks/robustness/run_sweep.py \
    --tiers highorder --galaxies ngc3610 --arms lsb --axes sma0

# Highorder tier — full sweep (2 galaxies × 4 arms × 3 axes)
uv run python benchmarks/robustness/run_sweep.py --tiers highorder

# HSC tier — single galaxy, two arms, sma0 axis (few seconds)
uv run python benchmarks/robustness/run_sweep.py \
    --tiers hsc --galaxies 10140088 --arms bare lsb --axes sma0

# HSC tier — full sweep (6 galaxies × 4 arms × 3 axes)
uv run python benchmarks/robustness/run_sweep.py --tiers hsc

# All tiers in one invocation (driver loops and writes per-tier outputs)
uv run python benchmarks/robustness/run_sweep.py \
    --tiers mocks huang2013 highorder hsc

# Rebuild profile + outlier QA figures from persisted FITS
uv run python benchmarks/robustness/build_figures.py

# Custom output directory (default: outputs/benchmark_robustness/)
uv run python benchmarks/robustness/run_sweep.py --output /tmp/robustness
```

## Outputs

Every run writes a per-tier subtree under
`outputs/benchmark_robustness/` plus a cross-tier `SUMMARY.md` at the
top level:

```
outputs/benchmark_robustness/
├── SUMMARY.md                 # cross-tier headline rollup
├── mocks/
│   ├── REPORT.md              # per-tier human-readable report
│   ├── results.json           # machine-readable rows + env metadata
│   ├── _summary.csv           # flat one-row-per-fit CSV
│   ├── sweep/{arm}/{obj_id}/{axis}_{value}.fits
│   ├── reference/{obj_id}/{obj_id}_reference.fits
│   └── figures/
│       ├── profiles/{obj_id}/{arm}_{axis}.png
│       └── outliers/{obj_id}/{arm}_{axis}_{value}_qa.png
├── huang2013/ ...
├── highorder/ ...
└── hsc/ ...
```

- `results.json` is machine-readable rows plus environment metadata
  (git SHA, Python/numpy/scipy versions, platform) via
  `benchmarks/utils/run_metadata.py`.
- `REPORT.md` is the per-tier human-readable rollup with per-arm and
  per-axis motion distributions, top outliers, and per-row walk detail.
- `SUMMARY.md` at the parent level aggregates headline numbers across
  every tier whose `results.json` currently exists on disk.
- Profile figures plot every perturbation as a thin grey trace in the
  background and highlight outliers (top-N by `profile_rel_rms`) in
  saturated viridis colors with labels.

## Tests

Unit tests live at:

- `tests/unit/test_robustness_metrics.py` — guards the metric math
  (relative-intensity deviation, pi-periodic angular MAD, interpolation-
  based comparison, bin thresholds).
- `tests/unit/test_robustness_datasets.py` — tier dispatch smoke tests
  and a huang2013 loader test that auto-skips when the external data
  root is not present locally.

Run both with:

```bash
uv run pytest tests/unit/test_robustness_metrics.py \
              tests/unit/test_robustness_datasets.py -v
```
