# Robustness Benchmark

Measures how isoster's outward/inward free fit responds to perturbations
of the user-supplied initial `sma0` and initial isophotal geometry
(`eps`, `pa`, `x0`, `y0`).

This is a **benchmark experiment**, not a pass/fail test — the goal is
to characterize the capture radius of the fit, not to gate the code on
a threshold. Findings land in
`outputs/benchmark_robustness/REPORT.md` and in the accompanying
design doc at `docs/agent/journal/2026-04-15_robustness_plan.md`.

## Layout

```
benchmarks/robustness/
  __init__.py
  datasets.py     # per-tier galaxy loaders + fiducial starting conditions
  metrics.py      # relative-intensity, angular-MAD, interpolated comparison
  run_sweep.py    # main entry — CLI + sweep driver + REPORT.md writer
  README.md       # this file
```

## Arms

| arm         | description                                                      |
|-------------|------------------------------------------------------------------|
| `bare`      | free fit, `max_retry_first_isophote=0`, no LSB features          |
| `retry`     | `bare` + `max_retry_first_isophote=5`                            |
| `retry_lsb` | `retry` + `lsb_auto_lock=True` + outer-region center regularization |

The reference fit per galaxy uses `retry_lsb` at the fiducial start —
on mocks this is truth-informed, on real galaxies it matches the
shipped LSB sweep's `B_std` arm.

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

# Full 1-D sweep on the mocks tier
uv run python benchmarks/robustness/run_sweep.py --tiers mocks

# Restrict to specific axes or galaxies
uv run python benchmarks/robustness/run_sweep.py \
    --tiers mocks \
    --axes sma0 eps \
    --galaxies mock_disk_low_n

# Huang2013 tier — single mock, bare arm, sma0 axis (a few seconds)
uv run python benchmarks/robustness/run_sweep.py \
    --tiers huang2013 --galaxies IC2597_mock1 --arms bare --axes sma0

# Huang2013 tier — full sweep (4 IC2597 mocks x 3 arms x 5 axes)
uv run python benchmarks/robustness/run_sweep.py --tiers huang2013

# Highorder tier — single galaxy, one arm, sma0 axis only (a few seconds
# after the first-run mask cache is built)
uv run python benchmarks/robustness/run_sweep.py \
    --tiers highorder --galaxies ngc3610 --arms retry_lsb --axes sma0

# Highorder tier — full sweep (2 galaxies × 3 arms × 5 axes)
uv run python benchmarks/robustness/run_sweep.py --tiers highorder

# HSC tier — single galaxy, two arms, sma0 axis (few seconds)
uv run python benchmarks/robustness/run_sweep.py \
    --tiers hsc --galaxies 10140088 --arms bare retry_lsb --axes sma0

# HSC tier — full sweep (6 galaxies × 3 arms × 5 axes)
uv run python benchmarks/robustness/run_sweep.py --tiers hsc

# Custom output directory (default: outputs/benchmark_robustness/)
uv run python benchmarks/robustness/run_sweep.py --output /tmp/robustness
```

## Outputs

Every run writes, per the `benchmarks/FRAMEWORK.md` §2 rules:

- `results.json` — machine-readable rows plus environment metadata
  (git SHA, Python/numpy/scipy versions, platform) via
  `benchmarks/utils/run_metadata.py`.
- `REPORT.md` — human-readable rollup with per-arm motion distribution
  and per-row detail tables.

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
