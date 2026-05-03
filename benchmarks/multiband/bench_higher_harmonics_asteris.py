"""Performance benchmark for the multiband_higher_harmonics enum.

Compares wall time on the asteris denoised B=5 cutout (object
37484563299062823) for FIVE configurations:

1. **single-band baseline** — ``isoster.fit_image`` on the i-band image.
   Sets the Stage-1 ≤2.5× quality bar (D17).
2. **multi-band independent** — ``multiband_higher_harmonics='independent'``,
   the Stage-1 default; per-band post-hoc higher-order fits.
3. **multi-band shared** — ``multiband_higher_harmonics='shared'``;
   one post-hoc joint refit of higher orders shared across bands.
4. **multi-band simultaneous_in_loop** — wider joint design matrix
   every iteration. Recovered single-band ISOFIT feature.
5. **multi-band simultaneous_original** — Ciambur-original variant:
   standard 5-param loop + one post-hoc joint refit over all orders.

Each configuration is timed three times after a single warm-up run
(numba JIT amortization + page-cache warm-up). The median of the three
runs is reported.

Decision criterion (Section 6.1 / Q-R3-2 / D17): all four multi-band
modes must stay within ≤ 2.5× of the single-band baseline.

Outputs:
- ``<out_dir>/asteris_<obj_id>_higher_harmonics_perf.json`` — raw timings + ratios.
- Console table.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Callable, List

import numpy as np

from isoster import IsosterConfig, fit_image
from isoster.multiband import IsosterConfigMB, fit_image_multiband
from isoster.multiband.numba_kernels_mb import warmup_numba_mb
from isoster.numba_kernels import warmup_numba


DEMO_BAND_FOLDERS = ["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"]
DEMO_BANDS = ["g", "r", "i", "z", "y"]
ASTERIS_OBJ_ID = "37484563299062823"
PERF_BAR = 2.5


def _galaxy_dir(asteris_root: Path, obj_id: str, band_folder: str) -> Path:
    cutout_name = (
        f"obj_{obj_id}_pdr3_dud_rev_8523_0_3_cutout768_uniform8_t8_v1"
    )
    return asteris_root / cutout_name / band_folder


def _load_denoised(asteris_root: Path, obj_id: str, band_folder: str) -> np.ndarray:
    from astropy.io import fits

    path = _galaxy_dir(asteris_root, obj_id, band_folder) / "denoised.fits"
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None:
                return np.asarray(hdu.data, dtype=np.float64)
    raise IOError(f"No data HDU found in {path}")


def _load_object_mask(asteris_root: Path, obj_id: str) -> np.ndarray:
    from astropy.io import fits

    mpath = _galaxy_dir(asteris_root, obj_id, "HSC-I") / "object_mask.fits"
    with fits.open(mpath) as hdul:
        data = hdul[0].data
    return np.asarray(data, dtype=np.float64) > 0.5


def _sky_std(image: np.ndarray) -> float:
    """Sigma-clipped sky std matching the asteris example script.

    Mirrors :func:`examples.example_asteris_denoised.common.sky_std` so the
    variance maps consumed by the joint solver are bit-identical to those
    used in the reference run at
    ``outputs/example_asteris_denoised/.../37484563299062823_multiband_isophotes.fits``.
    Earlier MAD-based estimators here gave a 12% larger sigma which shifted
    borderline outer-SMA convergence behavior and produced divergent
    geometry vs the reference (idx 64+ on this cutout).
    """
    from astropy.stats import sigma_clipped_stats

    _, _, std = sigma_clipped_stats(image, sigma=3.0, maxiters=5)
    return float(std)


def _uniform_variance_map(image: np.ndarray) -> np.ndarray:
    return np.full_like(image, _sky_std(image) ** 2, dtype=np.float64)


def _time_run(label: str, callable_: Callable[[], object], n_repeats: int = 3) -> dict:
    """Warm up once, then time ``n_repeats`` runs and return the median."""
    print(f"  warming up: {label} ...")
    callable_()
    times: List[float] = []
    for i in range(n_repeats):
        t0 = time.perf_counter()
        callable_()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"    run {i + 1}: {elapsed:.3f}s")
    return {
        "label": label,
        "times_s": times,
        "median_s": float(np.median(times)),
        "min_s": float(np.min(times)),
        "n_repeats": n_repeats,
    }


def _make_mb_cfg(
    mode: str,
    *,
    h: int,
    w: int,
) -> IsosterConfigMB:
    with warnings.catch_warnings():
        # The simultaneous_* modes emit an experimental UserWarning at
        # construction; suppress it during benchmarking.
        warnings.simplefilter("ignore", UserWarning)
        return IsosterConfigMB(
            bands=DEMO_BANDS, reference_band="i",
            sma0=20.0, eps=0.2, pa=0.0, astep=0.1, linear_growth=False,
            maxsma=float(np.hypot(h, w)) / 2.0, debug=True,
            compute_deviations=True, nclip=2, max_retry_first_isophote=5,
            band_weights=None,
            multiband_higher_harmonics=mode,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asteris-root",
        default=Path.home() / "Downloads" / "asteris" / "objs",
        type=Path,
    )
    parser.add_argument("--obj-id", default=ASTERIS_OBJ_ID)
    parser.add_argument(
        "--out-dir",
        default=Path("outputs/benchmark_multiband/higher_harmonics_perf"),
        type=Path,
    )
    parser.add_argument("--n-repeats", type=int, default=3)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Section 6 perf benchmark on asteris {args.obj_id} (B=5) ===")
    print("Loading inputs ...")
    images: List[np.ndarray] = []
    for folder in DEMO_BAND_FOLDERS:
        img = _load_denoised(args.asteris_root, args.obj_id, folder)
        images.append(img)
    mask = _load_object_mask(args.asteris_root, args.obj_id)
    variance_maps = [_uniform_variance_map(im) for im in images]
    h, w = images[0].shape
    print(f"  shape={images[0].shape}, n_masked={int(mask.sum())}")

    print("Warming up numba JITs ...")
    warmup_numba()
    warmup_numba_mb()

    # --- Single-band baseline (i-band) ---
    sb_cfg = IsosterConfig(
        sma0=20.0, eps=0.2, pa=0.0, astep=0.1, linear_growth=False,
        maxsma=float(np.hypot(h, w)) / 2.0, debug=True, nclip=2,
        max_retry_first_isophote=5,
    )
    i_idx = DEMO_BANDS.index("i")

    def _run_singleband() -> object:
        return fit_image(
            images[i_idx], mask=mask, config=sb_cfg,
            variance_map=variance_maps[i_idx],
        )

    sb_timing = _time_run("singleband (i-band)", _run_singleband, args.n_repeats)
    sb_med = sb_timing["median_s"]

    # --- Multi-band: 4 modes ---
    timings: List[dict] = [sb_timing]
    for mode in (
        "independent",
        "shared",
        "simultaneous_in_loop",
        "simultaneous_original",
    ):
        cfg = _make_mb_cfg(mode, h=h, w=w)

        def _run(cfg=cfg) -> object:
            return fit_image_multiband(
                images=images, masks=mask, config=cfg,
                variance_maps=variance_maps,
            )

        timings.append(_time_run(f"multiband {mode}", _run, args.n_repeats))

    # --- Summary ---
    print()
    print(f"{'configuration':<40} {'median (s)':>12} {'ratio vs SB':>14}  {'PASS':>5}")
    print("-" * 78)
    for t in timings:
        ratio = t["median_s"] / sb_med
        passed = ratio <= PERF_BAR or "single" in t["label"]
        print(
            f"{t['label']:<40} {t['median_s']:>12.3f} {ratio:>13.2f}x"
            f"  {'PASS' if passed else 'FAIL':>5}"
        )
    print()
    print(f"Stage-1 perf bar (D17 / Section 6.1 Q11): ≤{PERF_BAR}× single-band.")

    out = {
        "obj_id": args.obj_id,
        "shape": list(images[0].shape),
        "n_repeats": args.n_repeats,
        "results": timings,
        "ratios_vs_singleband": {
            t["label"]: t["median_s"] / sb_med for t in timings
        },
        "stage1_bar": PERF_BAR,
    }
    out_path = args.out_dir / f"asteris_{args.obj_id}_higher_harmonics_perf.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
