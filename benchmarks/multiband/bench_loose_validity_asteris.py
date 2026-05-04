"""Performance benchmark for the D9 loose-validity backport.

Compares wall time on the asteris denoised B=5 cutout
(object 37484563299062823) for three configurations:

1. **single-band baseline** — ``isoster.fit_image`` on the i-band image.
   This sets the Stage-1 ≤2.5× quality bar (D17).
2. **multi-band shared validity** (Stage-1 default) — the existing
   numba-accelerated joint design matrix path.
3. **multi-band loose validity** — the D9 backport path; uses the
   pure-NumPy ``build_joint_design_matrix_jagged`` builder.

Each configuration is timed three times after a single warm-up run
(numba JIT amortization + page-cache warm-up). The median of the
three runs is reported.

Decision criterion:
- If loose validity stays within the same ≤2.5× ratio as shared
  validity, the pure-NumPy jagged builder is fine and we keep it.
- If loose validity blows past the bar, we add a numba JIT for
  ``build_joint_design_matrix_jagged`` next.

Outputs:
- ``<out_dir>/asteris_loose_perf.json`` — raw timings + ratios.
- Console table.
"""

from __future__ import annotations

import argparse
import json
import time
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
    """Sigma-clipped sky std, mirroring asteris_demo's ``common.sky_std``."""
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 1.0
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    sigma = max(1.4826 * mad, 1e-6)
    keep = np.abs(finite - med) < 5.0 * sigma
    return float(np.std(finite[keep])) if keep.any() else sigma


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asteris-root",
        default=Path.home() / "Downloads" / "asteris" / "objs",
        type=Path,
        help="Root of the asteris cutout folders.",
    )
    parser.add_argument(
        "--obj-id", default=ASTERIS_OBJ_ID,
        help="SGA cutout object ID to benchmark on.",
    )
    parser.add_argument(
        "--out-dir",
        default=Path("outputs/benchmark_multiband/asteris_perf"),
        type=Path,
    )
    parser.add_argument("--n-repeats", type=int, default=3)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== D9 perf benchmark on asteris {args.obj_id} (B=5) ===")
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

    # --- Multi-band shared validity (Stage-1 default) ---
    mb_cfg_shared = IsosterConfigMB(
        bands=DEMO_BANDS, reference_band="i",
        sma0=20.0, eps=0.2, pa=0.0, astep=0.1, linear_growth=False,
        maxsma=float(np.hypot(h, w)) / 2.0, debug=True,
        compute_deviations=True, nclip=2, max_retry_first_isophote=5,
        band_weights=None,
    )

    def _run_mb_shared() -> object:
        return fit_image_multiband(
            images=images, masks=mask, config=mb_cfg_shared,
            variance_maps=variance_maps,
        )

    mb_shared_timing = _time_run("multiband shared", _run_mb_shared, args.n_repeats)

    # --- Multi-band loose validity (D9 backport path) ---
    mb_cfg_loose = IsosterConfigMB(
        bands=DEMO_BANDS, reference_band="i",
        sma0=20.0, eps=0.2, pa=0.0, astep=0.1, linear_growth=False,
        maxsma=float(np.hypot(h, w)) / 2.0, debug=True,
        compute_deviations=True, nclip=2, max_retry_first_isophote=5,
        band_weights=None,
        loose_validity=True,
    )

    def _run_mb_loose() -> object:
        return fit_image_multiband(
            images=images, masks=mask, config=mb_cfg_loose,
            variance_maps=variance_maps,
        )

    mb_loose_timing = _time_run("multiband loose", _run_mb_loose, args.n_repeats)

    # --- Optional: per_band_count normalization ---
    mb_cfg_loose_norm = IsosterConfigMB(
        bands=DEMO_BANDS, reference_band="i",
        sma0=20.0, eps=0.2, pa=0.0, astep=0.1, linear_growth=False,
        maxsma=float(np.hypot(h, w)) / 2.0, debug=True,
        compute_deviations=True, nclip=2, max_retry_first_isophote=5,
        band_weights=None,
        loose_validity=True,
        loose_validity_band_normalization="per_band_count",
    )

    def _run_mb_loose_norm() -> object:
        return fit_image_multiband(
            images=images, masks=mask, config=mb_cfg_loose_norm,
            variance_maps=variance_maps,
        )

    mb_loose_norm_timing = _time_run(
        "multiband loose (per_band_count)", _run_mb_loose_norm, args.n_repeats,
    )

    # --- Summary ---
    sb_med = sb_timing["median_s"]
    print()
    print(f"{'configuration':<40} {'median (s)':>12} {'ratio vs SB':>14}")
    print("-" * 68)
    for t in (sb_timing, mb_shared_timing, mb_loose_timing, mb_loose_norm_timing):
        ratio = t["median_s"] / sb_med
        print(f"{t['label']:<40} {t['median_s']:>12.3f} {ratio:>13.2f}x")
    print()
    bar = 2.5
    loose_ratio = mb_loose_timing["median_s"] / sb_med
    shared_ratio = mb_shared_timing["median_s"] / sb_med
    print(
        f"Stage-1 quality bar (D17): ≤{bar}× single-band. "
        f"Shared = {shared_ratio:.2f}× ({'PASS' if shared_ratio <= bar else 'FAIL'}); "
        f"Loose = {loose_ratio:.2f}× ({'PASS' if loose_ratio <= bar else 'FAIL'})."
    )

    out = {
        "obj_id": args.obj_id,
        "shape": list(images[0].shape),
        "n_repeats": args.n_repeats,
        "results": [sb_timing, mb_shared_timing, mb_loose_timing, mb_loose_norm_timing],
        "ratios_vs_singleband": {
            "shared": shared_ratio,
            "loose": loose_ratio,
            "loose_per_band_count": mb_loose_norm_timing["median_s"] / sb_med,
        },
        "stage1_bar": bar,
    }
    out_path = args.out_dir / f"asteris_{args.obj_id}_loose_perf.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
