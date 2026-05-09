"""Stage-3 Stage-C perf + numerical benchmark for ``lsb_auto_lock``.

Compares three multi-band configurations on the asteris denoised B=5
cutout (object 37484563299062823):

1. ``baseline`` — no auto-lock, no outer damping. Stage-1 default.
2. ``lock-mean`` — ``lsb_auto_lock=True`` with ``integrator='mean'``,
   matrix-mode joint solve preserved. The lock-fire path freezes
   geometry but keeps the linear solver.
3. ``lock-median`` — ``lsb_auto_lock=True`` with
   ``integrator='median'`` and ``fit_per_band_intens_jointly=False``
   (Stage-A S1 satisfied). Locked-region intercepts use per-band
   sigma-clipped median for contaminant robustness.

Stage-C quality bars (plan section 7, Phase 38):

- All three configs PASS the canonical Stage-1 ``≤ 2.5×`` single-band
  bar.
- Lock fires on at least one outward isophote on the asteris cutout
  (the LSB envelope is genuinely there at 4–5× re).
- Outer-tail (last 20 valid isophotes) eps / pa MAD shrink under
  lock-on vs baseline — the locked geometry is the lock anchor's,
  so MAD must be near zero on locked isophotes.

Outputs:
- ``<out_dir>/asteris_<obj_id>_lsb_auto_lock_perf.json`` — perf.
- ``<out_dir>/asteris_<obj_id>_lsb_auto_lock_compare.json`` — lock
  metadata + outer-tail diagnostics per config.
- One FITS per config under ``<out_dir>/`` for downstream QA.
- Console summary table.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from isoster import IsosterConfig, fit_image
from isoster.multiband import IsosterConfigMB, fit_image_multiband
from isoster.multiband.numba_kernels_mb import warmup_numba_mb
from isoster.multiband.utils_mb import isophote_results_mb_to_fits
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
    """Required ``astropy.stats.sigma_clipped_stats(sigma=3, maxiters=5)`` —
    MAD-based estimators are forbidden (12% bias)."""
    from astropy.stats import sigma_clipped_stats

    _, _, std = sigma_clipped_stats(image, sigma=3.0, maxiters=5)
    return float(std)


def _uniform_variance_map(image: np.ndarray) -> np.ndarray:
    return np.full_like(image, _sky_std(image) ** 2, dtype=np.float64)


def _time_run(label: str, callable_: Callable[[], object], n_repeats: int = 3) -> dict:
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


def _build_cfg(
    *, h: int, w: int, lock_on: bool, locked_integrator: str,
) -> IsosterConfigMB:
    with warnings.catch_warnings():
        # Auto-enable debug warning is expected when lock_on=True.
        warnings.simplefilter("ignore", UserWarning)
        kwargs = dict(
            bands=DEMO_BANDS, reference_band="i",
            sma0=20.0, eps=0.2, pa=0.0, astep=0.1, linear_growth=False,
            maxsma=float(np.hypot(h, w)) / 2.0, debug=True,
            compute_deviations=True, nclip=2, max_retry_first_isophote=5,
            band_weights=None,
            lsb_auto_lock=lock_on,
            lsb_auto_lock_integrator=locked_integrator,
            lsb_auto_lock_maxgerr=0.3,
            lsb_auto_lock_debounce=2,
        )
        if lock_on and locked_integrator == "median":
            kwargs["fit_per_band_intens_jointly"] = False  # Stage-A S1
        return IsosterConfigMB(**kwargs)


def _outer_tail_metrics(result: dict) -> Dict[str, object]:
    rows = result["isophotes"]
    smas = np.array([float(r.get("sma", float("nan"))) for r in rows])
    valid = np.isfinite(smas)
    valid_rows = [r for r, v in zip(rows, valid) if v]
    tail = valid_rows[-20:] if len(valid_rows) >= 20 else valid_rows
    intens_mad: Dict[str, float] = {}
    for b in DEMO_BANDS:
        col = np.array(
            [float(r.get(f"intens_{b}", float("nan"))) for r in tail],
            dtype=np.float64,
        )
        finite = col[np.isfinite(col)]
        intens_mad[b] = (
            float(np.median(np.abs(finite - np.median(finite))))
            if finite.size >= 4 else float("nan")
        )
    eps_arr = np.array([float(r.get("eps", float("nan"))) for r in tail])
    pa_arr = np.array([float(r.get("pa", float("nan"))) for r in tail])
    eps_mad = (
        float(np.median(np.abs(eps_arr - np.median(eps_arr))))
        if eps_arr.size else float("nan")
    )
    pa_mad = (
        float(np.median(np.abs(pa_arr - np.median(pa_arr))))
        if pa_arr.size else float("nan")
    )
    n_locked = sum(1 for iso in rows if bool(iso.get("lsb_locked", False)))
    return {
        "n_outer_isophotes": len(tail),
        "intens_mad_per_band": intens_mad,
        "eps_mad": eps_mad,
        "pa_mad": pa_mad,
        "n_locked": int(n_locked),
        "lock_fired": bool(result.get("lsb_auto_lock", False)),
        "lock_sma": (
            float(result["lsb_auto_lock_sma"])
            if result.get("lsb_auto_lock_sma") is not None else None
        ),
    }


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
        default=Path("outputs/benchmark_multiband/lsb_auto_lock_perf"),
        type=Path,
    )
    parser.add_argument("--n-repeats", type=int, default=3)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Stage-3 Stage-C lsb_auto_lock benchmark on asteris {args.obj_id} ===")
    print("Loading inputs ...")
    images: List[np.ndarray] = []
    for folder in DEMO_BAND_FOLDERS:
        images.append(_load_denoised(args.asteris_root, args.obj_id, folder))
    mask = _load_object_mask(args.asteris_root, args.obj_id)
    variance_maps = [_uniform_variance_map(im) for im in images]
    h, w = images[0].shape
    print(f"  shape={images[0].shape}, n_masked={int(mask.sum())}")

    print("Warming up numba JITs ...")
    warmup_numba()
    warmup_numba_mb()

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

    configs = [
        ("baseline", False, "mean"),
        ("lock-mean", True, "mean"),
        ("lock-median", True, "median"),
    ]
    timings: List[dict] = [sb_timing]
    results: Dict[str, dict] = {}
    fits_paths: Dict[str, str] = {}
    for label, lock_on, integrator in configs:
        cfg = _build_cfg(h=h, w=w, lock_on=lock_on, locked_integrator=integrator)
        def _run(cfg=cfg) -> object:
            return fit_image_multiband(
                images=images, masks=mask, config=cfg,
                variance_maps=variance_maps,
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            timings.append(_time_run(f"multiband {label}", _run, args.n_repeats))
            res = fit_image_multiband(
                images=images, masks=mask, config=cfg,
                variance_maps=variance_maps,
            )
        results[label] = res
        slug = label.replace("-", "_")
        fits_path = args.out_dir / f"asteris_{args.obj_id}_lsb_auto_lock_{slug}.fits"
        isophote_results_mb_to_fits(res, fits_path, overwrite=True)
        fits_paths[label] = str(fits_path)

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
    print(f"Stage-1 perf bar: ≤{PERF_BAR}× single-band.")

    print()
    print("Lock metadata + outer-tail diagnostics:")
    print(
        f"{'config':<15} {'fired':>6} {'sma':>8} {'n_lock':>7} "
        f"{'eps MAD':>10} {'pa MAD':>10}"
    )
    print("-" * 65)
    metrics: Dict[str, dict] = {}
    for label, _, _ in configs:
        m = _outer_tail_metrics(results[label])
        metrics[label] = m
        sma_str = f"{m['lock_sma']:.1f}" if m["lock_sma"] is not None else "—"
        print(
            f"{label:<15} {str(m['lock_fired']):>6} {sma_str:>8} "
            f"{m['n_locked']:>7} {m['eps_mad']:>10.3e} {m['pa_mad']:>10.3e}"
        )

    perf_path = args.out_dir / f"asteris_{args.obj_id}_lsb_auto_lock_perf.json"
    perf_path.write_text(json.dumps({
        "obj_id": args.obj_id,
        "shape": list(images[0].shape),
        "n_repeats": args.n_repeats,
        "perf_bar_vs_singleband": PERF_BAR,
        "results": timings,
        "ratios_vs_singleband": {t["label"]: t["median_s"] / sb_med for t in timings},
        "fits_paths": fits_paths,
    }, indent=2))
    print(f"Wrote {perf_path}")

    compare_path = args.out_dir / f"asteris_{args.obj_id}_lsb_auto_lock_compare.json"
    compare_path.write_text(json.dumps({
        "obj_id": args.obj_id,
        "lock_metadata_and_outer_tail": metrics,
    }, indent=2))
    print(f"Wrote {compare_path}")


if __name__ == "__main__":
    main()
