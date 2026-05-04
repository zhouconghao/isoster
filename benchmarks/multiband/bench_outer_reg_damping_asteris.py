"""Stage-3 Stage-B perf + numerical benchmark for outer-region damping.

Compares three multi-band configurations on the asteris denoised B=5
cutout (object 37484563299062823):

1. ``baseline``                 — no outer-region regularization. The
   Stage-1 default; Stage-A bit-identical reference for inner / mid
   isophotes.
2. ``damping center-only``      — ``use_outer_center_regularization=True``,
   ``outer_reg_weights={center: 1, eps: 0, pa: 0}``. Damps only x0/y0
   in the LSB regime; reproduces the historical single-band default
   before the {1,1,1} extension landed.
3. ``damping all axes``         — ``outer_reg_weights={center: 1, eps: 1,
   pa: 1}`` (Stage-B default). Damps all four geometry parameters and
   prevents the selector-asymmetry failure mode where center-only
   damping redirects the outer random walk into eps/pa.

Stage-B quality bars (plan section 7, Phase 38):

- All three configs PASS the canonical Stage-1 ``≤ 2.5×`` single-band
  bar.
- ``damping`` adds modest overhead vs ``baseline`` (the per-axis alpha
  is a handful of float ops per iteration); aim for ``≤ 1.20×`` of the
  baseline multi-band path.
- Outer-tail (last 20 valid isophotes) per-band intens MAD and PA
  scatter shrink under the {1,1,1} damper vs baseline — when masking
  produces saturated clipped jumps the damper suppresses them.

Outputs:
- ``<out_dir>/asteris_<obj_id>_outer_reg_damping_perf.json`` — raw
  timings + ratios + FITS paths.
- ``<out_dir>/asteris_<obj_id>_outer_reg_damping_compare.json`` —
  per-mode outer-tail diagnostics.
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
DAMPING_OVERHEAD_BAR = 1.20


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
    """Sigma-clipped sky std matching ``examples.example_asteris_denoised.common.sky_std``.

    Required: ``astropy.stats.sigma_clipped_stats(sigma=3, maxiters=5)``;
    MAD-based estimators are forbidden (12% bias, handover 2026-05-03).
    """
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


def _make_cfg(
    *, h: int, w: int, outer_reg: bool, weights: Dict[str, float],
) -> IsosterConfigMB:
    with warnings.catch_warnings():
        # Auto-enable geometry_convergence emits a UserWarning we want
        # suppressed at construction; the user-facing test suite
        # already covers the warning emission.
        warnings.simplefilter("ignore", UserWarning)
        return IsosterConfigMB(
            bands=DEMO_BANDS, reference_band="i",
            sma0=20.0, eps=0.2, pa=0.0, astep=0.1, linear_growth=False,
            maxsma=float(np.hypot(h, w)) / 2.0, debug=True,
            compute_deviations=True, nclip=2, max_retry_first_isophote=5,
            band_weights=None,
            use_outer_center_regularization=outer_reg,
            outer_reg_sma_onset=80.0,  # mid-galaxy → outer LSB transition
            outer_reg_strength=2.0,
            outer_reg_weights=weights,
        )


def _outer_tail_metrics(result: dict) -> Dict[str, Dict[str, float]]:
    """Per-band outer-tail (last 20 valid) intens MAD + geometry-axis scatter."""
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
        if finite.size < 4:
            intens_mad[b] = float("nan")
        else:
            intens_mad[b] = float(np.median(np.abs(finite - np.median(finite))))

    eps_arr = np.array([float(r.get("eps", float("nan"))) for r in tail])
    pa_arr = np.array([float(r.get("pa", float("nan"))) for r in tail])
    eps_mad = float(np.median(np.abs(eps_arr - np.median(eps_arr)))) if eps_arr.size else float("nan")
    pa_mad = float(np.median(np.abs(pa_arr - np.median(pa_arr)))) if pa_arr.size else float("nan")
    return {"intens_mad": intens_mad, "eps_mad": eps_mad, "pa_mad": pa_mad}


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
        default=Path("outputs/benchmark_multiband/outer_reg_damping_perf"),
        type=Path,
    )
    parser.add_argument("--n-repeats", type=int, default=3)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Stage-3 Stage-B outer-reg-damping benchmark on asteris {args.obj_id} ===")
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
        ("baseline (no outer-reg)", False, {"center": 1.0, "eps": 1.0, "pa": 1.0}),
        ("damping center-only", True, {"center": 1.0, "eps": 0.0, "pa": 0.0}),
        ("damping all axes", True, {"center": 1.0, "eps": 1.0, "pa": 1.0}),
    ]
    timings: List[dict] = [sb_timing]
    results: Dict[str, dict] = {}
    fits_paths: Dict[str, str] = {}
    for label, on, weights in configs:
        cfg = _make_cfg(h=h, w=w, outer_reg=on, weights=weights)
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
        slug = label.replace(" (no outer-reg)", "").replace(" ", "_").replace("-", "_")
        fits_path = args.out_dir / f"asteris_{args.obj_id}_{slug}.fits"
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

    baseline_med = next(t["median_s"] for t in timings if "baseline" in t["label"])
    damping_all_med = next(
        t["median_s"] for t in timings if "damping all axes" in t["label"]
    )
    damping_overhead = damping_all_med / baseline_med
    print()
    print(
        f"Damping-vs-baseline overhead: {damping_overhead:.2f}× "
        f"({'PASS' if damping_overhead <= DAMPING_OVERHEAD_BAR else 'FAIL'} "
        f"<= {DAMPING_OVERHEAD_BAR}×)"
    )
    print(f"Stage-1 perf bar: ≤{PERF_BAR}× single-band.")

    # Outer-tail diagnostics.
    print()
    print("Outer-tail (last 20 valid isophotes) diagnostics:")
    print(
        f"{'configuration':<40} {'eps MAD':>10} {'pa MAD':>10} "
        + "  ".join(f"intens_{b} MAD".rjust(15) for b in DEMO_BANDS)
    )
    print("-" * (62 + 17 * len(DEMO_BANDS)))
    metrics: Dict[str, Dict[str, object]] = {}
    for label, res in results.items():
        m = _outer_tail_metrics(res)
        metrics[label] = m
        intens_str = "  ".join(
            f"{m['intens_mad'][b]:>15.3e}"
            if isinstance(m["intens_mad"], dict)
            and np.isfinite(m["intens_mad"][b])
            else f"{'nan':>15}"
            for b in DEMO_BANDS
        )
        print(
            f"{label:<40} {m['eps_mad']:>10.3e} {m['pa_mad']:>10.3e} "
            f"{intens_str}"
        )

    perf_path = args.out_dir / f"asteris_{args.obj_id}_outer_reg_damping_perf.json"
    perf_path.write_text(json.dumps({
        "obj_id": args.obj_id,
        "shape": list(images[0].shape),
        "n_repeats": args.n_repeats,
        "perf_bar_vs_singleband": PERF_BAR,
        "damping_overhead_bar": DAMPING_OVERHEAD_BAR,
        "damping_overhead_observed": damping_overhead,
        "results": timings,
        "ratios_vs_singleband": {t["label"]: t["median_s"] / sb_med for t in timings},
        "fits_paths": fits_paths,
    }, indent=2))
    print(f"Wrote {perf_path}")

    compare_path = args.out_dir / f"asteris_{args.obj_id}_outer_reg_damping_compare.json"
    compare_path.write_text(json.dumps({
        "obj_id": args.obj_id,
        "outer_tail_metrics": metrics,
        "n_outer_isophotes": 20,
    }, indent=2))
    print(f"Wrote {compare_path}")


if __name__ == "__main__":
    main()
