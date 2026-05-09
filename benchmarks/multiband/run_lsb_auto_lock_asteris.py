"""LSB auto-lock demo on an asteris HSC g/r/i/z/y cutout.

Sweeps three lock configurations on the same asteris denoised cutout
(object 37484563299062823 by default) and writes:

- ``<out_dir>/asteris_<obj_id>_lsb_auto_lock_<config>.fits`` per config.
- ``<out_dir>/asteris_<obj_id>_lsb_auto_lock_<config>.png`` per-config
  QA mosaic.
- ``<out_dir>/asteris_<obj_id>_lsb_auto_lock_compare.png`` 4-row
  cross-config geometry-vs-SMA panel — outer-isophote behavior is
  where the lock's effect shows.
- ``<out_dir>/asteris_<obj_id>_lsb_auto_lock_stats.json`` per-config
  summary (n_isophotes, mean RMS, lock metadata, outer-tail eps/pa MAD).

Configurations:

1. ``baseline``     — ``lsb_auto_lock=False``. Stage-1 default.
2. ``lock-mean``    — ``lsb_auto_lock=True`` with
   ``lsb_auto_lock_integrator='mean'``. Matrix-mode joint solve
   preserved (Stage-A S1 satisfied by the mean integrator).
3. ``lock-median``  — ``lsb_auto_lock=True`` with
   ``lsb_auto_lock_integrator='median'`` and
   ``fit_per_band_intens_jointly=False``. Locked-region intercepts
   use per-band median for contaminant robustness; intercept mode
   flipped to satisfy Stage-A S1.

Caveat — the lock is designed for massive ellipticals / cD galaxies
with extended LSB envelopes; on barred / spiral systems it can pin
real outer-disc evolution to the inner reference. See
``docs/10-multiband.md`` for the galaxy-type caveat.

Stage-3 Stage-C (plan section 7).
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from isoster.multiband import (
    IsosterConfigMB,
    fit_image_multiband,
    isophote_results_mb_to_fits,
    plot_qa_summary_mb,
    subtract_outermost_sky_offset,
)

DEMO_BAND_FOLDERS = ["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"]
DEMO_BANDS = ["g", "r", "i", "z", "y"]
HSC_ZP = 27.0
HSC_PIXEL_SCALE_ARCSEC = 0.168
ASTERIS_OBJ_ID = "37484563299062823"

# (label, lsb_auto_lock, integrator). Order drives the cross-config
# comparison panel column order.
LOCK_CONFIGS = (
    ("baseline", False, "mean"),
    ("lock-mean", True, "mean"),
    ("lock-median", True, "median"),
)


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
    """Required ``astropy.stats.sigma_clipped_stats(sigma=3, maxiters=5)``;
    MAD-based estimators are forbidden (12% bias)."""
    from astropy.stats import sigma_clipped_stats

    _, _, std = sigma_clipped_stats(image, sigma=3.0, maxiters=5)
    return float(std)


def _uniform_variance_map(image: np.ndarray) -> np.ndarray:
    return np.full_like(image, _sky_std(image) ** 2, dtype=np.float64)


def _build_cfg(
    bands: List[str], h: int, w: int, *,
    lock_on: bool, integrator: str,
) -> IsosterConfigMB:
    with warnings.catch_warnings():
        # Suppress the auto-enable-debug warning during demo runs.
        warnings.simplefilter("ignore", UserWarning)
        kwargs = dict(
            bands=bands, reference_band="i",
            sma0=20.0, eps=0.2, pa=0.0, astep=0.1, linear_growth=False,
            maxsma=float(np.hypot(h, w)) / 2.0, debug=True,
            compute_deviations=True, nclip=2, max_retry_first_isophote=5,
            band_weights=None,
            lsb_auto_lock=lock_on,
            lsb_auto_lock_integrator=integrator,
            lsb_auto_lock_maxgerr=0.3,
            lsb_auto_lock_debounce=2,
        )
        if lock_on and integrator == "median":
            kwargs["fit_per_band_intens_jointly"] = False  # Stage-A S1
        return IsosterConfigMB(**kwargs)


def _outer_tail_metrics(result: dict, bands: List[str], n_outer: int = 20) -> dict:
    rows = result["isophotes"]
    smas = np.array([float(r.get("sma", float("nan"))) for r in rows])
    valid = np.isfinite(smas)
    valid_rows = [r for r, v in zip(rows, valid) if v]
    tail = valid_rows[-n_outer:] if len(valid_rows) >= n_outer else valid_rows

    intens_mad: Dict[str, float] = {}
    for b in bands:
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


def _summarize(result: dict, bands: List[str]) -> dict:
    iso = result["isophotes"]
    if not iso:
        return {"n_isophotes": 0}
    sma_arr = np.asarray([r["sma"] for r in iso], dtype=np.float64)
    stop_codes = [int(r["stop_code"]) for r in iso]
    rms = np.asarray([r["rms"] for r in iso], dtype=np.float64)
    rms_finite = rms[np.isfinite(rms)]
    return {
        "n_isophotes": len(iso),
        "n_converged": sum(1 for c in stop_codes if c == 0),
        "stop_code_counts": {
            str(c): stop_codes.count(c) for c in sorted(set(stop_codes))
        },
        "sma_range": [float(sma_arr.min()), float(sma_arr.max())],
        "mean_rms": float(np.nanmean(rms_finite)) if rms_finite.size else float("nan"),
        "median_rms": float(np.nanmedian(rms_finite)) if rms_finite.size else float("nan"),
        "outer_tail": _outer_tail_metrics(result, bands),
    }


def _comparison_figure(
    results_by_label: Dict[str, dict],
    out_path: Path,
    title_prefix: str,
) -> None:
    """4-row cross-config geometry-vs-SMA panel. Vertical line marks the
    lock-fire SMA per config (only on configs where the lock fired)."""
    rows = ("x0", "y0", "eps", "pa")
    n_rows = len(rows)
    cols = list(results_by_label.keys())
    n_cols = len(cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.6 * n_cols, 2.4 * n_rows),
        sharex=True, sharey="row",
    )
    if n_rows == 1:
        axes = axes[None, :]
    for col_idx, label in enumerate(cols):
        result = results_by_label.get(label)
        if result is None:
            continue
        iso = result["isophotes"]
        sma = np.asarray([r["sma"] for r in iso], dtype=np.float64)
        lock_sma = (
            float(result["lsb_auto_lock_sma"])
            if result.get("lsb_auto_lock_sma") is not None else None
        )
        for row_idx, key in enumerate(rows):
            ax = axes[row_idx, col_idx]
            y = np.asarray([float(r.get(key, np.nan)) for r in iso])
            ax.plot(sma, y, lw=1.0, color="#1f77b4", alpha=0.9)
            if lock_sma is not None:
                ax.axvline(lock_sma, color="#d62728", lw=0.8, ls="--", alpha=0.7)
            if row_idx == 0:
                ax.set_title(label, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(key)
            if row_idx == n_rows - 1:
                ax.set_xlabel("SMA (px)")
    fig.suptitle(title_prefix, fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(
    asteris_root: Path, obj_id: str, out_dir: Path,
    n_outer_for_sky: int = 8,
) -> None:
    print(f"=== Loading asteris cutout {obj_id} ===")
    images: List[np.ndarray] = []
    sky_rms_per_band: Dict[str, float] = {}
    for folder, b in zip(DEMO_BAND_FOLDERS, DEMO_BANDS):
        img = _load_denoised(asteris_root, obj_id, folder)
        images.append(img)
        sky_rms_per_band[b] = _sky_std(img)
    mask = _load_object_mask(asteris_root, obj_id)
    variance_maps = [_uniform_variance_map(im) for im in images]
    h, w = images[0].shape

    out_dir.mkdir(parents=True, exist_ok=True)
    softening = {b: max(s, 1e-6) for b, s in sky_rms_per_band.items()}

    results_by_label: Dict[str, dict] = {}
    stats_by_label: Dict[str, dict] = {}
    for label, lock_on, integrator in LOCK_CONFIGS:
        print(f"=== Running asteris {obj_id} with config={label!r} ===")
        cfg = _build_cfg(DEMO_BANDS, h, w, lock_on=lock_on, integrator=integrator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = fit_image_multiband(
                images=images, masks=mask, config=cfg, variance_maps=variance_maps,
            )
        results_by_label[label] = res
        stats_by_label[label] = _summarize(res, DEMO_BANDS)

        slug = label.replace("-", "_")
        fits_path = out_dir / f"asteris_{obj_id}_lsb_auto_lock_{slug}.fits"
        isophote_results_mb_to_fits(res, str(fits_path))
        print(f"  wrote {fits_path}")

        res_corr, sky_offsets = subtract_outermost_sky_offset(
            res, n_outer=n_outer_for_sky,
        )
        images_corr = [im - sky_offsets[b] for im, b in zip(images, DEMO_BANDS)]

        qa_path = out_dir / f"asteris_{obj_id}_lsb_auto_lock_{slug}.png"
        plot_qa_summary_mb(
            result=res_corr,
            images=images_corr,
            bands=DEMO_BANDS,
            sb_zeropoint=HSC_ZP,
            pixel_scale_arcsec=HSC_PIXEL_SCALE_ARCSEC,
            softening_per_band=softening,
            object_mask=mask,
            output_path=str(qa_path),
            title=f"asteris {obj_id} — lsb_auto_lock config: {label}",
        )
        print(
            f"  wrote {qa_path}  (sky offsets: "
            + ", ".join(f"{b}={sky_offsets[b]:+.3g}" for b in DEMO_BANDS) + ")"
        )

    cmp_path = out_dir / f"asteris_{obj_id}_lsb_auto_lock_compare.png"
    _comparison_figure(
        results_by_label, cmp_path,
        title_prefix=f"asteris {obj_id} — geometry vs SMA across lsb_auto_lock configs (red dashed = lock-fire SMA)",
    )
    print(f"Wrote {cmp_path}")

    stats_path = out_dir / f"asteris_{obj_id}_lsb_auto_lock_stats.json"
    stats_path.write_text(json.dumps(stats_by_label, indent=2))
    print(f"Wrote {stats_path}")

    print()
    print(
        f"{'config':<15} {'n_iso':>5} {'conv':>5} {'lock':>6} "
        f"{'lock SMA':>9} {'n_locked':>9} {'eps MAD':>10} {'pa MAD':>10}"
    )
    print("-" * 90)
    for label, _, _ in LOCK_CONFIGS:
        s = stats_by_label[label]
        ot = s["outer_tail"]
        sma_str = f"{ot['lock_sma']:.1f}" if ot["lock_sma"] is not None else "—"
        print(
            f"{label:<15} {s['n_isophotes']:>5} {s['n_converged']:>5} "
            f"{str(ot['lock_fired']):>6} {sma_str:>9} {ot['n_locked']:>9} "
            f"{ot['eps_mad']:>10.3e} {ot['pa_mad']:>10.3e}"
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
        default=Path("outputs/benchmark_multiband/lsb_auto_lock_asteris"),
        type=Path,
    )
    parser.add_argument("--n-outer-for-sky", type=int, default=8)
    args = parser.parse_args()
    run(args.asteris_root, args.obj_id, args.out_dir,
        n_outer_for_sky=args.n_outer_for_sky)


if __name__ == "__main__":
    main()
