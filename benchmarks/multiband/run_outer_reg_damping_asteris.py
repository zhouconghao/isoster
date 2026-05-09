"""Outer-region damping demo on an asteris HSC g/r/i/z/y cutout.

Sweeps three damping configurations on the same asteris denoised
cutout (object 37484563299062823 by default) and writes:

- ``<out_dir>/asteris_<obj_id>_outer_reg_<config>.fits`` per config.
- ``<out_dir>/asteris_<obj_id>_outer_reg_<config>.png`` per-config QA panel.
- ``<out_dir>/asteris_<obj_id>_outer_reg_compare.png`` 4-row cross-mode
  comparison of geometry (x0, y0, eps, pa) vs SMA — outer-reg's primary
  effect surface.
- ``<out_dir>/asteris_<obj_id>_outer_reg_stats.json`` per-config summary
  (n_isophotes, mean RMS, outer-tail eps / pa MAD).

Configurations:
1. ``baseline``        — ``use_outer_center_regularization=False``.
2. ``center-only``     — feature on, ``weights={center: 1, eps: 0, pa: 0}``
   (the historical single-band default before the {1,1,1} extension).
3. ``all-axes``        — feature on, ``weights={center: 1, eps: 1, pa: 1}``
   (Stage-B default; damps all four geometry parameters and prevents
   the selector-asymmetry failure mode where center-only damping
   redirects the outer random walk into eps/pa).

Stage-3 Stage-B (plan section 7).
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

# (label, use_outer_reg, weights). Order matters — drives column order
# in the comparison panel.
DAMPING_CONFIGS = (
    ("baseline", False, {"center": 1.0, "eps": 1.0, "pa": 1.0}),
    ("center-only", True, {"center": 1.0, "eps": 0.0, "pa": 0.0}),
    ("all-axes", True, {"center": 1.0, "eps": 1.0, "pa": 1.0}),
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
    """Sigma-clipped sky std matching ``examples.example_asteris_denoised.common.sky_std``.

    Required: ``astropy.stats.sigma_clipped_stats(sigma=3, maxiters=5)``.
    MAD-based estimators are forbidden (12% bias, handover 2026-05-03).
    """
    from astropy.stats import sigma_clipped_stats

    _, _, std = sigma_clipped_stats(image, sigma=3.0, maxiters=5)
    return float(std)


def _uniform_variance_map(image: np.ndarray) -> np.ndarray:
    return np.full_like(image, _sky_std(image) ** 2, dtype=np.float64)


def _build_cfg(
    bands: List[str], h: int, w: int, *,
    use_outer_reg: bool, weights: Dict[str, float],
) -> IsosterConfigMB:
    with warnings.catch_warnings():
        # Suppress the geometry_convergence auto-enable warning during
        # demo runs; the user-facing test suite already covers it.
        warnings.simplefilter("ignore", UserWarning)
        return IsosterConfigMB(
            bands=bands, reference_band="i",
            sma0=20.0, eps=0.2, pa=0.0, astep=0.1, linear_growth=False,
            maxsma=float(np.hypot(h, w)) / 2.0, debug=True,
            compute_deviations=True, nclip=2, max_retry_first_isophote=5,
            band_weights=None,
            use_outer_center_regularization=use_outer_reg,
            outer_reg_sma_onset=80.0,
            outer_reg_strength=2.0,
            outer_reg_weights=weights,
        )


def _outer_tail_metrics(result: dict, bands: List[str], n_outer: int = 20) -> dict:
    """Outer-tail diagnostics: per-band intens MAD, eps/pa MAD over the
    last ``n_outer`` valid isophotes. Outer-reg's primary effect is
    suppressing geometry scatter in this regime."""
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
    return {
        "n_outer_isophotes": len(tail),
        "intens_mad_per_band": intens_mad,
        "eps_mad": eps_mad,
        "pa_mad": pa_mad,
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
    """4-row cross-mode comparison of geometry vs SMA.

    Rows: x0, y0, eps, pa. Columns: one per damping config. Per-config
    overlay shows the geometry trajectory; outer-reg's job is to keep
    these flat (or near the inner-region reference) above the onset.
    """
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
        for row_idx, key in enumerate(rows):
            ax = axes[row_idx, col_idx]
            y = np.asarray([float(r.get(key, np.nan)) for r in iso])
            ax.plot(sma, y, lw=1.0, color="#1f77b4", alpha=0.9)
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
    for label, on, weights in DAMPING_CONFIGS:
        print(f"=== Running asteris {obj_id} with config={label!r} ===")
        cfg = _build_cfg(DEMO_BANDS, h, w, use_outer_reg=on, weights=weights)
        res = fit_image_multiband(
            images=images, masks=mask, config=cfg, variance_maps=variance_maps,
        )
        results_by_label[label] = res
        stats_by_label[label] = _summarize(res, DEMO_BANDS)

        slug = label.replace(" ", "_").replace("-", "_")
        fits_path = out_dir / f"asteris_{obj_id}_outer_reg_{slug}.fits"
        isophote_results_mb_to_fits(res, str(fits_path))
        print(f"  wrote {fits_path}")

        # Sky-offset post-process for the QA figure (mirrors the
        # higher-harmonics demo). Raw FITS keeps uncorrected values.
        res_corr, sky_offsets = subtract_outermost_sky_offset(
            res, n_outer=n_outer_for_sky,
        )
        images_corr = [im - sky_offsets[b] for im, b in zip(images, DEMO_BANDS)]

        qa_path = out_dir / f"asteris_{obj_id}_outer_reg_{slug}.png"
        plot_qa_summary_mb(
            result=res_corr,
            images=images_corr,
            bands=DEMO_BANDS,
            sb_zeropoint=HSC_ZP,
            pixel_scale_arcsec=HSC_PIXEL_SCALE_ARCSEC,
            softening_per_band=softening,
            object_mask=mask,
            output_path=str(qa_path),
            title=f"asteris {obj_id} — outer_reg config: {label}",
        )
        print(
            f"  wrote {qa_path}  (sky offsets: "
            + ", ".join(f"{b}={sky_offsets[b]:+.3g}" for b in DEMO_BANDS) + ")"
        )

    cmp_path = out_dir / f"asteris_{obj_id}_outer_reg_compare.png"
    _comparison_figure(
        results_by_label, cmp_path,
        title_prefix=f"asteris {obj_id} — geometry (x0/y0/eps/pa) vs SMA across damping configs",
    )
    print(f"Wrote {cmp_path}")

    stats_path = out_dir / f"asteris_{obj_id}_outer_reg_stats.json"
    stats_path.write_text(json.dumps(stats_by_label, indent=2))
    print(f"Wrote {stats_path}")

    # Console summary.
    print()
    print(
        f"{'config':<15} {'n_iso':>5} {'conv':>5} {'mean_rms':>10}  "
        f"{'eps MAD':>10} {'pa MAD':>10}"
    )
    print("-" * 70)
    for label, _, _ in DAMPING_CONFIGS:
        s = stats_by_label[label]
        ot = s["outer_tail"]
        print(
            f"{label:<15} {s['n_isophotes']:>5} {s['n_converged']:>5} "
            f"{s['mean_rms']:>10.3e}  "
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
        default=Path("outputs/benchmark_multiband/outer_reg_damping_asteris"),
        type=Path,
    )
    parser.add_argument(
        "--n-outer-for-sky", type=int, default=8,
        help="Number of outermost valid isophotes per band whose median "
        "I0_b is used as the inferred sky offset (subtracted from "
        "intens_<b> and the images before plotting). Default 8.",
    )
    args = parser.parse_args()
    run(args.asteris_root, args.obj_id, args.out_dir,
        n_outer_for_sky=args.n_outer_for_sky)


if __name__ == "__main__":
    main()
