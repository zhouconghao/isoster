"""Per-band curve-of-growth demo on an asteris HSC g/r/i/z/y cutout.

Single-config demo (``compute_cog=True``) on the asteris denoised
cutout (object 37484563299062823 by default). Writes:

- ``<out_dir>/asteris_<obj_id>_compute_cog.fits`` (Schema-1 round-
  trippable; per-band ``cog_<b>`` / ``cog_annulus_<b>`` plus shared
  ``area_annulus`` / ``flag_*`` columns are stamped into the
  ISOPHOTES table).
- ``<out_dir>/asteris_<obj_id>_compute_cog.png`` standard QA mosaic.
- ``<out_dir>/asteris_<obj_id>_compute_cog_curves.png`` per-band
  cumulative-flux profile vs SMA, plus the annular flux trace.
- ``<out_dir>/asteris_<obj_id>_compute_cog_stats.json`` summary
  (n_isophotes, total cog per band at the outermost isophote,
  half-light SMA per band — the SMA where cog reaches half its
  maximum).

Stage-3 Stage-D (plan section 7 S7).
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
BAND_COLORS = {b: c for b, c in zip(
    DEMO_BANDS, ("#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"),
)}


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
    """``astropy.stats.sigma_clipped_stats(sigma=3, maxiters=5)`` —
    MAD-based estimators forbidden (12% bias)."""
    from astropy.stats import sigma_clipped_stats

    _, _, std = sigma_clipped_stats(image, sigma=3.0, maxiters=5)
    return float(std)


def _uniform_variance_map(image: np.ndarray) -> np.ndarray:
    return np.full_like(image, _sky_std(image) ** 2, dtype=np.float64)


def _half_light_sma(sma: np.ndarray, cog: np.ndarray) -> float:
    """Linear interpolation of the SMA at which cog reaches half of
    its outermost finite value. Returns NaN if cog is all NaN or
    monotonically zero."""
    finite = np.isfinite(cog)
    if int(finite.sum()) < 2:
        return float("nan")
    sma_f = sma[finite]
    cog_f = cog[finite]
    cog_max = float(cog_f[-1])
    if cog_max <= 0.0:
        return float("nan")
    half = 0.5 * cog_max
    idx = int(np.searchsorted(cog_f, half))
    if idx == 0:
        return float(sma_f[0])
    if idx >= cog_f.size:
        return float(sma_f[-1])
    # Linear interpolation between idx-1 and idx.
    s_lo, s_hi = float(sma_f[idx - 1]), float(sma_f[idx])
    c_lo, c_hi = float(cog_f[idx - 1]), float(cog_f[idx])
    if c_hi == c_lo:
        return s_lo
    return s_lo + (half - c_lo) * (s_hi - s_lo) / (c_hi - c_lo)


def _curves_figure(
    result: dict, result_sky: dict, sky_offsets: Dict[str, float],
    bands: List[str], out_path: Path, title: str,
) -> None:
    """Two-panel figure: cumulative-flux profile (cog) and annular
    flux trace (cog_annulus), per-band overlay. Solid = raw CoG
    (joint solver's I0_b carries residual sky); dashed = background-
    corrected CoG (sky offset subtracted before trapezoidal
    integration). Past the LSB transition the two curves diverge —
    the raw CoG dips because the residual sky goes slightly negative
    on these denoised images."""
    rows = result["isophotes"]
    rows_sky = result_sky["isophotes"]
    sma = np.array([float(r["sma"]) for r in rows])
    fig, (ax_cog, ax_ann) = plt.subplots(
        2, 1, figsize=(7.0, 6.0), sharex=True,
    )
    for b in bands:
        cog_raw = np.array([float(r.get(f"cog_{b}", float("nan"))) for r in rows])
        ann_raw = np.array([float(r.get(f"cog_annulus_{b}", float("nan"))) for r in rows])
        cog_corr = np.array([float(r.get(f"cog_{b}", float("nan"))) for r in rows_sky])
        ann_corr = np.array([float(r.get(f"cog_annulus_{b}", float("nan"))) for r in rows_sky])
        color = BAND_COLORS.get(b, None)
        # Solid = raw, dashed = background-corrected.
        ax_cog.plot(sma, cog_raw, color=color, lw=1.4, ls="-", label=b)
        ax_cog.plot(sma, cog_corr, color=color, lw=1.4, ls="--", alpha=0.85)
        ax_ann.plot(sma, ann_raw, color=color, lw=1.0, ls="-")
        ax_ann.plot(sma, ann_corr, color=color, lw=1.0, ls="--", alpha=0.85)
    ax_cog.set_ylabel("Cumulative flux (cog)")
    band_legend = ax_cog.legend(loc="lower right", frameon=False, title="band")
    ax_cog.add_artist(band_legend)
    style_handles = [
        plt.Line2D([0], [0], color="0.3", ls="-", lw=1.4, label="raw"),
        plt.Line2D([0], [0], color="0.3", ls="--", lw=1.4, label="bg-corrected"),
    ]
    ax_cog.legend(handles=style_handles, loc="upper left", frameon=False)
    ax_cog.grid(True, alpha=0.3)
    ax_ann.set_ylabel("Annular flux (cog_annulus)")
    ax_ann.set_xlabel("SMA (px)")
    ax_ann.grid(True, alpha=0.3)
    ax_ann.axhline(0.0, color="0.6", lw=0.5, zorder=-1)
    sky_text = ", ".join(f"{b}={sky_offsets.get(b, 0.0):+.3g}" for b in bands)
    fig.suptitle(f"{title}\nsky offsets subtracted: {sky_text}", fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cfg = IsosterConfigMB(
            bands=DEMO_BANDS, reference_band="i",
            sma0=20.0, eps=0.2, pa=0.0, astep=0.1, linear_growth=False,
            maxsma=float(np.hypot(h, w)) / 2.0, debug=True,
            compute_deviations=True, nclip=2, max_retry_first_isophote=5,
            band_weights=None,
            compute_cog=True,
        )

    print(f"=== Running asteris {obj_id} with compute_cog=True ===")
    res = fit_image_multiband(
        images=images, masks=mask, config=cfg, variance_maps=variance_maps,
    )

    fits_path = out_dir / f"asteris_{obj_id}_compute_cog.fits"
    isophote_results_mb_to_fits(res, str(fits_path))
    print(f"  wrote {fits_path}")

    res_corr, sky_offsets = subtract_outermost_sky_offset(
        res, n_outer=n_outer_for_sky,
    )
    images_corr = [im - sky_offsets[b] for im, b in zip(images, DEMO_BANDS)]

    # Stage-3 Stage-D follow-up: recompute CoG with sky offsets
    # subtracted from intens_<b> before the trapezoidal integration.
    # The raw CoG (already stamped onto res by the driver) shows the
    # cumulative flux of the joint solver's per-band intercept,
    # which carries residual sky and dips past the LSB transition.
    # The background-corrected CoG asymptotes properly.
    from isoster.multiband.cog_mb import (
        add_cog_mb_to_isophotes, compute_cog_mb,
    )
    import copy as _copy
    res_bg = _copy.deepcopy(res)  # avoid stomping the raw cog_<b> columns
    cog_bg = compute_cog_mb(
        res_bg["isophotes"], bands=DEMO_BANDS, sky_offsets=sky_offsets,
    )
    add_cog_mb_to_isophotes(res_bg["isophotes"], DEMO_BANDS, cog_bg)

    qa_path = out_dir / f"asteris_{obj_id}_compute_cog.png"
    plot_qa_summary_mb(
        result=res_corr,
        images=images_corr,
        bands=DEMO_BANDS,
        sb_zeropoint=HSC_ZP,
        pixel_scale_arcsec=HSC_PIXEL_SCALE_ARCSEC,
        softening_per_band=softening,
        object_mask=mask,
        output_path=str(qa_path),
        title=f"asteris {obj_id} — compute_cog=True",
    )
    print(f"  wrote {qa_path}")

    curves_path = out_dir / f"asteris_{obj_id}_compute_cog_curves.png"
    _curves_figure(
        res, res_bg, sky_offsets, DEMO_BANDS, curves_path,
        title=f"asteris {obj_id} — per-band CoG (raw vs background-corrected)",
    )
    print(f"  wrote {curves_path}")

    # Per-band summary stats: cog totals + half-light SMAs for both
    # the raw CoG and the background-corrected CoG (sky offsets
    # subtracted before integration).
    rows = res["isophotes"]
    rows_bg = res_bg["isophotes"]
    sma_arr = np.array([float(r["sma"]) for r in rows])
    summary_raw: Dict[str, Dict[str, float]] = {}
    summary_bg: Dict[str, Dict[str, float]] = {}
    for b in DEMO_BANDS:
        cog_raw = np.array([float(r.get(f"cog_{b}", float("nan"))) for r in rows])
        cog_bg_arr = np.array([float(r.get(f"cog_{b}", float("nan"))) for r in rows_bg])
        finite_raw = np.isfinite(cog_raw)
        finite_bg = np.isfinite(cog_bg_arr)
        summary_raw[b] = {
            "cog_total_outermost": (
                float(cog_raw[finite_raw][-1]) if int(finite_raw.sum()) > 0
                else float("nan")
            ),
            "half_light_sma_px": _half_light_sma(sma_arr, cog_raw),
        }
        summary_bg[b] = {
            "cog_total_outermost": (
                float(cog_bg_arr[finite_bg][-1]) if int(finite_bg.sum()) > 0
                else float("nan")
            ),
            "half_light_sma_px": _half_light_sma(sma_arr, cog_bg_arr),
        }

    stats_path = out_dir / f"asteris_{obj_id}_compute_cog_stats.json"
    stats_path.write_text(json.dumps({
        "obj_id": obj_id,
        "n_isophotes": len(rows),
        "shape": list(images[0].shape),
        "sky_offsets": sky_offsets,
        "per_band_summary_raw": summary_raw,
        "per_band_summary_bg_corrected": summary_bg,
    }, indent=2))
    print(f"Wrote {stats_path}")

    print()
    print(
        f"{'band':<6} {'raw cog_total':>14} {'raw r1/2':>10} "
        f"{'bg cog_total':>14} {'bg r1/2':>10}"
    )
    print("-" * 60)
    for b in DEMO_BANDS:
        sr = summary_raw[b]
        sb = summary_bg[b]
        print(
            f"{b:<6} {sr['cog_total_outermost']:>14.4e} "
            f"{sr['half_light_sma_px']:>10.2f} "
            f"{sb['cog_total_outermost']:>14.4e} "
            f"{sb['half_light_sma_px']:>10.2f}"
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
        default=Path("outputs/benchmark_multiband/compute_cog_asteris"),
        type=Path,
    )
    parser.add_argument("--n-outer-for-sky", type=int, default=8)
    args = parser.parse_args()
    run(args.asteris_root, args.obj_id, args.out_dir,
        n_outer_for_sky=args.n_outer_for_sky)


if __name__ == "__main__":
    main()
