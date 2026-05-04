"""Per-band curve-of-growth demo on PGC006669 (LegacySurvey grz).

Single-config demo (``compute_cog=True``) on the PGC006669 cutout.
Writes:

- ``<out_dir>/PGC006669_compute_cog.fits`` (Schema-1 round-trippable).
- ``<out_dir>/PGC006669_compute_cog.png`` standard QA mosaic.
- ``<out_dir>/PGC006669_compute_cog_curves.png`` per-band cumulative
  flux profile + annular flux trace.
- ``<out_dir>/PGC006669_compute_cog_stats.json`` summary (cog total +
  half-light SMA per band).

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

from legacysurvey_loader import (
    LEGACYSURVEY_ZP,
    asinh_softening_from_log10_match,
    load_legacysurvey_grz,
)


GRZ_COLORS = {"g": "#1f77b4", "r": "#2ca02c", "z": "#9467bd"}


def _half_light_sma(sma: np.ndarray, cog: np.ndarray) -> float:
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
    s_lo, s_hi = float(sma_f[idx - 1]), float(sma_f[idx])
    c_lo, c_hi = float(cog_f[idx - 1]), float(cog_f[idx])
    if c_hi == c_lo:
        return s_lo
    return s_lo + (half - c_lo) * (s_hi - s_lo) / (c_hi - c_lo)


def _curves_figure(
    result: dict, result_sky: dict, sky_offsets: Dict[str, float],
    bands: List[str], out_path: Path, title: str,
) -> None:
    """Two-panel figure: cumulative-flux + annular-flux per band.
    Solid = raw CoG (joint solver's I0_b carries residual sky);
    dashed = background-corrected CoG (per-band sky offset
    subtracted before trapezoidal integration)."""
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
        color = GRZ_COLORS.get(b, None)
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


def run(galaxy_dir: Path, galaxy_prefix: str, out_dir: Path) -> None:
    cutout = load_legacysurvey_grz(galaxy_dir, galaxy_prefix)
    bands = list(cutout.bands)
    masks_per_band = [cutout.combined_mask.copy() for _ in bands]
    maxsma = float(min(cutout.shape) * 0.45)

    out_dir.mkdir(parents=True, exist_ok=True)
    softening = {
        b: asinh_softening_from_log10_match(
            cutout.pixel_scale_arcsec, LEGACYSURVEY_ZP, bright_mu=22.0,
        )
        for b in bands
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cfg = IsosterConfigMB(
            bands=bands, reference_band="r",
            harmonic_combination="joint",
            band_weights={b: 1.0 for b in bands},
            sma0=10.0, minsma=1.0, maxsma=maxsma,
            astep=0.10, linear_growth=False, debug=True,
            compute_cog=True,
        )

    print(f"=== Running PGC006669 with compute_cog=True ===")
    res = fit_image_multiband(
        cutout.images, masks_per_band, cfg, variance_maps=cutout.variances,
    )

    fits_path = out_dir / f"{galaxy_prefix}_compute_cog.fits"
    isophote_results_mb_to_fits(res, str(fits_path))
    print(f"  wrote {fits_path}")

    # Per-band sky correction post-process (median I0_b over the
    # outermost N=8 valid isophotes, mirroring the asteris pipeline).
    _, sky_offsets = subtract_outermost_sky_offset(res, n_outer=8)

    # Stage-3 Stage-D follow-up: recompute CoG with the sky offsets
    # subtracted from intens_<b> before the trapezoidal integration.
    from isoster.multiband.cog_mb import (
        add_cog_mb_to_isophotes, compute_cog_mb,
    )
    import copy as _copy
    res_bg = _copy.deepcopy(res)
    cog_bg = compute_cog_mb(
        res_bg["isophotes"], bands=bands, sky_offsets=sky_offsets,
    )
    add_cog_mb_to_isophotes(res_bg["isophotes"], bands, cog_bg)

    qa_path = out_dir / f"{galaxy_prefix}_compute_cog.png"
    plot_qa_summary_mb(
        result=res,
        images=cutout.images,
        bands=bands,
        sb_zeropoint=LEGACYSURVEY_ZP,
        pixel_scale_arcsec=cutout.pixel_scale_arcsec,
        softening_per_band=softening,
        object_mask=cutout.combined_mask,
        output_path=str(qa_path),
        title=f"PGC006669 — compute_cog=True",
    )
    print(f"  wrote {qa_path}")

    curves_path = out_dir / f"{galaxy_prefix}_compute_cog_curves.png"
    _curves_figure(
        res, res_bg, sky_offsets, bands, curves_path,
        title=f"PGC006669 — per-band CoG (raw vs background-corrected)",
    )
    print(f"  wrote {curves_path}")

    rows = res["isophotes"]
    rows_bg = res_bg["isophotes"]
    sma_arr = np.array([float(r["sma"]) for r in rows])
    summary_raw: Dict[str, Dict[str, float]] = {}
    summary_bg: Dict[str, Dict[str, float]] = {}
    for b in bands:
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

    stats_path = out_dir / f"{galaxy_prefix}_compute_cog_stats.json"
    stats_path.write_text(json.dumps({
        "galaxy_prefix": galaxy_prefix,
        "n_isophotes": len(rows),
        "shape": list(cutout.shape),
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
    for b in bands:
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
        "--galaxy-dir",
        default=Path("/Volumes/galaxy/isophote/sga2020/data/demo/PGC006669"),
        type=Path,
    )
    parser.add_argument("--galaxy-prefix", default="PGC006669-largegalaxy")
    parser.add_argument(
        "--out-dir",
        default=Path("outputs/benchmark_multiband/compute_cog_pgc"),
        type=Path,
    )
    args = parser.parse_args()
    run(args.galaxy_dir, args.galaxy_prefix, args.out_dir)


if __name__ == "__main__":
    main()
