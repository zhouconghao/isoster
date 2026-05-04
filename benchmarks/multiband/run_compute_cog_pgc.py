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
    result: dict, bands: List[str], out_path: Path, title: str,
) -> None:
    rows = result["isophotes"]
    sma = np.array([float(r["sma"]) for r in rows])
    fig, (ax_cog, ax_ann) = plt.subplots(
        2, 1, figsize=(7.0, 6.0), sharex=True,
    )
    for b in bands:
        cog = np.array([float(r.get(f"cog_{b}", float("nan"))) for r in rows])
        ann = np.array([float(r.get(f"cog_annulus_{b}", float("nan"))) for r in rows])
        ax_cog.plot(sma, cog, color=GRZ_COLORS.get(b, None), lw=1.4, label=b)
        ax_ann.plot(sma, ann, color=GRZ_COLORS.get(b, None), lw=1.0)
    ax_cog.set_ylabel("Cumulative flux (cog)")
    ax_cog.legend(loc="lower right", frameon=False)
    ax_cog.grid(True, alpha=0.3)
    ax_ann.set_ylabel("Annular flux (cog_annulus)")
    ax_ann.set_xlabel("SMA (px)")
    ax_ann.grid(True, alpha=0.3)
    ax_ann.axhline(0.0, color="0.6", lw=0.5, zorder=-1)
    fig.suptitle(title, fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
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
        res, bands, curves_path,
        title=f"PGC006669 — per-band CoG (cumulative flux + annular flux)",
    )
    print(f"  wrote {curves_path}")

    rows = res["isophotes"]
    sma_arr = np.array([float(r["sma"]) for r in rows])
    summary: Dict[str, Dict[str, float]] = {}
    for b in bands:
        cog = np.array([float(r.get(f"cog_{b}", float("nan"))) for r in rows])
        finite = np.isfinite(cog)
        cog_total = float(cog[finite][-1]) if int(finite.sum()) > 0 else float("nan")
        summary[b] = {
            "cog_total_outermost": cog_total,
            "half_light_sma_px": _half_light_sma(sma_arr, cog),
        }

    stats_path = out_dir / f"{galaxy_prefix}_compute_cog_stats.json"
    stats_path.write_text(json.dumps({
        "galaxy_prefix": galaxy_prefix,
        "n_isophotes": len(rows),
        "shape": list(cutout.shape),
        "per_band_summary": summary,
    }, indent=2))
    print(f"Wrote {stats_path}")

    print()
    print(f"{'band':<6} {'cog_total':>14} {'half-light SMA (px)':>20}")
    print("-" * 45)
    for b in bands:
        s = summary[b]
        print(
            f"{b:<6} {s['cog_total_outermost']:>14.4e} "
            f"{s['half_light_sma_px']:>20.2f}"
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
