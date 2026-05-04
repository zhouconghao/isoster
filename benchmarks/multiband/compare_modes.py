"""Compare multi-band joint fit vs three independent single-band fits.

Pipeline:

1. Load LegacySurvey grz cutout + COMBINED mask + per-band variance.
2. Read the existing multi-band Schema-1 result FITS.
3. Run ``isoster.fit_image`` on each band independently with knobs that
   match the multi-band config (same sma0/minsma/maxsma/astep/integrator).
4. Build per-band residuals for both modes (multi-band uses shared geometry
   + per-band ``intens_<b>``; single-band uses that band's own results).
5. Compute relative residuals stratified by SMA region (inner/mid/outer).
6. Render a single composite QA PNG and save a JSON sidecar.

Usage:
    uv run python benchmark_multiband/compare_modes.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from isoster import IsosterConfig, fit_image
from isoster.model import build_isoster_model
from isoster.multiband import isophote_results_mb_from_fits
from isoster.utils import isophote_results_to_fits

from legacysurvey_loader import (
    LEGACYSURVEY_ZP,
    asinh_softening_from_log10_match,
    load_legacysurvey_grz,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


_BAND_COLORS = {"g": "#1f77b4", "r": "#2ca02c", "z": "#d62728"}


# --------------------------------------------------------------------------- #
# Single-band fitting                                                         #
# --------------------------------------------------------------------------- #


def run_single_band(
    image: np.ndarray,
    variance: np.ndarray,
    mask: np.ndarray,
    *,
    sma0: float,
    minsma: float,
    maxsma: float,
    astep: float,
    integrator: str,
) -> dict:
    config = IsosterConfig(
        sma0=sma0,
        minsma=minsma,
        maxsma=maxsma,
        astep=astep,
        linear_growth=False,
        integrator=integrator,
        debug=True,
    )
    return fit_image(image, mask=mask, variance_map=variance, config=config)


# --------------------------------------------------------------------------- #
# Per-band MB-style residual                                                  #
# --------------------------------------------------------------------------- #


def mb_per_band_isophotes(mb_isophotes: Sequence[dict], band: str) -> List[dict]:
    """Convert Schema-1 multi-band isophotes into single-band-shaped dicts.

    The shared geometry columns stay; intensity and harmonics are taken
    from the band suffix. Used to drive ``build_isoster_model`` per band.
    """
    out: List[dict] = []
    for iso in mb_isophotes:
        d = dict(iso)
        d["intens"] = float(iso.get(f"intens_{band}", 0.0))
        for col in ("a3", "b3", "a4", "b4"):
            v = iso.get(f"{col}_{band}")
            if v is not None:
                d[col] = float(v)
        out.append(d)
    return out


def build_residual(image: np.ndarray, isophotes: Sequence[dict]) -> np.ndarray:
    model = build_isoster_model(image.shape, list(isophotes))
    return image - model


# --------------------------------------------------------------------------- #
# Region-stratified residual statistics                                       #
# --------------------------------------------------------------------------- #


def _ellipse_radius_grid(
    shape: Tuple[int, int],
    x0: float,
    y0: float,
    eps: float,
    pa_rad: float,
) -> np.ndarray:
    h, w = shape
    yy, xx = np.indices((h, w), dtype=np.float64)
    dx = xx - x0
    dy = yy - y0
    cos_pa = np.cos(pa_rad)
    sin_pa = np.sin(pa_rad)
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    return np.sqrt(x_rot ** 2 + (y_rot / max(1.0 - eps, 1e-3)) ** 2)


def _representative_geometry(isophotes: Sequence[dict]) -> Tuple[float, float, float, float]:
    valid = [
        iso for iso in isophotes
        if iso.get("valid", True) and float(iso.get("sma", 0.0)) > 0.0
        and np.isfinite(iso.get("eps", np.nan))
    ]
    if not valid:
        raise ValueError("No valid isophotes to derive elliptical-radius grid")
    sma_arr = np.array([float(iso["sma"]) for iso in valid])
    target = np.percentile(sma_arr, 60)
    iso = min(valid, key=lambda r: abs(float(r["sma"]) - target))
    return (
        float(iso["x0"]),
        float(iso["y0"]),
        float(iso["eps"]),
        float(np.deg2rad(float(iso["pa"]))),
    )


def residual_region_stats(
    image: np.ndarray,
    residual: np.ndarray,
    mask: np.ndarray,
    isophotes: Sequence[dict],
    sma_max_used: float,
) -> Dict[str, Dict[str, float]]:
    """Stratify residuals by elliptical-radius region.

    Returns a dict with keys ``inner``, ``mid``, ``outer`` and entries
    ``median_abs_residual``, ``rms_residual``, ``relative_residual``,
    ``n_pix``.
    """
    x0, y0, eps, pa_rad = _representative_geometry(isophotes)
    er = _ellipse_radius_grid(image.shape, x0, y0, eps, pa_rad)

    valid = (~mask) & np.isfinite(image) & np.isfinite(residual)
    valid &= er <= sma_max_used
    bounds = {
        "inner": (0.0, sma_max_used * 0.30),
        "mid":   (sma_max_used * 0.30, sma_max_used * 0.70),
        "outer": (sma_max_used * 0.70, sma_max_used),
    }
    out: Dict[str, Dict[str, float]] = {}
    for region, (lo, hi) in bounds.items():
        sel = valid & (er > lo) & (er <= hi)
        if not sel.any():
            out[region] = dict(
                median_abs_residual=float("nan"),
                rms_residual=float("nan"),
                relative_residual=float("nan"),
                n_pix=0,
            )
            continue
        r = residual[sel]
        m = image[sel]
        median_abs_r = float(np.median(np.abs(r)))
        rms = float(np.sqrt(np.mean(r ** 2)))
        ref_scale = float(np.median(np.abs(m))) if np.any(np.abs(m) > 0) else 0.0
        relative = median_abs_r / ref_scale if ref_scale > 0 else float("nan")
        out[region] = dict(
            median_abs_residual=median_abs_r,
            rms_residual=rms,
            relative_residual=relative,
            n_pix=int(sel.sum()),
        )
    return out


# --------------------------------------------------------------------------- #
# Plotting helpers                                                            #
# --------------------------------------------------------------------------- #


def _xaxis_arcsec_pow(sma: np.ndarray, pixel_scale: float) -> np.ndarray:
    """SMA**0.25 in arcsec, matching the multi-band plotter's axis."""
    return np.where(sma > 0, (sma * pixel_scale) ** 0.25, np.nan)


def _mu_per_arcsec2(intens: np.ndarray, zp: float, pixel_scale: float, soft: float) -> np.ndarray:
    pixarea = pixel_scale ** 2
    safe_intens = np.where(intens > soft, intens / pixarea, soft / pixarea)
    return -2.5 * np.log10(safe_intens) + zp


def _normalize_harmonic(coeffs: np.ndarray, intens: np.ndarray, sma: np.ndarray) -> np.ndarray:
    """Bender normalization A_n_norm = -A_n / (a · dI/da). Falls back to np.gradient(intens)."""
    out = np.full_like(coeffs, np.nan, dtype=np.float64)
    grad = np.gradient(intens, sma)
    denom = sma * grad
    nonzero = np.abs(denom) > 0
    out[nonzero] = -coeffs[nonzero] / denom[nonzero]
    return out


def _plot_isophote_overlay(
    ax, isophotes: Sequence[dict], n_rings: int, color: str
) -> None:
    valid = [
        iso for iso in isophotes
        if iso.get("valid", True) and float(iso.get("sma", 0.0)) > 0.0
        and np.isfinite(iso.get("eps", np.nan))
    ]
    if not valid:
        return
    valid_sorted = sorted(valid, key=lambda r: float(r["sma"]))
    if len(valid_sorted) <= n_rings:
        sample = valid_sorted
    else:
        idx = np.round(np.linspace(0, len(valid_sorted) - 1, n_rings)).astype(int)
        sample = [valid_sorted[i] for i in idx]
    for iso in sample:
        sma = float(iso["sma"])
        eps = float(iso["eps"])
        pa_deg = float(iso["pa"])
        ellipse = Ellipse(
            (float(iso["x0"]), float(iso["y0"])),
            width=2.0 * sma,
            height=2.0 * sma * (1.0 - eps),
            angle=pa_deg,
            facecolor="none",
            edgecolor=color,
            lw=0.7,
            alpha=0.85,
        )
        ax.add_patch(ellipse)


def render_comparison_qa(
    cutout,
    mb_result: dict,
    sb_results: Dict[str, dict],
    out_path: Path,
    n_overlay_rings: int = 6,
) -> dict:
    """Build the multi vs single comparison figure and return residual stats."""

    bands = cutout.bands
    pix = cutout.pixel_scale_arcsec
    zp = cutout.zp
    softening_per_band = {
        b: asinh_softening_from_log10_match(pix, zp, bright_mu=22.0) for b in bands
    }

    mb_isos = list(mb_result["isophotes"])

    fig = plt.figure(figsize=(20.0, 14.0), constrained_layout=False)
    outer = fig.add_gridspec(
        2, 2,
        hspace=0.20, wspace=0.10,
        left=0.06, right=0.985, top=0.94, bottom=0.06,
    )

    # ---- Top-left: SB profile ------------------------------------------- #
    ax_sb = fig.add_subplot(outer[0, 0])
    for b in bands:
        color = _BAND_COLORS.get(b, "k")
        soft = softening_per_band[b]
        sma_mb = np.array([float(iso["sma"]) for iso in mb_isos])
        i_mb = np.array([float(iso.get(f"intens_{b}", np.nan)) for iso in mb_isos])
        valid_mb = np.array([bool(iso.get("valid", True)) for iso in mb_isos])
        x_mb = _xaxis_arcsec_pow(sma_mb, pix)
        mu_mb = _mu_per_arcsec2(i_mb, zp, pix, soft)
        sel = valid_mb & np.isfinite(mu_mb)
        ax_sb.plot(x_mb[sel], mu_mb[sel], "o", ms=5.0, color=color,
                   label=f"{b} (multi-band)")

        sb_isos = sb_results[b]["isophotes"]
        sma_sb = np.array([float(iso["sma"]) for iso in sb_isos])
        i_sb = np.array([float(iso.get("intens", np.nan)) for iso in sb_isos])
        valid_sb = np.array([bool(iso.get("valid", True)) for iso in sb_isos])
        x_sb = _xaxis_arcsec_pow(sma_sb, pix)
        mu_sb = _mu_per_arcsec2(i_sb, zp, pix, soft)
        sel_sb = valid_sb & np.isfinite(mu_sb)
        ax_sb.plot(x_sb[sel_sb], mu_sb[sel_sb], "o", ms=6.5, mfc="none",
                   mec=color, mew=1.0, label=f"{b} (single-band)")

    ax_sb.invert_yaxis()
    ax_sb.set_xlabel(r"$\mathrm{SMA}^{0.25}$  [arcsec$^{0.25}$]", fontsize=12)
    ax_sb.set_ylabel(r"$\mu$  [mag/arcsec$^2$]", fontsize=12)
    ax_sb.set_title("Surface brightness — filled = MB joint, open = SB independent")
    ax_sb.grid(alpha=0.25, lw=0.5)
    ax_sb.legend(fontsize=8, ncol=3, loc="upper right")

    # ---- Top-right: 2x3 residual mosaic --------------------------------- #
    gs_mosaic = outer[0, 1].subgridspec(2, len(bands), hspace=0.06, wspace=0.04)
    residuals_mb: Dict[str, np.ndarray] = {}
    residuals_sb: Dict[str, np.ndarray] = {}
    sma_max_used: Dict[str, float] = {}

    for col, b in enumerate(bands):
        image = cutout.images[col]
        mb_iso_b = mb_per_band_isophotes(mb_isos, b)
        residuals_mb[b] = build_residual(image, mb_iso_b)
        residuals_sb[b] = build_residual(image, sb_results[b]["isophotes"])
        sma_max_used[b] = float(np.nanmax([
            float(iso["sma"]) for iso in mb_iso_b
            if iso.get("valid", True) and float(iso.get("sma", 0.0)) > 0.0
        ] or [10.0]))

    for col, b in enumerate(bands):
        image = cutout.images[col]
        finite = np.isfinite(image) & ~cutout.combined_mask
        clim = float(np.percentile(np.abs(np.concatenate([
            residuals_mb[b][finite].ravel(),
            residuals_sb[b][finite].ravel(),
        ])), 99.0))
        clim = max(clim, 1e-6)

        ax_mb = fig.add_subplot(gs_mosaic[0, col])
        ax_mb.imshow(residuals_mb[b], origin="lower", cmap="RdBu_r",
                     vmin=-clim, vmax=+clim)
        _plot_isophote_overlay(ax_mb, mb_isos, n_overlay_rings, color="k")
        ax_mb.set_title(f"residual {b} (multi-band)", color=_BAND_COLORS.get(b, "k"),
                        fontsize=10)
        ax_mb.set_xticks([])
        ax_mb.set_yticks([])

        ax_sb_resid = fig.add_subplot(gs_mosaic[1, col])
        ax_sb_resid.imshow(residuals_sb[b], origin="lower", cmap="RdBu_r",
                           vmin=-clim, vmax=+clim)
        _plot_isophote_overlay(ax_sb_resid, sb_results[b]["isophotes"], n_overlay_rings,
                               color="k")
        ax_sb_resid.set_title(f"residual {b} (single-band)", color=_BAND_COLORS.get(b, "k"),
                              fontsize=10)
        ax_sb_resid.set_xticks([])
        ax_sb_resid.set_yticks([])

    # ---- Bottom-left: harmonics ----------------------------------------- #
    gs_harm = outer[1, 0].subgridspec(4, 1, hspace=0.05)
    harm_titles = ["A3_norm", "B3_norm", "A4_norm", "B4_norm"]
    for row, key in enumerate(["a3", "b3", "a4", "b4"]):
        ax = fig.add_subplot(gs_harm[row, 0])
        for b in bands:
            color = _BAND_COLORS.get(b, "k")
            sma_mb = np.array([float(iso["sma"]) for iso in mb_isos])
            i_mb = np.array([float(iso.get(f"intens_{b}", np.nan)) for iso in mb_isos])
            v_mb = np.array([float(iso.get(f"{key}_{b}", np.nan)) for iso in mb_isos])
            norm_mb = _normalize_harmonic(v_mb, i_mb, sma_mb)
            x_mb = _xaxis_arcsec_pow(sma_mb, pix)
            sel = np.array([bool(iso.get("valid", True)) for iso in mb_isos]) & np.isfinite(norm_mb)
            ax.plot(x_mb[sel], norm_mb[sel], "o", ms=4.0, color=color)

            sb_isos = sb_results[b]["isophotes"]
            sma_sb = np.array([float(iso["sma"]) for iso in sb_isos])
            i_sb = np.array([float(iso.get("intens", np.nan)) for iso in sb_isos])
            v_sb = np.array([float(iso.get(key, np.nan)) for iso in sb_isos])
            norm_sb = _normalize_harmonic(v_sb, i_sb, sma_sb)
            x_sb = _xaxis_arcsec_pow(sma_sb, pix)
            sel_sb = np.array([bool(iso.get("valid", True)) for iso in sb_isos]) & np.isfinite(norm_sb)
            ax.plot(x_sb[sel_sb], norm_sb[sel_sb], "o", ms=5.5, mfc="none",
                    mec=color, mew=0.8)

        ax.axhline(0.0, color="0.3", lw=0.6, ls=":")
        ax.set_ylabel(harm_titles[row], fontsize=9)
        if row < 3:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r"$\mathrm{SMA}^{0.25}$  [arcsec$^{0.25}$]", fontsize=10)
        ax.grid(alpha=0.2, lw=0.4)

    # ---- Bottom-right: shared geometry ---------------------------------- #
    gs_geom = outer[1, 1].subgridspec(4, 1, hspace=0.05)
    geom_titles = ["eps", "PA [deg]", "x0 [pix]", "y0 [pix]"]
    for row, key in enumerate(["eps", "pa", "x0", "y0"]):
        ax = fig.add_subplot(gs_geom[row, 0])
        sma_mb = np.array([float(iso["sma"]) for iso in mb_isos])
        v_mb = np.array([float(iso.get(key, np.nan)) for iso in mb_isos])
        x_mb = _xaxis_arcsec_pow(sma_mb, pix)
        valid_mb = np.array([bool(iso.get("valid", True)) for iso in mb_isos]) & np.isfinite(v_mb)
        ax.plot(x_mb[valid_mb], v_mb[valid_mb], "o", ms=5.0, color="k",
                label="multi-band shared")

        for b in bands:
            color = _BAND_COLORS.get(b, "k")
            sb_isos = sb_results[b]["isophotes"]
            sma_sb = np.array([float(iso["sma"]) for iso in sb_isos])
            v_sb = np.array([float(iso.get(key, np.nan)) for iso in sb_isos])
            x_sb = _xaxis_arcsec_pow(sma_sb, pix)
            valid_sb = np.array([bool(iso.get("valid", True)) for iso in sb_isos]) & np.isfinite(v_sb)
            ax.plot(x_sb[valid_sb], v_sb[valid_sb], "o", ms=6.0, mfc="none",
                    mec=color, mew=0.9, label=f"{b} single-band" if row == 0 else None)

        ax.set_ylabel(geom_titles[row], fontsize=9)
        if row < 3:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r"$\mathrm{SMA}^{0.25}$  [arcsec$^{0.25}$]", fontsize=10)
        if row == 0:
            ax.legend(fontsize=8, ncol=4, loc="upper left")
        ax.grid(alpha=0.2, lw=0.4)

    fig.suptitle("PGC006669  —  multi-band joint vs single-band independent",
                 fontsize=14, y=0.995)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- Region-stratified statistics ----------------------------------- #
    stats: dict = {}
    for b in bands:
        stats[b] = {}
        stats[b]["multi_band"] = residual_region_stats(
            cutout.images[bands.index(b)],
            residuals_mb[b],
            cutout.combined_mask,
            mb_isos,
            sma_max_used[b],
        )
        stats[b]["single_band"] = residual_region_stats(
            cutout.images[bands.index(b)],
            residuals_sb[b],
            cutout.combined_mask,
            sb_results[b]["isophotes"],
            sma_max_used[b],
        )
    return stats


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


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
        default=Path("outputs/benchmark_multiband/PGC006669"),
        type=Path,
    )
    parser.add_argument("--sma0", type=float, default=10.0)
    parser.add_argument("--minsma", type=float, default=1.0)
    parser.add_argument("--maxsma", type=float, default=None)
    parser.add_argument("--astep", type=float, default=0.10)
    parser.add_argument("--integrator", default="median")
    args = parser.parse_args()

    cutout = load_legacysurvey_grz(args.galaxy_dir, args.galaxy_prefix)
    bands = cutout.bands
    if args.maxsma is None:
        args.maxsma = float(min(cutout.shape) * 0.45)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    mb_fits_path = args.out_dir / f"{args.galaxy_prefix}_multiband_isophotes.fits"
    if not mb_fits_path.exists():
        raise FileNotFoundError(
            f"Expected multi-band result at {mb_fits_path}. "
            f"Run run_isoster_multiband_pgc.py first."
        )
    mb_result = isophote_results_mb_from_fits(str(mb_fits_path))

    sb_results: Dict[str, dict] = {}
    for col, b in enumerate(bands):
        print(f"Running single-band fit on band {b} ...")
        sb_results[b] = run_single_band(
            cutout.images[col],
            cutout.variances[col],
            cutout.combined_mask,
            sma0=args.sma0,
            minsma=args.minsma,
            maxsma=args.maxsma,
            astep=args.astep,
            integrator=args.integrator,
        )
        sb_path = args.out_dir / f"{args.galaxy_prefix}_singleband_{b}.fits"
        isophote_results_to_fits(sb_results[b], str(sb_path))
        print(f"  -> {sb_path}")

    qa_path = args.out_dir / f"{args.galaxy_prefix}_compare_mb_vs_sb.png"
    stats = render_comparison_qa(cutout, mb_result, sb_results, qa_path)
    print(f"Wrote {qa_path}")

    stats_path = args.out_dir / f"{args.galaxy_prefix}_compare_mb_vs_sb_stats.json"
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2, default=float)
    print(f"Wrote {stats_path}")

    print("\nRelative residual (median |residual| / median |image|) per region:")
    print(f"{'band':>5} {'mode':>14} {'inner':>10} {'mid':>10} {'outer':>10}")
    for b in bands:
        for mode in ("multi_band", "single_band"):
            row = stats[b][mode]
            print(
                f"{b:>5} {mode:>14} "
                f"{row['inner']['relative_residual']:>10.4f} "
                f"{row['mid']['relative_residual']:>10.4f} "
                f"{row['outer']['relative_residual']:>10.4f}"
            )


if __name__ == "__main__":
    main()
