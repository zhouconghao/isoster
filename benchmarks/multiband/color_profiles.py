"""g-r and r-z color profiles: multi-band joint vs single-band forced.

For the SB / "forced-photometry" version we re-use the existing r-band
free fit as the geometry template and run ``isoster.fit_image`` in
forced mode on the g and z images (geometry locked to r, intensities
re-measured).

For the MB version we read ``intens_<b>`` and ``intens_err_<b>`` straight
from the joint Schema-1 result.

Both modes' color profiles are saved as JSON sidecars and a single
two-panel PNG.  This script is independent of ``compare_modes.py`` —
it does not touch or extend that figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from isoster import IsosterConfig, fit_image
from isoster.multiband import isophote_results_mb_from_fits
from isoster.utils import isophote_results_from_fits, isophote_results_to_fits

from legacysurvey_loader import load_legacysurvey_grz


_BAND_COLORS = {"g": "#1f77b4", "r": "#2ca02c", "z": "#d62728"}


# --------------------------------------------------------------------------- #
# Forced-photometry runs                                                      #
# --------------------------------------------------------------------------- #


def run_forced(
    image: np.ndarray,
    variance: np.ndarray,
    mask: np.ndarray,
    template_result: dict,
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
    return fit_image(
        image,
        mask=mask,
        variance_map=variance,
        config=config,
        template=template_result,
    )


# --------------------------------------------------------------------------- #
# Intensity → magnitude / color helpers                                       #
# --------------------------------------------------------------------------- #


def _flux_to_mag(intens: np.ndarray, zp: float, pixel_scale: float) -> np.ndarray:
    """Convert intensity (per pixel) to mag/arcsec^2 using μ = -2.5·log10(I/pixarea) + zp."""
    pixarea = pixel_scale ** 2
    safe = np.where(intens > 0, intens / pixarea, np.nan)
    return -2.5 * np.log10(safe) + zp


def _color(intens_a: np.ndarray, intens_b: np.ndarray, zp: float, pixel_scale: float) -> np.ndarray:
    """Color = μ_a - μ_b. The pixel-scale and ZP cancel; included for clarity."""
    return _flux_to_mag(intens_a, zp, pixel_scale) - _flux_to_mag(intens_b, zp, pixel_scale)


def _color_err(intens_a: np.ndarray, err_a: np.ndarray,
               intens_b: np.ndarray, err_b: np.ndarray) -> np.ndarray:
    """Magnitude error of the color, log10 form: σ = (2.5/ln10)·√((σ_a/I_a)² + (σ_b/I_b)²)."""
    coef = 2.5 / np.log(10.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_a = np.where(intens_a > 0, err_a / intens_a, np.nan)
        rel_b = np.where(intens_b > 0, err_b / intens_b, np.nan)
        return coef * np.sqrt(rel_a ** 2 + rel_b ** 2)


# --------------------------------------------------------------------------- #
# Profile builders                                                            #
# --------------------------------------------------------------------------- #


def mb_color_profile(mb_result: dict, band_a: str, band_b: str,
                     zp: float, pixel_scale: float) -> Dict[str, np.ndarray]:
    isos = list(mb_result["isophotes"])
    sma = np.array([float(iso["sma"]) for iso in isos])
    valid = np.array([bool(iso.get("valid", True)) for iso in isos])
    i_a = np.array([float(iso.get(f"intens_{band_a}", np.nan)) for iso in isos])
    i_b = np.array([float(iso.get(f"intens_{band_b}", np.nan)) for iso in isos])
    e_a = np.array([float(iso.get(f"intens_err_{band_a}", np.nan)) for iso in isos])
    e_b = np.array([float(iso.get(f"intens_err_{band_b}", np.nan)) for iso in isos])
    color = _color(i_a, i_b, zp, pixel_scale)
    err = _color_err(i_a, e_a, i_b, e_b)
    return {"sma": sma, "color": color, "color_err": err, "valid": valid}


def sb_color_profile(result_a: dict, result_b: dict,
                     zp: float, pixel_scale: float) -> Dict[str, np.ndarray]:
    """Single-band color profile: forced runs share the r-band geometry, so
    isophotes are aligned by index and SMA."""
    isos_a = list(result_a["isophotes"])
    isos_b = list(result_b["isophotes"])
    n = min(len(isos_a), len(isos_b))
    sma_a = np.array([float(isos_a[i]["sma"]) for i in range(n)])
    sma_b = np.array([float(isos_b[i]["sma"]) for i in range(n)])
    if not np.allclose(sma_a, sma_b, rtol=1e-6, atol=1e-6):
        raise ValueError(
            "SB forced color profile expected matching SMAs across template and target; "
            "got mismatch — check that the forced-photometry template was honored."
        )
    valid = np.array([
        bool(isos_a[i].get("valid", True)) and bool(isos_b[i].get("valid", True))
        for i in range(n)
    ])
    i_a = np.array([float(isos_a[i].get("intens", np.nan)) for i in range(n)])
    i_b = np.array([float(isos_b[i].get("intens", np.nan)) for i in range(n)])
    e_a = np.array([float(isos_a[i].get("intens_err", np.nan)) for i in range(n)])
    e_b = np.array([float(isos_b[i].get("intens_err", np.nan)) for i in range(n)])
    color = _color(i_a, i_b, zp, pixel_scale)
    err = _color_err(i_a, e_a, i_b, e_b)
    return {"sma": sma_a, "color": color, "color_err": err, "valid": valid}


# --------------------------------------------------------------------------- #
# Plotting                                                                    #
# --------------------------------------------------------------------------- #


def _xaxis_arcsec_pow(sma: np.ndarray, pixel_scale: float) -> np.ndarray:
    return np.where(sma > 0, (sma * pixel_scale) ** 0.25, np.nan)


def _bulk_ylim(values_a: np.ndarray, values_b: np.ndarray, pad: float = 0.4) -> Tuple[float, float]:
    """Tight y-limit driven by the central 95% of valid color points.

    The LSB tail can produce spurious extreme colors (intensity flips
    sign or hits the softening floor); clipping to the 2.5–97.5
    percentile range keeps the bulk of the profile readable.
    """
    finite = np.concatenate([
        values_a[np.isfinite(values_a)],
        values_b[np.isfinite(values_b)],
    ])
    if finite.size == 0:
        return (-1.0, 2.0)
    lo, hi = np.percentile(finite, [2.5, 97.5])
    span = max(hi - lo, 0.5)
    return (float(lo - pad * span), float(hi + pad * span))


def render_color_qa(
    profiles: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    pixel_scale: float,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.5), sharex=True)

    panel_specs = [("g_minus_r", r"$g - r$  [mag/arcsec$^2$]"),
                   ("r_minus_z", r"$r - z$  [mag/arcsec$^2$]")]

    for ax, (key, ylabel) in zip(axes, panel_specs):
        ylim_inputs: List[np.ndarray] = []
        for mode_label, mode_key, fill_style in [
            ("multi-band joint", "mb", "filled"),
            ("single-band forced", "sb", "open"),
        ]:
            prof = profiles[mode_key][key]
            sel = prof["valid"] & np.isfinite(prof["color"]) & np.isfinite(prof["color_err"])
            x = _xaxis_arcsec_pow(prof["sma"], pixel_scale)
            color = "#222222" if mode_key == "mb" else "#888888"
            mfc = color if fill_style == "filled" else "none"
            ms = 5.0 if fill_style == "filled" else 6.5
            ax.errorbar(
                x[sel],
                prof["color"][sel],
                yerr=prof["color_err"][sel],
                ls="none",
                marker="o",
                mfc=mfc,
                mec=color,
                ms=ms,
                mew=1.0,
                ecolor=color,
                elinewidth=0.7,
                capsize=2.0,
                label=mode_label,
            )
            ylim_inputs.append(prof["color"][sel])
        if len(ylim_inputs) == 2:
            ax.set_ylim(*_bulk_ylim(ylim_inputs[0], ylim_inputs[1]))
        ax.set_xlabel(r"$\mathrm{SMA}^{0.25}$  [arcsec$^{0.25}$]", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.axhline(0.0, color="0.5", lw=0.6, ls=":")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=9, loc="best")

    axes[0].set_title("g − r color profile")
    axes[1].set_title("r − z color profile")
    fig.suptitle("PGC006669 — color profiles, multi-band joint vs single-band forced",
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


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
    parser.add_argument(
        "--mb-suffix",
        default="",
        help="Suffix tagged onto the multi-band FITS filename (e.g. '_fixbg0'). "
        "When set, the output PNG/JSON also pick up the same suffix so the "
        "default-mode artifacts are not overwritten.",
    )
    args = parser.parse_args()

    cutout = load_legacysurvey_grz(args.galaxy_dir, args.galaxy_prefix)
    bands = cutout.bands
    if args.maxsma is None:
        args.maxsma = float(min(cutout.shape) * 0.45)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    mb_path = (
        args.out_dir / f"{args.galaxy_prefix}_multiband{args.mb_suffix}_isophotes.fits"
    )
    if not mb_path.exists():
        raise FileNotFoundError(f"Missing multi-band result at {mb_path}")
    mb_result = isophote_results_mb_from_fits(str(mb_path))

    sb_r_path = args.out_dir / f"{args.galaxy_prefix}_singleband_r.fits"
    if not sb_r_path.exists():
        raise FileNotFoundError(f"Missing r-band single-band result at {sb_r_path}. "
                                "Run compare_modes.py first.")
    sb_r_result = isophote_results_from_fits(str(sb_r_path))

    band_idx = {b: i for i, b in enumerate(bands)}
    forced_results: Dict[str, dict] = {"r": sb_r_result}
    for b in ("g", "z"):
        print(f"Running forced single-band fit on {b} (template = r) ...")
        forced = run_forced(
            cutout.images[band_idx[b]],
            cutout.variances[band_idx[b]],
            cutout.combined_mask,
            sb_r_result,
            sma0=args.sma0,
            minsma=args.minsma,
            maxsma=args.maxsma,
            astep=args.astep,
            integrator=args.integrator,
        )
        forced_results[b] = forced
        out_path = args.out_dir / f"{args.galaxy_prefix}_forced_{b}_on_r.fits"
        isophote_results_to_fits(forced, str(out_path))
        print(f"  -> {out_path}")

    profiles = {
        "mb": {
            "g_minus_r": mb_color_profile(mb_result, "g", "r", cutout.zp, cutout.pixel_scale_arcsec),
            "r_minus_z": mb_color_profile(mb_result, "r", "z", cutout.zp, cutout.pixel_scale_arcsec),
        },
        "sb": {
            "g_minus_r": sb_color_profile(forced_results["g"], forced_results["r"],
                                          cutout.zp, cutout.pixel_scale_arcsec),
            "r_minus_z": sb_color_profile(forced_results["r"], forced_results["z"],
                                          cutout.zp, cutout.pixel_scale_arcsec),
        },
    }

    qa_path = (
        args.out_dir / f"{args.galaxy_prefix}_color_profiles{args.mb_suffix}.png"
    )
    render_color_qa(profiles, cutout.pixel_scale_arcsec, qa_path)
    print(f"Wrote {qa_path}")

    serialisable = {
        mode: {
            color: {k: np.asarray(v).tolist() for k, v in prof.items()}
            for color, prof in by_color.items()
        }
        for mode, by_color in profiles.items()
    }
    json_path = (
        args.out_dir / f"{args.galaxy_prefix}_color_profiles{args.mb_suffix}.json"
    )
    with open(json_path, "w") as fh:
        json.dump(serialisable, fh, indent=2, default=float)
    print(f"Wrote {json_path}")

    print("\nMedian color (valid rings only):")
    print(f"{'mode':>22} {'g-r':>10} {'r-z':>10}")
    for label, mode_key in [("multi-band joint", "mb"), ("single-band forced", "sb")]:
        gr = profiles[mode_key]["g_minus_r"]
        rz = profiles[mode_key]["r_minus_z"]
        gr_vals = gr["color"][gr["valid"] & np.isfinite(gr["color"])]
        rz_vals = rz["color"][rz["valid"] & np.isfinite(rz["color"])]
        gr_med = float(np.median(gr_vals)) if gr_vals.size else float("nan")
        rz_med = float(np.median(rz_vals)) if rz_vals.size else float("nan")
        print(f"{label:>22} {gr_med:>10.3f} {rz_med:>10.3f}")


if __name__ == "__main__":
    main()
