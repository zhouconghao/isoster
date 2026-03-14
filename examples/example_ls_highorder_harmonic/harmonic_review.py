"""
Harmonic Visualization Review
=============================

Runs three EA-mode conditions on real galaxies to review the harmonic
overlay rendering:

1. **ea_no_harmonics** — ``compute_deviations=False``, no harmonic fitting
2. **ea_harmonics_34** — simultaneous [3, 4]
3. **ea_harmonics_34567** — simultaneous [3, 4, 5, 6, 7]

Generates:
- Per-condition extended QA figure
- 3-condition comparison QA
- Harmonic detail figure (for each harmonic condition)

Usage
-----
::

    uv run python examples/example_ls_highorder_harmonic/harmonic_review.py \\
        --galaxy ngc3610 --band-index 1

    uv run python examples/example_ls_highorder_harmonic/harmonic_review.py \\
        --galaxy eso243-49 --band-index 1
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

import isoster
from isoster.config import IsosterConfig
from isoster.model import build_isoster_model
from isoster.plotting import (
    configure_qa_plot_style,
    derive_arcsinh_parameters,
    draw_isophote_overlays,
    make_arcsinh_display_from_parameters,
    normalize_pa_degrees,
    plot_qa_summary_extended,
    set_x_limits_with_right_margin,
)

# Example-local modules
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from masking import make_object_mask  # noqa: E402
from shared import (  # noqa: E402
    BAND_NAMES,
    FITS_FILENAME,
    INITIAL_SMA,
    MASK_PARAMS,
    PIXEL_SCALE,
    SUPPORTED_GALAXIES,
    load_legacysurvey_fits,
)

# ---------------------------------------------------------------------------
# Data root
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).parent.parent.parent / "data"

# ---------------------------------------------------------------------------
# Review conditions (EA-mode only, 3 configs)
# ---------------------------------------------------------------------------

REVIEW_CONDITIONS = [
    "ea_no_harmonics",
    "ea_harmonics_34",
    "ea_harmonics_34567",
]

REVIEW_DISPLAY = {
    "ea_no_harmonics": "EA / no harmonics",
    "ea_harmonics_34": "EA / [3,4]",
    "ea_harmonics_34567": "EA / [3–7]",
}

REVIEW_COLORS = {
    "ea_no_harmonics": "#7f7f7f",
    "ea_harmonics_34": "#d62728",
    "ea_harmonics_34567": "#2ca02c",
}

REVIEW_MARKERS = {
    "ea_no_harmonics": "o",
    "ea_harmonics_34": "D",
    "ea_harmonics_34567": "P",
}


def make_review_configs(galaxy: str, sma0: float | None = None) -> dict[str, IsosterConfig]:
    """Build the three review IsosterConfig objects."""
    _sma0 = sma0 if sma0 is not None else INITIAL_SMA[galaxy]

    base_kwargs: dict[str, Any] = dict(
        sma0=_sma0,
        convergence_scaling="sector_area",
        geometry_damping=0.7,
        permissive_geometry=True,
        use_eccentric_anomaly=True,
    )

    configs = {
        "ea_no_harmonics": IsosterConfig(
            **base_kwargs,
            simultaneous_harmonics=False,
            compute_deviations=False,
        ),
        "ea_harmonics_34": IsosterConfig(
            **base_kwargs,
            simultaneous_harmonics=True,
            harmonic_orders=[3, 4],
        ),
        "ea_harmonics_34567": IsosterConfig(
            **base_kwargs,
            simultaneous_harmonics=True,
            harmonic_orders=[3, 4, 5, 6, 7],
        ),
    }
    return configs


# ---------------------------------------------------------------------------
# Harmonic detail figure
# ---------------------------------------------------------------------------

# Per-order colors for 1-D harmonic panels
_ORDER_COLORS = {
    3: "#1f77b4",
    4: "#ff7f0e",
    5: "#2ca02c",
    6: "#d62728",
    7: "#9467bd",
}


def plot_harmonic_detail(
    image: np.ndarray,
    mask: np.ndarray | None,
    condition_results: dict[str, list[dict[str, Any]]],
    condition_models: dict[str, np.ndarray],
    galaxy: str,
    filename: str,
) -> None:
    """Harmonic detail figure for two harmonic conditions side-by-side.

    Layout: 2 columns (one per harmonic condition) x 4 rows.

    Row 1 (tall): Image with harmonic isophote overlays.
    Row 2 (tall): Residual (data - model) with same overlays.
    Row 3: 1-D odd harmonic coefficients vs SMA^0.25.
    Row 4: 1-D even harmonic coefficients vs SMA^0.25.

    Parameters
    ----------
    image : 2D array
        Science image.
    mask : 2D bool array or None
        Bad-pixel mask.
    condition_results : dict
        Mapping condition_label -> list of isophote dicts. Should contain
        the two harmonic conditions (ea_harmonics_34, ea_harmonics_34567).
    condition_models : dict
        Mapping condition_label -> 2D model array.
    galaxy : str
        Galaxy name for title.
    filename : str
        Output path.
    """
    configure_qa_plot_style()

    harmonic_conditions = [
        c for c in ["ea_harmonics_34", "ea_harmonics_34567"]
        if c in condition_results
    ]
    n_cols = len(harmonic_conditions)
    if n_cols == 0:
        print("  No harmonic conditions to plot; skipping detail figure.")
        return

    # Arcsinh display parameters (shared across panels)
    low, high, scale, vmax = derive_arcsinh_parameters(image)
    display, vmin_disp, vmax_disp = make_arcsinh_display_from_parameters(
        image, low=low, high=high, scale=scale, vmax=vmax,
    )

    # Shared residual color scale
    all_abs_res = []
    for cond in harmonic_conditions:
        res_map = image - condition_models[cond]
        finite = res_map[np.isfinite(res_map)]
        if finite.size > 0:
            all_abs_res.append(np.abs(finite))
    if all_abs_res:
        res_limit = float(np.nanpercentile(np.concatenate(all_abs_res), 99.0))
        res_limit = max(res_limit, 0.05)
    else:
        res_limit = 1.0

    # Figure
    fig = plt.figure(figsize=(6.0 * n_cols, 16.0))
    outer = gridspec.GridSpec(
        4, n_cols, figure=fig,
        height_ratios=[3.0, 3.0, 1.2, 1.2],
        hspace=0.18, wspace=0.25,
    )

    fig.suptitle(
        f"{galaxy.upper()} — harmonic detail",
        fontsize=15, y=0.995,
    )

    for col_idx, cond in enumerate(harmonic_conditions):
        isophotes = condition_results[cond]
        model = condition_models[cond]
        residual = image - model

        # Row 1: Image + harmonic overlays
        ax_img = fig.add_subplot(outer[0, col_idx])
        ax_img.imshow(
            display, origin="lower", cmap="viridis",
            vmin=vmin_disp, vmax=vmax_disp, interpolation="nearest",
        )
        draw_isophote_overlays(
            ax_img, isophotes, step=5,
            line_width=0.8, alpha=0.8, edge_color="white",
            draw_harmonics=True,
        )
        ax_img.set_title(f"{REVIEW_DISPLAY[cond]} — image + overlays", fontsize=10)
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        # Row 2: Residual + harmonic overlays
        ax_res = fig.add_subplot(outer[1, col_idx])
        ax_res.imshow(
            residual, origin="lower", cmap="coolwarm",
            vmin=-res_limit, vmax=res_limit, interpolation="nearest",
        )
        draw_isophote_overlays(
            ax_res, isophotes, step=5,
            line_width=0.8, alpha=0.6, edge_color="black",
            draw_harmonics=True,
        )
        ax_res.set_title(f"{REVIEW_DISPLAY[cond]} — residual", fontsize=10)
        ax_res.set_xticks([])
        ax_res.set_yticks([])

        # Prepare profile arrays
        sma = np.array([r.get("sma", np.nan) for r in isophotes])
        xax = sma ** 0.25
        valid = np.isfinite(xax) & (sma > 1.0)

        # Detect available harmonic orders
        available_orders = set()
        for r in isophotes:
            for order in range(3, 8):
                if np.isfinite(r.get(f"a{order}", np.nan)):
                    available_orders.add(order)
        odd_orders = sorted(o for o in available_orders if o % 2 == 1)
        even_orders = sorted(o for o in available_orders if o % 2 == 0)

        # Row 3: Odd harmonics
        ax_odd = fig.add_subplot(outer[2, col_idx])
        odd_handles = []
        for order in odd_orders:
            color = _ORDER_COLORS.get(order, "black")
            an = np.array([r.get(f"a{order}", np.nan) for r in isophotes])
            bn = np.array([r.get(f"b{order}", np.nan) for r in isophotes])
            ok_a = valid & np.isfinite(an)
            ok_b = valid & np.isfinite(bn)
            if np.any(ok_a):
                ax_odd.scatter(
                    xax[ok_a], an[ok_a], s=14, marker="o",
                    facecolors=color, edgecolors=color, alpha=0.7,
                )
            if np.any(ok_b):
                ax_odd.scatter(
                    xax[ok_b], bn[ok_b], s=14, marker="o",
                    facecolors="none", edgecolors=color, alpha=0.7,
                )
            odd_handles.append(
                Line2D([], [], marker="o", linestyle="None",
                       color=color, markersize=5,
                       label=f"a{order} (filled), b{order} (open)")
            )

        ax_odd.axhline(0.0, color="gray", linestyle=":", linewidth=0.7, alpha=0.6)
        ax_odd.set_ylabel("odd harmonics")
        ax_odd.set_title(REVIEW_DISPLAY[cond], fontsize=9) if col_idx == 0 else None
        ax_odd.tick_params(labelbottom=False)
        ax_odd.grid(alpha=0.2)
        if odd_handles:
            ax_odd.legend(handles=odd_handles, fontsize=7, loc="upper right")

        # Row 4: Even harmonics
        ax_even = fig.add_subplot(outer[3, col_idx])
        even_handles = []
        for order in even_orders:
            color = _ORDER_COLORS.get(order, "black")
            an = np.array([r.get(f"a{order}", np.nan) for r in isophotes])
            bn = np.array([r.get(f"b{order}", np.nan) for r in isophotes])
            ok_a = valid & np.isfinite(an)
            ok_b = valid & np.isfinite(bn)
            if np.any(ok_a):
                ax_even.scatter(
                    xax[ok_a], an[ok_a], s=14, marker="s",
                    facecolors=color, edgecolors=color, alpha=0.7,
                )
            if np.any(ok_b):
                ax_even.scatter(
                    xax[ok_b], bn[ok_b], s=14, marker="s",
                    facecolors="none", edgecolors=color, alpha=0.7,
                )
            even_handles.append(
                Line2D([], [], marker="s", linestyle="None",
                       color=color, markersize=5,
                       label=f"a{order} (filled), b{order} (open)")
            )

        ax_even.axhline(0.0, color="gray", linestyle=":", linewidth=0.7, alpha=0.6)
        ax_even.set_ylabel("even harmonics")
        ax_even.set_xlabel(r"SMA$^{0.25}$ (pixel$^{0.25}$)")
        ax_even.grid(alpha=0.2)
        if even_handles:
            ax_even.legend(handles=even_handles, fontsize=7, loc="upper right")

        # Set x-limits
        if np.any(valid):
            set_x_limits_with_right_margin(ax_even, xax[valid])
            ax_odd.set_xlim(ax_even.get_xlim())

    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved harmonic detail figure → {filename}")


# ---------------------------------------------------------------------------
# Comparison QA (adapted for the 3 review conditions)
# ---------------------------------------------------------------------------

def plot_review_comparison_qa(
    galaxy: str,
    image: np.ndarray,
    mask: np.ndarray | None,
    condition_results: dict[str, list[dict[str, Any]]],
    condition_models: dict[str, np.ndarray],
    filename: str,
) -> None:
    """3-condition comparison QA figure.

    Layout: top row (3 residual panels) + 5 profile rows sharing SMA^0.25 x-axis.
    """
    configure_qa_plot_style()

    conditions = [c for c in REVIEW_CONDITIONS if c in condition_results]
    n_cond = len(conditions)
    if n_cond == 0:
        return

    # Residual maps and shared scale
    residuals = {}
    all_abs = []
    for cond in conditions:
        res_map = np.where(np.isfinite(image), image - condition_models[cond], np.nan)
        residuals[cond] = res_map
        finite = res_map[np.isfinite(res_map)]
        if finite.size > 0:
            all_abs.append(np.abs(finite))
    if all_abs:
        res_limit = float(np.nanpercentile(np.concatenate(all_abs), 99.0))
        res_limit = max(res_limit, 0.05)
    else:
        res_limit = 1.0

    # Figure layout
    n_profile_rows = 5
    fig_height = 3.5 + 1.5 * n_profile_rows
    fig = plt.figure(figsize=(14.0, fig_height))

    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[1.6, float(n_profile_rows)],
        hspace=0.12,
    )

    # Top: residual panels
    top = gridspec.GridSpecFromSubplotSpec(
        1, n_cond, subplot_spec=outer[0], wspace=0.05,
    )

    # Bottom: profile panels
    profile_height_ratios = [2.5, 1.2, 1.2, 1.2, 1.2]
    bottom = gridspec.GridSpecFromSubplotSpec(
        n_profile_rows, 1, subplot_spec=outer[1],
        height_ratios=profile_height_ratios, hspace=0.0,
    )

    fig.suptitle(
        f"{galaxy.upper()} — harmonic review comparison",
        fontsize=16, y=0.995,
    )

    # Residual panels
    for col, cond in enumerate(conditions):
        ax = fig.add_subplot(top[0, col])
        ax.imshow(
            residuals[cond], origin="lower", cmap="coolwarm",
            vmin=-res_limit, vmax=res_limit, interpolation="nearest",
        )
        ax.set_title(REVIEW_DISPLAY[cond], fontsize=9, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])

    # Profile panels
    def _arr(isos, key, default=np.nan):
        return np.array([r.get(key, default) for r in isos])

    ax_sb = fig.add_subplot(bottom[0])
    ax_odd = fig.add_subplot(bottom[1], sharex=ax_sb)
    ax_even = fig.add_subplot(bottom[2], sharex=ax_sb)
    ax_ba = fig.add_subplot(bottom[3], sharex=ax_sb)
    ax_pa = fig.add_subplot(bottom[4], sharex=ax_sb)

    # Detect all harmonic orders across conditions
    all_orders = set()
    for cond in conditions:
        for r in condition_results[cond]:
            for order in range(3, 8):
                if np.isfinite(r.get(f"a{order}", np.nan)):
                    all_orders.add(order)
    odd_orders = sorted(o for o in all_orders if o % 2 == 1)
    even_orders = sorted(o for o in all_orders if o % 2 == 0)

    legend_handles = []

    for cond in conditions:
        res = condition_results[cond]
        sma = _arr(res, "sma")
        intens = _arr(res, "intens")
        eps = _arr(res, "eps")
        pa_rad = _arr(res, "pa")
        stop = _arr(res, "stop_code", 0).astype(int)

        xax = sma ** 0.25
        vm = np.isfinite(xax) & (sma > 1.0)
        good = vm & (stop == 0)

        col = REVIEW_COLORS[cond]
        mrk = REVIEW_MARKERS[cond]
        label = REVIEW_DISPLAY[cond]
        scatter_kw = dict(s=16, marker=mrk, alpha=0.75)

        # Surface brightness
        sb_ok = vm & np.isfinite(intens) & (intens > 0)
        if np.any(sb_ok):
            ax_sb.scatter(
                xax[sb_ok], np.log10(intens[sb_ok]),
                facecolors=col, edgecolors=col, **scatter_kw, label=label,
            )

        # Odd harmonics
        for order in odd_orders:
            an = _arr(res, f"a{order}")
            ok = vm & np.isfinite(an)
            if np.any(ok):
                ax_odd.scatter(
                    xax[ok], an[ok],
                    facecolors=col, edgecolors=col, **scatter_kw,
                )

        # Even harmonics
        for order in even_orders:
            an = _arr(res, f"a{order}")
            ok = vm & np.isfinite(an)
            if np.any(ok):
                ax_even.scatter(
                    xax[ok], an[ok],
                    facecolors=col, edgecolors=col, **scatter_kw,
                )

        # Axis ratio
        ba = 1.0 - eps
        if np.any(vm & good):
            ax_ba.scatter(
                xax[vm & good], ba[vm & good],
                facecolors=col, edgecolors=col, **scatter_kw,
            )

        # PA
        pa_deg = normalize_pa_degrees(np.degrees(pa_rad))
        if np.any(vm & good):
            ax_pa.scatter(
                xax[vm & good], pa_deg[vm & good],
                facecolors=col, edgecolors=col, **scatter_kw,
            )

        legend_handles.append(
            Line2D([], [], marker=mrk, linestyle="None",
                   color=col, markersize=6, label=label)
        )

    # Axis labels and formatting
    ax_sb.set_ylabel(r"$\log_{10}(I)$")
    ax_sb.set_title("Surface brightness")
    ax_odd.set_ylabel(r"$a_n$ (odd)")
    ax_odd.axhline(0.0, color="gray", linestyle=":", linewidth=0.7, alpha=0.6)
    ax_even.set_ylabel(r"$a_n$ (even)")
    ax_even.axhline(0.0, color="gray", linestyle=":", linewidth=0.7, alpha=0.6)
    ax_ba.set_ylabel("axis ratio")
    ax_pa.set_ylabel("PA [deg]")
    ax_pa.set_xlabel(r"SMA$^{0.25}$ (pixel$^{0.25}$)")

    for ax in [ax_sb, ax_odd, ax_even, ax_ba]:
        ax.tick_params(labelbottom=False)
        ax.grid(alpha=0.2)
    ax_pa.grid(alpha=0.2)

    # X-limits
    all_xvals = []
    for cond in conditions:
        sma = _arr(condition_results[cond], "sma")
        vm = np.isfinite(sma) & (sma > 1.0)
        if np.any(vm):
            all_xvals.append(sma[vm] ** 0.25)
    if all_xvals:
        set_x_limits_with_right_margin(ax_pa, np.concatenate(all_xvals))

    # Legend
    ax_sb.legend(
        handles=legend_handles, loc="upper right", fontsize=9,
        ncol=1, frameon=True,
    )

    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.05, top=0.970, hspace=0.04)
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  Saved comparison QA → {filename}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Harmonic visualization review — EA-mode, 3 conditions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--galaxy",
        choices=list(SUPPORTED_GALAXIES),
        required=True,
        help="Galaxy to process.",
    )
    parser.add_argument(
        "--band-index",
        type=int,
        default=1,
        help="Band plane index (0-based) to extract from the 3D FITS cube.",
    )
    parser.add_argument(
        "--sma0",
        type=float,
        default=None,
        help="Override initial SMA (pixels).",
    )
    parser.add_argument(
        "--skip-mask",
        action="store_true",
        help="Use an empty mask instead of generating one.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_arguments()
    galaxy = args.galaxy
    band_index = args.band_index

    suffix = "_no_mask" if args.skip_mask else ""
    output_dir = Path("outputs/harmonic_review") / f"{galaxy}{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    band_name = BAND_NAMES[galaxy][band_index]
    print(f"\n{'='*60}")
    print(f"  Harmonic Visualization Review")
    print(f"  Galaxy : {galaxy}  (band {band_name}, index {band_index})")
    print(f"  Output : {output_dir}")
    print(f"{'='*60}\n")

    # Load image
    fits_path = _DATA_DIR / FITS_FILENAME[galaxy]
    if not fits_path.exists():
        sys.exit(f"ERROR: FITS file not found: {fits_path}")

    print("Loading image ...")
    image, pixel_scale = load_legacysurvey_fits(fits_path, band_index)
    print(f"  Image shape: {image.shape}, pixel scale: {PIXEL_SCALE[galaxy]} arcsec/px")

    # Generate mask
    if args.skip_mask:
        print("Skipping mask generation (--skip-mask).")
        mask = np.zeros(image.shape, dtype=bool)
    else:
        print("Generating mask ...")
        mask_kwargs = MASK_PARAMS.get(galaxy, {})
        center_xy = ((image.shape[1] - 1) / 2.0, (image.shape[0] - 1) / 2.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask = make_object_mask(image, center_xy=center_xy, **mask_kwargs)
    n_masked = int(np.sum(mask))
    print(f"  Masked pixels: {n_masked} / {mask.size} ({100.0*n_masked/mask.size:.1f}%)")

    # Build configs
    configs = make_review_configs(galaxy, sma0=args.sma0)

    # Run fits
    all_results: dict[str, list[dict]] = {}
    all_models: dict[str, np.ndarray] = {}

    for cond_label, config in configs.items():
        cond_dir = output_dir / cond_label
        cond_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  [{cond_label}]")
        print(f"    EA={config.use_eccentric_anomaly}  "
              f"sim_harm={config.simultaneous_harmonics}  "
              f"compute_dev={config.compute_deviations}  "
              f"orders={config.harmonic_orders}")

        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = isoster.fit_image(image, mask, config)
        elapsed = time.perf_counter() - t0

        isophotes = results["isophotes"]
        n_iso = len(isophotes)
        n_conv = sum(1 for r in isophotes if r.get("stop_code", -99) == 0)
        print(f"    {n_iso} isophotes, {n_conv} converged, {elapsed:.1f} s")

        # Save FITS
        isoster.isophote_results_to_fits(results, str(cond_dir / "isophotes.fits"))

        # Build model (use_harmonics=True only if harmonics were fitted)
        has_harmonics = config.simultaneous_harmonics or config.compute_deviations
        model = build_isoster_model(
            image.shape, isophotes,
            use_harmonics=has_harmonics,
        )

        # Per-condition QA
        harmonic_orders = config.harmonic_orders if has_harmonics else None
        plot_qa_summary_extended(
            title=f"{galaxy} — {REVIEW_DISPLAY[cond_label]}",
            image=image,
            isoster_model=model,
            isoster_res=isophotes,
            harmonic_orders=harmonic_orders,
            mask=mask,
            filename=str(cond_dir / "qa.png"),
        )
        print(f"    Saved QA → {cond_dir / 'qa.png'}")

        all_results[cond_label] = isophotes
        all_models[cond_label] = model

    # Comparison QA
    print("\nGenerating comparison QA ...")
    plot_review_comparison_qa(
        galaxy=galaxy,
        image=image,
        mask=mask,
        condition_results=all_results,
        condition_models=all_models,
        filename=str(output_dir / "comparison_qa.png"),
    )

    # Harmonic detail figures
    harmonic_conds = {
        c: all_results[c] for c in ["ea_harmonics_34", "ea_harmonics_34567"]
        if c in all_results
    }
    harmonic_models = {
        c: all_models[c] for c in harmonic_conds
    }

    if harmonic_conds:
        print("Generating harmonic detail figure ...")
        plot_harmonic_detail(
            image=image,
            mask=mask,
            condition_results=harmonic_conds,
            condition_models=harmonic_models,
            galaxy=galaxy,
            filename=str(output_dir / "harmonic_detail.png"),
        )

    print(f"\nDone. Outputs in: {output_dir}")
    print(f"  ls {output_dir}/\n")


if __name__ == "__main__":
    main()
