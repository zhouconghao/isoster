"""
Plotting utilities for isophote analysis visualization.

This module provides reusable plotting functions for QA figures that follow
the Huang2013 campaign baseline style (build_method_qa_figure convention).
Key features: stop-code-separated scatter, arcsinh image display with
colorbars, fractional residual maps, robust axis limits, and PA unwrap.
"""

from __future__ import annotations

import platform
import shutil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse as MPLEllipse

# ---------------------------------------------------------------------------
# Stop-code styling constants
# ---------------------------------------------------------------------------

STOP_CODE_STYLES = {
    0: {"color": "#1f77b4", "marker": "o", "label": "stop=0"},
    1: {"color": "#ff7f0e", "marker": "s", "label": "stop=1"},
    2: {"color": "#2ca02c", "marker": "^", "label": "stop=2"},
    3: {"color": "#d62728", "marker": "D", "label": "stop=3"},
    -1: {"color": "#9467bd", "marker": "X", "label": "stop=-1"},
}

MONOCHROME_STOP_MARKERS = {
    0: "o",
    1: "s",
    2: "^",
    3: "D",
    -1: "X",
}

MONOCHROME_STOP_COLORS = {
    0: "#111111",
    1: "#1f3b73",
    2: "#1b5e20",
    3: "#7f1d1d",
    -1: "#4b2e83",
}


# ---------------------------------------------------------------------------
# Style and text helpers
# ---------------------------------------------------------------------------

def style_for_stop_code(stop_code: int, monochrome: bool = False) -> dict[str, str]:
    """Return plotting style dict (color, marker, label) for a stop code."""
    if monochrome:
        marker = MONOCHROME_STOP_MARKERS.get(stop_code, "o")
        color = MONOCHROME_STOP_COLORS.get(stop_code, "#374151")
        return {"color": color, "marker": marker, "label": f"stop={stop_code}"}
    if stop_code in STOP_CODE_STYLES:
        return STOP_CODE_STYLES[stop_code]
    return {"color": "#7f7f7f", "marker": "o", "label": f"stop={stop_code}"}


def configure_qa_plot_style() -> None:
    """Apply shared plotting style for QA figures."""
    latex_available = (
        platform.system() != "Windows" and shutil.which("latex") is not None
    )
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "text.usetex": latex_available,
            "axes.labelsize": 15,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 14,
            "axes.unicode_minus": False,
        }
    )


def latex_safe_text(label_text: str) -> str:
    """Escape percent signs when LaTeX text rendering is active."""
    if plt.rcParams.get("text.usetex", False):
        return label_text.replace("%", r"\%")
    return label_text


# ---------------------------------------------------------------------------
# PA normalization (unwrap-aware)
# ---------------------------------------------------------------------------

def normalize_pa_degrees(pa_degrees: np.ndarray, anchor: float | None = None) -> np.ndarray:
    """Normalize PA values while preserving continuity across 180-degree periodicity.

    Uses double-angle unwrap to avoid spurious jumps larger than 90 degrees.
    
    Args:
        pa_degrees: Input PA array in degrees.
        anchor: If provided, shift the entire unwrapped profile by multiples of 
                180 to be as close as possible to this anchor value at the first 
                valid point.
    """
    pa = np.asarray(pa_degrees, dtype=float)
    output = np.full(pa.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(pa)
    if not np.any(finite_mask):
        return output

    # 1. Force into [0, 180) before unwrapping to ensure a clean start
    wrapped = np.mod(pa[finite_mask], 180.0)
    
    # 2. Double-angle unwrap (since PA has 180-deg period, 2*PA has 360-deg period)
    doubled_rad = np.deg2rad(2.0 * wrapped)
    unwrapped_rad = np.unwrap(doubled_rad)
    normalized = np.rad2deg(0.5 * unwrapped_rad)
    
    # 3. Optional anchoring to a global reference (to avoid large offsets from 0-180)
    if anchor is not None:
        first_val = normalized[0]
        offset = 180.0 * np.round((anchor - first_val) / 180.0)
        normalized += offset
    
    output[finite_mask] = normalized
    return output


def normalize_angle(angle_rad):
    """Normalize angle in radians to [0, 180) degrees (legacy compatibility)."""
    deg = np.degrees(angle_rad)
    return np.mod(deg, 180.0)


# ---------------------------------------------------------------------------
# Arcsinh display pipeline
# ---------------------------------------------------------------------------

def derive_arcsinh_parameters(
    image_values: np.ndarray,
    lower_percentile: float = 0.05,
    upper_percentile: float = 99.99,
    scale_percentile: float = 70.0,
) -> tuple[float, float, float, float]:
    """Derive arcsinh display parameters (low, high, scale, vmax) from image statistics."""
    finite_values = image_values[np.isfinite(image_values)]
    if finite_values.size == 0:
        return 0.0, 1.0, 1.0, 1.0

    low = float(np.nanpercentile(finite_values, lower_percentile))
    high = float(np.nanpercentile(finite_values, upper_percentile))
    if not np.isfinite(high) or high <= low:
        high = low + 1.0

    clipped = np.clip(image_values, low, high)
    shifted = np.clip(clipped - low, 0.0, None)
    positive = shifted[np.isfinite(shifted) & (shifted > 0.0)]
    scale = (
        float(np.nanpercentile(positive, scale_percentile)) if positive.size else 1.0
    )
    scale = max(scale, 1e-12)

    display = np.arcsinh(shifted / scale)
    finite_display = display[np.isfinite(display)]
    vmax = float(np.nanpercentile(finite_display, 99.8)) if finite_display.size else 1.0
    vmax = max(vmax, 1e-6)

    return low, high, scale, vmax


def make_arcsinh_display_from_parameters(
    image_values: np.ndarray,
    low: float,
    high: float,
    scale: float,
    vmax: float,
) -> tuple[np.ndarray, float, float]:
    """Build arcsinh display map from explicit scaling parameters.

    Returns (display_array, vmin, vmax).
    """
    clipped = np.clip(image_values, low, high)
    shifted = np.clip(clipped - low, 0.0, None)
    display = np.arcsinh(shifted / max(scale, 1e-12))
    return display, 0.0, max(vmax, 1e-6)


def make_arcsinh_display(
    image_values: np.ndarray,
    lower_percentile: float = 0.05,
    upper_percentile: float = 99.99,
    scale_percentile: float = 70.0,
) -> tuple[np.ndarray, float, float, float]:
    """Build arcsinh-scaled display map and limits.

    Returns (display_array, vmin, vmax, scale).
    """
    low, high, scale, vmax = derive_arcsinh_parameters(
        image_values,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        scale_percentile=scale_percentile,
    )
    display, vmin, vmax = make_arcsinh_display_from_parameters(
        image_values, low=low, high=high, scale=scale, vmax=vmax,
    )
    return display, vmin, vmax, scale


# ---------------------------------------------------------------------------
# Residual computation
# ---------------------------------------------------------------------------

def compute_fractional_residual_percent(
    image: np.ndarray, model: np.ndarray
) -> np.ndarray:
    """Compute residual map as ``100 * (model - data) / data``."""
    residual = np.full(image.shape, np.nan, dtype=float)
    valid = np.isfinite(image) & (np.abs(image) > 0.0)
    residual[valid] = 100.0 * (model[valid] - image[valid]) / image[valid]
    return residual


# ---------------------------------------------------------------------------
# Robust axis-limit helpers
# ---------------------------------------------------------------------------

def robust_limits(
    values: np.ndarray, lower_percentile: float = 5.0, upper_percentile: float = 95.0
) -> tuple[float, float]:
    """Compute robust plotting limits from finite values."""
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return -1.0, 1.0
    low = float(np.nanpercentile(finite_values, lower_percentile))
    high = float(np.nanpercentile(finite_values, upper_percentile))
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        delta = max(abs(low), 1.0)
        return low - delta, high + delta
    return low, high


def set_axis_limits_from_finite_values(
    axis,
    values: np.ndarray,
    invert: bool = False,
    margin_fraction: float = 0.08,
    min_margin: float = 0.05,
    lower_clip: float | None = None,
    upper_clip: float | None = None,
) -> None:
    """Set y-axis limits from finite data values with a comfortable margin."""
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return

    value_low = float(np.nanmin(finite_values))
    value_high = float(np.nanmax(finite_values))
    if value_high <= value_low:
        margin = max(min_margin, 0.05 * max(abs(value_low), 1.0))
    else:
        margin = max(min_margin, margin_fraction * (value_high - value_low))

    limit_low = value_low - margin
    limit_high = value_high + margin
    if lower_clip is not None:
        limit_low = max(limit_low, lower_clip)
    if upper_clip is not None:
        limit_high = min(limit_high, upper_clip)

    if limit_high <= limit_low:
        midpoint = 0.5 * (value_low + value_high)
        half_width = max(min_margin, 0.02)
        limit_low = midpoint - half_width
        limit_high = midpoint + half_width
        if lower_clip is not None:
            limit_low = max(limit_low, lower_clip)
        if upper_clip is not None:
            limit_high = min(limit_high, upper_clip)
        if limit_high <= limit_low:
            return

    if invert:
        axis.set_ylim(limit_high, limit_low)
    else:
        axis.set_ylim(limit_low, limit_high)


def set_x_limits_with_right_margin(
    axis,
    x_values: np.ndarray,
    min_margin: float = 0.02,
    margin_fraction: float = 0.03,
) -> None:
    """Set x-axis limits with a small right-edge margin for readability."""
    finite_values = np.asarray(x_values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return

    x_low = float(np.nanmin(finite_values))
    x_high = float(np.nanmax(finite_values))
    width = max(x_high - x_low, 0.0)
    right_margin = max(min_margin, margin_fraction * max(width, 1.0))
    axis.set_xlim(x_low, x_high + right_margin)


# ---------------------------------------------------------------------------
# Isophote overlay drawing
# ---------------------------------------------------------------------------

def draw_isophote_overlays(
    axis,
    isophotes,
    step: int = 10,
    line_width: float = 1.0,
    alpha: float = 0.7,
    edge_color: str | None = None,
) -> None:
    """Overlay selective isophotes on an image axis.

    Parameters
    ----------
    isophotes : list of dict
        Isophote results with keys: sma, x0, y0, eps, pa, stop_code.
    step : int
        Draw every *step*-th isophote.
    edge_color : str or None
        Fixed color override; if None, color by stop code.
    """
    if len(isophotes) == 0:
        return

    step = max(step, 1)
    for index in range(0, len(isophotes), step):
        iso = isophotes[index]
        sma = float(iso.get("sma", 0.0))
        if not np.isfinite(sma) or sma <= 1.0:
            continue

        x0 = float(iso.get("x0", 0.0))
        y0 = float(iso.get("y0", 0.0))
        eps = float(iso.get("eps", 0.0))
        pa_rad = float(iso.get("pa", 0.0))
        stop_code = int(iso.get("stop_code", 0))

        style = style_for_stop_code(stop_code)
        color = edge_color if edge_color is not None else style["color"]

        ellipse = MPLEllipse(
            (x0, y0),
            2.0 * sma,
            2.0 * sma * (1.0 - eps),
            angle=np.rad2deg(pa_rad),
            fill=False,
            linewidth=line_width,
            alpha=alpha,
            edgecolor=color,
        )
        axis.add_patch(ellipse)


# ---------------------------------------------------------------------------
# Scatter-by-stop-code profile plotting
# ---------------------------------------------------------------------------

def plot_profile_by_stop_code(
    axis,
    x_values: np.ndarray,
    y_values: np.ndarray,
    stop_codes: np.ndarray,
    y_errors: np.ndarray | None = None,
    marker_face: str = "filled",
    label_prefix: str = "",
    marker_size: float = 26.0,
    monochrome: bool = False,
) -> None:
    """Scatter/errorbar profile points separated by stop codes.

    Parameters
    ----------
    marker_face : str
        ``"filled"`` for solid markers, ``"open"`` for hollow.
    monochrome : bool
        If True, use monochrome palette (non-zero stop codes are hollow).
    """
    unique_stop_codes = sorted(
        {int(code) for code in stop_codes[np.isfinite(stop_codes)]},
        key=lambda code: (
            (0, 0) if code == 0 else (1, code) if code > 0 else (2, abs(code))
        ),
    )
    for stop_code in unique_stop_codes:
        mask = stop_codes == stop_code
        if not np.any(mask):
            continue
        style = style_for_stop_code(stop_code, monochrome=monochrome)
        if monochrome and stop_code != 0:
            face_color = "none"
        else:
            face_color = style["color"] if marker_face == "filled" else "none"

        if y_errors is not None:
            sanitized_errors = np.asarray(y_errors[mask], dtype=float)
            sanitized_errors[sanitized_errors < 0.0] = np.nan
            axis.errorbar(
                x_values[mask],
                y_values[mask],
                yerr=sanitized_errors,
                fmt=style["marker"],
                markersize=4.5,
                color=style["color"],
                mfc=face_color,
                mec=style["color"],
                capsize=1.8,
                linewidth=0.7,
                alpha=0.85,
                label=f"{label_prefix}{style['label']}",
            )
        else:
            axis.scatter(
                x_values[mask],
                y_values[mask],
                s=marker_size,
                marker=style["marker"],
                facecolors=face_color,
                edgecolors=style["color"],
                linewidths=0.9,
                alpha=0.9,
                label=f"{label_prefix}{style['label']}",
            )


# ---------------------------------------------------------------------------
# Main QA summary figure
# ---------------------------------------------------------------------------

def plot_qa_summary(
    title,
    image,
    isoster_model,
    isoster_res,
    photutils_res=None,
    mask=None,
    filename="qa_summary.png",
    relative_residual=False,
):
    """Generate a QA figure following the Huang2013 campaign baseline style.

    Parameters
    ----------
    title : str
        Figure title.
    image : 2D array
        Original input image.
    isoster_model : 2D array
        Reconstructed model from isoster isophotes.
    isoster_res : list of dict
        Results from isoster fitting (each dict has sma, intens, eps, pa, etc.).
    photutils_res : photutils.isophote.IsophoteList, optional
        Results from photutils fitting for comparison overlay.
    mask : 2D bool array, optional
        Bad-pixel mask (True = masked).
    filename : str
        Output filename.
    relative_residual : bool
        When True, show ``100 * (model - data) / data`` (fractional residual %).
        When False (default), show ``data - model`` (direct residual).
        Note: the sign convention differs between modes (direct uses
        data-model; fractional uses model-data).
    """
    configure_qa_plot_style()

    # --- Extract isoster arrays ------------------------------------------------
    def get_arr(key, default=np.nan):
        return np.array([r.get(key, default) for r in isoster_res])

    i_sma = get_arr("sma")
    i_intens = get_arr("intens")
    i_intens_err = get_arr("intens_err")
    i_eps = get_arr("eps")
    i_eps_err = get_arr("eps_err")
    i_pa = get_arr("pa")
    i_pa_err = get_arr("pa_err")
    i_x0 = get_arr("x0")
    i_x0_err = get_arr("x0_err")
    i_y0 = get_arr("y0")
    i_y0_err = get_arr("y0_err")
    i_a3 = get_arr("a3")
    i_b3 = get_arr("b3")
    i_a4 = get_arr("a4")
    i_b4 = get_arr("b4")
    i_stop = get_arr("stop_code", 0).astype(int)

    # Derived columns
    x_axis = i_sma ** 0.25
    valid_mask = np.isfinite(x_axis) & (i_sma > 1.0)

    i_ba = 1.0 - i_eps
    i_pa_deg = normalize_pa_degrees(np.degrees(i_pa))
    i_pa_err_deg = np.degrees(i_pa_err)

    # Center offsets relative to median
    median_x0 = np.nanmedian(i_x0[valid_mask]) if np.any(valid_mask) else 0.0
    median_y0 = np.nanmedian(i_y0[valid_mask]) if np.any(valid_mask) else 0.0
    i_dx = i_x0 - median_x0
    i_dy = i_y0 - median_y0

    # Determine panel count: base 5 + optional harmonics + optional CoG
    has_harmonics = np.any(np.isfinite(i_a3)) or np.any(np.isfinite(i_a4))
    has_cog = any("cog" in r for r in isoster_res)

    n_panels = 4  # SB, centroid, b/a, PA
    if has_harmonics:
        n_panels += 1
    if has_cog:
        n_panels += 1

    # Height ratios: SB panel 2.2x, rest 1.0x, last 1.2x
    height_ratios = [2.2] + [1.0] * (n_panels - 2) + [1.2]

    # --- Figure layout ---------------------------------------------------------
    fig = plt.figure(figsize=(13.6, 11.0))
    outer = gridspec.GridSpec(
        1, 2, figure=fig, width_ratios=[1.0, 2.01], wspace=0.27,
    )

    # Left column: 3 image rows, each with a colorbar sliver
    left = gridspec.GridSpecFromSubplotSpec(
        3, 2, subplot_spec=outer[0],
        width_ratios=[1.0, 0.04], hspace=0.10, wspace=-0.20,
    )

    # Right column: 1-D profile panels
    right = gridspec.GridSpecFromSubplotSpec(
        n_panels, 1, subplot_spec=outer[1],
        height_ratios=height_ratios, hspace=0.0,
    )

    fig.suptitle(title, fontsize=20, y=0.989)

    # --- Left column: 2-D panels ----------------------------------------------
    # Shared arcsinh parameters derived from the data image
    ref_low, ref_high, ref_scale, ref_vmax = derive_arcsinh_parameters(image)

    # Panel 1: Data + isophotes
    ax_img = fig.add_subplot(left[0, 0])
    img_display, img_vmin, img_vmax = make_arcsinh_display_from_parameters(
        image, low=ref_low, high=ref_high, scale=ref_scale, vmax=ref_vmax,
    )
    h_img = ax_img.imshow(
        img_display, origin="lower", cmap="viridis",
        vmin=img_vmin, vmax=img_vmax, interpolation="none",
    )
    if mask is not None:
        mask_overlay = np.zeros((*image.shape, 4))
        mask_overlay[mask] = [1, 0, 0, 0.4]
        ax_img.imshow(mask_overlay, origin="lower")

    overlay_step = max(1, len(isoster_res) // 15)
    draw_isophote_overlays(
        ax_img, isoster_res, step=overlay_step,
        line_width=1.2, alpha=0.8, edge_color="orangered",
    )
    ax_img.text(
        0.15, 0.9, "Data", fontsize=18, color="w",
        transform=ax_img.transAxes, ha="center", va="center",
        weight="bold", alpha=0.85,
    )

    ax_img_cb = fig.add_subplot(left[0, 1])
    fig.colorbar(h_img, cax=ax_img_cb).set_label(r"arcsinh((data $-$ p0.5) / scale)")

    # Panel 2: Model
    ax_mod = fig.add_subplot(left[1, 0])
    mod_display, mod_vmin, mod_vmax = make_arcsinh_display_from_parameters(
        isoster_model, low=ref_low, high=ref_high, scale=ref_scale, vmax=ref_vmax,
    )
    h_mod = ax_mod.imshow(
        mod_display, origin="lower", cmap="viridis",
        vmin=mod_vmin, vmax=mod_vmax, interpolation="none",
    )
    ax_mod.text(
        0.15, 0.9, "Model", fontsize=18, color="w",
        transform=ax_mod.transAxes, ha="center", va="center",
        weight="bold", alpha=0.85,
    )

    ax_mod_cb = fig.add_subplot(left[1, 1])
    fig.colorbar(h_mod, cax=ax_mod_cb).set_label(r"arcsinh((model $-$ p0.5) / scale)")

    # Panel 3: Residual (coolwarm)
    ax_res = fig.add_subplot(left[2, 0])
    if relative_residual:
        residual = compute_fractional_residual_percent(image, isoster_model)
        res_label = latex_safe_text("(model - data) / data [%]")
        res_clip_lo, res_clip_hi = 0.05, 8.0
    else:
        residual = np.where(np.isfinite(image), image - isoster_model, np.nan)
        res_label = "data - model"
        res_clip_lo, res_clip_hi = None, None
    abs_res = np.abs(residual[np.isfinite(residual)])
    res_limit = np.nanpercentile(abs_res, 99.0) if abs_res.size else 1.0
    if res_clip_lo is not None:
        res_limit = float(np.clip(res_limit, res_clip_lo, res_clip_hi))
    h_res = ax_res.imshow(
        residual, origin="lower", cmap="coolwarm",
        vmin=-res_limit, vmax=res_limit, interpolation="nearest",
    )
    ax_res.text(
        0.18, 0.9, "Residual", fontsize=18, color="k",
        transform=ax_res.transAxes, ha="center", va="center",
        weight="bold", alpha=0.85,
    )

    ax_res_cb = fig.add_subplot(left[2, 1])
    fig.colorbar(h_res, cax=ax_res_cb).set_label(res_label)

    for ax in [ax_img, ax_mod, ax_res]:
        ax.set_xlabel("x [pixel]")
        ax.set_ylabel("y [pixel]")

    # --- Right column: 1-D profiles -------------------------------------------
    panel_idx = 0

    # 1. Surface brightness (log10 I)
    ax_sb = fig.add_subplot(right[panel_idx])
    panel_idx += 1

    sb_valid = valid_mask & np.isfinite(i_intens) & (i_intens > 0)
    y_sb = np.full_like(i_intens, np.nan)
    y_sb[sb_valid] = np.log10(i_intens[sb_valid])
    y_sb_err = np.full_like(i_intens_err, np.nan)
    y_sb_err[sb_valid] = i_intens_err[sb_valid] / (i_intens[sb_valid] * np.log(10))

    plot_profile_by_stop_code(
        ax_sb, x_axis[sb_valid], y_sb[sb_valid], i_stop[sb_valid],
        y_errors=y_sb_err[sb_valid], marker_face="filled", monochrome=True,
    )

    if photutils_res is not None:
        _overlay_photutils_sb(ax_sb, photutils_res)

    ax_sb.set_ylabel(r"$\log_{10}(I)$")
    ax_sb.set_title("Surface brightness profile")
    ax_sb.grid(alpha=0.25)
    sb_for_limits = y_sb[sb_valid & np.isfinite(y_sb)]
    if sb_for_limits.size > 0:
        set_axis_limits_from_finite_values(
            ax_sb, sb_for_limits, margin_fraction=0.06, min_margin=0.2,
        )
    handles, labels = ax_sb.get_legend_handles_labels()
    if handles:
        ax_sb.legend(handles[:8], labels[:8], loc="upper right", fontsize=14, ncol=1)

    # 2. Center offset
    ax_cen = fig.add_subplot(right[panel_idx], sharex=ax_sb)
    panel_idx += 1

    plot_profile_by_stop_code(
        ax_cen, x_axis[valid_mask], i_dx[valid_mask], i_stop[valid_mask],
        y_errors=i_x0_err[valid_mask], marker_face="filled",
        label_prefix="dx ", monochrome=True,
    )
    plot_profile_by_stop_code(
        ax_cen, x_axis[valid_mask], i_dy[valid_mask], i_stop[valid_mask],
        y_errors=i_y0_err[valid_mask], marker_face="open",
        label_prefix="dy ", monochrome=True,
    )
    ax_cen.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_cen.set_ylabel("center offset [pix]")
    ax_cen.grid(alpha=0.25)
    centroid_legend = [
        Line2D([], [], marker="o", linestyle="None", color="black",
               markerfacecolor="black", markersize=4.8, label="X"),
        Line2D([], [], marker="o", linestyle="None", color="black",
               markerfacecolor="none", markersize=4.8, label="Y"),
    ]
    ax_cen.legend(handles=centroid_legend, loc="upper right", frameon=True,
                  fontsize=14, ncol=2)

    # 3. Axis ratio (b/a)
    ax_ba = fig.add_subplot(right[panel_idx], sharex=ax_sb)
    panel_idx += 1

    plot_profile_by_stop_code(
        ax_ba, x_axis[valid_mask], i_ba[valid_mask], i_stop[valid_mask],
        y_errors=i_eps_err[valid_mask], marker_face="filled", monochrome=True,
    )
    ba_for_limits = i_ba[valid_mask & np.isfinite(i_ba)]
    set_axis_limits_from_finite_values(
        ax_ba, ba_for_limits, margin_fraction=0.08, min_margin=0.03,
        lower_clip=0.0, upper_clip=1.0,
    )
    ax_ba.set_ylabel("axis ratio")
    ax_ba.grid(alpha=0.25)

    # 4. Position angle
    ax_pa = fig.add_subplot(right[panel_idx], sharex=ax_sb)
    panel_idx += 1

    plot_profile_by_stop_code(
        ax_pa, x_axis[valid_mask], i_pa_deg[valid_mask], i_stop[valid_mask],
        y_errors=i_pa_err_deg[valid_mask], marker_face="filled", monochrome=True,
    )
    pa_for_limits = i_pa_deg[valid_mask & np.isfinite(i_pa_deg) & (i_stop == 0)]
    if pa_for_limits.size > 1:
        pa_low, pa_high = robust_limits(pa_for_limits, 3, 97)
        pa_margin = max(3.0, 0.08 * (pa_high - pa_low + 1e-6))
        ax_pa.set_ylim(pa_low - pa_margin, pa_high + pa_margin)
    ax_pa.set_ylabel("PA [deg]")
    ax_pa.grid(alpha=0.25)

    # 5. Harmonics (A3/B3, A4/B4) — optional
    ax_harm = None
    if has_harmonics:
        ax_harm = fig.add_subplot(right[panel_idx], sharex=ax_sb)
        panel_idx += 1

        ax_harm.scatter(
            x_axis[valid_mask], i_a3[valid_mask], s=20, marker="o",
            facecolors="#1f77b4", edgecolors="#1f77b4", alpha=0.7, label="A3",
        )
        ax_harm.scatter(
            x_axis[valid_mask], i_b3[valid_mask], s=20, marker="s",
            facecolors="none", edgecolors="#1f77b4", alpha=0.7, label="B3",
        )
        ax_harm.scatter(
            x_axis[valid_mask], i_a4[valid_mask], s=20, marker="^",
            facecolors="#d62728", edgecolors="#d62728", alpha=0.7, label="A4",
        )
        ax_harm.scatter(
            x_axis[valid_mask], i_b4[valid_mask], s=20, marker="D",
            facecolors="none", edgecolors="#d62728", alpha=0.7, label="B4",
        )
        harm_values = np.concatenate([
            i_a3[valid_mask], i_b3[valid_mask],
            i_a4[valid_mask], i_b4[valid_mask],
        ])
        set_axis_limits_from_finite_values(
            ax_harm, harm_values, margin_fraction=0.10, min_margin=0.005,
        )
        ax_harm.axhline(0.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax_harm.set_ylabel("A/B harmonics")
        ax_harm.legend(loc="upper right", fontsize=12, ncol=4)
        ax_harm.grid(alpha=0.25)

    # 6. Curve of growth — optional
    ax_cog = None
    if has_cog:
        ax_cog = fig.add_subplot(right[panel_idx], sharex=ax_sb)
        panel_idx += 1

        cog_values = get_arr("cog")
        plot_profile_by_stop_code(
            ax_cog, x_axis[valid_mask], cog_values[valid_mask], i_stop[valid_mask],
            marker_face="filled", label_prefix="fit ", monochrome=True,
        )
        finite_cog = cog_values[valid_mask & np.isfinite(cog_values) & (cog_values > 0)]
        if finite_cog.size > 0 and np.all(finite_cog > 0):
            ax_cog.set_yscale("log")
        ax_cog.set_ylabel("CoG flux")
        ax_cog.grid(alpha=0.25)

    # X-axis label on bottom panel only, suppress tick labels on others
    all_right_axes = [ax_sb, ax_cen, ax_ba, ax_pa]
    if ax_harm is not None:
        all_right_axes.append(ax_harm)
    if ax_cog is not None:
        all_right_axes.append(ax_cog)

    for ax in all_right_axes[:-1]:
        ax.tick_params(labelbottom=False)
    all_right_axes[-1].set_xlabel(r"SMA$^{0.25}$ (pixel$^{0.25}$)")

    # X-limit with right margin
    if np.any(valid_mask):
        set_x_limits_with_right_margin(all_right_axes[-1], x_axis[valid_mask])

    # --- Final layout adjustment and save --------------------------------------
    fig.subplots_adjust(left=0.025, right=0.992, bottom=0.05, top=0.940, wspace=0.18)
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved QA figure to {filename}")


# ---------------------------------------------------------------------------
# Extended QA figure with per-order harmonic amplitude panels
# ---------------------------------------------------------------------------

def plot_qa_summary_extended(
    title,
    image,
    isoster_model,
    isoster_res,
    *,
    harmonic_orders: list[int] | None = None,
    harmonic_mode: str = "coefficients",
    normalize_harmonics: bool = False,
    relative_residual: bool = False,
    mask=None,
    filename="qa_summary_extended.png",
):
    """QA figure with extended harmonic visualization panels.

    Extends :func:`plot_qa_summary` with two harmonic-specific panels:

    * **Odd harmonics** — orders 3, 5, 7 (if present)
    * **Even harmonics** — orders 4, 6 (if present)

    Parameters
    ----------
    title : str
        Figure title.
    image : 2D array
        Original input image.
    isoster_model : 2D array
        Reconstructed 2D model from isoster isophotes.
    isoster_res : list of dict
        Isoster fitting results.
    harmonic_orders : list of int or None
        Which harmonic orders to show.  Auto-detected from the
        available ``a3``/``b3`` … ``a7``/``b7`` keys when None.
    harmonic_mode : str
        ``'coefficients'`` (default) shows individual ``a_n`` (filled) and
        ``b_n`` (open) per order.  ``'amplitude'`` shows
        ``A_n = sqrt(a_n^2 + b_n^2)`` per order.
    normalize_harmonics : bool
        Only used when ``harmonic_mode='amplitude'``.  When True, shows
        ``A_n / I`` instead of raw ``A_n``.
    relative_residual : bool
        When False (default), the residual map shows ``data - model``
        (absolute).  When True, shows ``(data - model) / data``
        (fractional).
    mask : 2D bool array, optional
        Bad-pixel mask (True = masked) for the data panel overlay.
    filename : str
        Output path.
    """
    configure_qa_plot_style()

    def get_arr(key, default=np.nan):
        return np.array([r.get(key, default) for r in isoster_res])

    i_sma = get_arr("sma")
    i_intens = get_arr("intens")
    i_intens_err = get_arr("intens_err")
    i_eps = get_arr("eps")
    i_eps_err = get_arr("eps_err")
    i_pa = get_arr("pa")
    i_pa_err = get_arr("pa_err")
    i_x0 = get_arr("x0")
    i_x0_err = get_arr("x0_err")
    i_y0 = get_arr("y0")
    i_y0_err = get_arr("y0_err")
    i_stop = get_arr("stop_code", 0).astype(int)

    # Determine which harmonic orders are present
    if harmonic_orders is None:
        harmonic_orders = []
        for order in [3, 4, 5, 6, 7]:
            key = f"a{order}"
            if any(np.isfinite(r.get(key, np.nan)) for r in isoster_res):
                harmonic_orders.append(order)

    # Pre-load harmonic arrays for all orders
    harm_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for order in harmonic_orders:
        harm_data[order] = (get_arr(f"a{order}"), get_arr(f"b{order}"))

    # Split into odd and even orders
    odd_orders = [o for o in harmonic_orders if o % 2 == 1]
    even_orders = [o for o in harmonic_orders if o % 2 == 0]

    x_axis = i_sma ** 0.25
    valid_mask = np.isfinite(x_axis) & (i_sma > 1.0)

    i_ba = 1.0 - i_eps
    i_pa_deg = normalize_pa_degrees(np.degrees(i_pa))
    i_pa_err_deg = np.degrees(i_pa_err)

    median_x0 = np.nanmedian(i_x0[valid_mask]) if np.any(valid_mask) else 0.0
    median_y0 = np.nanmedian(i_y0[valid_mask]) if np.any(valid_mask) else 0.0
    i_dx = i_x0 - median_x0
    i_dy = i_y0 - median_y0

    safe_intens = np.where(np.isfinite(i_intens) & (i_intens > 0.0), i_intens, np.nan)

    # 6 panels: SB, centroid, b/a, PA, odd harmonics, even harmonics
    n_panels = 6
    height_ratios = [2.2, 1.0, 1.0, 1.0, 1.4, 1.4]

    fig = plt.figure(figsize=(13.6, 13.0))
    outer = gridspec.GridSpec(
        1, 2, figure=fig, width_ratios=[1.0, 2.01], wspace=0.27,
    )
    left = gridspec.GridSpecFromSubplotSpec(
        3, 2, subplot_spec=outer[0],
        width_ratios=[1.0, 0.04], hspace=0.10, wspace=-0.20,
    )
    right = gridspec.GridSpecFromSubplotSpec(
        n_panels, 1, subplot_spec=outer[1],
        height_ratios=height_ratios, hspace=0.0,
    )
    fig.suptitle(title, fontsize=20, y=0.989)

    # --- Left panels: data / model / residual ---------------------------------
    ref_low, ref_high, ref_scale, ref_vmax = derive_arcsinh_parameters(image)

    ax_img = fig.add_subplot(left[0, 0])
    img_display, img_vmin, img_vmax = make_arcsinh_display_from_parameters(
        image, low=ref_low, high=ref_high, scale=ref_scale, vmax=ref_vmax,
    )
    h_img = ax_img.imshow(
        img_display, origin="lower", cmap="viridis",
        vmin=img_vmin, vmax=img_vmax, interpolation="none",
    )
    if mask is not None:
        mask_overlay = np.zeros((*image.shape, 4))
        mask_overlay[mask] = [1, 0, 0, 0.4]
        ax_img.imshow(mask_overlay, origin="lower")
    overlay_step = max(1, len(isoster_res) // 15)
    draw_isophote_overlays(
        ax_img, isoster_res, step=overlay_step,
        line_width=1.2, alpha=0.8, edge_color="orangered",
    )
    ax_img.text(0.15, 0.9, "Data", fontsize=18, color="w",
                transform=ax_img.transAxes, ha="center", va="center",
                weight="bold", alpha=0.85)
    ax_img_cb = fig.add_subplot(left[0, 1])
    fig.colorbar(h_img, cax=ax_img_cb).set_label(r"arcsinh((data $-$ p0.5) / scale)")

    ax_mod = fig.add_subplot(left[1, 0])
    mod_display, mod_vmin, mod_vmax = make_arcsinh_display_from_parameters(
        isoster_model, low=ref_low, high=ref_high, scale=ref_scale, vmax=ref_vmax,
    )
    h_mod = ax_mod.imshow(
        mod_display, origin="lower", cmap="viridis",
        vmin=mod_vmin, vmax=mod_vmax, interpolation="none",
    )
    ax_mod.text(0.15, 0.9, "Model", fontsize=18, color="w",
                transform=ax_mod.transAxes, ha="center", va="center",
                weight="bold", alpha=0.85)
    ax_mod_cb = fig.add_subplot(left[1, 1])
    fig.colorbar(h_mod, cax=ax_mod_cb).set_label(r"arcsinh((model $-$ p0.5) / scale)")

    # Residual panel — absolute (default) or fractional
    ax_res = fig.add_subplot(left[2, 0])
    if relative_residual:
        residual_map = compute_fractional_residual_percent(image, isoster_model)
        res_label = latex_safe_text("(data - model) / data [%]")
    else:
        residual_map = np.where(np.isfinite(image), image - isoster_model, np.nan)
        res_label = "data $-$ model"

    abs_res = np.abs(residual_map[np.isfinite(residual_map)])
    res_limit = float(np.clip(
        np.nanpercentile(abs_res, 99.0) if abs_res.size else 1.0, 0.05, None,
    ))
    h_res = ax_res.imshow(
        residual_map, origin="lower", cmap="coolwarm",
        vmin=-res_limit, vmax=res_limit, interpolation="nearest",
    )
    ax_res.text(0.18, 0.9, "Residual", fontsize=18, color="k",
                transform=ax_res.transAxes, ha="center", va="center",
                weight="bold", alpha=0.85)
    ax_res_cb = fig.add_subplot(left[2, 1])
    fig.colorbar(h_res, cax=ax_res_cb).set_label(res_label)

    for ax in [ax_img, ax_mod, ax_res]:
        ax.set_xlabel("x [pixel]")
        ax.set_ylabel("y [pixel]")

    # --- Right panels: 1-D profiles ------------------------------------------
    panel_idx = 0

    # 1. Surface brightness
    ax_sb = fig.add_subplot(right[panel_idx])
    panel_idx += 1
    sb_valid = valid_mask & np.isfinite(i_intens) & (i_intens > 0)
    y_sb = np.full_like(i_intens, np.nan)
    y_sb[sb_valid] = np.log10(i_intens[sb_valid])
    y_sb_err = np.full_like(i_intens_err, np.nan)
    y_sb_err[sb_valid] = i_intens_err[sb_valid] / (i_intens[sb_valid] * np.log(10))
    plot_profile_by_stop_code(
        ax_sb, x_axis[sb_valid], y_sb[sb_valid], i_stop[sb_valid],
        y_errors=y_sb_err[sb_valid], marker_face="filled", monochrome=True,
    )
    ax_sb.set_ylabel(r"$\log_{10}(I)$")
    ax_sb.set_title("Surface brightness profile")
    ax_sb.grid(alpha=0.25)
    sb_for_limits = y_sb[sb_valid & np.isfinite(y_sb)]
    if sb_for_limits.size > 0:
        set_axis_limits_from_finite_values(ax_sb, sb_for_limits, margin_fraction=0.06, min_margin=0.2)
    handles, labels = ax_sb.get_legend_handles_labels()
    if handles:
        ax_sb.legend(handles[:8], labels[:8], loc="upper right", fontsize=12, ncol=1)

    # 2. Center offset
    ax_cen = fig.add_subplot(right[panel_idx], sharex=ax_sb)
    panel_idx += 1
    plot_profile_by_stop_code(
        ax_cen, x_axis[valid_mask], i_dx[valid_mask], i_stop[valid_mask],
        y_errors=i_x0_err[valid_mask], marker_face="filled",
        label_prefix="dx ", monochrome=True,
    )
    plot_profile_by_stop_code(
        ax_cen, x_axis[valid_mask], i_dy[valid_mask], i_stop[valid_mask],
        y_errors=i_y0_err[valid_mask], marker_face="open",
        label_prefix="dy ", monochrome=True,
    )
    ax_cen.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_cen.set_ylabel("center [pix]")
    ax_cen.grid(alpha=0.25)

    # 3. Axis ratio
    ax_ba = fig.add_subplot(right[panel_idx], sharex=ax_sb)
    panel_idx += 1
    plot_profile_by_stop_code(
        ax_ba, x_axis[valid_mask], i_ba[valid_mask], i_stop[valid_mask],
        y_errors=i_eps_err[valid_mask], marker_face="filled", monochrome=True,
    )
    ba_for_limits = i_ba[valid_mask & np.isfinite(i_ba)]
    set_axis_limits_from_finite_values(
        ax_ba, ba_for_limits, margin_fraction=0.08, min_margin=0.03,
        lower_clip=0.0, upper_clip=1.0,
    )
    ax_ba.set_ylabel("axis ratio")
    ax_ba.grid(alpha=0.25)

    # 4. PA
    ax_pa = fig.add_subplot(right[panel_idx], sharex=ax_sb)
    panel_idx += 1
    plot_profile_by_stop_code(
        ax_pa, x_axis[valid_mask], i_pa_deg[valid_mask], i_stop[valid_mask],
        y_errors=i_pa_err_deg[valid_mask], marker_face="filled", monochrome=True,
    )
    pa_for_limits = i_pa_deg[valid_mask & np.isfinite(i_pa_deg) & (i_stop == 0)]
    if pa_for_limits.size > 1:
        pa_low, pa_high = robust_limits(pa_for_limits, 3, 97)
        pa_margin = max(3.0, 0.08 * (pa_high - pa_low + 1e-6))
        ax_pa.set_ylim(pa_low - pa_margin, pa_high + pa_margin)
    ax_pa.set_ylabel("PA [deg]")
    ax_pa.grid(alpha=0.25)

    # Colour cycle for harmonic orders
    order_colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]
    order_markers = ["o", "^", "s", "D", "v"]

    def _plot_harmonic_panel(ax, orders, panel_label):
        """Plot harmonic data on a panel for the given orders."""
        all_values = []
        for idx_o, order in enumerate(orders):
            if order not in harm_data:
                continue
            an, bn = harm_data[order]
            col = order_colors[idx_o % len(order_colors)]
            mrk = order_markers[idx_o % len(order_markers)]

            if harmonic_mode == "amplitude":
                # A_n = sqrt(a_n^2 + b_n^2), optionally normalized by I
                amplitude = np.sqrt(
                    np.where(np.isfinite(an), an, 0.0) ** 2
                    + np.where(np.isfinite(bn), bn, 0.0) ** 2
                )
                amplitude[~(np.isfinite(an) & np.isfinite(bn))] = np.nan
                if normalize_harmonics:
                    amplitude = amplitude / safe_intens
                show = valid_mask & np.isfinite(amplitude)
                if np.any(show):
                    ax.scatter(
                        x_axis[show], amplitude[show],
                        s=18, marker=mrk, facecolors=col, edgecolors=col,
                        alpha=0.75, label=f"$A_{{{order}}}$",
                    )
                    all_values.append(amplitude[show])
            else:
                # Coefficients mode: a_n (filled) and b_n (open)
                show_a = valid_mask & np.isfinite(an)
                show_b = valid_mask & np.isfinite(bn)
                if np.any(show_a):
                    ax.scatter(
                        x_axis[show_a], an[show_a],
                        s=18, marker=mrk, facecolors=col, edgecolors=col,
                        alpha=0.75, label=f"$a_{{{order}}}$",
                    )
                    all_values.append(an[show_a])
                if np.any(show_b):
                    ax.scatter(
                        x_axis[show_b], bn[show_b],
                        s=18, marker=mrk, facecolors="none", edgecolors=col,
                        linewidths=0.9, alpha=0.75, label=f"$b_{{{order}}}$",
                    )
                    all_values.append(bn[show_b])

        ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        if all_values:
            cat = np.concatenate(all_values)
            set_axis_limits_from_finite_values(
                ax, cat, margin_fraction=0.12, min_margin=0.001,
            )

        # Y-axis label
        if harmonic_mode == "amplitude":
            if normalize_harmonics:
                ax.set_ylabel(f"$A_n / I$ ({panel_label})")
            else:
                ax.set_ylabel(f"$A_n$ ({panel_label})")
        else:
            ax.set_ylabel(f"$a_n, b_n$ ({panel_label})")

        ax.legend(loc="upper right", fontsize=9, ncol=min(len(orders) * 2, 6))
        ax.grid(alpha=0.25)

    # 5. Odd harmonics panel
    ax_odd = fig.add_subplot(right[panel_idx], sharex=ax_sb)
    panel_idx += 1
    if odd_orders:
        _plot_harmonic_panel(ax_odd, odd_orders, "odd")
    else:
        ax_odd.set_ylabel(f"$a_n, b_n$ (odd)")
        ax_odd.text(0.5, 0.5, "no odd orders", color="gray", fontsize=10,
                    transform=ax_odd.transAxes, ha="center", va="center")
        ax_odd.grid(alpha=0.25)

    # 6. Even harmonics panel
    ax_even = fig.add_subplot(right[panel_idx], sharex=ax_sb)
    panel_idx += 1
    if even_orders:
        _plot_harmonic_panel(ax_even, even_orders, "even")
    else:
        ax_even.set_ylabel(f"$a_n, b_n$ (even)")
        ax_even.text(0.5, 0.5, "no even orders", color="gray", fontsize=10,
                     transform=ax_even.transAxes, ha="center", va="center")
        ax_even.grid(alpha=0.25)

    # X-axis housekeeping
    all_right_axes = [ax_sb, ax_cen, ax_ba, ax_pa, ax_odd, ax_even]
    for ax in all_right_axes[:-1]:
        ax.tick_params(labelbottom=False)
    all_right_axes[-1].set_xlabel(r"SMA$^{0.25}$ (pixel$^{0.25}$)")
    if np.any(valid_mask):
        set_x_limits_with_right_margin(all_right_axes[-1], x_axis[valid_mask])

    fig.subplots_adjust(left=0.025, right=0.992, bottom=0.04, top=0.940, wspace=0.18)
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved extended QA figure to {filename}")


def _overlay_photutils_sb(ax, photutils_res):
    """Overlay photutils surface brightness on the SB panel (open markers)."""
    p_sma = np.array([iso.sma for iso in photutils_res])
    p_intens = np.array([iso.intens for iso in photutils_res])
    p_intens_err = np.array([iso.int_err for iso in photutils_res])

    p_mask = (p_sma > 1.5) & np.isfinite(p_intens) & (p_intens > 0)
    p_x = p_sma[p_mask] ** 0.25
    p_y = np.log10(p_intens[p_mask])
    p_yerr = p_intens_err[p_mask] / (p_intens[p_mask] * np.log(10))

    ax.errorbar(
        p_x, p_y, yerr=p_yerr,
        fmt="o", mfc="none", mec="black", color="black",
        markersize=4, capsize=1.8, linewidth=0.7, alpha=0.8,
        label="photutils",
    )
