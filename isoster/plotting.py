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
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse as MPLEllipse
from matplotlib.patches import Polygon as MPLPolygon

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
# Multi-method comparison constants
# ---------------------------------------------------------------------------

# Default visual styles for multi-method comparison figures.
# Each method entry provides: color, marker, marker_face ("filled" or "none"),
# overlay_color, overlay_width, label.
METHOD_STYLES: dict[str, dict[str, str | float]] = {
    "isoster": {
        "color": "#1f77b4",
        "marker": "o",
        "marker_face": "filled",
        "overlay_color": "white",
        "overlay_width": 1.0,
        "label": "isoster",
    },
    "photutils": {
        "color": "#d62728",
        "marker": "s",
        "marker_face": "none",
        "overlay_color": "orangered",
        "overlay_width": 1.1,
        "label": "photutils",
    },
    "autoprof": {
        "color": "#2ca02c",
        "marker": "^",
        "marker_face": "filled",
        "overlay_color": "lime",
        "overlay_width": 0.9,
        "label": "AutoProf",
    },
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
    latex_available = platform.system() != "Windows" and shutil.which("latex") is not None
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
# Multi-method profile building
# ---------------------------------------------------------------------------


def build_method_profile(
    data: list[dict] | dict[str, np.ndarray],
) -> dict[str, np.ndarray] | None:
    """Convert isophote list or array dict into standardized profile arrays.

    Accepts either a list of isophote dicts (isoster/photutils format, each
    dict has scalar values per isophote) or a dict of arrays (autoprof format,
    each key maps to an ndarray).

    Parameters
    ----------
    data : list[dict] or dict[str, ndarray]
        Isophote data in either format.

    Returns
    -------
    dict[str, ndarray] or None
        Standardized profile dict with keys: sma, x_axis, intens, eps, pa.
        Optional keys included when present in input: stop_codes, x0, y0,
        intens_err, rms.  Returns None if input is empty.
    """
    if isinstance(data, list):
        if len(data) == 0:
            return None
        sma = np.array([iso["sma"] for iso in data])
        profile = {
            "sma": sma,
            "x_axis": sma**0.25,
            "intens": np.array([iso["intens"] for iso in data]),
            "eps": np.array([iso["eps"] for iso in data]),
            "pa": np.array([iso["pa"] for iso in data]),
        }
        if "stop_code" in data[0]:
            profile["stop_codes"] = np.array([iso.get("stop_code", 0) for iso in data])
        for key in ("x0", "y0", "intens_err", "rms", "eps_err", "pa_err", "x0_err", "y0_err"):
            if key in data[0]:
                profile[key] = np.array([iso.get(key, np.nan) for iso in data])
        # Preserve harmonic coefficients (a3, b3, a4, b4, ...)
        for key in data[0]:
            if len(key) >= 2 and key[0] in ("a", "b") and key[1:].isdigit():
                n = int(key[1:])
                if n >= 3:
                    profile[key] = np.array([iso.get(key, 0.0) for iso in data])
        return profile

    elif isinstance(data, dict):
        sma = np.asarray(data.get("sma", []))
        if sma.size == 0:
            return None
        profile = {
            "sma": sma,
            "x_axis": sma**0.25,
            "intens": np.asarray(data["intens"]),
            "eps": np.asarray(data["eps"]),
            "pa": np.asarray(data["pa"]),
        }
        for key in (
            "stop_codes",
            "stop_code",
            "x0",
            "y0",
            "intens_err",
            "rms",
            "eps_err",
            "pa_err",
            "x0_err",
            "y0_err",
        ):
            if key in data and isinstance(data[key], np.ndarray):
                out_key = "stop_codes" if key == "stop_code" else key
                profile[out_key] = data[key]
        # Preserve harmonic coefficients (a3, b3, a4, b4, ...)
        for key in data:
            if len(key) >= 2 and key[0] in ("a", "b") and key[1:].isdigit():
                n = int(key[1:])
                if n >= 3 and isinstance(data[key], np.ndarray):
                    profile[key] = data[key]
        return profile

    return None


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
    scale = float(np.nanpercentile(positive, scale_percentile)) if positive.size else 1.0
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
        image_values,
        low=low,
        high=high,
        scale=scale,
        vmax=vmax,
    )
    return display, vmin, vmax, scale


# ---------------------------------------------------------------------------
# Residual computation
# ---------------------------------------------------------------------------


def compute_fractional_residual_percent(image: np.ndarray, model: np.ndarray) -> np.ndarray:
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
    skip_innermost: bool = True,
) -> None:
    """Set x-axis limits with margins for readability.

    Parameters
    ----------
    skip_innermost : bool
        If True (default), the left edge starts from the second-smallest
        unique x value (with a small left margin) so that the innermost
        point where no isophote can be reliably defined does not dominate
        the x-range.
    """
    finite_values = np.asarray(x_values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return

    sorted_unique = np.unique(finite_values)
    if skip_innermost and sorted_unique.size >= 2:
        x_low = float(sorted_unique[1])
    else:
        x_low = float(sorted_unique[0])

    x_high = float(sorted_unique[-1])
    width = max(x_high - x_low, 0.0)
    right_margin = max(min_margin, margin_fraction * max(width, 1.0))
    left_margin = max(min_margin, margin_fraction * max(width, 1.0))
    axis.set_xlim(x_low - left_margin, x_high + right_margin)


# ---------------------------------------------------------------------------
# Isophote overlay drawing
# ---------------------------------------------------------------------------


def _detect_harmonic_orders(iso: dict) -> list[int]:
    """Return sorted list of harmonic orders present in an isophote dict."""
    orders = set()
    for key in iso:
        if len(key) >= 2 and key[0] in ("a", "b") and key[1:].isdigit():
            n = int(key[1:])
            if n >= 3:
                orders.add(n)
    return sorted(orders)


def _compute_harmonic_contour(
    iso: dict,
    n_points: int = 360,
) -> np.ndarray:
    """Compute the (x, y) contour of an isophote with harmonic deviations.

    Returns an (n_points, 2) array of pixel coordinates tracing the
    perturbed isophote shape.  The harmonic perturbation follows the
    convention used in ``model.py``:

        r_draw(E) = sma * (1 + sum_n [a_n sin(nE) + b_n cos(nE)])

    where E is the eccentric anomaly (or position angle, depending on
    the ``use_eccentric_anomaly`` flag stored in the isophote dict).

    Parameters
    ----------
    iso : dict
        Single isophote result dict with geometry keys and harmonic
        coefficients (a3, b3, a4, b4, ...).
    n_points : int
        Number of sample points around the contour.
    """
    sma = float(iso["sma"])
    eps = float(iso.get("eps", 0.0))
    pa_rad = float(iso.get("pa", 0.0))
    x0 = float(iso.get("x0", 0.0))
    y0 = float(iso.get("y0", 0.0))

    orders = _detect_harmonic_orders(iso)

    # Parametric angle: eccentric anomaly E in [0, 2pi)
    angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)

    # Harmonic perturbation: dr/r = sum_n [a_n sin(nE) + b_n cos(nE)]
    dr_over_r = np.zeros(n_points)
    for n in orders:
        an = float(iso.get(f"a{n}", 0.0))
        bn = float(iso.get(f"b{n}", 0.0))
        if not (np.isfinite(an) and np.isfinite(bn)):
            continue
        dr_over_r += an * np.sin(n * angles) + bn * np.cos(n * angles)

    scale = 1.0 + dr_over_r

    # Parametric ellipse in rotated frame, scaled by harmonic perturbation
    x_rot = sma * scale * np.cos(angles)
    y_rot = sma * (1.0 - eps) * scale * np.sin(angles)

    # Rotate by PA and translate to center
    cos_pa = np.cos(pa_rad)
    sin_pa = np.sin(pa_rad)
    x_pix = x0 + x_rot * cos_pa - y_rot * sin_pa
    y_pix = y0 + x_rot * sin_pa + y_rot * cos_pa

    return np.column_stack([x_pix, y_pix])


def draw_isophote_overlays(
    axis,
    isophotes,
    step: int = 10,
    line_width: float = 1.0,
    alpha: float = 0.7,
    edge_color: str | None = None,
    draw_harmonics: bool = True,
) -> None:
    """Overlay selective isophotes on an image axis.

    When ``draw_harmonics`` is True and harmonic coefficients (a3/b3,
    a4/b4, ...) are present in the isophote dicts, the overlays show
    the actual non-elliptical isophote shape.  Otherwise pure ellipses
    are drawn.

    Parameters
    ----------
    isophotes : list of dict
        Isophote results with keys: sma, x0, y0, eps, pa, stop_code.
    step : int
        Draw every *step*-th isophote.
    edge_color : str or None
        Fixed color override; if None, color by stop code.
    draw_harmonics : bool
        If True (default), incorporate harmonic deviations when
        available.  Set to False to force pure-ellipse overlays.
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

        has_harmonics = draw_harmonics and len(_detect_harmonic_orders(iso)) > 0

        if has_harmonics:
            contour = _compute_harmonic_contour(iso)
            patch = MPLPolygon(
                contour,
                closed=True,
                fill=False,
                linewidth=line_width,
                alpha=alpha,
                edgecolor=color,
            )
        else:
            patch = MPLEllipse(
                (x0, y0),
                2.0 * sma,
                2.0 * sma * (1.0 - eps),
                angle=np.rad2deg(pa_rad),
                fill=False,
                linewidth=line_width,
                alpha=alpha,
                edgecolor=color,
            )
        axis.add_patch(patch)


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
        key=lambda code: (0, 0) if code == 0 else (1, code) if code > 0 else (2, abs(code)),
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
    sb_zeropoint=None,
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
    x_axis = i_sma**0.25
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
        1,
        2,
        figure=fig,
        width_ratios=[1.0, 2.01],
        wspace=0.27,
    )

    # Left column: 3 image rows, each with a colorbar sliver
    left = gridspec.GridSpecFromSubplotSpec(
        3,
        2,
        subplot_spec=outer[0],
        width_ratios=[1.0, 0.04],
        hspace=0.10,
        wspace=-0.20,
    )

    # Right column: 1-D profile panels
    right = gridspec.GridSpecFromSubplotSpec(
        n_panels,
        1,
        subplot_spec=outer[1],
        height_ratios=height_ratios,
        hspace=0.0,
    )

    fig.suptitle(title, fontsize=20, y=0.989)

    # --- Left column: 2-D panels ----------------------------------------------
    # Shared arcsinh parameters derived from the data image
    ref_low, ref_high, ref_scale, ref_vmax = derive_arcsinh_parameters(image)

    # Panel 1: Data + isophotes
    ax_img = fig.add_subplot(left[0, 0])
    img_display, img_vmin, img_vmax = make_arcsinh_display_from_parameters(
        image,
        low=ref_low,
        high=ref_high,
        scale=ref_scale,
        vmax=ref_vmax,
    )
    h_img = ax_img.imshow(
        img_display,
        origin="lower",
        cmap="viridis",
        vmin=img_vmin,
        vmax=img_vmax,
        interpolation="none",
    )
    if mask is not None:
        mask_overlay = np.zeros((*image.shape, 4))
        mask_overlay[mask] = [1, 0, 0, 0.4]
        ax_img.imshow(mask_overlay, origin="lower")

    overlay_step = max(1, len(isoster_res) // 15)
    draw_isophote_overlays(
        ax_img,
        isoster_res,
        step=overlay_step,
        line_width=1.2,
        alpha=0.8,
    )
    ax_img.text(
        0.15,
        0.9,
        "Data",
        fontsize=18,
        color="w",
        transform=ax_img.transAxes,
        ha="center",
        va="center",
        weight="bold",
        alpha=0.85,
    )

    ax_img_cb = fig.add_subplot(left[0, 1])
    fig.colorbar(h_img, cax=ax_img_cb).set_label(r"arcsinh((data $-$ p0.5) / scale)")

    # Panel 2: Model
    ax_mod = fig.add_subplot(left[1, 0])
    mod_display, mod_vmin, mod_vmax = make_arcsinh_display_from_parameters(
        isoster_model,
        low=ref_low,
        high=ref_high,
        scale=ref_scale,
        vmax=ref_vmax,
    )
    h_mod = ax_mod.imshow(
        mod_display,
        origin="lower",
        cmap="viridis",
        vmin=mod_vmin,
        vmax=mod_vmax,
        interpolation="none",
    )
    ax_mod.text(
        0.15,
        0.9,
        "Model",
        fontsize=18,
        color="w",
        transform=ax_mod.transAxes,
        ha="center",
        va="center",
        weight="bold",
        alpha=0.85,
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
        residual,
        origin="lower",
        cmap="coolwarm",
        vmin=-res_limit,
        vmax=res_limit,
        interpolation="nearest",
    )
    ax_res.text(
        0.18,
        0.9,
        "Residual",
        fontsize=18,
        color="k",
        transform=ax_res.transAxes,
        ha="center",
        va="center",
        weight="bold",
        alpha=0.85,
    )

    ax_res_cb = fig.add_subplot(left[2, 1])
    fig.colorbar(h_res, cax=ax_res_cb).set_label(res_label)

    for ax in [ax_img, ax_mod, ax_res]:
        ax.set_xlabel("x [pixel]")
        ax.set_ylabel("y [pixel]")

    # --- Right column: 1-D profiles -------------------------------------------
    panel_idx = 0

    # 1. Surface brightness
    ax_sb = fig.add_subplot(right[panel_idx])
    panel_idx += 1

    sb_valid = valid_mask & np.isfinite(i_intens) & (i_intens > 0)
    y_sb = np.full_like(i_intens, np.nan)
    y_sb_err = np.full_like(i_intens_err, np.nan)
    if sb_zeropoint is not None:
        # Surface brightness: μ = -2.5 * log10(I) + zp
        y_sb[sb_valid] = -2.5 * np.log10(i_intens[sb_valid]) + sb_zeropoint
        y_sb_err[sb_valid] = 2.5 * i_intens_err[sb_valid] / (i_intens[sb_valid] * np.log(10))
        sb_ylabel = r"$\mu$ [mag/arcsec$^2$]"
    else:
        y_sb[sb_valid] = np.log10(i_intens[sb_valid])
        y_sb_err[sb_valid] = i_intens_err[sb_valid] / (i_intens[sb_valid] * np.log(10))
        sb_ylabel = r"$\log_{10}(I)$"

    plot_profile_by_stop_code(
        ax_sb,
        x_axis[sb_valid],
        y_sb[sb_valid],
        i_stop[sb_valid],
        y_errors=y_sb_err[sb_valid],
        marker_face="filled",
        monochrome=True,
    )

    if photutils_res is not None:
        _overlay_photutils_sb(ax_sb, photutils_res)

    ax_sb.set_ylabel(sb_ylabel)
    ax_sb.set_title("Surface brightness profile")
    ax_sb.grid(alpha=0.25)
    if sb_zeropoint is not None:
        ax_sb.invert_yaxis()
    sb_for_limits = y_sb[sb_valid & np.isfinite(y_sb)]
    if sb_for_limits.size > 0:
        set_axis_limits_from_finite_values(
            ax_sb,
            sb_for_limits,
            margin_fraction=0.06,
            min_margin=0.2,
            invert=sb_zeropoint is not None,
        )
    handles, labels = ax_sb.get_legend_handles_labels()
    if handles:
        ax_sb.legend(handles[:8], labels[:8], loc="upper right", fontsize=14, ncol=1)

    # 2. Center offset
    ax_cen = fig.add_subplot(right[panel_idx], sharex=ax_sb)
    panel_idx += 1

    plot_profile_by_stop_code(
        ax_cen,
        x_axis[valid_mask],
        i_dx[valid_mask],
        i_stop[valid_mask],
        y_errors=i_x0_err[valid_mask],
        marker_face="filled",
        label_prefix="dx ",
        monochrome=True,
    )
    plot_profile_by_stop_code(
        ax_cen,
        x_axis[valid_mask],
        i_dy[valid_mask],
        i_stop[valid_mask],
        y_errors=i_y0_err[valid_mask],
        marker_face="open",
        label_prefix="dy ",
        monochrome=True,
    )
    ax_cen.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_cen.set_ylabel("center offset [pix]")
    ax_cen.grid(alpha=0.25)
    centroid_legend = [
        Line2D([], [], marker="o", linestyle="None", color="black", markerfacecolor="black", markersize=4.8, label="X"),
        Line2D([], [], marker="o", linestyle="None", color="black", markerfacecolor="none", markersize=4.8, label="Y"),
    ]
    ax_cen.legend(handles=centroid_legend, loc="upper right", frameon=True, fontsize=14, ncol=2)
    # Y-limits from data values only (exclude errorbars to preserve detail)
    cen_vals = np.concatenate([i_dx[valid_mask], i_dy[valid_mask]])
    set_axis_limits_from_finite_values(ax_cen, cen_vals, margin_fraction=0.08, min_margin=0.3)

    # 3. Axis ratio (b/a)
    ax_ba = fig.add_subplot(right[panel_idx], sharex=ax_sb)
    panel_idx += 1

    plot_profile_by_stop_code(
        ax_ba,
        x_axis[valid_mask],
        i_ba[valid_mask],
        i_stop[valid_mask],
        y_errors=i_eps_err[valid_mask],
        marker_face="filled",
        monochrome=True,
    )
    ba_for_limits = i_ba[valid_mask & np.isfinite(i_ba)]
    set_axis_limits_from_finite_values(
        ax_ba,
        ba_for_limits,
        margin_fraction=0.08,
        min_margin=0.03,
        lower_clip=0.0,
        upper_clip=1.0,
    )
    ax_ba.set_ylabel("axis ratio")
    ax_ba.grid(alpha=0.25)

    # 4. Position angle
    ax_pa = fig.add_subplot(right[panel_idx], sharex=ax_sb)
    panel_idx += 1

    plot_profile_by_stop_code(
        ax_pa,
        x_axis[valid_mask],
        i_pa_deg[valid_mask],
        i_stop[valid_mask],
        y_errors=i_pa_err_deg[valid_mask],
        marker_face="filled",
        monochrome=True,
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
            x_axis[valid_mask],
            i_a3[valid_mask],
            s=20,
            marker="o",
            facecolors="#1f77b4",
            edgecolors="#1f77b4",
            alpha=0.7,
            label="A3",
        )
        ax_harm.scatter(
            x_axis[valid_mask],
            i_b3[valid_mask],
            s=20,
            marker="s",
            facecolors="none",
            edgecolors="#1f77b4",
            alpha=0.7,
            label="B3",
        )
        ax_harm.scatter(
            x_axis[valid_mask],
            i_a4[valid_mask],
            s=20,
            marker="^",
            facecolors="#d62728",
            edgecolors="#d62728",
            alpha=0.7,
            label="A4",
        )
        ax_harm.scatter(
            x_axis[valid_mask],
            i_b4[valid_mask],
            s=20,
            marker="D",
            facecolors="none",
            edgecolors="#d62728",
            alpha=0.7,
            label="B4",
        )
        harm_values = np.concatenate(
            [
                i_a3[valid_mask],
                i_b3[valid_mask],
                i_a4[valid_mask],
                i_b4[valid_mask],
            ]
        )
        set_axis_limits_from_finite_values(
            ax_harm,
            harm_values,
            margin_fraction=0.10,
            min_margin=0.005,
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
            ax_cog,
            x_axis[valid_mask],
            cog_values[valid_mask],
            i_stop[valid_mask],
            marker_face="filled",
            label_prefix="fit ",
            monochrome=True,
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
    sb_zeropoint=None,
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

    x_axis = i_sma**0.25
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
        1,
        2,
        figure=fig,
        width_ratios=[1.0, 2.01],
        wspace=0.27,
    )
    left = gridspec.GridSpecFromSubplotSpec(
        3,
        2,
        subplot_spec=outer[0],
        width_ratios=[1.0, 0.04],
        hspace=0.10,
        wspace=-0.20,
    )
    right = gridspec.GridSpecFromSubplotSpec(
        n_panels,
        1,
        subplot_spec=outer[1],
        height_ratios=height_ratios,
        hspace=0.0,
    )
    fig.suptitle(title, fontsize=20, y=0.989)

    # --- Left panels: data / model / residual ---------------------------------
    ref_low, ref_high, ref_scale, ref_vmax = derive_arcsinh_parameters(image)

    ax_img = fig.add_subplot(left[0, 0])
    img_display, img_vmin, img_vmax = make_arcsinh_display_from_parameters(
        image,
        low=ref_low,
        high=ref_high,
        scale=ref_scale,
        vmax=ref_vmax,
    )
    h_img = ax_img.imshow(
        img_display,
        origin="lower",
        cmap="viridis",
        vmin=img_vmin,
        vmax=img_vmax,
        interpolation="none",
    )
    if mask is not None:
        mask_overlay = np.zeros((*image.shape, 4))
        mask_overlay[mask] = [1, 0, 0, 0.4]
        ax_img.imshow(mask_overlay, origin="lower")
    overlay_step = max(1, len(isoster_res) // 15)
    draw_isophote_overlays(
        ax_img,
        isoster_res,
        step=overlay_step,
        line_width=1.2,
        alpha=0.8,
    )
    ax_img.text(
        0.15,
        0.9,
        "Data",
        fontsize=18,
        color="w",
        transform=ax_img.transAxes,
        ha="center",
        va="center",
        weight="bold",
        alpha=0.85,
    )
    ax_img_cb = fig.add_subplot(left[0, 1])
    fig.colorbar(h_img, cax=ax_img_cb).set_label(r"arcsinh((data $-$ p0.5) / scale)")

    ax_mod = fig.add_subplot(left[1, 0])
    mod_display, mod_vmin, mod_vmax = make_arcsinh_display_from_parameters(
        isoster_model,
        low=ref_low,
        high=ref_high,
        scale=ref_scale,
        vmax=ref_vmax,
    )
    h_mod = ax_mod.imshow(
        mod_display,
        origin="lower",
        cmap="viridis",
        vmin=mod_vmin,
        vmax=mod_vmax,
        interpolation="none",
    )
    ax_mod.text(
        0.15,
        0.9,
        "Model",
        fontsize=18,
        color="w",
        transform=ax_mod.transAxes,
        ha="center",
        va="center",
        weight="bold",
        alpha=0.85,
    )
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
    res_limit = float(
        np.clip(
            np.nanpercentile(abs_res, 99.0) if abs_res.size else 1.0,
            0.05,
            None,
        )
    )
    h_res = ax_res.imshow(
        residual_map,
        origin="lower",
        cmap="coolwarm",
        vmin=-res_limit,
        vmax=res_limit,
        interpolation="nearest",
    )
    ax_res.text(
        0.18,
        0.9,
        "Residual",
        fontsize=18,
        color="k",
        transform=ax_res.transAxes,
        ha="center",
        va="center",
        weight="bold",
        alpha=0.85,
    )
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
    y_sb_err = np.full_like(i_intens_err, np.nan)
    if sb_zeropoint is not None:
        y_sb[sb_valid] = -2.5 * np.log10(i_intens[sb_valid]) + sb_zeropoint
        y_sb_err[sb_valid] = 2.5 * i_intens_err[sb_valid] / (i_intens[sb_valid] * np.log(10))
        sb_ylabel = r"$\mu$ [mag/arcsec$^2$]"
    else:
        y_sb[sb_valid] = np.log10(i_intens[sb_valid])
        y_sb_err[sb_valid] = i_intens_err[sb_valid] / (i_intens[sb_valid] * np.log(10))
        sb_ylabel = r"$\log_{10}(I)$"
    plot_profile_by_stop_code(
        ax_sb,
        x_axis[sb_valid],
        y_sb[sb_valid],
        i_stop[sb_valid],
        y_errors=y_sb_err[sb_valid],
        marker_face="filled",
        monochrome=True,
    )
    ax_sb.set_ylabel(sb_ylabel)
    ax_sb.set_title("Surface brightness profile")
    ax_sb.grid(alpha=0.25)
    if sb_zeropoint is not None:
        ax_sb.invert_yaxis()
    sb_for_limits = y_sb[sb_valid & np.isfinite(y_sb)]
    if sb_for_limits.size > 0:
        set_axis_limits_from_finite_values(
            ax_sb, sb_for_limits, margin_fraction=0.06, min_margin=0.2,
            invert=sb_zeropoint is not None,
        )
    handles, labels = ax_sb.get_legend_handles_labels()
    if handles:
        ax_sb.legend(handles[:8], labels[:8], loc="upper right", fontsize=12, ncol=1)

    # 2. Center offset
    ax_cen = fig.add_subplot(right[panel_idx], sharex=ax_sb)
    panel_idx += 1
    plot_profile_by_stop_code(
        ax_cen,
        x_axis[valid_mask],
        i_dx[valid_mask],
        i_stop[valid_mask],
        y_errors=i_x0_err[valid_mask],
        marker_face="filled",
        label_prefix="dx ",
        monochrome=True,
    )
    plot_profile_by_stop_code(
        ax_cen,
        x_axis[valid_mask],
        i_dy[valid_mask],
        i_stop[valid_mask],
        y_errors=i_y0_err[valid_mask],
        marker_face="open",
        label_prefix="dy ",
        monochrome=True,
    )
    ax_cen.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_cen.set_ylabel("center [pix]")
    ax_cen.grid(alpha=0.25)
    # Y-limits from data values only (exclude errorbars to preserve detail)
    cen_vals = np.concatenate([i_dx[valid_mask], i_dy[valid_mask]])
    set_axis_limits_from_finite_values(ax_cen, cen_vals, margin_fraction=0.08, min_margin=0.3)

    # 3. Axis ratio
    ax_ba = fig.add_subplot(right[panel_idx], sharex=ax_sb)
    panel_idx += 1
    plot_profile_by_stop_code(
        ax_ba,
        x_axis[valid_mask],
        i_ba[valid_mask],
        i_stop[valid_mask],
        y_errors=i_eps_err[valid_mask],
        marker_face="filled",
        monochrome=True,
    )
    ba_for_limits = i_ba[valid_mask & np.isfinite(i_ba)]
    set_axis_limits_from_finite_values(
        ax_ba,
        ba_for_limits,
        margin_fraction=0.08,
        min_margin=0.03,
        lower_clip=0.0,
        upper_clip=1.0,
    )
    ax_ba.set_ylabel("axis ratio")
    ax_ba.grid(alpha=0.25)

    # 4. PA
    ax_pa = fig.add_subplot(right[panel_idx], sharex=ax_sb)
    panel_idx += 1
    plot_profile_by_stop_code(
        ax_pa,
        x_axis[valid_mask],
        i_pa_deg[valid_mask],
        i_stop[valid_mask],
        y_errors=i_pa_err_deg[valid_mask],
        marker_face="filled",
        monochrome=True,
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
                amplitude = np.sqrt(np.where(np.isfinite(an), an, 0.0) ** 2 + np.where(np.isfinite(bn), bn, 0.0) ** 2)
                amplitude[~(np.isfinite(an) & np.isfinite(bn))] = np.nan
                if normalize_harmonics:
                    amplitude = amplitude / safe_intens
                show = valid_mask & np.isfinite(amplitude)
                if np.any(show):
                    ax.scatter(
                        x_axis[show],
                        amplitude[show],
                        s=18,
                        marker=mrk,
                        facecolors=col,
                        edgecolors=col,
                        alpha=0.75,
                        label=f"$A_{{{order}}}$",
                    )
                    all_values.append(amplitude[show])
            else:
                # Coefficients mode: a_n (filled) and b_n (open)
                show_a = valid_mask & np.isfinite(an)
                show_b = valid_mask & np.isfinite(bn)
                if np.any(show_a):
                    ax.scatter(
                        x_axis[show_a],
                        an[show_a],
                        s=18,
                        marker=mrk,
                        facecolors=col,
                        edgecolors=col,
                        alpha=0.75,
                        label=f"$a_{{{order}}}$",
                    )
                    all_values.append(an[show_a])
                if np.any(show_b):
                    ax.scatter(
                        x_axis[show_b],
                        bn[show_b],
                        s=18,
                        marker=mrk,
                        facecolors="none",
                        edgecolors=col,
                        linewidths=0.9,
                        alpha=0.75,
                        label=f"$b_{{{order}}}$",
                    )
                    all_values.append(bn[show_b])

        ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        if all_values:
            cat = np.concatenate(all_values)
            set_axis_limits_from_finite_values(
                ax,
                cat,
                margin_fraction=0.12,
                min_margin=0.001,
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
        ax_odd.set_ylabel("$a_n, b_n$ (odd)")
        ax_odd.text(
            0.5, 0.5, "no odd orders", color="gray", fontsize=10, transform=ax_odd.transAxes, ha="center", va="center"
        )
        ax_odd.grid(alpha=0.25)

    # 6. Even harmonics panel
    ax_even = fig.add_subplot(right[panel_idx], sharex=ax_sb)
    panel_idx += 1
    if even_orders:
        _plot_harmonic_panel(ax_even, even_orders, "even")
    else:
        ax_even.set_ylabel("$a_n, b_n$ (even)")
        ax_even.text(
            0.5, 0.5, "no even orders", color="gray", fontsize=10, transform=ax_even.transAxes, ha="center", va="center"
        )
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


# ---------------------------------------------------------------------------
# Multi-method comparison QA figure
# ---------------------------------------------------------------------------


def _scatter_by_stop_code_in_method_color(
    axis,
    x: np.ndarray,
    y: np.ndarray,
    stop_codes: np.ndarray,
    base_color: str,
    y_errors: np.ndarray | None = None,
    label: str = "",
    marker_size: float = 22.0,
    label_stop_codes: bool = True,
) -> None:
    """Scatter points using stop-code markers in a single method color.

    Stop=0 points are filled; non-zero stop codes use open markers with
    the same base color.  This distinguishes convergence quality while
    keeping method identity clear via color.

    Parameters
    ----------
    label_stop_codes : bool
        If True, each stop code gets its own legend entry (e.g.
        "isoster", "isoster (stop=2)").  If False, only the first
        group (stop=0) gets the method label; non-zero groups are
        plotted but unlabeled in the legend.
    """
    unique_codes = sorted(
        {int(c) for c in stop_codes[np.isfinite(stop_codes)]},
        key=lambda c: (0, 0) if c == 0 else (1, c) if c > 0 else (2, abs(c)),
    )
    for code in unique_codes:
        mask = stop_codes == code
        if not np.any(mask):
            continue
        marker = MONOCHROME_STOP_MARKERS.get(code, "o")
        face = base_color if code == 0 else "none"
        if label_stop_codes:
            code_label = f"{label} (stop={code})" if code != 0 else label
        else:
            # Only label the first group (stop=0) with the method name
            code_label = label if code == 0 else None

        if y_errors is not None:
            errs = np.asarray(y_errors[mask], dtype=float)
            errs[errs < 0] = np.nan
            axis.errorbar(
                x[mask],
                y[mask],
                yerr=errs,
                fmt=marker,
                color=base_color,
                mfc=face,
                mec=base_color,
                markersize=4.2,
                capsize=1.5,
                linewidth=0.6,
                alpha=0.8,
                label=code_label,
                zorder=3,
            )
        else:
            axis.scatter(
                x[mask],
                y[mask],
                s=marker_size,
                marker=marker,
                facecolors=face,
                edgecolors=base_color,
                linewidths=0.9,
                alpha=0.8,
                label=code_label,
                zorder=3,
            )


def _build_isos_for_overlay(prof: dict[str, np.ndarray]) -> list[dict]:
    """Reconstruct minimal isophote dicts from profile arrays for overlay.

    Includes harmonic coefficients (a3, b3, a4, b4, ...) when present
    so that ``draw_isophote_overlays`` can render the perturbed shape.
    """
    if "x0" not in prof or "y0" not in prof:
        return []
    n = len(prof["sma"])
    # Detect harmonic keys in the profile dict
    harm_keys = [k for k in prof if len(k) >= 2 and k[0] in ("a", "b") and k[1:].isdigit() and int(k[1:]) >= 3]
    isos = []
    for i in range(n):
        iso = {
            "sma": float(prof["sma"][i]),
            "x0": float(prof["x0"][i]),
            "y0": float(prof["y0"][i]),
            "eps": float(prof["eps"][i]),
            "pa": float(prof["pa"][i]),
            "stop_code": int(prof["stop_codes"][i]) if "stop_codes" in prof else 0,
        }
        for k in harm_keys:
            iso[k] = float(prof[k][i])
        isos.append(iso)
    return isos


def _compute_residual_map(
    image: np.ndarray,
    model: np.ndarray,
    relative: bool = False,
) -> tuple[np.ndarray, str]:
    """Compute residual map and its colorbar label.

    Parameters
    ----------
    relative : bool
        If True, return ``100 * (model - data) / data`` (fractional %).
        If False (default), return ``data - model`` (absolute).
    """
    if relative:
        residual = compute_fractional_residual_percent(image, model)
        label = "(model-data)/data [%]"
    else:
        residual = np.where(np.isfinite(image), image - model, np.nan)
        label = "data - model"
    return residual, label


def _plot_residual_panel(
    fig,
    gs_slot,
    gs_cbar_slot,
    image: np.ndarray,
    model: np.ndarray,
    panel_title: str,
    relative_residual: bool = False,
    overlay_isos: list[dict] | None = None,
    overlay_step: int = 5,
    overlay_color: str = "white",
    overlay_width: float = 1.0,
) -> None:
    """Draw a single residual image panel with optional isophote overlays."""
    residual, cbar_label = _compute_residual_map(
        image,
        model,
        relative=relative_residual,
    )
    abs_vals = np.abs(residual[np.isfinite(residual)])
    res_limit = float(
        np.clip(
            np.nanpercentile(abs_vals, 99.0) if abs_vals.size else 1.0,
            0.05,
            8.0 if relative_residual else None,
        )
    )

    ax = fig.add_subplot(gs_slot)
    handle = ax.imshow(
        residual,
        origin="lower",
        cmap="coolwarm",
        vmin=-res_limit,
        vmax=res_limit,
        interpolation="nearest",
    )
    if overlay_isos:
        draw_isophote_overlays(
            ax,
            overlay_isos,
            step=overlay_step,
            line_width=overlay_width,
            alpha=0.7,
            edge_color=overlay_color,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.03,
        0.95,
        panel_title,
        color="black",
        fontsize=11,
        fontweight="bold",
        transform=ax.transAxes,
        va="top",
        ha="left",
    )

    ax_cbar = fig.add_subplot(gs_cbar_slot)
    fig.colorbar(handle, cax=ax_cbar).set_label(cbar_label, fontsize=8)


def plot_comparison_qa_figure(
    image: np.ndarray,
    profiles: dict[str, dict[str, np.ndarray]],
    title: str = "",
    output_path: str | Path = "qa_comparison.png",
    *,
    models: dict[str, np.ndarray] | None = None,
    mask: np.ndarray | None = None,
    method_styles: dict[str, dict] | None = None,
    relative_residual: bool = False,
    dpi: int = 150,
) -> None:
    """Multi-method comparison QA figure with 2D images and 1D profiles.

    Automatically selects one of three layout modes based on the number
    of methods that provide profiles:

    **Mode 1 — Solo (isoster only):**
    Left column: original image with isophote overlays (+ mask overlay),
    reconstructed 2D model, residual.

    **Mode 2 — One-on-one (isoster + one other):**
    Left column: original image (+ mask), isoster residual with isophote
    overlays, other method's residual with its isophote overlays.

    **Mode 3 — Three-way (isoster + both):**
    Left column: original image (+ mask), isoster residual + overlays,
    photutils residual + overlays, autoprof residual + overlays.

    Right column (all modes): SB (with errorbars), relative SB diff,
    ellipticity, PA, centroid offset — sharing SMA^0.25 x-axis.

    Parameters
    ----------
    image : 2D ndarray
        Original galaxy image.
    profiles : dict[str, dict[str, ndarray]]
        Method name -> profile arrays dict (from ``build_method_profile``).
        Required keys per profile: sma, x_axis, intens, eps, pa.
        Optional: stop_codes, x0, y0, intens_err, rms, runtime_seconds.
    title : str
        Figure title.
    output_path : str or Path
        Output file path for the saved figure.
    models : dict[str, 2D ndarray] or None
        Method name -> 2D reconstructed model.
    mask : 2D bool ndarray or None
        Bad-pixel mask (True = masked).  Overlaid on the data image panel.
    method_styles : dict or None
        Override default ``METHOD_STYLES``.
    relative_residual : bool
        If True, residual panels show ``(model - data) / data`` [%].
        If False (default), show ``data - model`` (absolute).
    dpi : int
        Figure resolution.
    """
    from pathlib import Path as _Path

    import matplotlib.gridspec as gridspec

    configure_qa_plot_style()
    plt.rcParams["text.usetex"] = False

    styles = dict(METHOD_STYLES)
    if method_styles is not None:
        styles.update(method_styles)

    if models is None:
        models = {}

    available = [m for m in profiles if profiles[m] is not None]
    n_methods = len(available)

    # --- Determine layout mode ---
    # Mode 1 (solo): 3 left rows (image, model, residual)
    # Mode 2 (one-on-one): 3 left rows (image, isoster residual, other residual)
    # Mode 3 (three-way): 4 left rows (image, isoster res, phot res, autoprof res)
    if n_methods <= 1:
        n_left_rows = 3  # image, model, residual
    elif n_methods == 2:
        n_left_rows = 3  # image, 2x residual
    else:
        n_left_rows = 1 + n_methods  # image + one residual per method

    n_right_rows = 5
    fig_height = max(12, n_left_rows * 3.2)

    fig = plt.figure(figsize=(15, fig_height), dpi=dpi)
    outer = gridspec.GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1.0, 1.8],
        wspace=0.30,
        top=0.95,
    )
    left = gridspec.GridSpecFromSubplotSpec(
        n_left_rows,
        2,
        subplot_spec=outer[0],
        width_ratios=[1.0, 0.04],
        hspace=0.12,
        wspace=0.02,
    )
    right = gridspec.GridSpecFromSubplotSpec(
        n_right_rows,
        1,
        subplot_spec=outer[1],
        height_ratios=[2.5, 1.0, 1.0, 1.0, 1.0],
        hspace=0.0,
    )

    # --- Title (runtime shown in SB panel instead) ---
    fig.suptitle(title, fontsize=16, y=0.975)

    # Collect runtime lines for later display in SB panel
    _runtime_lines = []
    for method_name in available:
        prof = profiles[method_name]
        style = styles.get(method_name, {"label": method_name})
        label = style.get("label", method_name)
        color = style.get("color", "black")
        rt = prof.get("runtime_seconds")
        if rt is not None:
            line = f"{label}: {float(rt):.2f}s"
            retries = prof.get("retries", 0)
            if retries > 0:
                line += f"; retry: {retries}"
            _runtime_lines.append((line, color))

    # --- Left column row 0: Original image ---
    ax_img = fig.add_subplot(left[0, 0])
    low, high, scale, vmax_val = derive_arcsinh_parameters(image)
    display, _, disp_vmax = make_arcsinh_display_from_parameters(
        image,
        low,
        high,
        scale,
        vmax_val,
    )
    handle_img = ax_img.imshow(
        display,
        cmap="viridis",
        origin="lower",
        vmin=0,
        vmax=disp_vmax,
        interpolation="none",
    )

    # Mask overlay (semi-transparent red)
    if mask is not None:
        mask_overlay = np.zeros((*image.shape, 4))
        mask_overlay[mask] = [1, 0, 0, 0.4]
        ax_img.imshow(mask_overlay, origin="lower")

    ax_cbar = fig.add_subplot(left[0, 1])
    fig.colorbar(handle_img, cax=ax_cbar)

    if n_methods <= 1:
        # Solo mode: overlays from isoster on the data image
        for method_name in available:
            prof = profiles[method_name]
            style = styles.get(method_name, {})
            isos = _build_isos_for_overlay(prof)
            if isos:
                overlay_step = max(1, len(isos) // 15)
                draw_isophote_overlays(
                    ax_img,
                    isos,
                    step=overlay_step,
                    line_width=style.get("overlay_width", 1.0),
                    alpha=0.8,
                    edge_color=style.get("overlay_color", "white"),
                )

    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.text(
        0.03,
        0.95,
        "Data",
        color="white",
        fontsize=11,
        fontweight="bold",
        transform=ax_img.transAxes,
        va="top",
        ha="left",
    )

    # Scale bar: ~1/10 of image size, rounded up to nearest 10
    img_size = max(image.shape)
    bar_length = int(np.ceil(img_size / 100.0)) * 10  # round up to 10s
    bar_x0 = image.shape[1] * 0.05
    bar_y0 = image.shape[0] * 0.05
    ax_img.plot(
        [bar_x0, bar_x0 + bar_length],
        [bar_y0, bar_y0],
        color="white",
        linewidth=2.5,
        solid_capstyle="butt",
    )
    ax_img.text(
        bar_x0 + bar_length / 2,
        bar_y0 + image.shape[0] * 0.03,
        f"{bar_length} pix",
        color="white",
        fontsize=9,
        ha="center",
        va="bottom",
    )

    # --- Left column: mode-dependent panels ---
    if n_methods <= 1 and available:
        # Mode 1 (solo): row 1 = model, row 2 = residual
        method_name = available[0]
        model = models.get(method_name)

        if model is not None:
            # Model panel
            ax_mod = fig.add_subplot(left[1, 0])
            mod_display, _, mod_vmax = make_arcsinh_display_from_parameters(
                model,
                low,
                high,
                scale,
                vmax_val,
            )
            h_mod = ax_mod.imshow(
                mod_display,
                origin="lower",
                cmap="viridis",
                vmin=0,
                vmax=mod_vmax,
                interpolation="none",
            )
            ax_mod.set_xticks([])
            ax_mod.set_yticks([])
            ax_mod.text(
                0.03,
                0.95,
                "Model",
                color="white",
                fontsize=11,
                fontweight="bold",
                transform=ax_mod.transAxes,
                va="top",
                ha="left",
            )
            ax_mod_cb = fig.add_subplot(left[1, 1])
            fig.colorbar(h_mod, cax=ax_mod_cb)

            # Residual panel
            _plot_residual_panel(
                fig,
                left[2, 0],
                left[2, 1],
                image,
                model,
                panel_title="Residual",
                relative_residual=relative_residual,
            )
        else:
            # No model available — leave panels blank
            for row_idx in (1, 2):
                ax_blank = fig.add_subplot(left[row_idx, 0])
                ax_blank.text(
                    0.5,
                    0.5,
                    "no model",
                    color="gray",
                    fontsize=12,
                    transform=ax_blank.transAxes,
                    ha="center",
                    va="center",
                )
                ax_blank.set_axis_off()

    elif n_methods >= 2:
        # Mode 2/3: one residual panel per method with isophote overlays
        for row_idx, method_name in enumerate(available, start=1):
            model = models.get(method_name)
            style = styles.get(method_name, {})
            label = style.get("label", method_name)
            isos = _build_isos_for_overlay(profiles[method_name])
            overlay_step = max(1, len(isos) // 15) if isos else 5

            if model is not None:
                _plot_residual_panel(
                    fig,
                    left[row_idx, 0],
                    left[row_idx, 1],
                    image,
                    model,
                    panel_title=f"{label} residual",
                    relative_residual=relative_residual,
                    overlay_isos=isos,
                    overlay_step=overlay_step,
                    overlay_color=style.get("overlay_color", "white"),
                    overlay_width=style.get("overlay_width", 1.0),
                )
            else:
                ax_blank = fig.add_subplot(left[row_idx, 0])
                ax_blank.text(
                    0.5,
                    0.5,
                    f"{label}: no model",
                    color="gray",
                    fontsize=12,
                    transform=ax_blank.transAxes,
                    ha="center",
                    va="center",
                )
                ax_blank.set_axis_off()

    elif not available:
        # No profiles at all — save bare figure
        output_path = _Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return

    # --- Right column: 1D profiles ---
    if not available:
        output_path = _Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return

    all_x = np.concatenate([profiles[m]["x_axis"] for m in available])

    # Panel 0: Surface brightness (with errorbars and stop-code markers)
    ax_sb = fig.add_subplot(right[0])
    for method_name in available:
        prof = profiles[method_name]
        style = styles.get(method_name, {})
        x = prof["x_axis"]
        intens = prof["intens"]
        valid = np.isfinite(intens) & (intens > 0)
        y = np.full_like(intens, np.nan)
        y[valid] = np.log10(intens[valid])

        # Error propagation: sigma_log10_I = intens_err / (I * ln(10))
        y_err = None
        if "intens_err" in prof:
            y_err = np.full_like(intens, np.nan)
            y_err[valid] = prof["intens_err"][valid] / (intens[valid] * np.log(10))

        if "stop_codes" in prof:
            _scatter_by_stop_code_in_method_color(
                ax_sb,
                x[valid],
                y[valid],
                prof["stop_codes"][valid],
                base_color=style["color"],
                y_errors=y_err[valid] if y_err is not None else None,
                label=style.get("label", method_name),
                marker_size=22,
                label_stop_codes=(n_methods <= 1),
            )
        else:
            mfc = style["color"] if style.get("marker_face") == "filled" else "none"
            if y_err is not None:
                ax_sb.errorbar(
                    x[valid],
                    y[valid],
                    yerr=y_err[valid],
                    fmt=style.get("marker", "o"),
                    color=style["color"],
                    mfc=mfc,
                    mec=style["color"],
                    markersize=4.2,
                    capsize=1.5,
                    linewidth=0.6,
                    alpha=0.7,
                    label=style.get("label", method_name),
                    zorder=3,
                )
            else:
                ax_sb.scatter(
                    x[valid],
                    y[valid],
                    color=style["color"],
                    marker=style.get("marker", "o"),
                    facecolors=mfc,
                    edgecolors=style["color"],
                    s=22,
                    alpha=0.7,
                    label=style.get("label", method_name),
                    zorder=3,
                )

    ax_sb.set_ylabel(r"$\log_{10}$(Intensity)")
    ax_sb.grid(alpha=0.25)
    ax_sb.legend(loc="upper right", fontsize=10)
    ax_sb.tick_params(labelbottom=False)
    set_x_limits_with_right_margin(ax_sb, all_x)
    # Data-driven Y-axis: range from data values, not error bars
    all_sb_vals = []
    for m in available:
        intens = profiles[m]["intens"]
        v = np.isfinite(intens) & (intens > 0)
        if np.any(v):
            all_sb_vals.append(np.log10(intens[v]))
    if all_sb_vals:
        all_sb = np.concatenate(all_sb_vals)
        set_axis_limits_from_finite_values(ax_sb, all_sb, invert=False)

    # Runtime annotation in bottom-left of SB panel
    if _runtime_lines:
        y_pos = 0.04
        for line_text, line_color in _runtime_lines:
            ax_sb.text(
                0.03,
                y_pos,
                line_text,
                color=line_color,
                fontsize=9,
                fontweight="bold",
                transform=ax_sb.transAxes,
                va="bottom",
                ha="left",
            )
            y_pos += 0.06

    # Panel 1: Relative SB difference (first method as reference)
    # Outliers beyond ±100% are shown as lower-limit markers (triangles)
    # and the Y-axis is capped at ±100% when such outliers exist.
    _DIFF_CAP = 100.0  # percent
    ax_diff = fig.add_subplot(right[1], sharex=ax_sb)
    has_outliers = False
    if n_methods >= 2:
        ref_method = available[0]
        ref_prof = profiles[ref_method]
        ref_sma = ref_prof["sma"]
        ref_intens = ref_prof["intens"]
        for method_name in available:
            if method_name == ref_method:
                continue
            prof = profiles[method_name]
            style = styles.get(method_name, {})
            from scipy.interpolate import interp1d

            valid_ref = np.isfinite(ref_intens) & (ref_intens > 0)
            if valid_ref.sum() < 2:
                continue
            interp_func = interp1d(
                ref_sma[valid_ref],
                ref_intens[valid_ref],
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            ref_at_method = interp_func(prof["sma"])
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_diff = 100.0 * (prof["intens"] - ref_at_method) / ref_at_method
            valid = np.isfinite(rel_diff)
            mfc = style["color"] if style.get("marker_face") == "filled" else "none"
            base_marker = style.get("marker", "o")

            # Split into normal points and outliers (|dI/I| > cap)
            inlier = valid & (np.abs(rel_diff) <= _DIFF_CAP)
            outlier = valid & (np.abs(rel_diff) > _DIFF_CAP)

            if np.any(outlier):
                has_outliers = True

            # Normal points
            ax_diff.scatter(
                prof["x_axis"][inlier],
                rel_diff[inlier],
                color=style["color"],
                marker=base_marker,
                facecolors=mfc,
                edgecolors=style["color"],
                s=18,
                alpha=0.7,
                label=style.get("label", method_name),
                zorder=3,
            )
            # Outlier points: clamp to ±cap, show as triangle markers
            if np.any(outlier):
                clamped = np.clip(rel_diff[outlier], -_DIFF_CAP, _DIFF_CAP)
                # Triangle pointing in the direction of the outlier
                outlier_markers = np.where(rel_diff[outlier] > 0, "^", "v")
                for om in ["^", "v"]:
                    om_mask = outlier_markers == om
                    if np.any(om_mask):
                        ax_diff.scatter(
                            prof["x_axis"][outlier][om_mask],
                            clamped[om_mask],
                            color=style["color"],
                            marker=om,
                            facecolors=style["color"],
                            edgecolors=style["color"],
                            s=30,
                            alpha=0.5,
                            zorder=4,
                        )

        ax_diff.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax_diff.set_ylabel("dI/I [%]")
        ax_diff.legend(loc="best", fontsize=9)

        # Cap Y-axis when outliers exist
        if has_outliers:
            ax_diff.set_ylim(-_DIFF_CAP * 1.08, _DIFF_CAP * 1.08)
    else:
        ax_diff.set_ylabel("dI/I [%]")
        ax_diff.text(
            0.5,
            0.5,
            "single method",
            color="gray",
            fontsize=10,
            transform=ax_diff.transAxes,
            ha="center",
            va="center",
        )
    ax_diff.grid(alpha=0.25)
    ax_diff.tick_params(labelbottom=False)

    # Panel 2: Ellipticity (with error bars when available)
    ax_eps = fig.add_subplot(right[2], sharex=ax_sb)
    for method_name in available:
        prof = profiles[method_name]
        style = styles.get(method_name, {})
        eps_err = prof.get("eps_err")
        mfc = style["color"] if style.get("marker_face") == "filled" else "none"
        if eps_err is not None:
            if "stop_codes" in prof:
                _scatter_by_stop_code_in_method_color(
                    ax_eps,
                    prof["x_axis"],
                    prof["eps"],
                    prof["stop_codes"],
                    base_color=style["color"],
                    y_errors=eps_err,
                    marker_size=18,
                    label_stop_codes=False,
                    label=style.get("label", method_name),
                )
            else:
                ax_eps.errorbar(
                    prof["x_axis"],
                    prof["eps"],
                    yerr=eps_err,
                    fmt=style.get("marker", "o"),
                    color=style["color"],
                    mfc=mfc,
                    mec=style["color"],
                    markersize=4.2,
                    capsize=1.5,
                    linewidth=0.6,
                    alpha=0.7,
                    zorder=3,
                )
        else:
            ax_eps.scatter(
                prof["x_axis"],
                prof["eps"],
                color=style["color"],
                marker=style.get("marker", "o"),
                facecolors=mfc,
                edgecolors=style["color"],
                s=18,
                alpha=0.7,
                zorder=3,
            )
    ax_eps.set_ylabel("Ellipticity")
    ax_eps.grid(alpha=0.25)
    ax_eps.tick_params(labelbottom=False)
    # Data-driven Y-axis: limits from data values only, not error bars
    all_eps = np.concatenate([profiles[m]["eps"] for m in available])
    set_axis_limits_from_finite_values(
        ax_eps,
        all_eps,
        margin_fraction=0.08,
        min_margin=0.03,
        lower_clip=0.0,
        upper_clip=1.0,
    )

    # Panel 3: PA (normalized with cross-method anchoring, with error bars)
    ax_pa = fig.add_subplot(right[3], sharex=ax_sb)
    all_pa_norm = []
    ref_pa_median = None
    for method_name in available:
        prof = profiles[method_name]
        style = styles.get(method_name, {})
        pa_deg = np.degrees(prof["pa"])

        # First method: normalize freely; subsequent methods: anchor to
        # the first method's median PA to avoid ~180 deg offsets
        pa_norm = normalize_pa_degrees(pa_deg, anchor=ref_pa_median)
        if ref_pa_median is None:
            finite_pa = pa_norm[np.isfinite(pa_norm)]
            if finite_pa.size > 0:
                ref_pa_median = float(np.nanmedian(finite_pa))

        all_pa_norm.append(pa_norm)
        pa_err_deg = None
        if "pa_err" in prof:
            pa_err_deg = np.degrees(prof["pa_err"])

        mfc = style["color"] if style.get("marker_face") == "filled" else "none"
        if pa_err_deg is not None:
            if "stop_codes" in prof:
                _scatter_by_stop_code_in_method_color(
                    ax_pa,
                    prof["x_axis"],
                    pa_norm,
                    prof["stop_codes"],
                    base_color=style["color"],
                    y_errors=pa_err_deg,
                    marker_size=18,
                    label_stop_codes=False,
                    label=style.get("label", method_name),
                )
            else:
                ax_pa.errorbar(
                    prof["x_axis"],
                    pa_norm,
                    yerr=pa_err_deg,
                    fmt=style.get("marker", "o"),
                    color=style["color"],
                    mfc=mfc,
                    mec=style["color"],
                    markersize=4.2,
                    capsize=1.5,
                    linewidth=0.6,
                    alpha=0.7,
                    zorder=3,
                )
        else:
            ax_pa.scatter(
                prof["x_axis"],
                pa_norm,
                color=style["color"],
                marker=style.get("marker", "o"),
                facecolors=mfc,
                edgecolors=style["color"],
                s=18,
                alpha=0.7,
                zorder=3,
            )
    ax_pa.set_ylabel("PA (deg)")
    ax_pa.grid(alpha=0.25)
    ax_pa.tick_params(labelbottom=False)
    # Data-driven PA limits from stop=0 data of first method, or all data
    if available and all_pa_norm:
        if "stop_codes" in profiles[available[0]]:
            sc = profiles[available[0]]["stop_codes"]
            pa_for_limits = all_pa_norm[0][sc == 0]
        else:
            pa_for_limits = all_pa_norm[0]
        pa_finite = pa_for_limits[np.isfinite(pa_for_limits)]
        if pa_finite.size > 1:
            pa_low, pa_high = robust_limits(pa_finite, 3, 97)
            pa_margin = max(3.0, 0.08 * (pa_high - pa_low + 1e-6))
            ax_pa.set_ylim(pa_low - pa_margin, pa_high + pa_margin)

    # Panel 4: Center offset (relative to isoster's robust median center)
    # Reference center: median of inner 10 stop=0 isophotes from the first
    # method (isoster), after 3-sigma clipping.
    ref_x0, ref_y0 = None, None
    first_method = available[0] if available else None
    if first_method and "x0" in profiles[first_method]:
        fprof = profiles[first_method]
        # Select stop=0 isophotes, sorted by SMA (innermost first)
        if "stop_codes" in fprof:
            good = fprof["stop_codes"] == 0
        else:
            good = np.isfinite(fprof["x0"])
        order = np.argsort(fprof["sma"])
        good_sorted = good[order]
        x0_sorted = fprof["x0"][order]
        y0_sorted = fprof["y0"][order]
        inner_x0 = x0_sorted[good_sorted][:10]
        inner_y0 = y0_sorted[good_sorted][:10]

        # 3-sigma clipping on radial distance from raw median
        if inner_x0.size > 0:
            mx, my = np.nanmedian(inner_x0), np.nanmedian(inner_y0)
            dist = np.sqrt((inner_x0 - mx) ** 2 + (inner_y0 - my) ** 2)
            sigma = np.nanstd(dist)
            if sigma > 0:
                keep = dist < 3 * sigma
                inner_x0 = inner_x0[keep]
                inner_y0 = inner_y0[keep]
            if inner_x0.size > 0:
                ref_x0 = float(np.nanmedian(inner_x0))
                ref_y0 = float(np.nanmedian(inner_y0))

    ax_cen = fig.add_subplot(right[4], sharex=ax_sb)
    all_offsets = []
    for method_name in available:
        prof = profiles[method_name]
        if "x0" not in prof:
            continue
        style = styles.get(method_name, {})
        # All methods use the isoster reference center if available;
        # fall back to own median if reference is not available
        if ref_x0 is not None:
            cx, cy = ref_x0, ref_y0
        else:
            cx = np.nanmedian(prof["x0"])
            cy = np.nanmedian(prof["y0"])
        offset = np.sqrt((prof["x0"] - cx) ** 2 + (prof["y0"] - cy) ** 2)
        all_offsets.append(offset)

        # Propagate center errors to offset error:
        # d(offset)/d(x0) ~ (x0-cx)/offset, etc.
        offset_err = None
        if "x0_err" in prof and "y0_err" in prof:
            safe_offset = np.where(offset > 0, offset, 1.0)
            dx = prof["x0"] - cx
            dy = prof["y0"] - cy
            offset_err = np.sqrt((dx / safe_offset * prof["x0_err"]) ** 2 + (dy / safe_offset * prof["y0_err"]) ** 2)
            # Where offset is ~0, use simple quadrature
            near_zero = offset < 1e-6
            if np.any(near_zero):
                offset_err[near_zero] = np.sqrt(prof["x0_err"][near_zero] ** 2 + prof["y0_err"][near_zero] ** 2)

        mfc = style["color"] if style.get("marker_face") == "filled" else "none"
        if offset_err is not None:
            if "stop_codes" in prof:
                _scatter_by_stop_code_in_method_color(
                    ax_cen,
                    prof["x_axis"],
                    offset,
                    prof["stop_codes"],
                    base_color=style["color"],
                    y_errors=offset_err,
                    marker_size=18,
                    label_stop_codes=False,
                    label=style.get("label", method_name),
                )
            else:
                ax_cen.errorbar(
                    prof["x_axis"],
                    offset,
                    yerr=offset_err,
                    fmt=style.get("marker", "o"),
                    color=style["color"],
                    mfc=mfc,
                    mec=style["color"],
                    markersize=4.2,
                    capsize=1.5,
                    linewidth=0.6,
                    alpha=0.7,
                    zorder=3,
                )
        else:
            ax_cen.scatter(
                prof["x_axis"],
                offset,
                color=style["color"],
                marker=style.get("marker", "o"),
                facecolors=mfc,
                edgecolors=style["color"],
                s=18,
                alpha=0.7,
                zorder=3,
            )
    ax_cen.set_ylabel(r"$\delta$ Cen (pix)")
    ax_cen.set_xlabel(r"SMA$^{0.25}$ (pix$^{0.25}$)")
    ax_cen.grid(alpha=0.25)
    # Data-driven Y-axis for center offset
    if all_offsets:
        all_off = np.concatenate(all_offsets)
        set_axis_limits_from_finite_values(
            ax_cen,
            all_off,
            margin_fraction=0.08,
            min_margin=0.05,
            lower_clip=0.0,
        )

    output_path = _Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


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
        p_x,
        p_y,
        yerr=p_yerr,
        fmt="o",
        mfc="none",
        mec="black",
        color="black",
        markersize=4,
        capsize=1.8,
        linewidth=0.7,
        alpha=0.8,
        label="photutils",
    )
