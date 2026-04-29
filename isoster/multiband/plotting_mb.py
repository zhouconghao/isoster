"""
Composite QA figure for multi-band isoster results.

Decision D15: a single composite PNG with four content blocks —

1. Per-band asinh-SB radial profile (one line per band).
2. Per-band residual mosaic (image - model reconstructed from joint
   geometry + per-band intensity).
3. Bender-normalized 3rd/4th-order harmonic profiles, one line per band
   per panel.
4. Shared geometry profile (eps, pa, x0, y0 vs SMA, single line each).

Surface-brightness and Bender-normalization conventions match
:mod:`isoster.plotting`. The asinh-SB error uses the log10-form
softening floor (``sigma_mu = 2.5/ln10 * sigma_I / max(|I|, scale)``).
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ..model import build_isoster_model
from ..plotting import _normalize_harmonic_for_plot, configure_qa_plot_style

logger = logging.getLogger(__name__)


# Stable per-band color cycle for up to 8 bands. Falls back to matplotlib's
# default cycle if the user has more bands than entries here.
_BAND_COLORS: tuple = (
    "#1f77b4",  # blue
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#ff7f0e",  # orange
    "#17becf",  # cyan
    "#e377c2",  # pink
    "#8c564b",  # brown
)


def _band_color(band_idx: int) -> str:
    if band_idx < len(_BAND_COLORS):
        return _BAND_COLORS[band_idx]
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#444"])
    return cycle[band_idx % len(cycle)]


def _intens_to_mu_asinh(
    intens: NDArray[np.floating],
    pixel_scale_arcsec: float,
    sb_zeropoint: float,
    softening_pix: float = 1.0,
) -> NDArray[np.floating]:
    """
    Lupton+ 1999 asinh-SB conversion (per-pixel intensity -> mag/arcsec^2).

    Converges to the standard log10 form in the bright limit. The
    softening parameter ``b`` is given in per-pixel intensity units;
    default ``softening_pix=1.0`` is a generic placeholder — callers in
    the asteris demo override with the per-band sky std.
    """
    pixarea = pixel_scale_arcsec**2
    I_per_arcsec2 = np.asarray(intens, dtype=np.float64) / pixarea
    b_per_arcsec2 = softening_pix / pixarea
    # asinh-mag: mu = -2.5/ln(10) * (asinh(I / 2b) + ln(b))
    return -2.5 / np.log(10.0) * (np.arcsinh(I_per_arcsec2 / (2.0 * b_per_arcsec2)) + np.log(b_per_arcsec2))


def _mu_err_log10_form(
    intens: NDArray[np.floating],
    sigma_intens: NDArray[np.floating],
    softening_pix: float = 1.0,
) -> NDArray[np.floating]:
    """
    log10-form softened errorbars matching the asinh-SB display.

    sigma_mu = (2.5 / ln 10) * sigma_I / max(|I|, scale)
    """
    intens = np.asarray(intens, dtype=np.float64)
    sigma_intens = np.asarray(sigma_intens, dtype=np.float64)
    denom = np.maximum(np.abs(intens), softening_pix)
    return 2.5 / np.log(10.0) * sigma_intens / denom


def _isophotes_to_per_band_singleband_lists(
    isophotes: Sequence[dict],
    bands: Sequence[str],
) -> Dict[str, List[dict]]:
    """
    Reshape multi-band isophote dicts into one single-band-style list per band.

    Used for :func:`isoster.model.build_isoster_model`, which expects
    bare ``intens, a3, b3, a4, b4`` keys (no per-band suffixes).
    """
    out: Dict[str, List[dict]] = {b: [] for b in bands}
    for iso in isophotes:
        if not bool(iso.get("valid", True)):
            continue
        for b in bands:
            row = {
                "sma": float(iso["sma"]),
                "x0": float(iso["x0"]),
                "y0": float(iso["y0"]),
                "eps": float(iso["eps"]),
                "pa": float(iso["pa"]),
                "intens": float(iso.get(f"intens_{b}", float("nan"))),
            }
            for n in (3, 4):
                row[f"a{n}"] = float(iso.get(f"a{n}_{b}", 0.0))
                row[f"b{n}"] = float(iso.get(f"b{n}_{b}", 0.0))
            if "use_eccentric_anomaly" in iso:
                row["use_eccentric_anomaly"] = bool(iso["use_eccentric_anomaly"])
            out[b].append(row)
    return out


# ---------------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------------


def _plot_sb_profile(
    ax: Axes,
    isophotes: Sequence[dict],
    bands: Sequence[str],
    sb_zeropoint: Optional[float],
    pixel_scale_arcsec: Optional[float],
    softening_per_band: Optional[Dict[str, float]] = None,
) -> None:
    """SB profile mu vs SMA, one line per band."""
    sma = np.array([float(iso["sma"]) for iso in isophotes])
    valid = np.array([bool(iso.get("valid", True)) for iso in isophotes])
    for b_idx, b in enumerate(bands):
        intens = np.array([float(iso.get(f"intens_{b}", np.nan)) for iso in isophotes])
        intens_err = np.array(
            [float(iso.get(f"intens_err_{b}", np.nan)) for iso in isophotes]
        )
        mask = valid & np.isfinite(intens) & np.isfinite(sma)
        if mask.sum() == 0:
            continue
        s_b = (
            softening_per_band.get(b, 1.0)
            if softening_per_band is not None
            else 1.0
        )
        if sb_zeropoint is not None and pixel_scale_arcsec is not None:
            mu = _intens_to_mu_asinh(
                intens[mask], pixel_scale_arcsec, sb_zeropoint, softening_pix=s_b,
            ) + sb_zeropoint
            mu_err = _mu_err_log10_form(intens[mask], intens_err[mask], softening_pix=s_b)
            ax.errorbar(
                sma[mask], mu, yerr=mu_err,
                color=_band_color(b_idx), label=b, marker="o", ms=3,
                lw=0.9, capsize=0, alpha=0.9,
            )
            ax.set_ylabel(r"$\mu$ [mag/arcsec$^2$]")
            ax.invert_yaxis()
        else:
            # Fall back to log10 intensity when SB constants are missing.
            with np.errstate(invalid="ignore", divide="ignore"):
                ax.semilogy(
                    sma[mask], np.maximum(intens[mask], 1e-9),
                    color=_band_color(b_idx), label=b, marker="o", ms=3, lw=0.9,
                )
            ax.set_ylabel("intensity (log)")
    ax.set_xlabel("SMA [pix]")
    ax.set_title("Surface-brightness profile (per band)")
    ax.legend(loc="best", fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.3)


def _plot_geometry_profile(
    ax_eps: Axes, ax_pa: Axes, ax_x0: Axes, ax_y0: Axes,
    isophotes: Sequence[dict],
) -> None:
    """Shared geometry profile (single line each)."""
    sma = np.array([float(iso["sma"]) for iso in isophotes])
    valid = np.array([bool(iso.get("valid", True)) for iso in isophotes])
    eps = np.array([float(iso["eps"]) for iso in isophotes])
    pa = np.array([float(iso["pa"]) for iso in isophotes])
    x0 = np.array([float(iso["x0"]) for iso in isophotes])
    y0 = np.array([float(iso["y0"]) for iso in isophotes])
    m = valid & np.isfinite(sma)
    line_kw = dict(marker="o", ms=2.5, lw=0.9, color="#222")
    ax_eps.plot(sma[m], eps[m], **line_kw)
    ax_eps.set_ylabel(r"$\epsilon$"); ax_eps.grid(True, alpha=0.3)
    ax_pa.plot(sma[m], np.rad2deg(pa[m]), **line_kw)
    ax_pa.set_ylabel("PA [deg]"); ax_pa.grid(True, alpha=0.3)
    ax_x0.plot(sma[m], x0[m], **line_kw)
    ax_x0.set_ylabel(r"$x_0$ [pix]"); ax_x0.grid(True, alpha=0.3)
    ax_y0.plot(sma[m], y0[m], **line_kw)
    ax_y0.set_ylabel(r"$y_0$ [pix]"); ax_y0.set_xlabel("SMA [pix]")
    ax_y0.grid(True, alpha=0.3)


def _plot_harmonics(
    axes: Sequence[Axes],
    isophotes: Sequence[dict],
    bands: Sequence[str],
) -> None:
    """4 small panels: A3, B3, A4, B4 normalized per band."""
    sma = np.array([float(iso["sma"]) for iso in isophotes])
    valid = np.array([bool(iso.get("valid", True)) for iso in isophotes])
    panel_specs = (
        (axes[0], "a3", r"$A_3 / (a \, dI/da)$"),
        (axes[1], "b3", r"$B_3 / (a \, dI/da)$"),
        (axes[2], "a4", r"$A_4 / (a \, dI/da)$"),
        (axes[3], "b4", r"$B_4 / (a \, dI/da)$"),
    )
    for b_idx, b in enumerate(bands):
        intens = np.array([float(iso.get(f"intens_{b}", np.nan)) for iso in isophotes])
        # Per-band gradient: prefer the debug `grad_<b>` column; otherwise
        # rely on _normalize_harmonic_for_plot's np.gradient fallback.
        grad = np.array([float(iso.get(f"grad_{b}", np.nan)) for iso in isophotes])
        for ax, harm_key, ylabel in panel_specs:
            harm = np.array([float(iso.get(f"{harm_key}_{b}", np.nan)) for iso in isophotes])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                normalized = _normalize_harmonic_for_plot(harm, sma, grad, intens)
            m = valid & np.isfinite(normalized)
            if m.sum() == 0:
                continue
            ax.plot(
                sma[m], normalized[m], color=_band_color(b_idx),
                marker="o", ms=2.5, lw=0.9, label=b if harm_key == "a3" else None,
            )
    for ax, _key, ylabel in panel_specs:
        ax.set_ylabel(ylabel); ax.grid(True, alpha=0.3)
    axes[3].set_xlabel("SMA [pix]")
    axes[0].legend(loc="best", fontsize=8, framealpha=0.85)


def _plot_residual_mosaic(
    axes: Sequence[Axes],
    isophotes: Sequence[dict],
    bands: Sequence[str],
    images: Sequence[NDArray[np.floating]],
) -> None:
    """Per-band residual = image - model reconstructed from joint isophotes."""
    per_band_lists = _isophotes_to_per_band_singleband_lists(isophotes, bands)
    h, w = np.asarray(images[0]).shape
    for b_idx, b in enumerate(bands):
        ax = axes[b_idx]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = build_isoster_model(
                    (h, w), per_band_lists[b], use_harmonics=True,
                )
        except Exception as e:  # noqa: BLE001 — fall back to image only
            logger.warning("model build failed for band %s: %s", b, e)
            ax.text(0.5, 0.5, f"model error: {e}", transform=ax.transAxes, ha="center")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        residual = np.asarray(images[b_idx], dtype=np.float64) - model
        # Symmetric asinh-stretch around zero.
        rms = float(np.std(residual))
        scale = max(1e-9, rms * 5.0)
        stretched = np.arcsinh(residual / scale)
        vmax = float(np.percentile(np.abs(stretched), 99.0))
        ax.imshow(
            stretched, origin="lower", cmap="RdBu_r",
            vmin=-vmax, vmax=vmax,
        )
        ax.set_title(f"residual ({b})", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def plot_qa_summary_mb(
    result: dict,
    images: Sequence[NDArray[np.floating]],
    *,
    bands: Optional[Sequence[str]] = None,
    sb_zeropoint: Optional[float] = None,
    pixel_scale_arcsec: Optional[float] = None,
    softening_per_band: Optional[Dict[str, float]] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (20.0, 14.0),
    title: Optional[str] = None,
) -> Figure:
    """
    Render the composite multi-band QA figure.

    Parameters
    ----------
    result
        Dict from :func:`isoster.multiband.fit_image_multiband`. Must
        contain ``'isophotes'`` and ``'bands'`` (or ``bands`` explicitly).
    images
        The same list of band images passed to the driver, in the same
        order as ``bands``. Used for the residual mosaic.
    bands
        Optional override for the band list. Default reads from
        ``result['bands']``.
    sb_zeropoint, pixel_scale_arcsec
        SB-conversion constants. Both must be set together
        (project rule); otherwise the SB panel falls back to log-
        intensity.
    softening_per_band
        Optional per-band asinh-SB softening (per-pixel intensity units).
        Default 1.0 per band.
    output_path
        If set, save the figure to this path.
    figsize, title
        Standard matplotlib knobs.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if matplotlib.get_backend().lower() == "agg":
        logger.debug("rendering multi-band QA on the Agg backend")
    # Reset usetex before plotting (carry-forward watch-out).
    plt.rcParams["text.usetex"] = False
    configure_qa_plot_style()
    plt.rcParams["text.usetex"] = False

    isophotes = list(result["isophotes"])
    if bands is None:
        bands = list(result.get("bands", []))
    if not bands:
        raise ValueError("bands is required (either via result['bands'] or explicit kwarg)")
    if len(images) != len(bands):
        raise ValueError(
            f"len(images) ({len(images)}) does not match len(bands) ({len(bands)})"
        )

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    n_bands = len(bands)

    # 4-row x ~6-col layout (rough sketch):
    #   Row 0: SB profile (cols 0-2)            | residual mosaic (cols 3-5, n_bands subpanels)
    #   Row 1: Bender harmonics (4 stacked)     | geometry (4 stacked)
    gs = fig.add_gridspec(
        nrows=8, ncols=6,
        height_ratios=[1, 1, 1, 1, 1, 1, 1, 1],
        width_ratios=[1, 1, 1, 1, 1, 1],
    )

    # SB profile occupies rows 0-3, cols 0-2
    ax_sb = fig.add_subplot(gs[0:4, 0:3])
    _plot_sb_profile(
        ax_sb, isophotes, bands, sb_zeropoint, pixel_scale_arcsec,
        softening_per_band=softening_per_band,
    )

    # Residual mosaic: rows 0-3, cols 3-5 split into n_bands columns.
    res_axes: List[Axes] = []
    if n_bands > 0:
        # If many bands, wrap to two rows (3 wide x 2 tall); otherwise single row.
        if n_bands <= 3:
            for b_idx in range(n_bands):
                col_lo = 3 + b_idx
                ax = fig.add_subplot(gs[0:4, col_lo:col_lo + 1])
                res_axes.append(ax)
        else:
            n_rows = 2
            n_cols = (n_bands + 1) // 2
            row_height = 4 // n_rows
            for b_idx in range(n_bands):
                r = b_idx // n_cols
                c = b_idx % n_cols
                ax = fig.add_subplot(
                    gs[r * row_height:(r + 1) * row_height, 3 + c:3 + c + 1]
                )
                res_axes.append(ax)
        _plot_residual_mosaic(res_axes, isophotes, bands, images)

    # Bender harmonics: rows 4-7, cols 0-2 split into 4 small panels stacked.
    ax_a3 = fig.add_subplot(gs[4, 0:3])
    ax_b3 = fig.add_subplot(gs[5, 0:3], sharex=ax_a3)
    ax_a4 = fig.add_subplot(gs[6, 0:3], sharex=ax_a3)
    ax_b4 = fig.add_subplot(gs[7, 0:3], sharex=ax_a3)
    _plot_harmonics([ax_a3, ax_b3, ax_a4, ax_b4], isophotes, bands)

    # Geometry profile: rows 4-7, cols 3-5 split into 4 stacked panels.
    ax_eps = fig.add_subplot(gs[4, 3:6])
    ax_pa = fig.add_subplot(gs[5, 3:6], sharex=ax_eps)
    ax_x0 = fig.add_subplot(gs[6, 3:6], sharex=ax_eps)
    ax_y0 = fig.add_subplot(gs[7, 3:6], sharex=ax_eps)
    _plot_geometry_profile(ax_eps, ax_pa, ax_x0, ax_y0, isophotes)

    if title is not None:
        fig.suptitle(title, fontsize=12)

    if output_path is not None:
        fig.savefig(output_path, dpi=120, bbox_inches="tight")

    return fig
