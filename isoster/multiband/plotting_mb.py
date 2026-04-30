"""
Composite QA figure for multi-band isoster results.

Layout (decision D15, refined 2026-04-30 to follow the single-band QA
style and the asteris-pair runner conventions):

* Left column (stacked, all sharing x = SMA^0.25 in arcsec):
    1. Surface-brightness profile (asinh-SB, Lupton+ 1999) with per-band
       errorbars.
    2. Bender-normalized A_3 / B_3 / A_4 / B_4 — four compact panels.
    3. eps, PA, (x0, y0) — three compact geometry panels.
* Right column:
    - i-band cutout with representative isophotes overplotted.
    - Per-band residual mosaic (image - model) with the object mask
      overlaid translucently. Color scaling is computed from unmasked
      pixels only so weak residual structure is visible.

All 1-D points are shown as scatter markers with errorbars (no
connecting line). Surface-brightness conversions follow the project
rule: ``zp`` and ``pixel_scale_arcsec`` are passed as separate
arguments and never pre-combined.
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
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from numpy.typing import NDArray

from ..model import build_isoster_model
from ..plotting import _normalize_harmonic_for_plot, configure_qa_plot_style

logger = logging.getLogger(__name__)


# Stable per-band color cycle.
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


# ---------------------------------------------------------------------------
# asinh-SB conversion (Lupton+ 1999) — mirrors examples/example_asteris_denoised/run_isoster_pair.py
# ---------------------------------------------------------------------------


def _intens_to_mu_asinh(
    intens: NDArray[np.floating],
    zeropoint: float,
    pixel_scale_arcsec: float,
    scale: float,
) -> NDArray[np.floating]:
    """``mu_asinh = -2.5/ln(10) * asinh(I_pp/(2B)) + zp - 2.5*log10(B)``.

    With ``I_pp = I / pixarea`` and ``B = scale / pixarea``. Converges
    to the standard log10 form for ``|I| >> scale``; well-defined for
    any I, including negatives. Identical to the recipe in
    ``run_isoster_pair.intens_to_mu_asinh``.
    """
    pixarea = pixel_scale_arcsec * pixel_scale_arcsec
    intens = np.asarray(intens, dtype=np.float64)
    i_pp = intens / pixarea
    big_b = scale / pixarea
    return (
        -2.5 / np.log(10.0) * np.arcsinh(i_pp / (2.0 * big_b))
        + zeropoint
        - 2.5 * np.log10(big_b)
    )


def _mu_asinh_error(
    intens: NDArray[np.floating],
    intens_err: NDArray[np.floating],
    scale: float,
) -> NDArray[np.floating]:
    """log10-form softened errorbar: ``2.5/ln10 * sigma_I / max(|I|, scale)``.

    Matches the conventional log10 visual feel in the high-S/N regime
    and saturates at ``2.5/ln10 * sigma_I/scale`` when ``|I| -> 0``.
    """
    intens = np.asarray(intens, dtype=np.float64)
    intens_err = np.asarray(intens_err, dtype=np.float64)
    denom = np.maximum(np.abs(intens), scale)
    return 2.5 / np.log(10.0) * intens_err / denom


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _xaxis_arcsec_pow(sma_pix: NDArray[np.floating], pixel_scale_arcsec: float) -> NDArray[np.floating]:
    """X-axis = (SMA in arcsec)^0.25 — matches the single-band QA convention."""
    sma_arcsec = np.asarray(sma_pix, dtype=np.float64) * float(pixel_scale_arcsec)
    out = np.full_like(sma_arcsec, np.nan, dtype=np.float64)
    good = sma_arcsec > 0
    out[good] = sma_arcsec[good] ** 0.25
    return out


def _isophotes_to_per_band_singleband_lists(
    isophotes: Sequence[dict],
    bands: Sequence[str],
) -> Dict[str, List[dict]]:
    """Reshape multi-band isophote dicts into one single-band-style list per band."""
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


def _resolve_softening(
    bands: Sequence[str],
    isophotes: Sequence[dict],
    softening_per_band: Optional[Dict[str, float]],
) -> Dict[str, float]:
    """
    Choose an asinh-SB softening per band.

    User-provided values win. Otherwise: median of finite ``intens_err_<b>``,
    falling back to ``1.0`` if no errors are present.
    """
    out: Dict[str, float] = {}
    for b in bands:
        if softening_per_band is not None and b in softening_per_band:
            out[b] = max(float(softening_per_band[b]), 1e-12)
            continue
        errs = np.array(
            [float(iso.get(f"intens_err_{b}", np.nan)) for iso in isophotes],
            dtype=np.float64,
        )
        finite = errs[np.isfinite(errs) & (errs > 0)]
        out[b] = max(float(np.median(finite)) if finite.size else 1.0, 1e-12)
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
    softening_per_band: Dict[str, float],
) -> None:
    sma = np.array([float(iso["sma"]) for iso in isophotes])
    valid = np.array([bool(iso.get("valid", True)) for iso in isophotes])
    if pixel_scale_arcsec is None:
        pixel_scale_arcsec = 1.0  # falls back to log-intensity below
    x = _xaxis_arcsec_pow(sma, pixel_scale_arcsec)

    has_sb = sb_zeropoint is not None and pixel_scale_arcsec is not None
    for b_idx, b in enumerate(bands):
        intens = np.array([float(iso.get(f"intens_{b}", np.nan)) for iso in isophotes])
        intens_err = np.array(
            [float(iso.get(f"intens_err_{b}", np.nan)) for iso in isophotes]
        )
        mask = valid & np.isfinite(intens) & np.isfinite(x)
        if mask.sum() == 0:
            continue
        scale_b = softening_per_band.get(b, 1.0)
        if has_sb:
            mu = _intens_to_mu_asinh(intens[mask], sb_zeropoint, pixel_scale_arcsec, scale_b)  # type: ignore[arg-type]
            mu_err = _mu_asinh_error(intens[mask], intens_err[mask], scale_b)
            ax.errorbar(
                x[mask], mu, yerr=mu_err,
                color=_band_color(b_idx), label=b,
                marker="o", ms=6.0, mfc=_band_color(b_idx), mec="white",
                mew=0.6, lw=0, elinewidth=0.9, capsize=0, alpha=0.95,
            )
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                mu = np.where(intens[mask] > 0, np.log10(np.maximum(intens[mask], 1e-30)), np.nan)
            ax.errorbar(
                x[mask], mu, yerr=None,
                color=_band_color(b_idx), label=b,
                marker="o", ms=6.0, lw=0,
            )
    if has_sb:
        ax.set_ylabel(r"$\mu$ [mag/arcsec$^2$]")
        ax.invert_yaxis()
    else:
        ax.set_ylabel(r"$\log_{10}\,I$")
    ax.legend(loc="best", fontsize=12, framealpha=0.85, markerscale=0.9)
    ax.grid(True, alpha=0.3)


def _plot_harmonic(
    ax: Axes,
    harm_key: str,
    ylabel: str,
    isophotes: Sequence[dict],
    bands: Sequence[str],
    pixel_scale_arcsec: float,
) -> None:
    sma = np.array([float(iso["sma"]) for iso in isophotes])
    valid = np.array([bool(iso.get("valid", True)) for iso in isophotes])
    x = _xaxis_arcsec_pow(sma, pixel_scale_arcsec)
    for b_idx, b in enumerate(bands):
        intens = np.array([float(iso.get(f"intens_{b}", np.nan)) for iso in isophotes])
        grad = np.array([float(iso.get(f"grad_{b}", np.nan)) for iso in isophotes])
        harm = np.array([float(iso.get(f"{harm_key}_{b}", np.nan)) for iso in isophotes])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            normalized = _normalize_harmonic_for_plot(harm, sma, grad, intens)
        m = valid & np.isfinite(normalized) & np.isfinite(x)
        if m.sum() == 0:
            continue
        ax.plot(
            x[m], normalized[m], color=_band_color(b_idx),
            marker="o", ms=5.0, lw=0, alpha=0.9,
        )
    ax.axhline(0.0, color="#666", lw=0.5, alpha=0.5)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)


def _plot_eps(
    ax: Axes, isophotes: Sequence[dict], pixel_scale_arcsec: float,
) -> None:
    sma = np.array([float(iso["sma"]) for iso in isophotes])
    valid = np.array([bool(iso.get("valid", True)) for iso in isophotes])
    eps = np.array([float(iso["eps"]) for iso in isophotes])
    eps_err = np.array(
        [float(iso.get("eps_err", 0.0)) for iso in isophotes], dtype=np.float64
    )
    x = _xaxis_arcsec_pow(sma, pixel_scale_arcsec)
    m = valid & np.isfinite(x) & np.isfinite(eps)
    if m.any():
        yerr = np.where(eps_err > 0, eps_err, np.nan)[m]
        if not np.all(np.isnan(yerr)):
            ax.errorbar(
                x[m], eps[m], yerr=yerr,
                color="#222", marker="o", ms=5.0, lw=0,
                elinewidth=0.7, capsize=0,
            )
        else:
            ax.plot(x[m], eps[m], color="#222", marker="o", ms=5.0, lw=0)
    ax.set_ylabel(r"$\epsilon$", fontsize=12)
    ax.grid(True, alpha=0.3)


def _plot_pa(ax: Axes, isophotes: Sequence[dict], pixel_scale_arcsec: float) -> None:
    sma = np.array([float(iso["sma"]) for iso in isophotes])
    valid = np.array([bool(iso.get("valid", True)) for iso in isophotes])
    pa = np.array([float(iso["pa"]) for iso in isophotes])
    pa_err = np.array(
        [float(iso.get("pa_err", 0.0)) for iso in isophotes], dtype=np.float64
    )
    pa_deg = np.rad2deg(pa) % 180.0
    x = _xaxis_arcsec_pow(sma, pixel_scale_arcsec)
    m = valid & np.isfinite(x) & np.isfinite(pa_deg)
    if m.any():
        yerr = np.where(pa_err > 0, np.rad2deg(pa_err), np.nan)[m]
        if not np.all(np.isnan(yerr)):
            ax.errorbar(
                x[m], pa_deg[m], yerr=yerr,
                color="#222", marker="o", ms=5.0, lw=0,
                elinewidth=0.7, capsize=0,
            )
        else:
            ax.plot(x[m], pa_deg[m], color="#222", marker="o", ms=5.0, lw=0)
    ax.set_ylabel("PA [deg]", fontsize=12)
    ax.grid(True, alpha=0.3)


def _plot_center(
    ax: Axes, isophotes: Sequence[dict], pixel_scale_arcsec: float,
) -> None:
    """x0 / y0 vs SMA, with each shown relative to the median for clarity."""
    sma = np.array([float(iso["sma"]) for iso in isophotes])
    valid = np.array([bool(iso.get("valid", True)) for iso in isophotes])
    x0 = np.array([float(iso["x0"]) for iso in isophotes])
    y0 = np.array([float(iso["y0"]) for iso in isophotes])
    x = _xaxis_arcsec_pow(sma, pixel_scale_arcsec)
    m = valid & np.isfinite(x) & np.isfinite(x0) & np.isfinite(y0)
    if not m.any():
        ax.set_ylabel(r"$\Delta x_0,\Delta y_0$ [pix]", fontsize=12)
        ax.grid(True, alpha=0.3)
        return
    x0_med = float(np.median(x0[m]))
    y0_med = float(np.median(y0[m]))
    ax.plot(x[m], x0[m] - x0_med, color="#1f77b4", marker="o", ms=5.0, lw=0,
            label=r"$\Delta x_0$")
    ax.plot(x[m], y0[m] - y0_med, color="#d62728", marker="s", ms=5.0, lw=0,
            label=r"$\Delta y_0$")
    ax.axhline(0.0, color="#666", lw=0.5, alpha=0.5)
    ax.set_ylabel(r"$\Delta_{\rm c}$ [pix]", fontsize=12)
    ax.legend(loc="best", fontsize=10, framealpha=0.85)
    ax.grid(True, alpha=0.3)


def _plot_image_with_isophotes(
    ax: Axes,
    image: NDArray[np.floating],
    isophotes: Sequence[dict],
    band_label: str,
    n_rings: int = 8,
) -> None:
    """i-band cutout with N evenly-log-spaced isophote ellipses overplotted."""
    valid_iso = [iso for iso in isophotes if bool(iso.get("valid", True)) and iso.get("sma", 0.0) > 0]
    h, w = np.asarray(image).shape

    # Asinh stretch for the background.
    img = np.asarray(image, dtype=np.float64)
    finite = img[np.isfinite(img)]
    if finite.size:
        scale = float(np.nanpercentile(np.abs(finite), 70.0)) or 1e-6
        vmax = float(np.nanpercentile(np.abs(finite), 99.7)) or 1.0
    else:
        scale, vmax = 1e-3, 1.0
    stretched = np.arcsinh(img / max(scale, 1e-12)) / np.arcsinh(max(vmax, 1e-12) / max(scale, 1e-12))
    ax.imshow(stretched, origin="lower", cmap="gray_r", vmin=-0.05, vmax=1.05)

    # Pick representative isophotes evenly in log(sma).
    if valid_iso:
        smas = np.array([iso["sma"] for iso in valid_iso])
        if smas.size > n_rings:
            log_targets = np.linspace(np.log(smas.min()), np.log(smas.max()), n_rings)
            chosen_idx = [int(np.argmin(np.abs(np.log(smas) - lt))) for lt in log_targets]
            chosen_idx = sorted(set(chosen_idx))
        else:
            chosen_idx = list(range(len(valid_iso)))
        cmap = plt.get_cmap("plasma")
        for k, idx in enumerate(chosen_idx):
            iso = valid_iso[idx]
            color = cmap(k / max(len(chosen_idx) - 1, 1))
            sma = float(iso["sma"])
            eps = float(iso["eps"])
            pa = float(iso["pa"])
            ellipse = Ellipse(
                xy=(float(iso["x0"]), float(iso["y0"])),
                width=2.0 * sma,
                height=2.0 * sma * (1.0 - eps),
                angle=float(np.rad2deg(pa)),
                facecolor="none",
                edgecolor=color,
                linewidth=1.0,
                alpha=0.9,
            )
            ax.add_patch(ellipse)
    ax.set_title(f"{band_label} + representative isophotes", fontsize=10)
    ax.set_xlim(0, w); ax.set_ylim(0, h)
    ax.set_xticks([]); ax.set_yticks([])


def _plot_residual_panel(
    ax: Axes,
    image: NDArray[np.floating],
    model: NDArray[np.floating],
    band_label: str,
    mask: Optional[NDArray[np.bool_]] = None,
) -> None:
    """One residual panel with mask overlay; scale from unmasked pixels only."""
    residual = np.asarray(image, dtype=np.float64) - np.asarray(model, dtype=np.float64)
    if mask is not None:
        unmasked = residual[~np.asarray(mask, dtype=bool) & np.isfinite(residual)]
    else:
        unmasked = residual[np.isfinite(residual)]
    if unmasked.size == 0:
        unmasked = residual[np.isfinite(residual)]
    if unmasked.size == 0:
        rms = 1e-6
    else:
        rms = float(np.std(unmasked))
    scale = max(rms, 1e-9) * 5.0
    stretched = np.arcsinh(residual / scale)
    if unmasked.size:
        unmasked_stretched = np.arcsinh(unmasked / scale)
        vmax = float(np.percentile(np.abs(unmasked_stretched), 99.0))
    else:
        vmax = float(np.percentile(np.abs(stretched), 99.0))
    vmax = max(vmax, 1e-3)
    ax.imshow(stretched, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    if mask is not None:
        # Translucent overlay where mask is True.
        overlay_cmap = ListedColormap([(0, 0, 0, 0), (0.2, 0.2, 0.2, 0.45)])
        ax.imshow(np.asarray(mask, dtype=bool), origin="lower", cmap=overlay_cmap, vmin=0, vmax=1)
    ax.set_title(f"residual ({band_label})", fontsize=10)
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
    object_mask: Optional[NDArray[np.bool_]] = None,
    reference_band_idx: Optional[int] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (18.0, 16.0),
    title: Optional[str] = None,
) -> Figure:
    """
    Render the composite multi-band QA figure.

    Layout:
    * Left column (1-D profiles, all sharing x = SMA^(1/4) in arcsec):
      SB, A_3, B_3, A_4, B_4, eps, PA, center offsets.
    * Right column (2-D images): reference-band cutout with
      representative isophotes overplotted (top), per-band residual
      mosaic with mask overlay (bottom).

    Parameters
    ----------
    result, images, bands : as before.
    sb_zeropoint, pixel_scale_arcsec
        SB conversion constants. Both must be set together
        (project rule); if either is None the SB panel falls back to
        log10(I) on a unit x-axis.
    softening_per_band
        Per-band asinh-SB softening (per-pixel intensity units). When
        omitted, derived from the median of finite ``intens_err_<b>``.
    object_mask
        ``(H, W)`` boolean mask (True = bad pixel) overlaid on every
        residual panel and used to compute the per-panel color scale
        from unmasked pixels only.
    reference_band_idx
        Index into ``bands`` for the cutout-with-isophotes panel.
        Default tries ``result['reference_band']`` and falls back to 0.
    """
    if matplotlib.get_backend().lower() == "agg":
        logger.debug("rendering multi-band QA on the Agg backend")
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

    # Resolve x-axis pixel scale (only used for axis labelling).
    pix = pixel_scale_arcsec if pixel_scale_arcsec is not None else 1.0

    # asinh-SB softening per band.
    softening = _resolve_softening(bands, isophotes, softening_per_band)

    # Reference band selection for the cutout panel.
    if reference_band_idx is None:
        ref_band = str(result.get("reference_band", bands[0]))
        try:
            reference_band_idx = list(bands).index(ref_band)
        except ValueError:
            reference_band_idx = 0

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    n_bands = len(bands)

    # --- gridspec: 24 row units, 8 col units. -------------------------------
    # Left column 1-D stack, cols 0-3:
    #   SB         rows 0-7   (8 units, biggest panel)
    #   A3         rows 8-9
    #   B3         rows 10-11
    #   A4         rows 12-13
    #   B4         rows 14-15
    #   eps        rows 16-17
    #   PA         rows 18-19
    #   center     rows 20-23
    # Right column, cols 4-7:
    #   image+iso  rows 0-11  (top half)
    #   residual mosaic rows 12-23 (bottom half)
    gs = fig.add_gridspec(nrows=24, ncols=8)

    ax_sb = fig.add_subplot(gs[0:8, 0:4])
    ax_a3 = fig.add_subplot(gs[8:10, 0:4], sharex=ax_sb)
    ax_b3 = fig.add_subplot(gs[10:12, 0:4], sharex=ax_sb)
    ax_a4 = fig.add_subplot(gs[12:14, 0:4], sharex=ax_sb)
    ax_b4 = fig.add_subplot(gs[14:16, 0:4], sharex=ax_sb)
    ax_eps = fig.add_subplot(gs[16:18, 0:4], sharex=ax_sb)
    ax_pa = fig.add_subplot(gs[18:20, 0:4], sharex=ax_sb)
    ax_ctr = fig.add_subplot(gs[20:24, 0:4], sharex=ax_sb)

    _plot_sb_profile(ax_sb, isophotes, bands, sb_zeropoint, pixel_scale_arcsec, softening)
    _plot_harmonic(ax_a3, "a3", r"$A_3 / (a\,dI/da)$", isophotes, bands, pix)
    _plot_harmonic(ax_b3, "b3", r"$B_3 / (a\,dI/da)$", isophotes, bands, pix)
    _plot_harmonic(ax_a4, "a4", r"$A_4 / (a\,dI/da)$", isophotes, bands, pix)
    _plot_harmonic(ax_b4, "b4", r"$B_4 / (a\,dI/da)$", isophotes, bands, pix)
    _plot_eps(ax_eps, isophotes, pix)
    _plot_pa(ax_pa, isophotes, pix)
    _plot_center(ax_ctr, isophotes, pix)

    # Hide x-tick labels on every stacked panel except the bottom one.
    for ax in (ax_sb, ax_a3, ax_b3, ax_a4, ax_b4, ax_eps, ax_pa):
        ax.tick_params(axis="x", labelbottom=False)
    if pixel_scale_arcsec is not None:
        ax_ctr.set_xlabel(r"$\mathrm{SMA}^{0.25}$  [arcsec$^{0.25}$]", fontsize=12)
    else:
        ax_ctr.set_xlabel(r"$\mathrm{SMA}^{0.25}$  [pix$^{0.25}$]", fontsize=12)

    # --- right column: image + isophotes ------------------------------------
    ax_img = fig.add_subplot(gs[0:12, 4:8])
    _plot_image_with_isophotes(
        ax_img, images[reference_band_idx], isophotes,
        band_label=bands[reference_band_idx],
    )

    # --- right column: residual mosaic --------------------------------------
    per_band_lists = _isophotes_to_per_band_singleband_lists(isophotes, bands)
    h, w = np.asarray(images[0]).shape
    # Tile B residual subplots inside rows 12-23, cols 4-7. Use up to 3 cols
    # per row; B=5 -> 2 rows of 3 cells (one cell empty).
    n_cols_res = min(3, n_bands)
    n_rows_res = (n_bands + n_cols_res - 1) // n_cols_res
    res_row_height = max(1, 12 // n_rows_res)
    res_col_width = max(1, 4 // n_cols_res)
    for b_idx, b in enumerate(bands):
        r = b_idx // n_cols_res
        c = b_idx % n_cols_res
        row_lo = 12 + r * res_row_height
        row_hi = row_lo + res_row_height
        col_lo = 4 + c * res_col_width
        col_hi = col_lo + res_col_width
        ax_r = fig.add_subplot(gs[row_lo:row_hi, col_lo:col_hi])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = build_isoster_model(
                    (h, w), per_band_lists[b], use_harmonics=True,
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("model build failed for band %s: %s", b, e)
            ax_r.text(0.5, 0.5, f"model error", transform=ax_r.transAxes, ha="center")
            ax_r.set_xticks([]); ax_r.set_yticks([])
            continue
        _plot_residual_panel(ax_r, images[b_idx], model, b, mask=object_mask)

    if title is not None:
        fig.suptitle(title, fontsize=13)

    if output_path is not None:
        fig.savefig(output_path, dpi=120, bbox_inches="tight")

    return fig
