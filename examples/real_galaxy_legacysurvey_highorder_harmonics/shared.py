"""
Shared helpers for the LegacySurvey high-order harmonics example campaign.

Provides:
- ``GALAXY_CONFIGS``  — per-galaxy metadata (pixel scale, band names, etc.)
- ``load_legacysurvey_fits()``  — load a single band from the 3D FITS cube
- ``make_isoster_configs()``  — build the 6 fitting conditions as IsosterConfig dicts
- ``CONDITION_LABELS`` / ``CONDITION_ORDER``  — canonical ordering
- ``plot_harmonic_comparison_qa()``  — cross-condition comparison figure
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from astropy.io import fits

from isoster.config import IsosterConfig
from isoster.plotting import (
    configure_qa_plot_style,
    derive_arcsinh_parameters,
    draw_isophote_overlays,
    make_arcsinh_display_from_parameters,
    normalize_pa_degrees,
    plot_profile_by_stop_code,
    robust_limits,
    set_axis_limits_from_finite_values,
    set_x_limits_with_right_margin,
    latex_safe_text,
)

# ---------------------------------------------------------------------------
# Per-galaxy metadata
# ---------------------------------------------------------------------------

#: Supported galaxy names (lower-case, matching filenames under examples/data/)
SUPPORTED_GALAXIES = ("eso243-49", "ngc3610")

#: Pixel scale in arcsec / pixel for each galaxy
PIXEL_SCALE = {
    "eso243-49": 0.25,
    "ngc3610": 1.0,
}

#: Band names corresponding to FITS data-cube indices 0, 1, 2
BAND_NAMES = {
    "eso243-49": ["g", "r", "z"],
    "ngc3610": ["g", "r", "i"],
}

#: FITS file name for each galaxy
FITS_FILENAME = {
    "eso243-49": "eso243-49.fits",
    "ngc3610": "ngc3610.fits",
}

#: Initial SMA in pixels (bright center estimate)
INITIAL_SMA = {
    "eso243-49": 6.0,
    "ngc3610": 6.0,
}

#: Masking parameters per galaxy (passed to masking.make_object_mask)
MASK_PARAMS: dict[str, dict[str, Any]] = {
    "eso243-49": dict(
        on_galaxy=True,
        on_galaxy_box=32,   # large box to follow the extended disk envelope
        on_galaxy_filter=3,
        on_galaxy_npixels=10,
        on_galaxy_dilate_fwhm=8.0,
    ),
    "ngc3610": dict(
        on_galaxy=True,
        on_galaxy_box=32,   # large box to follow the smooth elliptical envelope
        on_galaxy_npixels=10,
        on_galaxy_dilate_fwhm=8.0,
    ),
}

# ---------------------------------------------------------------------------
# Six fitting conditions (2 × 3 grid)
# ---------------------------------------------------------------------------

#: Canonical condition order
CONDITION_ORDER = [
    "pa_baseline",
    "ea_baseline",
    "pa_harmonics_34",
    "ea_harmonics_34",
    "pa_harmonics_34567",
    "ea_harmonics_34567",
]

#: Short display label for each condition (used in figure legends)
CONDITION_DISPLAY = {
    "pa_baseline": "PA / no-sim",
    "ea_baseline": "EA / no-sim",
    "pa_harmonics_34": "PA / [3,4]",
    "ea_harmonics_34": "EA / [3,4]",
    "pa_harmonics_34567": "PA / [3–7]",
    "ea_harmonics_34567": "EA / [3–7]",
}

#: Colours for cross-condition comparison lines
CONDITION_COLORS = {
    "pa_baseline": "#1f77b4",
    "ea_baseline": "#aec7e8",
    "pa_harmonics_34": "#d62728",
    "ea_harmonics_34": "#f4a6a6",
    "pa_harmonics_34567": "#2ca02c",
    "ea_harmonics_34567": "#98df8a",
}

#: Markers for cross-condition comparison scatter
CONDITION_MARKERS = {
    "pa_baseline": "o",
    "ea_baseline": "s",
    "pa_harmonics_34": "^",
    "ea_harmonics_34": "D",
    "pa_harmonics_34567": "P",
    "ea_harmonics_34567": "v",
}


def make_isoster_configs(galaxy: str, sma0: float | None = None) -> dict[str, IsosterConfig]:
    """Build the six IsosterConfig objects for the fitting conditions.

    Parameters
    ----------
    galaxy : str
        Galaxy name (must be in :data:`SUPPORTED_GALAXIES`).
    sma0 : float or None
        Initial SMA override.  Uses :data:`INITIAL_SMA` default when None.

    Returns
    -------
    dict mapping condition label → IsosterConfig
    """
    _sma0 = sma0 if sma0 is not None else INITIAL_SMA[galaxy]

    # Shared baseline kwargs for all conditions
    base_kwargs: dict[str, Any] = dict(
        sma0=_sma0,
        convergence_scaling="sector_area",
        geometry_damping=0.7,
        permissive_geometry=True,
    )

    configs = {}
    for label in CONDITION_ORDER:
        use_ea = label.startswith("ea_")
        if "34567" in label:
            orders = [3, 4, 5, 6, 7]
            sim = True
        elif "34" in label and "harmonics" in label:
            orders = [3, 4]
            sim = True
        else:
            # baseline: no simultaneous harmonics; [3,4] computed post-hoc only
            orders = [3, 4]
            sim = False

        configs[label] = IsosterConfig(
            **base_kwargs,
            use_eccentric_anomaly=use_ea,
            simultaneous_harmonics=sim,
            harmonic_orders=orders,
        )

    return configs


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_legacysurvey_fits(
    fits_path: str | Path,
    band_index: int,
) -> tuple[np.ndarray, float]:
    """Load one band from a LegacySurvey 3D FITS cube.

    Parameters
    ----------
    fits_path : str or Path
        Path to the FITS file (PRIMARY HDU expected, shape ``(3, H, W)``).
    band_index : int
        Band plane to extract (0, 1, or 2).

    Returns
    -------
    image : 2D float64 array of shape ``(H, W)``
    pixel_scale : float
        Pixel scale in arcsec/pixel derived from FITS header (``CD2_2`` or
        ``CDELT2``), or default 1.0 if header is absent / non-standard.
    """
    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(np.float64)
        header = hdul[0].header

    if data.ndim == 3:
        image = data[band_index]
    elif data.ndim == 2:
        image = data
    else:
        raise ValueError(f"Unexpected data shape {data.shape} in {fits_path}")

    # Derive pixel scale from WCS header (arcsec/pixel)
    pixel_scale = 1.0
    for key in ("CD2_2", "CDELT2"):
        if key in header:
            val = abs(float(header[key]))
            if val > 0.0:
                # If WCS uses degrees, convert; if already arcsec, keep
                pixel_scale = val * 3600.0 if val < 0.01 else val
                break

    return image, pixel_scale


# ---------------------------------------------------------------------------
# Cross-condition comparison figure
# ---------------------------------------------------------------------------

def plot_harmonic_comparison_qa(
    galaxy: str,
    image: np.ndarray,
    mask: np.ndarray | None,
    condition_results: dict[str, list[dict[str, Any]]],
    *,
    filename: str = "comparison_qa.png",
) -> None:
    """All-condition comparison figure for a single galaxy/band.

    Layout
    ------
    *Top row* — 6 small image panels, one per condition, with isophote overlays
    and mask overlay.

    *Bottom section* — 6 profile panels sharing the same SMA^0.25 x-axis,
    one colour per condition:
      1. Surface brightness log10(I)
      2. a4 / intens (key boxy/disky discriminator)
      3. A4 amplitude sqrt(a4² + b4²) / intens
      4. A3 amplitude sqrt(a3² + b3²) / intens (asymmetry)
      5. Axis ratio b/a
      6. PA (degrees)

    Parameters
    ----------
    galaxy : str
        Galaxy name string (used in title).
    image : 2D array
        Science image (for display).
    mask : 2D bool array or None
        Bad-pixel mask overlay (True = bad).
    condition_results : dict
        Mapping condition_label → list of isophote dicts from ``fit_image()``.
    filename : str
        Output path.
    """
    configure_qa_plot_style()

    conditions = [c for c in CONDITION_ORDER if c in condition_results]
    n_conditions = len(conditions)

    # --------------------------------------------------------------------------
    # Figure layout: top row (images) + 6 profile rows
    # --------------------------------------------------------------------------
    n_profile_rows = 6
    fig_height = 3.0 * (1 + n_conditions // 3) + 1.5 * n_profile_rows
    fig = plt.figure(figsize=(14.0, fig_height))

    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[1.6, float(n_profile_rows)],
        hspace=0.12,
    )

    # Top: 6 image panels in one row
    top = gridspec.GridSpecFromSubplotSpec(
        1, n_conditions, subplot_spec=outer[0], wspace=0.05,
    )

    # Bottom: profile panels
    profile_height_ratios = [2.5, 1.2, 1.2, 1.2, 1.2, 1.2]
    bottom = gridspec.GridSpecFromSubplotSpec(
        n_profile_rows, 1, subplot_spec=outer[1],
        height_ratios=profile_height_ratios, hspace=0.0,
    )

    fig.suptitle(
        f"{galaxy.upper()}  —  harmonic conditions comparison",
        fontsize=16, y=0.995,
    )

    # Pre-compute arcsinh parameters from the full image once
    ref_low, ref_high, ref_scale, ref_vmax = derive_arcsinh_parameters(image)
    img_display, img_vmin, img_vmax = make_arcsinh_display_from_parameters(
        image, low=ref_low, high=ref_high, scale=ref_scale, vmax=ref_vmax,
    )

    # ------------------------------------------------------------------
    # Top row: image panels
    # ------------------------------------------------------------------
    for col, cond in enumerate(conditions):
        ax_im = fig.add_subplot(top[0, col])
        ax_im.imshow(
            img_display, origin="lower", cmap="viridis",
            vmin=img_vmin, vmax=img_vmax, interpolation="none",
        )
        if mask is not None:
            mask_rgba = np.zeros((*image.shape, 4))
            mask_rgba[mask] = [1, 0, 0, 0.35]
            ax_im.imshow(mask_rgba, origin="lower")

        res = condition_results[cond]
        step = max(1, len(res) // 12)
        draw_isophote_overlays(
            ax_im, res, step=step, line_width=0.9, alpha=0.8,
            edge_color=CONDITION_COLORS[cond],
        )
        ax_im.set_title(CONDITION_DISPLAY[cond], fontsize=9, pad=2)
        ax_im.set_xticks([])
        ax_im.set_yticks([])

    # ------------------------------------------------------------------
    # Helper to extract arrays from isophote list
    # ------------------------------------------------------------------
    def _arr(isos, key, default=np.nan):
        return np.array([r.get(key, default) for r in isos])

    # Build profile axes
    ax_sb = fig.add_subplot(bottom[0])
    ax_a4_frac = fig.add_subplot(bottom[1], sharex=ax_sb)
    ax_amp4 = fig.add_subplot(bottom[2], sharex=ax_sb)
    ax_amp3 = fig.add_subplot(bottom[3], sharex=ax_sb)
    ax_ba = fig.add_subplot(bottom[4], sharex=ax_sb)
    ax_pa_panel = fig.add_subplot(bottom[5], sharex=ax_sb)

    legend_handles = []

    for cond in conditions:
        res = condition_results[cond]
        sma = _arr(res, "sma")
        intens = _arr(res, "intens")
        intens_err = _arr(res, "intens_err")
        eps = _arr(res, "eps")
        pa_rad = _arr(res, "pa")
        a3 = _arr(res, "a3")
        b3 = _arr(res, "b3")
        a4 = _arr(res, "a4")
        b4 = _arr(res, "b4")
        stop = _arr(res, "stop_code", 0).astype(int)

        xax = sma ** 0.25
        vm = np.isfinite(xax) & (sma > 1.0)
        safe_i = np.where(np.isfinite(intens) & (intens > 0.0), intens, np.nan)

        col = CONDITION_COLORS[cond]
        mrk = CONDITION_MARKERS[cond]
        label = CONDITION_DISPLAY[cond]

        good = vm & (stop == 0)

        scatter_kw = dict(s=16, marker=mrk, alpha=0.75)

        # 1. Surface brightness
        sb_ok = vm & np.isfinite(intens) & (intens > 0)
        y_sb = np.full_like(intens, np.nan)
        y_sb[sb_ok] = np.log10(intens[sb_ok])
        ax_sb.scatter(xax[sb_ok], y_sb[sb_ok],
                      facecolors=col, edgecolors=col, **scatter_kw, label=label)

        # 2. a4 / intens
        a4f = a4 / safe_i
        a4_ok = vm & np.isfinite(a4f)
        ax_a4_frac.scatter(xax[a4_ok], a4f[a4_ok],
                           facecolors=col, edgecolors=col, **scatter_kw)

        # 3. A4 amplitude
        amp4 = np.sqrt(
            np.where(np.isfinite(a4), a4, 0.0) ** 2
            + np.where(np.isfinite(b4), b4, 0.0) ** 2
        ) / safe_i
        amp4[~(np.isfinite(a4) & np.isfinite(b4))] = np.nan
        amp4_ok = vm & np.isfinite(amp4)
        ax_amp4.scatter(xax[amp4_ok], amp4[amp4_ok],
                        facecolors=col, edgecolors=col, **scatter_kw)

        # 4. A3 amplitude
        amp3 = np.sqrt(
            np.where(np.isfinite(a3), a3, 0.0) ** 2
            + np.where(np.isfinite(b3), b3, 0.0) ** 2
        ) / safe_i
        amp3[~(np.isfinite(a3) & np.isfinite(b3))] = np.nan
        amp3_ok = vm & np.isfinite(amp3)
        ax_amp3.scatter(xax[amp3_ok], amp3[amp3_ok],
                        facecolors=col, edgecolors=col, **scatter_kw)

        # 5. Axis ratio
        ba = 1.0 - eps
        ax_ba.scatter(xax[vm & good], ba[vm & good],
                      facecolors=col, edgecolors=col, **scatter_kw)

        # 6. PA
        pa_deg = normalize_pa_degrees(np.degrees(pa_rad))
        ax_pa_panel.scatter(xax[vm & good], pa_deg[vm & good],
                            facecolors=col, edgecolors=col, **scatter_kw)

        legend_handles.append(
            Line2D([], [], marker=mrk, linestyle="None",
                   color=col, markersize=6, label=label)
        )

    # Axis labels / formatting
    ax_sb.set_ylabel(r"$\log_{10}(I)$")
    ax_sb.set_title("Surface brightness")
    ax_a4_frac.set_ylabel(r"$a_4 / I$")
    ax_a4_frac.axhline(0.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
    ax_a4_frac.text(0.01, 0.95, "disky", color="gray", fontsize=8,
                    transform=ax_a4_frac.transAxes, va="top")
    ax_a4_frac.text(0.01, 0.05, "boxy", color="gray", fontsize=8,
                    transform=ax_a4_frac.transAxes, va="bottom")
    ax_amp4.set_ylabel(r"$A_4 / I$")
    ax_amp3.set_ylabel(r"$A_3 / I$")
    ax_ba.set_ylabel("axis ratio")
    ax_pa_panel.set_ylabel("PA [deg]")
    ax_pa_panel.set_xlabel(r"SMA$^{0.25}$ (pixel$^{0.25}$)")

    for ax in [ax_sb, ax_a4_frac, ax_amp4, ax_amp3, ax_ba]:
        ax.tick_params(labelbottom=False)
        ax.grid(alpha=0.2)
    ax_pa_panel.grid(alpha=0.2)

    # Collect x data for x-limit
    all_xvals = []
    for cond in conditions:
        res = condition_results[cond]
        sma = _arr(res, "sma")
        vm = np.isfinite(sma) & (sma > 1.0)
        if np.any(vm):
            all_xvals.append((sma[vm] ** 0.25))

    if all_xvals:
        xcat = np.concatenate(all_xvals)
        set_x_limits_with_right_margin(ax_pa_panel, xcat)

    # Robust y-limits (ignore outlier error-bar-driven extremes)
    for ax, collect_fn, clip_lo, clip_hi in [
        (ax_sb, lambda c, r: np.log10(np.maximum(_arr(r, "intens"), 1e-30)),
         None, None),
        (ax_ba, lambda c, r: 1.0 - _arr(r, "eps"), 0.0, 1.0),
    ]:
        vals = []
        for cond in conditions:
            res = condition_results[cond]
            try:
                v = collect_fn(cond, res)
                vm = np.isfinite(v)
                if np.any(vm):
                    vals.append(v[vm])
            except Exception:  # noqa: BLE001
                pass
        if vals:
            cat = np.concatenate(vals)
            set_axis_limits_from_finite_values(
                ax, cat, margin_fraction=0.06, min_margin=0.05,
                lower_clip=clip_lo, upper_clip=clip_hi,
            )

    # PA limits from stop=0 points only
    pa_vals = []
    for cond in conditions:
        res = condition_results[cond]
        sma = _arr(res, "sma")
        stop = _arr(res, "stop_code", 0).astype(int)
        pa_rad = _arr(res, "pa")
        vm = np.isfinite(sma) & (sma > 1.0) & (stop == 0)
        pa_deg = normalize_pa_degrees(np.degrees(pa_rad))
        if np.any(vm & np.isfinite(pa_deg)):
            pa_vals.append(pa_deg[vm & np.isfinite(pa_deg)])
    if pa_vals:
        all_pa = np.concatenate(pa_vals)
        lo, hi = robust_limits(all_pa, 3, 97)
        margin = max(3.0, 0.08 * (hi - lo + 1e-6))
        ax_pa_panel.set_ylim(lo - margin, hi + margin)

    # Legend on SB panel
    ax_sb.legend(
        handles=legend_handles, loc="upper right", fontsize=9,
        ncol=2, frameon=True,
    )

    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.05, top=0.970, hspace=0.04)
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved comparison QA figure to {filename}")
