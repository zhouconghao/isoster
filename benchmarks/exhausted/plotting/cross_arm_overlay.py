"""Per-galaxy cross-arm overlay figure.

Layout mirrors :func:`isoster.plotting.plot_qa_summary` but overlays
every arm for one galaxy rather than showing one arm's 2-D panels.

Figure layout (2 columns):

  +----------------+-------------------------------+
  |                | Surface brightness (tall)     |
  |                +-------------------------------+
  |  Scoreboard    | Center offset (dx, dy)        |
  |  (all arms,    +-------------------------------+
  |  labels +      | Axis ratio (b/a)              |
  |  composite     +-------------------------------+
  |  score)        | Position angle (deg)          |
  +----------------+-------------------------------+

Every right-column panel shares the same ``sma^0.25`` x-axis, starting
at ``1.5^0.25`` (we drop the ``sma <= 1.5`` regime because it is
dominated by the central-pixel bookkeeping isophote and seeding noise).

Recipe elements copied from the per-arm QA:

- Y-axis inverted for SB when zp/pixscale supplied.
- ``1 - eps`` (axis ratio) instead of raw eps.
- ``dx`` / ``dy`` relative to median ``x0, y0`` (not raw drift).
- PA double-angle-unwrapped via :func:`normalize_pa_degrees`.
- Robust y-limits via :func:`set_axis_limits_from_finite_values`.
- X-range + right margin via :func:`set_x_limits_with_right_margin`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.lines import Line2D

from isoster.plotting import (
    configure_qa_plot_style,
    normalize_pa_degrees,
    robust_limits,
    set_axis_limits_from_finite_values,
    set_x_limits_with_right_margin,
)

SMA_MIN_PIX = 1.5
X_AXIS_MIN = SMA_MIN_PIX ** 0.25  # ≈ 1.1067

SCORE_WARN_COLOR = "#ff9800"
SCORE_SKIPPED_COLOR = "#9e9e9e"
ERROR_LABEL_COLOR = "#c62828"


def plot_cross_arm_overlay(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    sb_zeropoint: float | None = None,
    pixel_scale_arcsec: float | None = None,
    title: str | None = None,
) -> Path | None:
    """Render the cross-arm overlay to ``output_path``.

    ``rows`` is the full inventory for one ``(galaxy, tool)`` pair.
    Each row must carry ``arm_id``, ``status``, ``flag_severity_max``,
    ``composite_score``, and a ``profile_path`` for arms that ran
    successfully.

    Returns ``output_path`` on success, ``None`` if no arm has a
    loadable profile (degenerate case — no plot produced).
    """
    if not rows:
        return None
    configure_qa_plot_style()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Order rows: ok/cached by composite_score ascending, then errors
    # (flag_severity_max >= 2), then skipped, at the bottom. This
    # makes the scoreboard readable top-to-bottom.
    score_ordered = sorted(
        rows,
        key=lambda r: (
            _sort_bucket(r),
            _score(r),
            str(r.get("arm_id", "")),
        ),
    )

    # Arms we have profiles for (the ones that draw curves).
    plottable = [r for r in score_ordered if _has_profile(r)]
    if not plottable:
        return None

    # Per-arm color — stable across panels. Bold palette, not tab20.
    palette = _palette(len(score_ordered))
    # Assign in scoreboard order so colors align with ranking.
    color_by_arm: dict[str, Any] = {
        str(r["arm_id"]): palette[i] for i, r in enumerate(score_ordered)
    }

    # Pick the score winner as the reference for the difference panels.
    ok_plottable = [r for r in score_ordered if _has_profile(r)]
    ref_row = ok_plottable[0] if ok_plottable else None
    ref_profile = _load_profile(ref_row["profile_path"]) if ref_row is not None else None

    # --- figure ----------------------------------------------------------
    # Taller figure to host the two new difference panels (6 rows on the right).
    fig = plt.figure(figsize=(20.5, 15.5))
    outer = gridspec.GridSpec(
        1, 2, figure=fig, width_ratios=[1.22, 1.85], wspace=0.12
    )

    ax_score = fig.add_subplot(outer[0, 0])

    right = gridspec.GridSpecFromSubplotSpec(
        6,
        1,
        subplot_spec=outer[0, 1],
        height_ratios=[2.2, 1.0, 1.0, 1.0, 1.3, 1.3],
        hspace=0.0,
    )
    ax_sb = fig.add_subplot(right[0])
    ax_cen = fig.add_subplot(right[1], sharex=ax_sb)
    ax_ba = fig.add_subplot(right[2], sharex=ax_sb)
    ax_pa = fig.add_subplot(right[3], sharex=ax_sb)
    ax_dsb = fig.add_subplot(right[4], sharex=ax_sb)
    ax_dcog = fig.add_subplot(right[5], sharex=ax_sb)
    right_axes = [ax_sb, ax_cen, ax_ba, ax_pa, ax_dsb, ax_dcog]

    if title:
        fig.suptitle(title, fontsize=30, y=0.995)

    # --- scoreboard ------------------------------------------------------
    _draw_scoreboard(ax_score, score_ordered, color_by_arm)

    # --- overlay loop ----------------------------------------------------
    all_x: list[np.ndarray] = []
    sb_values: list[float] = []
    dr_values: list[float] = []
    ba_values: list[float] = []
    pa_values_finite: list[float] = []
    dsb_values: list[float] = []
    dcog_values: list[float] = []

    # Pre-compute reference interpolation helpers for the difference
    # panels. The reference profile is the highest-scoring arm.
    ref_sma = None
    ref_intens = None
    ref_cog = None
    ref_arm_id = ""
    if ref_profile is not None and ref_row is not None:
        ref_arm_id = str(ref_row["arm_id"])
        _ref_sma = np.asarray(ref_profile["sma"], dtype=float)
        _ref_intens = np.asarray(ref_profile["intens"], dtype=float)
        _ref_mask = np.isfinite(_ref_sma) & np.isfinite(_ref_intens) & (_ref_sma > 0)
        if np.any(_ref_mask):
            order = np.argsort(_ref_sma[_ref_mask])
            ref_sma = _ref_sma[_ref_mask][order]
            ref_intens = _ref_intens[_ref_mask][order]
            # Keep strictly monotonic-increasing sma for np.interp.
            uniq_idx = np.concatenate(
                ([True], np.diff(ref_sma) > 0)
            )
            ref_sma = ref_sma[uniq_idx]
            ref_intens = ref_intens[uniq_idx]
            # Build a reference CoG by cumulative elliptical-shell integration.
            if "cog" in ref_profile:
                _ref_cog_raw = np.asarray(ref_profile["cog"], dtype=float)[_ref_mask][
                    order
                ][uniq_idx]
                if np.any(np.isfinite(_ref_cog_raw) & (_ref_cog_raw > 0)):
                    ref_cog = _ref_cog_raw

    for row in plottable:
        arm_id = str(row["arm_id"])
        color = color_by_arm[arm_id]
        try:
            profile = _load_profile(row["profile_path"])
        except (FileNotFoundError, OSError):
            continue
        if profile is None or len(profile.get("sma", [])) == 0:
            continue

        sma = np.asarray(profile["sma"], dtype=float)
        keep = np.isfinite(sma) & (sma > SMA_MIN_PIX)
        if not np.any(keep):
            continue
        x = sma[keep] ** 0.25
        all_x.append(x)

        # SB panel
        intens = np.asarray(profile["intens"], dtype=float)[keep]
        mu = _to_surface_brightness(intens, sb_zeropoint, pixel_scale_arcsec)
        _plot_line(ax_sb, x, mu, color, arm_id)
        sb_values.extend(mu[np.isfinite(mu)].tolist())

        # Center offset combined: dr = sqrt(dx^2 + dy^2)
        x0 = np.asarray(profile["x0"], dtype=float)[keep]
        y0 = np.asarray(profile["y0"], dtype=float)[keep]
        med_x0 = float(np.nanmedian(x0))
        med_y0 = float(np.nanmedian(y0))
        dr = np.sqrt((x0 - med_x0) ** 2 + (y0 - med_y0) ** 2)
        _plot_line(ax_cen, x, dr, color, arm_id)
        dr_values.extend(dr[np.isfinite(dr)].tolist())

        # Axis ratio b/a = 1 - eps
        eps = np.asarray(profile["eps"], dtype=float)[keep]
        ba = 1.0 - eps
        _plot_line(ax_ba, x, ba, color, arm_id)
        ba_values.extend(ba[np.isfinite(ba)].tolist())

        # PA, double-angle-unwrapped degrees
        pa_rad = np.asarray(profile["pa"], dtype=float)[keep]
        pa_deg = normalize_pa_degrees(np.degrees(pa_rad))
        _plot_line(ax_pa, x, pa_deg, color, arm_id)
        pa_values_finite.extend(pa_deg[np.isfinite(pa_deg)].tolist())

        # Relative-difference panels (intensity + CoG) vs. the score winner.
        sma_valid = sma[keep]
        if ref_sma is not None and ref_intens is not None and arm_id != ref_arm_id:
            ref_here = np.interp(
                sma_valid, ref_sma, ref_intens, left=np.nan, right=np.nan
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                d_intens_pct = np.where(
                    (np.abs(ref_here) > 0) & np.isfinite(ref_here),
                    (intens - ref_here) / ref_here * 100.0,
                    np.nan,
                )
            _plot_line(ax_dsb, x, d_intens_pct, color, arm_id)
            dsb_values.extend(d_intens_pct[np.isfinite(d_intens_pct)].tolist())

            if "cog" in profile and ref_cog is not None:
                cog_this = np.asarray(profile["cog"], dtype=float)[keep]
                ref_cog_here = np.interp(
                    sma_valid, ref_sma, ref_cog, left=np.nan, right=np.nan
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    d_cog_pct = np.where(
                        (np.abs(ref_cog_here) > 0) & np.isfinite(ref_cog_here),
                        (cog_this - ref_cog_here) / ref_cog_here * 100.0,
                        np.nan,
                    )
                _plot_line(ax_dcog, x, d_cog_pct, color, arm_id)
                dcog_values.extend(d_cog_pct[np.isfinite(d_cog_pct)].tolist())

    # --- panel cosmetics -------------------------------------------------
    panel_label_fs = 21
    tick_fs = 16
    legend_fs = 16

    # SB panel
    ax_sb.set_ylabel(
        r"$\mu$ [mag/arcsec$^2$]" if sb_zeropoint is not None else r"$\log_{10}(I)$",
        fontsize=panel_label_fs,
    )
    ax_sb.set_title("Surface brightness profile", fontsize=panel_label_fs)
    ax_sb.grid(alpha=0.25)
    ax_sb.tick_params(axis="both", labelsize=tick_fs)
    if sb_zeropoint is not None:
        ax_sb.invert_yaxis()
    if sb_values:
        set_axis_limits_from_finite_values(
            ax_sb,
            np.asarray(sb_values),
            margin_fraction=0.06,
            min_margin=0.2,
            invert=sb_zeropoint is not None,
        )

    # Arm legend: use BOTH low-SMA-faint and high-SMA-bright corners.
    # With y inverted (SB mode), "faint" is top and "bright" is bottom.
    _place_arm_legends(
        ax_sb,
        color_by_arm,
        score_ordered_arms=[str(r["arm_id"]) for r in score_ordered],
        legend_fs=legend_fs,
        sb_inverted=sb_zeropoint is not None,
    )

    # Center offset (combined dr)
    ax_cen.axhline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
    ax_cen.set_ylabel(r"$\Delta r$ [pix]", fontsize=panel_label_fs)
    ax_cen.grid(alpha=0.25)
    ax_cen.tick_params(axis="both", labelsize=tick_fs)
    dr_arr = np.asarray(dr_values, dtype=float)
    if dr_arr.size:
        set_axis_limits_from_finite_values(
            ax_cen, dr_arr, margin_fraction=0.08, min_margin=0.3, lower_clip=0.0
        )

    # Axis ratio
    ax_ba.set_ylabel(r"axis ratio $1-\epsilon$", fontsize=panel_label_fs)
    ax_ba.grid(alpha=0.25)
    ax_ba.tick_params(axis="both", labelsize=tick_fs)
    ba_arr = np.asarray(ba_values, dtype=float)
    if ba_arr.size:
        set_axis_limits_from_finite_values(
            ax_ba,
            ba_arr,
            margin_fraction=0.08,
            min_margin=0.03,
            lower_clip=0.0,
            upper_clip=1.0,
        )

    # PA
    ax_pa.set_ylabel("PA [deg]", fontsize=panel_label_fs)
    ax_pa.grid(alpha=0.25)
    ax_pa.tick_params(axis="both", labelsize=tick_fs)
    pa_arr = np.asarray(pa_values_finite, dtype=float)
    if pa_arr.size > 1:
        pa_low, pa_high = robust_limits(pa_arr, 3, 97)
        pa_margin = max(3.0, 0.08 * (pa_high - pa_low + 1e-6))
        ax_pa.set_ylim(pa_low - pa_margin, pa_high + pa_margin)

    # Relative intensity difference
    ax_dsb.axhline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
    ref_tag = ref_arm_id if ref_arm_id else "ref"
    ax_dsb.set_ylabel(
        r"$\Delta I/I_\mathrm{ref}$ [%]",
        fontsize=panel_label_fs,
    )
    ax_dsb.grid(alpha=0.25)
    ax_dsb.tick_params(axis="both", labelsize=tick_fs)
    dsb_arr = np.asarray(dsb_values, dtype=float)
    if dsb_arr.size > 1:
        lo, hi = robust_limits(dsb_arr, 2, 98)
        span = max(0.5, max(abs(lo), abs(hi)))
        ax_dsb.set_ylim(-span * 1.1, span * 1.1)
    ax_dsb.text(
        0.015,
        0.92,
        f"reference arm: {ref_tag}",
        transform=ax_dsb.transAxes,
        fontsize=legend_fs,
        color="black",
        weight="bold",
        alpha=0.8,
    )

    # Relative CoG difference
    ax_dcog.axhline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
    ax_dcog.set_ylabel(
        r"$\Delta \mathrm{CoG}/\mathrm{CoG}_\mathrm{ref}$ [%]",
        fontsize=panel_label_fs,
    )
    ax_dcog.grid(alpha=0.25)
    ax_dcog.tick_params(axis="both", labelsize=tick_fs)
    dcog_arr = np.asarray(dcog_values, dtype=float)
    if dcog_arr.size > 1:
        lo, hi = robust_limits(dcog_arr, 2, 98)
        span = max(0.5, max(abs(lo), abs(hi)))
        ax_dcog.set_ylim(-span * 1.1, span * 1.1)

    # x-axis label on the bottom panel only
    for ax in right_axes[:-1]:
        ax.tick_params(labelbottom=False)
    right_axes[-1].set_xlabel(
        r"SMA$^{0.25}$ (pixel$^{0.25}$)", fontsize=panel_label_fs
    )
    if all_x:
        xs = np.concatenate(all_x)
        set_x_limits_with_right_margin(right_axes[-1], xs)
        # Clamp the lower limit to the SMA^0.25 floor.
        lo, hi = right_axes[-1].get_xlim()
        right_axes[-1].set_xlim(max(lo, X_AXIS_MIN), hi)

    # Final layout: shrink inter-column gap, widen left to avoid label clip.
    fig.subplots_adjust(
        left=0.10, right=0.985, bottom=0.045, top=0.950, wspace=0.10, hspace=0.0
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Scoreboard
# ---------------------------------------------------------------------------


def _draw_scoreboard(
    ax, rows: list[dict[str, Any]], color_by_arm: dict[str, Any]
) -> None:
    """Horizontal bar chart: one row per arm, sorted by rank.

    Each bar uses the arm's own palette color (matching the right-column
    lines) EXCEPT skipped arms (grey). Flag severity is conveyed by
    annotation style: ``ERROR`` uses bold red text; ``WARN`` uses
    non-bold orange text. This keeps bar color consistent with the
    right-column lines (color = arm identity).
    """
    n = len(rows)
    y_positions = np.arange(n)
    labels: list[str] = []
    bar_values: list[float] = []
    bar_colors: list[Any] = []
    score_texts: list[str] = []
    score_colors: list[str] = []
    score_weights: list[str] = []
    flag_texts: list[str] = []

    finite_scores = [
        _score(r)
        for r in rows
        if _score(r) < 1e5 and np.isfinite(_score(r))
    ]
    score_cap = max(finite_scores) * 1.25 if finite_scores else 1.0
    score_cap = max(score_cap, 1e-3)

    for r in rows:
        arm_id = str(r.get("arm_id", ""))
        status = str(r.get("status", "")).lower()
        severity = _as_float(r.get("flag_severity_max"), 0.0)
        score = _score(r)

        arm_color = color_by_arm.get(arm_id, "#1f77b4")
        if status not in ("ok", "cached"):
            colour = SCORE_SKIPPED_COLOR
            bar_value = score_cap
            score_text = status.upper()
            score_color = "black"
            score_weight = "normal"
        elif severity >= 2:
            # Keep the arm color on the bar; mark the LABEL with bold red.
            colour = arm_color
            bar_value = score_cap
            score_text = f"ERROR  {_fmt_score(score)}"
            score_color = ERROR_LABEL_COLOR
            score_weight = "bold"
        elif severity >= 1:
            colour = arm_color
            bar_value = min(score, score_cap)
            score_text = _fmt_score(score)
            score_color = SCORE_WARN_COLOR
            score_weight = "normal"
        else:
            colour = arm_color
            bar_value = min(score, score_cap)
            score_text = _fmt_score(score)
            score_color = "black"
            score_weight = "normal"

        labels.append(arm_id)
        bar_values.append(bar_value)
        bar_colors.append(colour)
        score_texts.append(score_text)
        score_colors.append(score_color)
        score_weights.append(score_weight)
        flag_texts.append(_compact_flags(r.get("flags", "")))

    ax.barh(
        y_positions,
        bar_values,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.35,
    )
    ax.set_yticks(y_positions)
    # Arm labels: keep SMALL per spec (only axis labels that must stay)
    ax.set_yticklabels(labels, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, score_cap)
    ax.set_xlabel(
        "composite score (lower wins)", fontsize=18
    )
    ax.tick_params(axis="x", labelsize=14)
    ax.set_title(
        "Lower score wins",
        fontsize=22,
        weight="bold",
        pad=10,
    )
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.tick_params(axis="y", length=0)

    # Inline annotations: score (right of bar) + flags (even further right)
    x_right = score_cap
    for y, bar_v, text, tcolor, tweight, flag_text in zip(
        y_positions, bar_values, score_texts, score_colors, score_weights, flag_texts
    ):
        tx = min(bar_v + 0.02 * score_cap, x_right * 0.55)
        ax.text(
            tx,
            y,
            text,
            va="center",
            ha="left",
            fontsize=14,
            color=tcolor,
            weight=tweight,
        )
        if flag_text:
            ax.text(
                x_right * 0.995,
                y,
                flag_text,
                va="center",
                ha="right",
                fontsize=12,
                color=ERROR_LABEL_COLOR,
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_profile(row: dict[str, Any]) -> bool:
    status = str(row.get("status", "")).lower()
    if status not in ("ok", "cached"):
        return False
    path = row.get("profile_path")
    if not path:
        return False
    try:
        return Path(str(path)).is_file()
    except TypeError:
        return False


def _sort_bucket(row: dict[str, Any]) -> int:
    status = str(row.get("status", "")).lower()
    if status not in ("ok", "cached"):
        return 2
    if _as_float(row.get("flag_severity_max"), 0.0) >= 2:
        return 1
    return 0


def _score(row: dict[str, Any]) -> float:
    value = row.get("composite_score")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("inf")
    if not np.isfinite(value):
        return float("inf")
    return value


def _as_float(value: Any, fallback: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return fallback
    if value != value:  # NaN
        return fallback
    return value


def _fmt_score(score: float) -> str:
    if score >= 1e5:
        return f"{score:.1e}"
    if score >= 100:
        return f"{score:.0f}"
    if score >= 10:
        return f"{score:.1f}"
    return f"{score:.2f}"


def _compact_flags(flags_field: Any) -> str:
    if flags_field is None:
        return ""
    text = str(flags_field).strip()
    if not text:
        return ""
    names = [n for n in text.split(",") if n]
    shortmap = {
        "first_isophote_failure": "FIF",
        "no_fit": "NOFIT",
        "few_isophotes": "FEW",
        "high_stopcode": "STOP",
        "any_stop_m1": "m1",
        "center_drift": "drift",
        "pa_instability": "pa",
        "eps_instability": "eps",
        "high_outer_resid_frac": "outR",
    }
    return " ".join(shortmap.get(n, n) for n in names[:5])


def _plot_line(
    ax, x: np.ndarray, y: np.ndarray, color: Any, label: str, linestyle: str = "-"
) -> None:
    ax.plot(
        x,
        y,
        color=color,
        linestyle=linestyle,
        linewidth=1.4,
        alpha=0.9,
        marker=".",
        markersize=4,
    )


def _place_arm_legends(
    ax,
    color_by_arm: dict[str, Any],
    *,
    score_ordered_arms: list[str],
    legend_fs: int,
    sb_inverted: bool,
) -> None:
    """Split the arm legend across two SB corners.

    With SB y-axis inverted (default), "faint" is at the top (high μ)
    and "bright" is at the bottom (low μ). We place:

      - top-right corner (faint end): second half of the arm list
      - bottom-left corner (bright end): first half of the arm list

    Without y inversion (log10 intensity mode) the corner semantics
    swap. Either way the legend lives in two empty SB corners.
    """
    if not score_ordered_arms:
        return
    half = (len(score_ordered_arms) + 1) // 2
    first_group = score_ordered_arms[:half]
    second_group = score_ordered_arms[half:]

    def _handles(arms: list[str]):
        return [
            Line2D(
                [], [], color=color_by_arm.get(a, "#000000"),
                linewidth=2.0, marker=".", markersize=6, label=a,
            )
            for a in arms
        ]

    if sb_inverted:
        # faint = top -> top-right gets the less-important half
        bright_corner_loc = "lower left"
        faint_corner_loc = "upper right"
    else:
        bright_corner_loc = "upper left"
        faint_corner_loc = "lower right"

    if first_group:
        leg1 = ax.legend(
            handles=_handles(first_group),
            loc=bright_corner_loc,
            fontsize=legend_fs,
            ncol=1,
            framealpha=0.85,
            labelspacing=0.28,
            handlelength=1.6,
        )
        ax.add_artist(leg1)
    if second_group:
        ax.legend(
            handles=_handles(second_group),
            loc=faint_corner_loc,
            fontsize=legend_fs,
            ncol=1,
            framealpha=0.85,
            labelspacing=0.28,
            handlelength=1.6,
        )


def _load_profile(path: str) -> dict[str, np.ndarray] | None:
    with fits.open(path) as hdul:
        table_hdu = None
        for hdu in hdul:
            if getattr(hdu, "name", "").upper() == "ISOPHOTES" and hasattr(hdu, "data"):
                table_hdu = hdu
                break
        if table_hdu is None:
            for hdu in hdul[1:]:
                if hasattr(hdu, "data") and hdu.data is not None:
                    table_hdu = hdu
                    break
        if table_hdu is None:
            return None
        data = table_hdu.data
        return {col.lower(): np.asarray(data[col]) for col in data.names}


def _to_surface_brightness(
    intens: np.ndarray,
    zeropoint: float | None,
    pixel_scale_arcsec: float | None,
) -> np.ndarray:
    if zeropoint is None or pixel_scale_arcsec is None:
        out = np.full_like(intens, np.nan, dtype=float)
        mask = intens > 0
        out[mask] = np.log10(intens[mask])
        return out
    pixel_area = pixel_scale_arcsec**2
    out = np.full_like(intens, np.nan, dtype=float)
    mask = intens > 0
    out[mask] = -2.5 * np.log10(intens[mask] / pixel_area) + zeropoint
    return out


def _palette(n: int):
    """Bold, high-contrast palette for ~23 arms.

    Uses Glasbey-style hand-picked distinct colors; falls back to a
    concatenation of tab10 + Set1 + Dark2 if more than 24 arms are
    requested. Tuned for readability against white backgrounds.
    """
    bold = [
        "#e41a1c",  # vivid red
        "#377eb8",  # strong blue
        "#4daf4a",  # grass green
        "#984ea3",  # purple
        "#ff7f00",  # orange
        "#ffff33",  # yellow (sparingly; strong background contrast)
        "#a65628",  # brown
        "#f781bf",  # pink
        "#17becf",  # teal
        "#9467bd",  # lavender
        "#2ca02c",  # emerald
        "#d62728",  # crimson
        "#1f77b4",  # navy blue
        "#8c564b",  # dark brown
        "#bcbd22",  # olive
        "#66c2a5",  # aqua
        "#fc8d62",  # salmon
        "#8da0cb",  # slate
        "#e78ac3",  # rose
        "#a6d854",  # lime
        "#ffd92f",  # gold
        "#e5c494",  # tan
        "#b3b3b3",  # stone
        "#525252",  # charcoal
    ]
    if n <= len(bold):
        return bold[:n]
    # Cycle with slight alpha shifts if we ever exceed 24 arms.
    return [bold[i % len(bold)] for i in range(n)]
