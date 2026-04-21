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

SCORE_OK_COLOR = "#1f77b4"
SCORE_WARN_COLOR = "#ff7f0e"
SCORE_ERROR_COLOR = "#c62828"
SCORE_SKIPPED_COLOR = "#9e9e9e"


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

    # Per-arm color — stable across panels. Use tab20 palette.
    palette = _palette(len(plottable))
    color_by_arm: dict[str, Any] = {
        str(r["arm_id"]): palette[i] for i, r in enumerate(plottable)
    }

    # --- figure ----------------------------------------------------------
    fig = plt.figure(figsize=(18.5, 11.5))
    outer = gridspec.GridSpec(
        1, 2, figure=fig, width_ratios=[1.0, 1.85], wspace=0.22
    )

    ax_score = fig.add_subplot(outer[0, 0])

    right = gridspec.GridSpecFromSubplotSpec(
        4,
        1,
        subplot_spec=outer[0, 1],
        height_ratios=[2.2, 1.0, 1.0, 1.1],
        hspace=0.0,
    )
    ax_sb = fig.add_subplot(right[0])
    ax_cen = fig.add_subplot(right[1], sharex=ax_sb)
    ax_ba = fig.add_subplot(right[2], sharex=ax_sb)
    ax_pa = fig.add_subplot(right[3], sharex=ax_sb)
    right_axes = [ax_sb, ax_cen, ax_ba, ax_pa]

    if title:
        fig.suptitle(title, fontsize=20, y=0.995)

    # --- scoreboard ------------------------------------------------------
    _draw_scoreboard(ax_score, score_ordered, color_by_arm)

    # --- overlay loop ----------------------------------------------------
    all_x: list[np.ndarray] = []
    sb_values: list[float] = []
    dx_values: list[float] = []
    dy_values: list[float] = []
    ba_values: list[float] = []
    pa_values_finite: list[float] = []

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

        # Center offsets relative to median x0/y0 — matches per-arm QA
        x0 = np.asarray(profile["x0"], dtype=float)[keep]
        y0 = np.asarray(profile["y0"], dtype=float)[keep]
        med_x0 = float(np.nanmedian(x0))
        med_y0 = float(np.nanmedian(y0))
        dx = x0 - med_x0
        dy = y0 - med_y0
        _plot_line(ax_cen, x, dx, color, f"{arm_id} dx", linestyle="-")
        _plot_line(ax_cen, x, dy, color, f"{arm_id} dy", linestyle="--")
        dx_values.extend(dx[np.isfinite(dx)].tolist())
        dy_values.extend(dy[np.isfinite(dy)].tolist())

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

    # --- panel cosmetics -------------------------------------------------
    # SB panel
    ax_sb.set_ylabel(
        r"$\mu$ [mag/arcsec$^2$]" if sb_zeropoint is not None else r"$\log_{10}(I)$"
    )
    ax_sb.set_title("Surface brightness profile")
    ax_sb.grid(alpha=0.25)
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

    # Center offset
    ax_cen.axhline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
    ax_cen.set_ylabel("center offset [pix]")
    ax_cen.grid(alpha=0.25)
    cen_vals = np.asarray(dx_values + dy_values, dtype=float)
    if cen_vals.size:
        set_axis_limits_from_finite_values(
            ax_cen, cen_vals, margin_fraction=0.08, min_margin=0.3
        )
    ax_cen.legend(
        handles=[
            Line2D([], [], color="black", linestyle="-", label="dx"),
            Line2D([], [], color="black", linestyle="--", label="dy"),
        ],
        loc="upper right",
        fontsize=12,
        ncol=2,
    )

    # Axis ratio
    ax_ba.set_ylabel(r"axis ratio $1-\epsilon$")
    ax_ba.grid(alpha=0.25)
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
    ax_pa.set_ylabel("PA [deg]")
    ax_pa.grid(alpha=0.25)
    pa_arr = np.asarray(pa_values_finite, dtype=float)
    if pa_arr.size > 1:
        pa_low, pa_high = robust_limits(pa_arr, 3, 97)
        pa_margin = max(3.0, 0.08 * (pa_high - pa_low + 1e-6))
        ax_pa.set_ylim(pa_low - pa_margin, pa_high + pa_margin)

    # x-axis label on the bottom panel only
    for ax in right_axes[:-1]:
        ax.tick_params(labelbottom=False)
    right_axes[-1].set_xlabel(r"SMA$^{0.25}$ (pixel$^{0.25}$)")
    if all_x:
        xs = np.concatenate(all_x)
        set_x_limits_with_right_margin(right_axes[-1], xs)
        # Clamp the lower limit to the SMA^0.25 floor.
        lo, hi = right_axes[-1].get_xlim()
        right_axes[-1].set_xlim(max(lo, X_AXIS_MIN), hi)

    # Final layout
    fig.subplots_adjust(
        left=0.035, right=0.99, bottom=0.05, top=0.955, wspace=0.2, hspace=0.0
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

    Each bar shows the composite score on a capped-linear axis, with a
    text annotation for score / n_iso / flags next to the arm label.
    Error-flagged arms are red; skipped arms grey; normal arms use the
    per-arm color (matching the right-column lines).
    """
    n = len(rows)
    y_positions = np.arange(n)
    labels: list[str] = []
    bar_values: list[float] = []
    bar_colors: list[Any] = []
    annotations: list[str] = []
    flag_annotations: list[str] = []

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

        if status not in ("ok", "cached"):
            colour = SCORE_SKIPPED_COLOR
            bar_value = score_cap
            score_text = f"{status.upper()}"
        elif severity >= 2:
            colour = SCORE_ERROR_COLOR
            bar_value = score_cap
            score_text = f"ERR  {_fmt_score(score)}"
        elif severity >= 1:
            colour = SCORE_WARN_COLOR
            bar_value = min(score, score_cap)
            score_text = _fmt_score(score)
        else:
            colour = color_by_arm.get(arm_id, SCORE_OK_COLOR)
            bar_value = min(score, score_cap)
            score_text = _fmt_score(score)

        labels.append(arm_id)
        bar_values.append(bar_value)
        bar_colors.append(colour)
        annotations.append(score_text)
        flag_annotations.append(_compact_flags(r.get("flags", "")))

    ax.barh(
        y_positions,
        bar_values,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.3,
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, score_cap)
    ax.set_xlabel("composite score (lower = better)")
    ax.set_title(
        "Arm ranking — grey=skipped, red=error, orange=warn",
        fontsize=12,
    )
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.tick_params(axis="y", length=0)

    # Inline annotations: score (right of bar) + flags (even further right)
    x_right = score_cap
    for y, bar_v, score_text, flag_text in zip(
        y_positions, bar_values, annotations, flag_annotations
    ):
        tx = min(bar_v + 0.02 * score_cap, x_right * 0.62)
        ax.text(
            tx,
            y,
            score_text,
            va="center",
            ha="left",
            fontsize=10,
            color="black",
        )
        if flag_text:
            ax.text(
                x_right * 0.99,
                y,
                flag_text,
                va="center",
                ha="right",
                fontsize=8,
                color="#c62828",
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
        linewidth=0.9,
        alpha=0.85,
        marker=".",
        markersize=3,
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
    import matplotlib.cm as cm

    base = cm.get_cmap("tab20", max(n, 1))
    return [base(i % base.N) for i in range(n)]
