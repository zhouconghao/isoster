"""Per-galaxy cross-arm overlay figure.

Overlays all isoster arms for one galaxy onto a compact 5-panel
figure (one SB profile, three geometry traces, one score bar) so the
user can eyeball arm behavior without opening per-arm PNGs one by one.

Panels (left-to-right, top-to-bottom):

  1. Surface brightness in mag/arcsec^2 vs sma^(1/4)
  2. Ellipticity eps vs sma
  3. Position angle (deg, 180-unwrapped) vs sma
  4. Center drift (sqrt((x0-x0_ref)^2 + (y0-y0_ref)^2)) vs sma
  5. Horizontal bar chart of composite_score per arm (lower is better)

``matplotlib.use("Agg")`` is set by the plotting package __init__.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


def plot_cross_arm_overlay(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    sb_zeropoint: float | None = None,
    pixel_scale_arcsec: float | None = None,
    title: str | None = None,
    max_arms: int = 30,
) -> Path | None:
    """Render the 5-panel overlay to ``output_path``.

    ``rows`` must carry at least ``arm_id``, ``status``,
    ``profile_path``, and ``composite_score``. Rows that are skipped
    or errored contribute a bar to panel 5 but no curve elsewhere.
    Returns ``output_path`` on success, ``None`` if every row lacked
    an on-disk profile.
    """
    ok_rows = [r for r in rows if _has_profile(r)]
    if not ok_rows:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ok_rows.sort(key=lambda r: (_score(r), str(r.get("arm_id", ""))))
    if len(ok_rows) > max_arms:
        ok_rows = ok_rows[:max_arms]
    arm_ids = [str(r["arm_id"]) for r in ok_rows]
    palette = _palette(len(arm_ids))
    color_by_arm = dict(zip(arm_ids, palette))

    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)
    ax_sb = fig.add_subplot(gs[0, 0:2])
    ax_eps = fig.add_subplot(gs[0, 2])
    ax_pa = fig.add_subplot(gs[1, 2])
    ax_center = fig.add_subplot(gs[1, 0:2])
    ax_score = fig.add_subplot(gs[2, :])

    # Reference center taken from the best-score row (same galaxy, so the
    # adapter's initial center would work too, but we re-use the best fit's
    # recovered (x0, y0) to keep drift plotting meaningful).
    x0_ref, y0_ref = _get_reference_center(ok_rows[0])

    for row in ok_rows:
        arm_id = str(row["arm_id"])
        color = color_by_arm[arm_id]
        try:
            profile = _load_profile(row["profile_path"])
        except FileNotFoundError:
            continue
        if len(profile) == 0:
            continue

        sma = np.asarray(profile["sma"], dtype=float)
        intens = np.asarray(profile["intens"], dtype=float)
        eps = np.asarray(profile["eps"], dtype=float)
        pa_rad = np.asarray(profile["pa"], dtype=float)
        x0 = np.asarray(profile["x0"], dtype=float)
        y0 = np.asarray(profile["y0"], dtype=float)

        # SB panel
        mu = _to_surface_brightness(intens, sb_zeropoint, pixel_scale_arcsec)
        sma_quarter = np.power(np.clip(sma, 0.0, None), 0.25)
        ax_sb.plot(sma_quarter, mu, marker=".", linewidth=0.8, color=color, label=arm_id)

        # Eps / PA
        ax_eps.plot(sma, eps, marker=".", linewidth=0.8, color=color)
        pa_deg = _unwrap_pa_deg(pa_rad)
        ax_pa.plot(sma, pa_deg, marker=".", linewidth=0.8, color=color)

        # Drift
        drift = np.sqrt((x0 - x0_ref) ** 2 + (y0 - y0_ref) ** 2)
        ax_center.plot(sma, drift, marker=".", linewidth=0.8, color=color)

    # SB axis conventions: invert y because fainter is higher magnitude.
    ax_sb.invert_yaxis()
    ax_sb.set_xlabel(r"$\mathrm{sma}^{1/4}$ (pix$^{1/4}$)")
    if sb_zeropoint is not None:
        ax_sb.set_ylabel(r"$\mu$ (mag/arcsec$^2$)")
    else:
        ax_sb.set_ylabel(r"$\log_{10}(I)$")
    ax_sb.grid(True, linestyle=":", alpha=0.4)

    ax_eps.set_xlabel("sma (pix)")
    ax_eps.set_ylabel(r"$\epsilon$")
    ax_eps.grid(True, linestyle=":", alpha=0.4)

    ax_pa.set_xlabel("sma (pix)")
    ax_pa.set_ylabel("PA (deg)")
    ax_pa.grid(True, linestyle=":", alpha=0.4)

    ax_center.set_xlabel("sma (pix)")
    ax_center.set_ylabel("center drift (pix)")
    ax_center.set_yscale("symlog", linthresh=0.1)
    ax_center.grid(True, linestyle=":", alpha=0.4)

    # Score bar chart over ALL rows, not just those with profiles.
    all_rows = sorted(
        rows, key=lambda r: (_score(r), str(r.get("arm_id", "")))
    )
    score_ids = [str(r["arm_id"]) for r in all_rows]
    scores = np.array([_score(r) for r in all_rows], dtype=float)
    # Cap huge penalty scores so the bar chart stays readable.
    finite_scores = scores[np.isfinite(scores) & (scores < 1e5)]
    cap = float(finite_scores.max()) if finite_scores.size > 0 else 1.0
    plot_scores = np.where(scores > cap * 5, cap * 5, scores)
    bar_colors = [
        color_by_arm.get(aid, "#888888") if _score(row) < 1e5 else "#cc3333"
        for aid, row in zip(score_ids, all_rows)
    ]
    ax_score.barh(score_ids, plot_scores, color=bar_colors)
    ax_score.invert_yaxis()
    ax_score.set_xlabel("composite score (lower is better; red = error-flagged)")
    ax_score.grid(True, axis="x", linestyle=":", alpha=0.4)

    ax_sb.legend(ncol=3, fontsize=7, loc="lower left")
    if title:
        fig.suptitle(title)

    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_profile(row: dict[str, Any]) -> bool:
    path = row.get("profile_path")
    if not path:
        return False
    try:
        return Path(path).is_file()
    except TypeError:
        return False


def _score(row: dict[str, Any]) -> float:
    value = row.get("composite_score")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("inf")
    if value != value:  # NaN
        return float("inf")
    return value


def _load_profile(path: str) -> dict[str, np.ndarray]:
    with fits.open(path) as hdul:
        # ISOPHOTES HDU or first table HDU
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
            raise FileNotFoundError(f"no isophote table in {path}")
        data = table_hdu.data
        return {col.lower(): np.asarray(data[col]) for col in data.names}


def _to_surface_brightness(
    intens: np.ndarray,
    zeropoint: float | None,
    pixel_scale_arcsec: float | None,
) -> np.ndarray:
    if zeropoint is None or pixel_scale_arcsec is None:
        # Fallback: log10 of intensity (intens already per pixel).
        out = np.full_like(intens, np.nan, dtype=float)
        mask = intens > 0
        out[mask] = np.log10(intens[mask])
        return out
    pixel_area = pixel_scale_arcsec**2
    out = np.full_like(intens, np.nan, dtype=float)
    mask = intens > 0
    out[mask] = -2.5 * np.log10(intens[mask] / pixel_area) + zeropoint
    return out


def _unwrap_pa_deg(pa_rad: np.ndarray) -> np.ndarray:
    pa_deg = np.rad2deg(pa_rad)
    wrapped = np.mod(pa_deg, 180.0)
    doubled = np.deg2rad(2.0 * wrapped)
    unwrapped = np.unwrap(doubled)
    return np.rad2deg(0.5 * unwrapped)


def _get_reference_center(row: dict[str, Any]) -> tuple[float, float]:
    """Read the final-arm ``(x0, y0)`` from the profile's innermost isophote.

    The drift panel uses this as the reference so curves line up at zero
    near the center regardless of which arm is being plotted.
    """
    try:
        profile = _load_profile(row["profile_path"])
    except Exception:  # noqa: BLE001 - fall back to origin
        return 0.0, 0.0
    if "x0" not in profile or "y0" not in profile or len(profile["x0"]) == 0:
        return 0.0, 0.0
    # Pick the row at the smallest non-zero sma when possible.
    sma = np.asarray(profile["sma"], dtype=float)
    mask = sma > 0
    if mask.any():
        idx = np.argmin(sma[mask])
        positive_indices = np.nonzero(mask)[0]
        chosen = positive_indices[idx]
    else:
        chosen = 0
    return float(profile["x0"][chosen]), float(profile["y0"][chosen])


def _palette(n: int) -> list:
    import matplotlib.cm as cm

    base = cm.get_cmap("tab20", max(n, 1))
    return [base(i % base.N) for i in range(n)]
