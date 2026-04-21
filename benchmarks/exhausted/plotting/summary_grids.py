"""Multi-galaxy summary grids per (tool, arm).

Two grids per arm:

- ``summary_profiles_<arm>.png``: N-column grid of SB profiles across
  every galaxy, colored uniformly per galaxy so the user can
  eyeball where this arm works vs. where it breaks.
- ``summary_residuals_<arm>.png``: N-column grid of the image /
  residual heatmap pairs, one column per galaxy.

Ported in spirit from ``sga_isoster/scripts/photutils_plots.py``; the
layout is simplified so it stays readable with a handful of galaxies
(typical for smoke tests) and scales to dozens once real datasets land.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


def plot_summary_profiles(
    rows_by_galaxy: dict[str, list[dict[str, Any]]],
    arm_id: str,
    output_path: Path,
    *,
    sb_zeropoint: float | None = None,
    pixel_scale_arcsec: float | None = None,
    max_cols: int = 4,
) -> Path | None:
    """Plot SB profile for one arm across all galaxies in a grid."""
    entries: list[tuple[str, dict[str, Any]]] = []
    for galaxy_id, rows in rows_by_galaxy.items():
        row = _find_row(rows, arm_id)
        if row is None:
            continue
        entries.append((galaxy_id, row))
    if not entries:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ncols = min(max_cols, len(entries))
    nrows = int(math.ceil(len(entries) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.6 * ncols, 2.8 * nrows),
        constrained_layout=True,
        squeeze=False,
    )

    for idx, (galaxy_id, row) in enumerate(entries):
        ax = axes[idx // ncols][idx % ncols]
        title = f"{galaxy_id}"
        status = str(row.get("status", ""))
        if status != "ok":
            title += f"\n({status})"
        ax.set_title(title, fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.set_xlabel(r"$\mathrm{sma}^{1/4}$")
        ax.set_ylabel(r"$\mu$ (mag/arcsec$^2$)" if sb_zeropoint is not None else r"$\log_{10}(I)$")
        profile_path = row.get("profile_path")
        if not profile_path:
            ax.text(0.5, 0.5, "no profile", transform=ax.transAxes, ha="center")
            continue
        try:
            profile = _load_profile(profile_path)
        except FileNotFoundError:
            ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center")
            continue
        sma = np.asarray(profile["sma"], dtype=float)
        intens = np.asarray(profile["intens"], dtype=float)
        stop_codes = np.asarray(profile.get("stop_code", np.zeros_like(sma)), dtype=int)
        sma_q = np.power(np.clip(sma, 0.0, None), 0.25)
        mu = _to_surface_brightness(intens, sb_zeropoint, pixel_scale_arcsec)
        for code, color in (
            (0, "#1f77b4"),
            (1, "#ff7f0e"),
            (2, "#2ca02c"),
            (3, "#d62728"),
            (-1, "#9467bd"),
        ):
            mask = stop_codes == code
            if mask.any():
                ax.scatter(sma_q[mask], mu[mask], s=10, color=color, label=f"stop={code}")
        ax.invert_yaxis()
        if idx == 0:
            ax.legend(fontsize=6, loc="upper right")

    # Hide unused grid cells
    for idx in range(len(entries), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f"Arm: {arm_id} — SB profiles across galaxies")
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def plot_summary_residuals(
    rows_by_galaxy: dict[str, list[dict[str, Any]]],
    arm_id: str,
    output_path: Path,
    *,
    max_cols: int = 4,
) -> Path | None:
    """Plot image + residual pairs for one arm across all galaxies."""
    entries: list[tuple[str, dict[str, Any]]] = []
    for galaxy_id, rows in rows_by_galaxy.items():
        row = _find_row(rows, arm_id)
        if row is None:
            continue
        model_path = row.get("model_path")
        if not model_path or not Path(str(model_path)).is_file():
            continue
        entries.append((galaxy_id, row))
    if not entries:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ncols = min(max_cols, len(entries))
    nrows = int(math.ceil(len(entries) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 3.2 * nrows),
        constrained_layout=True,
        squeeze=False,
    )

    for idx, (galaxy_id, row) in enumerate(entries):
        ax = axes[idx // ncols][idx % ncols]
        ax.set_title(galaxy_id, fontsize=9)
        ax.axis("off")
        try:
            with fits.open(row["model_path"]) as hdul:
                residual = None
                for hdu in hdul[1:]:
                    if getattr(hdu, "name", "").upper() == "RESIDUAL":
                        residual = np.asarray(hdu.data, dtype=np.float32)
                        break
                if residual is None:
                    continue
        except (FileNotFoundError, OSError):
            continue
        vmax = float(np.nanpercentile(np.abs(residual), 97)) if residual.size > 0 else 1.0
        vmax = max(vmax, 1e-6)
        ax.imshow(
            residual,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=+vmax,
            origin="lower",
        )

    for idx in range(len(entries), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f"Arm: {arm_id} — residual maps (red=model under, blue=over)")
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_row(rows: list[dict[str, Any]], arm_id: str) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get("arm_id", "")) == arm_id:
            return row
    return None


def _load_profile(path: str) -> dict[str, np.ndarray]:
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
            raise FileNotFoundError(f"no isophote table in {path}")
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
