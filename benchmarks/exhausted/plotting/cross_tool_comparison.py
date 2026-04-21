"""Per-galaxy cross-tool comparison QA figure.

For a single galaxy, overlays the "winning" arm of each enabled tool
(``isoster::ref_default`` + ``photutils::baseline_median`` +
``autoprof::baseline`` by default) using
:func:`isoster.plotting.plot_comparison_qa_figure`.

The winner selection prefers the tool's canonical default arm; if that
arm is not in the inventory (or is skipped/errored), falls back to the
tool's lowest-``composite_score`` ok row. Returns ``None`` and writes
nothing when fewer than 2 tools have a usable row.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from isoster.plotting import build_method_profile, plot_comparison_qa_figure

DEFAULT_ARM_PER_TOOL: dict[str, str] = {
    "isoster": "ref_default",
    "photutils": "baseline_median",
    "autoprof": "baseline",
}


def plot_cross_tool_comparison(
    inventories: dict[str, list[dict[str, Any]]],
    output_path: Path,
    *,
    image: np.ndarray,
    mask: np.ndarray | None = None,
    sb_zeropoint: float | None = None,
    pixel_scale_arcsec: float | None = None,
    title: str | None = None,
) -> Path | None:
    """Render the cross-tool comparison figure.

    ``inventories`` maps ``tool_name -> [rows]`` for one galaxy.
    Selects one representative row per tool, loads its
    ``profile.fits`` into a list of isophote dicts, and calls
    :func:`plot_comparison_qa_figure` with up to three methods.
    """
    selected: dict[str, list[dict[str, Any]]] = {}
    arm_used: dict[str, str] = {}
    for tool, rows in inventories.items():
        chosen = _choose_row(tool, rows)
        if chosen is None:
            continue
        profile_path = chosen.get("profile_path") or ""
        if not profile_path or not Path(str(profile_path)).is_file():
            continue
        iso_dicts = _load_profile_as_iso_dicts(profile_path)
        if not iso_dicts:
            continue
        selected[tool] = iso_dicts
        arm_used[tool] = str(chosen.get("arm_id", "?"))

    if len(selected) < 2:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert isophote lists to standardized profile dicts.
    profiles: dict[str, dict[str, np.ndarray]] = {}
    for tool, iso_list in selected.items():
        profile = build_method_profile(iso_list)
        if profile is not None:
            profiles[tool] = profile
    if len(profiles) < 2:
        return None

    computed_title = title or "Cross-tool comparison: " + ", ".join(
        f"{t}::{arm_used[t]}" for t in profiles
    )
    try:
        plot_comparison_qa_figure(
            image=np.asarray(image, dtype=np.float64),
            profiles=profiles,
            title=computed_title,
            output_path=str(output_path),
            mask=mask,
        )
    except Exception as exc:  # noqa: BLE001 - QA must not abort
        output_path.with_suffix(".png.err.txt").write_text(
            f"{type(exc).__name__}: {exc}\n"
        )
        return None
    return output_path


def _choose_row(
    tool: str, rows: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Prefer the tool's default arm; fall back to the lowest-score ok row."""
    default_arm = DEFAULT_ARM_PER_TOOL.get(tool)
    ok_rows = [
        r for r in rows
        if str(r.get("status", "")).lower() in {"ok", "cached"}
        and float(r.get("flag_severity_max", 0.0) or 0.0) < 2
    ]
    if default_arm:
        for r in ok_rows:
            if str(r.get("arm_id", "")) == default_arm:
                return r
    if not ok_rows:
        return None
    ok_rows.sort(
        key=lambda r: float(r.get("composite_score", float("inf")) or float("inf"))
    )
    return ok_rows[0]


def _load_profile_as_iso_dicts(path: str) -> list[dict[str, Any]]:
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
            return []
        data = table_hdu.data
        n = len(data)
        cols = data.names
        out: list[dict[str, Any]] = []
        for i in range(n):
            row: dict[str, Any] = {}
            for c in cols:
                value = data[c][i]
                try:
                    row[c.lower()] = (
                        float(value) if np.isscalar(value) else value
                    )
                except (TypeError, ValueError):
                    row[c.lower()] = value
            out.append(row)
        return out
