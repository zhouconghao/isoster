"""Cross-arm and cross-tool statistical summaries.

Two roles:

1. Computing a per-row ``composite_score`` from the campaign YAML's
   ``summary.composite_score_weights`` (lower is better). Rows with a
   severity-error flag or non-ok status receive a huge penalty so
   pathological fits never win a ranking.

2. Writing CSV + markdown tables that pivot the inventory into:
   - per-galaxy cross-arm comparison
   - per-tool cross-arm summary (medians across galaxies)

The module does not mutate inventory FITS files on disk — callers feed
inventory rows in, tables come out.

Score axes (all normalized to be order 1 at "typical mild disruption"):

   FIDELITY
     - resid_rms_inner / image_sigma
     - resid_rms_mid   / image_sigma
     - resid_rms_outer / image_sigma  (weighted 2x by default; LSB
       is where arms actually differ)
   CENTER STABILITY
     - combined_drift_pix / centroid_tol_pix  (default tol = 2.0 px)
   CONVERGENCE HEALTH
     - n_stop_m1              (integer count)
     - frac_stop_nonzero      (fraction 0-1)
   COMPLETENESS
     - max(0, 1 - n_iso / n_iso_ref)  (ref = max n_iso across ok arms)
   SPEED
     - wall_time_fit_s        (tie-breaker)

PA and eps drift are explicitly *not* in the score: they can be
genuinely astrophysical (e.g. isophote twisting in ellipticals).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..analysis.inventory import INVENTORY_COLUMNS
from ..analysis.quality_flags import SEVERITY_ERROR

# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[str, float] = {
    # Fidelity
    "resid_inner": 1.0,
    "resid_mid": 1.0,
    "resid_outer": 2.0,
    # Center stability
    "centroid_drift": 1.0,
    "centroid_tol_pix": 2.0,   # divisor, not a weight
    # Convergence health
    "n_stop_m1": 2.0,
    "frac_stop_nonzero": 5.0,
    # Completeness
    "n_iso_completeness": 3.0,
    # Speed
    "wall_time": 0.1,
}

# Rows flagged as ERROR get this penalty added to the score so they are
# never ranked "best" regardless of per-metric values.
ERROR_FLAG_PENALTY = 1_000_000.0

# Divisor floor to keep normalized terms finite on noiseless mocks or
# empty bundles.
MIN_SIGMA_VALUE = 1e-6


def compute_composite_score(
    row: dict[str, Any],
    weights: dict[str, float] | None = None,
    *,
    image_sigma: float = 1.0,
    n_iso_ref: int = 1,
) -> float:
    """Return the composite score for one row. Lower is better.

    ``weights`` keys (all optional; missing keys default to
    :data:`DEFAULT_WEIGHTS`):

        resid_inner, resid_mid, resid_outer,
        centroid_drift, centroid_tol_pix,
        n_stop_m1, frac_stop_nonzero,
        n_iso_completeness,
        wall_time

    The only exception to the per-term sum is when the row itself is
    not ``status in {"ok", "cached"}`` or carries a severity-error
    flag — then the returned score is :data:`ERROR_FLAG_PENALTY` so
    the row sinks to the bottom of any sort.
    """
    weights = weights if weights is not None else DEFAULT_WEIGHTS
    status = str(row.get("status", "")).lower()
    if status not in {"ok", "cached"}:
        return ERROR_FLAG_PENALTY
    severity = _as_float(row.get("flag_severity_max"), 0.0)
    if severity >= SEVERITY_ERROR:
        return ERROR_FLAG_PENALTY

    sigma = max(float(image_sigma), MIN_SIGMA_VALUE)
    centroid_tol = max(float(weights.get("centroid_tol_pix", 2.0)), MIN_SIGMA_VALUE)
    n_iso_ref_eff = max(int(n_iso_ref), 1)

    # FIDELITY -- residuals normalized by image noise
    fidelity = (
        float(weights.get("resid_inner", 1.0))
        * _as_float(row.get("resid_rms_inner"), 0.0)
        / sigma
        + float(weights.get("resid_mid", 1.0))
        * _as_float(row.get("resid_rms_mid"), 0.0)
        / sigma
        + float(weights.get("resid_outer", 2.0))
        * _as_float(row.get("resid_rms_outer"), 0.0)
        / sigma
    )

    # CENTER STABILITY -- centroid drift only (PA and eps excluded)
    centroid = (
        float(weights.get("centroid_drift", 1.0))
        * _as_float(row.get("combined_drift_pix"), 0.0)
        / centroid_tol
    )

    # CONVERGENCE HEALTH
    behavior = (
        float(weights.get("n_stop_m1", 2.0)) * _as_float(row.get("n_stop_m1"), 0.0)
        + float(weights.get("frac_stop_nonzero", 5.0))
        * _as_float(row.get("frac_stop_nonzero"), 0.0)
    )

    # COMPLETENESS -- short fits penalized relative to the galaxy's best arm
    n_iso = _as_float(row.get("n_iso"), 0.0)
    completeness = (
        float(weights.get("n_iso_completeness", 3.0))
        * max(0.0, 1.0 - n_iso / n_iso_ref_eff)
    )

    # SPEED -- tie-breaker; never outweighs quality
    speed = float(weights.get("wall_time", 0.1)) * _as_float(
        row.get("wall_time_fit_s"), 0.0
    )

    return float(fidelity + centroid + behavior + completeness + speed)


def apply_composite_scores(
    rows: Iterable[dict[str, Any]],
    weights: dict[str, float] | None = None,
    *,
    image_sigma: float = 1.0,
    n_iso_ref: int | None = None,
) -> list[dict[str, Any]]:
    """Return a new list of rows with ``composite_score`` filled in.

    ``n_iso_ref`` defaults to the max ``n_iso`` across rows with
    ``status in {"ok", "cached"}`` and no severity-error flag; floored
    at 1. Callers usually let this auto-compute per galaxy.
    """
    rows = [dict(row) for row in rows]
    if n_iso_ref is None:
        eligible = [
            int(r.get("n_iso", 0))
            for r in rows
            if str(r.get("status", "")).lower() in {"ok", "cached"}
            and _as_float(r.get("flag_severity_max"), 0.0) < SEVERITY_ERROR
        ]
        n_iso_ref = max(eligible) if eligible else 1
    for row in rows:
        row["composite_score"] = compute_composite_score(
            row,
            weights,
            image_sigma=image_sigma,
            n_iso_ref=n_iso_ref,
        )
        # Record the context for audit so the score is reproducible
        # from the inventory alone.
        row["image_sigma_adu"] = float(image_sigma)
        row["n_iso_ref_used"] = int(n_iso_ref)
    return rows


# ---------------------------------------------------------------------------
# Cross-arm tables
# ---------------------------------------------------------------------------

CROSS_ARM_COLUMNS: tuple[str, ...] = (
    "arm_id",
    "status",
    "composite_score",
    "flag_severity_max",
    "flags",
    "wall_time_fit_s",
    "n_iso",
    "n_stop_m1",
    "frac_stop_nonzero",
    "combined_drift_pix",
    "resid_rms_inner",
    "resid_rms_mid",
    "resid_rms_outer",
    "image_sigma_adu",
    "max_dpa_deg",    # reported for reference, not scored
    "max_deps",       # reported for reference, not scored
    "first_isophote_failure",
    "first_isophote_retry_attempts",
)


def write_per_galaxy_cross_arm_table(
    rows: list[dict[str, Any]],
    galaxy_id: str,
    output_dir: Path,
    *,
    sort_by: str = "composite_score",
) -> tuple[Path, Path]:
    """Write ``cross_arm_table.csv`` + ``.md`` for one galaxy.

    ``rows`` is typically the full inventory for one (galaxy, tool)
    pair (already carrying ``composite_score``). Returns the two
    output paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(
        rows,
        key=lambda r: (
            _as_float(r.get(sort_by), float("inf")),
            str(r.get("arm_id", "")),
        ),
    )
    csv_path = output_dir / "cross_arm_table.csv"
    md_path = output_dir / "cross_arm_table.md"
    _write_csv(csv_path, sorted_rows, CROSS_ARM_COLUMNS)
    _write_markdown(
        md_path,
        title=f"Cross-arm table: {galaxy_id}",
        rows=sorted_rows,
        columns=CROSS_ARM_COLUMNS,
    )
    return csv_path, md_path


def write_per_tool_cross_arm_summary(
    rows_by_galaxy: dict[str, list[dict[str, Any]]],
    tool: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Aggregate all galaxies: median per arm.

    Input: ``{galaxy_id: [rows]}``. Output: ``cross_arm_summary.csv``
    and ``cross_arm_summary.md`` with one row per arm.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # arm_id -> list[row]
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for rows in rows_by_galaxy.values():
        for row in rows:
            by_arm.setdefault(row["arm_id"], []).append(row)

    aggregated: list[dict[str, Any]] = []
    for arm_id, arm_rows in by_arm.items():
        ok_rows = [
            r for r in arm_rows if str(r.get("status", "")).lower() in {"ok", "cached"}
        ]
        aggregated.append(
            {
                "arm_id": arm_id,
                "n_galaxies_total": len(arm_rows),
                "n_galaxies_ok": len(ok_rows),
                "median_composite_score": _median(
                    [r.get("composite_score") for r in ok_rows]
                ),
                "median_wall_time_fit_s": _median(
                    [r.get("wall_time_fit_s") for r in ok_rows]
                ),
                "median_combined_drift_pix": _median(
                    [r.get("combined_drift_pix") for r in ok_rows]
                ),
                "median_resid_rms_inner": _median(
                    [r.get("resid_rms_inner") for r in ok_rows]
                ),
                "median_resid_rms_mid": _median(
                    [r.get("resid_rms_mid") for r in ok_rows]
                ),
                "median_resid_rms_outer": _median(
                    [r.get("resid_rms_outer") for r in ok_rows]
                ),
                "n_first_isophote_failure": sum(
                    1 for r in arm_rows if bool(r.get("first_isophote_failure"))
                ),
                "max_flag_severity": max(
                    (_as_float(r.get("flag_severity_max"), 0.0) for r in arm_rows),
                    default=0.0,
                ),
            }
        )
    summary_cols = (
        "arm_id",
        "n_galaxies_ok",
        "n_galaxies_total",
        "median_composite_score",
        "median_wall_time_fit_s",
        "median_combined_drift_pix",
        "median_resid_rms_inner",
        "median_resid_rms_mid",
        "median_resid_rms_outer",
        "n_first_isophote_failure",
        "max_flag_severity",
    )
    aggregated.sort(
        key=lambda r: _as_float(r.get("median_composite_score"), float("inf"))
    )
    csv_path = output_dir / "cross_arm_summary.csv"
    md_path = output_dir / "cross_arm_summary.md"
    _write_csv(csv_path, aggregated, summary_cols)
    _write_markdown(
        md_path,
        title=f"Cross-arm summary: tool={tool}",
        rows=aggregated,
        columns=summary_cols,
    )
    return csv_path, md_path


# ---------------------------------------------------------------------------
# CSV / markdown helpers
# ---------------------------------------------------------------------------


def _write_csv(
    path: Path, rows: list[dict[str, Any]], columns: Iterable[str]
) -> None:
    import csv

    columns = list(columns)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            rendered = {col: _format_for_csv(row.get(col, "")) for col in columns}
            writer.writerow(rendered)


def _write_markdown(
    path: Path, title: str, rows: list[dict[str, Any]], columns: Iterable[str]
) -> None:
    columns = list(columns)
    lines: list[str] = [f"# {title}", "", "| " + " | ".join(columns) + " |"]
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for row in rows:
        cells = [_format_for_markdown(row.get(col, "")) for col in columns]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def _format_for_csv(value: Any) -> str:
    if isinstance(value, float):
        if not np.isfinite(value):
            return ""
        return f"{value:.6g}"
    return str(value) if value is not None else ""


def _format_for_markdown(value: Any) -> str:
    if isinstance(value, float):
        if not np.isfinite(value):
            return "—"
        return f"{value:.4g}"
    if isinstance(value, bool):
        return "✓" if value else ""
    if value is None:
        return ""
    return str(value)


def _median(values: list[Any]) -> float:
    cleaned = [_as_float(v, float("nan")) for v in values]
    cleaned = [v for v in cleaned if np.isfinite(v)]
    if not cleaned:
        return float("nan")
    return float(np.median(cleaned))


def _as_float(value: Any, fallback: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return fallback
    if value != value:  # NaN
        return fallback
    return value


def write_cross_tool_table(
    per_tool_rows: dict[str, dict[str, list[dict[str, Any]]]],
    output_dir: Path,
    *,
    default_arm_per_tool: dict[str, str] | None = None,
) -> tuple[Path, Path] | None:
    """Pivot per-tool inventories to a tool-vs-galaxy table.

    Picks each tool's default arm (overridable via
    ``default_arm_per_tool``) and, per galaxy, tabulates composite_score,
    wall_time_fit_s, n_iso, combined_drift_pix, resid_rms_outer, flags.
    Returns ``(csv_path, md_path)`` or ``None`` if fewer than two tools
    contributed a row.
    """
    if default_arm_per_tool is None:
        default_arm_per_tool = {
            "isoster": "ref_default",
            "photutils": "baseline_median",
            "autoprof": "baseline",
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    all_galaxies: set[str] = set()
    for rows_by_galaxy in per_tool_rows.values():
        all_galaxies.update(rows_by_galaxy.keys())
    if len(per_tool_rows) < 2 or not all_galaxies:
        return None

    rows: list[dict[str, Any]] = []
    for galaxy_id in sorted(all_galaxies):
        for tool, rows_by_galaxy in per_tool_rows.items():
            galaxy_rows = rows_by_galaxy.get(galaxy_id, [])
            if not galaxy_rows:
                continue
            chosen = _pick_arm_row(galaxy_rows, default_arm_per_tool.get(tool))
            if chosen is None:
                continue
            rows.append(
                {
                    "galaxy_id": galaxy_id,
                    "tool": tool,
                    "arm_id": chosen.get("arm_id", ""),
                    "status": chosen.get("status", ""),
                    "composite_score": _as_float(chosen.get("composite_score"), float("nan")),
                    "wall_time_fit_s": _as_float(chosen.get("wall_time_fit_s"), float("nan")),
                    "n_iso": int(chosen.get("n_iso", 0) or 0),
                    "combined_drift_pix": _as_float(chosen.get("combined_drift_pix"), float("nan")),
                    "resid_rms_inner": _as_float(chosen.get("resid_rms_inner"), float("nan")),
                    "resid_rms_outer": _as_float(chosen.get("resid_rms_outer"), float("nan")),
                    "flags": chosen.get("flags", ""),
                }
            )
    if not rows:
        return None

    cols = (
        "galaxy_id",
        "tool",
        "arm_id",
        "status",
        "composite_score",
        "wall_time_fit_s",
        "n_iso",
        "combined_drift_pix",
        "resid_rms_inner",
        "resid_rms_outer",
        "flags",
    )
    csv_path = output_dir / "cross_tool_table.csv"
    md_path = output_dir / "cross_tool_table.md"
    _write_csv(csv_path, rows, cols)
    _write_markdown(
        md_path,
        title="Cross-tool table (default arm per tool)",
        rows=rows,
        columns=cols,
    )
    return csv_path, md_path


def _pick_arm_row(
    rows: list[dict[str, Any]], default_arm: str | None
) -> dict[str, Any] | None:
    ok_rows = [
        r for r in rows
        if str(r.get("status", "")).lower() in {"ok", "cached"}
        and _as_float(r.get("flag_severity_max"), 0.0) < SEVERITY_ERROR
    ]
    if default_arm:
        for r in ok_rows:
            if str(r.get("arm_id", "")) == default_arm:
                return r
    if not ok_rows:
        return None
    ok_rows.sort(key=lambda r: _as_float(r.get("composite_score"), float("inf")))
    return ok_rows[0]


__all__ = [
    "compute_composite_score",
    "apply_composite_scores",
    "write_per_galaxy_cross_arm_table",
    "write_per_tool_cross_arm_summary",
    "write_cross_tool_table",
    "DEFAULT_WEIGHTS",
    "ERROR_FLAG_PENALTY",
    "CROSS_ARM_COLUMNS",
    "INVENTORY_COLUMNS",
]
