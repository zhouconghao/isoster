"""Cross-arm and cross-tool statistical summaries.

Two roles:

1. Computing a per-row ``composite_score`` from the campaign YAML's
   ``summary.composite_score_weights`` (lower is better). Rows with a
   severity-error flag receive a huge penalty so pathological fits
   (e.g. ``first_isophote_failure``) never win a ranking.

2. Writing CSV + markdown tables that pivot the inventory into:
   - per-galaxy cross-arm comparison
   - per-tool cross-arm summary (medians across galaxies)
   - (future, Phase D) per-galaxy cross-tool comparison

The module does not mutate inventory FITS files on disk — callers feed
inventory rows in, tables come out.
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
    "spline_rms_center": 1.0,
    "max_dpa_deg": 1.0,
    "max_deps": 1.0,
    "n_stop_m1": 2.0,
    "elapsed_s": -0.02,
}

# Rows flagged as ERROR get this penalty added to the score so they are
# never ranked "best" regardless of per-metric values.
ERROR_FLAG_PENALTY = 1_000_000.0


def compute_composite_score(
    row: dict[str, Any], weights: dict[str, float] | None = None
) -> float:
    """Return the composite score for one row. Lower is better.

    ``weights`` keys:
      ``spline_rms_center``, ``max_dpa_deg``, ``max_deps``,
      ``n_stop_m1``, ``elapsed_s``. Unknown keys are ignored; missing
      keys default to zero.

    Missing or NaN metric values drop out of the sum (treated as zero)
    rather than propagating NaN across every downstream table. The
    only exception is when the row itself is not ``status == "ok"`` or
    has a severity-error flag — then the returned score is
    ``ERROR_FLAG_PENALTY`` (a huge positive number) so the row sinks
    to the bottom of any sort.
    """
    weights = weights if weights is not None else DEFAULT_WEIGHTS
    status = str(row.get("status", "")).lower()
    if status not in {"ok", "cached"}:
        return ERROR_FLAG_PENALTY
    severity = _as_float(row.get("flag_severity_max"), 0.0)
    if severity >= SEVERITY_ERROR:
        return ERROR_FLAG_PENALTY

    # Metric-weight mapping: metric column name -> weight key
    contributions: dict[str, str] = {
        "spline_rms_center": "spline_rms_center",
        "max_dpa_deg": "max_dpa_deg",
        "max_deps": "max_deps",
        "n_stop_m1": "n_stop_m1",
        "wall_time_fit_s": "elapsed_s",
    }
    score = 0.0
    for metric_key, weight_key in contributions.items():
        w = float(weights.get(weight_key, 0.0))
        if w == 0.0:
            continue
        value = _as_float(row.get(metric_key), 0.0)
        score += w * value
    return float(score)


def apply_composite_scores(
    rows: Iterable[dict[str, Any]], weights: dict[str, float] | None = None
) -> list[dict[str, Any]]:
    """Return a new list of rows with ``composite_score`` filled in."""
    scored: list[dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        row["composite_score"] = compute_composite_score(row, weights)
        scored.append(row)
    return scored


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
    "spline_rms_center",
    "max_dpa_deg",
    "max_deps",
    "outer_gerr_median",
    "resid_rms_inner",
    "resid_rms_mid",
    "resid_rms_outer",
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
    """Aggregate all galaxies: median/median-absolute-deviation per arm.

    Input: ``{galaxy_id: [rows]}``. Output: ``cross_arm_summary.csv``
    and ``cross_arm_summary.md`` with one row per arm and columns for
    median composite score, median drift, median wall time, plus the
    count of galaxies for which each arm ran ``ok``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # arm_id -> list[row]
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for rows in rows_by_galaxy.values():
        for row in rows:
            by_arm.setdefault(row["arm_id"], []).append(row)

    aggregated: list[dict[str, Any]] = []
    for arm_id, arm_rows in by_arm.items():
        ok_rows = [r for r in arm_rows if str(r.get("status", "")).lower() in {"ok", "cached"}]
        aggregated.append(
            {
                "arm_id": arm_id,
                "n_galaxies_total": len(arm_rows),
                "n_galaxies_ok": len(ok_rows),
                "median_composite_score": _median([r.get("composite_score") for r in ok_rows]),
                "median_wall_time_fit_s": _median([r.get("wall_time_fit_s") for r in ok_rows]),
                "median_combined_drift_pix": _median([r.get("combined_drift_pix") for r in ok_rows]),
                "median_spline_rms_center": _median([r.get("spline_rms_center") for r in ok_rows]),
                "median_max_dpa_deg": _median([r.get("max_dpa_deg") for r in ok_rows]),
                "median_max_deps": _median([r.get("max_deps") for r in ok_rows]),
                "median_resid_rms_outer": _median([r.get("resid_rms_outer") for r in ok_rows]),
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
        "median_spline_rms_center",
        "median_max_dpa_deg",
        "median_max_deps",
        "median_resid_rms_outer",
        "n_first_isophote_failure",
        "max_flag_severity",
    )
    aggregated.sort(key=lambda r: _as_float(r.get("median_composite_score"), float("inf")))
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


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: Iterable[str]) -> None:
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


__all__ = [
    "compute_composite_score",
    "apply_composite_scores",
    "write_per_galaxy_cross_arm_table",
    "write_per_tool_cross_arm_summary",
    "DEFAULT_WEIGHTS",
    "ERROR_FLAG_PENALTY",
    "CROSS_ARM_COLUMNS",
    "INVENTORY_COLUMNS",
]
