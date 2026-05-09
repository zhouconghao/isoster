"""Quality flags evaluated from a metric row dict.

Each flag is a deterministic predicate over scalar metrics; severity
is the numeric max of any flag triggered. The taxonomy below was
designed for residual-metric-contract v1.1 outputs (see
``residual_metrics.py`` and ``profile_io.py``) but works for any row
dict that exposes the same key names.

Thresholds are configurable -- see ``DEFAULT_THRESHOLDS`` for the
defaults that worked well on ~70k SGA-2020 r-band cutouts. Tune for
your sample (compact galaxies probably want lower ``few_isophotes``
threshold; deep imaging may want larger ``large_drift`` cap).

Usage::

    from quality_flags import evaluate, DEFAULT_THRESHOLDS

    row = {
        "n_iso": 60,
        "max_dpa_deg": 22.4,
        "abs_resid_over_sigma_inner": 4.1,
        ...
    }
    result = evaluate(row)
    # {"flags": "LARGE_DPA(1)", "flag_severity_max": 1.0}

    # custom thresholds:
    th = {**DEFAULT_THRESHOLDS, "few_isophotes_lt": 3, "large_dpa_gt": 45.0}
    result = evaluate(row, thresholds=th)
"""

from __future__ import annotations

import math
from typing import Any

SEVERITY_ERROR = 3.0

# (flag_id, severity, predicate_name)
_FLAG_DEFINITIONS: list[tuple[str, float, str]] = [
    ("FEW_ISOPHOTES", 1.0, "few_isophotes"),
    ("FIRST_ISOPHOTE_RETRY", 0.5, "first_isophote_retry"),
    ("HIGH_NONZERO_STOP_FRAC", 1.0, "high_nonzero_stop_frac"),
    ("LARGE_DRIFT", 1.5, "large_drift"),
    ("LARGE_DPA", 1.0, "large_dpa"),
    ("LARGE_DEPS", 1.0, "large_deps"),
    ("INNER_RESID_LARGE", 1.5, "inner_resid_large"),
    ("OUTER_RESID_LARGE", 1.0, "outer_resid_large"),
    ("FAILED", 3.0, "is_failed"),
    ("SKIPPED", 0.0, "is_skipped"),
]


DEFAULT_THRESHOLDS: dict[str, float] = {
    # Below ``few_isophotes_lt`` rings = unusable fit.
    "few_isophotes_lt": 6,
    # ``frac_stop_nonzero`` above this = isophote engine struggled.
    "high_nonzero_stop_frac_gt": 0.4,
    # ``combined_drift_pix`` above this = centre wandered too far.
    "large_drift_gt_pix": 5.0,
    # ``max_dpa_deg`` above this = PA jumped too much between rings.
    "large_dpa_gt_deg": 30.0,
    # ``max_deps`` above this = ellipticity jumped too much.
    "large_deps_gt": 0.2,
    # ``abs_resid_over_sigma_inner`` above this = central misfit.
    "inner_resid_large_gt_sigma": 5.0,
    # ``abs_resid_over_sigma_outer`` above this = outer-disk misfit.
    "outer_resid_large_gt_sigma": 2.5,
}


def _safe_float(value: Any) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return f


def _few_isophotes(row: dict[str, Any], th: dict[str, float]) -> bool:
    return int(row.get("n_iso", 0) or 0) < int(th["few_isophotes_lt"])


def _first_isophote_retry(row: dict[str, Any], th: dict[str, float]) -> bool:
    return bool(int(row.get("first_isophote_retry_attempts", 0) or 0) > 0)


def _high_nonzero_stop_frac(row: dict[str, Any], th: dict[str, float]) -> bool:
    f = _safe_float(row.get("frac_stop_nonzero"))
    return math.isfinite(f) and f > th["high_nonzero_stop_frac_gt"]


def _large_drift(row: dict[str, Any], th: dict[str, float]) -> bool:
    f = _safe_float(row.get("combined_drift_pix"))
    return math.isfinite(f) and f > th["large_drift_gt_pix"]


def _large_dpa(row: dict[str, Any], th: dict[str, float]) -> bool:
    f = _safe_float(row.get("max_dpa_deg"))
    return math.isfinite(f) and f > th["large_dpa_gt_deg"]


def _large_deps(row: dict[str, Any], th: dict[str, float]) -> bool:
    f = _safe_float(row.get("max_deps"))
    return math.isfinite(f) and f > th["large_deps_gt"]


def _inner_resid_large(row: dict[str, Any], th: dict[str, float]) -> bool:
    f = _safe_float(row.get("abs_resid_over_sigma_inner"))
    return math.isfinite(f) and f > th["inner_resid_large_gt_sigma"]


def _outer_resid_large(row: dict[str, Any], th: dict[str, float]) -> bool:
    f = _safe_float(row.get("abs_resid_over_sigma_outer"))
    return math.isfinite(f) and f > th["outer_resid_large_gt_sigma"]


def _is_failed(row: dict[str, Any], th: dict[str, float]) -> bool:
    status = str(row.get("status", "")).lower()
    return status in {"failed", "config_error"} or status.startswith("error")


def _is_skipped(row: dict[str, Any], th: dict[str, float]) -> bool:
    return str(row.get("status", "")) == "skipped"


_PREDICATES = {
    "few_isophotes": _few_isophotes,
    "first_isophote_retry": _first_isophote_retry,
    "high_nonzero_stop_frac": _high_nonzero_stop_frac,
    "large_drift": _large_drift,
    "large_dpa": _large_dpa,
    "large_deps": _large_deps,
    "inner_resid_large": _inner_resid_large,
    "outer_resid_large": _outer_resid_large,
    "is_failed": _is_failed,
    "is_skipped": _is_skipped,
}


def evaluate(
    row: dict[str, Any],
    *,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Return ``{"flags": "...", "flag_severity_max": float}``.

    ``flags`` is a comma-separated list ``"NAME(severity)"``; empty
    string when no flag triggers. ``flag_severity_max`` is the max
    numeric severity over the triggered flags (0.0 when none trigger).

    Pass a custom ``thresholds`` dict to override the defaults; any
    key not provided falls back to ``DEFAULT_THRESHOLDS``.
    """
    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        th.update(thresholds)

    triggered: list[tuple[str, float]] = []
    for flag_id, severity, predicate_name in _FLAG_DEFINITIONS:
        try:
            if _PREDICATES[predicate_name](row, th):
                triggered.append((flag_id, severity))
        except Exception:  # noqa: BLE001 - never let flag evaluation fail
            continue
    if not triggered:
        return {"flags": "", "flag_severity_max": 0.0}
    flags = ",".join(f"{name}({sev:g})" for name, sev in triggered)
    return {
        "flags": flags,
        "flag_severity_max": max(sev for _, sev in triggered),
    }


def evaluate_flags(
    row: dict[str, Any],
    *,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for the existing exhausted fitter API."""
    return evaluate(row, thresholds=thresholds)


__all__ = ["evaluate", "evaluate_flags", "DEFAULT_THRESHOLDS", "SEVERITY_ERROR"]
