"""Per-fit quality flags and their severity levels.

Ported from ``sga_isoster/scripts/analyze_{photutils,autoprof}.py`` and
extended with the ``first_isophote_failure`` flag that the Phase B
smoke campaign exposed: an arm whose single surviving isophote is the
central pixel at ``sma=0`` is not a legitimate fit, yet naive drift
ranking happily promotes it to the top.

Severity ladder (higher = worse; used for sorting and composite
scoring):

    0 = info       (noteworthy but benign)
    1 = warn       (call the fit into question but keep it)
    2 = error      (fit should be excluded from cross-arm rankings)

The returned ``flags`` field is a comma-separated string so the
inventory FITS can store it in a simple text column.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SEVERITY_INFO = 0
SEVERITY_WARN = 1
SEVERITY_ERROR = 2


@dataclass(frozen=True)
class FlagRule:
    name: str
    severity: int
    description: str


RULES: tuple[FlagRule, ...] = (
    FlagRule("no_fit", SEVERITY_ERROR, "zero isophotes produced"),
    FlagRule(
        "first_isophote_failure",
        SEVERITY_ERROR,
        "only the central pixel survived; first ellipse fit failed even after retry",
    ),
    FlagRule("few_isophotes", SEVERITY_WARN, "fewer than 5 isophotes"),
    FlagRule("high_stopcode", SEVERITY_WARN, "frac_stop_nonzero > 0.3"),
    FlagRule("any_stop_m1", SEVERITY_WARN, "at least one stop_code=-1"),
    FlagRule("center_drift", SEVERITY_WARN, "combined_drift_pix > 5 pix"),
    FlagRule("pa_instability", SEVERITY_WARN, "max_dpa_deg > 45 deg"),
    FlagRule("eps_instability", SEVERITY_WARN, "max_deps > 0.3"),
    FlagRule(
        "high_outer_resid_frac",
        SEVERITY_WARN,
        "frac_above_3sigma_outer > 0.05",
    ),
)


def evaluate_flags(row: dict[str, Any]) -> dict[str, Any]:
    """Apply every rule to an inventory row. Returns ``{flags, flag_severity_max}``.

    The ``row`` must already carry the metric columns produced by
    ``metrics.summarize_fit`` plus, ideally, the residual-zone and
    first-isophote diagnostics. Missing columns default to neutral
    values so the flag check never raises.
    """
    fired: list[str] = []
    max_severity = -1
    for rule in RULES:
        if _rule_triggers(rule.name, row):
            fired.append(rule.name)
            max_severity = max(max_severity, rule.severity)
    return {
        "flags": ",".join(fired),
        "flag_severity_max": float(max_severity) if max_severity >= 0 else 0.0,
    }


def _rule_triggers(name: str, row: dict[str, Any]) -> bool:
    if name == "no_fit":
        return int(row.get("n_iso", 0)) == 0
    if name == "first_isophote_failure":
        # Direct: the results dict said so.
        if bool(row.get("first_isophote_failure", False)):
            return True
        # Fallback: one-isophote fits whose histogram is pure zero indicate
        # only the central pixel survived (retry metadata may be missing on
        # rows loaded from older caches).
        n_iso = int(row.get("n_iso", 0))
        return n_iso == 1 and int(row.get("n_stop_0", 0)) == 1
    if name == "few_isophotes":
        return 0 < int(row.get("n_iso", 0)) < 5
    if name == "high_stopcode":
        return _as_float(row.get("frac_stop_nonzero"), 0.0) > 0.3
    if name == "any_stop_m1":
        return int(row.get("n_stop_m1", 0)) > 0
    if name == "center_drift":
        return _as_float(row.get("combined_drift_pix"), 0.0) > 5.0
    if name == "pa_instability":
        return _as_float(row.get("max_dpa_deg"), 0.0) > 45.0
    if name == "eps_instability":
        return _as_float(row.get("max_deps"), 0.0) > 0.3
    if name == "high_outer_resid_frac":
        return _as_float(row.get("frac_above_3sigma_outer"), 0.0) > 0.05
    return False


def _as_float(value: Any, fallback: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return fallback
    if value != value:  # NaN
        return fallback
    return value
