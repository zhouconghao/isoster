"""Metric helpers for the robustness benchmark.

All helpers compare a perturbed isophote list to a reference list and return
dimensionless, frame-invariant scalars. The profile comparison uses
*relative* intensity variation (dimensionless) rather than magnitude units,
so the bright core and the LSB tail share the same scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

# Characterization bin edges — first guesses from
# docs/agent/journal/2026-04-15_robustness_plan.md §8. These are used to
# color summary tables, not as pass/fail gates.
BIN_EDGES = {
    "profile_frac_overlap": {"good_min": 0.90, "warn_min": 0.70},
    "profile_rel_rms": {"good_max": 0.02, "warn_max": 0.10},
    "profile_rel_max": {"good_max": 0.05, "warn_max": 0.25},
    "eps_mad": {"good_max": 0.02, "warn_max": 0.05},
    "pa_mad_deg": {"good_max": 2.0, "warn_max": 5.0},
    "center_max_drift": {"good_max": 0.5, "warn_max": 2.0},
}


def _to_array(values: Iterable, key: str) -> np.ndarray:
    return np.asarray([row[key] for row in values], dtype=np.float64)


def relative_intensity_deviation(
    intens_ref: np.ndarray,
    intens_pert: np.ndarray,
    low_signal_threshold: float,
) -> tuple[float, float, int]:
    """Return ``(rel_rms, rel_max, n_low_signal)`` on a matched grid.

    Points with ``|intens_ref| < low_signal_threshold`` are dropped from the
    relative-deviation calculation and counted in ``n_low_signal`` instead —
    they are not informative because the denominator is near zero.
    """
    ref = np.asarray(intens_ref, dtype=np.float64)
    pert = np.asarray(intens_pert, dtype=np.float64)
    mask = np.abs(ref) >= low_signal_threshold
    n_low = int(np.sum(~mask))
    if not np.any(mask):
        return float("nan"), float("nan"), n_low
    rel = (pert[mask] - ref[mask]) / ref[mask]
    return float(np.sqrt(np.mean(rel * rel))), float(np.max(np.abs(rel))), n_low


def _wrap_pa_to_half_pi(delta: np.ndarray) -> np.ndarray:
    """Wrap an angular difference in radians to the semicircle [-pi/2, pi/2]."""
    wrapped = np.mod(delta + np.pi / 2.0, np.pi) - np.pi / 2.0
    return wrapped


def angular_mad(pa_ref: np.ndarray, pa_pert: np.ndarray) -> float:
    """Median absolute deviation of ``pa_pert - pa_ref`` with pi periodicity."""
    if pa_ref.size == 0:
        return float("nan")
    delta = _wrap_pa_to_half_pi(np.asarray(pa_pert) - np.asarray(pa_ref))
    return float(np.median(np.abs(delta)))


def center_drift(
    x0_ref: np.ndarray,
    y0_ref: np.ndarray,
    x0_pert: np.ndarray,
    y0_pert: np.ndarray,
) -> float:
    """Maximum Euclidean center drift between reference and perturbed isophotes."""
    if x0_ref.size == 0:
        return float("nan")
    dx = np.asarray(x0_pert) - np.asarray(x0_ref)
    dy = np.asarray(y0_pert) - np.asarray(y0_ref)
    return float(np.max(np.sqrt(dx * dx + dy * dy)))


@dataclass
class ProfileComparison:
    """Bundle of metrics comparing a perturbed fit to a reference fit."""

    n_ref: int
    n_pert: int
    n_overlap: int
    frac_overlap: float
    profile_rel_rms: float
    profile_rel_max: float
    n_low_signal: int
    eps_mad: float
    pa_mad_deg: float
    center_max_drift: float


def _interp_sorted(
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    x_query: np.ndarray,
) -> np.ndarray:
    """Linear interpolation that assumes ``x_ref`` is strictly monotonic.

    Values of ``x_query`` outside the reference range are returned as NaN.
    This is the single place where the robustness metrics reach into numpy's
    ``interp``; downstream code handles the NaN mask.
    """
    inside = (x_query >= x_ref[0]) & (x_query <= x_ref[-1])
    out = np.full_like(x_query, np.nan, dtype=np.float64)
    out[inside] = np.interp(x_query[inside], x_ref, y_ref)
    return out


def compare_to_reference(
    reference_isophotes: Sequence[dict],
    perturbed_isophotes: Sequence[dict],
    low_signal_threshold: float,
) -> ProfileComparison:
    """Compare two lists of isophote dicts by interpolating the reference.

    The reference is resampled onto the perturbed SMA grid via linear
    interpolation, so the two fits are always compared at the same radii
    without a matching tolerance. Perturbed SMAs that fall outside the
    reference's SMA support are excluded from the metrics and counted in
    ``frac_overlap`` instead.
    """
    nan_bundle = ProfileComparison(
        n_ref=len(reference_isophotes) if reference_isophotes else 0,
        n_pert=len(perturbed_isophotes) if perturbed_isophotes else 0,
        n_overlap=0,
        frac_overlap=float("nan"),
        profile_rel_rms=float("nan"),
        profile_rel_max=float("nan"),
        n_low_signal=0,
        eps_mad=float("nan"),
        pa_mad_deg=float("nan"),
        center_max_drift=float("nan"),
    )
    if not reference_isophotes or not perturbed_isophotes:
        return nan_bundle

    ref_sma = _to_array(reference_isophotes, "sma")
    pert_sma = _to_array(perturbed_isophotes, "sma")
    order = np.argsort(ref_sma)
    ref_sma = ref_sma[order]
    # Ensure strict monotonicity: drop duplicate SMAs (keeps the first).
    keep = np.concatenate(([True], np.diff(ref_sma) > 0))
    ref_sma = ref_sma[keep]
    if ref_sma.size < 2:
        return nan_bundle

    def resample(field: str) -> np.ndarray:
        values = _to_array(reference_isophotes, field)[order][keep]
        return _interp_sorted(ref_sma, values, pert_sma)

    ref_intens_on_pert = resample("intens")
    ref_eps_on_pert = resample("eps")
    ref_x0_on_pert = resample("x0")
    ref_y0_on_pert = resample("y0")
    ref_pa_on_pert = resample("pa")  # radians; wrap handled in angular_mad

    inside = np.isfinite(ref_intens_on_pert)
    n_overlap = int(np.sum(inside))
    frac_overlap = n_overlap / float(pert_sma.size)
    if n_overlap == 0:
        return ProfileComparison(
            n_ref=int(ref_sma.size),
            n_pert=int(pert_sma.size),
            n_overlap=0,
            frac_overlap=frac_overlap,
            profile_rel_rms=float("nan"),
            profile_rel_max=float("nan"),
            n_low_signal=0,
            eps_mad=float("nan"),
            pa_mad_deg=float("nan"),
            center_max_drift=float("nan"),
        )

    pert_intens = _to_array(perturbed_isophotes, "intens")[inside]
    ref_intens = ref_intens_on_pert[inside]
    rel_rms, rel_max, n_low = relative_intensity_deviation(
        ref_intens, pert_intens, low_signal_threshold
    )

    pert_eps = _to_array(perturbed_isophotes, "eps")[inside]
    ref_eps = ref_eps_on_pert[inside]
    eps_mad_value = float(np.median(np.abs(pert_eps - ref_eps)))

    pert_pa = _to_array(perturbed_isophotes, "pa")[inside]
    ref_pa = ref_pa_on_pert[inside]
    pa_mad_rad = angular_mad(ref_pa, pert_pa)
    pa_mad_deg = float(np.degrees(pa_mad_rad)) if np.isfinite(pa_mad_rad) else float("nan")

    pert_x0 = _to_array(perturbed_isophotes, "x0")[inside]
    pert_y0 = _to_array(perturbed_isophotes, "y0")[inside]
    ref_x0 = ref_x0_on_pert[inside]
    ref_y0 = ref_y0_on_pert[inside]
    drift = center_drift(ref_x0, ref_y0, pert_x0, pert_y0)

    return ProfileComparison(
        n_ref=int(ref_sma.size),
        n_pert=int(pert_sma.size),
        n_overlap=n_overlap,
        frac_overlap=frac_overlap,
        profile_rel_rms=rel_rms,
        profile_rel_max=rel_max,
        n_low_signal=n_low,
        eps_mad=eps_mad_value,
        pa_mad_deg=pa_mad_deg,
        center_max_drift=drift,
    )


def _bin_upper(value: float, good_max: float, warn_max: float) -> str:
    if not np.isfinite(value):
        return "fail"
    if value <= good_max:
        return "good"
    if value <= warn_max:
        return "warn"
    return "fail"


def _bin_lower(value: float, good_min: float, warn_min: float) -> str:
    if not np.isfinite(value):
        return "fail"
    if value >= good_min:
        return "good"
    if value >= warn_min:
        return "warn"
    return "fail"


def bin_metrics(comparison: ProfileComparison, first_iso_failure: bool) -> dict[str, str]:
    """Bucket each metric into ``good`` / ``warn`` / ``fail`` using §8 bins."""
    bins: dict[str, str] = {}
    bins["first_iso_failure"] = "fail" if first_iso_failure else "good"
    bins["profile_frac_overlap"] = _bin_lower(
        comparison.frac_overlap,
        BIN_EDGES["profile_frac_overlap"]["good_min"],
        BIN_EDGES["profile_frac_overlap"]["warn_min"],
    )
    for key, attr in (
        ("profile_rel_rms", "profile_rel_rms"),
        ("profile_rel_max", "profile_rel_max"),
        ("eps_mad", "eps_mad"),
        ("pa_mad_deg", "pa_mad_deg"),
        ("center_max_drift", "center_max_drift"),
    ):
        edges = BIN_EDGES[key]
        bins[key] = _bin_upper(
            getattr(comparison, attr),
            edges["good_max"],
            edges["warn_max"],
        )
    return bins


def worst_bin(bins: dict[str, str]) -> str:
    """Collapse per-metric bins into a single worst-case label."""
    order = {"good": 0, "warn": 1, "fail": 2}
    worst = "good"
    for label in bins.values():
        if order[label] > order[worst]:
            worst = label
    return worst
