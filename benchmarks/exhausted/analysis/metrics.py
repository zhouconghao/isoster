"""Per-fit metrics.

Ported from ``examples/example_hsc_edge_real/common.py`` with the
interface generalized so the campaign runner can feed any isoster
result dict (``fit_image`` output) regardless of dataset.

Every helper tolerates partial or empty inputs and returns ``nan`` or
zero rather than raising, so inventory rows stay homogeneous.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import UnivariateSpline

Isophote = dict[str, Any]
IsophoteList = list[Isophote]


# ---------------------------------------------------------------------------
# Stop-code helpers
# ---------------------------------------------------------------------------


def stop_code_histogram(isophotes: IsophoteList) -> dict[int, int]:
    """Return ``{stop_code: count}``.  Missing stop codes are recorded as ``-99``."""
    counts: dict[int, int] = {}
    for iso in isophotes:
        code = int(iso.get("stop_code", -99))
        counts[code] = counts.get(code, 0) + 1
    return counts


def stop_code_summary_string(isophotes: IsophoteList) -> str:
    """Compact ``'k:v k:v'`` string for logs."""
    counts = stop_code_histogram(isophotes)
    return "  ".join(f"{k}:{v}" for k, v in sorted(counts.items()))


def count_stop_code(isophotes: IsophoteList, code: int) -> int:
    return sum(1 for iso in isophotes if int(iso.get("stop_code", -99)) == code)


def frac_stop_nonzero(isophotes: IsophoteList) -> float:
    if not isophotes:
        return float("nan")
    return sum(1 for iso in isophotes if int(iso.get("stop_code", 0)) != 0) / len(isophotes)


# ---------------------------------------------------------------------------
# LSB / lock-aware slicing
# ---------------------------------------------------------------------------


def pre_lock_outward(isophotes: IsophoteList, sma0: float) -> IsophoteList:
    """Outward isophotes with ``sma >= sma0`` excluding any LSB-locked tail."""
    out: IsophoteList = []
    for iso in isophotes:
        if iso.get("sma", 0.0) < sma0:
            continue
        if iso.get("lsb_locked", False):
            continue
        out.append(iso)
    return out


def locked_tail(isophotes: IsophoteList) -> IsophoteList:
    return [iso for iso in isophotes if iso.get("lsb_locked", False)]


# ---------------------------------------------------------------------------
# Geometry-drift metrics
# ---------------------------------------------------------------------------


def combined_drift(
    isophotes: IsophoteList, x0_ref: float, y0_ref: float
) -> tuple[float, float, float]:
    """Return ``(max_dx, max_dy, combined)`` relative to ``(x0_ref, y0_ref)``.

    ``combined = sqrt(max_dx**2 + max_dy**2)``. Zeros for an empty list.
    """
    if not isophotes:
        return 0.0, 0.0, 0.0
    dx = np.array([abs(iso["x0"] - x0_ref) for iso in isophotes])
    dy = np.array([abs(iso["y0"] - y0_ref) for iso in isophotes])
    mdx = float(np.max(dx))
    mdy = float(np.max(dy))
    return mdx, mdy, float(np.sqrt(mdx**2 + mdy**2))


def spline_rms(isophotes: IsophoteList) -> float:
    """Combined ``x0``/``y0`` RMS around a smoothing spline in sma.

    Returns ``nan`` for fewer than 6 isophotes (not enough for a cubic
    smoothing spline).
    """
    if len(isophotes) < 6:
        return float("nan")
    sma = np.array([iso["sma"] for iso in isophotes])
    x0s = np.array([iso["x0"] for iso in isophotes])
    y0s = np.array([iso["y0"] for iso in isophotes])
    order = np.argsort(sma)
    sma, x0s, y0s = sma[order], x0s[order], y0s[order]
    try:
        spl_x = UnivariateSpline(sma, x0s, k=3, s=len(sma))
        spl_y = UnivariateSpline(sma, y0s, k=3, s=len(sma))
    except Exception:  # noqa: BLE001 - fragile scipy spline failure mode
        return float("nan")
    rx = x0s - spl_x(sma)
    ry = y0s - spl_y(sma)
    return float(np.sqrt(np.mean(rx**2) + np.mean(ry**2)))


def outward_drift_from_anchor(
    isophotes: IsophoteList, sma0: float
) -> tuple[float, float, float, float]:
    """Return ``(max|dx|, max|dy|, anchor_x0, anchor_y0)`` over outward fits."""
    outward = [iso for iso in isophotes if iso.get("sma", 0.0) >= sma0]
    if not outward:
        return float("nan"), float("nan"), float("nan"), float("nan")
    anchor = outward[0]
    dx = np.array([abs(iso["x0"] - anchor["x0"]) for iso in outward])
    dy = np.array([abs(iso["y0"] - anchor["y0"]) for iso in outward])
    return (
        float(np.nanmax(dx)),
        float(np.nanmax(dy)),
        float(anchor["x0"]),
        float(anchor["y0"]),
    )


def locked_tail_drift(isophotes: IsophoteList) -> tuple[float, float]:
    tail = locked_tail(isophotes)
    if not tail:
        return float("nan"), float("nan")
    x0s = np.array([iso["x0"] for iso in tail])
    y0s = np.array([iso["y0"] for iso in tail])
    return float(x0s.max() - x0s.min()), float(y0s.max() - y0s.min())


def max_dpa_deg(isophotes: IsophoteList) -> float:
    """Max absolute PA excursion across outer isophotes, 180°-unwrapped."""
    pa_values = [iso.get("pa") for iso in isophotes if iso.get("pa") is not None]
    if len(pa_values) < 2:
        return float("nan")
    pa_arr = np.asarray(pa_values, dtype=float)
    pa_deg = np.rad2deg(pa_arr)
    # Fold onto [0, 180), unwrap via doubled-angle trick.
    wrapped = np.mod(pa_deg, 180.0)
    doubled = np.deg2rad(2.0 * wrapped)
    unwrapped = np.unwrap(doubled)
    pa_cont_deg = np.rad2deg(0.5 * unwrapped)
    return float(pa_cont_deg.max() - pa_cont_deg.min())


def max_deps(isophotes: IsophoteList) -> float:
    eps_values = [iso.get("eps") for iso in isophotes if iso.get("eps") is not None]
    if len(eps_values) < 2:
        return float("nan")
    eps_arr = np.asarray(eps_values, dtype=float)
    return float(eps_arr.max() - eps_arr.min())


# ---------------------------------------------------------------------------
# Gradient diagnostics (requires debug=True)
# ---------------------------------------------------------------------------


def outer_gerr_median(isophotes: IsophoteList, sma_threshold: float) -> float:
    """Median ``|grad_error / grad|`` on isophotes with ``sma > threshold``.

    Returns ``nan`` if fewer than 3 eligible finite ratios.
    """
    ratios: list[float] = []
    for iso in isophotes:
        if iso.get("sma", 0.0) <= sma_threshold:
            continue
        grad = iso.get("grad")
        gerr = iso.get("grad_error")
        if grad is None or gerr is None:
            continue
        try:
            g = float(grad)
            ge = float(gerr)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(g) or not np.isfinite(ge) or g == 0.0:
            continue
        ratios.append(abs(ge / g))
    if len(ratios) < 3:
        return float("nan")
    return float(np.median(ratios))


def reference_centroid(
    results: dict[str, Any], isophotes: IsophoteList, sma0: float
) -> tuple[float, float]:
    """Prefer the outer-reg inner reference; fall back to the anchor at ``sma0``."""
    x0_ref = results.get("outer_reg_x0_ref")
    y0_ref = results.get("outer_reg_y0_ref")
    if x0_ref is not None and y0_ref is not None:
        return float(x0_ref), float(y0_ref)
    anchor = next((iso for iso in isophotes if iso.get("sma") == sma0), None)
    if anchor is None:
        anchor = next(
            (iso for iso in isophotes if iso.get("sma", 0.0) >= sma0),
            isophotes[0] if isophotes else None,
        )
    if anchor is None:
        return float("nan"), float("nan")
    return float(anchor["x0"]), float(anchor["y0"])


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------


def summarize_fit(
    results: dict[str, Any],
    sma0: float,
    lsb_sma_threshold_pix: float | None = None,
) -> dict[str, Any]:
    """Return a flat metric dict keyed by metric name.

    Always populates the following columns so inventory FITS stays
    homogeneous regardless of fit behavior: ``n_iso``, ``n_stop_0``,
    ``n_stop_1``, ``n_stop_2``, ``n_stop_m1``, ``frac_stop_nonzero``,
    ``combined_drift_pix``, ``spline_rms_center``, ``max_dpa_deg``,
    ``max_deps``, ``outer_gerr_median``, ``outward_drift_x``,
    ``outward_drift_y``, ``n_locked``, ``locked_drift_x``, ``locked_drift_y``.

    ``lsb_sma_threshold_pix`` controls which isophotes feed
    ``outer_gerr_median``. When ``None`` we fall back to ``2 * sma0``.
    """
    isophotes: IsophoteList = results.get("isophotes", [])
    histogram = stop_code_histogram(isophotes)

    x0_ref, y0_ref = reference_centroid(results, isophotes, sma0)
    outward = pre_lock_outward(isophotes, sma0)
    _max_dx, _max_dy, combined = combined_drift(outward, x0_ref, y0_ref)
    outward_dx, outward_dy, _ax, _ay = outward_drift_from_anchor(outward, sma0)
    locked_dx, locked_dy = locked_tail_drift(isophotes)

    outer_thr = lsb_sma_threshold_pix if lsb_sma_threshold_pix is not None else 2.0 * sma0
    outer_gerr = outer_gerr_median(isophotes, outer_thr)

    return {
        "n_iso": len(isophotes),
        "n_stop_0": int(histogram.get(0, 0)),
        "n_stop_1": int(histogram.get(1, 0)),
        "n_stop_2": int(histogram.get(2, 0)),
        "n_stop_m1": int(histogram.get(-1, 0)),
        "frac_stop_nonzero": frac_stop_nonzero(isophotes),
        "combined_drift_pix": combined,
        "spline_rms_center": spline_rms(outward),
        "max_dpa_deg": max_dpa_deg(outward),
        "max_deps": max_deps(outward),
        "outer_gerr_median": outer_gerr,
        "outward_drift_x": outward_dx,
        "outward_drift_y": outward_dy,
        "n_locked": len(locked_tail(isophotes)),
        "locked_drift_x": locked_dx,
        "locked_drift_y": locked_dy,
        "stop_code_hist": stop_code_summary_string(isophotes),
    }
