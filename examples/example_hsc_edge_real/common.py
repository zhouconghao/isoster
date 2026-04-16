"""Shared helpers for the example_hsc_edge_real runners.

Centralizes the galaxy list, FITS I/O, target anchor loading, and the LSB /
outskirt drift metrics used by ``run_baseline.py`` and
``run_lsb_outer_sweep.py``. All three BCGs are massive cluster centrals
observed in HSC coadds; see ``mask-strategy.md`` for how the custom masks
are built.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.interpolate import UnivariateSpline

EXAMPLE_DIR = Path(__file__).resolve().parent
DATA_DIR = EXAMPLE_DIR / "data"
OUTPUT_ROOT = EXAMPLE_DIR.parents[1] / "outputs" / "example_hsc_edge_real"

BAND = "HSC_I"
SB_ZEROPOINT = 27.0

GALAXIES = [
    ("37498869835124888", "cluster BCG with multiple companions"),
    ("42177291811318246", "close NW companion + bright edge star"),
    ("42310032070569600", "extended star halo + blended neighbour"),
]


def _galaxy_dir(obj_id: str) -> Path:
    return DATA_DIR / obj_id


def load_galaxy_data(obj_id: str, band: str = BAND):
    """Return ``(image, variance, mask)`` for one galaxy.

    The mask is read from ``{obj_id}_{band}_mask_custom.fits`` (detection-based
    custom mask, True=masked) and cast to bool.
    """
    gdir = _galaxy_dir(obj_id)
    image = fits.getdata(gdir / f"{obj_id}_{band}_image.fits").astype(np.float64)
    variance = fits.getdata(gdir / f"{obj_id}_{band}_variance.fits").astype(np.float64)
    mask = fits.getdata(gdir / f"{obj_id}_{band}_mask_custom.fits").astype(bool)
    return image, variance, mask


def load_target_anchor(obj_id: str, band: str = BAND) -> tuple[float, float]:
    """Read the target anchor ``(x0, y0)`` from the custom mask FITS header.

    The anchor is stored as ``X_OBJ`` / ``Y_OBJ`` (0-indexed smoothed-peak
    pixel coordinates) by ``build_custom_masks.py``. Hard-fails if either key
    is missing so downstream runners cannot silently fall back to frame center.
    """
    mask_path = _galaxy_dir(obj_id) / f"{obj_id}_{band}_mask_custom.fits"
    header = fits.getheader(mask_path)
    if "X_OBJ" not in header or "Y_OBJ" not in header:
        raise KeyError(
            f"{mask_path.name}: missing X_OBJ / Y_OBJ header keys. "
            "Rebuild the mask with build_custom_masks.py."
        )
    return float(header["X_OBJ"]), float(header["Y_OBJ"])


def stop_code_histogram(isophotes) -> dict:
    """Return a dict of ``stop_code -> count`` for an isophote list."""
    counts: dict = {}
    for iso in isophotes:
        code = iso.get("stop_code", -99)
        counts[code] = counts.get(code, 0) + 1
    return counts


def stop_code_summary_string(isophotes) -> str:
    """Compact ``"k:v k:v"`` histogram for logging."""
    counts = stop_code_histogram(isophotes)
    return "  ".join(f"{k}:{v}" for k, v in sorted(counts.items()))


def count_stop_code(isophotes, code: int) -> int:
    return sum(1 for iso in isophotes if iso.get("stop_code", -99) == code)


def pre_lock_outward(isophotes, sma0: float):
    """Outward isophotes with sma >= sma0, excluding any auto-lock tail."""
    out = []
    for iso in isophotes:
        if iso.get("sma", 0.0) < sma0:
            continue
        if iso.get("lsb_locked", False):
            continue
        out.append(iso)
    return out


def locked_tail(isophotes):
    return [iso for iso in isophotes if iso.get("lsb_locked", False)]


def combined_drift(isos, x0_ref: float, y0_ref: float):
    """Return ``(max_dx, max_dy, combined)`` relative to ``(x0_ref, y0_ref)``.

    ``combined = sqrt(max_dx**2 + max_dy**2)``. Returns zeros for an empty
    list so downstream table printing stays clean.
    """
    if not isos:
        return 0.0, 0.0, 0.0
    dx = np.array([abs(iso["x0"] - x0_ref) for iso in isos])
    dy = np.array([abs(iso["y0"] - y0_ref) for iso in isos])
    mdx = float(np.max(dx))
    mdy = float(np.max(dy))
    return mdx, mdy, float(np.sqrt(mdx**2 + mdy**2))


def spline_rms(isos) -> float:
    """Combined x0/y0 RMS around a smoothing spline in sma.

    Guarded for small samples (<6 isophotes returns NaN).
    """
    if len(isos) < 6:
        return float("nan")
    sma = np.array([iso["sma"] for iso in isos])
    x0s = np.array([iso["x0"] for iso in isos])
    y0s = np.array([iso["y0"] for iso in isos])
    order = np.argsort(sma)
    sma, x0s, y0s = sma[order], x0s[order], y0s[order]
    try:
        spl_x = UnivariateSpline(sma, x0s, k=3, s=len(sma))
        spl_y = UnivariateSpline(sma, y0s, k=3, s=len(sma))
    except Exception:
        return float("nan")
    rx = x0s - spl_x(sma)
    ry = y0s - spl_y(sma)
    return float(np.sqrt(np.mean(rx**2) + np.mean(ry**2)))


def outward_drift_from_anchor(isophotes, sma0: float):
    """Return ``(max|dx|, max|dy|, anchor_x0, anchor_y0)`` over outward fits.

    The anchor is the first isophote at or above ``sma0``. NaNs if no outward
    isophotes exist.
    """
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


def locked_tail_drift(isophotes):
    """Peak-to-peak ``(dx, dy)`` across the LSB-lock locked tail (NaN if empty)."""
    tail = locked_tail(isophotes)
    if not tail:
        return float("nan"), float("nan")
    x0s = np.array([iso["x0"] for iso in tail])
    y0s = np.array([iso["y0"] for iso in tail])
    return float(x0s.max() - x0s.min()), float(y0s.max() - y0s.min())


def outer_gerr_median(isophotes, sma_threshold: float) -> float:
    """Median ``|grad_error / grad|`` on isophotes with sma > threshold.

    Requires ``debug=True`` in the IsosterConfig (otherwise ``grad`` and
    ``grad_error`` are not populated). NaN if fewer than 3 eligible isophotes.
    """
    ratios = []
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


def reference_centroid(results, isophotes, sma0: float):
    """Prefer the outer-reg inner reference; else the anchor isophote at sma0."""
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
