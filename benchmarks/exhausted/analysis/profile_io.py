"""Tool-agnostic readers for per-isophote profile FITS tables.

This module collects the gotchas we hit reading profile FITS produced
by ``photutils.isophote.IsophoteList.to_table()``, ``isoster``, and
``autoprof``. Drop into any project that needs to compute scalar
metrics from those profiles. Pure-Python (``numpy`` + ``astropy.io``).

Profile-table conventions handled here:

1. **`pa` column units.** ``photutils`` writes ``pa`` in degrees with
   ``TUNIT='deg'`` attached to the column. ``np.asarray()`` silently
   strips the unit; downstream code that treats the bare value as
   radians produces ~57x errors. ``read_pa_in_radians`` honours the
   FITS unit metadata.

2. **`eps` vs `ellipticity` column name.** ``photutils`` and
   ``autoprof`` write ``ellipticity`` / ``ellipticity_err``;
   ``isoster`` writes both ``eps`` and ``ellipticity`` (with ``eps``
   as the canonical name in older versions). Always alias.

3. **`sma=0` central-pixel sentinel row.** Both ``photutils`` and
   ``isoster`` prepend a row at index 0 representing the un-fitted
   central pixel: ``sma=0, eps=0, pa=0, stop_code=0``. ``autoprof``
   does not write a sentinel. Any diff-loop computing ``max_dpa``,
   ``max_deps``, drift, or stop-code statistics MUST filter ``sma>0``
   first; otherwise the sentinel-to-row-1 jump produces spurious
   60-90 deg pa diffs and 0.4-0.6 eps diffs.

This module exposes:

- ``read_pa_in_radians(tbl)`` -- safe pa reader honouring TUNIT.
- ``read_eps(tbl)`` -- handles the eps/ellipticity alias.
- ``valid_isophote_mask(tbl)`` -- boolean mask of real-fit rows
  (drops the ``sma=0`` sentinel).
- ``mid_row_geometry(profile_path)`` -- a fallback initial-geometry
  dict from the median real-fit row of a profile FITS, useful when
  no catalog geometry is available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.table import Table


def read_pa_in_radians(tbl: Table) -> np.ndarray | None:
    """Read the ``pa`` column from a profile FITS in radians.

    Honours the column unit metadata: ``photutils.IsophoteList.to_table()``
    writes ``pa`` in degrees with ``u.deg`` attached, which round-trips
    through FITS as ``TUNIT='deg'`` on the column. ``np.asarray()``
    silently strips this; treating bare degrees as radians produces
    roughly ``180/pi ~ 57`` times wrong values in any downstream
    computation.

    Returns ``None`` when the column is missing.
    """
    if "pa" not in tbl.colnames:
        return None
    col = tbl["pa"]
    raw = np.asarray(col, dtype=np.float64)
    unit = getattr(col, "unit", None)
    if unit is not None and str(unit).lower() in ("deg", "degree", "degrees"):
        return np.deg2rad(raw)
    return raw


def read_eps(tbl: Table) -> np.ndarray | None:
    """Read ellipticity, handling the ``eps`` vs ``ellipticity`` alias.

    Prefers ``eps`` if both are present; falls back to ``ellipticity``.
    Returns ``None`` when neither exists.
    """
    if "eps" in tbl.colnames:
        return np.asarray(tbl["eps"], dtype=np.float64)
    if "ellipticity" in tbl.colnames:
        return np.asarray(tbl["ellipticity"], dtype=np.float64)
    return None


def valid_isophote_mask(tbl: Table) -> np.ndarray | None:
    """Boolean mask selecting real fitted isophote rows.

    Drops the ``sma=0`` central-pixel sentinel that photutils + isoster
    write at index 0 (``sma=0, eps=0, pa=0, stop_code=0``). Autoprof
    has no sentinel, so all rows are "valid" there.

    Returns ``None`` when the profile has no ``sma`` column.
    """
    if "sma" not in tbl.colnames:
        return None
    sma = np.asarray(tbl["sma"], dtype=np.float64)
    return np.isfinite(sma) & (sma > 0)


def mid_row_geometry(profile_path: str | Path) -> dict[str, float] | None:
    """Mid-row geometry from an isoster/photutils/autoprof profile FITS.

    Useful as a fallback initial-geometry source when no catalog
    geometry is available. Returns a dict with::

        x0, y0           -- centre in pixels
        eps              -- 1 - b/a (clipped to >= 1e-6)
        pa_rad           -- math convention, CCW from +x
        maxsma_pix       -- largest fitted sma
        min_sma_pix      -- smallest real-fit sma (sentinel excluded)

    Returns ``None`` if the profile has fewer than 5 real-fit rows.
    """
    profile_path = Path(profile_path)
    with fits.open(profile_path) as hdul:
        data = hdul[1].data
        header = hdul[1].header
    tbl = Table(data)
    valid = valid_isophote_mask(tbl)
    if valid is None or int(valid.sum()) < 5:
        return None
    rows = np.where(valid)[0]
    mid = int(rows[len(rows) // 2])
    sma = np.asarray(tbl["sma"], dtype=np.float64)

    pa_value = float(tbl["pa"][mid])
    pa_unit = ""
    for idx, name in enumerate(tbl.colnames, start=1):
        if name.lower() == "pa":
            pa_unit = str(header.get(f"TUNIT{idx}", "")).strip().lower()
            break
    if pa_unit in ("deg", "degree", "degrees"):
        pa_value = float(np.deg2rad(pa_value))

    eps_arr = read_eps(tbl)
    eps_mid = float(eps_arr[mid]) if eps_arr is not None else 0.0

    return {
        "x0": float(tbl["x0"][mid]),
        "y0": float(tbl["y0"][mid]),
        "eps": float(max(eps_mid, 1e-6)),
        "pa_rad": pa_value,
        "maxsma_pix": float(np.nanmax(sma)),
        "min_sma_pix": float(np.min(sma[valid])),
    }


def profile_summary(profile_path: str | Path) -> dict[str, Any]:
    """One-stop scalar summary of a profile FITS.

    Computes the depth + drift metrics commonly emitted by isophote
    fitters (``n_iso``, ``min_sma_pix``, ``max_sma_pix``, ``max_dpa_deg``,
    ``max_deps``, drift, stop-code histogram, ``faintest_sb``) on the
    real-fit rows only -- the ``sma=0`` sentinel is filtered out per
    the convention above.

    Returns an empty dict if the file does not exist or carries no
    usable rows.
    """
    profile_path = Path(profile_path)
    if not profile_path.is_file():
        return {}
    with fits.open(profile_path) as hdul:
        table_hdu_index = None
        for hdu_index, hdu in enumerate(hdul):
            if getattr(hdu, "name", "").upper() == "ISOPHOTES" and hasattr(hdu, "data"):
                table_hdu_index = hdu_index
                break
        if table_hdu_index is None:
            for hdu_index, hdu in enumerate(hdul[1:], start=1):
                if hasattr(hdu, "data") and hdu.data is not None:
                    table_hdu_index = hdu_index
                    break
        if table_hdu_index is None:
            return {}
    tbl = Table.read(profile_path, hdu=table_hdu_index)
    out: dict[str, Any] = {}

    sma_raw = np.asarray(tbl["sma"], dtype=np.float64) if "sma" in tbl.colnames else None
    sb_raw = np.asarray(tbl["sma_sb"], dtype=np.float64) if "sma_sb" in tbl.colnames else None
    eps_raw = read_eps(tbl)
    pa_rad_raw = read_pa_in_radians(tbl)
    x0_raw = np.asarray(tbl["x0"], dtype=np.float64) if "x0" in tbl.colnames else None
    y0_raw = np.asarray(tbl["y0"], dtype=np.float64) if "y0" in tbl.colnames else None
    stop_raw = np.asarray(tbl["stop_code"], dtype=np.float64) if "stop_code" in tbl.colnames else None

    valid = valid_isophote_mask(tbl)

    def _apply(arr: np.ndarray | None) -> np.ndarray | None:
        if arr is None or valid is None or valid.size != arr.size:
            return arr
        return arr[valid]

    sma = _apply(sma_raw)
    eps = _apply(eps_raw)
    pa_rad = _apply(pa_rad_raw)
    x0 = _apply(x0_raw)
    y0 = _apply(y0_raw)
    stop = _apply(stop_raw)

    if sma is not None and sma.size > 0:
        out["max_sma_pix"] = float(np.nanmax(sma))
        out["min_sma_pix"] = float(np.min(sma))

    if sb_raw is not None and np.any(np.isfinite(sb_raw)):
        out["faintest_sb"] = float(np.nanmax(sb_raw))

    if x0 is not None and y0 is not None and x0.size > 1:
        dx = np.diff(x0)
        dy = np.diff(y0)
        with np.errstate(invalid="ignore"):
            dr = np.sqrt(dx * dx + dy * dy)
        out["combined_drift_pix"] = float(np.nanmax(dr)) if dr.size else float("nan")
        out["outward_drift_x"] = float(x0[-1] - x0[0])
        out["outward_drift_y"] = float(y0[-1] - y0[0])

    if pa_rad is not None and pa_rad.size > 1:
        pa_deg = np.rad2deg(pa_rad) % 180.0
        d = np.diff(pa_deg)
        d = (d + 90.0) % 180.0 - 90.0
        with np.errstate(invalid="ignore"):
            max_d = float(np.nanmax(np.abs(d))) if d.size else 0.0
        out["max_dpa_deg"] = max_d

    if eps is not None and eps.size > 1:
        d_eps = np.diff(eps)
        out["max_deps"] = float(np.nanmax(np.abs(d_eps))) if d_eps.size else 0.0

    if stop is not None and stop.size > 0:
        n = stop.size
        out["n_iso"] = int(n)
        out["n_stop_0"] = int(np.sum(stop == 0))
        out["n_stop_1"] = int(np.sum(stop == 1))
        out["n_stop_2"] = int(np.sum(stop == 2))
        out["n_stop_m1"] = int(np.sum(stop == -1))
        out["frac_stop_nonzero"] = float((n - out["n_stop_0"]) / n) if n > 0 else float("nan")

    return out


__all__ = [
    "read_pa_in_radians",
    "read_eps",
    "valid_isophote_mask",
    "mid_row_geometry",
    "profile_summary",
]
