"""Multi-band curve-of-growth photometry.

Per-band cumulative flux profiles built from the shared-geometry
isophote list. Mirrors :mod:`isoster.cog` with two key differences:

1. **Shared geometry, per-band intensities.** The annular-area /
   crossing-flag computation depends only on the shared geometry
   (``sma``, ``eps``, ``x0``, ``y0``, ``pa``) and runs once per fit;
   per-band cumulative flux uses each band's ``intens_<b>`` column.
2. **Per-band column suffixes.** ``cog_<b>`` and ``cog_annulus_<b>``
   are stamped onto each row dict; the shared geometry columns
   (``area_annulus``, ``flag_cross``, ``flag_negative_area``) carry
   no suffix because they are identical across bands by construction.

Stage-3 Stage-D (plan section 7 S7).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from numpy.typing import NDArray

from ..cog import compute_ellipse_area, detect_crossing


def compute_cog_mb(
    isophotes: Sequence[Dict[str, object]],
    bands: Sequence[str],
    fix_center: bool = False,
    fix_geometry: bool = False,
    sky_offsets: Dict[str, float] | None = None,
) -> Dict[str, object]:
    """
    Compute per-band CoG for a multi-band isophote list.

    Parameters
    ----------
    isophotes : list of dict
        Multi-band isophote rows (each carrying shared geometry plus
        ``intens_<b>`` per band). Must be ordered ascending in ``sma``
        — the same convention as the driver's ``final_list``.
    bands : sequence of str
        Band names in the order matching ``IsosterConfigMB.bands``.
    fix_center, fix_geometry : bool
        Forwarded to :func:`isoster.cog.detect_crossing` to gate the
        crossing diagnostics. Multi-band typically uses fix_center=
        fix_geometry=False (free shared geometry).
    sky_offsets : dict mapping band name to offset, optional
        Per-band scalar subtracted from ``intens_<b>`` BEFORE the
        trapezoidal annulus average is computed. Use this to feed in
        the inferred sky offsets from
        :func:`isoster.multiband.plotting_mb.subtract_outermost_sky_offset`
        (or a user-supplied sky estimator) so the cumulative flux
        does not asymptote / dip in the LSB regime where the joint
        solver's per-band intercept tracks residual sky. Default
        ``None`` (no correction) preserves the historical raw-CoG
        behavior bit-identically; passing an empty dict ``{}`` is
        equivalent. Bands missing from the dict get a zero offset.

    Returns
    -------
    dict
        Keys:
          - ``sma``: shared SMA array.
          - ``area_annulus``: shared annular area, length-N float64
            (negative-annulus entries zero-clamped).
          - ``area_annulus_raw``: raw annular areas (may be negative).
          - ``flag_cross``: shared boolean array.
          - ``flag_negative_area``: shared boolean array.
          - ``cog_<b>``: per-band cumulative flux array (length N) for
            each band.
          - ``cog_annulus_<b>``: per-band annular flux array.
    """
    n_iso = len(isophotes)
    if n_iso == 0:
        empty = np.array([])
        out: Dict[str, object] = {
            "sma": empty,
            "area_annulus": empty,
            "area_annulus_raw": empty,
            "flag_cross": np.array([], dtype=bool),
            "flag_negative_area": np.array([], dtype=bool),
        }
        for b in bands:
            out[f"cog_{b}"] = empty
            out[f"cog_annulus_{b}"] = empty
        return out

    # --- Shared geometry block (single-band CoG semantics) ---
    sma = np.array([float(iso["sma"]) for iso in isophotes], dtype=np.float64)  # type: ignore[arg-type]
    eps = np.array([float(iso["eps"]) for iso in isophotes], dtype=np.float64)  # type: ignore[arg-type]
    areas = compute_ellipse_area(sma, eps)
    area_annulus_raw = np.diff(areas, prepend=0.0)

    crossing_info = detect_crossing(
        list(isophotes),
        fix_center=fix_center,
        fix_geometry=fix_geometry,
    )
    flag_cross = crossing_info["flag_cross"]
    flag_negative_area = crossing_info["flag_negative_area"]

    area_annulus = area_annulus_raw.copy()
    area_annulus[flag_negative_area] = 0.0

    out_dict: Dict[str, object] = {
        "sma": sma,
        "area_annulus": area_annulus,
        "area_annulus_raw": area_annulus_raw,
        "flag_cross": np.asarray(flag_cross, dtype=bool),
        "flag_negative_area": np.asarray(flag_negative_area, dtype=bool),
    }

    # --- Per-band cumulative flux ---
    sky_map: Dict[str, float] = dict(sky_offsets) if sky_offsets else {}
    for b in bands:
        intens_col = np.array(
            [
                float(iso[f"intens_{b}"])  # type: ignore[arg-type]
                if f"intens_{b}" in iso
                else float("nan")
                for iso in isophotes
            ],
            dtype=np.float64,
        )
        # Optional per-band sky correction: subtract a scalar offset
        # before the trapezoidal average. Sky offset is a downstream
        # correction (the fitter reports faithful per-band intercepts);
        # without it, the cumulative flux can dip past the LSB
        # transition because the joint solver's I0_b carries the
        # residual sky.
        sky_b = float(sky_map.get(b, 0.0))
        if sky_b != 0.0:
            intens_col = intens_col - sky_b

        # Trapezoidal annulus integration: average of inner+outer per-band
        # intensities. First annulus uses the band's own intensity (no
        # inner). NaNs in intens_<b> propagate — the band-drop case
        # under loose validity intentionally yields NaN cog at that
        # isophote.
        intens_inner = np.concatenate([[intens_col[0]], intens_col[:-1]])
        intens_avg = 0.5 * (intens_inner + intens_col)
        intens_avg[0] = intens_col[0]

        cog_annulus_b = intens_avg * area_annulus
        cog_b = np.cumsum(cog_annulus_b)

        out_dict[f"cog_{b}"] = cog_b
        out_dict[f"cog_annulus_{b}"] = cog_annulus_b

    return out_dict


def add_cog_mb_to_isophotes(
    isophotes: List[Dict[str, object]],
    bands: Sequence[str],
    cog_results: Dict[str, object],
) -> None:
    """Stamp the multi-band CoG output onto each row dict in place.

    Per-band columns ``cog_<b>`` / ``cog_annulus_<b>`` and shared
    columns ``area_annulus`` / ``flag_cross`` / ``flag_negative_area``
    appear on every row. Schema 1 round-trip handles them
    automatically via ``Table(rows=isophotes)`` auto-inferring columns.
    """
    n_iso = len(isophotes)
    cog_arr = cog_results.get(f"cog_{bands[0]}") if bands else None
    n_cog = int(np.asarray(cog_arr).size) if cog_arr is not None else 0
    if n_iso != n_cog:
        raise ValueError(
            f"Length mismatch: {n_iso} isophotes vs {n_cog} CoG entries. "
            f"Ensure compute_cog_mb() was called on the same isophote list."
        )
    area_annulus: NDArray[np.float64] = np.asarray(cog_results["area_annulus"], dtype=np.float64)
    flag_cross: NDArray[np.bool_] = np.asarray(cog_results["flag_cross"], dtype=bool)
    flag_negative_area: NDArray[np.bool_] = np.asarray(cog_results["flag_negative_area"], dtype=bool)
    cog_per_band: Dict[str, NDArray[np.float64]] = {
        b: np.asarray(cog_results[f"cog_{b}"], dtype=np.float64) for b in bands
    }
    cog_ann_per_band: Dict[str, NDArray[np.float64]] = {
        b: np.asarray(cog_results[f"cog_annulus_{b}"], dtype=np.float64) for b in bands
    }
    for i, iso in enumerate(isophotes):
        iso["area_annulus"] = float(area_annulus[i])
        iso["flag_cross"] = bool(flag_cross[i])
        iso["flag_negative_area"] = bool(flag_negative_area[i])
        for b in bands:
            iso[f"cog_{b}"] = float(cog_per_band[b][i])
            iso[f"cog_annulus_{b}"] = float(cog_ann_per_band[b][i])
