"""Shared inventory FITS writer.

Centralizes the per-galaxy-per-tool ``inventory.fits`` schema so the
fitters, runner, and stats modules agree on column order and dtypes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from astropy.io import fits
from astropy.table import Table

INVENTORY_COLUMNS: tuple[str, ...] = (
    "galaxy_id",
    "tool",
    "arm_id",
    "status",
    "error_msg",
    "wall_time_fit_s",
    "wall_time_total_s",
    "n_iso",
    "n_stop_0",
    "n_stop_1",
    "n_stop_2",
    "n_stop_m1",
    "frac_stop_nonzero",
    "stop_code_hist",
    "combined_drift_pix",
    "spline_rms_center",
    "max_dpa_deg",
    "max_deps",
    "outer_gerr_median",
    "outward_drift_x",
    "outward_drift_y",
    "n_locked",
    "locked_drift_x",
    "locked_drift_y",
    "first_isophote_failure",
    "first_isophote_retry_attempts",
    "first_isophote_retry_stop_codes",
    "resid_rms_inner",
    "resid_rms_mid",
    "resid_rms_outer",
    "resid_median_inner",
    "resid_median_mid",
    "resid_median_outer",
    "frac_above_3sigma_outer",
    "image_sigma_adu",
    "sigma_method",
    "n_iso_ref_used",
    "flags",
    "flag_severity_max",
    "composite_score",
    "qa_path",
    "profile_path",
    "model_path",
    "config_path",
)

INTEGER_COLUMNS = frozenset(
    {
        "n_iso",
        "n_stop_0",
        "n_stop_1",
        "n_stop_2",
        "n_stop_m1",
        "n_locked",
        "first_isophote_retry_attempts",
        "n_iso_ref_used",
    }
)

BOOLEAN_COLUMNS = frozenset({"first_isophote_failure"})

FLOAT_COLUMNS = frozenset(
    {
        "wall_time_fit_s",
        "wall_time_total_s",
        "frac_stop_nonzero",
        "combined_drift_pix",
        "spline_rms_center",
        "max_dpa_deg",
        "max_deps",
        "outer_gerr_median",
        "outward_drift_x",
        "outward_drift_y",
        "locked_drift_x",
        "locked_drift_y",
        "resid_rms_inner",
        "resid_rms_mid",
        "resid_rms_outer",
        "resid_median_inner",
        "resid_median_mid",
        "resid_median_outer",
        "frac_above_3sigma_outer",
        "image_sigma_adu",
        "flag_severity_max",
        "composite_score",
    }
)


def column_default(column: str) -> Any:
    if column in INTEGER_COLUMNS:
        return 0
    if column in BOOLEAN_COLUMNS:
        return False
    if column in FLOAT_COLUMNS:
        return float("nan")
    return ""


def write_inventory(rows: Iterable[dict[str, Any]], path: Path) -> None:
    """Write one row per fit to ``path`` as an astropy ``Table`` / FITS.

    Missing fields are filled with type-appropriate defaults. The
    output file has a single ``INVENTORY`` binary table HDU.
    """
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ordered: dict[str, list[Any]] = {}
    for col in INVENTORY_COLUMNS:
        default = column_default(col)
        values: list[Any] = []
        for row in rows:
            value = row.get(col, default)
            if col in INTEGER_COLUMNS:
                value = _coerce_int(value, default)
            elif col in BOOLEAN_COLUMNS:
                value = bool(value)
            elif col in FLOAT_COLUMNS:
                value = _coerce_float(value, default)
            else:
                value = "" if value is None else str(value)
            values.append(value)
        ordered[col] = values
    table = Table(ordered)
    table.meta["EXTNAME"] = "INVENTORY"
    # Force overwrite so re-runs replace old rows.
    table.write(path, overwrite=True)


def read_inventory(path: Path) -> Table:
    """Read an inventory FITS; raises a clear error if the file is missing."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"inventory not found: {path}")
    with fits.open(path) as hdul:
        # Prefer the INVENTORY extension; fall back to first table HDU.
        for hdu in hdul:
            if getattr(hdu, "name", "").upper() == "INVENTORY" and hasattr(hdu, "data"):
                return Table(hdu.data)
    return Table.read(path)


def _coerce_int(value: Any, default: int) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return default
    return coerced


def _coerce_float(value: Any, default: float) -> float:
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return default
    return coerced


__all__ = [
    "INVENTORY_COLUMNS",
    "write_inventory",
    "read_inventory",
    "column_default",
    "INTEGER_COLUMNS",
    "BOOLEAN_COLUMNS",
    "FLOAT_COLUMNS",
]
