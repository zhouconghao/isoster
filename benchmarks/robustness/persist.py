"""Per-row isophote FITS persistence for the robustness benchmark.

Writes one ``BinTableHDU`` per fit (perturbation row or reference) so that
1-D profiles can be reconstructed offline without re-running the sweep.
Layout follows ``docs/agent/journal/2026-04-15_robustness_plan.md`` §9:

``outputs/benchmark_robustness/{tier}/sweep/{arm}/{obj_id}/{axis}_{value:+.2f}.fits``
``outputs/benchmark_robustness/{tier}/reference/{obj_id}/{obj_id}_reference.fits``

The writer stashes per-row metadata (start conditions, success flag,
stop-code histogram, error message, ...) in the FITS primary header using
``HIERARCH`` keywords so long names survive. The reader returns the
isophote columns as a dict of ``numpy`` arrays plus the metadata dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from astropy.io import fits
from astropy.table import Table

ISOPHOTE_COLUMNS: tuple[str, ...] = (
    "sma",
    "intens",
    "intens_err",
    "rms",
    "eps",
    "eps_err",
    "pa",
    "pa_err",
    "x0",
    "x0_err",
    "y0",
    "y0_err",
    "a3",
    "a3_err",
    "b3",
    "b3_err",
    "a4",
    "a4_err",
    "b4",
    "b4_err",
    "tflux_e",
    "tflux_c",
    "npix_e",
    "npix_c",
    "stop_code",
    "niter",
    "valid",
    "ndata",
    "nflag",
    "grad",
    "grad_error",
    "grad_r_error",
)

_INT_COLUMNS = frozenset({"npix_e", "npix_c", "stop_code", "niter", "ndata", "nflag"})
_BOOL_COLUMNS = frozenset({"valid"})


def _row_path(
    output_root: Path,
    tier: str,
    arm: str,
    obj_id: str,
    axis: str,
    value: float,
) -> Path:
    """Return the per-row FITS path under ``{tier}/sweep/{arm}/{obj_id}/``."""
    return (
        output_root
        / tier
        / "sweep"
        / arm
        / obj_id
        / f"{axis}_{value:+.2f}.fits"
    )


def _reference_path(output_root: Path, tier: str, obj_id: str) -> Path:
    """Return the reference-fit FITS path under ``{tier}/reference/{obj_id}/``."""
    return (
        output_root
        / tier
        / "reference"
        / obj_id
        / f"{obj_id}_reference.fits"
    )


def _isophotes_to_table(isophotes: Sequence[Mapping[str, Any]]) -> Table:
    """Pack a list of isophote dicts into an astropy Table with stable dtypes."""
    n = len(isophotes)
    data: Dict[str, np.ndarray] = {}
    for col in ISOPHOTE_COLUMNS:
        if col in _INT_COLUMNS:
            arr = np.zeros(n, dtype=np.int64)
            for i, iso in enumerate(isophotes):
                arr[i] = int(iso.get(col, 0))
        elif col in _BOOL_COLUMNS:
            arr = np.zeros(n, dtype=bool)
            for i, iso in enumerate(isophotes):
                arr[i] = bool(iso.get(col, False))
        else:
            arr = np.full(n, np.nan, dtype=np.float64)
            for i, iso in enumerate(isophotes):
                v = iso.get(col, np.nan)
                arr[i] = float(v) if v is not None else np.nan
        data[col] = arr
    return Table(data, names=ISOPHOTE_COLUMNS)


def _set_header(header: fits.Header, meta: Mapping[str, Any]) -> None:
    """Write ``meta`` keys into ``header`` using HIERARCH for long names.

    Values that are ``None`` are stored as empty strings. Floats / ints /
    bools pass through directly; anything else is coerced to ``str`` and
    clipped to 60 chars to stay well under the FITS card limit.
    """
    for key, value in meta.items():
        # HIERARCH is needed for keys longer than 8 chars or with lowercase
        card_key = f"HIERARCH {key}" if len(key) > 8 or not key.isupper() else key
        if value is None:
            card_value: Any = ""
        elif isinstance(value, bool):
            card_value = bool(value)
        elif isinstance(value, (int, np.integer)):
            card_value = int(value)
        elif isinstance(value, (float, np.floating)):
            fv = float(value)
            card_value = fv if np.isfinite(fv) else "nan"
        else:
            card_value = str(value)[:60]
        try:
            header[card_key] = card_value
        except (ValueError, fits.VerifyError):
            # Last-resort: stringify and retry under a shortened key.
            header[card_key] = str(card_value)[:60]


def write_isophote_fits(
    path: Path,
    isophotes: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> Path:
    """Write ``isophotes`` + ``metadata`` to a single-HDU FITS file.

    The primary HDU is empty. The ``ISOPHOTES`` binary table carries the
    isophote columns; the primary header carries per-row metadata via
    HIERARCH keywords. Returns the written path for logging convenience.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    primary = fits.PrimaryHDU()
    _set_header(primary.header, metadata)
    table = _isophotes_to_table(isophotes)
    bintable = fits.BinTableHDU(table, name="ISOPHOTES")
    fits.HDUList([primary, bintable]).writeto(path, overwrite=True)
    return path


def write_row_fits(
    output_root: Path,
    tier: str,
    arm: str,
    obj_id: str,
    axis: str,
    value: float,
    isophotes: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> Path:
    """Convenience wrapper: compute the per-row path and write."""
    path = _row_path(output_root, tier, arm, obj_id, axis, value)
    return write_isophote_fits(path, isophotes, metadata)


def write_reference_fits(
    output_root: Path,
    tier: str,
    obj_id: str,
    isophotes: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> Path:
    """Convenience wrapper: compute the reference path and write."""
    path = _reference_path(output_root, tier, obj_id)
    return write_isophote_fits(path, isophotes, metadata)


def read_isophote_fits(path: Path) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Read a persisted isophote FITS back into column arrays + metadata.

    Used by the figure-building code so it does not need to re-run a fit.
    """
    primary_header = fits.getheader(str(path), ext=0)
    table_data = fits.getdata(str(path), extname="ISOPHOTES")
    table = Table(table_data)
    columns: Dict[str, np.ndarray] = {
        name: np.asarray(table[name]) for name in table.colnames
    }
    metadata: Dict[str, Any] = {}
    for card in primary_header.cards:
        key = card.keyword
        if key in ("SIMPLE", "BITPIX", "NAXIS", "EXTEND", "COMMENT", "HISTORY", ""):
            continue
        metadata[key] = card.value
    return columns, metadata
