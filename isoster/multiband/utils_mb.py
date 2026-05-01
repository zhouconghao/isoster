"""
FITS I/O for multi-band isoster results (Schema 1).

Decision D14: a single ``ISOPHOTES`` BinTableHDU with shared-geometry
columns plus per-band-suffixed columns (``intens_<b>``, ``a3_<b>``, ...).
The CONFIG HDU mirrors the existing single-band layout (PARAM/VALUE
rows, JSON-serialized values), with all multi-band-specific fields
(``bands``, ``reference_band``, ``band_weights``,
``harmonic_combination``, ``variance_mode``) recorded automatically by
``IsosterConfigMB.model_dump()``.

Provides :func:`load_bands_from_hdus` as the recommended way to extract
``(images, masks, variance_maps, bands)`` tuples from a list of FITS
HDUs whose ``FILTER`` headers carry the band identity.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from astropy.io import fits
from astropy.table import Table
from numpy.typing import NDArray

from ..utils import _NumpyEncoder, _build_config_hdu, _parse_config_hdu  # type: ignore[attr-defined]
from .config_mb import IsosterConfigMB

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Multi-band CONFIG parser (overrides the single-band parse to reconstruct
# IsosterConfigMB instead of IsosterConfig)
# ---------------------------------------------------------------------------


def _parse_config_hdu_mb(hdu: fits.BinTableHDU) -> Optional[IsosterConfigMB]:
    """
    Reconstruct an :class:`IsosterConfigMB` from a CONFIG BinTableHDU.

    Mirrors :func:`isoster.utils._parse_config_hdu` but targets the
    multi-band config class. Returns ``None`` on failure (forward-
    compatibility with newer schemas).
    """
    config_tbl = Table.read(hdu)
    if len(config_tbl) == 0:
        return None
    parsed: dict = {}
    for row in config_tbl:
        key = str(row["PARAM"])
        try:
            parsed[key] = json.loads(row["VALUE"])
        except (json.JSONDecodeError, TypeError):
            logger.warning("Skipping unparseable multi-band config key '%s'", key)
            continue
    known = set(IsosterConfigMB.model_fields.keys())
    filtered = {k: v for k, v in parsed.items() if k in known}
    try:
        return IsosterConfigMB(**filtered)
    except Exception as e:  # noqa: BLE001 — defensive
        logger.warning("Could not reconstruct IsosterConfigMB from FITS: %s", e)
        return None


# ---------------------------------------------------------------------------
# Schema 1 writer / reader
# ---------------------------------------------------------------------------


# Top-level multi-band keys persisted on the result dict — these are
# also recorded into the CONFIG HDU through model_dump so reading back
# only needs to reconstruct from CONFIG.
_TOP_LEVEL_MB_KEYS = (
    "bands",
    "multiband",
    "harmonic_combination",
    "reference_band",
    "band_weights",
    "variance_mode",
)


def isophote_results_mb_to_astropy_table(results: dict) -> Table:
    """
    Convert a multi-band isoster result dict into an astropy Table.

    Per-isophote rows include shared-geometry columns and per-band
    suffixed columns. Astropy infers column dtypes; rows missing a
    field are filled with masked / NaN values.
    """
    isophotes = results["isophotes"]
    if not isophotes:
        return Table()
    # Use Table.from_pandas-style row construction. astropy's
    # Table(rows=...) handles missing keys per row by emitting masked
    # entries.
    return Table(rows=isophotes)


def isophote_results_mb_to_fits(
    results: dict, filename: Union[str, Path], overwrite: bool = True,
) -> None:
    """
    Save multi-band isoster results to a multi-HDU FITS file (Schema 1).

    HDU layout:

    * 0: ``PrimaryHDU`` with ``MULTIBND=True`` and ``BANDS=<comma-list>`` headers.
    * 1: ``ISOPHOTES`` BinTable with shared-geometry + per-band-suffix columns.
    * 2: ``CONFIG`` BinTable (PARAM/VALUE rows from IsosterConfigMB.model_dump()).
    """
    tbl = isophote_results_mb_to_astropy_table(results)
    iso_hdu = fits.table_to_hdu(tbl)
    iso_hdu.name = "ISOPHOTES"

    config_hdu = _build_config_hdu(results)

    primary = fits.PrimaryHDU()
    primary.header["MULTIBND"] = (True, "Multi-band schema 1 result")
    bands = list(results.get("bands", []))
    if bands:
        primary.header["BANDS"] = (",".join(bands), "comma-separated band names")
    primary.header["REFBAND"] = (
        str(results.get("reference_band", "")),
        "diagnostic reference band",
    )
    primary.header["HARMCMB"] = (
        str(results.get("harmonic_combination", "joint")),
        "harmonic combination mode",
    )
    primary.header["VARMODE"] = (
        str(results.get("variance_mode", "ols")),
        "variance mode (wls or ols)",
    )

    hdulist = fits.HDUList([primary, iso_hdu, config_hdu])
    hdulist.writeto(filename, overwrite=overwrite)


def isophote_results_mb_from_fits(filename: Union[str, Path]) -> dict:
    """
    Load multi-band isoster results from a Schema-1 FITS file.

    Inverse of :func:`isophote_results_mb_to_fits`. The returned dict
    has the same shape as the in-memory result dict (`'isophotes'`
    list of dicts plus the multi-band top-level keys), suitable for
    feeding back into plotting routines or other downstream tools.
    """
    with fits.open(filename) as hdulist:
        try:
            iso_tbl = Table.read(hdulist["ISOPHOTES"])
        except KeyError:
            iso_tbl = Table.read(hdulist[1])

        config: Optional[IsosterConfigMB] = None
        if "CONFIG" in hdulist:
            config = _parse_config_hdu_mb(hdulist["CONFIG"])

        # Fall back to PrimaryHDU headers if CONFIG is missing or stale.
        primary_hdr = hdulist[0].header
        bands_from_hdr = primary_hdr.get("BANDS")
        ref_from_hdr = primary_hdr.get("REFBAND")
        harm_from_hdr = primary_hdr.get("HARMCMB")
        var_from_hdr = primary_hdr.get("VARMODE")

    isophotes: List[dict] = []
    for row in iso_tbl:
        out: dict = {}
        for col in iso_tbl.colnames:
            val = row[col]
            if isinstance(val, np.integer):
                val = int(val)
            elif isinstance(val, np.floating):
                val = float(val)
            elif isinstance(val, np.bool_):
                val = bool(val)
            elif isinstance(val, np.str_):
                val = str(val)
            out[col] = val
        isophotes.append(out)

    fix_bg_zero = False
    if config is not None:
        bands = list(config.bands)
        ref_band = config.reference_band
        harm = config.harmonic_combination
        variance_mode = config.variance_mode if config.variance_mode is not None else "ols"
        band_weights = config.resolved_band_weights()
        fix_bg_zero = bool(config.fix_per_band_background_to_zero)
    else:
        bands = bands_from_hdr.split(",") if bands_from_hdr else []
        ref_band = ref_from_hdr or (bands[0] if bands else "")
        harm = harm_from_hdr or "joint"
        variance_mode = var_from_hdr or "ols"
        band_weights = {b: 1.0 for b in bands}

    return {
        "isophotes": isophotes,
        "config": config,
        "bands": bands,
        "multiband": True,
        "harmonic_combination": harm,
        "reference_band": ref_band,
        "band_weights": band_weights,
        "variance_mode": variance_mode,
        "fix_per_band_background_to_zero": fix_bg_zero,
    }


# ---------------------------------------------------------------------------
# Bands loader helper (decision D8)
# ---------------------------------------------------------------------------


def load_bands_from_hdus(
    hdus: Sequence[Union[fits.PrimaryHDU, fits.ImageHDU]],
    *,
    filter_keyword: str = "FILTER",
    drop_prefix: str = "HSC-",
) -> Tuple[
    List[NDArray[np.floating]],
    List[Optional[NDArray[np.bool_]]],
    Optional[List[NDArray[np.floating]]],
    List[str],
]:
    """
    Convenience helper to extract aligned multi-band inputs from FITS HDUs.

    Each HDU's ``FILTER`` header value is sanitized (strip an optional
    ``HSC-`` prefix, lowercase) and used as the band name. Returns the
    image arrays, a list of all-``None`` masks (placeholder), an all-
    ``None`` variance-map list (placeholder), and the band-name list.

    This is intentionally minimal: callers that want masks or variance
    maps should construct them separately and pass them to
    :func:`fit_image_multiband`. The helper exists primarily so the
    asteris demo can do
    ``images, _, _, bands = load_bands_from_hdus(hdus)``.

    Parameters
    ----------
    hdus
        Sequence of HDUs; each must carry the FILTER header keyword.
    filter_keyword
        Header keyword to read for the band name. Default ``"FILTER"``.
    drop_prefix
        Prefix to strip from the filter value (case-insensitive).
        Default ``"HSC-"``.

    Returns
    -------
    images, masks, variance_maps, bands
        Aligned for direct unpacking into the multi-band driver.
    """
    images: List[NDArray[np.floating]] = []
    bands: List[str] = []
    for i, hdu in enumerate(hdus):
        if filter_keyword not in hdu.header:
            raise KeyError(
                f"hdus[{i}] is missing the required '{filter_keyword}' header keyword."
            )
        raw = str(hdu.header[filter_keyword]).strip()
        if drop_prefix and raw.upper().startswith(drop_prefix.upper()):
            raw = raw[len(drop_prefix):]
        # Replace remaining hyphens with underscores so the resulting
        # band name matches IsosterConfigMB's regex.
        raw = raw.replace("-", "_")
        bands.append(raw.lower())
        images.append(np.asarray(hdu.data, dtype=np.float64))
    masks: List[Optional[NDArray[np.bool_]]] = [None] * len(hdus)
    variance_maps: Optional[List[NDArray[np.floating]]] = None
    return images, masks, variance_maps, bands
