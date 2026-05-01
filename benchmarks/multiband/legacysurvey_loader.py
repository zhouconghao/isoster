"""LegacySurvey grz loader for the multi-band isoster benchmark.

Handles the LegacySurvey/SGA-2020 layout (CompImageHDU images, inverse
variance instead of variance, pre-built combined mask) and exposes a
single `load_legacysurvey_grz` helper that returns numpy arrays plus the
photometric metadata isoster needs.

Conventions enforced here (per `CLAUDE.md`):

- LegacySurvey zeropoint `zp = 22.5` (image BUNIT = nanomaggy).
- DECaLS pixel scale read from the WCS (`|CD1_1|`).  Falls back to the
  catalogued 0.262 arcsec/pix only if missing.
- Variance is computed as `1.0 / invvar` with `invvar <= 0` clamped to
  the sanitization sentinel `1e-30` and the corresponding pixels added
  to the mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from astropy.io import fits

LEGACYSURVEY_ZP = 22.5
DECALS_PIXEL_SCALE_ARCSEC = 0.262
DEFAULT_BANDS: tuple[str, ...] = ("g", "r", "z")


@dataclass(frozen=True)
class LegacySurveyCutout:
    bands: List[str]
    images: List[np.ndarray]
    variances: List[np.ndarray]
    invvar_zero_masks: List[np.ndarray]
    combined_mask: np.ndarray
    pixel_scale_arcsec: float
    zp: float
    shape: tuple[int, int]
    headers: Dict[str, fits.Header]
    extra_mask_layers: Dict[str, np.ndarray]


def _read_image(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None:
                return np.asarray(hdu.data, dtype=np.float32), hdu.header
    raise IOError(f"No data HDU found in {path}")


def _pixel_scale_from_header(header: fits.Header) -> float:
    cd11 = header.get("CD1_1")
    if cd11 is not None:
        return float(abs(cd11) * 3600.0)
    cdelt1 = header.get("CDELT1")
    if cdelt1 is not None:
        return float(abs(cdelt1) * 3600.0)
    return DECALS_PIXEL_SCALE_ARCSEC


def load_legacysurvey_grz(
    galaxy_dir: str | Path,
    galaxy_prefix: str,
    bands: Sequence[str] = DEFAULT_BANDS,
    combined_mask_filename: Optional[str] = None,
) -> LegacySurveyCutout:
    """Load LegacySurvey images, inverse-variance, and combined mask.

    Parameters
    ----------
    galaxy_dir
        Directory containing the SGA-2020 cutout files.
    galaxy_prefix
        File-name prefix (e.g. ``"PGC006669-largegalaxy"``).
    bands
        Tuple of band letters; defaults to ``("g", "r", "z")``.
    combined_mask_filename
        Optional override for the combined-mask file.  If ``None`` we
        try ``<galaxy>-mask.fits`` (SGA-2020 helper output).

    Returns
    -------
    LegacySurveyCutout
        Image stack, variance stack, per-band invvar-zero masks, the
        combined object/contaminant mask, the pixel scale, and the
        zeropoint.  Callers that want a single broadcast mask can pass
        ``cutout.combined_mask`` straight to ``fit_image_multiband``.
    """

    galaxy_dir = Path(galaxy_dir)
    images: List[np.ndarray] = []
    variances: List[np.ndarray] = []
    invvar_zero_masks: List[np.ndarray] = []
    headers: Dict[str, fits.Header] = {}
    pixel_scales: List[float] = []
    shape: tuple[int, int] | None = None

    for band in bands:
        img_path = galaxy_dir / f"{galaxy_prefix}-image-{band}.fits.fz"
        invvar_path = galaxy_dir / f"{galaxy_prefix}-invvar-{band}.fits.fz"
        image, header = _read_image(img_path)
        invvar, _ = _read_image(invvar_path)
        if image.shape != invvar.shape:
            raise ValueError(
                f"Image and invvar shapes disagree for band {band}: "
                f"{image.shape} vs {invvar.shape}"
            )
        if shape is None:
            shape = image.shape
        elif image.shape != shape:
            raise ValueError(
                f"Band {band} shape {image.shape} disagrees with first band {shape}"
            )

        bad = ~np.isfinite(invvar) | (invvar <= 0.0)
        safe_invvar = np.where(bad, 1.0, invvar)
        variance = 1.0 / safe_invvar
        variance[bad] = 1e30

        images.append(image.astype(np.float32))
        variances.append(variance.astype(np.float32))
        invvar_zero_masks.append(bad)
        headers[band] = header
        pixel_scales.append(_pixel_scale_from_header(header))

    pixel_scale = float(np.median(pixel_scales))
    if not np.allclose(pixel_scales, pixel_scale, rtol=1e-6):
        raise ValueError(
            f"Per-band pixel scales disagree: {dict(zip(bands, pixel_scales))}"
        )

    assert shape is not None, "load loop must populate shape from at least one band"

    if combined_mask_filename is None:
        combined_mask_filename = f"{galaxy_prefix.split('-')[0]}-mask.fits"
    combined_path = galaxy_dir / combined_mask_filename
    extra_layers: Dict[str, np.ndarray] = {}
    if combined_path.exists():
        with fits.open(combined_path) as hdul:
            combined_mask = np.asarray(hdul[0].data, dtype=bool)
            for hdu in hdul[1:]:
                ext = hdu.header.get("EXTNAME", f"HDU{len(extra_layers)+1}")
                if hdu.data is not None:
                    extra_layers[ext] = np.asarray(hdu.data, dtype=bool)
    else:
        combined_mask = np.zeros(shape, dtype=bool)

    for bad in invvar_zero_masks:
        combined_mask = combined_mask | bad

    return LegacySurveyCutout(
        bands=list(bands),
        images=images,
        variances=variances,
        invvar_zero_masks=invvar_zero_masks,
        combined_mask=combined_mask,
        pixel_scale_arcsec=pixel_scale,
        zp=LEGACYSURVEY_ZP,
        shape=shape,
        headers=headers,
        extra_mask_layers=extra_layers,
    )


def asinh_softening_from_log10_match(
    pixel_scale_arcsec: float,
    zp: float,
    bright_mu: float = 22.0,
) -> float:
    """Pick the asinh softening intensity (per pixel, native flux units).

    The asinh-SB convention used by the multi-band plotter approaches the
    log10 form for ``I >> softening``.  We choose the softening such that
    at a "high SB" reference (default mu = 22 mag/arcsec^2 — well above
    sky for LegacySurvey grz) the asinh and log10 expressions agree to
    better than 0.01 mag.

    Returns the per-pixel intensity floor in the same flux units as the
    image (nanomaggies for LegacySurvey).
    """

    pixarea = pixel_scale_arcsec ** 2
    intensity_per_pix = 10.0 ** ((zp - bright_mu) / 2.5) * pixarea
    return float(intensity_per_pix * 1e-3)
