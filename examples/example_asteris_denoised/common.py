"""Shared helpers for the example_asteris_denoised runners.

The asteris dataset is a small set of HSC i-band cutouts (768x768) where each
galaxy ships *two* image versions:

* ``noisy.fits``     — original HSC coadd (zp=27, pixel scale 0.168 arcsec/pix).
* ``denoised.fits``  — same cutout after a learned denoising step. Same WCS
  and astrometry as the noisy version, but lower per-pixel sky RMS.

The data folder is not copied into the repo; users point ``DATA_ROOT`` at
``~/Downloads/asteris/objs`` (or override via the ``ASTERIS_DATA_ROOT``
environment variable).

This module centralizes paths, photometric constants, the galaxy list, and a
couple of helpers (sigma-clipped sky std, uniform variance map) that the mask
builder and the isoster runner both use.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

EXAMPLE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = EXAMPLE_DIR.parents[1] / "outputs" / "example_asteris_denoised"

# Photometric constants for HSC coadds.
SB_ZEROPOINT = 27.0
PIXEL_SCALE_ARCSEC = 0.168

# Detection band: only HSC-I is used in this example, but the helpers accept
# a band argument so the same code can be reused for the other four filters.
DEFAULT_BAND = "HSC-I"

# Default location of the asteris cutouts. Override with ASTERIS_DATA_ROOT.
DATA_ROOT = Path(
    os.environ.get(
        "ASTERIS_DATA_ROOT",
        str(Path.home() / "Downloads" / "asteris" / "objs"),
    )
).expanduser()

# Each entry: (short_id, full_folder_name, description). The short id is what
# we use as the directory key in outputs/ and as the value of OBJECT in
# generated FITS headers.
GALAXIES = [
    (
        "37484563299062823",
        "obj_37484563299062823_pdr3_dud_rev_8523_0_3_cutout768_uniform8_t8_v1",
        "asteris demo galaxy A",
    ),
    (
        "37485405112652042",
        "obj_37485405112652042_pdr3_dud_rev_8523_6_7_cutout768_uniform8_t8_v1",
        "asteris demo galaxy B",
    ),
    (
        "37493917737837940",
        "obj_37493917737837940_pdr3_dud_rev_8525_4_5_cutout768_uniform8_t8_v1",
        "asteris demo galaxy C",
    ),
    (
        "37494055176793569",
        "obj_37494055176793569_pdr3_dud_rev_8525_5_5_cutout768_uniform8_t8_v1",
        "asteris demo galaxy D",
    ),
    (
        "38553980090996188",
        "obj_38553980090996188_pdr3_dud_rev_8766_5_4_cutout768_uniform8_t8_v1",
        "asteris demo galaxy E",
    ),
]


def galaxy_dir(obj_id: str, band: str = DEFAULT_BAND) -> Path:
    """Return the source directory holding ``noisy.fits`` / ``denoised.fits``."""
    full = next((full for sid, full, _ in GALAXIES if sid == obj_id), None)
    if full is None:
        raise KeyError(f"Unknown asteris object id: {obj_id}")
    return DATA_ROOT / full / band


def output_dir(obj_id: str) -> Path:
    """Return (and create) the per-galaxy output directory under ``outputs/``."""
    out = OUTPUT_ROOT / obj_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_pair(obj_id: str, band: str = DEFAULT_BAND):
    """Load ``(noisy, denoised)`` images as float64 arrays."""
    gdir = galaxy_dir(obj_id, band)
    noisy = fits.getdata(gdir / "noisy.fits").astype(np.float64)
    denoised = fits.getdata(gdir / "denoised.fits").astype(np.float64)
    if noisy.shape != denoised.shape:
        raise ValueError(
            f"{obj_id}: noisy {noisy.shape} and denoised {denoised.shape} "
            "shapes do not match"
        )
    return noisy, denoised


def sky_std(image: np.ndarray, sigma: float = 3.0, maxiters: int = 5) -> float:
    """Sigma-clipped standard deviation, used as a uniform sky RMS estimate."""
    _, _, std = sigma_clipped_stats(image, sigma=sigma, maxiters=maxiters)
    return float(std)


def uniform_variance_map(image: np.ndarray) -> np.ndarray:
    """Return a constant variance map matching ``image.shape``.

    The variance is ``sky_std(image) ** 2``. Using a uniform variance gives
    the WLS fitter clean photon-noise error bars on extracted intensities
    (``intens_err`` becomes ``sky_std / sqrt(N_eff)``) without bleeding any
    real galaxy structure into the weights.
    """
    sigma = sky_std(image)
    return np.full(image.shape, sigma * sigma, dtype=np.float64)


def load_target_anchor_from_mask(mask_path: Path) -> tuple[float, float]:
    """Read the target anchor from a mask FITS header (``X_OBJ`` / ``Y_OBJ``)."""
    header = fits.getheader(mask_path)
    if "X_OBJ" not in header or "Y_OBJ" not in header:
        raise KeyError(
            f"{mask_path.name}: missing X_OBJ / Y_OBJ keys; rebuild the mask."
        )
    return float(header["X_OBJ"]), float(header["Y_OBJ"])
