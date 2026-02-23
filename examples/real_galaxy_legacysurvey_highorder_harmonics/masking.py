"""
Object masking for real galaxy images using a two-stage photutils pipeline.

Migrated inline from the confeti masking strategy.  Mask convention:
``True`` = masked pixel (bad), matching isoster's expected mask format.

Stage 1 detects and masks field contaminants (stars, other galaxies).
Stage 2 (optional) detects compact on-galaxy sources (star-forming knots,
superposed stars) by fitting and subtracting the smooth galaxy envelope first.
"""

from __future__ import annotations

import warnings

import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.io import fits
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import SourceCatalog, detect_sources


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_object_mask(
    image: np.ndarray,
    *,
    # Stage-1 background / detection
    box_size: int = 32,
    filter_size: int = 3,
    detect_fwhm: float = 3.0,
    nsigma: float = 1.5,
    npixels: int = 5,
    dilate_fwhm: float = 12.0,
    # Stage-2 on-galaxy compact sources
    on_galaxy: bool = False,
    on_galaxy_box: int = 8,
    on_galaxy_filter: int = 3,
    on_galaxy_nsigma: float = 1.5,
    on_galaxy_npixels: int = 5,
    on_galaxy_dilate_fwhm: float = 10.0,
    # Center of target galaxy (image coordinates, 0-indexed)
    center_xy: tuple[float, float] | None = None,
) -> np.ndarray:
    """Generate a bad-pixel mask for a galaxy image.

    Parameters
    ----------
    image : 2D float array
        Science image (background not pre-subtracted).
    box_size : int
        Box size for Stage-1 ``Background2D`` estimator.
    filter_size : int
        Median filter size for Stage-1 background mesh smoothing.
    detect_fwhm : float
        FWHM (pixels) of the Gaussian kernel used for source detection convolution.
    nsigma : float
        Detection threshold in units of background RMS.
    npixels : int
        Minimum connected pixels required to form a source.
    dilate_fwhm : float
        FWHM (pixels) of dilation kernel applied to Stage-1 contaminant segments.
    on_galaxy : bool
        When True, also run Stage-2 to detect compact on-galaxy sources.
    on_galaxy_box : int
        Box size for Stage-2 local background (should be small, e.g. 4–8).
    on_galaxy_filter : int
        Filter size for Stage-2 background mesh smoothing.
    on_galaxy_nsigma : float
        Detection threshold for Stage-2 compact-source detection.
    on_galaxy_npixels : int
        Minimum connected pixels for Stage-2 sources.
    on_galaxy_dilate_fwhm : float
        Dilation FWHM for Stage-2 compact source masks.
    center_xy : tuple (x, y) or None
        Pixel coordinates of the target galaxy centre (0-indexed).
        Defaults to the image centre.

    Returns
    -------
    mask : 2D bool array
        True where pixels are bad/masked.
    """
    h, w = image.shape
    if center_xy is None:
        center_xy = ((w - 1) / 2.0, (h - 1) / 2.0)

    # ------------------------------------------------------------------
    # Stage 1 — field contaminants
    # ------------------------------------------------------------------
    mask_stage1 = _detect_field_contaminants(
        image,
        center_xy=center_xy,
        box_size=box_size,
        filter_size=filter_size,
        detect_fwhm=detect_fwhm,
        nsigma=nsigma,
        npixels=npixels,
        dilate_fwhm=dilate_fwhm,
    )

    combined_mask = mask_stage1.copy()

    # ------------------------------------------------------------------
    # Stage 2 — on-galaxy compact sources (optional)
    # ------------------------------------------------------------------
    if on_galaxy:
        mask_stage2 = _detect_on_galaxy(
            image,
            stage1_mask=mask_stage1,
            center_xy=center_xy,
            box_size=on_galaxy_box,
            filter_size=on_galaxy_filter,
            nsigma=on_galaxy_nsigma,
            npixels=on_galaxy_npixels,
            dilate_fwhm=on_galaxy_dilate_fwhm,
        )
        combined_mask |= mask_stage2

    return combined_mask


def save_mask_fits(mask: np.ndarray, output_path: str) -> None:
    """Save a boolean mask as a FITS file (uint8, 1 = masked)."""
    hdu = fits.PrimaryHDU(mask.astype(np.uint8))
    hdu.header["COMMENT"] = "Bad-pixel mask: 1 = masked (True)"
    hdu.writeto(output_path, overwrite=True)


def load_mask_fits(fits_path: str) -> np.ndarray:
    """Load a boolean mask saved by :func:`save_mask_fits`."""
    with fits.open(fits_path) as hdul:
        return hdul[0].data.astype(bool)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_background(
    image: np.ndarray,
    box_size: int,
    filter_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (background_map, rms_map) from Background2D."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bkg = Background2D(
            image,
            box_size=box_size,
            filter_size=filter_size,
            bkg_estimator=MedianBackground(),
        )
    return bkg.background, bkg.background_rms


def _make_dilation_kernel(fwhm: float) -> Gaussian2DKernel:
    """Return a Gaussian2DKernel with the given FWHM for dilation."""
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return Gaussian2DKernel(sigma)


def _find_target_label(catalog: SourceCatalog, center_xy: tuple[float, float]) -> int:
    """Return the segmentation label of the source nearest to center_xy."""
    cx, cy = center_xy
    min_dist = np.inf
    target_label = -1
    for src in catalog:
        dx = float(src.xcentroid) - cx
        dy = float(src.ycentroid) - cy
        dist = np.hypot(dx, dy)
        if dist < min_dist:
            min_dist = dist
            target_label = int(src.label)
    return target_label


def _detect_field_contaminants(
    image: np.ndarray,
    *,
    center_xy: tuple[float, float],
    box_size: int,
    filter_size: int,
    detect_fwhm: float,
    nsigma: float,
    npixels: int,
    dilate_fwhm: float,
) -> np.ndarray:
    """Stage-1: detect and dilate field contaminants, excluding the target galaxy."""
    bkg, rms = _build_background(image, box_size=box_size, filter_size=filter_size)
    subtracted = image - bkg

    detect_kernel = _make_dilation_kernel(detect_fwhm)
    convolved = convolve(subtracted, detect_kernel, normalize_kernel=True)

    threshold = nsigma * rms
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        segmap = detect_sources(convolved, threshold, npixels=npixels)

    if segmap is None or segmap.nlabels == 0:
        return np.zeros(image.shape, dtype=bool)

    catalog = SourceCatalog(subtracted, segmap)
    target_label = _find_target_label(catalog, center_xy)

    # Build contaminant mask: all segments except the target galaxy
    contaminant_data = segmap.data.copy()
    if target_label > 0:
        contaminant_data[contaminant_data == target_label] = 0

    contaminant_bool = contaminant_data > 0

    # Dilate contaminants
    dilate_kernel = _make_dilation_kernel(dilate_fwhm)
    dilate_kernel_arr = np.asarray(dilate_kernel)
    dilated = convolve(
        contaminant_bool.astype(float), dilate_kernel_arr, normalize_kernel=True,
    )
    mask = dilated > 0.01  # threshold at 1% of kernel sum

    return mask.astype(bool)


def _detect_on_galaxy(
    image: np.ndarray,
    *,
    stage1_mask: np.ndarray,
    center_xy: tuple[float, float],
    box_size: int,
    filter_size: int,
    nsigma: float,
    npixels: int,
    dilate_fwhm: float,
) -> np.ndarray:
    """Stage-2: detect compact on-galaxy sources by subtracting a smooth envelope.

    Uses a small-box Background2D to model the smooth galaxy profile, then
    detects residual compact sources.  The nucleus (source nearest to the
    galaxy centre) is excluded from the mask.
    """
    # Mask already-known bad pixels when fitting the envelope
    fill_value = float(np.nanmedian(image))
    masked_image = image.copy()
    masked_image[stage1_mask] = fill_value

    local_bkg, local_rms = _build_background(
        masked_image, box_size=box_size, filter_size=filter_size,
    )
    residual = image - local_bkg

    threshold = nsigma * local_rms
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        segmap = detect_sources(residual, threshold, npixels=npixels)

    if segmap is None or segmap.nlabels == 0:
        return np.zeros(image.shape, dtype=bool)

    catalog = SourceCatalog(residual, segmap)
    nucleus_label = _find_target_label(catalog, center_xy)

    compact_data = segmap.data.copy()
    if nucleus_label > 0:
        compact_data[compact_data == nucleus_label] = 0

    compact_bool = compact_data > 0

    # Dilate compact-source masks
    dilate_kernel = _make_dilation_kernel(dilate_fwhm)
    dilate_kernel_arr = np.asarray(dilate_kernel)
    dilated = convolve(
        compact_bool.astype(float), dilate_kernel_arr, normalize_kernel=True,
    )
    mask = dilated > 0.01

    return mask.astype(bool)
