#!/usr/bin/env python3
"""Extract HSC coadd images from HDF files into per-galaxy FITS files.

Reads HSC coadd HDF files produced by the HSC pipeline, extracts image,
variance, PSF, and mask layers per band, and writes them as standard FITS
files ready for isoster fitting.

The most important step is constructing the **object mask** that masks
contaminating sources (neighbors, bright stars, bad pixels) while leaving
the target galaxy unmasked. The HDF mask bitplane encodes:

    MASK_BIT_HSC       =  1  # HSC pipeline mask
    MASK_BIT_DET       =  2  # HSC detection mask
    MASK_BIT_BADPIX    =  4  # Bad pixel mask
    MASK_BIT_BRIGHT    =  8  # Gaia bright star mask
    MASK_BIT_EXTREME   = 16  # Extreme value pixel mask
    MASK_BIT_ITER      = 32  # Full detection mask (all detected objects)
    MASK_BIT_UNSHARPED = 64  # Central object exclusion mask (the target)

The object mask for isoster is:
    (ITER & ~UNSHARPED) | BRIGHT | BADPIX | EXTREME

This masks all detected objects EXCEPT the target galaxy, plus bright
stars, bad pixels, and extreme-value pixels.

Input
-----
    HDF files in ``HDF_DIR`` (one per galaxy), each containing:
    - ``ori_data/{band}/DATA``: science image
    - ``ori_data/{band}/VAR``: variance map
    - ``ori_data/{band}/PSF``: PSF image
    - ``mask/MASK_BIT``: combined bitplane mask (band-independent)

Output
------
    Per-galaxy subdirectory under ``OUTPUT_DIR`` with FITS files:
    - ``{obj_id}_{band}_image.fits``: science image
    - ``{obj_id}_{band}_variance.fits``: variance map
    - ``{obj_id}_{band}_psf.fits``: PSF image
    - ``{obj_id}_mask_combined.fits``: combined object mask (bool, True=bad)
    - ``{obj_id}_mask_layers.fits``: multi-extension FITS with individual mask layers

Usage
-----
    uv run python examples/example_hsc_edgecases/extract_hdf_to_fits.py
    uv run python examples/example_hsc_edgecases/extract_hdf_to_fits.py --hdf-dir /path/to/hdfs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_HDF_DIR = Path("/Users/shuang/Desktop/isoster_test_sample")
OUTPUT_DIR = Path(__file__).resolve().parent / "data"

BANDS = ["HSC-G", "HSC-R", "HSC-I"]

# Galaxy descriptions for FITS headers
GALAXY_INFO = {
    "10140056": "artifact",
    "10140002": "nearby bright star",
    "10140009": "blending bright star",
    "10140088": "clear case",
    "10140093": "small blending source",
    "10140006": "nearby large galaxy",
}

# Mask bit definitions
MASK_BIT_HSC = 1
MASK_BIT_DET = 2
MASK_BIT_BADPIX = 4
MASK_BIT_BRIGHT = 8
MASK_BIT_EXTREME = 16
MASK_BIT_ITER = 32
MASK_BIT_UNSHARPED = 64

# Objects where the BRIGHT star mask should be excluded from the combined mask.
# 10140009: bright star blended with galaxy core — keep unmasked for edge-case testing.
EXCLUDE_BRIGHT_OBJECTS = {"10140009"}

MASK_LAYERS = {
    "HSC": MASK_BIT_HSC,
    "DET": MASK_BIT_DET,
    "BADPIX": MASK_BIT_BADPIX,
    "BRIGHT": MASK_BIT_BRIGHT,
    "EXTREME": MASK_BIT_EXTREME,
    "ITER": MASK_BIT_ITER,
    "UNSHARPED": MASK_BIT_UNSHARPED,
}


def build_object_mask(mask_bit, band_mask=None, exclude_bright=False,
                      central_clear_radius=50):
    """Build the combined object mask for isoster from the HDF bitplane.

    Key insight on HSC mask semantics:
    - ITER (bit 32): full detection footprint — includes ALL detected objects
      including the target galaxy.
    - UNSHARPED (bit 64): detections found AFTER the central object model was
      subtracted (unsharp masking). These are neighbor/contaminant footprints.
    - Per-band MASK bit 5 (value 32): DETECTED flag from the HSC pipeline,
      which runs source detection after central object subtraction. This
      covers neighbor sources more completely than UNSHARPED alone.

    Therefore:
    - Target galaxy footprint = ITER & ~UNSHARPED (detected only in the original)
    - Neighbor footprints = UNSHARPED | per-band DETECTED

    The object mask for isoster masks:
    - Neighbor detections (UNSHARPED + per-band DETECTED)
    - Bright star halos (BRIGHT), unless exclude_bright=True
    - Bad pixels (BADPIX)
    - Extreme value pixels outside the target footprint (EXTREME & ~target),
      since the bright galaxy core is often flagged as EXTREME

    A central clearing step removes any mask within ``central_clear_radius``
    of the image center. The HSC pipeline often flags sub-structure (HII
    regions, star-forming knots) inside the target galaxy as separate
    detections; these spurious mask islands would corrupt isophote fitting.
    Only BADPIX is preserved inside the cleared zone.

    Args:
        mask_bit: 2D uint8 array of packed mask bits from the HDF file.
        band_mask: Optional 2D int array of the per-band HSC pipeline MASK.
            When provided, its DETECTED flag (bit 5) supplements the neighbor
            mask for better coverage of extended neighbor outskirts.
        exclude_bright: If True, omit the BRIGHT star mask layer. Useful for
            cases where a bright star is blended with the target core and
            the user wants to test isoster behavior on that edge case.
        central_clear_radius: Radius in pixels around the image center within
            which the mask is cleared (except BADPIX). Set to 0 to disable.

    Returns:
        Boolean mask array (True = masked/bad pixel).
    """
    iter_mask = (mask_bit & MASK_BIT_ITER) > 0
    neighbors = (mask_bit & MASK_BIT_UNSHARPED) > 0
    target_footprint = iter_mask & ~neighbors

    # Per-band DETECTED flag covers neighbor outskirts missed by UNSHARPED
    if band_mask is not None:
        hsc_detected = (band_mask & 32) > 0  # HSC pipeline DETECTED (bit 5)
        neighbors = neighbors | hsc_detected

    badpix = (mask_bit & MASK_BIT_BADPIX) > 0
    extreme = (mask_bit & MASK_BIT_EXTREME) > 0

    # Only mask EXTREME pixels outside the target galaxy footprint
    extreme_outside_target = extreme & ~target_footprint

    combined = neighbors | badpix | extreme_outside_target

    if not exclude_bright:
        bright = (mask_bit & MASK_BIT_BRIGHT) > 0
        combined = combined | bright

    # Clear mask in the central region of the target galaxy.
    # Sub-structure detections (HII regions, knots) inside the galaxy are
    # spurious for isophote fitting; only true bad pixels should survive.
    if central_clear_radius > 0:
        h, w = mask_bit.shape
        cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        central_zone = ((xx - cx) ** 2 + (yy - cy) ** 2) <= central_clear_radius ** 2
        # Preserve only BADPIX inside the central zone
        combined[central_zone] = badpix[central_zone]

    return combined


def extract_one_galaxy(hdf_path, output_dir, bands=None):
    """Extract all bands and masks from one HDF file into FITS files.

    Args:
        hdf_path: Path to the HDF file.
        output_dir: Directory to write output FITS files.
        bands: List of band names to extract (default: BANDS).

    Returns:
        dict with extraction summary statistics.
    """
    if bands is None:
        bands = BANDS

    obj_id = hdf_path.stem
    galaxy_dir = output_dir / obj_id
    galaxy_dir.mkdir(parents=True, exist_ok=True)

    desc = GALAXY_INFO.get(obj_id, "unknown")
    stats = {"obj_id": obj_id, "description": desc, "bands": {}}

    # Per-galaxy overrides
    exclude_bright = obj_id in EXCLUDE_BRIGHT_OBJECTS

    with h5py.File(hdf_path, "r") as f:
        # --- Mask bitplane (band-independent) ---
        mask_bit = f["mask/MASK_BIT"][:]
        stats["image_shape"] = list(mask_bit.shape)

        # Write individual mask layers as multi-extension FITS
        hdu_list = [fits.PrimaryHDU()]
        hdu_list[0].header["OBJECT"] = obj_id
        hdu_list[0].header["DESCRIP"] = desc
        for layer_name, bit_val in MASK_LAYERS.items():
            layer = ((mask_bit & bit_val) > 0).astype(np.uint8)
            ext = fits.ImageHDU(data=layer, name=layer_name)
            ext.header["BITVAL"] = bit_val
            ext.header["COMMENT"] = f"Mask layer: {layer_name} (bit {bit_val})"
            hdu_list.append(ext)
        fits.HDUList(hdu_list).writeto(
            galaxy_dir / f"{obj_id}_mask_layers.fits", overwrite=True
        )

        # --- Per-band data ---
        for band in bands:
            band_key = f"ori_data/{band}"
            if band_key not in f:
                print(f"  WARNING: {band_key} not found in {hdf_path.name}, skipping")
                continue

            band_safe = band.replace("-", "_")
            band_stats = {}

            # Science image
            image = f[f"{band_key}/DATA"][:]
            hdu = fits.PrimaryHDU(data=image.astype(np.float32))
            hdu.header["OBJECT"] = obj_id
            hdu.header["FILTER"] = band
            hdu.header["BUNIT"] = "counts"
            hdu.header["DESCRIP"] = desc
            hdu.writeto(galaxy_dir / f"{obj_id}_{band_safe}_image.fits", overwrite=True)
            band_stats["image_min"] = round(float(np.nanmin(image)), 4)
            band_stats["image_max"] = round(float(np.nanmax(image)), 4)
            band_stats["image_center"] = round(float(image[image.shape[0] // 2, image.shape[1] // 2]), 4)

            # Variance map
            variance = f[f"{band_key}/VAR"][:]
            hdu = fits.PrimaryHDU(data=variance.astype(np.float32))
            hdu.header["OBJECT"] = obj_id
            hdu.header["FILTER"] = band
            hdu.header["BUNIT"] = "counts^2"
            hdu.header["DESCRIP"] = desc
            hdu.writeto(galaxy_dir / f"{obj_id}_{band_safe}_variance.fits", overwrite=True)
            band_stats["var_min"] = round(float(np.nanmin(variance)), 6)
            band_stats["var_max"] = round(float(np.nanmax(variance)), 6)

            # PSF
            psf = f[f"{band_key}/PSF"][:]
            hdu = fits.PrimaryHDU(data=psf.astype(np.float64))
            hdu.header["OBJECT"] = obj_id
            hdu.header["FILTER"] = band
            hdu.header["DESCRIP"] = desc
            hdu.writeto(galaxy_dir / f"{obj_id}_{band_safe}_psf.fits", overwrite=True)
            band_stats["psf_shape"] = list(psf.shape)

            # Per-band combined mask (uses both MASK_BIT and per-band MASK)
            band_hsc_mask = f[f"{band_key}/MASK"][:]
            combined_mask = build_object_mask(
                mask_bit, band_mask=band_hsc_mask, exclude_bright=exclude_bright,
            )
            hdu = fits.PrimaryHDU(data=combined_mask.astype(np.uint8))
            hdu.header["OBJECT"] = obj_id
            hdu.header["FILTER"] = band
            hdu.header["MASKTYPE"] = "combined object mask for isoster"
            hdu.header["DESCRIP"] = desc
            hdu.header["NOBRIGHT"] = exclude_bright
            hdu.header["COMMENT"] = "True (1) = masked pixel."
            hdu.writeto(galaxy_dir / f"{obj_id}_{band_safe}_mask.fits", overwrite=True)
            mask_frac = np.sum(combined_mask) / combined_mask.size * 100
            band_stats["mask_fraction_pct"] = round(mask_frac, 1)

            stats["bands"][band] = band_stats

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract HSC coadd HDF files into per-galaxy FITS files."
    )
    parser.add_argument(
        "--hdf-dir",
        type=Path,
        default=DEFAULT_HDF_DIR,
        help="Directory containing HDF files (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for FITS files (default: %(default)s)",
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        default=BANDS,
        help="Bands to extract (default: %(default)s)",
    )
    args = parser.parse_args()

    bands = args.bands

    hdf_files = sorted(args.hdf_dir.glob("*.hdf"))
    if not hdf_files:
        print(f"No HDF files found in {args.hdf_dir}")
        return

    print(f"Found {len(hdf_files)} HDF files in {args.hdf_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Bands: {bands}")
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for hdf_path in hdf_files:
        obj_id = hdf_path.stem
        desc = GALAXY_INFO.get(obj_id, "unknown")
        print(f"Extracting {obj_id} ({desc})...")
        stats = extract_one_galaxy(hdf_path, args.output_dir, bands=bands)
        all_stats[obj_id] = stats

        h, w = stats["image_shape"]
        print(f"  Image: {h}x{w}")
        for band, bs in stats["bands"].items():
            print(f"  {band}: center={bs['image_center']:.2f}, "
                  f"range=[{bs['image_min']:.4f}, {bs['image_max']:.4f}], "
                  f"mask={bs['mask_fraction_pct']:.1f}%, PSF={bs['psf_shape']}")
        print()

    print(f"Done. Extracted {len(all_stats)} galaxies to {args.output_dir}")

    # Print summary table
    print("\nSummary:")
    i_band = "HSC-I"
    print(f"{'ID':>10s}  {'Description':<25s}  {'Shape':>12s}  {'I-Mask%':>8s}")
    print("-" * 62)
    for obj_id, s in sorted(all_stats.items()):
        shape_str = f"{s['image_shape'][0]}x{s['image_shape'][1]}"
        i_mask = s["bands"].get(i_band, {}).get("mask_fraction_pct", 0.0)
        print(f"{obj_id:>10s}  {s['description']:<25s}  {shape_str:>12s}  {i_mask:7.1f}%")


if __name__ == "__main__":
    main()
