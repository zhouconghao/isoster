#!/usr/bin/env python3
"""Extract HSC coadd images from HDF files into per-galaxy FITS files.

Adapted from ``examples/example_hsc_edgecases/extract_hdf_to_fits.py`` for
the three real-galaxy edge cases in ``examples/example_hsc_edge_real/data``.

Key differences from the original edge-cases extractor:

1. The target galaxy is NOT guaranteed to be at the frame center. For each
   galaxy we locate the target footprint ``ITER & ~UNSHARPED`` connected
   component closest to the image center, take its centroid as the target
   ``(x0, y0)``, and write both values into the FITS headers so downstream
   scripts do not have to recompute them.

2. A "default" combined mask is still written (using the same recipe as the
   original extractor) for baseline comparison. Custom per-galaxy masks are
   produced by a separate script (``build_custom_masks.py``) and are NOT
   generated here.

3. We do not hardcode per-galaxy BRIGHT exclusions; the default mask is just
   a neutral baseline. Custom handling is deferred to the custom-mask script.

Inputs
------
HDF files with structure::

    ori_data/HSC-G/{DATA, VAR, PSF, MASK}
    ori_data/HSC-R/{DATA, VAR, PSF, MASK}
    ori_data/HSC-I/{DATA, VAR, PSF, MASK}
    mask/MASK_BIT                             # band-independent uint8 bitplane

Outputs
-------
Per-galaxy subdirectory under ``OUTPUT_DIR`` with the following FITS files:

    {obj_id}_{band}_image.fits     # science image (float32)
    {obj_id}_{band}_variance.fits  # variance map  (float32)
    {obj_id}_{band}_psf.fits       # PSF image     (float64)
    {obj_id}_{band}_mask_default.fits  # default combined mask (uint8)
    {obj_id}_mask_layers.fits      # multi-extension FITS of individual bit layers
    {obj_id}_target_info.txt       # plain-text summary of derived target center

Usage
-----
    uv run python examples/example_hsc_edge_real/extract_hdf_to_fits.py
    uv run python examples/example_hsc_edge_real/extract_hdf_to_fits.py --hdf-dir /path/to/hdfs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits
from scipy import ndimage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_HDF_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"

BANDS = ["HSC-G", "HSC-R", "HSC-I"]

# Short descriptions for the three real edge-case galaxies.
GALAXY_INFO = {
    "37498869835124888": "very extended target, large ITER footprint",
    "42177291811318246": "bright star + extended neighbor",
    "42310032070569600": "bright star halo overlapping outskirts",
}

# Mask bit definitions (same semantics as the original edge-case extractor)
MASK_BIT_HSC = 1
MASK_BIT_DET = 2
MASK_BIT_BADPIX = 4
MASK_BIT_BRIGHT = 8
MASK_BIT_EXTREME = 16
MASK_BIT_ITER = 32
MASK_BIT_UNSHARPED = 64

MASK_LAYERS = {
    "HSC": MASK_BIT_HSC,
    "DET": MASK_BIT_DET,
    "BADPIX": MASK_BIT_BADPIX,
    "BRIGHT": MASK_BIT_BRIGHT,
    "EXTREME": MASK_BIT_EXTREME,
    "ITER": MASK_BIT_ITER,
    "UNSHARPED": MASK_BIT_UNSHARPED,
}


# ---------------------------------------------------------------------------
# Target-center detection
# ---------------------------------------------------------------------------
def find_target_center(mask_bit, image, search_halfwidth=200, smooth_sigma=5.0):
    """Locate the target galaxy center from the reference image.

    These HDF cutouts are produced so that the target galaxy sits near the
    frame center, so the target is the brightest source within a box of
    ``2*search_halfwidth`` pixels around the geometric middle. We smooth the
    reference image with a Gaussian kernel and pick the brightest pixel in
    that search box. This works even in crowded cluster fields where the
    ``ITER & ~UNSHARPED`` footprint merges multiple galaxies into one blob
    (using the largest-component centroid in that case lands far from the
    real BCG).

    The footprint is still parsed to report diagnostic counts, but not used
    for centering.

    Args:
        mask_bit: 2D uint8 array of packed mask bits from the HDF file.
        image: 2D reference image used to find the brightness peak
            (typically the i-band science image).
        search_halfwidth: Half-size of the central search box in pixels.
        smooth_sigma: Gaussian smoothing sigma before peak detection.

    Returns:
        Dict with keys:
            x0, y0           : int peak pixel (0-indexed)
            peak_smoothed    : float smoothed flux at the peak
            peak_raw         : float raw flux at the peak
            target_fp_npix   : pixel count of the ITER & ~UNSHARPED footprint
            n_components     : total number of footprint components
    """
    h, w = mask_bit.shape
    cx_img, cy_img = w // 2, h // 2

    x0 = max(cx_img - search_halfwidth, 0)
    x1 = min(cx_img + search_halfwidth, w)
    y0 = max(cy_img - search_halfwidth, 0)
    y1 = min(cy_img + search_halfwidth, h)

    sub = image[y0:y1, x0:x1]
    sm = ndimage.gaussian_filter(sub.astype(np.float64), sigma=smooth_sigma)
    py_sub, px_sub = np.unravel_index(int(np.argmax(sm)), sm.shape)
    peak_smoothed = float(sm[py_sub, px_sub])
    peak_y = int(y0 + py_sub)
    peak_x = int(x0 + px_sub)
    peak_raw = float(image[peak_y, peak_x])

    iter_m = (mask_bit & MASK_BIT_ITER) > 0
    uns = (mask_bit & MASK_BIT_UNSHARPED) > 0
    target_fp = iter_m & ~uns
    labels, n_components = ndimage.label(target_fp)

    return {
        "x0": float(peak_x),
        "y0": float(peak_y),
        "peak_smoothed": peak_smoothed,
        "peak_raw": peak_raw,
        "target_fp_npix": int(target_fp.sum()),
        "n_components": int(n_components),
    }


# ---------------------------------------------------------------------------
# Default combined mask
# ---------------------------------------------------------------------------
def build_default_mask(mask_bit, band_mask, target_x0, target_y0,
                       central_clear_radius=50):
    """Build the baseline combined object mask.

    Mirrors the recipe from ``example_hsc_edgecases/extract_hdf_to_fits.py``
    but with a target-centric central-clearing zone rather than assuming
    the target sits at the image center.

    Args:
        mask_bit: Band-independent uint8 bitplane (shape ``(h, w)``).
        band_mask: Per-band HSC pipeline MASK (int), used for its DETECTED
            flag (bit 5 = 32) to better cover neighbor outskirts.
        target_x0, target_y0: Target center in pixel coordinates.
        central_clear_radius: Radius in pixels around the target center
            within which only BADPIX is preserved.

    Returns:
        Boolean mask array (``True`` = masked/bad pixel).
    """
    iter_mask = (mask_bit & MASK_BIT_ITER) > 0
    neighbors = (mask_bit & MASK_BIT_UNSHARPED) > 0
    target_footprint = iter_mask & ~neighbors

    if band_mask is not None:
        hsc_detected = (band_mask & 32) > 0
        neighbors = neighbors | hsc_detected

    badpix = (mask_bit & MASK_BIT_BADPIX) > 0
    extreme = (mask_bit & MASK_BIT_EXTREME) > 0
    bright = (mask_bit & MASK_BIT_BRIGHT) > 0

    extreme_outside_target = extreme & ~target_footprint

    combined = neighbors | badpix | extreme_outside_target | bright

    if central_clear_radius > 0:
        h, w = mask_bit.shape
        yy, xx = np.ogrid[:h, :w]
        central_zone = ((xx - target_x0) ** 2 + (yy - target_y0) ** 2) \
            <= central_clear_radius ** 2
        combined[central_zone] = badpix[central_zone]

    return combined


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def extract_one_galaxy(hdf_path, output_dir, bands=None):
    """Extract all bands and masks from one HDF file into FITS files."""
    if bands is None:
        bands = BANDS

    obj_id = hdf_path.stem
    galaxy_dir = output_dir / obj_id
    galaxy_dir.mkdir(parents=True, exist_ok=True)

    desc = GALAXY_INFO.get(obj_id, "unknown")
    stats = {"obj_id": obj_id, "description": desc, "bands": {}}

    with h5py.File(hdf_path, "r") as f:
        mask_bit = f["mask/MASK_BIT"][:]
        stats["image_shape"] = list(mask_bit.shape)

        # Load the i-band image first so we can use it as the target-center
        # reference. Fall back to whatever band is available if HSC-I missing.
        ref_band = "HSC-I" if "ori_data/HSC-I/DATA" in f else bands[0]
        ref_image = f[f"ori_data/{ref_band}/DATA"][:]

        target = find_target_center(mask_bit, ref_image)
        stats["target"] = target

        # --- Write mask layers (multi-extension FITS) ---
        hdu_list = [fits.PrimaryHDU()]
        hdu_list[0].header["OBJECT"] = obj_id
        hdu_list[0].header["DESCRIP"] = desc
        hdu_list[0].header["X_OBJ"] = (target["x0"], "target galaxy x center (0-indexed)")
        hdu_list[0].header["Y_OBJ"] = (target["y0"], "target galaxy y center (0-indexed)")
        hdu_list[0].header["PEAK_SM"] = (target["peak_smoothed"], "smoothed flux at target peak")
        hdu_list[0].header["PEAK_RAW"] = (target["peak_raw"], "raw flux at target peak")
        hdu_list[0].header["N_FP_PIX"] = (target["target_fp_npix"], "pixels in ITER & ~UNSHARPED")
        hdu_list[0].header["N_FP_CMP"] = (target["n_components"], "target-footprint component count")
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
            hdu.header["X_OBJ"] = (target["x0"], "target galaxy x center (0-indexed)")
            hdu.header["Y_OBJ"] = (target["y0"], "target galaxy y center (0-indexed)")
            hdu.writeto(galaxy_dir / f"{obj_id}_{band_safe}_image.fits", overwrite=True)
            band_stats["image_min"] = round(float(np.nanmin(image)), 4)
            band_stats["image_max"] = round(float(np.nanmax(image)), 4)
            ty, tx = int(round(target["y0"])), int(round(target["x0"]))
            band_stats["image_at_target"] = round(float(image[ty, tx]), 4)

            # Variance
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

            # Default combined mask (per-band, uses per-band HSC MASK too)
            band_hsc_mask = f[f"{band_key}/MASK"][:]
            default_mask = build_default_mask(
                mask_bit,
                band_mask=band_hsc_mask,
                target_x0=target["x0"],
                target_y0=target["y0"],
            )
            hdu = fits.PrimaryHDU(data=default_mask.astype(np.uint8))
            hdu.header["OBJECT"] = obj_id
            hdu.header["FILTER"] = band
            hdu.header["MASKTYPE"] = "default baseline mask"
            hdu.header["DESCRIP"] = desc
            hdu.header["X_OBJ"] = (target["x0"], "target galaxy x center (0-indexed)")
            hdu.header["Y_OBJ"] = (target["y0"], "target galaxy y center (0-indexed)")
            hdu.header["COMMENT"] = "True (1) = masked pixel (default recipe)."
            hdu.writeto(
                galaxy_dir / f"{obj_id}_{band_safe}_mask_default.fits",
                overwrite=True,
            )
            band_stats["default_mask_pct"] = round(
                float(default_mask.sum()) / default_mask.size * 100, 2
            )

            stats["bands"][band] = band_stats

    # Plain-text target summary per galaxy (handy for quick grep/review).
    info_path = galaxy_dir / f"{obj_id}_target_info.txt"
    with info_path.open("w") as fp:
        fp.write(f"obj_id         : {obj_id}\n")
        fp.write(f"description    : {desc}\n")
        fp.write(f"image_shape    : {stats['image_shape']}\n")
        fp.write(f"target_x0      : {target['x0']:.3f}\n")
        fp.write(f"target_y0      : {target['y0']:.3f}\n")
        fp.write(f"peak_smoothed  : {target['peak_smoothed']:.3f}\n")
        fp.write(f"peak_raw       : {target['peak_raw']:.3f}\n")
        fp.write(f"target_fp_npix : {target['target_fp_npix']}\n")
        fp.write(f"n_components   : {target['n_components']}\n")

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

    hdf_files = sorted(args.hdf_dir.glob("*.hdf"))
    if not hdf_files:
        print(f"No HDF files found in {args.hdf_dir}")
        return

    print(f"Found {len(hdf_files)} HDF files in {args.hdf_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Bands: {args.bands}")
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for hdf_path in hdf_files:
        obj_id = hdf_path.stem
        desc = GALAXY_INFO.get(obj_id, "unknown")
        print(f"Extracting {obj_id} ({desc})...")
        stats = extract_one_galaxy(hdf_path, args.output_dir, bands=args.bands)
        all_stats[obj_id] = stats

        h, w = stats["image_shape"]
        t = stats["target"]
        print(f"  Image: {h}x{w}")
        print(f"  Target center: ({t['x0']:.1f}, {t['y0']:.1f})  "
              f"peak_sm={t['peak_smoothed']:.2f}  peak_raw={t['peak_raw']:.2f}  "
              f"fp_npix={t['target_fp_npix']}  n_comp={t['n_components']}")
        for band, bs in stats["bands"].items():
            print(f"  {band}: peak_at_target={bs['image_at_target']:.2f}, "
                  f"range=[{bs['image_min']:.4f}, {bs['image_max']:.4f}], "
                  f"default_mask={bs['default_mask_pct']:.1f}%, PSF={bs['psf_shape']}")
        print()

    print(f"Done. Extracted {len(all_stats)} galaxies to {args.output_dir}")

    # Summary table
    print("\nSummary:")
    i_band = "HSC-I"
    print(f"{'ID':>20s}  {'Shape':>12s}  {'Target (x,y)':>18s}  {'I def-mask%':>11s}")
    print("-" * 68)
    for obj_id, s in sorted(all_stats.items()):
        shape_str = f"{s['image_shape'][0]}x{s['image_shape'][1]}"
        i_pct = s["bands"].get(i_band, {}).get("default_mask_pct", 0.0)
        cen_str = f"({s['target']['x0']:.0f}, {s['target']['y0']:.0f})"
        print(f"{obj_id:>20s}  {shape_str:>12s}  {cen_str:>18s}  {i_pct:10.1f}%")


if __name__ == "__main__":
    main()
