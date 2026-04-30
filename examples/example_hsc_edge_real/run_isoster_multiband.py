"""
Multi-band joint isoster demo on the example_hsc_edge_real BCG dataset.

Single-target demo (object 37498869835124888) using all three available
bands (HSC g / r / i). For each band the demo loads the image, variance
map, and *both* available masks (``*_mask_custom.fits`` and
``*_mask_default.fits``), then picks the one with the smaller masked-
pixel fraction so we lose as little signal as possible. Per-band masks
and variance maps are then fed into :func:`fit_image_multiband`.

Outputs land under ``outputs/example_hsc_edge_real/<id>/``:

* ``<id>_multiband_isophotes.fits`` — Schema-1 multi-band result.
* ``<id>_multiband_qa.png`` — composite QA figure with sky-offset
  post-process applied.
* ``<id>_multiband_qa_no_sky.png`` — same fit, no sky-offset
  post-process (useful for comparing the residual mosaics).

Anchor (x0, y0) is read from the HSC_I custom mask header
(``X_OBJ`` / ``Y_OBJ``).

Usage::

    uv run python examples/example_hsc_edge_real/run_isoster_multiband.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402

EXAMPLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLE_DIR))
from common import (  # noqa: E402
    DATA_DIR, OUTPUT_ROOT, PIXEL_SCALE_ARCSEC, SB_ZEROPOINT, load_target_anchor,
)

from isoster.multiband import (  # noqa: E402
    IsosterConfigMB,
    fit_image_multiband,
    isophote_results_mb_to_fits,
    subtract_outermost_sky_offset,
)
from isoster.multiband.plotting_mb import plot_qa_summary_mb  # noqa: E402

DEMO_OBJ_ID = "37498869835124888"
DEMO_BANDS = ["g", "r", "i"]
DEMO_BAND_FOLDERS = ["HSC_G", "HSC_R", "HSC_I"]


def _band_dir(obj_id: str) -> Path:
    return DATA_DIR / obj_id


def _load_band(obj_id: str, band_folder: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Load image, variance, and the smaller-fraction mask for one band.

    Returns ``(image, variance, mask, mask_kind)`` where ``mask_kind`` is
    ``'custom'`` or ``'default'`` for downstream logging.
    """
    gdir = _band_dir(obj_id)
    image = fits.getdata(gdir / f"{obj_id}_{band_folder}_image.fits").astype(np.float64)
    variance = fits.getdata(gdir / f"{obj_id}_{band_folder}_variance.fits").astype(np.float64)
    m_custom = fits.getdata(gdir / f"{obj_id}_{band_folder}_mask_custom.fits").astype(bool)
    m_default = fits.getdata(gdir / f"{obj_id}_{band_folder}_mask_default.fits").astype(bool)

    n_custom = int(m_custom.sum())
    n_default = int(m_default.sum())
    if n_custom <= n_default:
        return image, variance, m_custom, "custom"
    return image, variance, m_default, "default"


def main() -> int:
    obj_id = DEMO_OBJ_ID
    print(f"=== Multi-band isoster (HSC edge-real demo): object {obj_id} ===")

    out_dir = OUTPUT_ROOT / obj_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load g / r / i: image + variance + smaller-fraction mask each.
    images: List[np.ndarray] = []
    variance_maps: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    print(f"  loading {len(DEMO_BANDS)} bands from {DATA_DIR / obj_id}:")
    for _band, folder in zip(DEMO_BANDS, DEMO_BAND_FOLDERS):
        img, var, mask, mkind = _load_band(obj_id, folder)
        n_pix = img.size
        print(
            f"    {folder}: shape={img.shape}, "
            f"mask_kind={mkind} ({mask.sum()} px, {mask.sum()/n_pix:.3f}), "
            f"sky_var_median={float(np.nanmedian(var)):.4g}"
        )
        images.append(img)
        variance_maps.append(var)
        masks.append(mask)

    # 2. Anchor from HSC_I custom-mask header.
    x0, y0 = load_target_anchor(obj_id, band="HSC_I")
    h, w = images[0].shape
    print(f"  anchor (HSC_I mask): x0={x0:.2f}, y0={y0:.2f}; cutout={(h, w)}")

    # 3. Multi-band config: equal band weights, joint mode, geometric astep.
    cfg = IsosterConfigMB(
        bands=DEMO_BANDS,
        reference_band="i",
        x0=x0, y0=y0,
        sma0=20.0,
        eps=0.2, pa=0.0,
        astep=0.1, linear_growth=False,
        maxsma=float(np.hypot(h, w)) / 2.0,
        debug=True,
        compute_deviations=True,
        nclip=2,
        max_retry_first_isophote=5,
        band_weights=None,
    )
    print(
        f"  config: bands={cfg.bands}, reference_band={cfg.reference_band}, "
        f"sma0={cfg.sma0}, maxsma={cfg.maxsma:.1f}, "
        f"harmonic_combination={cfg.harmonic_combination}"
    )

    # 4. Fit (timed).
    print("  running fit_image_multiband ...")
    t0 = time.perf_counter()
    result = fit_image_multiband(
        images=images, masks=masks, config=cfg, variance_maps=variance_maps,
    )
    fit_time = time.perf_counter() - t0
    n_iso = len(result["isophotes"])
    n_valid = sum(1 for iso in result["isophotes"] if iso["valid"])
    print(f"  done: {n_iso} isophotes ({n_valid} valid) in {fit_time:.3f} s")

    # 5. Save Schema-1 FITS.
    fits_path = out_dir / f"{obj_id}_multiband_isophotes.fits"
    isophote_results_mb_to_fits(result, fits_path)
    print(f"  wrote {fits_path}")

    # 6. Per-band asinh-SB softening = sigma_sky from variance map.
    softening_per_band: Dict[str, float] = {}
    for b, var in zip(DEMO_BANDS, variance_maps):
        sigma = float(np.sqrt(np.nanmedian(var)))
        softening_per_band[b] = max(sigma, 1e-6)

    # 7. Sky-offset post-process (median of outermost rings per band) and
    # corresponding shifted images for the residual mosaic.
    n_outer_for_sky = 8
    result_corr, sky_offsets = subtract_outermost_sky_offset(
        result, n_outer=n_outer_for_sky,
    )
    print(f"  sky offsets from outermost {n_outer_for_sky} rings:")
    for b in DEMO_BANDS:
        print(f"    {b}: {sky_offsets[b]:+.6g}")
    images_corr = [im - sky_offsets[b] for im, b in zip(images, DEMO_BANDS)]

    # For the residual mosaic mask overlay, use the union of the per-band
    # masks (so overlapping bad pixels show consistently).
    union_mask = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        union_mask |= m

    # 8. With-sky-correction QA artifact.
    qa_path = out_dir / f"{obj_id}_multiband_qa.png"
    title = f"hsc_edge_real {obj_id} - multiband joint fit (g/r/i)"
    fig = plot_qa_summary_mb(
        result_corr, images_corr,
        sb_zeropoint=SB_ZEROPOINT,
        pixel_scale_arcsec=PIXEL_SCALE_ARCSEC,
        softening_per_band=softening_per_band,
        object_mask=union_mask,
        output_path=qa_path,
        title=title,
    )
    plt.close(fig)
    print(f"  wrote {qa_path}")

    # 9. Companion no-sky-correction QA artifact.
    qa_no_sky_path = out_dir / f"{obj_id}_multiband_qa_no_sky.png"
    fig_no_sky = plot_qa_summary_mb(
        result, images,
        sb_zeropoint=SB_ZEROPOINT,
        pixel_scale_arcsec=PIXEL_SCALE_ARCSEC,
        softening_per_band=softening_per_band,
        object_mask=union_mask,
        output_path=qa_no_sky_path,
        title=title + " (no sky correction)",
    )
    plt.close(fig_no_sky)
    print(f"  wrote {qa_no_sky_path}")

    # 10. Optional sanity: median geometry over the well-fit mid-radius range.
    valid = [
        iso for iso in result["isophotes"]
        if iso["valid"] and 30.0 <= iso["sma"] <= 200.0
    ]
    if valid:
        eps_med = float(np.median([iso["eps"] for iso in valid]))
        pa_med_deg = float(np.rad2deg(np.median([iso["pa"] for iso in valid])))
        print(
            f"  geometry sanity (valid 30 <= sma <= 200): "
            f"eps={eps_med:.3f}  pa={pa_med_deg:.1f} deg"
        )

    print("=== Demo complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
