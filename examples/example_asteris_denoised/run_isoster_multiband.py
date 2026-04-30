"""
Stage-1 multi-band isoster demo on the asteris denoised dataset.

Single-target demo (object 37484563299062823) — the only asteris galaxy
with all five HSC bands (G/R/I/Z/Y) available. Loads the five denoised
cutouts, the existing object mask (built by ``build_object_mask.py``),
and a per-band uniform-variance map derived from the sigma-clipped sky
RMS. Runs :func:`isoster.multiband.fit_image_multiband` and writes:

* ``<id>_multiband_isophotes.fits`` — Schema-1 multi-band result.
* ``<id>_multiband_qa.png`` — composite QA figure (decision D15).

Reference geometry from the existing i-band single-band run is loaded
from ``<id>_denoised_free_isophotes.fits`` and overlaid on the geometry
panel of the QA figure as a "in-family with reference" sanity check.

Usage::

    uv run python examples/example_asteris_denoised/run_isoster_multiband.py

Set ``ASTERIS_DATA_ROOT`` to override the default cutout location.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")  # safe headless rendering on macOS
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402

EXAMPLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLE_DIR))
from common import (  # noqa: E402
    DATA_ROOT, GALAXIES, PIXEL_SCALE_ARCSEC, SB_ZEROPOINT,
    galaxy_dir, output_dir, sky_std, uniform_variance_map,
)

from isoster.multiband import (  # noqa: E402
    IsosterConfigMB,
    fit_image_multiband,
    isophote_results_mb_to_fits,
    subtract_outermost_sky_offset,
)
from isoster.multiband.plotting_mb import plot_qa_summary_mb  # noqa: E402

DEMO_OBJ_ID = "37484563299062823"
DEMO_BANDS = ["g", "r", "i", "z", "y"]
DEMO_BAND_FOLDERS = ["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"]


def _load_denoised_band(obj_id: str, band_folder: str) -> np.ndarray:
    """Load the ``denoised.fits`` cutout from one HSC band folder."""
    gdir = galaxy_dir(obj_id, band_folder)
    path = gdir / "denoised.fits"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing denoised cutout for {obj_id}/{band_folder} at {path}. "
            f"Set ASTERIS_DATA_ROOT to a directory that contains the asteris "
            f"objs hierarchy, or symlink the data into "
            f"{DATA_ROOT}."
        )
    return fits.getdata(path).astype(np.float64)


def _load_object_mask(obj_id: str) -> np.ndarray:
    """The existing HSC-I object mask is the shared mask for all bands."""
    mpath = galaxy_dir(obj_id, "HSC-I") / "object_mask.fits"
    if not mpath.exists():
        raise FileNotFoundError(
            f"Object mask not found at {mpath}. Run "
            f"`build_object_mask.py` first to generate it."
        )
    raw = fits.getdata(mpath)
    # The mask saved by build_object_mask.py is a float weight map (1.0 = bad).
    # Convert into a boolean mask using a 0.5 threshold to match isoster's
    # convention (True = bad).
    return raw > 0.5


def _load_reference_isophotes(obj_id: str) -> List[dict]:
    """Load the existing single-band i-band free fit as a geometry overlay."""
    out_dir = output_dir(obj_id)
    ref_path = out_dir / f"{obj_id}_denoised_free_isophotes.fits"
    if not ref_path.exists():
        print(
            f"[warn] reference i-band single-band fit not found at {ref_path}; "
            f"the QA figure will skip the reference overlay. "
            f"Run `run_isoster_pair.py` first to produce it."
        )
        return []
    with fits.open(ref_path) as hdulist:
        try:
            tbl = hdulist["ISOPHOTES"]
        except KeyError:
            tbl = hdulist[1]
        rows = []
        for row in tbl.data:  # type: ignore[union-attr]
            rows.append(
                {
                    "sma": float(row["sma"]),
                    "x0": float(row["x0"]),
                    "y0": float(row["y0"]),
                    "eps": float(row["eps"]),
                    "pa": float(row["pa"]),
                    "stop_code": int(row["stop_code"]),
                    "valid": bool(row["intens"] == row["intens"]),  # NaN check
                }
            )
        return rows


def main() -> int:
    obj_id = DEMO_OBJ_ID
    print(f"=== Multi-band isoster demo: object {obj_id} ===")
    out_dir = output_dir(obj_id)

    # 1. Load the five denoised bands.
    images: List[np.ndarray] = []
    sky_rms_per_band: dict = {}
    for band, folder in zip(DEMO_BANDS, DEMO_BAND_FOLDERS):
        img = _load_denoised_band(obj_id, folder)
        sky_rms_per_band[band] = sky_std(img)
        images.append(img)
        print(
            f"  loaded {folder}: shape={img.shape}, sky_rms={sky_rms_per_band[band]:.4g}"
        )

    # 2. Shared object mask (HSC-I).
    mask = _load_object_mask(obj_id)
    print(f"  loaded object_mask: shape={mask.shape}, n_masked={int(mask.sum())}")

    # 3. Per-band uniform variance maps from the sigma-clipped sky RMS.
    variance_maps = [uniform_variance_map(im) for im in images]

    # 4. Multi-band config — start near image center, geometric astep,
    #    maxsma reaches the cutout corner so the deepest LSB tail is sampled.
    h, w = images[0].shape
    cfg = IsosterConfigMB(
        bands=DEMO_BANDS,
        reference_band="i",
        sma0=20.0,
        eps=0.2, pa=0.0,
        astep=0.1, linear_growth=False,
        maxsma=float(np.hypot(h, w)) / 2.0,
        debug=True,
        compute_deviations=True,
        nclip=2,
        max_retry_first_isophote=5,
        # Per-band weights: equal across bands by default, in line with
        # decision D12. Modify here to up-weight one band relative to others.
        band_weights=None,
    )
    print(
        f"  config: bands={cfg.bands}, reference_band={cfg.reference_band}, "
        f"harmonic_combination={cfg.harmonic_combination}, sma0={cfg.sma0}, "
        f"maxsma={cfg.maxsma:.1f}"
    )

    # 5. Run the joint multi-band fit.
    print("  running fit_image_multiband ...")
    result = fit_image_multiband(
        images=images, masks=mask, config=cfg, variance_maps=variance_maps,
    )
    n_iso = len(result["isophotes"])
    n_valid = sum(1 for iso in result["isophotes"] if iso["valid"])
    print(f"  done: {n_iso} isophotes ({n_valid} valid)")

    # 6. Save Schema-1 FITS.
    fits_path = out_dir / f"{obj_id}_multiband_isophotes.fits"
    isophote_results_mb_to_fits(result, fits_path)
    print(f"  wrote {fits_path}")

    # 7. Composite QA figure.
    softening_per_band = {b: max(s, 1e-6) for b, s in sky_rms_per_band.items()}
    qa_path = out_dir / f"{obj_id}_multiband_qa.png"
    title = f"asteris {obj_id} - multiband joint fit (5 bands)"

    # Post-process: subtract per-band outer-ring sky residual so the SB
    # profile reflects the galaxy signal rather than asymptoting to the
    # per-band I0_b plateau (decision D11 / Stage-2+ revisit). Subtract
    # the same offset from each band's image so the residual mosaic
    # remains image - model in matched units.
    n_outer_for_sky = 8
    result_sky_corr, sky_offsets = subtract_outermost_sky_offset(
        result, n_outer=n_outer_for_sky,
    )
    print(f"  sky offsets from outermost {n_outer_for_sky} rings:")
    for b in DEMO_BANDS:
        print(f"    {b}: {sky_offsets[b]:+.6g}")
    images_sky_corr = [im - sky_offsets[b] for im, b in zip(images, DEMO_BANDS)]

    fig = plot_qa_summary_mb(
        result_sky_corr, images_sky_corr,
        sb_zeropoint=SB_ZEROPOINT,
        pixel_scale_arcsec=PIXEL_SCALE_ARCSEC,
        softening_per_band=softening_per_band,
        object_mask=mask,
        output_path=qa_path,
        title=title,
    )
    plt.close(fig)
    print(f"  wrote {qa_path}")

    # 8. Optional: print a brief comparison vs the existing i-band single-band
    #    reference fit (geometry sanity check).
    ref_isophotes = _load_reference_isophotes(obj_id)
    if ref_isophotes:
        ref_eps = np.array([iso["eps"] for iso in ref_isophotes if iso["valid"]])
        ref_pa = np.array([iso["pa"] for iso in ref_isophotes if iso["valid"]])
        mb_eps = np.array(
            [float(iso["eps"]) for iso in result["isophotes"] if iso["valid"]]
        )
        mb_pa = np.array(
            [float(iso["pa"]) for iso in result["isophotes"] if iso["valid"]]
        )
        print("  geometry sanity vs i-band reference (median over valid rings):")
        print(
            f"    eps:  multi-band={np.median(mb_eps):.3f}   i-band={np.median(ref_eps):.3f}"
        )
        print(
            f"    pa:   multi-band={np.rad2deg(np.median(mb_pa)):.2f} deg   "
            f"i-band={np.rad2deg(np.median(ref_pa)):.2f} deg"
        )

    print("=== Demo complete ===")
    # Mark the GALAXIES list as used so linters do not complain.
    _ = GALAXIES
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
