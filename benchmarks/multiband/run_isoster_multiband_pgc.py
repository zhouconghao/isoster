"""Multi-band isoster benchmark driver for LegacySurvey grz cutouts.

Loads grz images + invvar + combined mask for one SGA-2020 galaxy,
configures `IsosterConfigMB`, calls `fit_image_multiband`, writes the
Schema-1 FITS product, and renders the composite QA figure.

DO NOT run this script before the user has approved the mask QA figure
(`qa_masks.py`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isoster.multiband import (
    IsosterConfigMB,
    fit_image_multiband,
    isophote_results_mb_to_fits,
    plot_qa_summary_mb,
    subtract_outermost_sky_offset,
)

from legacysurvey_loader import (
    LEGACYSURVEY_ZP,
    asinh_softening_from_log10_match,
    load_legacysurvey_grz,
)


def run(
    galaxy_dir: Path,
    galaxy_prefix: str,
    out_dir: Path,
    sky_offset_n_outer: int = 5,
    fix_per_band_background_to_zero: bool = False,
) -> None:
    cutout = load_legacysurvey_grz(galaxy_dir, galaxy_prefix)
    bands = cutout.bands
    maxsma = float(min(cutout.shape) * 0.45)

    config = IsosterConfigMB(
        bands=bands,
        reference_band="r",
        harmonic_combination="joint",
        band_weights={b: 1.0 for b in bands},
        integrator="median",
        sma0=10.0,
        minsma=1.0,
        maxsma=maxsma,
        astep=0.10,
        linear_growth=False,
        debug=True,
        fix_per_band_background_to_zero=fix_per_band_background_to_zero,
    )

    result = fit_image_multiband(
        images=cutout.images,
        masks=cutout.combined_mask,
        variance_maps=cutout.variances,
        config=config,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_fixbg0" if fix_per_band_background_to_zero else ""
    fits_path = out_dir / f"{galaxy_prefix}_multiband{suffix}_isophotes.fits"
    isophote_results_mb_to_fits(result, str(fits_path))

    softening = {
        b: asinh_softening_from_log10_match(
            cutout.pixel_scale_arcsec, LEGACYSURVEY_ZP, bright_mu=22.0
        )
        for b in bands
    }

    qa_raw = out_dir / f"{galaxy_prefix}_multiband{suffix}_qa.png"
    plot_qa_summary_mb(
        result=result,
        images=cutout.images,
        bands=bands,
        sb_zeropoint=LEGACYSURVEY_ZP,
        pixel_scale_arcsec=cutout.pixel_scale_arcsec,
        softening_per_band=softening,
        object_mask=cutout.combined_mask,
        output_path=str(qa_raw),
    )

    result_sky, sky_offsets = subtract_outermost_sky_offset(
        result, n_outer=sky_offset_n_outer
    )
    qa_sky = out_dir / f"{galaxy_prefix}_multiband{suffix}_qa_sky_corrected.png"
    plot_qa_summary_mb(
        result=result_sky,
        images=[img - sky_offsets[b] for img, b in zip(cutout.images, bands)],
        bands=bands,
        sb_zeropoint=LEGACYSURVEY_ZP,
        pixel_scale_arcsec=cutout.pixel_scale_arcsec,
        softening_per_band=softening,
        object_mask=cutout.combined_mask,
        output_path=str(qa_sky),
    )

    print(f"Wrote {fits_path}")
    print(f"Wrote {qa_raw}")
    print(f"Wrote {qa_sky}")
    print(f"Sky offsets per band: {sky_offsets}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--galaxy-dir",
        default=Path("/Volumes/galaxy/isophote/sga2020/data/demo/PGC006669"),
        type=Path,
    )
    parser.add_argument("--galaxy-prefix", default="PGC006669-largegalaxy")
    parser.add_argument(
        "--out-dir",
        default=Path("outputs/benchmark_multiband/PGC006669"),
        type=Path,
    )
    parser.add_argument(
        "--fix-per-band-background-to-zero",
        action="store_true",
        help="Drop the per-band intercept columns from the joint solver "
        "(D11 backport). intens_<b> becomes the band's ring mean.",
    )
    args = parser.parse_args()
    run(
        args.galaxy_dir, args.galaxy_prefix, args.out_dir,
        fix_per_band_background_to_zero=args.fix_per_band_background_to_zero,
    )


if __name__ == "__main__":
    main()
