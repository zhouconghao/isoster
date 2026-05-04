"""Loose-validity demo (D9 backport) on PGC006669 with a planted per-band
artifact.

Workflow:

1. Load PGC006669 grz cutout via ``legacysurvey_loader``.
2. Plant a 30-degree mask arc in the **r-band only** at SMA ~ 35-45 px,
   simulating a satellite trail / bleed that affects one band but not
   the others.
3. Run the multi-band fit twice on the same inputs:
   - shared-validity (default): the per-band artifact drops samples
     across **all** bands at the affected radii.
   - loose-validity: each band keeps its own surviving samples; only
     the r-band loses coverage.
4. Render the loose-validity composite QA — the figure auto-includes
   the new ``n_valid_<b>/n_attempted`` panel below the geometry block,
   which makes the per-band coverage drop visible.
5. Print a per-radius comparison table of `n_valid_<b>` so the user
   can see exactly where the two modes diverge.

Output:
- ``<prefix>_multiband_loose_isophotes.fits``
- ``<prefix>_multiband_loose_qa.png``
- ``<prefix>_multiband_loose_vs_shared_summary.txt``

DO NOT delete the planted-artifact mask without rerunning — the demo
is meaningless if both fits see identical inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from isoster.multiband import (
    IsosterConfigMB,
    fit_image_multiband,
    isophote_results_mb_to_fits,
    plot_qa_summary_mb,
)

from legacysurvey_loader import LEGACYSURVEY_ZP, asinh_softening_from_log10_match, load_legacysurvey_grz


def plant_band_arc_artifact(
    base_mask: np.ndarray,
    shape: tuple[int, int],
    sma_inner_pix: float,
    sma_outer_pix: float,
    angle_lo_rad: float,
    angle_hi_rad: float,
    x0: float | None = None,
    y0: float | None = None,
) -> np.ndarray:
    """Return a copy of base_mask with an annular-arc region added.

    The arc is defined in geometric image coordinates (radius and
    angle from the cutout center), so it does not assume any galaxy
    geometry — it is a synthetic per-band artifact, not a real one.
    """
    h, w = shape
    if x0 is None:
        x0 = (w - 1) / 2.0
    if y0 is None:
        y0 = (h - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r = np.hypot(xx - x0, yy - y0)
    phi = np.arctan2(yy - y0, xx - x0)
    arc = (r >= sma_inner_pix) & (r <= sma_outer_pix) & (phi >= angle_lo_rad) & (phi <= angle_hi_rad)
    new_mask = base_mask.copy()
    new_mask[arc] = True
    return new_mask


def run(
    galaxy_dir: Path,
    galaxy_prefix: str,
    out_dir: Path,
    arc_band: str = "r",
    sma_inner_pix: float = 35.0,
    sma_outer_pix: float = 45.0,
    arc_lo_rad: float = -0.4,
    arc_hi_rad: float = 0.4,
) -> None:
    cutout = load_legacysurvey_grz(galaxy_dir, galaxy_prefix)
    bands = cutout.bands
    if arc_band not in bands:
        raise ValueError(f"arc_band={arc_band!r} not in cutout bands {bands}")

    # Build per-band masks: every band starts from the COMBINED mask,
    # the arc band gets the planted artifact added.
    masks_per_band: list[np.ndarray] = []
    for b in bands:
        if b == arc_band:
            masks_per_band.append(
                plant_band_arc_artifact(
                    cutout.combined_mask,
                    cutout.shape,
                    sma_inner_pix,
                    sma_outer_pix,
                    arc_lo_rad,
                    arc_hi_rad,
                )
            )
        else:
            masks_per_band.append(cutout.combined_mask.copy())

    arc_extra = int(masks_per_band[bands.index(arc_band)].sum() - cutout.combined_mask.sum())
    print(f"Planted {arc_extra} extra masked pixels in band {arc_band!r}.")

    maxsma = float(min(cutout.shape) * 0.45)
    base_kwargs = dict(
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
    )

    cfg_shared = IsosterConfigMB(**base_kwargs)
    cfg_loose = IsosterConfigMB(**base_kwargs, loose_validity=True)

    print("Running shared-validity multi-band fit ...")
    res_shared = fit_image_multiband(
        cutout.images, masks_per_band, cfg_shared, variance_maps=cutout.variances,
    )
    print("Running loose-validity multi-band fit ...")
    res_loose = fit_image_multiband(
        cutout.images, masks_per_band, cfg_loose, variance_maps=cutout.variances,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fits_path = out_dir / f"{galaxy_prefix}_multiband_loose_isophotes.fits"
    isophote_results_mb_to_fits(res_loose, str(fits_path))
    print(f"Wrote {fits_path}")

    softening = {
        b: asinh_softening_from_log10_match(cutout.pixel_scale_arcsec, LEGACYSURVEY_ZP, bright_mu=22.0)
        for b in bands
    }
    qa_path = out_dir / f"{galaxy_prefix}_multiband_loose_qa.png"
    plot_qa_summary_mb(
        result=res_loose,
        images=cutout.images,
        bands=bands,
        sb_zeropoint=LEGACYSURVEY_ZP,
        pixel_scale_arcsec=cutout.pixel_scale_arcsec,
        softening_per_band=softening,
        object_mask=cutout.combined_mask,
        output_path=str(qa_path),
        title=f"PGC006669 — multi-band loose validity ({arc_band}-band arc artifact)",
    )
    print(f"Wrote {qa_path}")

    # Per-radius comparison table.
    summary_lines = [
        f"# PGC006669 loose-validity demo (arc planted in {arc_band}-band,",
        f"# SMA in [{sma_inner_pix:.1f}, {sma_outer_pix:.1f}] px,",
        f"# phi in [{arc_lo_rad:+.2f}, {arc_hi_rad:+.2f}] rad)",
        "",
        f"{'sma':>7} | {'mode':>6} | "
        + " | ".join(f"valid_{b}={'-':>3} n_valid_{b}={'-':>4}" for b in bands)
        + " | stop_code",
    ]
    isos_shared = {iso["sma"]: iso for iso in res_shared["isophotes"]}
    isos_loose = {iso["sma"]: iso for iso in res_loose["isophotes"]}
    smas_to_inspect = [
        sma for sma in sorted(isos_loose) if sma_inner_pix - 5 <= sma <= sma_outer_pix + 5
    ]
    for sma in smas_to_inspect:
        for label, isos in (("shared", isos_shared), ("loose ", isos_loose)):
            iso = isos.get(sma)
            if iso is None:
                continue
            cells = " | ".join(
                f"valid_{b}={'Y' if iso.get('valid', False) else 'N':>3} "
                f"n_valid_{b}={int(iso.get(f'n_valid_{b}', 0)):>4}"
                for b in bands
            )
            summary_lines.append(
                f"{sma:7.2f} | {label:>6} | {cells} | stop={int(iso.get('stop_code', -99))}"
            )
    summary_path = out_dir / f"{galaxy_prefix}_multiband_loose_vs_shared_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"Wrote {summary_path}")


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
    parser.add_argument("--arc-band", default="r")
    parser.add_argument("--sma-inner-pix", type=float, default=35.0)
    parser.add_argument("--sma-outer-pix", type=float, default=45.0)
    parser.add_argument("--arc-lo-rad", type=float, default=-0.4)
    parser.add_argument("--arc-hi-rad", type=float, default=0.4)
    args = parser.parse_args()
    run(
        args.galaxy_dir, args.galaxy_prefix, args.out_dir,
        arc_band=args.arc_band,
        sma_inner_pix=args.sma_inner_pix,
        sma_outer_pix=args.sma_outer_pix,
        arc_lo_rad=args.arc_lo_rad,
        arc_hi_rad=args.arc_hi_rad,
    )


if __name__ == "__main__":
    main()
