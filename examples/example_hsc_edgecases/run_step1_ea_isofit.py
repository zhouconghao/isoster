#!/usr/bin/env python3
"""Step 1 EA+ISOFIT run: eccentric anomaly sampling with simultaneous harmonics.

Uses the best sigma-clipping setup (asymmetric: sclip_low=3.0, sclip_high=2.0,
nclip=3) combined with eccentric anomaly sampling and simultaneous harmonic
fitting up to order 4.

Usage
-----
    uv run python examples/example_hsc_edgecases/run_step1_ea_isofit.py
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.rcParams["text.usetex"] = False

import numpy as np
from astropy.io import fits

from isoster import fit_image
from isoster.config import IsosterConfig
from isoster.model import build_isoster_model
from isoster.plotting import plot_qa_summary
from isoster.utils import isophote_results_to_fits

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "outputs" / "example_hsc_edgecases" / "step1_ea_isofit"
)

GALAXIES = [
    ("10140088", "clear case"),
    ("10140002", "nearby bright star"),
    ("10140006", "nearby large galaxy"),
    ("10140009", "blending bright star"),
    ("10140056", "artifact"),
    ("10140093", "small blending source"),
]

BAND = "HSC_I"
SB_ZEROPOINT = 27.0

CONFIG_KWARGS = dict(
    sma0=10.0,
    minsma=0.0,
    astep=0.1,
    linear_growth=False,
    eps=0.2,
    pa=0.0,
    fix_center=False,
    fix_pa=False,
    fix_eps=False,
    # Asymmetric sigma clipping (best from sclip experiments)
    sclip=3.0,
    nclip=3,
    sclip_low=3.0,
    sclip_high=2.0,
    # Eccentric anomaly sampling
    use_eccentric_anomaly=True,
    # Simultaneous harmonic fitting (ISOFIT)
    simultaneous_harmonics=True,
    harmonic_orders=[3, 4],
    compute_deviations=True,
    full_photometry=True,
    compute_cog=True,
    debug=True,
    max_retry_first_isophote=5,
)


def load_galaxy_data(obj_id):
    """Load i-band image, variance, and mask for one galaxy."""
    galaxy_dir = DATA_DIR / obj_id
    image = fits.getdata(galaxy_dir / f"{obj_id}_{BAND}_image.fits").astype(np.float64)
    variance = fits.getdata(galaxy_dir / f"{obj_id}_{BAND}_variance.fits").astype(np.float64)
    mask = fits.getdata(galaxy_dir / f"{obj_id}_{BAND}_mask.fits").astype(bool)
    return image, variance, mask


def stop_code_summary(isophotes):
    """Return a dict of stop_code -> count."""
    counts = {}
    for iso in isophotes:
        code = iso.get("stop_code", -99)
        counts[code] = counts.get(code, 0) + 1
    return counts


def run_one_galaxy(obj_id, desc):
    """Run EA+ISOFIT fitting on one galaxy."""
    image, variance, mask = load_galaxy_data(obj_id)
    h, w = image.shape
    x0, y0 = w / 2.0, h / 2.0
    max_sma = min(h, w) / 2.0 - 10

    config = IsosterConfig(x0=x0, y0=y0, maxsma=max_sma, **CONFIG_KWARGS)

    t0 = time.perf_counter()
    results = fit_image(image, mask=mask, config=config, variance_map=variance)
    elapsed = time.perf_counter() - t0

    isophotes = results["isophotes"]
    sc_counts = stop_code_summary(isophotes)
    sc_str = "  ".join(f"{k}:{v}" for k, v in sorted(sc_counts.items()))
    fail_flag = "  *FIRST_ISO_FAIL*" if results.get("first_isophote_failure") else ""
    print(f"  {obj_id} ({desc}): {len(isophotes)} iso, {elapsed:.2f}s, {sc_str}{fail_flag}")

    # Save results
    galaxy_out = OUTPUT_DIR / obj_id
    galaxy_out.mkdir(parents=True, exist_ok=True)
    tag = "step1_ea_isofit"
    isophote_results_to_fits(results, str(galaxy_out / f"{obj_id}_{tag}_results.fits"))

    # QA figure
    model = build_isoster_model(image.shape, isophotes, use_harmonics=True)
    plot_qa_summary(
        title=f"{obj_id} — {desc} ({tag})",
        image=image,
        isoster_model=model,
        isoster_res=isophotes,
        mask=mask,
        filename=str(galaxy_out / f"{obj_id}_{tag}_qa.png"),
        relative_residual=False,
        sb_zeropoint=SB_ZEROPOINT,
    )

    return {
        "obj_id": obj_id,
        "description": desc,
        "n_isophotes": len(isophotes),
        "elapsed_s": round(elapsed, 2),
        "stop_codes": sc_counts,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 1: EA + ISOFIT + asymmetric clipping on HSC edge cases")
    print(f"  EA=True, simultaneous_harmonics=True, orders=[3,4]")
    print(f"  sclip_low=3.0, sclip_high=2.0, nclip=3")
    print(f"Output: {OUTPUT_DIR}")

    all_results = []
    for obj_id, desc in GALAXIES:
        result = run_one_galaxy(obj_id, desc)
        all_results.append(result)

    print(f"\n  {'ID':>10s}  {'Description':<25s}  {'N_iso':>5s}  {'Time':>5s}  Stop Codes")
    print(f"  {'-'*72}")
    for r in all_results:
        sc_str = "  ".join(f"{k}:{v}" for k, v in sorted(r["stop_codes"].items()))
        print(f"  {r['obj_id']:>10s}  {r['description']:<25s}  {r['n_isophotes']:5d}  "
              f"{r['elapsed_s']:4.1f}s  {sc_str}")


if __name__ == "__main__":
    main()
