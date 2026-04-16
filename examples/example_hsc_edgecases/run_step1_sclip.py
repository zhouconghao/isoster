#!/usr/bin/env python3
"""Step 1 sigma-clipping experiments: test whether aggressive clipping reduces centroid drift.

The default isoster config uses nclip=0 (no sigma clipping). Contaminating sources
(bright stars, neighbors) inject outlier pixels along the isophote, biasing the
harmonic fit and pulling the centroid off-center.

This script tests several sigma-clipping configurations to see if clipping
can mitigate the centroid drift without degrading the fit quality.

Variants:
- sclip_mild:       sclip=3.0, nclip=3  (standard 3-sigma, 3 iterations)
- sclip_moderate:   sclip=2.5, nclip=3  (tighter threshold)
- sclip_aggressive: sclip=2.0, nclip=5  (aggressive clipping)
- sclip_asymmetric: sclip_low=2.0, sclip_high=3.0, nclip=3  (clip low outliers harder)

Usage
-----
    uv run python examples/example_hsc_edgecases/run_step1_sclip.py
    uv run python examples/example_hsc_edgecases/run_step1_sclip.py --variant sclip_moderate
"""

from __future__ import annotations

import argparse
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
OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "example_hsc_edgecases"

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
PIXEL_SCALE_ARCSEC = 0.168  # HSC coadd

# Shared base config (same as step1 baseline except sigma clipping)
BASE_CONFIG = dict(
    sma0=10.0,
    minsma=0.0,
    astep=0.1,
    linear_growth=False,
    eps=0.2,
    pa=0.0,
    fix_center=False,
    fix_pa=False,
    fix_eps=False,
    compute_deviations=True,
    full_photometry=True,
    compute_cog=True,
    debug=True,
    max_retry_first_isophote=5,
)

VARIANTS = {
    "sclip_mild": {
        "label": "3-sigma, 3 iterations",
        "config_overrides": {"sclip": 3.0, "nclip": 3},
    },
    "sclip_moderate": {
        "label": "2.5-sigma, 3 iterations",
        "config_overrides": {"sclip": 2.5, "nclip": 3},
    },
    "sclip_aggressive": {
        "label": "2.0-sigma, 5 iterations",
        "config_overrides": {"sclip": 2.0, "nclip": 5},
    },
    "sclip_asymmetric": {
        "label": "low=3.0, high=2.0, 3 iterations",
        "config_overrides": {"sclip": 3.0, "nclip": 3, "sclip_low": 3.0, "sclip_high": 2.0},
    },
}


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


def run_one_galaxy(obj_id, desc, output_dir, variant_name, variant_cfg):
    """Run isoster on one galaxy with a specific sigma-clipping variant."""
    image, variance, mask = load_galaxy_data(obj_id)
    h, w = image.shape
    x0, y0 = w / 2.0, h / 2.0
    max_sma = min(h, w) / 2.0 - 10

    cfg_kwargs = {**BASE_CONFIG, **variant_cfg["config_overrides"]}
    config = IsosterConfig(x0=x0, y0=y0, maxsma=max_sma, **cfg_kwargs)

    t0 = time.perf_counter()
    results = fit_image(image, mask=mask, config=config, variance_map=variance)
    elapsed = time.perf_counter() - t0

    isophotes = results["isophotes"]
    sc_counts = stop_code_summary(isophotes)
    sc_str = "  ".join(f"{k}:{v}" for k, v in sorted(sc_counts.items()))
    fail_flag = "  *FIRST_ISO_FAIL*" if results.get("first_isophote_failure") else ""
    print(f"  {obj_id} ({desc}): {len(isophotes)} iso, {elapsed:.2f}s, {sc_str}{fail_flag}")

    # Save results
    galaxy_out = output_dir / obj_id
    galaxy_out.mkdir(parents=True, exist_ok=True)
    tag = f"step1_{variant_name}"
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
        pixel_scale_arcsec=PIXEL_SCALE_ARCSEC,
    )

    return {
        "obj_id": obj_id,
        "description": desc,
        "n_isophotes": len(isophotes),
        "elapsed_s": round(elapsed, 2),
        "stop_codes": sc_counts,
    }


def run_variant(variant_name):
    """Run all galaxies for one sigma-clipping variant."""
    variant_cfg = VARIANTS[variant_name]
    output_dir = OUTPUT_ROOT / f"step1_{variant_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Variant: step1_{variant_name} — {variant_cfg['label']}")
    overrides = variant_cfg["config_overrides"]
    print(f"  Config:  sclip={overrides.get('sclip', 3.0)}, nclip={overrides.get('nclip', 0)}"
          + (f", sclip_low={overrides['sclip_low']}" if "sclip_low" in overrides else "")
          + (f", sclip_high={overrides['sclip_high']}" if "sclip_high" in overrides else ""))
    print(f"  Output:  {output_dir}")
    print(f"{'='*70}")

    all_results = []
    for obj_id, desc in GALAXIES:
        result = run_one_galaxy(obj_id, desc, output_dir, variant_name, variant_cfg)
        all_results.append(result)

    print(f"\n  {'ID':>10s}  {'Description':<25s}  {'N_iso':>5s}  {'Time':>5s}  Stop Codes")
    print(f"  {'-'*72}")
    for r in all_results:
        sc_str = "  ".join(f"{k}:{v}" for k, v in sorted(r["stop_codes"].items()))
        print(f"  {r['obj_id']:>10s}  {r['description']:<25s}  {r['n_isophotes']:5d}  "
              f"{r['elapsed_s']:4.1f}s  {sc_str}")
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Step 1 sigma-clipping experiments on HSC edge cases."
    )
    parser.add_argument(
        "--variant",
        choices=list(VARIANTS.keys()) + ["all"],
        default="all",
        help="Which variant(s) to run (default: all).",
    )
    args = parser.parse_args()

    variants_to_run = list(VARIANTS.keys()) if args.variant == "all" else [args.variant]

    print("Step 1: Sigma-clipping experiments on HSC edge cases")
    for variant_name in variants_to_run:
        run_variant(variant_name)


if __name__ == "__main__":
    main()
