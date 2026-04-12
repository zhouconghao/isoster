#!/usr/bin/env python3
"""Step 1: Run isoster on HSC edge-case galaxies with free geometry.

Fits isophotes to all 6 galaxies in the HSC-I band with completely free
geometry (center, ellipticity, and PA all unconstrained).

Run variants (selected via ``--variant``):
- ``baseline``: full run with variance map, mask, and harmonics (default)
- ``no_variance``: same but without the variance map (OLS instead of WLS)
- ``no_mask``: same but without the object mask
- ``no_harmonics``: same but with compute_deviations=False

Each variant writes results and QA figures under
``outputs/example_hsc_edgecases/step1_{variant}/{obj_id}/``.

Usage
-----
    # Run all variants
    uv run python examples/example_hsc_edgecases/run_step1_free_fit.py --variant all

    # Run a single variant
    uv run python examples/example_hsc_edgecases/run_step1_free_fit.py --variant baseline
    uv run python examples/example_hsc_edgecases/run_step1_free_fit.py --variant no_variance
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BAND = "HSC_I"
SB_ZEROPOINT = 27.0

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

# Per-variant overrides and fitting options
VARIANTS = {
    "baseline": {
        "label": "baseline (WLS + mask + harmonics)",
        "config_overrides": {},
        "use_variance": True,
        "use_mask": True,
    },
    "no_variance": {
        "label": "no variance map (OLS)",
        "config_overrides": {},
        "use_variance": False,
        "use_mask": True,
    },
    "no_mask": {
        "label": "no object mask",
        "config_overrides": {},
        "use_variance": True,
        "use_mask": False,
    },
    "no_harmonics": {
        "label": "no higher-order harmonics",
        "config_overrides": {"compute_deviations": False},
        "use_variance": True,
        "use_mask": True,
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
    """Return a dict of stop_code -> count from the isophote list."""
    counts = {}
    for iso in isophotes:
        code = iso.get("stop_code", -99)
        counts[code] = counts.get(code, 0) + 1
    return counts


def run_one_galaxy(obj_id, desc, output_dir, variant_name, variant_cfg):
    """Run isoster on one galaxy with a specific variant and save results + QA.

    Args:
        obj_id: Galaxy ID string.
        desc: Short description of the edge case.
        output_dir: Directory to write outputs.
        variant_name: Name of the run variant (e.g., "baseline").
        variant_cfg: Dict with config_overrides, use_variance, use_mask.

    Returns:
        dict with timing, stop code summary, and isophote count.
    """
    image, variance, mask = load_galaxy_data(obj_id)
    h, w = image.shape
    x0, y0 = w / 2.0, h / 2.0
    max_sma = min(h, w) / 2.0 - 10

    # Merge base config with variant overrides
    cfg_kwargs = {**BASE_CONFIG, **variant_cfg["config_overrides"]}
    config = IsosterConfig(x0=x0, y0=y0, maxsma=max_sma, **cfg_kwargs)

    # Select inputs based on variant
    fit_variance = variance if variant_cfg["use_variance"] else None
    fit_mask = mask if variant_cfg["use_mask"] else None

    mask_info = f"{np.sum(mask)/mask.size*100:.1f}%" if variant_cfg["use_mask"] else "disabled"
    var_info = "WLS" if variant_cfg["use_variance"] else "OLS"
    print(f"  {obj_id} ({desc}): {h}x{w}, mask={mask_info}, {var_info}")

    # Run fitting
    t0 = time.perf_counter()
    results = fit_image(image, mask=fit_mask, config=config, variance_map=fit_variance)
    elapsed = time.perf_counter() - t0

    isophotes = results["isophotes"]
    n_iso = len(isophotes)
    sc_counts = stop_code_summary(isophotes)

    sc_str = "  ".join(f"{k}:{v}" for k, v in sorted(sc_counts.items()))
    fail_flag = "  *FIRST_ISO_FAIL*" if results.get("first_isophote_failure") else ""
    print(f"    {n_iso} iso, {elapsed:.2f}s, stop_codes: {sc_str}{fail_flag}")

    # Save results
    galaxy_out = output_dir / obj_id
    galaxy_out.mkdir(parents=True, exist_ok=True)

    tag = f"step1_{variant_name}"
    fits_path = galaxy_out / f"{obj_id}_{tag}_results.fits"
    isophote_results_to_fits(results, str(fits_path))

    # Build model (use harmonics only if they were computed)
    use_harmonics = cfg_kwargs.get("compute_deviations", True)
    model = build_isoster_model(image.shape, isophotes, use_harmonics=use_harmonics)

    # Generate QA figure — always show the mask overlay for context
    qa_path = galaxy_out / f"{obj_id}_{tag}_qa.png"
    plot_qa_summary(
        title=f"{obj_id} — {desc} ({tag})",
        image=image,
        isoster_model=model,
        isoster_res=isophotes,
        mask=mask,  # always show mask in QA even if not used in fitting
        filename=str(qa_path),
        relative_residual=False,
        sb_zeropoint=SB_ZEROPOINT,
    )

    return {
        "obj_id": obj_id,
        "description": desc,
        "n_isophotes": n_iso,
        "elapsed_s": round(elapsed, 2),
        "stop_codes": sc_counts,
        "first_isophote_failure": results.get("first_isophote_failure", False),
    }


def run_variant(variant_name):
    """Run all galaxies for one variant."""
    variant_cfg = VARIANTS[variant_name]
    output_dir = OUTPUT_ROOT / f"step1_{variant_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Variant: step1_{variant_name} — {variant_cfg['label']}")
    print(f"  Output:  {output_dir}")
    print(f"{'='*70}")

    all_results = []
    for obj_id, desc in GALAXIES:
        result = run_one_galaxy(obj_id, desc, output_dir, variant_name, variant_cfg)
        all_results.append(result)

    # Summary table
    print(f"\n  {'ID':>10s}  {'Description':<25s}  {'N_iso':>5s}  {'Time':>5s}  Stop Codes")
    print(f"  {'-'*72}")
    for r in all_results:
        sc_str = "  ".join(f"{k}:{v}" for k, v in sorted(r["stop_codes"].items()))
        flag = " *FAIL*" if r["first_isophote_failure"] else ""
        print(f"  {r['obj_id']:>10s}  {r['description']:<25s}  {r['n_isophotes']:5d}  "
              f"{r['elapsed_s']:4.1f}s  {sc_str}{flag}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: free-geometry isophote fitting on HSC edge cases."
    )
    parser.add_argument(
        "--variant",
        choices=list(VARIANTS.keys()) + ["all"],
        default="baseline",
        help="Which variant(s) to run (default: baseline).",
    )
    args = parser.parse_args()

    variants_to_run = list(VARIANTS.keys()) if args.variant == "all" else [args.variant]

    print("Step 1: Free-geometry isophote fitting on HSC edge cases")
    for variant_name in variants_to_run:
        run_variant(variant_name)


if __name__ == "__main__":
    main()
