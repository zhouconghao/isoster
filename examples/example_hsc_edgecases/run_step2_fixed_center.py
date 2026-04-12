#!/usr/bin/env python3
"""Step 2: Fixed-center isophote fitting with free PA and ellipticity.

Estimates the galaxy center from step1 results using four strategies,
then runs isoster with fix_center=True. Uses mask, variance, and
aggressive asymmetric sigma clipping.

Center estimation strategies (from step1_sclip_asymmetric results):
- inner_5:        median of inner 5 stop=0 isophotes
- inner_10:       median of inner 10 stop=0 isophotes
- intens_weighted: intensity-weighted mean of all stop=0 isophotes
- clipped_median:  2-sigma clipped median of all stop=0 isophotes

Usage
-----
    uv run python examples/example_hsc_edgecases/run_step2_fixed_center.py
    uv run python examples/example_hsc_edgecases/run_step2_fixed_center.py --strategy inner_5
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
from isoster.utils import isophote_results_to_fits, isophote_results_from_fits

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "example_hsc_edgecases"
STEP1_DIR = OUTPUT_ROOT / "step1_sclip_asymmetric"

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

# Step2 config: fixed center, free PA and eps, aggressive asymmetric clipping
BASE_CONFIG = dict(
    sma0=10.0,
    minsma=0.0,
    astep=0.1,
    linear_growth=False,
    eps=0.2,
    pa=0.0,
    fix_center=True,
    fix_pa=False,
    fix_eps=False,
    # Aggressive asymmetric sigma clipping
    sclip=3.0,
    nclip=5,
    sclip_low=3.0,
    sclip_high=1.5,
    compute_deviations=True,
    full_photometry=True,
    compute_cog=True,
    debug=True,
    max_retry_first_isophote=5,
)

STRATEGIES = ["inner_5", "inner_10", "intens_weighted", "clipped_median"]


def estimate_center(step1_isophotes, strategy):
    """Estimate galaxy center from step1 isophote results.

    Args:
        step1_isophotes: List of isophote dicts from step1.
        strategy: One of 'inner_5', 'inner_10', 'intens_weighted', 'clipped_median'.

    Returns:
        Tuple (x0, y0) of estimated center coordinates.
    """
    good = [i for i in step1_isophotes if i["sma"] > 0 and i["stop_code"] == 0]
    if len(good) == 0:
        # Fallback: use all isophotes with sma > 0
        good = [i for i in step1_isophotes if i["sma"] > 0]
    if len(good) == 0:
        raise ValueError("No usable isophotes in step1 results")

    x0s = np.array([i["x0"] for i in good])
    y0s = np.array([i["y0"] for i in good])

    if strategy == "inner_5":
        n = min(5, len(good))
        return float(np.median(x0s[:n])), float(np.median(y0s[:n]))

    elif strategy == "inner_10":
        n = min(10, len(good))
        return float(np.median(x0s[:n])), float(np.median(y0s[:n]))

    elif strategy == "intens_weighted":
        intens = np.array([abs(i["intens"]) for i in good])
        w = intens / np.sum(intens)
        return float(np.sum(w * x0s)), float(np.sum(w * y0s))

    elif strategy == "clipped_median":
        med_x, med_y = np.median(x0s), np.median(y0s)
        dist = np.sqrt((x0s - med_x) ** 2 + (y0s - med_y) ** 2)
        sig = np.std(dist)
        keep = dist < 2 * sig if sig > 0 else np.ones(len(dist), dtype=bool)
        return float(np.median(x0s[keep])), float(np.median(y0s[keep]))

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


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


def run_one_galaxy(obj_id, desc, strategy, output_dir):
    """Run step2 fixed-center fitting on one galaxy."""
    # Load step1 results to estimate center
    step1_path = STEP1_DIR / obj_id / f"{obj_id}_step1_sclip_asymmetric_results.fits"
    step1_res = isophote_results_from_fits(str(step1_path))
    x0_est, y0_est = estimate_center(step1_res["isophotes"], strategy)

    image, variance, mask = load_galaxy_data(obj_id)
    h, w = image.shape
    max_sma = min(h, w) / 2.0 - 10

    config = IsosterConfig(x0=x0_est, y0=y0_est, maxsma=max_sma, **BASE_CONFIG)

    t0 = time.perf_counter()
    results = fit_image(image, mask=mask, config=config, variance_map=variance)
    elapsed = time.perf_counter() - t0

    isophotes = results["isophotes"]
    sc_counts = stop_code_summary(isophotes)
    sc_str = "  ".join(f"{k}:{v}" for k, v in sorted(sc_counts.items()))
    print(f"  {obj_id} ({desc}): center=({x0_est:.2f}, {y0_est:.2f}), "
          f"{len(isophotes)} iso, {elapsed:.2f}s, {sc_str}")

    # Save results
    galaxy_out = output_dir / obj_id
    galaxy_out.mkdir(parents=True, exist_ok=True)
    tag = f"step2_{strategy}"
    isophote_results_to_fits(results, str(galaxy_out / f"{obj_id}_{tag}_results.fits"))

    # QA figure
    model = build_isoster_model(image.shape, isophotes, use_harmonics=True)
    plot_qa_summary(
        title=f"{obj_id} — {desc} (step2: {strategy})",
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
        "strategy": strategy,
        "x0": x0_est,
        "y0": y0_est,
        "n_isophotes": len(isophotes),
        "elapsed_s": round(elapsed, 2),
        "stop_codes": sc_counts,
    }


def run_strategy(strategy):
    """Run all galaxies for one center estimation strategy."""
    output_dir = OUTPUT_ROOT / f"step2_{strategy}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Step 2: fixed center — strategy={strategy}")
    print(f"  fix_center=True, sclip_high=1.5, nclip=5")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}")

    all_results = []
    for obj_id, desc in GALAXIES:
        result = run_one_galaxy(obj_id, desc, strategy, output_dir)
        all_results.append(result)

    print(f"\n  {'ID':>10s}  {'Center':>18s}  {'N_iso':>5s}  {'Time':>5s}  Stop Codes")
    print(f"  {'-'*72}")
    for r in all_results:
        sc_str = "  ".join(f"{k}:{v}" for k, v in sorted(r["stop_codes"].items()))
        cen = f"({r['x0']:.2f}, {r['y0']:.2f})"
        print(f"  {r['obj_id']:>10s}  {cen:>18s}  {r['n_isophotes']:5d}  "
              f"{r['elapsed_s']:4.1f}s  {sc_str}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: fixed-center isophote fitting on HSC edge cases."
    )
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES + ["all"],
        default="all",
        help="Center estimation strategy (default: all).",
    )
    args = parser.parse_args()

    strategies = STRATEGIES if args.strategy == "all" else [args.strategy]

    print("Step 2: Fixed-center isophote fitting on HSC edge cases")
    print(f"  Center estimated from step1_sclip_asymmetric results")

    all_strategy_results = {}
    for strategy in strategies:
        results = run_strategy(strategy)
        all_strategy_results[strategy] = results

    # Cross-strategy comparison
    if len(all_strategy_results) > 1:
        print(f"\n{'='*80}")
        print("Cross-strategy comparison (RMS centroid offset from fixed center)")
        print(f"{'='*80}")
        print(f"{'Galaxy':>10s}", end="")
        for s in strategies:
            print(f"  {s:>16s}", end="")
        print()
        print("-" * (12 + 18 * len(strategies)))

        for obj_id, desc in GALAXIES:
            print(f"{obj_id:>10s}", end="")
            for strategy in strategies:
                results = all_strategy_results[strategy]
                r = next(x for x in results if x["obj_id"] == obj_id)
                n0 = r["stop_codes"].get(0, 0)
                n2 = r["stop_codes"].get(2, 0)
                print(f"  {n0:>3d}/{n2:>2d}(0/2)   ", end="")
            print()


if __name__ == "__main__":
    main()
