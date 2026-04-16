#!/usr/bin/env python3
"""Baseline isoster run on the three real HSC edge-case BCGs.

Single configuration = isoster defaults plus the session's non-negotiables:
variance-weighted fit, custom detection-based mask, ``debug=True``, default
harmonic orders [3, 4] with no simultaneous ISOFIT, free geometry, and
``max_retry_first_isophote=5`` for robustness on cluster-BCG starting
isophotes. No sigma clipping, default ``integrator="mean"``, no LSB features.

This is the reference result that every arm in ``run_lsb_outer_sweep.py``
is compared against.

Usage
-----
    uv run python examples/example_hsc_edge_real/run_baseline.py
    uv run python examples/example_hsc_edge_real/run_baseline.py --galaxies 37498869835124888
"""

from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.rcParams["text.usetex"] = False

import numpy as np

from isoster import fit_image
from isoster.config import IsosterConfig
from isoster.model import build_isoster_model
from isoster.plotting import plot_qa_summary
from isoster.utils import isophote_results_to_fits

from common import (
    GALAXIES,
    OUTPUT_ROOT,
    PIXEL_SCALE_ARCSEC,
    SB_ZEROPOINT,
    combined_drift,
    count_stop_code,
    load_galaxy_data,
    load_target_anchor,
    outer_gerr_median,
    outward_drift_from_anchor,
    pre_lock_outward,
    reference_centroid,
    spline_rms,
    stop_code_summary_string,
)

# Outer-region threshold used for the outer_gerr_median probe. Matches the
# outer_reg_sma_onset default and the LSB auto-lock sma regime on the
# edgecases example. Keep in sync with run_lsb_outer_sweep.py.
OUTER_SMA_THRESHOLD = 50.0

BASELINE_CONFIG = dict(
    # Starting geometry (defaults). Anchor comes from the mask header.
    eps=0.2,
    pa=0.0,
    # SMA schedule (defaults).
    sma0=10.0,
    minsma=0.0,
    astep=0.1,
    linear_growth=False,
    # Fully free geometry.
    fix_center=False,
    fix_pa=False,
    fix_eps=False,
    # User mandates.
    debug=True,
    compute_deviations=True,  # defaults to harmonic_orders=[3,4]
    # Analysis outputs (non-default but cheap and useful).
    full_photometry=True,
    compute_cog=True,
    # BCG starting-isophote robustness (matches edgecases convention).
    max_retry_first_isophote=5,
    # Defaults left explicit for the record:
    # sclip=3.0, nclip=0   -> no sigma clipping
    # integrator="mean"
    # simultaneous_harmonics=False
    # lsb_auto_lock=False
    # use_outer_center_regularization=False
)


def run_one(obj_id: str, desc: str, output_dir):
    image, variance, mask = load_galaxy_data(obj_id)
    x0_anchor, y0_anchor = load_target_anchor(obj_id)
    h, w = image.shape
    max_sma = min(h, w) / 2.0 - 10

    config = IsosterConfig(
        x0=x0_anchor, y0=y0_anchor, maxsma=max_sma, **BASELINE_CONFIG
    )

    mask_pct = mask.sum() / mask.size * 100
    print(
        f"  {obj_id} ({desc}): {h}x{w}, anchor=({x0_anchor:.1f}, {y0_anchor:.1f}),"
        f" mask={mask_pct:.1f}%"
    )

    t0 = time.perf_counter()
    results = fit_image(image, mask=mask, config=config, variance_map=variance)
    elapsed = time.perf_counter() - t0

    isophotes = results["isophotes"]
    n_iso = len(isophotes)
    sc_str = stop_code_summary_string(isophotes)
    n_m1 = count_stop_code(isophotes, -1)
    first_fail = results.get("first_isophote_failure", False)

    x0_ref, y0_ref = reference_centroid(results, isophotes, BASELINE_CONFIG["sma0"])
    outward = pre_lock_outward(isophotes, BASELINE_CONFIG["sma0"])
    pl_dx, pl_dy, pl_comb = combined_drift(outward, x0_ref, y0_ref)
    pl_rms = spline_rms(outward)
    anchor_dx, anchor_dy, anchor_x, anchor_y = outward_drift_from_anchor(
        isophotes, BASELINE_CONFIG["sma0"]
    )
    gerr_med = outer_gerr_median(isophotes, OUTER_SMA_THRESHOLD)

    fail_flag = "  *FIRST_ISO_FAIL*" if first_fail else ""
    print(
        f"    {n_iso} iso, {elapsed:.2f}s, stop_codes: {sc_str}"
        f"  n_sc-1={n_m1}{fail_flag}"
    )
    print(
        f"    outward drift vs anchor: max|dx|={anchor_dx:.2f} px"
        f"  max|dy|={anchor_dy:.2f} px  spline_rms={pl_rms:.2f} px"
    )
    print(
        f"    outer |grad_err/grad| median (sma>{OUTER_SMA_THRESHOLD:.0f}): "
        f"{gerr_med:.3f}"
    )

    galaxy_out = output_dir / obj_id
    galaxy_out.mkdir(parents=True, exist_ok=True)
    tag = "baseline"
    fits_path = galaxy_out / f"{obj_id}_{tag}_results.fits"
    isophote_results_to_fits(results, str(fits_path))

    model = build_isoster_model(image.shape, isophotes, use_harmonics=True)
    qa_path = galaxy_out / f"{obj_id}_{tag}_qa.png"
    plot_qa_summary(
        title=f"{obj_id} - {desc} ({tag})",
        image=image,
        isoster_model=model,
        isoster_res=isophotes,
        mask=mask,
        filename=str(qa_path),
        relative_residual=False,
        sb_zeropoint=SB_ZEROPOINT,
        pixel_scale_arcsec=PIXEL_SCALE_ARCSEC,
    )

    return {
        "obj_id": obj_id,
        "desc": desc,
        "n_iso": n_iso,
        "elapsed": elapsed,
        "first_isophote_failure": first_fail,
        "n_stop_m1": n_m1,
        "stop_codes": sc_str,
        "pre_lock_max_dx": pl_dx,
        "pre_lock_max_dy": pl_dy,
        "pre_lock_combined": pl_comb,
        "pre_lock_rms": pl_rms,
        "outward_drift_x": anchor_dx,
        "outward_drift_y": anchor_dy,
        "anchor_x0": anchor_x,
        "anchor_y0": anchor_y,
        "outer_gerr_median": gerr_med,
    }


def print_summary(rows):
    print()
    print(
        f"  {'ID':>18s}  {'N_iso':>5s}  {'T(s)':>5s}  "
        f"{'out_dx':>6s}  {'out_dy':>6s}  {'rms':>6s}  "
        f"{'gerr':>6s}  {'sc-1':>4s}  stop_codes"
    )
    print("  " + "-" * 92)
    for r in rows:
        rms_str = f"{r['pre_lock_rms']:6.2f}" if np.isfinite(r["pre_lock_rms"]) else "   nan"
        gerr_str = (
            f"{r['outer_gerr_median']:6.3f}"
            if np.isfinite(r["outer_gerr_median"])
            else "   nan"
        )
        print(
            f"  {r['obj_id']:>18s}  {r['n_iso']:>5d}  {r['elapsed']:>5.1f}  "
            f"{r['outward_drift_x']:>6.2f}  {r['outward_drift_y']:>6.2f}  "
            f"{rms_str}  {gerr_str}  {r['n_stop_m1']:>4d}  {r['stop_codes']}"
        )


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--galaxies",
        nargs="+",
        default=None,
        help="Restrict to these galaxy IDs (default: all 3).",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    galaxies = GALAXIES
    if args.galaxies:
        wanted = set(args.galaxies)
        galaxies = [g for g in GALAXIES if g[0] in wanted]
        if not galaxies:
            raise SystemExit(f"No matching galaxies for {args.galaxies}")

    output_dir = OUTPUT_ROOT / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("  Baseline isoster run on HSC edge-real BCGs")
    print(f"  Output: {output_dir}")
    print("=" * 76)

    rows = []
    for obj_id, desc in galaxies:
        rows.append(run_one(obj_id, desc, output_dir))

    print_summary(rows)


if __name__ == "__main__":
    main()
