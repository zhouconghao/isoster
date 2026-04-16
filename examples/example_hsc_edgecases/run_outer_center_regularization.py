#!/usr/bin/env python3
"""Outer-region center regularization benchmark on HSC edge cases.

Sweeps the 6 HSC edge-case galaxies through a set of outer-region center
regularization arms plus a baseline (lock-only, no outer regularization),
and prints a per-arm summary table. Outputs per-galaxy FITS and QA figures
under ``outputs/example_hsc_edgecases/outer_center_regularization/{arm}/{obj_id}/``.

Arms explored:
- baseline: automatic LSB lock only (same as ``run_lsb_auto_lock.py``).
- s{0.5,2,8}_on50: penalty at strength 0.5/2/8, sma_onset=50.

Metrics per galaxy x arm:
- pre_lock combined drift (max |dx|, |dy|) relative to the inner reference
  centroid (or the anchor isophote when no reference is built).
- pre_lock center spline RMS (robust to outlier isophotes).
- auto-lock commit sma (should stay within a few growth steps of baseline).
- stop-code distribution.
- inner reference (x0_ref, y0_ref), for audit.

Usage
-----
    uv run python examples/example_hsc_edgecases/run_outer_center_regularization.py
"""

from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.rcParams["text.usetex"] = False

import numpy as np
from astropy.io import fits
from scipy.interpolate import UnivariateSpline

from isoster import fit_image
from isoster.config import IsosterConfig
from isoster.model import build_isoster_model
from isoster.plotting import plot_qa_summary
from isoster.utils import isophote_results_to_fits

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
    sclip_low=3.0,
    sclip_high=2.0,
    nclip=3,
    # Automatic LSB geometry lock (always on so we can compare the soft+hard combo)
    lsb_auto_lock=True,
    lsb_auto_lock_maxgerr=0.3,
    lsb_auto_lock_debounce=2,
    lsb_auto_lock_integrator="median",
)

# Arm schedule. Each tuple is (arm_name, extra_cfg_kwargs).
ARMS = [
    ("baseline", dict()),
    ("s0p5_on50", dict(use_outer_center_regularization=True,
                       outer_reg_strength=0.5, outer_reg_sma_onset=50.0, outer_reg_sma_width=15.0)),
    ("s2p0_on50", dict(use_outer_center_regularization=True,
                       outer_reg_strength=2.0, outer_reg_sma_onset=50.0, outer_reg_sma_width=15.0)),
    ("s8p0_on50", dict(use_outer_center_regularization=True,
                       outer_reg_strength=8.0, outer_reg_sma_onset=50.0, outer_reg_sma_width=15.0)),
]


def load_galaxy_data(obj_id):
    galaxy_dir = DATA_DIR / obj_id
    image = fits.getdata(galaxy_dir / f"{obj_id}_{BAND}_image.fits").astype(np.float64)
    variance = fits.getdata(galaxy_dir / f"{obj_id}_{BAND}_variance.fits").astype(np.float64)
    mask = fits.getdata(galaxy_dir / f"{obj_id}_{BAND}_mask.fits").astype(bool)
    return image, variance, mask


def stop_code_summary(isophotes):
    counts = {}
    for iso in isophotes:
        code = iso.get("stop_code", -99)
        counts[code] = counts.get(code, 0) + 1
    return counts


def pre_lock_outward(isophotes, sma0):
    """Outward isophotes excluding any auto-lock-frozen tail."""
    out = []
    for iso in isophotes:
        if iso.get("sma", 0.0) < sma0:
            continue
        if iso.get("lsb_locked", False):
            continue
        out.append(iso)
    return out


def combined_drift(isos, x0_ref, y0_ref):
    if not isos:
        return 0.0, 0.0, 0.0
    dx = np.array([abs(iso["x0"] - x0_ref) for iso in isos])
    dy = np.array([abs(iso["y0"] - y0_ref) for iso in isos])
    return float(np.max(dx)), float(np.max(dy)), float(np.sqrt(np.max(dx) ** 2 + np.max(dy) ** 2))


def spline_rms(isos):
    """Fit a smoothing spline to x0(sma) and y0(sma); return combined RMS.

    Guarded for small sample counts (<6): returns NaN when the fit is not
    meaningful.
    """
    if len(isos) < 6:
        return float("nan")
    sma = np.array([iso["sma"] for iso in isos])
    x0s = np.array([iso["x0"] for iso in isos])
    y0s = np.array([iso["y0"] for iso in isos])
    order = np.argsort(sma)
    sma, x0s, y0s = sma[order], x0s[order], y0s[order]
    try:
        spl_x = UnivariateSpline(sma, x0s, k=3, s=len(sma))
        spl_y = UnivariateSpline(sma, y0s, k=3, s=len(sma))
    except Exception:
        return float("nan")
    rx = x0s - spl_x(sma)
    ry = y0s - spl_y(sma)
    return float(np.sqrt(np.mean(rx ** 2) + np.mean(ry ** 2)))


def run_one(obj_id, desc, arm_name, arm_cfg, output_dir):
    image, variance, mask = load_galaxy_data(obj_id)
    h, w = image.shape
    x0, y0 = w / 2.0, h / 2.0
    max_sma = min(h, w) / 2.0 - 10

    cfg_kwargs = {**BASE_CONFIG, **arm_cfg}
    config = IsosterConfig(x0=x0, y0=y0, maxsma=max_sma, **cfg_kwargs)

    t0 = time.perf_counter()
    results = fit_image(image, mask=mask, config=config, variance_map=variance)
    elapsed = time.perf_counter() - t0

    isophotes = results["isophotes"]
    n_iso = len(isophotes)
    sc_counts = stop_code_summary(isophotes)
    sc_str = ",".join(f"{k}:{v}" for k, v in sorted(sc_counts.items()))

    transition = results.get("lsb_auto_lock_sma")
    locked_count = results.get("lsb_auto_lock_count", 0)
    x0_ref = results.get("outer_reg_x0_ref")
    y0_ref = results.get("outer_reg_y0_ref")

    # Choose a reference centroid for drift metrics. Prefer the outer_reg
    # inner reference when available; else fall back to the anchor (sma=sma0).
    if x0_ref is None or y0_ref is None:
        anchor = next((iso for iso in isophotes if iso.get("sma") == BASE_CONFIG["sma0"]), None)
        if anchor is None:
            anchor = next((iso for iso in isophotes if iso.get("sma", 0) >= BASE_CONFIG["sma0"]), isophotes[0])
        x0_ref_metric = float(anchor["x0"])
        y0_ref_metric = float(anchor["y0"])
    else:
        x0_ref_metric = float(x0_ref)
        y0_ref_metric = float(y0_ref)

    pre_lock = pre_lock_outward(isophotes, BASE_CONFIG["sma0"])
    mdx, mdy, combined = combined_drift(pre_lock, x0_ref_metric, y0_ref_metric)
    srms = spline_rms(pre_lock)

    # Save per-galaxy results + QA figure
    galaxy_out = output_dir / obj_id
    galaxy_out.mkdir(parents=True, exist_ok=True)
    tag = f"outer_reg_{arm_name}"
    isophote_results_to_fits(results, str(galaxy_out / f"{obj_id}_{tag}_results.fits"))

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
        "desc": desc,
        "arm": arm_name,
        "n_iso": n_iso,
        "elapsed": elapsed,
        "transition_sma": transition,
        "locked_count": locked_count,
        "x0_ref": x0_ref_metric,
        "y0_ref": y0_ref_metric,
        "pre_lock_max_dx": mdx,
        "pre_lock_max_dy": mdy,
        "pre_lock_combined": combined,
        "pre_lock_spline_rms": srms,
        "stop_codes": sc_str,
    }


def print_summary(rows):
    # Table grouped by galaxy, one row per arm.
    arm_width = max(len(a[0]) for a in ARMS)
    print()
    print(
        f"  {'ID':>10s}  {'arm':<{arm_width}s}  "
        f"{'combined':>9s}  {'max_dx':>7s}  {'max_dy':>7s}  "
        f"{'rms':>6s}  {'lock_sma':>8s}  {'locked':>6s}  "
        f"{'x0_ref':>7s}  {'y0_ref':>7s}  stop_codes"
    )
    print("  " + "-" * 110)
    by_galaxy = {}
    for r in rows:
        by_galaxy.setdefault(r["obj_id"], []).append(r)

    for obj_id in [g[0] for g in GALAXIES]:
        for r in by_galaxy.get(obj_id, []):
            ts = f"{r['transition_sma']:8.2f}" if r["transition_sma"] is not None else "      --"
            rms_str = f"{r['pre_lock_spline_rms']:6.2f}" if np.isfinite(r["pre_lock_spline_rms"]) else "   nan"
            print(
                f"  {r['obj_id']:>10s}  {r['arm']:<{arm_width}s}  "
                f"{r['pre_lock_combined']:>9.2f}  {r['pre_lock_max_dx']:>7.2f}  {r['pre_lock_max_dy']:>7.2f}  "
                f"{rms_str}  {ts}  {r['locked_count']:>6d}  "
                f"{r['x0_ref']:>7.2f}  {r['y0_ref']:>7.2f}  {r['stop_codes']}"
            )
        print()  # blank line between galaxies for readability


def main():
    print("=" * 72)
    print("  Outer center regularization benchmark on HSC edge cases")
    print(f"  Output: {OUTPUT_ROOT}/outer_center_regularization")
    print("=" * 72)

    all_rows = []
    for arm_name, arm_cfg in ARMS:
        arm_dir = OUTPUT_ROOT / "outer_center_regularization" / arm_name
        arm_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n-- arm: {arm_name} --")
        for obj_id, desc in GALAXIES:
            print(f"  fitting {obj_id} ({desc}) ...")
            row = run_one(obj_id, desc, arm_name, deepcopy(arm_cfg), arm_dir)
            print(
                f"    combined={row['pre_lock_combined']:.2f} px  "
                f"rms={row['pre_lock_spline_rms']:.2f}  "
                f"lock_sma={row['transition_sma']}  locked={row['locked_count']}"
            )
            all_rows.append(row)

    print_summary(all_rows)


if __name__ == "__main__":
    main()
