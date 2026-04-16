#!/usr/bin/env python3
"""Run isoster automatic LSB geometry lock on HSC edge-case galaxies.

Fits isophotes to all 6 galaxies in the HSC-I band starting in free mode and
allowing the LSB auto-lock detector to commit a lock to the last clean
isophote once the outward fit enters the LSB regime. Uses the same base
config as ``run_step1_free_fit.py`` (variant=baseline) so the outputs are
directly comparable to the step1 baseline.

Usage
-----
    uv run python examples/example_hsc_edgecases/run_lsb_auto_lock.py
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
    # Asymmetric sigma clipping: clip bright outliers tighter (contaminants
    # are bright neighbours), not the faint side. This is the single biggest
    # inner-region improvement on the HSC edge-case galaxies — see the
    # run_step1_sclip asymmetric variant for the standalone baseline.
    sclip_low=3.0,
    sclip_high=2.0,
    nclip=3,
    # Automatic LSB geometry lock
    lsb_auto_lock=True,
    lsb_auto_lock_maxgerr=0.3,
    lsb_auto_lock_debounce=2,
    lsb_auto_lock_integrator="median",
)


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


def centroid_drift_outward_from_anchor(isophotes, anchor_sma):
    """Max |delta_x|, |delta_y| in pixels on outward isophotes with sma > anchor_sma,
    measured relative to the anchor (immediately before the lock) x0/y0.

    Returns (max_dx, max_dy, anchor_x0, anchor_y0). Anchor is taken as the
    outward isophote with sma closest to but not exceeding ``anchor_sma``.
    """
    outward = [iso for iso in isophotes if iso["sma"] >= BASE_CONFIG["sma0"]]
    if not outward:
        return (np.nan, np.nan, np.nan, np.nan)
    anchor = outward[0]
    for iso in outward:
        if iso["sma"] <= anchor_sma:
            anchor = iso
        else:
            break
    dx = np.array([abs(iso["x0"] - anchor["x0"]) for iso in outward])
    dy = np.array([abs(iso["y0"] - anchor["y0"]) for iso in outward])
    return (float(np.nanmax(dx)), float(np.nanmax(dy)), anchor["x0"], anchor["y0"])


def run_one(obj_id, desc, output_dir):
    image, variance, mask = load_galaxy_data(obj_id)
    h, w = image.shape
    x0, y0 = w / 2.0, h / 2.0
    max_sma = min(h, w) / 2.0 - 10

    config = IsosterConfig(x0=x0, y0=y0, maxsma=max_sma, **BASE_CONFIG)

    print(f"  {obj_id} ({desc}): {h}x{w}, mask={np.sum(mask)/mask.size*100:.1f}%")

    t0 = time.perf_counter()
    results = fit_image(image, mask=mask, config=config, variance_map=variance)
    elapsed = time.perf_counter() - t0

    isophotes = results["isophotes"]
    n_iso = len(isophotes)
    sc_counts = stop_code_summary(isophotes)
    sc_str = "  ".join(f"{k}:{v}" for k, v in sorted(sc_counts.items()))

    transition = results.get("lsb_auto_lock_sma")
    locked_count = results.get("lsb_auto_lock_count", 0)

    if transition is not None:
        # Drift in the LOCKED region (should be exactly zero)
        locked_isos = [iso for iso in isophotes if iso.get("lsb_locked")]
        if locked_isos:
            x0s = np.array([iso["x0"] for iso in locked_isos])
            y0s = np.array([iso["y0"] for iso in locked_isos])
            locked_drift_x = float(np.max(x0s) - np.min(x0s))
            locked_drift_y = float(np.max(y0s) - np.min(y0s))
        else:
            locked_drift_x = locked_drift_y = np.nan

        # Drift across the whole outward range, measured relative to anchor
        drift_dx, drift_dy, ax, ay = centroid_drift_outward_from_anchor(isophotes, transition)
    else:
        locked_drift_x = locked_drift_y = np.nan
        # Free run never triggered; fall back to drift over the whole outward range
        drift_dx, drift_dy, ax, ay = centroid_drift_outward_from_anchor(isophotes, float("inf"))

    print(f"    {n_iso} iso, {elapsed:.2f}s, stop_codes: {sc_str}")
    if transition is not None:
        print(
            f"    lock at sma={transition:.2f}, {locked_count} locked iso, "
            f"locked drift (x,y) = ({locked_drift_x:.2e}, {locked_drift_y:.2e}) px"
        )
    else:
        print("    auto-lock detector never triggered (stayed in free mode)")
    print(
        f"    outward max |dx|={drift_dx:.2f} px, max |dy|={drift_dy:.2f} px "
        f"(anchor x0,y0 = {ax:.2f}, {ay:.2f})"
    )

    # Save results + QA figure
    galaxy_out = output_dir / obj_id
    galaxy_out.mkdir(parents=True, exist_ok=True)
    tag = "lsb_auto_lock"
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
        "n_iso": n_iso,
        "elapsed": elapsed,
        "transition_sma": transition,
        "locked_count": locked_count,
        "locked_drift_x": locked_drift_x,
        "locked_drift_y": locked_drift_y,
        "outward_drift_x": drift_dx,
        "outward_drift_y": drift_dy,
    }


def main():
    output_dir = OUTPUT_ROOT / "lsb_auto_lock"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Automatic LSB geometry lock on HSC edge-case galaxies")
    print(f"  Output: {output_dir}")
    print("=" * 72)

    rows = []
    for obj_id, desc in GALAXIES:
        rows.append(run_one(obj_id, desc, output_dir))

    print()
    print(f"  {'ID':>10s}  {'Description':<22s}  {'N':>4s}  "
          f"{'T(s)':>5s}  {'lock_sma':>8s}  {'locked':>6s}  "
          f"{'lock_dx':>8s}  {'lock_dy':>8s}  {'out_dx':>7s}  {'out_dy':>7s}")
    print(f"  {'-'*110}")
    for r in rows:
        ts = (f"{r['transition_sma']:8.2f}" if r["transition_sma"] is not None else "      --")
        print(
            f"  {r['obj_id']:>10s}  {r['desc']:<22s}  {r['n_iso']:>4d}  "
            f"{r['elapsed']:5.1f}  {ts}  {r['locked_count']:>6d}  "
            f"{r['locked_drift_x']:>8.1e}  {r['locked_drift_y']:>8.1e}  "
            f"{r['outward_drift_x']:>7.2f}  {r['outward_drift_y']:>7.2f}"
        )


if __name__ == "__main__":
    main()
