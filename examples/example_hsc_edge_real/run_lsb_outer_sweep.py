#!/usr/bin/env python3
"""LSB / outer-region configuration sweep on HSC edge-real BCGs.

Crosses sigma clipping, main-fit integrator, and ``lsb_auto_lock`` /
outer-region center regularization axes to probe how different outskirt
treatments affect the three real HSC edge-case BCGs. Everything shared with
``run_baseline.py`` (variance map, custom mask, ``debug=True``, default
harmonic orders [3, 4], no simultaneous harmonics) is held fixed. Each arm
changes one axis from the baseline, plus two combined arms.

Arms (11 total)
---------------
1.  ``baseline``         — identical to ``run_baseline.py`` (self-contained reference)
2.  ``sclip_sym3``       — symmetric ``sclip=3.0, nclip=3``
3.  ``sclip_asym_32``    — asymmetric ``sclip_low=3.0, sclip_high=2.0, nclip=3``
4.  ``sclip_asym_15``    — aggressive ``sclip_low=3.0, sclip_high=1.5, nclip=5``
5.  ``int_median``       — ``integrator="median"`` everywhere
6.  ``int_adaptive_50``  — ``integrator="adaptive", lsb_sma_threshold=50``
7.  ``lsb_lock``         — ``lsb_auto_lock=True`` (default maxgerr=0.3, median integrator)
8.  ``lock_outer_reg``   — arm 7 + ``use_outer_center_regularization=True, strength=2.0``
9.  ``lock_asym_32``     — arm 7 + arm 3 sclip
10. ``no_harmonics``     — baseline with ``compute_deviations=False``
11. ``outer_reg_only``   — soft outer center regularization only (no auto-lock)

Outputs
-------
- Per-galaxy × arm FITS + QA PNG:
  ``outputs/example_hsc_edge_real/lsb_outer_sweep/{arm}/{obj_id}/...``
- Summary tables (runtime, drift, gradient quality) printed to stdout and
  saved as ``outputs/example_hsc_edge_real/lsb_outer_sweep/_summary.{csv,md}``.

Usage
-----
    uv run python examples/example_hsc_edge_real/run_lsb_outer_sweep.py
    uv run python examples/example_hsc_edge_real/run_lsb_outer_sweep.py --smoke
    uv run python examples/example_hsc_edge_real/run_lsb_outer_sweep.py \
        --galaxies 37498869835124888 --arms baseline lsb_lock lock_asym_32
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

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
    SB_ZEROPOINT,
    combined_drift,
    count_stop_code,
    load_galaxy_data,
    load_target_anchor,
    locked_tail_drift,
    outer_gerr_median,
    outward_drift_from_anchor,
    pre_lock_outward,
    reference_centroid,
    spline_rms,
    stop_code_summary_string,
)

SWEEP_DIR = OUTPUT_ROOT / "lsb_outer_sweep"

# Outer-region threshold for the gradient-quality probe. Matches the
# outer_reg_sma_onset default and run_baseline.py.
OUTER_SMA_THRESHOLD = 50.0

# Shared free-fit base, identical to run_baseline.py so ``baseline`` here
# reproduces that runner exactly.
BASE_CONFIG = dict(
    eps=0.2,
    pa=0.0,
    sma0=10.0,
    minsma=0.0,
    astep=0.1,
    linear_growth=False,
    fix_center=False,
    fix_pa=False,
    fix_eps=False,
    debug=True,
    compute_deviations=True,
    full_photometry=True,
    compute_cog=True,
    max_retry_first_isophote=5,
)

# Per-arm delta kwargs on top of BASE_CONFIG. Order defines the default sweep
# schedule.
ARMS = {
    "baseline": dict(),
    "sclip_sym3": dict(sclip=3.0, nclip=3),
    "sclip_asym_32": dict(sclip_low=3.0, sclip_high=2.0, nclip=3),
    "sclip_asym_15": dict(sclip_low=3.0, sclip_high=1.5, nclip=5),
    "int_median": dict(integrator="median"),
    "int_adaptive_50": dict(integrator="adaptive", lsb_sma_threshold=50.0),
    "lsb_lock": dict(
        lsb_auto_lock=True,
        lsb_auto_lock_maxgerr=0.3,
        lsb_auto_lock_debounce=2,
        lsb_auto_lock_integrator="median",
    ),
    "lock_outer_reg": dict(
        lsb_auto_lock=True,
        lsb_auto_lock_maxgerr=0.3,
        lsb_auto_lock_debounce=2,
        lsb_auto_lock_integrator="median",
        use_outer_center_regularization=True,
        outer_reg_strength=2.0,
        outer_reg_sma_onset=50.0,
        outer_reg_sma_width=20.0,
    ),
    "lock_asym_32": dict(
        sclip_low=3.0,
        sclip_high=2.0,
        nclip=3,
        lsb_auto_lock=True,
        lsb_auto_lock_maxgerr=0.3,
        lsb_auto_lock_debounce=2,
        lsb_auto_lock_integrator="median",
    ),
    # --- batch 2 ---
    "no_harmonics": dict(compute_deviations=False),
    "outer_reg_only": dict(
        use_outer_center_regularization=True,
        outer_reg_strength=2.0,
        outer_reg_sma_onset=50.0,
        outer_reg_sma_width=20.0,
    ),
    "outer_reg_asym_32": dict(
        sclip_low=3.0,
        sclip_high=2.0,
        nclip=3,
        use_outer_center_regularization=True,
        outer_reg_strength=2.0,
        outer_reg_sma_onset=50.0,
        outer_reg_sma_width=20.0,
    ),
}


def build_config(arm_name: str) -> dict:
    if arm_name not in ARMS:
        raise ValueError(f"unknown arm '{arm_name}' (known: {list(ARMS)})")
    cfg = dict(BASE_CONFIG)
    cfg.update(ARMS[arm_name])
    return cfg


def run_one(obj_id: str, desc: str, arm_name: str, arm_dir: Path, save_qa: bool):
    image, variance, mask = load_galaxy_data(obj_id)
    x0_anchor, y0_anchor = load_target_anchor(obj_id)
    h, w = image.shape
    max_sma = min(h, w) / 2.0 - 10

    cfg_kwargs = build_config(arm_name)
    config = IsosterConfig(
        x0=x0_anchor, y0=y0_anchor, maxsma=max_sma, **cfg_kwargs
    )

    t0 = time.perf_counter()
    results = fit_image(image, mask=mask, config=config, variance_map=variance)
    elapsed = time.perf_counter() - t0

    isophotes = results["isophotes"]
    n_iso = len(isophotes)
    sc_str = stop_code_summary_string(isophotes)
    n_m1 = count_stop_code(isophotes, -1)
    first_fail = results.get("first_isophote_failure", False)

    transition = results.get("lsb_auto_lock_sma")
    locked_count = results.get("lsb_auto_lock_count", 0) or 0

    x0_ref, y0_ref = reference_centroid(results, isophotes, BASE_CONFIG["sma0"])
    pre_lock = pre_lock_outward(isophotes, BASE_CONFIG["sma0"])
    pl_dx, pl_dy, pl_comb = combined_drift(pre_lock, x0_ref, y0_ref)
    pl_rms = spline_rms(pre_lock)
    lock_dx, lock_dy = locked_tail_drift(isophotes)
    out_dx, out_dy, anchor_x, anchor_y = outward_drift_from_anchor(
        isophotes, BASE_CONFIG["sma0"]
    )
    gerr_med = outer_gerr_median(isophotes, OUTER_SMA_THRESHOLD)

    if save_qa:
        galaxy_out = arm_dir / obj_id
        galaxy_out.mkdir(parents=True, exist_ok=True)
        tag = f"sweep_{arm_name}"
        isophote_results_to_fits(
            results, str(galaxy_out / f"{obj_id}_{tag}_results.fits")
        )
        model = build_isoster_model(image.shape, isophotes, use_harmonics=True)
        plot_qa_summary(
            title=f"{obj_id} - {desc} ({arm_name})",
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
        "desc": desc,
        "arm": arm_name,
        "n_iso": n_iso,
        "elapsed": elapsed,
        "first_isophote_failure": first_fail,
        "transition_sma": transition,
        "locked_count": locked_count,
        "x0_ref": x0_ref,
        "y0_ref": y0_ref,
        "pre_lock_max_dx": pl_dx,
        "pre_lock_max_dy": pl_dy,
        "pre_lock_combined": pl_comb,
        "pre_lock_rms": pl_rms,
        "locked_drift_x": lock_dx,
        "locked_drift_y": lock_dy,
        "outward_drift_x": out_dx,
        "outward_drift_y": out_dy,
        "anchor_x0": anchor_x,
        "anchor_y0": anchor_y,
        "outer_gerr_median": gerr_med,
        "n_stop_m1": n_m1,
        "stop_codes": sc_str,
    }


def _fmt(value, fmt: str, nan_token: str = "   nan") -> str:
    try:
        if value is None or not np.isfinite(value):
            return nan_token
    except TypeError:
        return nan_token
    return format(value, fmt)


def print_runtime_table(rows, arms, galaxy_order):
    arm_w = max(9, max(len(a) for a in arms))
    header = f"  {'ID':>18s}  " + "  ".join(f"{a:>{arm_w}s}" for a in arms)
    print()
    print("  Runtime (s) per galaxy x arm")
    print(header)
    print("  " + "-" * (len(header) - 2))
    by_key = {(r["obj_id"], r["arm"]): r for r in rows}
    for obj_id in galaxy_order:
        cells = []
        for arm in arms:
            row = by_key.get((obj_id, arm))
            cells.append(
                f"{row['elapsed']:>{arm_w}.2f}" if row else f"{'--':>{arm_w}s}"
            )
        print(f"  {obj_id:>18s}  " + "  ".join(cells))
    totals = []
    for arm in arms:
        arm_rows = [r for r in rows if r["arm"] == arm]
        tot = sum(r["elapsed"] for r in arm_rows) if arm_rows else None
        totals.append(
            f"{tot:>{arm_w}.1f}" if tot is not None else f"{'--':>{arm_w}s}"
        )
    print(f"  {'TOTAL':>18s}  " + "  ".join(totals))


def print_drift_table(rows, arms, galaxy_order):
    arm_w = max(15, max(len(a) for a in arms))
    print()
    print("  Drift + gradient quality summary")
    print(
        f"  {'ID':>18s}  {'arm':<{arm_w}s}  "
        f"{'n_iso':>5s}  {'T(s)':>5s}  {'lock_sma':>8s}  {'locked':>6s}  "
        f"{'pl_comb':>7s}  {'pl_rms':>6s}  {'out_dx':>6s}  {'out_dy':>6s}  "
        f"{'gerr50':>7s}  {'sc-1':>4s}  stop_codes"
    )
    print("  " + "-" * (48 + arm_w + 60))
    by_galaxy: dict = {}
    for r in rows:
        by_galaxy.setdefault(r["obj_id"], []).append(r)
    for obj_id in galaxy_order:
        gal_rows = sorted(
            by_galaxy.get(obj_id, []), key=lambda r: arms.index(r["arm"])
        )
        for r in gal_rows:
            ts = (
                f"{r['transition_sma']:8.2f}"
                if r["transition_sma"] is not None
                else "      --"
            )
            print(
                f"  {r['obj_id']:>18s}  {r['arm']:<{arm_w}s}  "
                f"{r['n_iso']:>5d}  {r['elapsed']:>5.1f}  {ts}  "
                f"{r['locked_count']:>6d}  "
                f"{_fmt(r['pre_lock_combined'], '7.2f'):>7s}  "
                f"{_fmt(r['pre_lock_rms'], '6.2f'):>6s}  "
                f"{_fmt(r['outward_drift_x'], '6.2f'):>6s}  "
                f"{_fmt(r['outward_drift_y'], '6.2f'):>6s}  "
                f"{_fmt(r['outer_gerr_median'], '7.3f'):>7s}  "
                f"{r['n_stop_m1']:>4d}  {r['stop_codes']}"
            )
        print()


def write_summary_files(rows, arms, galaxy_order, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "_summary.csv"
    fieldnames = [
        "obj_id",
        "desc",
        "arm",
        "n_iso",
        "elapsed",
        "first_isophote_failure",
        "transition_sma",
        "locked_count",
        "x0_ref",
        "y0_ref",
        "pre_lock_max_dx",
        "pre_lock_max_dy",
        "pre_lock_combined",
        "pre_lock_rms",
        "locked_drift_x",
        "locked_drift_y",
        "outward_drift_x",
        "outward_drift_y",
        "anchor_x0",
        "anchor_y0",
        "outer_gerr_median",
        "n_stop_m1",
        "stop_codes",
    ]
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    md_path = out_dir / "_summary.md"
    lines: list = []
    lines.append("# LSB / outer-region sweep - HSC edge-real BCGs\n")
    lines.append(
        "Baseline: free fit, variance-weighted, custom mask, `debug=True`, "
        "`harmonic_orders=[3,4]`, no `simultaneous_harmonics`, no sigma "
        "clipping, `integrator=\"mean\"`, no LSB features.\n"
    )
    lines.append(
        "Arms change one axis from the baseline; `lock_outer_reg` and "
        "`lock_asym_32` are combined-arm probes.\n"
    )
    lines.append("## Runtime (s)\n")
    lines.append("| ID | " + " | ".join(arms) + " |")
    lines.append("|" + "---|" * (len(arms) + 1))
    by_key = {(r["obj_id"], r["arm"]): r for r in rows}
    for obj_id in galaxy_order:
        cells = []
        for arm in arms:
            row = by_key.get((obj_id, arm))
            cells.append(f"{row['elapsed']:.2f}" if row else "--")
        lines.append(f"| {obj_id} | " + " | ".join(cells) + " |")
    totals = []
    for arm in arms:
        arm_rows = [r for r in rows if r["arm"] == arm]
        totals.append(
            f"{sum(r['elapsed'] for r in arm_rows):.1f}" if arm_rows else "--"
        )
    lines.append("| **TOTAL** | " + " | ".join(totals) + " |")
    lines.append("")

    lines.append("## Drift + gradient quality\n")
    lines.append(
        "| ID | arm | n_iso | T(s) | lock_sma | locked | pl_comb | pl_rms "
        "| out_dx | out_dy | gerr50 | sc-1 | stop_codes |"
    )
    lines.append("|" + "---|" * 13)
    by_galaxy: dict = {}
    for r in rows:
        by_galaxy.setdefault(r["obj_id"], []).append(r)
    for obj_id in galaxy_order:
        gal_rows = sorted(
            by_galaxy.get(obj_id, []), key=lambda r: arms.index(r["arm"])
        )
        for r in gal_rows:
            ts = (
                f"{r['transition_sma']:.2f}"
                if r["transition_sma"] is not None
                else "--"
            )
            lines.append(
                f"| {r['obj_id']} | {r['arm']} | {r['n_iso']} | "
                f"{r['elapsed']:.1f} | {ts} | {r['locked_count']} | "
                f"{_fmt(r['pre_lock_combined'], '.2f', 'nan')} | "
                f"{_fmt(r['pre_lock_rms'], '.2f', 'nan')} | "
                f"{_fmt(r['outward_drift_x'], '.2f', 'nan')} | "
                f"{_fmt(r['outward_drift_y'], '.2f', 'nan')} | "
                f"{_fmt(r['outer_gerr_median'], '.3f', 'nan')} | "
                f"{r['n_stop_m1']} | {r['stop_codes']} |"
            )
    md_path.write_text("\n".join(lines) + "\n")
    return csv_path, md_path


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Run only the first galaxy (still all arms).",
    )
    ap.add_argument(
        "--galaxies",
        nargs="+",
        default=None,
        help="Restrict to these galaxy IDs (default: all 3).",
    )
    ap.add_argument(
        "--arms",
        nargs="+",
        default=None,
        help=f"Restrict to these arms. Default: all ({list(ARMS)}).",
    )
    ap.add_argument(
        "--no-qa",
        action="store_true",
        help="Skip per-galaxy FITS + QA PNG output (faster, less disk).",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    galaxies = GALAXIES
    if args.smoke:
        galaxies = GALAXIES[:1]
    if args.galaxies:
        wanted = set(args.galaxies)
        galaxies = [g for g in GALAXIES if g[0] in wanted]
        if not galaxies:
            raise SystemExit(f"No matching galaxies for {args.galaxies}")
    galaxy_order = [g[0] for g in galaxies]

    arms = args.arms or list(ARMS.keys())
    for arm in arms:
        build_config(arm)  # fail fast on unknown arm names

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 76)
    print("  LSB / outer-region sweep on HSC edge-real BCGs")
    print(f"  Arms: {arms}")
    print(f"  Galaxies: {galaxy_order}")
    print(f"  Output: {SWEEP_DIR}")
    print("=" * 76)

    rows = []
    for arm in arms:
        arm_dir = SWEEP_DIR / arm
        print(f"\n-- arm: {arm} --")
        for obj_id, desc in galaxies:
            print(f"  fitting {obj_id} ({desc}) ...", flush=True)
            row = run_one(obj_id, desc, arm, arm_dir, save_qa=not args.no_qa)
            ts = (
                f"{row['transition_sma']:.2f}"
                if row["transition_sma"] is not None
                else "--"
            )
            print(
                f"    {row['n_iso']} iso, {row['elapsed']:.2f}s, "
                f"lock_sma={ts}, locked={row['locked_count']}, "
                f"pl_comb={_fmt(row['pre_lock_combined'], '.2f', 'nan')} px, "
                f"gerr50={_fmt(row['outer_gerr_median'], '.3f', 'nan')}"
            )
            rows.append(row)

    print_runtime_table(rows, arms, galaxy_order)
    print_drift_table(rows, arms, galaxy_order)

    csv_path, md_path = write_summary_files(rows, arms, galaxy_order, SWEEP_DIR)
    print(f"\nSummary CSV: {csv_path}")
    print(f"Summary MD : {md_path}")


if __name__ == "__main__":
    main()
