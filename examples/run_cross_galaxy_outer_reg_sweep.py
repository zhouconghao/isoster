#!/usr/bin/env python3
"""Cross-galaxy sweep of outer-region regularization modes.

Validates the damping / solver / baseline trio on every HSC galaxy in
both existing example sets:

- ``example_hsc_edgecases``: 6 mock galaxies with controlled edge cases
  (bright-star blending, artifact, nearby companions). Anchor at image
  center.
- ``example_hsc_edge_real``: 3 real HSC coadd BCGs (cluster centrals).
  Anchor from ``X_OBJ``/``Y_OBJ`` header keys on the custom mask FITS.

Arms (6)
--------
1. ``baseline``            - no outer_reg.
2. ``damping_default``     - mode=damping, default params (onset=50,
   width=20, strength=2, weights={center:1, eps:0, pa:0}). What a user
   gets by flipping ``use_outer_center_regularization=True`` with
   defaults.
3. ``damping_full``        - mode=damping, onset=50, width=20, str=2,
   weights={1,1,1}.
4. ``damping_recommended`` - mode=damping, onset=100, width=20, str=4,
   weights={1,1,1}. The final recommended config.
5. ``solver_standard``     - mode=solver, onset=50, width=20, str=2,
   weights={1,1,1}.
6. ``solver_strong``       - mode=solver, onset=50, width=20, str=8,
   weights={1,1,1}.

``geometry_convergence=True`` is auto-enabled by the config validator
for every arm with ``use_outer_center_regularization=True``.

Output layout
-------------
    outputs/cross_galaxy_outer_reg_sweep/
        <arm>/
            <obj_id>/                 # mock or real, combined
                <obj_id>_<arm>_results.fits
                <obj_id>_<arm>_qa.png
        _summary.csv
        _summary.md

Metrics (outer region, sma >= sma0 for the source set)
- ``elapsed``             wall-clock seconds for ``fit_image``.
- ``n_iso``, stop codes.
- ``pl_comb``             max combined center drift (px) vs. anchor iso.
- ``max_dpa_deg``         max |delta pa| between adjacent outer iso.
- ``max_deps``            max |delta eps| between adjacent outer iso.
- ``n_sat_pa``            count of outer steps with |dpa| >= 90% of
                          clip_max_pa (i.e. saturated clipped PA jumps).
- ``n_sat_eps``           same for eps.
- ``eps_min``, ``eps_max``, ``pa_min_deg``, ``pa_max_deg``.

Usage
-----
    uv run python examples/run_cross_galaxy_outer_reg_sweep.py
    uv run python examples/run_cross_galaxy_outer_reg_sweep.py --no-qa
    uv run python examples/run_cross_galaxy_outer_reg_sweep.py \\
        --galaxies 37498869835124888 10140088 --arms baseline damping_recommended
"""

from __future__ import annotations

import argparse
import csv
import time
import warnings
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

EXAMPLES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLES_DIR.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "cross_galaxy_outer_reg_sweep"

BAND = "HSC_I"
SB_ZEROPOINT = 27.0
PIXEL_SCALE_ARCSEC = 0.168  # HSC coadd


# Galaxy registry. Each entry describes where to find the data and how
# to pick the anchor. ``source`` is the example-folder name; ``kind``
# is either ``"mock"`` (edgecases, anchor = image center) or ``"real"``
# (edge_real, anchor from X_OBJ/Y_OBJ mask header).
GALAXIES = [
    # --- mock (example_hsc_edgecases) ---
    {"obj_id": "10140088", "desc": "clear case",               "source": "example_hsc_edgecases", "kind": "mock"},
    {"obj_id": "10140002", "desc": "nearby bright star",       "source": "example_hsc_edgecases", "kind": "mock"},
    {"obj_id": "10140006", "desc": "nearby large galaxy",      "source": "example_hsc_edgecases", "kind": "mock"},
    {"obj_id": "10140009", "desc": "blending bright star",     "source": "example_hsc_edgecases", "kind": "mock"},
    {"obj_id": "10140056", "desc": "artifact",                 "source": "example_hsc_edgecases", "kind": "mock"},
    {"obj_id": "10140093", "desc": "near galaxy cluster",      "source": "example_hsc_edgecases", "kind": "mock"},
    # --- real (example_hsc_edge_real) ---
    {"obj_id": "37498869835124888", "desc": "BCG + companions",          "source": "example_hsc_edge_real", "kind": "real"},
    {"obj_id": "42177291811318246", "desc": "BCG + NW companion + star", "source": "example_hsc_edge_real", "kind": "real"},
    {"obj_id": "42310032070569600", "desc": "BCG + star halo + blend",   "source": "example_hsc_edge_real", "kind": "real"},
]


# Fit knobs held fixed across all arms. Matches the existing
# ``run_outer_reg_param_sweep.py`` base for the real galaxies, which is
# the same configuration used to probe the mocks in the earlier
# ``run_lsb_mode_sweep.py``.
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


ARMS = {
    "baseline": dict(),
    "damping_default": dict(
        use_outer_center_regularization=True,
        outer_reg_mode="damping",
        outer_reg_sma_onset=50.0,
        outer_reg_sma_width=20.0,
        outer_reg_strength=2.0,
        # default weights = {"center": 1.0, "eps": 0.0, "pa": 0.0}
    ),
    "damping_full": dict(
        use_outer_center_regularization=True,
        outer_reg_mode="damping",
        outer_reg_sma_onset=50.0,
        outer_reg_sma_width=20.0,
        outer_reg_strength=2.0,
        outer_reg_weights={"center": 1.0, "eps": 1.0, "pa": 1.0},
    ),
    "damping_recommended": dict(
        use_outer_center_regularization=True,
        outer_reg_mode="damping",
        outer_reg_sma_onset=100.0,
        outer_reg_sma_width=20.0,
        outer_reg_strength=4.0,
        outer_reg_weights={"center": 1.0, "eps": 1.0, "pa": 1.0},
    ),
    "solver_standard": dict(
        use_outer_center_regularization=True,
        outer_reg_mode="solver",
        outer_reg_sma_onset=50.0,
        outer_reg_sma_width=20.0,
        outer_reg_strength=2.0,
        outer_reg_weights={"center": 1.0, "eps": 1.0, "pa": 1.0},
    ),
    "solver_strong": dict(
        use_outer_center_regularization=True,
        outer_reg_mode="solver",
        outer_reg_sma_onset=50.0,
        outer_reg_sma_width=20.0,
        outer_reg_strength=8.0,
        outer_reg_weights={"center": 1.0, "eps": 1.0, "pa": 1.0},
    ),
}


def build_config(arm_name: str) -> dict:
    if arm_name not in ARMS:
        raise ValueError(f"unknown arm '{arm_name}' (known: {list(ARMS)})")
    cfg = dict(BASE_CONFIG)
    cfg.update(ARMS[arm_name])
    return cfg


def _galaxy_dir(gal: dict) -> Path:
    return EXAMPLES_DIR / gal["source"] / "data" / gal["obj_id"]


def load_galaxy_data(gal: dict):
    """Return ``(image, variance, mask, x0, y0)`` for a galaxy entry.

    Handles both mock and real data layouts: mock uses a plain
    ``_mask.fits`` with anchor at image center; real uses a custom
    mask with ``X_OBJ``/``Y_OBJ`` header keys for the anchor.
    """
    gdir = _galaxy_dir(gal)
    obj_id = gal["obj_id"]
    image = fits.getdata(gdir / f"{obj_id}_{BAND}_image.fits").astype(np.float64)
    variance = fits.getdata(gdir / f"{obj_id}_{BAND}_variance.fits").astype(np.float64)
    if gal["kind"] == "real":
        mask_path = gdir / f"{obj_id}_{BAND}_mask_custom.fits"
        mask = fits.getdata(mask_path).astype(bool)
        header = fits.getheader(mask_path)
        if "X_OBJ" not in header or "Y_OBJ" not in header:
            raise KeyError(
                f"{mask_path.name}: missing X_OBJ/Y_OBJ header keys"
            )
        x0 = float(header["X_OBJ"])
        y0 = float(header["Y_OBJ"])
    else:
        mask = fits.getdata(gdir / f"{obj_id}_{BAND}_mask.fits").astype(bool)
        h, w = image.shape
        x0 = w / 2.0
        y0 = h / 2.0
    return image, variance, mask, x0, y0


def stop_code_summary(isophotes) -> str:
    counts: dict = {}
    for iso in isophotes:
        code = iso.get("stop_code", -99)
        counts[code] = counts.get(code, 0) + 1
    return " ".join(f"{k}:{v}" for k, v in sorted(counts.items()))


def outer_metrics(isophotes, sma_threshold, clip_max_eps, clip_max_pa):
    """Compute outer-region smoothness/drift metrics.

    Returns a dict with keys:
      pl_comb: max combined center drift (px) vs first outer isophote.
      max_dpa_deg, max_deps: worst adjacent-isophote step.
      n_sat_pa, n_sat_eps: count of steps >= 90% of per-iteration clip.
      eps_min, eps_max, pa_min_deg, pa_max_deg: ranges in outer region.
    """
    sma = np.array([i["sma"] for i in isophotes])
    x0 = np.array([i["x0"] for i in isophotes])
    y0 = np.array([i["y0"] for i in isophotes])
    eps = np.array([i["eps"] for i in isophotes])
    pa_rad = np.array([i["pa"] for i in isophotes])
    order = np.argsort(sma)
    sma, x0, y0, eps, pa_rad = sma[order], x0[order], y0[order], eps[order], pa_rad[order]
    m = sma >= sma_threshold
    if m.sum() < 2:
        return {
            "pl_comb": float("nan"),
            "max_dpa_deg": float("nan"),
            "max_deps": float("nan"),
            "n_sat_pa": 0,
            "n_sat_eps": 0,
            "eps_min": float("nan"),
            "eps_max": float("nan"),
            "pa_min_deg": float("nan"),
            "pa_max_deg": float("nan"),
        }
    x0o, y0o, eps_o = x0[m], y0[m], eps[m]
    pa_o_deg = np.degrees(pa_rad[m])
    max_dx = float(np.max(np.abs(x0o - x0o[0])))
    max_dy = float(np.max(np.abs(y0o - y0o[0])))
    pl_comb = float(np.sqrt(max_dx**2 + max_dy**2))
    deps = np.abs(np.diff(eps_o))
    # Wrap delta PA onto (-pi/2, pi/2] so mod-pi artefacts don't blow up
    dpa_rad = np.diff(pa_rad[m])
    dpa_rad = ((dpa_rad + 0.5 * np.pi) % np.pi) - 0.5 * np.pi
    dpa_deg = np.abs(np.degrees(dpa_rad))
    n_sat_pa = int(np.sum(dpa_deg >= 0.9 * np.degrees(clip_max_pa)))
    n_sat_eps = int(np.sum(deps >= 0.9 * clip_max_eps))
    return {
        "pl_comb": pl_comb,
        "max_dpa_deg": float(dpa_deg.max()),
        "max_deps": float(deps.max()),
        "n_sat_pa": n_sat_pa,
        "n_sat_eps": n_sat_eps,
        "eps_min": float(eps_o.min()),
        "eps_max": float(eps_o.max()),
        "pa_min_deg": float(pa_o_deg.min()),
        "pa_max_deg": float(pa_o_deg.max()),
    }


def run_one(gal: dict, arm_name: str, arm_dir: Path, save_qa: bool):
    image, variance, mask, x0, y0 = load_galaxy_data(gal)
    h, w = image.shape
    max_sma = min(h, w) / 2.0 - 10

    cfg_kwargs = build_config(arm_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config = IsosterConfig(x0=x0, y0=y0, maxsma=max_sma, **cfg_kwargs)

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = fit_image(image, mask=mask, config=config, variance_map=variance)
    elapsed = time.perf_counter() - t0

    isophotes = results["isophotes"]
    sc_str = stop_code_summary(isophotes)
    n_iso = len(isophotes)
    sc0 = sum(1 for i in isophotes if i.get("stop_code") == 0)
    sc2 = sum(1 for i in isophotes if i.get("stop_code") == 2)
    scm1 = sum(1 for i in isophotes if i.get("stop_code") == -1)

    sma0 = BASE_CONFIG["sma0"]
    clip_max_eps = config.clip_max_eps if config.clip_max_eps is not None else 0.1
    clip_max_pa = config.clip_max_pa if config.clip_max_pa is not None else 0.5
    om = outer_metrics(isophotes, sma0, clip_max_eps, clip_max_pa)

    if save_qa:
        galaxy_out = arm_dir / gal["obj_id"]
        galaxy_out.mkdir(parents=True, exist_ok=True)
        tag = arm_name
        isophote_results_to_fits(
            results, str(galaxy_out / f"{gal['obj_id']}_{tag}_results.fits")
        )
        model = build_isoster_model(image.shape, isophotes, use_harmonics=True)
        plot_qa_summary(
            title=f"{gal['obj_id']} - {gal['desc']} ({arm_name})",
            image=image,
            isoster_model=model,
            isoster_res=isophotes,
            mask=mask,
            filename=str(galaxy_out / f"{gal['obj_id']}_{tag}_qa.png"),
            relative_residual=False,
            sb_zeropoint=SB_ZEROPOINT,
            pixel_scale_arcsec=PIXEL_SCALE_ARCSEC,
        )

    return {
        "obj_id": gal["obj_id"],
        "desc": gal["desc"],
        "kind": gal["kind"],
        "source": gal["source"],
        "arm": arm_name,
        "elapsed": elapsed,
        "n_iso": n_iso,
        "sc_0": sc0,
        "sc_2": sc2,
        "sc_m1": scm1,
        "stop_codes": sc_str,
        **om,
    }


def _fmt(value, fmt: str, nan_token: str = "nan") -> str:
    try:
        if value is None or not np.isfinite(value):
            return nan_token
    except TypeError:
        return nan_token
    return format(value, fmt)


def print_summary(rows, arms, galaxy_order):
    by_key = {(r["obj_id"], r["arm"]): r for r in rows}
    arm_w = max(11, max(len(a) for a in arms))

    print()
    print("  Runtime (s) per galaxy x arm")
    header = f"  {'ID':>18s}  {'kind':>4s}  " + "  ".join(f"{a:>{arm_w}s}" for a in arms)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for obj_id in galaxy_order:
        kind = next((r["kind"] for r in rows if r["obj_id"] == obj_id), "")
        cells = []
        for arm in arms:
            r = by_key.get((obj_id, arm))
            cells.append(f"{r['elapsed']:>{arm_w}.2f}" if r else f"{'--':>{arm_w}s}")
        print(f"  {obj_id:>18s}  {kind:>4s}  " + "  ".join(cells))
    totals = []
    for arm in arms:
        arm_rows = [r for r in rows if r["arm"] == arm]
        totals.append(f"{sum(r['elapsed'] for r in arm_rows):>{arm_w}.2f}" if arm_rows else f"{'--':>{arm_w}s}")
    print(f"  {'TOTAL':>18s}  {'':>4s}  " + "  ".join(totals))

    print()
    print("  Outer-region smoothness + drift (outer sma >= sma0 = %.0f px)" % BASE_CONFIG["sma0"])
    print(
        f"  {'ID':>18s}  {'arm':<{arm_w}s}  "
        f"{'n_iso':>5s}  {'sc=0':>4s}  {'sc=2':>4s}  {'T(s)':>5s}  "
        f"{'pl_comb':>7s}  {'max_dpa':>7s}  {'max_deps':>8s}  "
        f"{'sat_pa':>6s}  {'sat_eps':>7s}  {'eps_rng':>11s}  {'pa_rng(deg)':>12s}"
    )
    print("  " + "-" * (42 + arm_w + 90))
    by_galaxy: dict = {}
    for r in rows:
        by_galaxy.setdefault(r["obj_id"], []).append(r)
    for obj_id in galaxy_order:
        gal_rows = sorted(by_galaxy.get(obj_id, []), key=lambda r: arms.index(r["arm"]))
        for r in gal_rows:
            print(
                f"  {r['obj_id']:>18s}  {r['arm']:<{arm_w}s}  "
                f"{r['n_iso']:>5d}  {r['sc_0']:>4d}  {r['sc_2']:>4d}  {r['elapsed']:>5.2f}  "
                f"{_fmt(r['pl_comb'], '7.2f'):>7s}  "
                f"{_fmt(r['max_dpa_deg'], '7.2f'):>7s}  "
                f"{_fmt(r['max_deps'], '8.3f'):>8s}  "
                f"{r['n_sat_pa']:>6d}  {r['n_sat_eps']:>7d}  "
                f"[{_fmt(r['eps_min'], '.2f')},{_fmt(r['eps_max'], '.2f')}]  "
                f"[{_fmt(r['pa_min_deg'], '5.1f')},{_fmt(r['pa_max_deg'], '5.1f')}]"
            )
        print()


def write_summary_files(rows, arms, galaxy_order, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "_summary.csv"
    fieldnames = [
        "obj_id", "desc", "kind", "source", "arm",
        "elapsed", "n_iso", "sc_0", "sc_2", "sc_m1", "stop_codes",
        "pl_comb", "max_dpa_deg", "max_deps",
        "n_sat_pa", "n_sat_eps",
        "eps_min", "eps_max", "pa_min_deg", "pa_max_deg",
    ]
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    md_path = out_dir / "_summary.md"
    lines: list = []
    lines.append("# Cross-galaxy outer-reg sweep\n")
    lines.append(
        "Compares six arms across all mock (example_hsc_edgecases) and "
        "real (example_hsc_edge_real) HSC galaxies. `geometry_convergence=True` "
        "is auto-enabled by the config validator for every non-baseline arm.\n"
    )
    lines.append("## Arms\n")
    for arm in arms:
        lines.append(f"- **{arm}**: {ARMS[arm] or 'no outer_reg'}")
    lines.append("")

    lines.append("## Runtime (s)\n")
    lines.append("| ID | kind | " + " | ".join(arms) + " |")
    lines.append("|" + "---|" * (len(arms) + 2))
    by_key = {(r["obj_id"], r["arm"]): r for r in rows}
    for obj_id in galaxy_order:
        kind = next((r["kind"] for r in rows if r["obj_id"] == obj_id), "")
        cells = []
        for arm in arms:
            r = by_key.get((obj_id, arm))
            cells.append(f"{r['elapsed']:.2f}" if r else "--")
        lines.append(f"| {obj_id} | {kind} | " + " | ".join(cells) + " |")
    totals = []
    for arm in arms:
        arm_rows = [r for r in rows if r["arm"] == arm]
        totals.append(f"{sum(r['elapsed'] for r in arm_rows):.2f}" if arm_rows else "--")
    lines.append(f"| **TOTAL** |  | " + " | ".join(totals) + " |")
    lines.append("")

    lines.append("## Outer-region smoothness + drift\n")
    lines.append(
        "| ID | arm | n_iso | sc=0 | sc=2 | T(s) | pl_comb | max_dpa | max_deps | sat_pa | sat_eps | eps_rng | pa_rng(deg) |"
    )
    lines.append("|" + "---|" * 13)
    by_galaxy: dict = {}
    for r in rows:
        by_galaxy.setdefault(r["obj_id"], []).append(r)
    for obj_id in galaxy_order:
        gal_rows = sorted(by_galaxy.get(obj_id, []), key=lambda r: arms.index(r["arm"]))
        for r in gal_rows:
            lines.append(
                f"| {r['obj_id']} | {r['arm']} | {r['n_iso']} | {r['sc_0']} | "
                f"{r['sc_2']} | {r['elapsed']:.2f} | "
                f"{_fmt(r['pl_comb'], '.2f')} | "
                f"{_fmt(r['max_dpa_deg'], '.2f')} | "
                f"{_fmt(r['max_deps'], '.3f')} | "
                f"{r['n_sat_pa']} | {r['n_sat_eps']} | "
                f"[{_fmt(r['eps_min'], '.2f')},{_fmt(r['eps_max'], '.2f')}] | "
                f"[{_fmt(r['pa_min_deg'], '.1f')},{_fmt(r['pa_max_deg'], '.1f')}] |"
            )
    md_path.write_text("\n".join(lines) + "\n")
    return csv_path, md_path


def parse_args():
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument(
        "--galaxies",
        nargs="+",
        default=None,
        help="Restrict to these galaxy IDs. Default: all 9 (6 mock + 3 real).",
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
    if args.galaxies:
        wanted = set(args.galaxies)
        galaxies = [g for g in GALAXIES if g["obj_id"] in wanted]
        missing = wanted - {g["obj_id"] for g in galaxies}
        if missing:
            raise SystemExit(
                f"No match for {sorted(missing)}. Known: "
                f"{[g['obj_id'] for g in GALAXIES]}"
            )
    galaxy_order = [g["obj_id"] for g in galaxies]

    arms = args.arms or list(ARMS.keys())
    for arm in arms:
        build_config(arm)  # fail fast on unknown arm names

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("=" * 76)
    print("  Cross-galaxy outer-reg sweep")
    print(f"  Arms:     {arms}")
    print(f"  Galaxies: {galaxy_order}")
    print(f"  Output:   {OUTPUT_ROOT}")
    print("=" * 76)

    rows = []
    for arm in arms:
        arm_dir = OUTPUT_ROOT / arm
        print(f"\n-- arm: {arm} --")
        for gal in galaxies:
            print(f"  fitting {gal['obj_id']} ({gal['kind']}: {gal['desc']}) ...", flush=True)
            row = run_one(gal, arm, arm_dir, save_qa=not args.no_qa)
            print(
                f"    {row['n_iso']} iso, {row['elapsed']:.2f}s, "
                f"sc=[0:{row['sc_0']},2:{row['sc_2']},-1:{row['sc_m1']}], "
                f"pl_comb={_fmt(row['pl_comb'], '.2f')}, "
                f"max_dpa={_fmt(row['max_dpa_deg'], '.2f')} deg, "
                f"sat_pa={row['n_sat_pa']}, sat_eps={row['n_sat_eps']}"
            )
            rows.append(row)

    print_summary(rows, arms, galaxy_order)
    csv_path, md_path = write_summary_files(rows, arms, galaxy_order, OUTPUT_ROOT)
    print(f"\nSummary CSV: {csv_path}")
    print(f"Summary MD : {md_path}")


if __name__ == "__main__":
    main()
