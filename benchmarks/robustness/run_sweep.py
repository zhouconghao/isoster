#!/usr/bin/env python3
"""Robustness benchmark: isoster sensitivity to initial SMA and geometry.

Runs a 1-D perturbation sweep on ``(sma0, eps, pa, dx, dy)`` around a
per-galaxy fiducial start and records how far the resulting isophote
sequence walks from a reference fit. Three fit arms (``bare``,
``retry``, ``retry_lsb``) measure how much ``max_retry_first_isophote``
and the LSB features extend the capture radius.

Design document:
``docs/agent/journal/2026-04-15_robustness_plan.md``.

Usage
-----
Quick smoke test (one mock galaxy, sub-minute, sma0 axis only)::

    uv run python benchmarks/robustness/run_sweep.py --quick

Full single-tier run::

    uv run python benchmarks/robustness/run_sweep.py --tiers mocks

All tiers, restrict to the sma0 and eps axes::

    uv run python benchmarks/robustness/run_sweep.py \
        --tiers mocks huang2013 highorder hsc \
        --axes sma0 eps

Outputs default to ``outputs/benchmark_robustness/`` (singular
``benchmark_`` prefix per ``benchmarks/FRAMEWORK.md`` §1).
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import benchmarks.robustness.datasets as datasets  # noqa: E402
import benchmarks.robustness.metrics as metrics  # noqa: E402
from benchmarks.utils.run_metadata import (  # noqa: E402
    collect_environment_metadata,
    write_json,
)
from isoster import fit_image  # noqa: E402
from isoster.config import IsosterConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Constants: arms, axes, grid
# ---------------------------------------------------------------------------

ARMS = ("bare", "retry", "retry_lsb")
AXES = ("sma0", "eps", "pa", "dx", "dy")

SMA0_FACTORS: Sequence[float] = (
    0.25, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 3.0, 5.0,
)
EPS_VALUES: Sequence[float] = (0.00, 0.05, 0.15, 0.30, 0.50, 0.70)
PA_VALUES_RAD: Sequence[float] = (
    -np.pi / 2.0, -np.pi / 4.0, 0.0, np.pi / 6.0, np.pi / 4.0,
    np.pi / 3.0, np.pi / 2.0,
)
DX_VALUES: Sequence[float] = (-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0)
DY_VALUES: Sequence[float] = (-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0)

# Low-signal cutoff for the relative-intensity denominator. Reference
# intensities below this floor (per pixel) are dropped from the relative
# deviation calculation to keep the ratio well defined.
DEFAULT_LOW_SIGNAL_FLOOR = 1.0e-6

# Shared isoster config base — identical per-arm except for the feature
# deltas in ``ARM_DELTAS``. Knobs that are tied to the data (x0, y0, sma0,
# eps, pa, maxsma) are injected per galaxy / per perturbation downstream.
BASE_CONFIG: Dict[str, Any] = dict(
    minsma=0.0,
    astep=0.1,
    linear_growth=False,
    fix_center=False,
    fix_pa=False,
    fix_eps=False,
    compute_deviations=True,
    full_photometry=False,
    compute_cog=False,
    debug=False,
    sclip_low=3.0,
    sclip_high=2.0,
    nclip=3,
)

ARM_DELTAS: Dict[str, Dict[str, Any]] = {
    "bare": dict(
        max_retry_first_isophote=0,
    ),
    "retry": dict(
        max_retry_first_isophote=5,
    ),
    "retry_lsb": dict(
        max_retry_first_isophote=5,
        lsb_auto_lock=True,
        lsb_auto_lock_maxgerr=0.3,
        lsb_auto_lock_debounce=2,
        lsb_auto_lock_integrator="median",
        use_outer_center_regularization=True,
        outer_reg_strength=2.0,
        outer_reg_sma_onset=50.0,
        outer_reg_sma_width=15.0,
    ),
}

# The reference fit uses the best shipped configuration (``retry_lsb``) at
# the fiducial start. On mocks this is truth-informed; on real galaxies it
# is the LSB-sweep B_std equivalent.
REFERENCE_ARM = "retry_lsb"


# ---------------------------------------------------------------------------
# Per-galaxy fit
# ---------------------------------------------------------------------------


def build_config(
    galaxy: datasets.GalaxyData,
    arm: str,
    sma0: float,
    eps: float,
    pa: float,
    x0: float,
    y0: float,
) -> IsosterConfig:
    """Assemble the isoster config for one arm × perturbation.

    Merge order (later wins): ``BASE_CONFIG`` → arm deltas → tier-level
    ``galaxy.config_overrides`` → per-perturbation anchors (x0, y0,
    sma0, eps, pa, maxsma, minsma). Per-perturbation values always win
    so the robustness axis being swept is never accidentally clobbered.
    """
    merged = dict(BASE_CONFIG)
    merged.update(ARM_DELTAS[arm])
    merged.update(galaxy.config_overrides)
    merged.update(
        dict(
            x0=x0,
            y0=y0,
            sma0=sma0,
            eps=eps,
            pa=pa,
            maxsma=galaxy.maxsma,
            minsma=galaxy.minsma,
        )
    )
    return IsosterConfig(**merged)


def run_fit(
    galaxy: datasets.GalaxyData,
    arm: str,
    sma0: float,
    eps: float,
    pa: float,
    x0: float,
    y0: float,
) -> Dict[str, Any]:
    """Run a single isoster fit; always catch exceptions and report them."""
    t0 = time.perf_counter()
    try:
        config = build_config(galaxy, arm, sma0, eps, pa, x0, y0)
        results = fit_image(
            galaxy.image,
            mask=galaxy.mask,
            config=config,
            variance_map=galaxy.variance_map,
        )
        elapsed = time.perf_counter() - t0
        return {
            "success": True,
            "elapsed": elapsed,
            "results": results,
            "error": None,
        }
    except Exception as exc:  # pragma: no cover - benchmark robustness path
        elapsed = time.perf_counter() - t0
        return {
            "success": False,
            "elapsed": elapsed,
            "results": {"isophotes": []},
            "error": f"{type(exc).__name__}: {exc}",
        }


def stop_code_histogram(isophotes: List[Dict[str, Any]]) -> str:
    counts: Dict[int, int] = {}
    for iso in isophotes:
        code = int(iso.get("stop_code", -99))
        counts[code] = counts.get(code, 0) + 1
    return ",".join(f"{k}:{v}" for k, v in sorted(counts.items()))


def count_accepted(isophotes: List[Dict[str, Any]]) -> int:
    return sum(1 for iso in isophotes if iso.get("stop_code") in (0, 1, 2))


# ---------------------------------------------------------------------------
# Reference fit caching
# ---------------------------------------------------------------------------


def build_reference_fit(galaxy: datasets.GalaxyData) -> Dict[str, Any]:
    """Run the reference fit for a galaxy at the fiducial start."""
    fit = run_fit(
        galaxy,
        arm=REFERENCE_ARM,
        sma0=galaxy.fiducial_sma0,
        eps=galaxy.fiducial_eps,
        pa=galaxy.fiducial_pa,
        x0=galaxy.fiducial_x0,
        y0=galaxy.fiducial_y0,
    )
    if not fit["success"]:
        raise RuntimeError(
            f"reference fit failed for {galaxy.spec.obj_id}: {fit['error']}"
        )
    return fit


# ---------------------------------------------------------------------------
# Perturbation generators
# ---------------------------------------------------------------------------


def perturbations_for_axis(
    galaxy: datasets.GalaxyData,
    axis: str,
) -> List[Dict[str, Any]]:
    """Yield starting conditions along one 1-D axis.

    Each entry is a dict with ``axis``, ``value``, and the five starting
    parameters (``sma0``, ``eps``, ``pa``, ``x0``, ``y0``).
    """
    rows: List[Dict[str, Any]] = []
    base = dict(
        sma0=galaxy.fiducial_sma0,
        eps=galaxy.fiducial_eps,
        pa=galaxy.fiducial_pa,
        x0=galaxy.fiducial_x0,
        y0=galaxy.fiducial_y0,
    )
    if axis == "sma0":
        for factor in SMA0_FACTORS:
            row = dict(base)
            row["sma0"] = max(1.0, galaxy.fiducial_sma0 * factor)
            rows.append({"axis": "sma0", "value": factor, **row})
    elif axis == "eps":
        for value in EPS_VALUES:
            row = dict(base)
            row["eps"] = float(value)
            rows.append({"axis": "eps", "value": float(value), **row})
    elif axis == "pa":
        for value in PA_VALUES_RAD:
            row = dict(base)
            row["pa"] = float(value)
            rows.append({"axis": "pa", "value": float(value), **row})
    elif axis == "dx":
        for value in DX_VALUES:
            row = dict(base)
            row["x0"] = galaxy.fiducial_x0 + float(value)
            rows.append({"axis": "dx", "value": float(value), **row})
    elif axis == "dy":
        for value in DY_VALUES:
            row = dict(base)
            row["y0"] = galaxy.fiducial_y0 + float(value)
            rows.append({"axis": "dy", "value": float(value), **row})
    else:
        raise ValueError(f"unknown axis: {axis!r}")
    return rows


def quick_perturbations(galaxy: datasets.GalaxyData) -> List[Dict[str, Any]]:
    """Minimal grid for the ``--quick`` smoke path: three sma0 factors."""
    base = dict(
        sma0=galaxy.fiducial_sma0,
        eps=galaxy.fiducial_eps,
        pa=galaxy.fiducial_pa,
        x0=galaxy.fiducial_x0,
        y0=galaxy.fiducial_y0,
    )
    rows: List[Dict[str, Any]] = []
    for factor in (0.5, 1.0, 2.0):
        row = dict(base)
        row["sma0"] = max(1.0, galaxy.fiducial_sma0 * factor)
        rows.append({"axis": "sma0", "value": factor, **row})
    return rows


# ---------------------------------------------------------------------------
# Per-row metric computation
# ---------------------------------------------------------------------------


def row_for_fit(
    galaxy: datasets.GalaxyData,
    arm: str,
    axis: str,
    value: float,
    perturbation: Dict[str, Any],
    fit: Dict[str, Any],
    reference_isophotes: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute one row of the results table."""
    results = fit["results"]
    isophotes = results.get("isophotes", [])
    first_iso_failure = bool(results.get("first_isophote_failure", False))
    retry_log = results.get("first_isophote_retry_log", []) or []

    comparison = metrics.compare_to_reference(
        reference_isophotes,
        isophotes,
        low_signal_threshold=DEFAULT_LOW_SIGNAL_FLOOR,
    )
    bins = metrics.bin_metrics(comparison, first_iso_failure=first_iso_failure)
    worst = metrics.worst_bin(bins)

    return {
        "tier": galaxy.spec.tier,
        "obj_id": galaxy.spec.obj_id,
        "description": galaxy.spec.description,
        "arm": arm,
        "axis": axis,
        "value": float(value),
        "start_sma0": float(perturbation["sma0"]),
        "start_eps": float(perturbation["eps"]),
        "start_pa": float(perturbation["pa"]),
        "start_x0": float(perturbation["x0"]),
        "start_y0": float(perturbation["y0"]),
        "elapsed": float(fit["elapsed"]),
        "success": bool(fit["success"]),
        "error": fit["error"],
        "n_iso_total": len(isophotes),
        "n_iso_accepted": count_accepted(isophotes),
        "first_iso_failure": first_iso_failure,
        "first_iso_retry": len(retry_log),
        "stop_code_hist": stop_code_histogram(isophotes),
        "comparison": asdict(comparison),
        "bins": bins,
        "worst_bin": worst,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_sweep(
    tiers: Sequence[str],
    arms: Sequence[str],
    axes: Sequence[str],
    galaxy_filter: Optional[Sequence[str]],
    quick: bool,
) -> Dict[str, Any]:
    """Run the full sweep and return the in-memory results payload."""
    all_rows: List[Dict[str, Any]] = []
    references: Dict[str, Dict[str, Any]] = {}

    for tier in tiers:
        specs = datasets.list_galaxies(tier)
        if galaxy_filter:
            wanted = set(galaxy_filter)
            specs = [s for s in specs if s.obj_id in wanted]
        if not specs:
            print(f"  [tier {tier}] no galaxies (filter={galaxy_filter}) — skipping")
            continue
        if quick:
            specs = specs[:1]
        for spec in specs:
            print(f"  [{tier}] loading {spec.obj_id} ({spec.description})")
            galaxy = datasets.load_galaxy(spec)
            print(
                f"    fiducial: sma0={galaxy.fiducial_sma0:.2f} "
                f"eps={galaxy.fiducial_eps:.2f} pa={galaxy.fiducial_pa:.2f} "
                f"center=({galaxy.fiducial_x0:.1f},{galaxy.fiducial_y0:.1f}) "
                f"maxsma={galaxy.maxsma:.1f}"
            )
            print(f"    building reference fit ({REFERENCE_ARM}) ...", flush=True)
            t0 = time.perf_counter()
            reference_fit = build_reference_fit(galaxy)
            reference_isophotes = reference_fit["results"].get("isophotes", [])
            print(
                f"    reference: {len(reference_isophotes)} isophotes in "
                f"{time.perf_counter() - t0:.1f}s"
            )
            references[spec.obj_id] = {
                "elapsed": reference_fit["elapsed"],
                "n_iso": len(reference_isophotes),
                "stop_codes": stop_code_histogram(reference_isophotes),
            }

            for arm in arms:
                if quick:
                    perturbations = quick_perturbations(galaxy)
                else:
                    perturbations = []
                    for axis in axes:
                        perturbations.extend(perturbations_for_axis(galaxy, axis))
                for pert in perturbations:
                    axis = pert["axis"]
                    value = pert["value"]
                    print(
                        f"    [{arm}] {axis}={value:+.2f} ...",
                        end="",
                        flush=True,
                    )
                    fit = run_fit(
                        galaxy,
                        arm=arm,
                        sma0=pert["sma0"],
                        eps=pert["eps"],
                        pa=pert["pa"],
                        x0=pert["x0"],
                        y0=pert["y0"],
                    )
                    row = row_for_fit(
                        galaxy, arm, axis, value, pert, fit, reference_isophotes
                    )
                    all_rows.append(row)
                    summary = (
                        f" ok ({fit['elapsed']:.1f}s, "
                        f"{row['n_iso_accepted']}/{row['n_iso_total']} iso, "
                        f"worst={row['worst_bin']})"
                    )
                    if not fit["success"]:
                        summary = f" FAIL ({fit['elapsed']:.1f}s: {fit['error']})"
                    print(summary)

    return {
        "rows": all_rows,
        "references": references,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def write_results_json(
    output_dir: Path,
    payload: Dict[str, Any],
    run_config: Dict[str, Any],
) -> Path:
    """Write the machine-readable results file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "results.json"
    body = {
        "environment": collect_environment_metadata(project_root=PROJECT_ROOT),
        "run_config": run_config,
        "references": payload["references"],
        "rows": payload["rows"],
    }
    write_json(out, body)
    return out


def _worst_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"good": 0, "warn": 0, "fail": 0}
    for row in rows:
        counts[row["worst_bin"]] = counts.get(row["worst_bin"], 0) + 1
    return counts


def write_report_md(
    output_dir: Path,
    payload: Dict[str, Any],
    run_config: Dict[str, Any],
) -> Path:
    """Write the human-readable REPORT.md rollup."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = payload["rows"]
    refs = payload["references"]

    lines: List[str] = []
    lines.append("# Robustness benchmark — report\n")
    lines.append(
        "One-dimensional perturbation sweep over ``(sma0, eps, pa, dx, dy)`` "
        "around per-galaxy fiducial starts, measuring how far the isophote "
        "sequence walks from a reference fit. See "
        "``docs/agent/journal/2026-04-15_robustness_plan.md`` for design.\n"
    )
    lines.append("## Run configuration\n")
    lines.append(f"- Tiers: `{run_config['tiers']}`")
    lines.append(f"- Arms: `{run_config['arms']}`")
    lines.append(f"- Axes: `{run_config['axes']}`")
    lines.append(f"- Galaxies filter: `{run_config['galaxies']}`")
    lines.append(f"- Quick mode: `{run_config['quick']}`")
    lines.append(
        f"- Total elapsed: {run_config['total_elapsed_seconds']:.1f} s\n"
    )

    lines.append("## Reference fits\n")
    if refs:
        lines.append("| galaxy | n_iso | elapsed (s) | stop codes |")
        lines.append("|---|---|---|---|")
        for obj_id, meta in refs.items():
            lines.append(
                f"| {obj_id} | {meta['n_iso']} | {meta['elapsed']:.2f} | "
                f"{meta['stop_codes']} |"
            )
    else:
        lines.append("(no reference fits were run)")
    lines.append("")

    lines.append("## Motion distribution per arm\n")
    if rows:
        arms_in_run = sorted({r["arm"] for r in rows})
        lines.append("| arm | n_rows | good | warn | fail |")
        lines.append("|---|---|---|---|---|")
        for arm in arms_in_run:
            arm_rows = [r for r in rows if r["arm"] == arm]
            counts = _worst_counts(arm_rows)
            lines.append(
                f"| {arm} | {len(arm_rows)} | {counts['good']} | "
                f"{counts['warn']} | {counts['fail']} |"
            )
    else:
        lines.append("(no sweep rows produced)")
    lines.append("")

    lines.append("## Per-row detail\n")
    if rows:
        lines.append(
            "| tier | galaxy | arm | axis | value | n_ok | rel_rms | rel_max | overlap | first_iso | worst |"
        )
        lines.append("|" + "---|" * 11)
        for row in rows:
            comp = row["comparison"]
            lines.append(
                f"| {row['tier']} | {row['obj_id']} | {row['arm']} | "
                f"{row['axis']} | {row['value']:+.2f} | "
                f"{row['n_iso_accepted']}/{row['n_iso_total']} | "
                f"{comp['profile_rel_rms']:.3f} | "
                f"{comp['profile_rel_max']:.3f} | "
                f"{comp['frac_overlap']:.2f} | "
                f"{'✗' if row['first_iso_failure'] else '✓'} | "
                f"{row['worst_bin']} |"
            )
    lines.append("")

    out = output_dir / "REPORT.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def print_summary_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("\n(no rows produced)")
        return
    print()
    print("  Summary")
    print(
        f"  {'tier':<10s} {'galaxy':<20s} {'arm':<10s} "
        f"{'axis':<6s} {'value':>7s} {'n_ok':>5s} {'rel_rms':>8s} "
        f"{'rel_max':>8s} {'worst':<6s}"
    )
    print("  " + "-" * 90)
    for r in rows:
        comp = r["comparison"]
        rel_rms = comp.get("profile_rel_rms", float("nan"))
        rel_max = comp.get("profile_rel_max", float("nan"))
        print(
            f"  {r['tier']:<10s} {r['obj_id']:<20s} {r['arm']:<10s} "
            f"{r['axis']:<6s} {r['value']:>7.2f} {r['n_iso_accepted']:>5d} "
            f"{rel_rms:>8.3f} {rel_max:>8.3f} {r['worst_bin']:<6s}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Robustness benchmark: isoster sensitivity to initial SMA and "
            "isophotal geometry. See docs/agent/journal/"
            "2026-04-15_robustness_plan.md."
        ),
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Fast smoke test: one mock galaxy, one arm, three sma0 factors.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output directory (default outputs/benchmark_robustness/).",
    )
    ap.add_argument(
        "--tiers",
        nargs="+",
        default=None,
        choices=list(datasets.TIERS),
        help="Restrict to these tiers (default: all).",
    )
    ap.add_argument(
        "--arms",
        nargs="+",
        default=None,
        choices=list(ARMS),
        help="Restrict to these fit arms (default: all three).",
    )
    ap.add_argument(
        "--axes",
        nargs="+",
        default=None,
        choices=list(AXES),
        help="Restrict to these perturbation axes (default: all five).",
    )
    ap.add_argument(
        "--galaxies",
        nargs="+",
        default=None,
        help="Restrict to these galaxy IDs within each tier.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.quick:
        tiers = ("mocks",)
        arms = ("bare",)
        axes = ("sma0",)
    else:
        tiers = tuple(args.tiers) if args.tiers else datasets.TIERS
        arms = tuple(args.arms) if args.arms else ARMS
        axes = tuple(args.axes) if args.axes else AXES

    output_root = args.output or (
        PROJECT_ROOT / "outputs" / "benchmark_robustness"
    )

    print("=" * 76)
    print("  Robustness benchmark")
    print(f"  Tiers:  {tiers}")
    print(f"  Arms:   {arms}")
    print(f"  Axes:   {axes}")
    print(f"  Quick:  {args.quick}")
    print(f"  Output: {output_root}")
    print("=" * 76)

    t_start = time.perf_counter()
    payload = run_sweep(
        tiers=tiers,
        arms=arms,
        axes=axes,
        galaxy_filter=args.galaxies,
        quick=args.quick,
    )
    total_elapsed = time.perf_counter() - t_start

    run_config = {
        "tiers": list(tiers),
        "arms": list(arms),
        "axes": list(axes),
        "galaxies": list(args.galaxies) if args.galaxies else None,
        "quick": bool(args.quick),
        "total_elapsed_seconds": total_elapsed,
    }
    json_path = write_results_json(output_root, payload, run_config)
    report_path = write_report_md(output_root, payload, run_config)

    print_summary_table(payload["rows"])
    print(f"\n  Total elapsed: {total_elapsed:.1f}s")
    print(f"  Results JSON:  {json_path}")
    print(f"  REPORT md:     {report_path}")


if __name__ == "__main__":
    main()
