#!/usr/bin/env python3
"""Robustness benchmark: isoster sensitivity to initial SMA and geometry.

Runs a 1-D perturbation sweep on ``(sma0, eps, pa)`` around a
per-galaxy fiducial start and records how far the resulting isophote
sequence walks from a reference fit. Four fit arms (``bare``, ``retry``,
``lsb``, ``ea_lsb``) measure how much ``max_retry_first_isophote``, the
LSB features, and eccentric-anomaly sampling each extend the capture
radius. The per-galaxy fiducial center ``(x0, y0)`` is treated as a
known user input and is never perturbed by the sweep.

Design document:
``docs/agent/journal/2026-04-15_robustness_plan.md``.

Outputs
-------
Every run writes a per-tier subtree under ``outputs/benchmark_robustness/``::

    outputs/benchmark_robustness/
    ├── SUMMARY.md                 # cross-tier headline rollup
    ├── mocks/
    │   ├── REPORT.md              # per-tier human-readable report
    │   ├── results.json           # machine-readable rows + references
    │   ├── _summary.csv           # flat one-row-per-fit CSV
    │   ├── sweep/{arm}/{obj_id}/{axis}_{value}.fits
    │   ├── reference/{obj_id}/{obj_id}_reference.fits
    │   └── figures/
    │       ├── profiles/{obj_id}/{arm}_{axis}.png
    │       └── outliers/{obj_id}/{arm}_{axis}_{value}_qa.png
    ├── huang2013/...
    ├── highorder/...
    └── hsc/...

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
import benchmarks.robustness.persist as persist  # noqa: E402
from benchmarks.utils.run_metadata import (  # noqa: E402
    collect_environment_metadata,
    write_json,
)
from isoster import fit_image  # noqa: E402
from isoster.config import IsosterConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Constants: arms, axes, grid
# ---------------------------------------------------------------------------

ARMS = ("bare", "retry", "lsb", "ea_lsb")
AXES = ("sma0", "eps", "pa")

SMA0_FACTORS: Sequence[float] = (
    0.25, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 3.0, 5.0,
)
EPS_VALUES: Sequence[float] = (0.00, 0.05, 0.15, 0.30, 0.50, 0.70)
PA_VALUES_RAD: Sequence[float] = (
    -np.pi / 2.0, -np.pi / 4.0, 0.0, np.pi / 6.0, np.pi / 4.0,
    np.pi / 3.0, np.pi / 2.0,
)

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

_LSB_DELTAS: Dict[str, Any] = dict(
    max_retry_first_isophote=5,
    lsb_auto_lock=True,
    lsb_auto_lock_maxgerr=0.3,
    lsb_auto_lock_debounce=2,
    lsb_auto_lock_integrator="median",
    use_outer_center_regularization=True,
    outer_reg_strength=2.0,
    outer_reg_sma_onset=50.0,
    outer_reg_sma_width=15.0,
)

ARM_DELTAS: Dict[str, Dict[str, Any]] = {
    "bare": dict(
        max_retry_first_isophote=0,
    ),
    "retry": dict(
        max_retry_first_isophote=5,
    ),
    "lsb": dict(_LSB_DELTAS),
    "ea_lsb": dict(_LSB_DELTAS, use_eccentric_anomaly=True),
}

# The reference fit uses the best shipped configuration (``lsb``) at the
# fiducial start. On mocks this is truth-informed; on real galaxies it is
# the LSB-sweep B_std equivalent. ``ea_lsb`` is the twin arm that also
# switches on eccentric-anomaly sampling for high-eps galaxies.
REFERENCE_ARM = "lsb"


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
    parameters (``sma0``, ``eps``, ``pa``, ``x0``, ``y0``). The center
    parameters ``(x0, y0)`` are always held at the per-galaxy fiducial
    — the sweep only perturbs ``sma0``, ``eps``, and ``pa``.
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
            rows.append({"axis": "sma0", "value": float(factor), **row})
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
        rows.append({"axis": "sma0", "value": float(factor), **row})
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


def _display_path(path: Path) -> str:
    """Return ``path`` relative to ``PROJECT_ROOT`` when possible, else absolute."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _reference_metadata(
    galaxy: datasets.GalaxyData,
    fit: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the header metadata dict for a reference fit FITS."""
    isos = fit["results"].get("isophotes", [])
    return {
        "tier": galaxy.spec.tier,
        "obj_id": galaxy.spec.obj_id,
        "description": galaxy.spec.description,
        "arm": REFERENCE_ARM,
        "kind": "reference",
        "axis": "fiducial",
        "value": 0.0,
        "start_sma0": float(galaxy.fiducial_sma0),
        "start_eps": float(galaxy.fiducial_eps),
        "start_pa": float(galaxy.fiducial_pa),
        "start_x0": float(galaxy.fiducial_x0),
        "start_y0": float(galaxy.fiducial_y0),
        "maxsma": float(galaxy.maxsma),
        "elapsed": float(fit["elapsed"]),
        "success": bool(fit["success"]),
        "error": fit.get("error") or "",
        "n_iso_total": len(isos),
        "n_iso_accepted": count_accepted(isos),
        "stop_code_hist": stop_code_histogram(isos),
    }


def _row_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    """Slim the row dict to header-friendly scalars (no nested comparison)."""
    return {
        "tier": row["tier"],
        "obj_id": row["obj_id"],
        "description": row["description"],
        "arm": row["arm"],
        "kind": "perturbed",
        "axis": row["axis"],
        "value": float(row["value"]),
        "start_sma0": float(row["start_sma0"]),
        "start_eps": float(row["start_eps"]),
        "start_pa": float(row["start_pa"]),
        "start_x0": float(row["start_x0"]),
        "start_y0": float(row["start_y0"]),
        "elapsed": float(row["elapsed"]),
        "success": bool(row["success"]),
        "error": row.get("error") or "",
        "n_iso_total": int(row["n_iso_total"]),
        "n_iso_accepted": int(row["n_iso_accepted"]),
        "first_iso_failure": bool(row["first_iso_failure"]),
        "first_iso_retry": int(row["first_iso_retry"]),
        "stop_code_hist": row["stop_code_hist"],
        "worst_bin": row["worst_bin"],
    }


def run_tier(
    tier: str,
    arms: Sequence[str],
    axes: Sequence[str],
    galaxy_filter: Optional[Sequence[str]],
    quick: bool,
    output_root: Path,
) -> Dict[str, Any]:
    """Run the sweep for a single tier and return its payload.

    Side effect: writes one FITS binary table per fit under
    ``output_root/{tier}/sweep/{arm}/{obj_id}/`` and one reference table
    per galaxy under ``output_root/{tier}/reference/{obj_id}/``.
    """
    tier_rows: List[Dict[str, Any]] = []
    references: Dict[str, Dict[str, Any]] = {}

    specs = datasets.list_galaxies(tier)
    if galaxy_filter:
        wanted = set(galaxy_filter)
        specs = [s for s in specs if s.obj_id in wanted]
    if not specs:
        print(f"  [tier {tier}] no galaxies (filter={galaxy_filter}) — skipping")
        return {"rows": tier_rows, "references": references}
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
        ref_path = persist.write_reference_fits(
            output_root,
            tier=tier,
            obj_id=spec.obj_id,
            isophotes=reference_isophotes,
            metadata=_reference_metadata(galaxy, reference_fit),
        )
        print(f"    reference FITS: {_display_path(ref_path)}")
        references[spec.obj_id] = {
            "elapsed": reference_fit["elapsed"],
            "n_iso": len(reference_isophotes),
            "stop_codes": stop_code_histogram(reference_isophotes),
            "fits_path": _display_path(ref_path),
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
                    galaxy,
                    arm,
                    axis,
                    value,
                    pert,
                    fit,
                    reference_isophotes,
                )
                row_isos = fit["results"].get("isophotes", [])
                row_path = persist.write_row_fits(
                    output_root,
                    tier=tier,
                    arm=arm,
                    obj_id=spec.obj_id,
                    axis=axis,
                    value=value,
                    isophotes=row_isos,
                    metadata=_row_metadata(row),
                )
                row["fits_path"] = _display_path(row_path)
                tier_rows.append(row)
                summary = (
                    f" ok ({fit['elapsed']:.1f}s, "
                    f"{row['n_iso_accepted']}/{row['n_iso_total']} iso, "
                    f"worst={row['worst_bin']})"
                )
                if not fit["success"]:
                    summary = f" FAIL ({fit['elapsed']:.1f}s: {fit['error']})"
                print(summary)

    return {
        "rows": tier_rows,
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


def write_summary_csv(output_dir: Path, payload: Dict[str, Any]) -> Path:
    """Write a flat one-row-per-fit CSV for pandas / spreadsheet use.

    Columns carry the primary scalar metrics plus the binned worst label
    and the relative path of the corresponding isophote FITS. Fields
    match the header cards written by ``persist.write_row_fits``.
    """
    import csv

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "_summary.csv"
    rows = payload["rows"]
    fieldnames = [
        "tier",
        "obj_id",
        "arm",
        "axis",
        "value",
        "start_sma0",
        "start_eps",
        "start_pa",
        "start_x0",
        "start_y0",
        "success",
        "elapsed",
        "n_iso_total",
        "n_iso_accepted",
        "first_iso_failure",
        "first_iso_retry",
        "stop_code_hist",
        "n_ref",
        "n_pert",
        "n_overlap",
        "frac_overlap",
        "profile_rel_rms",
        "profile_rel_max",
        "n_low_signal",
        "eps_mad",
        "pa_mad_deg",
        "center_max_drift",
        "worst_bin",
        "fits_path",
    ]
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            comp = row.get("comparison", {}) or {}
            writer.writerow(
                {
                    "tier": row["tier"],
                    "obj_id": row["obj_id"],
                    "arm": row["arm"],
                    "axis": row["axis"],
                    "value": row["value"],
                    "start_sma0": row["start_sma0"],
                    "start_eps": row["start_eps"],
                    "start_pa": row["start_pa"],
                    "start_x0": row["start_x0"],
                    "start_y0": row["start_y0"],
                    "success": row["success"],
                    "elapsed": row["elapsed"],
                    "n_iso_total": row["n_iso_total"],
                    "n_iso_accepted": row["n_iso_accepted"],
                    "first_iso_failure": row["first_iso_failure"],
                    "first_iso_retry": row["first_iso_retry"],
                    "stop_code_hist": row["stop_code_hist"],
                    "n_ref": comp.get("n_ref"),
                    "n_pert": comp.get("n_pert"),
                    "n_overlap": comp.get("n_overlap"),
                    "frac_overlap": comp.get("frac_overlap"),
                    "profile_rel_rms": comp.get("profile_rel_rms"),
                    "profile_rel_max": comp.get("profile_rel_max"),
                    "n_low_signal": comp.get("n_low_signal"),
                    "eps_mad": comp.get("eps_mad"),
                    "pa_mad_deg": comp.get("pa_mad_deg"),
                    "center_max_drift": comp.get("center_max_drift"),
                    "worst_bin": row["worst_bin"],
                    "fits_path": row.get("fits_path", ""),
                }
            )
    return out


def _worst_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"good": 0, "warn": 0, "fail": 0}
    for row in rows:
        counts[row["worst_bin"]] = counts.get(row["worst_bin"], 0) + 1
    return counts


def _finite_values(rows: List[Dict[str, Any]], field: str) -> np.ndarray:
    """Extract a finite numpy array for a metric field across ``rows``."""
    vals: List[float] = []
    for r in rows:
        comp = r.get("comparison") or {}
        v = comp.get(field)
        if v is None:
            continue
        fv = float(v)
        if np.isfinite(fv):
            vals.append(fv)
    return np.asarray(vals, dtype=np.float64)


def _fmt_stats(arr: np.ndarray, fmt: str = "{:.3f}") -> str:
    """Format min / median / p90 / p99 / max as a single pipe-joined cell."""
    if arr.size == 0:
        return "n/a"
    return " / ".join(
        fmt.format(v)
        for v in (
            np.min(arr),
            np.median(arr),
            np.percentile(arr, 90),
            np.percentile(arr, 99),
            np.max(arr),
        )
    )


def write_report_md(
    output_dir: Path,
    payload: Dict[str, Any],
    run_config: Dict[str, Any],
) -> Path:
    """Write the human-readable per-tier REPORT.md rollup.

    The report leads with distributions of the four primary scalar
    metrics (``profile_rel_rms``, ``eps_mad``, ``pa_mad_deg``,
    ``center_max_drift``) per (galaxy, arm, axis). Characterization bins
    are relegated to a footnote — they are reporting color, not the
    headline. Top-K outliers per (galaxy, arm) are listed with their
    exact perturbation and metric values so they are easy to jump to.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = payload["rows"]
    refs = payload["references"]
    tier = run_config.get("tier", "?")

    lines: List[str] = []
    lines.append(f"# Robustness benchmark — tier `{tier}`")
    lines.append("")
    lines.append(
        "One-dimensional perturbation sweep over ``(sma0, eps, pa)``"
        " around per-galaxy fiducial starts, measuring how far the isophote"
        " sequence walks from a reference fit. The fiducial center"
        " ``(x0, y0)`` is treated as known user input and is not perturbed."
        " Reference arm: "
        f"``{REFERENCE_ARM}`` at the fiducial start. Design doc:"
        " ``docs/agent/journal/2026-04-15_robustness_plan.md``."
    )
    lines.append("")
    lines.append(
        "Stat cells are formatted as ``min / median / p90 / p99 / max``"
        " across the perturbation values on the axis. ``profile_rel_rms``"
        " is dimensionless (relative-intensity deviation); ``eps_mad``"
        " is dimensionless; ``pa_mad_deg`` is degrees; ``center_max_drift``"
        " is pixels. Rows with zero overlap against the reference fit"
        " contribute NaN and are excluded from the distributions."
    )
    lines.append("")

    lines.append("## Run configuration")
    lines.append("")
    lines.append(f"- Tier: `{run_config['tier']}`")
    lines.append(f"- Arms: `{run_config['arms']}`")
    lines.append(f"- Axes: `{run_config['axes']}`")
    lines.append(f"- Galaxies filter: `{run_config['galaxies']}`")
    lines.append(f"- Quick mode: `{run_config['quick']}`")
    lines.append(
        f"- Total elapsed: {run_config['total_elapsed_seconds']:.1f} s"
    )
    lines.append("")

    lines.append("## Reference fits")
    lines.append("")
    if refs:
        lines.append("| galaxy | n_iso | elapsed (s) | stop codes |")
        lines.append("|" + "---|" * 4)
        for obj_id, meta in refs.items():
            lines.append(
                f"| {obj_id} | {meta['n_iso']} | {meta['elapsed']:.2f} |"
                f" {meta['stop_codes']} |"
            )
    else:
        lines.append("(no reference fits were run)")
    lines.append("")

    galaxies = sorted({r["obj_id"] for r in rows})
    arms_in_run = [a for a in run_config["arms"] if any(r["arm"] == a for r in rows)]
    axes_in_run = [a for a in run_config["axes"] if any(r["axis"] == a for r in rows)]

    # ------------------------------------------------------------------
    # Section: quantitative summary per (galaxy × arm)
    # ------------------------------------------------------------------
    lines.append("## Quantitative summary per (galaxy × arm)")
    lines.append("")
    lines.append(
        "Aggregates every perturbation on every axis for the given"
        " (galaxy, arm). This is the headline number for 'how much does"
        " the fit move when you bump sma0, eps, or pa'."
    )
    lines.append("")
    for galaxy in galaxies:
        lines.append(f"### {galaxy}")
        lines.append("")
        lines.append(
            "| arm | n | profile_rel_rms (min/med/p90/p99/max)"
            " | eps_mad | pa_mad (deg) | center_drift (pix) |"
        )
        lines.append("|" + "---|" * 6)
        for arm in arms_in_run:
            arm_rows = [
                r for r in rows if r["obj_id"] == galaxy and r["arm"] == arm
            ]
            if not arm_rows:
                continue
            rel = _finite_values(arm_rows, "profile_rel_rms")
            eps = _finite_values(arm_rows, "eps_mad")
            pa = _finite_values(arm_rows, "pa_mad_deg")
            drift = _finite_values(arm_rows, "center_max_drift")
            lines.append(
                f"| {arm} | {len(arm_rows)} | {_fmt_stats(rel)}"
                f" | {_fmt_stats(eps)} | {_fmt_stats(pa, '{:.2f}')}"
                f" | {_fmt_stats(drift, '{:.2f}')} |"
            )
        lines.append("")

    # ------------------------------------------------------------------
    # Section: per-axis distributions
    # ------------------------------------------------------------------
    lines.append("## Per-axis distributions")
    lines.append("")
    lines.append(
        "How each individual axis (`sma0`, `eps`, `pa`) widens or tightens"
        " the fit's capture radius. Reading: if the ``profile_rel_rms``"
        " median for ``pa`` is 0.10 but the ``sma0`` median is 0.30, then"
        " a wrong ``sma0`` guess moves the fit more than a wrong PA."
    )
    lines.append("")
    for galaxy in galaxies:
        for arm in arms_in_run:
            arm_rows = [
                r for r in rows if r["obj_id"] == galaxy and r["arm"] == arm
            ]
            if not arm_rows:
                continue
            lines.append(f"### {galaxy} — {arm}")
            lines.append("")
            lines.append(
                "| axis | n | profile_rel_rms (min/med/p90/p99/max)"
                " | eps_mad | pa_mad (deg) | center_drift (pix) |"
            )
            lines.append("|" + "---|" * 6)
            for axis in axes_in_run:
                axis_rows = [r for r in arm_rows if r["axis"] == axis]
                if not axis_rows:
                    continue
                rel = _finite_values(axis_rows, "profile_rel_rms")
                eps = _finite_values(axis_rows, "eps_mad")
                pa = _finite_values(axis_rows, "pa_mad_deg")
                drift = _finite_values(axis_rows, "center_max_drift")
                lines.append(
                    f"| {axis} | {len(axis_rows)} | {_fmt_stats(rel)}"
                    f" | {_fmt_stats(eps)} | {_fmt_stats(pa, '{:.2f}')}"
                    f" | {_fmt_stats(drift, '{:.2f}')} |"
                )
            lines.append("")

    # ------------------------------------------------------------------
    # Section: top outliers per (galaxy × arm)
    # ------------------------------------------------------------------
    lines.append("## Top outliers per (galaxy × arm)")
    lines.append("")
    lines.append(
        "Top-5 rows by ``profile_rel_rms`` within each (galaxy, arm)."
        " These are the perturbations that walk the fit the farthest"
        " from the reference; they are the first targets for QA figures."
    )
    lines.append("")
    for galaxy in galaxies:
        for arm in arms_in_run:
            arm_rows = [
                r for r in rows if r["obj_id"] == galaxy and r["arm"] == arm
            ]
            if not arm_rows:
                continue
            scored = sorted(
                arm_rows,
                key=lambda r: (
                    -float((r.get("comparison") or {}).get("profile_rel_rms", -np.inf)),
                ),
            )[:5]
            lines.append(f"### {galaxy} — {arm}")
            lines.append("")
            lines.append(
                "| axis | value | rel_rms | rel_max | eps_mad | pa_mad (deg) |"
                " center_drift | n_ok | first_iso_fail |"
            )
            lines.append("|" + "---|" * 9)
            for row in scored:
                comp = row.get("comparison") or {}
                lines.append(
                    f"| {row['axis']} | {row['value']:+.2f} |"
                    f" {comp.get('profile_rel_rms', float('nan')):.3f} |"
                    f" {comp.get('profile_rel_max', float('nan')):.3f} |"
                    f" {comp.get('eps_mad', float('nan')):.3f} |"
                    f" {comp.get('pa_mad_deg', float('nan')):.2f} |"
                    f" {comp.get('center_max_drift', float('nan')):.2f} |"
                    f" {row['n_iso_accepted']}/{row['n_iso_total']} |"
                    f" {'✗' if row['first_iso_failure'] else '✓'} |"
                )
            lines.append("")

    # ------------------------------------------------------------------
    # Section: axis-walk detail tables
    # ------------------------------------------------------------------
    lines.append("## Axis-walk detail per (galaxy × arm × axis)")
    lines.append("")
    lines.append(
        "One row per perturbation value. Useful for spotting monotonic"
        " vs. non-monotonic responses as you walk the axis away from the"
        " fiducial. ``n_ok`` is ``accepted / total`` isophotes; ``first_iso``"
        " marks rows where the initial isophote failed."
    )
    lines.append("")
    for galaxy in galaxies:
        for arm in arms_in_run:
            for axis in axes_in_run:
                axis_rows = sorted(
                    [
                        r
                        for r in rows
                        if r["obj_id"] == galaxy
                        and r["arm"] == arm
                        and r["axis"] == axis
                    ],
                    key=lambda r: float(r["value"]),
                )
                if not axis_rows:
                    continue
                lines.append(f"### {galaxy} — {arm} — {axis}")
                lines.append("")
                lines.append(
                    "| value | n_ok | rel_rms | rel_max | eps_mad |"
                    " pa_mad (deg) | center_drift | overlap | first_iso |"
                )
                lines.append("|" + "---|" * 9)
                for row in axis_rows:
                    comp = row.get("comparison") or {}
                    lines.append(
                        f"| {row['value']:+.2f} |"
                        f" {row['n_iso_accepted']}/{row['n_iso_total']} |"
                        f" {comp.get('profile_rel_rms', float('nan')):.3f} |"
                        f" {comp.get('profile_rel_max', float('nan')):.3f} |"
                        f" {comp.get('eps_mad', float('nan')):.3f} |"
                        f" {comp.get('pa_mad_deg', float('nan')):.2f} |"
                        f" {comp.get('center_max_drift', float('nan')):.2f} |"
                        f" {comp.get('frac_overlap', float('nan')):.2f} |"
                        f" {'✗' if row['first_iso_failure'] else '✓'} |"
                    )
                lines.append("")

    # ------------------------------------------------------------------
    # Footnote: characterization bins (not the headline)
    # ------------------------------------------------------------------
    lines.append("## Appendix — characterization bins")
    lines.append("")
    lines.append(
        "Counts of the ``good / warn / fail`` bins defined in the design"
        " doc §8. These are reporting colors, not gates; distributions"
        " above are the primary signal."
    )
    lines.append("")
    if rows:
        lines.append("| galaxy | arm | n_rows | good | warn | fail |")
        lines.append("|---|---|---|---|---|---|")
        for galaxy in galaxies:
            for arm in arms_in_run:
                arm_rows = [
                    r for r in rows if r["obj_id"] == galaxy and r["arm"] == arm
                ]
                if not arm_rows:
                    continue
                counts = _worst_counts(arm_rows)
                lines.append(
                    f"| {galaxy} | {arm} | {len(arm_rows)}"
                    f" | {counts['good']} | {counts['warn']} | {counts['fail']} |"
                )
    else:
        lines.append("(no sweep rows produced)")
    lines.append("")

    out = output_dir / "REPORT.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def write_parent_summary_md(output_root: Path) -> Path:
    """Aggregate per-tier results into a cross-tier headline rollup.

    Walks every ``{output_root}/{tier}/results.json`` sibling and
    produces ``{output_root}/SUMMARY.md`` containing a row per
    (tier, galaxy, arm) with the four primary metrics summarized as
    ``median / p90 / p99 / max``. Tiers whose ``results.json`` is
    missing are listed as ``(not yet run)``. This file is rebuilt from
    scratch on every call so it always reflects the current disk state.
    """
    import json

    output_root.mkdir(parents=True, exist_ok=True)
    tier_dirs = sorted(
        p for p in output_root.iterdir() if p.is_dir() and p.name in datasets.TIERS
    )

    lines: List[str] = []
    lines.append("# Robustness benchmark — cross-tier summary")
    lines.append("")
    lines.append(
        "One row per (tier, galaxy, arm) showing how far the sweep walked"
        " the fit from the reference across all perturbations. Stat cells"
        " are ``median / p90 / p99 / max``. Follow the ``REPORT.md`` link"
        " for the tier-level per-axis breakdown and figures."
    )
    lines.append("")
    lines.append("| tier | report | galaxies | fits | total elapsed (s) |")
    lines.append("|---|---|---|---|---|")
    for tier in datasets.TIERS:
        tier_dir = output_root / tier
        results = tier_dir / "results.json"
        if not results.exists():
            lines.append(f"| {tier} | (not yet run) | — | — | — |")
            continue
        body = json.loads(results.read_text())
        refs = body.get("references", {})
        rows = body.get("rows", [])
        rc = body.get("run_config", {})
        elapsed = float(rc.get("total_elapsed_seconds", 0.0))
        lines.append(
            f"| {tier} | [`{tier}/REPORT.md`]({tier}/REPORT.md) |"
            f" {len(refs)} | {len(rows)} | {elapsed:.1f} |"
        )
    lines.append("")

    for tier_dir in tier_dirs:
        results = tier_dir / "results.json"
        if not results.exists():
            continue
        body = json.loads(results.read_text())
        rows = body.get("rows", [])
        if not rows:
            continue
        tier = tier_dir.name
        lines.append(f"## {tier}")
        lines.append("")
        galaxies = sorted({r["obj_id"] for r in rows})
        arms_in = []
        seen_arms: set = set()
        for r in rows:
            if r["arm"] not in seen_arms:
                arms_in.append(r["arm"])
                seen_arms.add(r["arm"])
        lines.append(
            "| galaxy | arm | n | profile_rel_rms | eps_mad |"
            " pa_mad (deg) | center_drift (pix) |"
        )
        lines.append("|" + "---|" * 7)
        for galaxy in galaxies:
            for arm in arms_in:
                arm_rows = [
                    r for r in rows if r["obj_id"] == galaxy and r["arm"] == arm
                ]
                if not arm_rows:
                    continue
                rel = _finite_values(arm_rows, "profile_rel_rms")
                eps = _finite_values(arm_rows, "eps_mad")
                pa = _finite_values(arm_rows, "pa_mad_deg")
                drift = _finite_values(arm_rows, "center_max_drift")
                lines.append(
                    f"| {galaxy} | {arm} | {len(arm_rows)}"
                    f" | {_fmt_short(rel)}"
                    f" | {_fmt_short(eps)}"
                    f" | {_fmt_short(pa, '{:.2f}')}"
                    f" | {_fmt_short(drift, '{:.2f}')} |"
                )
        lines.append("")

    out = output_root / "SUMMARY.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def _fmt_short(arr: np.ndarray, fmt: str = "{:.3f}") -> str:
    """Compact stat cell: ``median / p90 / p99 / max`` (parent summary)."""
    if arr.size == 0:
        return "n/a"
    return " / ".join(
        fmt.format(v)
        for v in (
            np.median(arr),
            np.percentile(arr, 90),
            np.percentile(arr, 99),
            np.max(arr),
        )
    )


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
        help="Restrict to these perturbation axes (default: all three).",
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

    all_rows: List[Dict[str, Any]] = []
    total_start = time.perf_counter()
    for tier in tiers:
        print(f"\n[tier: {tier}]")
        tier_dir = output_root / tier
        t_start = time.perf_counter()
        payload = run_tier(
            tier=tier,
            arms=arms,
            axes=axes,
            galaxy_filter=args.galaxies,
            quick=args.quick,
            output_root=output_root,
        )
        tier_elapsed = time.perf_counter() - t_start

        run_config = {
            "tier": tier,
            "arms": list(arms),
            "axes": list(axes),
            "galaxies": list(args.galaxies) if args.galaxies else None,
            "quick": bool(args.quick),
            "total_elapsed_seconds": tier_elapsed,
        }
        json_path = write_results_json(tier_dir, payload, run_config)
        report_path = write_report_md(tier_dir, payload, run_config)
        summary_csv_path = write_summary_csv(tier_dir, payload)
        all_rows.extend(payload["rows"])
        print(f"  [{tier}] elapsed: {tier_elapsed:.1f}s")
        print(f"  [{tier}] results JSON: {_display_path(json_path)}")
        print(f"  [{tier}] REPORT.md:    {_display_path(report_path)}")
        print(f"  [{tier}] summary CSV:  {_display_path(summary_csv_path)}")

    total_elapsed = time.perf_counter() - total_start
    parent_summary = write_parent_summary_md(output_root)

    print_summary_table(all_rows)
    print(f"\n  Total elapsed: {total_elapsed:.1f}s")
    print(f"  Parent SUMMARY: {_display_path(parent_summary)}")


if __name__ == "__main__":
    main()
