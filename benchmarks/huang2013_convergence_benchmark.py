"""Huang2013 20-galaxy convergence benchmark.

Tests 4 convergence configurations (baseline + 3 candidates) across the
first 20 galaxies with mock2 data, comparing against photutils baselines.
No per-galaxy artifacts are saved — only aggregate analysis.

Usage:
    uv run python benchmarks/huang2013_convergence_benchmark.py

Output:
    outputs/huang2013_convergence_benchmark/
        summary_metrics.json
        summary_report.md
        comparison_figure.png
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure matplotlib cache dirs exist before import
if "XDG_CACHE_HOME" not in os.environ:
    xdg_cache_dir = PROJECT_ROOT / "outputs" / "tmp" / "xdg-cache"
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache_dir)
if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = PROJECT_ROOT / "outputs" / "tmp" / "mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

import matplotlib.pyplot as plt  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.table import Table  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from isoster import fit_image  # noqa: E402
from isoster.config import IsosterConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
HUANG_ROOT = Path("/Users/mac/work/hsc/huang2013")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "huang2013_convergence_benchmark"
MOCK_ID = 2

STOP_CODES = [0, 1, 2, 3, -1]

# The 20 galaxies (alphabetical first 20 with mock2 data)
GALAXY_NAMES = [
    "ESO185-G054", "ESO221-G026", "IC1459", "IC1633", "IC2006",
    "IC2311", "IC2597", "IC3370", "IC3896", "IC4296",
    "IC4329", "IC4742", "IC4765", "IC4797", "IC4889",
    "IC5328", "NGC1052", "NGC1172", "NGC1199", "NGC1209",
]

# Convergence params that must be explicitly reset (base config JSONs
# were saved before these params existed).
CONVERGENCE_DEFAULTS = {
    "convergence_scaling": "none",
    "geometry_damping": 1.0,
    "geometry_convergence": False,
    "geometry_tolerance": 0.01,
    "geometry_stable_iters": 3,
}

# 4 configurations: baseline + 3 candidates from NGC1209 benchmark
CONFIGURATIONS: list[tuple[str, dict[str, Any]]] = [
    ("Baseline", {
        "convergence_scaling": "none",
        "geometry_damping": 1.0,
        "geometry_convergence": False,
    }),
    ("A: sector_area", {
        "convergence_scaling": "sector_area",
        "geometry_damping": 1.0,
        "geometry_convergence": False,
    }),
    ("A+B", {
        "convergence_scaling": "sector_area",
        "geometry_damping": 0.7,
        "geometry_convergence": False,
    }),
    ("A+B+C", {
        "convergence_scaling": "sector_area",
        "geometry_damping": 0.7,
        "geometry_convergence": True,
    }),
]


# ---------------------------------------------------------------------------
# Data discovery and loading
# ---------------------------------------------------------------------------

def galaxy_paths(name: str) -> dict[str, Path]:
    """Return paths for a galaxy's mock2 data files."""
    root = HUANG_ROOT / name
    mock_dir = root / f"mock{MOCK_ID}"
    return {
        "image": root / f"{name}_mock{MOCK_ID}.fits",
        "phot_table": mock_dir / f"{name}_mock{MOCK_ID}_photutils_baseline_profile.fits",
        "config": mock_dir / f"{name}_mock{MOCK_ID}_isoster_baseline_run.json",
    }


def discover_galaxies() -> list[str]:
    """Validate all 20 galaxies have required mock2 files.

    Returns the list of valid galaxy names and prints warnings for missing ones.
    """
    valid = []
    for name in GALAXY_NAMES:
        paths = galaxy_paths(name)
        missing = [k for k, p in paths.items() if not p.exists()]
        if missing:
            print(f"  WARNING: {name} missing {missing}, skipping")
        else:
            valid.append(name)
    return valid


def load_image(path: Path) -> np.ndarray:
    """Load a 2-D image array from a FITS file."""
    with fits.open(path) as hdu_list:
        for hdu in hdu_list:
            if hdu.data is not None and hdu.data.ndim == 2:
                return hdu.data.astype(np.float64)
    raise ValueError(f"No 2-D image data found in {path}")


def load_base_config(path: Path) -> dict[str, Any]:
    """Load the isoster fit config from the saved run JSON."""
    with open(path) as f:
        return json.load(f)["fit_config"]


# ---------------------------------------------------------------------------
# Metric helpers (standalone, from convergence_diagnostic.py)
# ---------------------------------------------------------------------------

def stop_code_counts(isophotes: list[dict]) -> dict[int, int]:
    """Count isophotes by stop code."""
    codes = [iso["stop_code"] for iso in isophotes]
    return {sc: codes.count(sc) for sc in STOP_CODES}


def match_sma(
    isoster_isos: list[dict],
    phot_sma: np.ndarray,
    tolerance_frac: float = 0.02,
) -> list[tuple[dict, int]]:
    """Match isoster isophotes to photutils SMA values within tolerance."""
    matched = []
    iso_sma = np.array([iso["sma"] for iso in isoster_isos])
    for pi, ps in enumerate(phot_sma):
        diffs = np.abs(iso_sma - ps)
        best = np.argmin(diffs)
        if diffs[best] < tolerance_frac * ps:
            matched.append((isoster_isos[best], pi))
    return matched


def compute_metrics(
    matched: list[tuple[dict, int]],
    phot_table: Table,
) -> dict[str, Any]:
    """Compute accuracy metrics from matched isophote pairs.

    Only considers isophotes with stop_code == 0 from both sides.
    """
    if not matched:
        return {
            "n_matched": 0,
            "med_rel_intens": np.nan, "max_rel_intens": np.nan,
            "med_abs_eps": np.nan, "max_abs_eps": np.nan,
            "med_abs_pa": np.nan, "max_abs_pa": np.nan,
        }

    rel_intens, abs_eps, abs_pa = [], [], []
    for iso, pi in matched:
        if iso["stop_code"] != 0 or phot_table["stop_code"][pi] != 0:
            continue
        phot_intens = phot_table["intens"][pi]
        if np.abs(phot_intens) < 1e-10:
            continue

        rel_intens.append(
            np.abs(iso["intens"] - phot_intens) / np.abs(phot_intens)
        )
        abs_eps.append(np.abs(iso["eps"] - phot_table["eps"][pi]))
        dpa = iso["pa"] - phot_table["pa"][pi]
        dpa = np.abs((dpa + np.pi / 2) % np.pi - np.pi / 2)
        abs_pa.append(dpa)

    if not rel_intens:
        return {
            "n_matched": 0,
            "med_rel_intens": np.nan, "max_rel_intens": np.nan,
            "med_abs_eps": np.nan, "max_abs_eps": np.nan,
            "med_abs_pa": np.nan, "max_abs_pa": np.nan,
        }

    return {
        "n_matched": len(rel_intens),
        "med_rel_intens": float(np.median(rel_intens)),
        "max_rel_intens": float(np.max(rel_intens)),
        "med_abs_eps": float(np.median(abs_eps)),
        "max_abs_eps": float(np.max(abs_eps)),
        "med_abs_pa": float(np.median(abs_pa)),
        "max_abs_pa": float(np.max(abs_pa)),
    }


# ---------------------------------------------------------------------------
# Config building
# ---------------------------------------------------------------------------

def build_merged_config(base: dict, overrides: dict) -> dict[str, Any]:
    """Merge base config + convergence defaults + per-config overrides."""
    merged = {**base}
    # Reset all convergence params to explicit defaults first
    merged.update(CONVERGENCE_DEFAULTS)
    # Apply this config's specific overrides
    merged.update(overrides)
    return merged


# ---------------------------------------------------------------------------
# Single galaxy-config fit
# ---------------------------------------------------------------------------

def run_single(
    galaxy_name: str,
    config_label: str,
    image: np.ndarray,
    merged_config: dict[str, Any],
    phot_table: Table,
) -> dict[str, Any]:
    """Run a single isoster configuration on one galaxy.

    Returns a dict with stop counts, accuracy metrics, and timing.
    On failure, returns NaN metrics and zero counts.
    """
    try:
        cfg = IsosterConfig(**merged_config)
        t0 = time.perf_counter()
        results = fit_image(image, mask=None, config=cfg)
        elapsed = time.perf_counter() - t0

        isophotes = results["isophotes"]
        counts = stop_code_counts(isophotes)
        matched = match_sma(isophotes, np.array(phot_table["sma"]))
        metrics = compute_metrics(matched, phot_table)

        return {
            "galaxy": galaxy_name,
            "config": config_label,
            "wall_time": elapsed,
            "n_isophotes": len(isophotes),
            "stop_counts": counts,
            "metrics": metrics,
            "error": None,
        }
    except Exception as exc:
        return {
            "galaxy": galaxy_name,
            "config": config_label,
            "wall_time": np.nan,
            "n_isophotes": 0,
            "stop_counts": {sc: 0 for sc in STOP_CODES},
            "metrics": {
                "n_matched": 0,
                "med_rel_intens": np.nan, "max_rel_intens": np.nan,
                "med_abs_eps": np.nan, "max_abs_eps": np.nan,
                "med_abs_pa": np.nan, "max_abs_pa": np.nan,
            },
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(
    all_results: list[dict],
) -> dict[str, dict[str, Any]]:
    """Compute cross-galaxy summary statistics per configuration.

    Returns a dict keyed by config label with aggregate metrics.
    """
    config_labels = [label for label, _ in CONFIGURATIONS]
    agg = {}

    for label in config_labels:
        subset = [r for r in all_results if r["config"] == label]
        n_galaxies = len(subset)
        ok = [r for r in subset if r["error"] is None]
        failed = [r for r in subset if r["error"] is not None]

        # Stop=2 per galaxy
        stop2_per_galaxy = [r["stop_counts"].get(2, 0) for r in ok]
        total_stop2 = sum(stop2_per_galaxy)
        galaxies_with_stop2 = sum(1 for s in stop2_per_galaxy if s > 0)

        # Total isophotes and converged fraction
        total_iso = sum(r["n_isophotes"] for r in ok)
        total_sc0 = sum(r["stop_counts"].get(0, 0) for r in ok)
        converged_frac = total_sc0 / total_iso if total_iso > 0 else 0.0

        # Wall times
        wall_times = [r["wall_time"] for r in ok]
        total_time = sum(wall_times)
        mean_time = np.mean(wall_times) if wall_times else np.nan

        # Accuracy: median-of-medians and max-of-medians across galaxies
        med_di = [r["metrics"]["med_rel_intens"] for r in ok]
        med_eps = [r["metrics"]["med_abs_eps"] for r in ok]
        med_pa = [r["metrics"]["med_abs_pa"] for r in ok]

        agg[label] = {
            "n_galaxies": n_galaxies,
            "n_ok": len(ok),
            "n_failed": len(failed),
            "failed_galaxies": [r["galaxy"] for r in failed],
            # Stop=2
            "total_stop2": total_stop2,
            "mean_stop2": float(np.mean(stop2_per_galaxy)) if stop2_per_galaxy else 0.0,
            "median_stop2": float(np.median(stop2_per_galaxy)) if stop2_per_galaxy else 0.0,
            "max_stop2": max(stop2_per_galaxy) if stop2_per_galaxy else 0,
            "galaxies_with_stop2": galaxies_with_stop2,
            "stop2_per_galaxy": stop2_per_galaxy,
            # Converged fraction
            "total_isophotes": total_iso,
            "total_converged": total_sc0,
            "converged_fraction": converged_frac,
            # Wall time
            "total_time": total_time,
            "mean_time": float(mean_time),
            # Accuracy (median-of-medians, max-of-medians)
            "med_of_med_di": float(np.nanmedian(med_di)),
            "max_of_med_di": float(np.nanmax(med_di)) if med_di else np.nan,
            "med_of_med_eps": float(np.nanmedian(med_eps)),
            "max_of_med_eps": float(np.nanmax(med_eps)) if med_eps else np.nan,
            "med_of_med_pa_deg": float(np.degrees(np.nanmedian(med_pa))),
            "max_of_med_pa_deg": float(np.degrees(np.nanmax(med_pa))) if med_pa else np.nan,
            # Raw per-galaxy arrays for figure
            "med_di_values": med_di,
            "med_eps_values": med_eps,
            "med_pa_values": [float(np.degrees(v)) for v in med_pa],
        }

    return agg


def per_galaxy_improvement(
    all_results: list[dict],
    baseline_label: str = "Baseline",
) -> dict[str, dict[str, int]]:
    """Count galaxies improved / same / worse vs baseline stop=2 per config."""
    config_labels = [label for label, _ in CONFIGURATIONS if label != baseline_label]
    baseline_map = {
        r["galaxy"]: r["stop_counts"].get(2, 0)
        for r in all_results if r["config"] == baseline_label and r["error"] is None
    }

    improvement = {}
    for label in config_labels:
        improved, same, worse = 0, 0, 0
        for r in all_results:
            if r["config"] != label or r["error"] is not None:
                continue
            if r["galaxy"] not in baseline_map:
                continue
            bl_s2 = baseline_map[r["galaxy"]]
            this_s2 = r["stop_counts"].get(2, 0)
            if this_s2 < bl_s2:
                improved += 1
            elif this_s2 == bl_s2:
                same += 1
            else:
                worse += 1
        improvement[label] = {
            "improved": improved, "same": same, "worse": worse,
        }
    return improvement


# ---------------------------------------------------------------------------
# Stdout reporting
# ---------------------------------------------------------------------------

def print_stdout_table(agg: dict, improvement: dict) -> None:
    """Print a concise aggregate summary to stdout."""
    baseline_time = agg.get("Baseline", {}).get("total_time", 1.0)

    print("\n" + "=" * 90)
    print("AGGREGATE SUMMARY (20 galaxies × 4 configs)")
    print("=" * 90)

    header = (
        "| Config          | stop2_tot | gal_w/s2 | conv_frac | time(s) "
        "| speedup | med|dI/I| | med|deps| | med|dPA|° |"
    )
    sep = "|" + "|".join(["---"] * 10) + "|"
    print(header)
    print(sep)

    for label, _ in CONFIGURATIONS:
        a = agg[label]
        speedup = baseline_time / a["total_time"] if a["total_time"] > 0 else 0
        print(
            f"| {label:<15s} "
            f"| {a['total_stop2']:>9d} "
            f"| {a['galaxies_with_stop2']:>8d} "
            f"| {a['converged_fraction']:>9.3f} "
            f"| {a['total_time']:>7.1f} "
            f"| {speedup:>7.2f}x "
            f"| {a['med_of_med_di']:>9.4f} "
            f"| {a['med_of_med_eps']:>9.4f} "
            f"| {a['med_of_med_pa_deg']:>9.2f} |"
        )

    print("\nPer-galaxy improvement vs Baseline:")
    for label, counts in improvement.items():
        print(f"  {label}: improved={counts['improved']}, "
              f"same={counts['same']}, worse={counts['worse']}")
    print()


# ---------------------------------------------------------------------------
# Output: JSON
# ---------------------------------------------------------------------------

def clean_for_json(obj: Any) -> Any:
    """Recursively convert numpy types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if not np.isnan(obj) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    return obj


def save_summary_json(
    all_results: list[dict],
    agg: dict,
    improvement: dict,
    output_path: Path,
) -> None:
    """Save all raw per-galaxy-config metrics and aggregate stats to JSON."""
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_galaxies": len(GALAXY_NAMES),
        "n_configurations": len(CONFIGURATIONS),
        "per_galaxy_results": all_results,
        "aggregate": agg,
        "improvement_vs_baseline": improvement,
    }
    output_path.write_text(
        json.dumps(clean_for_json(payload), indent=2) + "\n"
    )
    print(f"Summary JSON: {output_path}")


# ---------------------------------------------------------------------------
# Output: markdown report
# ---------------------------------------------------------------------------

def save_summary_report(
    all_results: list[dict],
    agg: dict,
    improvement: dict,
    output_path: Path,
) -> None:
    """Generate a markdown summary report."""
    baseline_time = agg["Baseline"]["total_time"]

    lines = [
        "# Huang2013 20-Galaxy Convergence Benchmark Report",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}",
        f"**Galaxies**: {len(GALAXY_NAMES)}",
        f"**Configurations**: {len(CONFIGURATIONS)}",
        f"**Total fits**: {len(all_results)}",
        "",
        "## Configurations",
        "",
        "| Label | convergence_scaling | geometry_damping | geometry_convergence |",
        "|-------|---------------------|------------------|---------------------|",
    ]
    for label, overrides in CONFIGURATIONS:
        lines.append(
            f"| {label} "
            f"| {overrides.get('convergence_scaling', 'none')} "
            f"| {overrides.get('geometry_damping', 1.0)} "
            f"| {overrides.get('geometry_convergence', False)} |"
        )

    lines.extend([
        "",
        "## Aggregate Summary",
        "",
        "| Config | total stop=2 | galaxies w/ stop=2 | conv. frac "
        "| total time(s) | speedup | med\\|dI/I\\| | med\\|deps\\| | med\\|dPA\\|(deg) |",
        "|--------|-------------|-------------------|----------"
        "|---------------|---------|------------|------------|----------------|",
    ])

    for label, _ in CONFIGURATIONS:
        a = agg[label]
        speedup = baseline_time / a["total_time"] if a["total_time"] > 0 else 0
        lines.append(
            f"| {label} "
            f"| {a['total_stop2']} "
            f"| {a['galaxies_with_stop2']} "
            f"| {a['converged_fraction']:.3f} "
            f"| {a['total_time']:.1f} "
            f"| {speedup:.2f}x "
            f"| {a['med_of_med_di']:.4f} "
            f"| {a['med_of_med_eps']:.4f} "
            f"| {a['med_of_med_pa_deg']:.2f} |"
        )

    # Per-galaxy stop=2 table
    lines.extend([
        "",
        "## Per-Galaxy Stop=2 Counts",
        "",
    ])

    # Header row: galaxies as columns
    header_cells = ["Config"] + [g[:8] for g in GALAXY_NAMES]
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|" + "|".join(["---"] * (len(GALAXY_NAMES) + 1)) + "|")

    for label, _ in CONFIGURATIONS:
        cells = [label]
        for gname in GALAXY_NAMES:
            match = [
                r for r in all_results
                if r["config"] == label and r["galaxy"] == gname
            ]
            if match and match[0]["error"] is None:
                s2 = match[0]["stop_counts"].get(2, 0)
                cells.append(f"**{s2}**" if s2 > 0 else "0")
            else:
                cells.append("ERR")
        lines.append("| " + " | ".join(cells) + " |")

    # Improvement summary
    lines.extend([
        "",
        "## Improvement vs Baseline",
        "",
    ])
    for label, counts in improvement.items():
        total = counts["improved"] + counts["same"] + counts["worse"]
        lines.append(
            f"- **{label}**: {counts['improved']}/{total} improved, "
            f"{counts['same']}/{total} same, "
            f"{counts['worse']}/{total} worse"
        )

    # Failed galaxies
    any_failed = False
    for label, _ in CONFIGURATIONS:
        failed = agg[label].get("failed_galaxies", [])
        if failed:
            if not any_failed:
                lines.extend(["", "## Failed Fits", ""])
                any_failed = True
            lines.append(f"- **{label}**: {', '.join(failed)}")

    output_path.write_text("\n".join(lines) + "\n")
    print(f"Summary report: {output_path}")


# ---------------------------------------------------------------------------
# Output: 2×2 comparison figure
# ---------------------------------------------------------------------------

def save_comparison_figure(
    all_results: list[dict],
    agg: dict,
    output_path: Path,
) -> None:
    """Generate a 2×2 (16×14") comparison figure.

    Top-left: Stop=2 heatmap (4 configs × 20 galaxies).
    Top-right: Accuracy boxplots (|dI/I|, |deps|, |dPA|).
    Bottom-left: Wall time bars with speedup annotations.
    Bottom-right: Paired scatter (baseline vs best candidate stop=2).
    """
    config_labels = [label for label, _ in CONFIGURATIONS]
    n_configs = len(config_labels)
    n_galaxies = len(GALAXY_NAMES)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    ax_tl, ax_tr, ax_bl, ax_br = axes.flat

    # --- Build stop=2 matrix: rows=configs, cols=galaxies ---
    stop2_matrix = np.full((n_configs, n_galaxies), np.nan)
    for ci, label in enumerate(config_labels):
        for gi, gname in enumerate(GALAXY_NAMES):
            match = [
                r for r in all_results
                if r["config"] == label and r["galaxy"] == gname
                and r["error"] is None
            ]
            if match:
                stop2_matrix[ci, gi] = match[0]["stop_counts"].get(2, 0)

    # --- Top-left: Stop=2 heatmap ---
    im = ax_tl.imshow(
        stop2_matrix, aspect="auto", cmap="YlOrRd",
        norm=Normalize(vmin=0, vmax=max(np.nanmax(stop2_matrix), 1)),
    )
    ax_tl.set_xticks(range(n_galaxies))
    ax_tl.set_xticklabels(
        [g[:8] for g in GALAXY_NAMES], rotation=70, ha="right", fontsize=6,
    )
    ax_tl.set_yticks(range(n_configs))
    ax_tl.set_yticklabels(config_labels, fontsize=8)
    ax_tl.set_title("Stop=2 counts per galaxy", fontsize=11)
    fig.colorbar(im, ax=ax_tl, shrink=0.6, label="count")

    # Annotate cells where count > 0
    for ci in range(n_configs):
        for gi in range(n_galaxies):
            val = stop2_matrix[ci, gi]
            if not np.isnan(val) and val > 0:
                ax_tl.text(
                    gi, ci, f"{int(val)}", ha="center", va="center",
                    fontsize=6, fontweight="bold",
                    color="white" if val > np.nanmax(stop2_matrix) * 0.6 else "black",
                )

    # --- Top-right: Accuracy boxplots (3 stacked sub-axes) ---
    ax_tr.remove()
    # Use the top-right gridspec cell for 3 vertically stacked boxplots
    gs_inner = axes[0, 1].get_gridspec()
    gs_right = gs_inner[0, 1].subgridspec(3, 1, hspace=0.45)
    ax_di = fig.add_subplot(gs_right[0])
    ax_eps = fig.add_subplot(gs_right[1])
    ax_pa = fig.add_subplot(gs_right[2])

    config_colors = plt.cm.Set2(np.linspace(0, 0.8, n_configs))

    for metric_ax, metric_key, ylabel, title in [
        (ax_di, "med_di_values", "med |dI/I|", "|dI/I| distribution"),
        (ax_eps, "med_eps_values", "med |deps|", "|deps| distribution"),
        (ax_pa, "med_pa_values", "med |dPA| (deg)", "|dPA| distribution"),
    ]:
        box_data = []
        for label in config_labels:
            vals = agg[label][metric_key]
            # Filter NaN
            vals = [v for v in vals if not np.isnan(v)]
            box_data.append(vals)

        bp = metric_ax.boxplot(
            box_data, vert=True, patch_artist=True,
            tick_labels=[l[:7] for l in config_labels],
            widths=0.6,
        )
        for patch, color in zip(bp["boxes"], config_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        metric_ax.set_ylabel(ylabel, fontsize=8)
        metric_ax.set_title(title, fontsize=9)
        metric_ax.tick_params(axis="x", labelsize=7)
        metric_ax.tick_params(axis="y", labelsize=7)
        metric_ax.grid(True, alpha=0.3, axis="y")

    # --- Bottom-left: Wall time horizontal bars ---
    total_times = [agg[label]["total_time"] for label in config_labels]
    baseline_time = total_times[0]

    bars = ax_bl.barh(
        range(n_configs), total_times,
        color=config_colors, edgecolor="k", lw=0.5,
    )
    ax_bl.set_yticks(range(n_configs))
    ax_bl.set_yticklabels(config_labels, fontsize=9)
    ax_bl.set_xlabel("Total wall time (s)")
    ax_bl.set_title("Wall time across 20 galaxies")
    ax_bl.invert_yaxis()

    for bar, t in zip(bars, total_times):
        speedup = baseline_time / t if t > 0 else 0
        annotation = f"{t:.0f}s"
        if abs(speedup - 1.0) > 0.05:
            annotation += f" ({speedup:.2f}x)"
        ax_bl.text(
            bar.get_width() + max(total_times) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            annotation, va="center", fontsize=8,
        )

    # --- Bottom-right: Paired scatter (baseline vs best candidate stop=2) ---
    # Best candidate = config with lowest total stop=2 (excluding baseline)
    candidate_labels = [l for l in config_labels if l != "Baseline"]
    best_candidate = min(
        candidate_labels,
        key=lambda l: agg[l]["total_stop2"],
    )

    baseline_s2 = []
    candidate_s2 = []
    galaxy_labels = []

    for gi, gname in enumerate(GALAXY_NAMES):
        bl_match = [
            r for r in all_results
            if r["config"] == "Baseline" and r["galaxy"] == gname
            and r["error"] is None
        ]
        cand_match = [
            r for r in all_results
            if r["config"] == best_candidate and r["galaxy"] == gname
            and r["error"] is None
        ]
        if bl_match and cand_match:
            bl_val = bl_match[0]["stop_counts"].get(2, 0)
            cand_val = cand_match[0]["stop_counts"].get(2, 0)
            baseline_s2.append(bl_val)
            candidate_s2.append(cand_val)
            galaxy_labels.append(gname)

    baseline_s2 = np.array(baseline_s2)
    candidate_s2 = np.array(candidate_s2)

    # Color by improvement: green=improved, gray=same, red=worse
    scatter_colors = []
    for bl, cand in zip(baseline_s2, candidate_s2):
        if cand < bl:
            scatter_colors.append("tab:green")
        elif cand == bl:
            scatter_colors.append("tab:gray")
        else:
            scatter_colors.append("tab:red")

    max_val = max(np.max(baseline_s2), np.max(candidate_s2), 1)
    ax_br.plot([0, max_val], [0, max_val], "k--", lw=0.8, alpha=0.5, label="identity")
    ax_br.scatter(
        baseline_s2, candidate_s2,
        s=60, c=scatter_colors, edgecolors="k", lw=0.5, zorder=5,
    )

    # Label points with galaxy names for non-zero cases
    for i, gname in enumerate(galaxy_labels):
        if baseline_s2[i] > 0 or candidate_s2[i] > 0:
            ax_br.annotate(
                gname[:8], (baseline_s2[i], candidate_s2[i]),
                textcoords="offset points", xytext=(4, 4),
                fontsize=6, ha="left",
            )

    ax_br.set_xlabel("Baseline stop=2")
    ax_br.set_ylabel(f"{best_candidate} stop=2")
    ax_br.set_title(f"Paired: Baseline vs {best_candidate}")
    ax_br.set_aspect("equal")
    ax_br.set_xlim(-0.5, max_val + 1)
    ax_br.set_ylim(-0.5, max_val + 1)
    ax_br.grid(True, alpha=0.3)

    # Legend for colors
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:green",
               markersize=8, label="Improved"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:gray",
               markersize=8, label="Same"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:red",
               markersize=8, label="Worse"),
    ]
    ax_br.legend(handles=legend_elements, fontsize=7, loc="upper left")

    fig.suptitle(
        "Huang2013 20-Galaxy Convergence Benchmark",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison figure: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Huang2013 20-Galaxy Convergence Benchmark")
    print(f"  {len(GALAXY_NAMES)} galaxies × {len(CONFIGURATIONS)} configs "
          f"= {len(GALAXY_NAMES) * len(CONFIGURATIONS)} fits")
    print("=" * 70)

    # Discover and validate galaxies
    print("\nValidating galaxy data files ...")
    valid_galaxies = discover_galaxies()
    print(f"  {len(valid_galaxies)}/{len(GALAXY_NAMES)} galaxies ready")
    if not valid_galaxies:
        print("ERROR: No valid galaxies found. Exiting.")
        sys.exit(1)

    # Main loop: galaxy (outer) × config (inner)
    all_results = []
    total_start = time.perf_counter()

    for gi, gname in enumerate(valid_galaxies):
        paths = galaxy_paths(gname)
        print(f"\n[{gi + 1}/{len(valid_galaxies)}] {gname}")

        # Load once per galaxy
        image = load_image(paths["image"])
        phot_table = Table.read(paths["phot_table"])
        base_config = load_base_config(paths["config"])
        print(f"  image={image.shape}, phot={len(phot_table)} iso, "
              f"sma0={base_config['sma0']}")

        for label, overrides in CONFIGURATIONS:
            merged = build_merged_config(base_config, overrides)
            result = run_single(gname, label, image, merged, phot_table)

            status = "OK" if result["error"] is None else f"FAIL: {result['error'][:60]}"
            sc2 = result["stop_counts"].get(2, 0)
            t = result["wall_time"]
            t_str = f"{t:.2f}s" if not np.isnan(t) else "N/A"
            print(f"  {label:<15s}: {t_str}, sc2={sc2}, {status}")

            all_results.append(result)

    total_elapsed = time.perf_counter() - total_start
    print(f"\nTotal wall time: {total_elapsed:.1f}s")

    # Aggregate
    agg = aggregate_results(all_results)
    improvement = per_galaxy_improvement(all_results)

    # Stdout summary
    print_stdout_table(agg, improvement)

    # Save outputs
    save_summary_json(all_results, agg, improvement,
                      OUTPUT_DIR / "summary_metrics.json")
    save_summary_report(all_results, agg, improvement,
                        OUTPUT_DIR / "summary_report.md")
    save_comparison_figure(all_results, agg,
                           OUTPUT_DIR / "comparison_figure.png")

    # Final check
    expected_fits = len(valid_galaxies) * len(CONFIGURATIONS)
    actual_fits = len(all_results)
    failed_fits = sum(1 for r in all_results if r["error"] is not None)
    print(f"\nFits: {actual_fits}/{expected_fits} attempted, "
          f"{actual_fits - failed_fits} succeeded, {failed_fits} failed")
    print("Done.")


if __name__ == "__main__":
    main()
