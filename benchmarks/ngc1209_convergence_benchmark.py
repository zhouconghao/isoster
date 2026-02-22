"""NGC1209_mock2 convergence benchmark with full artifact output.

Tests 10 convergence configurations, saves per-config FITS profiles,
QA figures, and a comparison analysis with 4-panel summary figure.

Usage:
    uv run python benchmarks/ngc1209_convergence_benchmark.py
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

# Add huang2013 examples to path for shared utilities
HUANG_EXAMPLES = PROJECT_ROOT / "examples" / "huang2013"
if str(HUANG_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(HUANG_EXAMPLES))

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

from isoster.config import IsosterConfig  # noqa: E402

# Import shared Huang2013 utilities
from huang2013_shared import (  # noqa: E402
    build_method_qa_figure,
    build_model_image,
    dump_json,
    prepare_profile_table,
    read_mock_image,
    run_isoster_fit,
    run_with_runtime_profile,
    summarize_table,
    DEFAULT_PIXEL_SCALE_ARCSEC,
    DEFAULT_REDSHIFT,
    DEFAULT_ZEROPOINT,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
IMAGE_PATH = Path("/Users/mac/work/hsc/huang2013/NGC1209/NGC1209_mock2.fits")
PHOT_PATH = Path(
    "/Users/mac/work/hsc/huang2013/NGC1209/mock2/"
    "NGC1209_mock2_photutils_baseline_profile.fits"
)
CONFIG_PATH = Path(
    "/Users/mac/work/hsc/huang2013/NGC1209/mock2/"
    "NGC1209_mock2_isoster_baseline_run.json"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ngc1209_convergence_stop2_test"
PER_CONFIG_DIR = OUTPUT_DIR / "per_config"

GALAXY_NAME = "NGC1209"
MOCK_ID = 2
COG_SUBPIXELS = 32
OVERLAY_STEP = 10
QA_DPI = 150

STOP_CODES = [0, 1, 2, 3, -1]

# ---------------------------------------------------------------------------
# 10 convergence configurations
# ---------------------------------------------------------------------------
# Convergence params that must be explicitly reset (base config may lack them
# since they were added after the config JSON was saved).
CONVERGENCE_DEFAULTS = {
    "convergence_scaling": "none",
    "geometry_damping": 1.0,
    "geometry_convergence": False,
    "geometry_tolerance": 0.01,
    "geometry_stable_iters": 3,
}

CONFIGURATIONS: list[tuple[str, str, dict[str, Any]]] = [
    # (label, stem, overrides)
    ("Baseline", "baseline",
     {"convergence_scaling": "none", "geometry_damping": 1.0,
      "geometry_convergence": False}),
    ("A: sector_area", "sector_area",
     {"convergence_scaling": "sector_area", "geometry_damping": 1.0,
      "geometry_convergence": False}),
    ("A-alt: sqrt_sma", "sqrt_sma",
     {"convergence_scaling": "sqrt_sma", "geometry_damping": 1.0,
      "geometry_convergence": False}),
    ("B: damping=0.7", "damping_07",
     {"convergence_scaling": "none", "geometry_damping": 0.7,
      "geometry_convergence": False}),
    ("B-alt: damping=0.5", "damping_05",
     {"convergence_scaling": "none", "geometry_damping": 0.5,
      "geometry_convergence": False}),
    ("C: geom_conv", "geom_conv",
     {"convergence_scaling": "none", "geometry_damping": 1.0,
      "geometry_convergence": True}),
    ("A+B", "sector_damping_07",
     {"convergence_scaling": "sector_area", "geometry_damping": 0.7,
      "geometry_convergence": False}),
    ("A+C", "sector_geom_conv",
     {"convergence_scaling": "sector_area", "geometry_damping": 1.0,
      "geometry_convergence": True}),
    ("B+C", "damping_07_geom_conv",
     {"convergence_scaling": "none", "geometry_damping": 0.7,
      "geometry_convergence": True}),
    ("A+B+C", "all_three",
     {"convergence_scaling": "sector_area", "geometry_damping": 0.7,
      "geometry_convergence": True}),
]


# ---------------------------------------------------------------------------
# Metric helpers (standalone, adapted from convergence_diagnostic.py)
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
# Per-config pipeline
# ---------------------------------------------------------------------------

def load_base_config(path: Path) -> dict[str, Any]:
    """Load fit config from the saved run JSON."""
    with open(path) as f:
        return json.load(f)["fit_config"]


def build_merged_config(base: dict, overrides: dict) -> dict[str, Any]:
    """Merge base config + convergence defaults + per-config overrides."""
    merged = {**base}
    # Reset all convergence params to "off" defaults first
    merged.update(CONVERGENCE_DEFAULTS)
    # Apply this config's specific overrides
    merged.update(overrides)
    return merged


def run_single_config(
    label: str,
    stem: str,
    image: np.ndarray,
    merged_config: dict[str, Any],
    phot_table: Table,
) -> dict[str, Any]:
    """Run a single convergence config and save all artifacts.

    Returns a summary dict with metrics, stop counts, and timing.
    """
    print(f"\n  [{stem}] {label}")
    print(f"    convergence_scaling={merged_config['convergence_scaling']}, "
          f"geometry_damping={merged_config['geometry_damping']}, "
          f"geometry_convergence={merged_config['geometry_convergence']}")

    # 1. Fit
    (isophotes, validated_config), runtime_meta, _ = run_with_runtime_profile(
        run_isoster_fit, image, merged_config
    )
    wall_time = runtime_meta["wall_time_seconds"]
    print(f"    Fit: {len(isophotes)} isophotes in {wall_time:.2f}s")

    # 2. Profile table
    profile_table = prepare_profile_table(
        isophotes, image,
        redshift=DEFAULT_REDSHIFT,
        pixel_scale_arcsec=DEFAULT_PIXEL_SCALE_ARCSEC,
        zeropoint_mag=DEFAULT_ZEROPOINT,
        cog_subpixels=COG_SUBPIXELS,
        method_name="isoster",
    )

    # 3. Model image
    model_image = build_model_image(image.shape, profile_table, "isoster")

    # 4. QA figure
    qa_path = PER_CONFIG_DIR / f"{stem}_qa.png"
    build_method_qa_figure(
        image=image,
        profile_table=profile_table,
        model_image=model_image,
        output_path=qa_path,
        method_name="isoster",
        galaxy_name=f"{GALAXY_NAME} mock{MOCK_ID}",
        mock_id=MOCK_ID,
        pixel_scale_arcsec=DEFAULT_PIXEL_SCALE_ARCSEC,
        redshift=DEFAULT_REDSHIFT,
        runtime_metadata=runtime_meta,
        overlay_step=OVERLAY_STEP,
        dpi=QA_DPI,
    )
    print(f"    QA figure: {qa_path.name}")

    # 5. Save profile FITS and ECSV
    fits_path = PER_CONFIG_DIR / f"{stem}_profile.fits"
    ecsv_path = PER_CONFIG_DIR / f"{stem}_profile.ecsv"
    profile_table.write(fits_path, format="fits", overwrite=True)
    profile_table.write(ecsv_path, format="ascii.ecsv", overwrite=True)

    # 6. Save run JSON
    run_json_path = PER_CONFIG_DIR / f"{stem}_run.json"
    table_summary = summarize_table(profile_table)
    run_payload = {
        "label": label,
        "stem": stem,
        "fit_config": validated_config,
        "convergence_overrides": {
            k: merged_config[k] for k in CONVERGENCE_DEFAULTS
        },
        "runtime": runtime_meta,
        "table_summary": table_summary,
    }
    dump_json(run_json_path, run_payload)

    # 7. Compute comparison metrics vs photutils
    counts = stop_code_counts(isophotes)
    matched = match_sma(isophotes, np.array(phot_table["sma"]))
    metrics = compute_metrics(matched, phot_table)

    print(f"    Stop codes: sc0={counts[0]}, sc1={counts[1]}, "
          f"sc2={counts[2]}, sc3={counts[3]}, sc-1={counts[-1]}")
    print(f"    vs photutils: n_match={metrics['n_matched']}, "
          f"med_dI/I={metrics['med_rel_intens']:.4f}")

    return {
        "label": label,
        "stem": stem,
        "wall_time": wall_time,
        "n_isophotes": len(isophotes),
        "stop_counts": counts,
        "metrics": metrics,
        "table_summary": table_summary,
    }


# ---------------------------------------------------------------------------
# Summary reporting
# ---------------------------------------------------------------------------

def print_summary_table(all_results: list[dict]) -> None:
    """Print a Markdown summary table to stdout."""
    header = (
        "| Config | Time(s) | N_iso | sc=0 | sc=1 | sc=2 | sc=3 | sc=-1 "
        "| N_match | med_dI/I | max_dI/I | med_deps | max_deps "
        "| med_dPA | max_dPA |"
    )
    sep = "|" + "|".join(["---"] * 15) + "|"
    print("\n" + header)
    print(sep)

    for r in all_results:
        sc = r["stop_counts"]
        m = r["metrics"]
        row = (
            f"| {r['label']:<20s} "
            f"| {r['wall_time']:>6.2f} "
            f"| {r['n_isophotes']:>5d} "
            f"| {sc.get(0, 0):>4d} "
            f"| {sc.get(1, 0):>4d} "
            f"| {sc.get(2, 0):>4d} "
            f"| {sc.get(3, 0):>4d} "
            f"| {sc.get(-1, 0):>5d} "
            f"| {m['n_matched']:>7d} "
            f"| {m['med_rel_intens']:.4f} "
            f"| {m['max_rel_intens']:.4f} "
            f"| {m['med_abs_eps']:.4f} "
            f"| {m['max_abs_eps']:.4f} "
            f"| {np.degrees(m['med_abs_pa']):>6.2f} "
            f"| {np.degrees(m['max_abs_pa']):>6.2f} |"
        )
        print(row)
    print()


def save_summary_json(all_results: list[dict], output_path: Path) -> None:
    """Save all metrics to a JSON file."""
    # Convert numpy types for JSON serialization
    def clean(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        return obj

    dump_json(output_path, clean(all_results))
    print(f"Summary JSON: {output_path}")


def save_summary_report(all_results: list[dict], output_path: Path) -> None:
    """Generate a markdown summary report."""
    baseline = all_results[0]
    baseline_time = baseline["wall_time"]

    lines = [
        "# NGC1209 mock2 — Convergence Benchmark Report",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}",
        f"**Configurations tested**: {len(all_results)}",
        f"**Baseline wall time**: {baseline_time:.2f}s",
        f"**Baseline stop=2 count**: {baseline['stop_counts'].get(2, 0)}",
        "",
        "## Summary Table",
        "",
        "| Config | Time(s) | Speedup | N_iso | sc=0 | sc=2 "
        "| med \\|dI/I\\| | max \\|dI/I\\| | med \\|deps\\| | med \\|dPA\\|(deg) |",
        "|--------|---------|---------|-------|------|------"
        "|-------------|-------------|-------------|-----------------|",
    ]

    for r in all_results:
        sc = r["stop_counts"]
        m = r["metrics"]
        speedup = baseline_time / r["wall_time"] if r["wall_time"] > 0 else 0
        lines.append(
            f"| {r['label']:<20s} "
            f"| {r['wall_time']:>6.2f} "
            f"| {speedup:>6.2f}x "
            f"| {r['n_isophotes']:>5d} "
            f"| {sc.get(0, 0):>4d} "
            f"| {sc.get(2, 0):>4d} "
            f"| {m['med_rel_intens']:.4f} "
            f"| {m['max_rel_intens']:.4f} "
            f"| {m['med_abs_eps']:.4f} "
            f"| {np.degrees(m['med_abs_pa']):.2f} |"
        )

    # Find best config (lowest stop=2, then fastest)
    best = min(
        all_results,
        key=lambda r: (r["stop_counts"].get(2, 0), r["wall_time"]),
    )
    lines.extend([
        "",
        "## Key Findings",
        "",
        f"- **Best config**: {best['label']} "
        f"(sc2={best['stop_counts'].get(2, 0)}, "
        f"{best['wall_time']:.2f}s, "
        f"{baseline_time / best['wall_time']:.1f}x speedup)",
        "",
        "### Stop-code=2 elimination",
        "",
    ])

    for r in all_results:
        sc2 = r["stop_counts"].get(2, 0)
        marker = " **[ELIMINATED]**" if sc2 == 0 else ""
        lines.append(f"- {r['label']}: {sc2}{marker}")

    lines.extend([
        "",
        "## Output Files",
        "",
        "```",
        "per_config/",
    ])
    for _, stem, _ in CONFIGURATIONS:
        lines.append(
            f"  {stem}_profile.fits, {stem}_profile.ecsv, "
            f"{stem}_qa.png, {stem}_run.json"
        )
    lines.extend([
        "summary_metrics.json",
        "summary_report.md",
        "comparison_figure.png",
        "```",
    ])

    output_path.write_text("\n".join(lines) + "\n")
    print(f"Summary report: {output_path}")


# ---------------------------------------------------------------------------
# 4-panel comparison figure
# ---------------------------------------------------------------------------

def save_comparison_figure(
    all_results: list[dict],
    phot_table: Table,
    output_path: Path,
) -> None:
    """Generate a 4-panel comparison figure.

    Top-left: Horizontal bar chart of stop=2 counts.
    Top-right: Scatter of median |dI/I| vs stop=2 count.
    Bottom-left: Wall time bar chart with speedup annotations.
    Bottom-right: 1D SB profile overlay (photutils, baseline, best).
    """
    labels = [r["label"] for r in all_results]
    stop2 = [r["stop_counts"].get(2, 0) for r in all_results]
    med_di = [r["metrics"]["med_rel_intens"] for r in all_results]
    times = [r["wall_time"] for r in all_results]
    baseline_time = times[0]

    n = len(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, n))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_tl, ax_tr, ax_bl, ax_br = axes.flat

    # --- Top-left: stop=2 bar chart ---
    bars = ax_tl.barh(range(n), stop2, color=colors, edgecolor="k", lw=0.5)
    ax_tl.set_yticks(range(n))
    ax_tl.set_yticklabels(labels, fontsize=8)
    ax_tl.set_xlabel("Stop-code = 2 count")
    ax_tl.set_title("Max-iteration failures per config")
    ax_tl.invert_yaxis()
    for bar, val in zip(bars, stop2):
        ax_tl.text(
            bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontsize=9,
        )

    # --- Top-right: accuracy vs stop=2 scatter ---
    ax_tr.scatter(stop2, med_di, s=80, c=colors, edgecolors="k", lw=0.5,
                  zorder=5)
    for i, label in enumerate(labels):
        ax_tr.annotate(
            label, (stop2[i], med_di[i]),
            textcoords="offset points", xytext=(6, 4), fontsize=7, ha="left",
        )
    ax_tr.set_xlabel("Stop-code = 2 count")
    ax_tr.set_ylabel("Median |dI/I| vs photutils")
    ax_tr.set_title("Accuracy vs convergence failures")
    ax_tr.grid(True, alpha=0.3)

    # --- Bottom-left: wall time bar chart ---
    bars_t = ax_bl.barh(range(n), times, color=colors, edgecolor="k", lw=0.5)
    ax_bl.set_yticks(range(n))
    ax_bl.set_yticklabels(labels, fontsize=8)
    ax_bl.set_xlabel("Wall time (s)")
    ax_bl.set_title("Wall time per config")
    ax_bl.invert_yaxis()
    for bar, t in zip(bars_t, times):
        speedup = baseline_time / t if t > 0 else 0
        annotation = f"{t:.1f}s"
        if speedup > 1.05:
            annotation += f" ({speedup:.1f}x)"
        elif speedup < 0.95:
            annotation += f" ({speedup:.1f}x)"
        ax_bl.text(
            bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            annotation, va="center", fontsize=8,
        )

    # --- Bottom-right: 1D SB overlay ---
    # Load per-config profiles for baseline and best
    best_idx = min(
        range(n),
        key=lambda i: (all_results[i]["stop_counts"].get(2, 0),
                       all_results[i]["wall_time"]),
    )
    baseline_stem = CONFIGURATIONS[0][1]
    best_stem = CONFIGURATIONS[best_idx][1]

    baseline_prof = PER_CONFIG_DIR / f"{baseline_stem}_profile.ecsv"
    best_prof = PER_CONFIG_DIR / f"{best_stem}_profile.ecsv"

    # Photutils reference
    phot_sma = np.array(phot_table["sma"])
    phot_intens = np.array(phot_table["intens"])
    phot_mask = phot_intens > 0
    ax_br.plot(
        phot_sma[phot_mask] ** 0.25, phot_intens[phot_mask],
        "k--", lw=1.5, label="photutils", zorder=3,
    )

    # Baseline and best isoster profiles
    for prof_path, cfg_label, color, marker in [
        (baseline_prof, all_results[0]["label"], "C3", "o"),
        (best_prof, all_results[best_idx]["label"], "C0", "s"),
    ]:
        if prof_path.exists():
            tbl = Table.read(prof_path, format="ascii.ecsv")
            sma = np.array(tbl["sma"])
            intens = np.array(tbl["intens"])
            sc = np.array(tbl["stop_code"])
            good = (intens > 0) & (sc == 0)
            ax_br.scatter(
                sma[good] ** 0.25, intens[good],
                s=12, color=color, marker=marker, label=cfg_label,
                zorder=4, alpha=0.8,
            )

    ax_br.set_yscale("log")
    ax_br.set_xlabel("SMA$^{0.25}$ (pix)")
    ax_br.set_ylabel("Intensity (counts)")
    ax_br.set_title("1D SB profiles: photutils vs isoster configs")
    ax_br.legend(fontsize=8, loc="upper right")
    ax_br.grid(True, alpha=0.3)

    fig.suptitle(
        f"{GALAXY_NAME} mock{MOCK_ID} — Convergence strategy comparison",
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
    PER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"{GALAXY_NAME} mock{MOCK_ID} — Convergence Benchmark (10 configs)")
    print("=" * 70)

    # Load inputs
    print("\nLoading image ...", end=" ", flush=True)
    image, header = read_mock_image(IMAGE_PATH)
    print(f"shape={image.shape}, dtype={image.dtype}")

    print("Loading photutils baseline ...", end=" ", flush=True)
    phot_table = Table.read(PHOT_PATH)
    print(f"{len(phot_table)} isophotes, "
          f"SMA [{phot_table['sma'].min():.1f}, "
          f"{phot_table['sma'].max():.1f}]")

    print("Loading base config ...", end=" ", flush=True)
    base_config = load_base_config(CONFIG_PATH)
    print(f"sma0={base_config['sma0']}, maxsma={base_config['maxsma']:.1f}")

    # Run all 10 configurations
    print(f"\nRunning {len(CONFIGURATIONS)} configurations:")
    all_results = []
    for label, stem, overrides in CONFIGURATIONS:
        merged = build_merged_config(base_config, overrides)
        result = run_single_config(
            label, stem, image, merged, phot_table,
        )
        all_results.append(result)

    # Summary table to stdout
    print_summary_table(all_results)

    # Save artifacts
    save_summary_json(all_results, OUTPUT_DIR / "summary_metrics.json")
    save_summary_report(all_results, OUTPUT_DIR / "summary_report.md")
    save_comparison_figure(all_results, phot_table,
                           OUTPUT_DIR / "comparison_figure.png")

    # Final count check
    n_expected = len(CONFIGURATIONS) * 4
    n_files = len(list(PER_CONFIG_DIR.iterdir()))
    print(f"\nPer-config files: {n_files} / {n_expected} expected")
    print("Done.")


if __name__ == "__main__":
    main()
