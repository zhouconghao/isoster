"""Convergence diagnostic benchmark for NGC1209_mock2.

Compares 7 convergence configurations against photutils baseline,
reporting stop-code distributions, wall-clock times, and profile
accuracy metrics.

Usage:
    uv run python benchmarks/convergence_diagnostic.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

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

from isoster import fit_image  # noqa: E402
from isoster.config import IsosterConfig  # noqa: E402

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
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Stop codes to track
STOP_CODES = [0, 1, 2, 3, -1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image(path: Path) -> np.ndarray:
    """Load the 2-D image array from a FITS file."""
    with fits.open(path) as hdu_list:
        for hdu in hdu_list:
            if hdu.data is not None and hdu.data.ndim == 2:
                return hdu.data.astype(np.float64)
    raise ValueError(f"No 2-D image data found in {path}")


def load_photutils_table(path: Path) -> Table:
    """Load the photutils baseline profile table."""
    return Table.read(path)


def load_base_config(path: Path) -> dict:
    """Load the base fit config dict from the run JSON."""
    with open(path) as f:
        run_data = json.load(f)
    return run_data["fit_config"]


def build_config(base: dict, overrides: dict) -> IsosterConfig:
    """Build an IsosterConfig from the base dict with overrides applied."""
    merged = {**base, **overrides}
    return IsosterConfig(**merged)


def stop_code_counts(isophotes: list[dict]) -> dict[int, int]:
    """Count isophotes by stop code."""
    codes = [iso["stop_code"] for iso in isophotes]
    return {sc: codes.count(sc) for sc in STOP_CODES}


def match_sma(isoster_isos: list[dict], phot_sma: np.ndarray,
              tolerance_frac: float = 0.02) -> list[tuple[dict, int]]:
    """Match isoster isophotes to photutils SMA values within tolerance.

    Returns list of (isoster_isophote, phot_index) pairs.
    """
    matched = []
    iso_sma = np.array([iso["sma"] for iso in isoster_isos])
    for pi, ps in enumerate(phot_sma):
        diffs = np.abs(iso_sma - ps)
        best = np.argmin(diffs)
        if diffs[best] < tolerance_frac * ps:
            matched.append((isoster_isos[best], pi))
    return matched


def compute_metrics(matched: list[tuple[dict, int]],
                    phot_table: Table) -> dict:
    """Compute accuracy metrics from matched isophote pairs.

    Ignores isophotes with stop_code != 0 for both isoster and photutils.
    """
    if not matched:
        return {
            "n_matched": 0,
            "med_rel_intens": np.nan, "max_rel_intens": np.nan,
            "med_abs_eps": np.nan, "max_abs_eps": np.nan,
            "med_abs_pa": np.nan, "max_abs_pa": np.nan,
        }

    rel_intens = []
    abs_eps = []
    abs_pa = []

    for iso, pi in matched:
        # Skip non-converged isophotes from either side
        if iso["stop_code"] != 0:
            continue
        if phot_table["stop_code"][pi] != 0:
            continue

        phot_intens = phot_table["intens"][pi]
        if np.abs(phot_intens) < 1e-10:
            continue

        rel_intens.append(np.abs(iso["intens"] - phot_intens) / np.abs(phot_intens))
        abs_eps.append(np.abs(iso["eps"] - phot_table["eps"][pi]))

        # PA difference with wrapping (radians)
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


def run_single(label: str, image: np.ndarray, base_config: dict,
               overrides: dict, phot_table: Table) -> dict:
    """Run a single isoster configuration and collect diagnostics."""
    print(f"  Running: {label} ...", end=" ", flush=True)

    cfg = build_config(base_config, overrides)
    t0 = time.perf_counter()
    results = fit_image(image, mask=None, config=cfg)
    elapsed = time.perf_counter() - t0

    isophotes = results["isophotes"]
    counts = stop_code_counts(isophotes)
    matched = match_sma(isophotes, np.array(phot_table["sma"]))
    metrics = compute_metrics(matched, phot_table)

    print(f"done ({elapsed:.2f}s, {len(isophotes)} iso, "
          f"stop2={counts.get(2, 0)})")

    return {
        "label": label,
        "wall_time": elapsed,
        "n_isophotes": len(isophotes),
        "stop_counts": counts,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_markdown_table(all_results: list[dict]) -> None:
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
            f"| {r['wall_time']:.2f} "
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
            f"| {np.degrees(m['med_abs_pa']):.2f} "
            f"| {np.degrees(m['max_abs_pa']):.2f} |"
        )
        print(row)
    print()


def save_figure(all_results: list[dict], output_path: Path) -> None:
    """Save a 2-panel diagnostic figure."""
    labels = [r["label"] for r in all_results]
    stop2 = [r["stop_counts"].get(2, 0) for r in all_results]
    med_di = [r["metrics"]["med_rel_intens"] for r in all_results]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: bar chart of stop=2 counts
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    bars = ax_left.barh(range(len(labels)), stop2, color=colors, edgecolor="k",
                        linewidth=0.5)
    ax_left.set_yticks(range(len(labels)))
    ax_left.set_yticklabels(labels, fontsize=9)
    ax_left.set_xlabel("Stop-code = 2 count")
    ax_left.set_title("Max-iteration failures per config")
    ax_left.invert_yaxis()
    # Annotate bars with counts
    for bar, val in zip(bars, stop2):
        ax_left.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                     str(val), va="center", fontsize=9)

    # Right: scatter of median rel intens diff vs stop=2 count
    ax_right.scatter(stop2, med_di, s=80, c=colors, edgecolors="k",
                     linewidths=0.5, zorder=5)
    for i, label in enumerate(labels):
        ax_right.annotate(
            label, (stop2[i], med_di[i]),
            textcoords="offset points", xytext=(6, 4), fontsize=7.5,
            ha="left",
        )
    ax_right.set_xlabel("Stop-code = 2 count")
    ax_right.set_ylabel("Median |dI/I| vs photutils")
    ax_right.set_title("Accuracy vs convergence failures")
    ax_right.grid(True, alpha=0.3)

    fig.suptitle("NGC1209 mock2 — Convergence diagnostic", fontsize=13,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CONFIGURATIONS = [
    ("Baseline",        {"convergence_scaling": "none",
                         "geometry_damping": 1.0,
                         "geometry_convergence": False}),
    ("A: sector_area",  {"convergence_scaling": "sector_area",
                         "geometry_damping": 1.0,
                         "geometry_convergence": False}),
    ("A-alt: sqrt_sma", {"convergence_scaling": "sqrt_sma",
                         "geometry_damping": 1.0,
                         "geometry_convergence": False}),
    ("B: damping=0.7",  {"convergence_scaling": "none",
                         "geometry_damping": 0.7,
                         "geometry_convergence": False}),
    ("C: geom_conv",    {"convergence_scaling": "none",
                         "geometry_damping": 1.0,
                         "geometry_convergence": True}),
    ("A+B combined",    {"convergence_scaling": "sector_area",
                         "geometry_damping": 0.7,
                         "geometry_convergence": False}),
    ("A+C combined",    {"convergence_scaling": "sector_area",
                         "geometry_damping": 1.0,
                         "geometry_convergence": True}),
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Convergence Diagnostic Benchmark — NGC1209 mock2")
    print("=" * 70)

    # Load inputs
    print("\nLoading image ...", end=" ", flush=True)
    image = load_image(IMAGE_PATH)
    print(f"shape={image.shape}, dtype={image.dtype}")

    print("Loading photutils baseline ...", end=" ", flush=True)
    phot_table = load_photutils_table(PHOT_PATH)
    print(f"{len(phot_table)} isophotes, "
          f"SMA range [{phot_table['sma'].min():.1f}, "
          f"{phot_table['sma'].max():.1f}]")

    print("Loading base config ...", end=" ", flush=True)
    base_config = load_base_config(CONFIG_PATH)
    print(f"sma0={base_config['sma0']}, maxsma={base_config['maxsma']:.1f}, "
          f"eps={base_config['eps']:.3f}")

    # Run all configurations
    print(f"\nRunning {len(CONFIGURATIONS)} configurations:\n")
    all_results = []
    for label, overrides in CONFIGURATIONS:
        result = run_single(label, image, base_config, overrides, phot_table)
        all_results.append(result)

    # Report
    print_markdown_table(all_results)

    # Figure
    output_fig = OUTPUT_DIR / "convergence_diagnostic.png"
    save_figure(all_results, output_fig)

    print("Done.")


if __name__ == "__main__":
    main()
