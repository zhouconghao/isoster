"""Exhaustive IC3370 configuration benchmark.

Tests every meaningful isoster configuration parameter (individually and in
combination) against a photutils baseline on the IC3370_mock2 galaxy.

Usage:
    uv run python benchmarks/ic3370_exhausted/run_benchmark.py
    uv run python benchmarks/ic3370_exhausted/run_benchmark.py --configs S00,S08
    uv run python benchmarks/ic3370_exhausted/run_benchmark.py --skip-photutils --skip-model
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
from isoster.model import build_isoster_model  # noqa: E402
from isoster.plotting import plot_qa_summary, plot_qa_summary_extended  # noqa: E402

from config_registry import (  # noqa: E402
    BASELINE_OVERRIDES,
    CONFIGURATIONS,
    EXTENDED_HARMONIC_CONFIGS,
    NEEDS_REFERENCE_GEOMETRY,
    PHOTUTILS_FIT_CONFIG,
)

# Stop codes to track
STOP_CODES = [0, 1, 2, 3, -1]

DEFAULT_IMAGE_PATH = PROJECT_ROOT / "examples" / "data" / "IC3370_mock2.fits"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "benchmark_ic3370_exhausted"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Exhaustive IC3370 configuration benchmark"
    )
    parser.add_argument(
        "--image", type=Path, default=DEFAULT_IMAGE_PATH,
        help="Path to IC3370_mock2.fits"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output directory"
    )
    parser.add_argument(
        "--configs", type=str, default=None,
        help="Comma-separated config IDs to run (e.g., S00,S08,C03). Default: all."
    )
    parser.add_argument(
        "--skip-photutils", action="store_true",
        help="Skip photutils baseline (reuse existing P00 results)"
    )
    parser.add_argument(
        "--skip-qa-figures", action="store_true",
        help="Skip per-config QA figures"
    )
    parser.add_argument(
        "--skip-model", action="store_true",
        help="Skip 2D model building and residuals"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image(path: Path) -> np.ndarray:
    """Load 2D image array from FITS."""
    with fits.open(path) as hdu_list:
        for hdu in hdu_list:
            if hdu.data is not None and hdu.data.ndim == 2:
                return hdu.data.astype(np.float64)
    raise ValueError(f"No 2D image data found in {path}")


def stop_code_counts(isophotes: list[dict]) -> dict[int, int]:
    """Count isophotes by stop code."""
    codes = [iso["stop_code"] for iso in isophotes]
    return {sc: codes.count(sc) for sc in STOP_CODES}


def match_sma(
    isoster_isos: list[dict],
    phot_sma: np.ndarray,
    tolerance_frac: float = 0.02,
) -> list[tuple[dict, int]]:
    """Match isoster isophotes to photutils SMA values within tolerance.

    Returns list of (isoster_isophote, phot_index) pairs.
    """
    matched = []
    iso_sma = np.array([iso["sma"] for iso in isoster_isos])
    for pi, ps in enumerate(phot_sma):
        if ps < 1e-10:
            continue
        diffs = np.abs(iso_sma - ps)
        best = np.argmin(diffs)
        if diffs[best] < tolerance_frac * ps:
            matched.append((isoster_isos[best], pi))
    return matched


def build_config(overrides: dict) -> IsosterConfig:
    """Build IsosterConfig from baseline + overrides."""
    merged = {**BASELINE_OVERRIDES, **overrides}
    return IsosterConfig(**merged)


# ---------------------------------------------------------------------------
# Photutils baseline
# ---------------------------------------------------------------------------

def run_photutils_baseline(
    image: np.ndarray,
    output_dir: Path,
) -> tuple[Table, list[dict], float]:
    """Run photutils.isophote fitting and save results.

    Returns (photutils_table, photutils_isophotes_as_dicts).
    """
    from photutils.isophote import Ellipse, EllipseGeometry

    cfg = PHOTUTILS_FIT_CONFIG
    geometry = EllipseGeometry(
        x0=cfg["x0"],
        y0=cfg["y0"],
        sma=cfg["sma0"],
        eps=cfg["eps"],
        pa=np.deg2rad(cfg["pa_deg"]),
    )
    ellipse = Ellipse(image, geometry)

    print("  Running photutils.isophote.Ellipse.fit_image() ...", flush=True)
    t0 = time.perf_counter()
    isolist = ellipse.fit_image(
        step=cfg["astep"],
        minsma=cfg["minsma"],
        maxsma=cfg["maxsma"],
        maxgerr=cfg["maxgerr"],
        nclip=cfg["nclip"],
        sclip=cfg["sclip"],
        integrmode=cfg["integrmode"],
    )
    elapsed = time.perf_counter() - t0

    table = isolist.to_table()
    p00_dir = output_dir / "P00"
    p00_dir.mkdir(parents=True, exist_ok=True)
    table.write(p00_dir / "photutils_baseline.fits", overwrite=True)

    print(f"  Photutils done: {elapsed:.2f}s, {len(table)} isophotes")

    # Convert to list-of-dicts for metrics
    phot_isos = []
    for row in table:
        phot_isos.append({
            "sma": float(row["sma"]),
            "intens": float(row["intens"]),
            "eps": float(row["ellipticity"]),
            "pa": float(np.radians(row["pa"].value if hasattr(row["pa"], "value") else row["pa"])),
            "stop_code": int(row["stop_code"]),
            "x0": float(row["x0"]),
            "y0": float(row["y0"]),
        })

    return table, phot_isos, elapsed


# ---------------------------------------------------------------------------
# Reference geometry from photutils
# ---------------------------------------------------------------------------

def _col_values(table: Table, colname: str) -> np.ndarray:
    """Extract column values as plain numpy array, stripping astropy units."""
    col = table[colname]
    if hasattr(col, "value"):
        return np.array(col.value, dtype=np.float64)
    return np.array(col, dtype=np.float64)


def estimate_effective_radius(phot_table: Table) -> float:
    """Estimate effective radius from photutils CoG (half-light radius).

    Uses trapezoidal integration of intensity * 2*pi*sma*(1-eps)*astep
    on the sorted isophote table.
    """
    sma = _col_values(phot_table, "sma")
    intens = _col_values(phot_table, "intens")
    eps = _col_values(phot_table, "ellipticity")

    # Sort by SMA
    order = np.argsort(sma)
    sma = sma[order]
    intens = intens[order]
    eps = eps[order]

    # Cumulative flux using elliptical annuli
    cumflux = np.zeros(len(sma))
    for i in range(1, len(sma)):
        r_inner = sma[i - 1]
        r_outer = sma[i]
        avg_intens = 0.5 * (intens[i - 1] + intens[i])
        avg_eps = 0.5 * (eps[i - 1] + eps[i])
        area = np.pi * (r_outer**2 - r_inner**2) * (1.0 - avg_eps)
        cumflux[i] = cumflux[i - 1] + avg_intens * area

    total_flux = cumflux[-1]
    if total_flux <= 0:
        return sma[-1] / 2.0  # fallback

    half_flux = total_flux / 2.0
    idx = np.searchsorted(cumflux, half_flux)
    if idx >= len(sma):
        return sma[-1]
    if idx == 0:
        return sma[0]

    # Linear interpolation
    frac = (half_flux - cumflux[idx - 1]) / (cumflux[idx] - cumflux[idx - 1])
    re = sma[idx - 1] + frac * (sma[idx] - sma[idx - 1])
    return float(re)


def derive_reference_geometry(
    phot_table: Table,
    re_pix: float,
    n_re: float = 3.0,
) -> dict:
    """Compute median centroid, axis ratio, PA from photutils within n_re x Re.

    Returns dict with x0, y0, eps, pa (radians) usable as config overrides.
    """
    sma = _col_values(phot_table, "sma")
    mask = (sma > 0) & (sma < n_re * re_pix)
    stop = _col_values(phot_table, "stop_code").astype(int)
    mask &= (stop == 0)

    if mask.sum() < 3:
        # Fallback: use all converged isophotes
        mask = (stop == 0)

    x0 = float(np.median(_col_values(phot_table, "x0")[mask]))
    y0 = float(np.median(_col_values(phot_table, "y0")[mask]))
    eps = float(np.median(_col_values(phot_table, "ellipticity")[mask]))
    # PA column from photutils is in degrees — convert to radians
    pa_deg = _col_values(phot_table, "pa")
    pa = float(np.radians(np.median(pa_deg[mask])))

    return {"x0": x0, "y0": y0, "eps": eps, "pa": pa}


# ---------------------------------------------------------------------------
# Per-zone metrics
# ---------------------------------------------------------------------------

def compute_metrics_by_zone(
    matched: list[tuple[dict, int]],
    phot_table: Table,
    re_pix: float,
    n_inner: float = 1.0,
    n_outer: float = 4.0,
) -> dict:
    """Profile accuracy metrics per radial zone (inner/mid/outer).

    Only uses stop_code==0 isophotes from both sides.
    """
    zones = {"inner": [], "mid": [], "outer": []}
    sma_inner = n_inner * re_pix
    sma_outer = n_outer * re_pix

    # Pre-extract photutils columns as plain arrays
    phot_stop = _col_values(phot_table, "stop_code").astype(int)
    phot_intens = _col_values(phot_table, "intens")
    phot_eps = _col_values(phot_table, "ellipticity")
    phot_pa_rad = np.radians(_col_values(phot_table, "pa"))

    for iso, pi in matched:
        if iso["stop_code"] != 0:
            continue
        if phot_stop[pi] != 0:
            continue

        sma = iso["sma"]
        p_intens = phot_intens[pi]
        if abs(p_intens) < 1e-10:
            continue

        entry = {
            "rel_intens": abs(iso["intens"] - p_intens) / abs(p_intens),
            "abs_eps": abs(iso["eps"] - phot_eps[pi]),
            "abs_pa": _pa_diff(iso["pa"], phot_pa_rad[pi]),
        }

        if sma < sma_inner:
            zones["inner"].append(entry)
        elif sma < sma_outer:
            zones["mid"].append(entry)
        else:
            zones["outer"].append(entry)

    result = {}
    for zone_name, entries in zones.items():
        if not entries:
            result[zone_name] = {
                "n": 0,
                "med_rel_intens": np.nan, "max_rel_intens": np.nan,
                "med_abs_eps": np.nan, "max_abs_eps": np.nan,
                "med_abs_pa_deg": np.nan, "max_abs_pa_deg": np.nan,
            }
        else:
            ri = [e["rel_intens"] for e in entries]
            ae = [e["abs_eps"] for e in entries]
            ap = [e["abs_pa"] for e in entries]
            result[zone_name] = {
                "n": len(entries),
                "med_rel_intens": float(np.median(ri)),
                "max_rel_intens": float(np.max(ri)),
                "med_abs_eps": float(np.median(ae)),
                "max_abs_eps": float(np.max(ae)),
                "med_abs_pa_deg": float(np.degrees(np.median(ap))),
                "max_abs_pa_deg": float(np.degrees(np.max(ap))),
            }
    return result


def _pa_diff(pa1: float, pa2: float) -> float:
    """Absolute PA difference with wrapping (radians), range [0, pi/2]."""
    d = pa1 - pa2
    return abs((d + np.pi / 2) % np.pi - np.pi / 2)


def compute_model_residual_stats(
    image: np.ndarray,
    model: np.ndarray,
) -> dict:
    """Fractional residual statistics for 2D model.

    Restricts to pixels where model is above 1% of its peak to avoid
    background-dominated pixels inflating fractional residuals.
    """
    model_peak = np.nanmax(model)
    if model_peak <= 0:
        return {"model_frac_med": np.nan, "model_rms": np.nan}

    threshold = 0.01 * model_peak
    valid = (model > threshold) & (np.abs(image) > threshold)
    if valid.sum() == 0:
        return {"model_frac_med": np.nan, "model_rms": np.nan}

    frac = (model[valid] - image[valid]) / image[valid]
    return {
        "model_frac_med": float(np.median(np.abs(frac))),
        "model_rms": float(np.sqrt(np.mean(frac**2))),
    }


# ---------------------------------------------------------------------------
# Single config runner
# ---------------------------------------------------------------------------

def run_single_config(
    config_id: str,
    description: str,
    image: np.ndarray,
    overrides: dict,
    phot_table: Table,
    re_pix: float,
    output_dir: Path,
    *,
    skip_qa: bool = False,
    skip_model: bool = False,
    phot_isos: list[dict] | None = None,
) -> dict:
    """Run one isoster configuration and collect all metrics."""
    print(f"  [{config_id}] {description} ...", end=" ", flush=True)

    cfg = build_config(overrides)
    cfg_dir = output_dir / config_id
    cfg_dir.mkdir(parents=True, exist_ok=True)

    # Suppress config validation warnings for intentional configs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        t0 = time.perf_counter()
        results = fit_image(image, mask=None, config=cfg)
        elapsed = time.perf_counter() - t0

    isophotes = results["isophotes"]
    counts = stop_code_counts(isophotes)
    matched = match_sma(isophotes, np.array(phot_table["sma"]))
    zone_metrics = compute_metrics_by_zone(matched, phot_table, re_pix)

    # Optional 2D model
    model_stats = {}
    model_2d = None
    if not skip_model:
        try:
            model_2d = build_isoster_model(image.shape, isophotes)
            model_stats = compute_model_residual_stats(image, model_2d)
        except Exception as exc:
            print(f"[model failed: {exc}]", end=" ")
            model_stats = {"model_frac_med": np.nan, "model_rms": np.nan}

    # Optional QA figure
    if not skip_qa and model_2d is not None:
        try:
            qa_path = str(cfg_dir / f"qa_{config_id}.png")
            # Use photutils isolist for overlay if available
            plot_qa_summary(
                title=f"IC3370 — {config_id}: {description}",
                image=image,
                isoster_model=model_2d,
                isoster_res=isophotes,
                photutils_res=None,  # raw isolist not available here
                filename=qa_path,
            )
            # Extended QA for harmonic configs
            if config_id in EXTENDED_HARMONIC_CONFIGS:
                ext_path = str(cfg_dir / f"qa_{config_id}_extended.png")
                plot_qa_summary_extended(
                    title=f"IC3370 — {config_id}: {description}",
                    image=image,
                    isoster_model=model_2d,
                    isoster_res=isophotes,
                    filename=ext_path,
                )
        except Exception as exc:
            print(f"[QA failed: {exc}]", end=" ")

    # Save isophotes as JSON
    _save_isophotes_json(isophotes, cfg_dir / f"isophotes_{config_id}.json")

    print(f"done ({elapsed:.2f}s, {len(isophotes)} iso, "
          f"stop0={counts.get(0, 0)}, stop2={counts.get(2, 0)})")

    return {
        "config_id": config_id,
        "description": description,
        "wall_time": elapsed,
        "n_isophotes": len(isophotes),
        "stop_counts": counts,
        "n_matched": len(matched),
        "zone_metrics": zone_metrics,
        "model_stats": model_stats,
    }


def _save_isophotes_json(isophotes: list[dict], path: Path) -> None:
    """Save isophotes to a compact JSON file."""
    # Keep only essential keys for compact storage
    keys_to_keep = [
        "sma", "intens", "intens_err", "eps", "eps_err",
        "pa", "pa_err", "x0", "y0", "stop_code", "ndata", "nflag", "niter",
        "a3", "b3", "a4", "b4", "rms",
    ]
    compact = []
    for iso in isophotes:
        row = {}
        for k in keys_to_keep:
            if k in iso:
                v = iso[k]
                if isinstance(v, (np.floating, np.integer)):
                    v = float(v) if isinstance(v, np.floating) else int(v)
                row[k] = v
        compact.append(row)
    with open(path, "w") as f:
        json.dump(compact, f, indent=1)


# ---------------------------------------------------------------------------
# Summary figure
# ---------------------------------------------------------------------------

def generate_summary_figure(
    all_results: list[dict],
    phot_result: dict | None,
    output_dir: Path,
) -> None:
    """Generate two 4-panel summary figures: one vs P00, one vs S00.

    Both include P00 and all isoster configs in the bar charts.  The
    scatter panels compare against either P00 (photutils) or S00 (isoster
    baseline) as the reference.
    """
    # Build combined list: P00 first, then all isoster results
    combined = []
    if phot_result:
        combined.append(phot_result)
    combined.extend(all_results)

    ids = [r["config_id"] for r in combined]
    n = len(ids)
    colors = plt.cm.tab20(np.linspace(0, 1, n))

    # --- Helper to draw one figure ---
    def _draw_figure(results_list, ref_label, filename):
        r_ids = [r["config_id"] for r in results_list]
        r_n = len(r_ids)
        r_colors = plt.cm.tab20(np.linspace(0, 1, r_n))

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Panel 1: Convergence counts (stop=0 vs total)
        ax = axes[0, 0]
        stop0 = [r["stop_counts"].get(0, 0) for r in results_list]
        stop2 = [r["stop_counts"].get(2, 0) for r in results_list]
        total = [r["n_isophotes"] for r in results_list]
        x = np.arange(r_n)
        ax.barh(x, total, color="lightgray", edgecolor="k", linewidth=0.3,
                label="total")
        ax.barh(x, stop0, color="steelblue", edgecolor="k", linewidth=0.3,
                label="stop=0")
        ax.set_yticks(x)
        ax.set_yticklabels(r_ids, fontsize=7)
        ax.set_xlabel("Isophote count")
        ax.set_title("Convergence: stop=0 / total")
        ax.invert_yaxis()
        ax.legend(fontsize=8)

        # Panel 2: Wall time
        ax = axes[0, 1]
        times = [r["wall_time"] for r in results_list]
        bars = ax.barh(x, times, color=r_colors, edgecolor="k", linewidth=0.3)
        ax.set_yticks(x)
        ax.set_yticklabels(r_ids, fontsize=7)
        ax.set_xlabel("Wall time (s)")
        ax.set_title("Runtime")
        ax.invert_yaxis()
        for bar, val in zip(bars, times):
            if np.isfinite(val):
                ax.text(bar.get_width() + 0.02,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.2f}", va="center", fontsize=6)

        # Panel 3: Intensity accuracy (mid-zone median) vs stop=2 count
        ax = axes[1, 0]
        mid_med_intens = []
        for r in results_list:
            zm = r.get("zone_metrics", {}).get("mid", {})
            mid_med_intens.append(zm.get("med_rel_intens", np.nan))
        ax.scatter(stop2, mid_med_intens, s=60, c=r_colors, edgecolors="k",
                   linewidths=0.5, zorder=5)
        for i, cid in enumerate(r_ids):
            if np.isfinite(mid_med_intens[i]):
                ax.annotate(cid, (stop2[i], mid_med_intens[i]),
                            textcoords="offset points", xytext=(5, 3),
                            fontsize=6)
        ax.set_xlabel("Stop-code = 2 count")
        ax.set_ylabel(f"Median |dI/I| vs {ref_label} (mid zone)")
        ax.set_title(f"Accuracy vs convergence failures (ref: {ref_label})")
        ax.grid(True, alpha=0.3)

        # Panel 4: eps accuracy vs PA accuracy (mid-zone)
        ax = axes[1, 1]
        mid_med_eps = []
        mid_med_pa = []
        for r in results_list:
            zm = r.get("zone_metrics", {}).get("mid", {})
            mid_med_eps.append(zm.get("med_abs_eps", np.nan))
            mid_med_pa.append(zm.get("med_abs_pa_deg", np.nan))
        ax.scatter(mid_med_eps, mid_med_pa, s=60, c=r_colors, edgecolors="k",
                   linewidths=0.5, zorder=5)
        for i, cid in enumerate(r_ids):
            if np.isfinite(mid_med_eps[i]) and np.isfinite(mid_med_pa[i]):
                ax.annotate(cid, (mid_med_eps[i], mid_med_pa[i]),
                            textcoords="offset points", xytext=(5, 3),
                            fontsize=6)
        ax.set_xlabel(f"Median |d_eps| vs {ref_label} (mid zone)")
        ax.set_ylabel(f"Median |d_PA| deg vs {ref_label} (mid zone)")
        ax.set_title(f"Geometry accuracy (ref: {ref_label})")
        ax.grid(True, alpha=0.3)

        fig.suptitle(f"IC3370_mock2 — Configuration Benchmark (ref: {ref_label})",
                     fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = output_dir / filename
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Summary figure saved: {out_path}")

    # Figure 1: vs P00 (photutils baseline) — all configs including P00
    _draw_figure(combined, "P00", "summary_vs_P00.png")

    # Figure 2: vs S00 (isoster baseline) — isoster configs only
    _draw_figure(all_results, "S00", "summary_vs_S00.png")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def generate_markdown_report(
    all_results: list[dict],
    phot_result: dict | None,
    re_pix: float,
    output_dir: Path,
) -> None:
    """Generate report.md with full metrics tables."""
    lines = [
        "# IC3370 Exhaustive Configuration Benchmark Report\n",
        f"Effective radius (Re): {re_pix:.1f} px\n",
        "## Global Metrics\n",
        "| Config | Time(s) | N_iso | sc=0 | sc=1 | sc=2 | sc=3 | sc=-1 "
        "| N_match | Model_med% | Model_RMS |",
        "|--------|---------|-------|------|------|------|------|-------|"
        "---------|-----------|-----------|",
    ]

    all_to_report = []
    if phot_result:
        all_to_report.append(phot_result)
    all_to_report.extend(all_results)

    for r in all_to_report:
        sc = r["stop_counts"]
        ms = r.get("model_stats", {})
        cid = r.get("config_id", r.get("label", "?"))
        lines.append(
            f"| {cid} | {r['wall_time']:.2f} | {r['n_isophotes']} "
            f"| {sc.get(0, 0)} | {sc.get(1, 0)} | {sc.get(2, 0)} "
            f"| {sc.get(3, 0)} | {sc.get(-1, 0)} "
            f"| {r.get('n_matched', '-')} "
            f"| {ms.get('model_frac_med', np.nan):.4f} "
            f"| {ms.get('model_rms', np.nan):.4f} |"
        )

    lines.append("")
    lines.append("## Per-Zone Accuracy (vs Photutils Baseline)\n")

    for zone_name in ["inner", "mid", "outer"]:
        lines.append(f"### Zone: {zone_name}\n")
        lines.append(
            "| Config | N | med_dI/I | max_dI/I | med_deps | max_deps "
            "| med_dPA(deg) | max_dPA(deg) |"
        )
        lines.append(
            "|--------|---|----------|----------|----------|----------"
            "|--------------|--------------|"
        )
        for r in all_results:
            cid = r["config_id"]
            zm = r["zone_metrics"].get(zone_name, {})
            lines.append(
                f"| {cid} | {zm.get('n', 0)} "
                f"| {zm.get('med_rel_intens', np.nan):.4f} "
                f"| {zm.get('max_rel_intens', np.nan):.4f} "
                f"| {zm.get('med_abs_eps', np.nan):.4f} "
                f"| {zm.get('max_abs_eps', np.nan):.4f} "
                f"| {zm.get('med_abs_pa_deg', np.nan):.2f} "
                f"| {zm.get('max_abs_pa_deg', np.nan):.2f} |"
            )
        lines.append("")

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines))
    print(f"Report saved: {report_path}")


# ---------------------------------------------------------------------------
# QA gallery
# ---------------------------------------------------------------------------

def compile_qa_gallery(output_dir: Path) -> None:
    """Copy all QA figures into qa_gallery/ subfolder."""
    gallery_dir = output_dir / "qa_gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for cfg_dir in sorted(output_dir.iterdir()):
        if not cfg_dir.is_dir():
            continue
        if cfg_dir.name == "qa_gallery":
            continue
        for png in cfg_dir.glob("qa_*.png"):
            dest = gallery_dir / png.name
            shutil.copy2(png, dest)
            count += 1

    print(f"QA gallery: {count} figures copied to {gallery_dir}")


# ---------------------------------------------------------------------------
# JIT warmup
# ---------------------------------------------------------------------------

def _warmup_jit(image: np.ndarray) -> None:
    """Run a tiny fit_image call to trigger numba JIT compilation.

    Uses a small SMA range so it finishes quickly (~1-2s on first call).
    Results are discarded.
    """
    warmup_cfg = build_config({
        "maxsma": 20.0,
        "maxit": 10,
        "minit": 3,
    })
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        fit_image(image, mask=None, config=warmup_cfg)
        elapsed = time.perf_counter() - t0
    print(f"  JIT warmup done ({elapsed:.2f}s)")


# ---------------------------------------------------------------------------
# P00 QA figure
# ---------------------------------------------------------------------------

def _generate_p00_qa(
    image: np.ndarray,
    phot_table: Table,
    output_dir: Path,
) -> None:
    """Build a photutils model and generate QA figure for P00.

    Converts the photutils table into isoster-style dicts so we can reuse
    plot_qa_summary and build_isoster_model.
    """
    # Convert photutils table to isoster-compatible isophote dicts
    phot_isos_for_model = []
    phot_sma = _col_values(phot_table, "sma")
    phot_intens = _col_values(phot_table, "intens")
    phot_eps = _col_values(phot_table, "ellipticity")
    phot_pa_rad = np.radians(_col_values(phot_table, "pa"))
    phot_stop = _col_values(phot_table, "stop_code").astype(int)
    phot_x0 = _col_values(phot_table, "x0")
    phot_y0 = _col_values(phot_table, "y0")

    # Optional columns
    has_intens_err = "intens_err" in phot_table.colnames
    has_eps_err = "ellipticity_err" in phot_table.colnames
    has_pa_err = "pa_err" in phot_table.colnames

    for i in range(len(phot_table)):
        iso = {
            "sma": float(phot_sma[i]),
            "intens": float(phot_intens[i]),
            "eps": float(phot_eps[i]),
            "pa": float(phot_pa_rad[i]),
            "stop_code": int(phot_stop[i]),
            "x0": float(phot_x0[i]),
            "y0": float(phot_y0[i]),
        }
        if has_intens_err:
            iso["intens_err"] = float(_col_values(phot_table, "intens_err")[i])
        if has_eps_err:
            iso["eps_err"] = float(_col_values(phot_table, "ellipticity_err")[i])
        if has_pa_err:
            iso["pa_err"] = float(np.radians(_col_values(phot_table, "pa_err")[i]))
        phot_isos_for_model.append(iso)

    # Build 2D model from photutils isophotes
    try:
        model = build_isoster_model(image.shape, phot_isos_for_model)
    except Exception as exc:
        print(f"  P00 model build failed: {exc}")
        return

    # QA figure
    p00_dir = output_dir / "P00"
    p00_dir.mkdir(parents=True, exist_ok=True)
    qa_path = str(p00_dir / "qa_P00.png")
    plot_qa_summary(
        title="IC3370 — P00: Photutils baseline",
        image=image,
        isoster_model=model,
        isoster_res=phot_isos_for_model,
        filename=qa_path,
    )


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("IC3370 Exhaustive Configuration Benchmark")
    print("=" * 70)

    # Load image
    print(f"\nLoading image: {args.image}")
    image = load_image(args.image)
    print(f"  Shape: {image.shape}, dtype: {image.dtype}")

    # --- P00: Photutils baseline ---
    phot_table = None
    phot_isos = None
    phot_result = None
    phot_elapsed = None

    p00_fits = output_dir / "P00" / "photutils_baseline.fits"
    if args.skip_photutils and p00_fits.exists():
        print("\nLoading existing photutils baseline ...")
        phot_table = Table.read(p00_fits)
        phot_elapsed = np.nan  # unknown from cached file
        print(f"  {len(phot_table)} isophotes from {p00_fits}")
    else:
        print("\nRunning photutils baseline (P00) ...")
        phot_table, phot_isos, phot_elapsed = run_photutils_baseline(
            image, output_dir
        )

    # Always build phot_result from the table so P00 appears in reports
    phot_stop = _col_values(phot_table, "stop_code").astype(int)
    phot_result = {
        "config_id": "P00",
        "description": "Photutils baseline",
        "wall_time": phot_elapsed,
        "n_isophotes": len(phot_table),
        "stop_counts": {
            sc: int(np.sum(phot_stop == sc)) for sc in STOP_CODES
        },
        "n_matched": len(phot_table),
        "model_stats": {},
    }

    # Derive effective radius and reference geometry
    re_pix = estimate_effective_radius(phot_table)
    print(f"\nEffective radius (Re): {re_pix:.1f} px")

    ref_geom = derive_reference_geometry(phot_table, re_pix)
    print(f"Reference geometry (within 3 Re): x0={ref_geom['x0']:.1f}, "
          f"y0={ref_geom['y0']:.1f}, eps={ref_geom['eps']:.3f}, "
          f"pa={np.degrees(ref_geom['pa']):.1f} deg")

    # --- P00 QA figure (photutils model + residual) ---
    if not args.skip_qa_figures and not args.skip_model:
        print("\nGenerating P00 photutils QA figure ...")
        _generate_p00_qa(image, phot_table, output_dir)

    # --- Numba JIT warmup ---
    print("\nWarming up numba JIT (dry run) ...")
    _warmup_jit(image)

    # --- Filter configs if requested ---
    configs_to_run = CONFIGURATIONS
    if args.configs:
        requested = set(args.configs.split(","))
        configs_to_run = [
            (cid, desc, ov) for cid, desc, ov in CONFIGURATIONS
            if cid in requested
        ]
        print(f"\nRunning subset: {[c[0] for c in configs_to_run]}")

    # --- Run all isoster configs ---
    print(f"\nRunning {len(configs_to_run)} isoster configurations:\n")
    all_results = []
    timing_log = {}

    for config_id, description, overrides in configs_to_run:
        # Inject reference geometry for S22/S23
        if config_id in NEEDS_REFERENCE_GEOMETRY:
            overrides = {**overrides}  # copy to avoid mutation
            if config_id == "S22":
                overrides["x0"] = ref_geom["x0"]
                overrides["y0"] = ref_geom["y0"]
            elif config_id == "S23":
                overrides["x0"] = ref_geom["x0"]
                overrides["y0"] = ref_geom["y0"]
                overrides["eps"] = ref_geom["eps"]
                overrides["pa"] = ref_geom["pa"]

        result = run_single_config(
            config_id=config_id,
            description=description,
            image=image,
            overrides=overrides,
            phot_table=phot_table,
            re_pix=re_pix,
            output_dir=output_dir,
            skip_qa=args.skip_qa_figures,
            skip_model=args.skip_model,
            phot_isos=phot_isos,
        )
        all_results.append(result)
        timing_log[config_id] = {
            "wall_time": result["wall_time"],
            "n_isophotes": result["n_isophotes"],
            "stop_counts": result["stop_counts"],
        }

    # --- Save timing log ---
    if phot_result:
        timing_log["P00"] = {
            "wall_time": phot_result["wall_time"],
            "n_isophotes": phot_result["n_isophotes"],
            "stop_counts": phot_result["stop_counts"],
        }
    timing_path = output_dir / "timing_log.json"
    with open(timing_path, "w") as f:
        json.dump(timing_log, f, indent=2)
    print(f"\nTiming log saved: {timing_path}")

    # --- Summary figures ---
    print("\nGenerating summary figures ...")
    generate_summary_figure(all_results, phot_result, output_dir)

    # --- Markdown report ---
    print("Generating markdown report ...")
    generate_markdown_report(all_results, phot_result, re_pix, output_dir)

    # --- QA gallery ---
    if not args.skip_qa_figures:
        compile_qa_gallery(output_dir)

    print("\n" + "=" * 70)
    print("Benchmark complete.")
    print(f"Output: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
