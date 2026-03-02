"""Exhaustive isoster configuration sweep benchmark.

Tests every meaningful isoster configuration parameter (individually and in
combination) against a photutils baseline on a single galaxy FITS image.

The sweep covers 39 configurations (P00 + S00-S23 + C01-C12) defined in
config_registry.py.  Galaxy geometry is supplied via CLI arguments or
auto-detected from the image.

Usage:
    # Default run on the bundled IC3370 mock (geometry auto-detected)
    uv run python benchmarks/exhausted/run_benchmark.py

    # Explicit geometry for IC3370_mock2 (reproduces the canonical run)
    uv run python benchmarks/exhausted/run_benchmark.py \\
        --image data/IC3370_mock2.fits \\
        --x0 566 --y0 566 --eps 0.239 --pa-deg -27.99 --sma0 6 --maxsma 283

    # Run on a custom image with auto-detected geometry
    uv run python benchmarks/exhausted/run_benchmark.py \\
        --image data/ngc3610.fits --band-index 2 --galaxy-label ngc3610

    # Run a subset of configs
    uv run python benchmarks/exhausted/run_benchmark.py --configs S00,S08,C03

    # Quick smoke test (S00 + S08 only, no QA figures, no model)
    uv run python benchmarks/exhausted/run_benchmark.py --quick
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
    CONFIGURATIONS,
    EXTENDED_HARMONIC_CONFIGS,
    NEEDS_REFERENCE_GEOMETRY,
    PARAMETER_BASELINE,
    PHOTUTILS_PARAMETER_CONFIG,
)

# Stop codes to track
STOP_CODES = [0, 1, 2, 3, -1]

DEFAULT_IMAGE_PATH = PROJECT_ROOT / "data" / "IC3370_mock2.fits"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "benchmark_exhausted"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Exhaustive isoster configuration sweep benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Image selection
    parser.add_argument(
        "--image", type=Path, default=DEFAULT_IMAGE_PATH,
        help="Path to a 2D (or 3D-cube) galaxy FITS file.",
    )
    parser.add_argument(
        "--band-index", type=int, default=0,
        help="Band plane index to extract when the FITS contains a 3D cube (0-based).",
    )
    parser.add_argument(
        "--galaxy-label", type=str, default=None,
        help="Short label used in titles and output folder names. "
             "Defaults to the image file stem.",
    )

    # Geometry (all optional — auto-detected if omitted)
    geo = parser.add_argument_group(
        "geometry",
        "Initial ellipse geometry. Omit any argument to auto-detect from the image.",
    )
    geo.add_argument("--x0", type=float, default=None,
                     help="Galaxy centre x-coordinate (pixels, 0-indexed).")
    geo.add_argument("--y0", type=float, default=None,
                     help="Galaxy centre y-coordinate (pixels, 0-indexed).")
    geo.add_argument("--eps", type=float, default=None,
                     help="Initial ellipticity (0 = circular, <1).")
    geo.add_argument("--pa-deg", type=float, default=None,
                     help="Initial position angle in degrees (photutils convention).")
    geo.add_argument("--sma0", type=float, default=None,
                     help="Starting semi-major axis in pixels.")
    geo.add_argument("--maxsma", type=float, default=None,
                     help="Maximum semi-major axis in pixels.")

    # Output
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory. Defaults to outputs/benchmark_exhausted/<galaxy-label>/.",
    )

    # Config selection
    parser.add_argument(
        "--configs", type=str, default=None,
        help="Comma-separated config IDs to run (e.g., S00,S08,C03). Default: all.",
    )

    # Quick mode
    parser.add_argument(
        "--quick", action="store_true",
        help="Smoke test: run S00 + S08 only, skip QA figures and 2D model.",
    )

    # Skip flags
    parser.add_argument(
        "--skip-photutils", action="store_true",
        help="Skip photutils baseline run (reuse existing P00/photutils_baseline.fits).",
    )
    parser.add_argument(
        "--skip-qa-figures", action="store_true",
        help="Skip per-config QA figures.",
    )
    parser.add_argument(
        "--skip-model", action="store_true",
        help="Skip 2D model building and residuals.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(path: Path, band_index: int = 0) -> np.ndarray:
    """Load a 2D image array from a FITS file.

    For a 3D data cube, extracts the plane at ``band_index``.
    For a 2D array, returns it directly.
    """
    with fits.open(path) as hdu_list:
        for hdu in hdu_list:
            if hdu.data is None:
                continue
            data = hdu.data
            if data.ndim == 2:
                return data.astype(np.float64)
            if data.ndim == 3:
                if band_index >= data.shape[0]:
                    raise ValueError(
                        f"band_index={band_index} out of range for cube with "
                        f"{data.shape[0]} planes."
                    )
                return data[band_index].astype(np.float64)
    raise ValueError(f"No usable 2D image data found in {path}")


# ---------------------------------------------------------------------------
# Geometry auto-detection
# ---------------------------------------------------------------------------

def auto_detect_geometry(image: np.ndarray) -> dict:
    """Derive sensible starting geometry from image shape.

    Returns a dict with x0, y0, eps, pa (radians), sma0, maxsma.
    Center is taken as the image centre; eps and pa use conservative defaults.
    """
    h, w = image.shape
    x0 = (w - 1) / 2.0
    y0 = (h - 1) / 2.0
    half_size = min(x0, y0)
    return {
        "x0": x0,
        "y0": y0,
        "eps": 0.2,
        "pa": 0.0,
        "sma0": max(5.0, half_size * 0.04),
        "maxsma": half_size * 0.90,
    }


def build_geometry(args: argparse.Namespace, image: np.ndarray) -> dict:
    """Resolve geometry from CLI args, filling missing values by auto-detection.

    Returns a dict with x0, y0, eps, pa (radians), sma0, maxsma.
    """
    auto = auto_detect_geometry(image)
    x0 = args.x0 if args.x0 is not None else auto["x0"]
    y0 = args.y0 if args.y0 is not None else auto["y0"]
    eps = args.eps if args.eps is not None else auto["eps"]
    pa = np.deg2rad(args.pa_deg) if args.pa_deg is not None else auto["pa"]
    sma0 = args.sma0 if args.sma0 is not None else auto["sma0"]
    maxsma = args.maxsma if args.maxsma is not None else auto["maxsma"]
    return {
        "x0": x0,
        "y0": y0,
        "eps": eps,
        "pa": pa,
        "sma0": sma0,
        "maxsma": maxsma,
    }


# ---------------------------------------------------------------------------
# Helpers
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


def build_config(baseline: dict, overrides: dict) -> IsosterConfig:
    """Build IsosterConfig from baseline + overrides."""
    merged = {**baseline, **overrides}
    return IsosterConfig(**merged)


# ---------------------------------------------------------------------------
# Photutils baseline
# ---------------------------------------------------------------------------

def run_photutils_baseline(
    image: np.ndarray,
    geometry: dict,
    output_dir: Path,
) -> tuple[Table, list[dict], float]:
    """Run photutils.isophote fitting and save results.

    Returns (photutils_table, photutils_isophotes_as_dicts, elapsed_seconds).
    """
    from photutils.isophote import Ellipse, EllipseGeometry

    cfg = PHOTUTILS_PARAMETER_CONFIG

    geo = EllipseGeometry(
        x0=geometry["x0"],
        y0=geometry["y0"],
        sma=geometry["sma0"],
        eps=geometry["eps"],
        pa=geometry["pa"],
    )
    ellipse = Ellipse(image, geo)

    print("  Running photutils.isophote.Ellipse.fit_image() ...", flush=True)
    t0 = time.perf_counter()
    isolist = ellipse.fit_image(
        step=cfg["astep"],
        minsma=cfg["minsma"],
        maxsma=geometry["maxsma"],
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

    Uses trapezoidal integration of intensity * elliptical annulus area.
    """
    sma = _col_values(phot_table, "sma")
    intens = _col_values(phot_table, "intens")
    eps = _col_values(phot_table, "ellipticity")

    order = np.argsort(sma)
    sma = sma[order]
    intens = intens[order]
    eps = eps[order]

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
        return sma[-1] / 2.0

    half_flux = total_flux / 2.0
    idx = np.searchsorted(cumflux, half_flux)
    if idx >= len(sma):
        return sma[-1]
    if idx == 0:
        return sma[0]

    frac = (half_flux - cumflux[idx - 1]) / (cumflux[idx] - cumflux[idx - 1])
    return float(sma[idx - 1] + frac * (sma[idx] - sma[idx - 1]))


def derive_reference_geometry(
    phot_table: Table,
    re_pix: float,
    n_re: float = 3.0,
) -> dict:
    """Compute median centroid, axis ratio, PA from photutils within n_re * Re.

    Returns dict with x0, y0, eps, pa (radians) usable as config overrides.
    """
    sma = _col_values(phot_table, "sma")
    stop = _col_values(phot_table, "stop_code").astype(int)
    mask = (sma > 0) & (sma < n_re * re_pix) & (stop == 0)

    if mask.sum() < 3:
        mask = (stop == 0)

    x0 = float(np.median(_col_values(phot_table, "x0")[mask]))
    y0 = float(np.median(_col_values(phot_table, "y0")[mask]))
    eps = float(np.median(_col_values(phot_table, "ellipticity")[mask]))
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
    """Fractional residual statistics for a 2D model.

    Restricts to pixels where model exceeds 1% of its peak, avoiding
    background-dominated pixels from inflating fractional residuals.
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
    baseline: dict,
    overrides: dict,
    phot_table: Table,
    re_pix: float,
    output_dir: Path,
    galaxy_label: str,
    *,
    skip_qa: bool = False,
    skip_model: bool = False,
) -> dict:
    """Run one isoster configuration and collect all metrics."""
    print(f"  [{config_id}] {description} ...", end=" ", flush=True)

    cfg = build_config(baseline, overrides)
    cfg_dir = output_dir / config_id
    cfg_dir.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        t0 = time.perf_counter()
        results = fit_image(image, mask=None, config=cfg)
        elapsed = time.perf_counter() - t0

    isophotes = results["isophotes"]
    counts = stop_code_counts(isophotes)
    matched = match_sma(isophotes, np.array(phot_table["sma"]))
    zone_metrics = compute_metrics_by_zone(matched, phot_table, re_pix)

    model_stats = {}
    model_2d = None
    if not skip_model:
        try:
            model_2d = build_isoster_model(image.shape, isophotes)
            model_stats = compute_model_residual_stats(image, model_2d)
        except Exception as exc:
            print(f"[model failed: {exc}]", end=" ")
            model_stats = {"model_frac_med": np.nan, "model_rms": np.nan}

    if not skip_qa and model_2d is not None:
        try:
            qa_path = str(cfg_dir / f"qa_{config_id}.png")
            plot_qa_summary(
                title=f"{galaxy_label} — {config_id}: {description}",
                image=image,
                isoster_model=model_2d,
                isoster_res=isophotes,
                photutils_res=None,
                filename=qa_path,
            )
            if config_id in EXTENDED_HARMONIC_CONFIGS:
                ext_path = str(cfg_dir / f"qa_{config_id}_extended.png")
                plot_qa_summary_extended(
                    title=f"{galaxy_label} — {config_id}: {description}",
                    image=image,
                    isoster_model=model_2d,
                    isoster_res=isophotes,
                    filename=ext_path,
                )
        except Exception as exc:
            print(f"[QA failed: {exc}]", end=" ")

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
    galaxy_label: str,
) -> None:
    """Generate two 4-panel summary figures: one vs P00 (photutils), one vs S00."""
    combined = []
    if phot_result:
        combined.append(phot_result)
    combined.extend(all_results)

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

        fig.suptitle(
            f"{galaxy_label} — Configuration Sweep (ref: {ref_label})",
            fontsize=14, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = output_dir / filename
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Summary figure saved: {out_path}")

    _draw_figure(combined, "P00", "summary_vs_P00.png")
    _draw_figure(all_results, "S00", "summary_vs_S00.png")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def generate_markdown_report(
    all_results: list[dict],
    phot_result: dict | None,
    re_pix: float,
    output_dir: Path,
    galaxy_label: str,
) -> None:
    """Generate report.md with full metrics tables."""
    lines = [
        f"# {galaxy_label} — Exhaustive Configuration Sweep Report\n",
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
    """Copy all per-config QA figures into qa_gallery/ subfolder."""
    gallery_dir = output_dir / "qa_gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for cfg_dir in sorted(output_dir.iterdir()):
        if not cfg_dir.is_dir() or cfg_dir.name == "qa_gallery":
            continue
        for png in cfg_dir.glob("qa_*.png"):
            dest = gallery_dir / png.name
            shutil.copy2(png, dest)
            count += 1

    print(f"QA gallery: {count} figures copied to {gallery_dir}")


# ---------------------------------------------------------------------------
# P00 QA figure
# ---------------------------------------------------------------------------

def _generate_p00_qa(
    image: np.ndarray,
    phot_table: Table,
    output_dir: Path,
    galaxy_label: str,
) -> None:
    """Build a photutils model and generate QA figure for P00."""
    phot_isos_for_model = []
    phot_sma = _col_values(phot_table, "sma")
    phot_intens = _col_values(phot_table, "intens")
    phot_eps = _col_values(phot_table, "ellipticity")
    phot_pa_rad = np.radians(_col_values(phot_table, "pa"))
    phot_stop = _col_values(phot_table, "stop_code").astype(int)
    phot_x0 = _col_values(phot_table, "x0")
    phot_y0 = _col_values(phot_table, "y0")

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

    try:
        model = build_isoster_model(image.shape, phot_isos_for_model)
    except Exception as exc:
        print(f"  P00 model build failed: {exc}")
        return

    p00_dir = output_dir / "P00"
    p00_dir.mkdir(parents=True, exist_ok=True)
    qa_path = str(p00_dir / "qa_P00.png")
    plot_qa_summary(
        title=f"{galaxy_label} — P00: Photutils baseline",
        image=image,
        isoster_model=model,
        isoster_res=phot_isos_for_model,
        filename=qa_path,
    )


# ---------------------------------------------------------------------------
# JIT warmup
# ---------------------------------------------------------------------------

def _warmup_jit(image: np.ndarray, baseline: dict) -> None:
    """Run a tiny fit_image call to trigger numba JIT compilation."""
    warmup_maxsma = min(20.0, baseline.get("maxsma", 100.0) * 0.07)
    warmup_cfg = build_config(baseline, {"maxsma": warmup_maxsma, "maxit": 10, "minit": 3})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        fit_image(image, mask=None, config=warmup_cfg)
        elapsed = time.perf_counter() - t0
    print(f"  JIT warmup done ({elapsed:.2f}s)")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Galaxy label
    galaxy_label = args.galaxy_label or args.image.stem

    # Output directory
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = DEFAULT_OUTPUT_DIR / galaxy_label
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Exhaustive Configuration Sweep — {galaxy_label}")
    print("=" * 70)

    # Load image
    print(f"\nLoading image: {args.image}  (band-index={args.band_index})")
    image = load_image(args.image, band_index=args.band_index)
    print(f"  Shape: {image.shape}, dtype: {image.dtype}")

    # Resolve geometry
    geometry = build_geometry(args, image)
    print(
        f"\nGeometry: x0={geometry['x0']:.1f}, y0={geometry['y0']:.1f}, "
        f"eps={geometry['eps']:.3f}, pa={np.degrees(geometry['pa']):.1f} deg, "
        f"sma0={geometry['sma0']:.1f}, maxsma={geometry['maxsma']:.1f}"
    )

    # Build baseline (fitting params + geometry)
    baseline = {**PARAMETER_BASELINE, **geometry}

    # Quick mode
    if args.quick:
        configs_filter: set[str] | None = {"S00", "S08"}
        skip_qa = True
        skip_model = True
        print("\n[Quick mode] Running S00 + S08 only, no QA figures, no 2D model.")
    else:
        configs_filter = None
        skip_qa = args.skip_qa_figures
        skip_model = args.skip_model

    # Override filter from --configs flag (takes priority over --quick)
    if args.configs:
        configs_filter = set(args.configs.split(","))

    # --- P00: Photutils baseline ---
    phot_table = None
    phot_result = None
    phot_elapsed = None

    p00_fits = output_dir / "P00" / "photutils_baseline.fits"
    if args.skip_photutils and p00_fits.exists():
        print("\nLoading existing photutils baseline ...")
        phot_table = Table.read(p00_fits)
        phot_elapsed = np.nan
        print(f"  {len(phot_table)} isophotes from {p00_fits}")
    else:
        print("\nRunning photutils baseline (P00) ...")
        phot_table, _, phot_elapsed = run_photutils_baseline(
            image, geometry, output_dir
        )

    phot_stop = _col_values(phot_table, "stop_code").astype(int)
    phot_result = {
        "config_id": "P00",
        "description": "Photutils baseline",
        "wall_time": phot_elapsed,
        "n_isophotes": len(phot_table),
        "stop_counts": {sc: int(np.sum(phot_stop == sc)) for sc in STOP_CODES},
        "n_matched": len(phot_table),
        "model_stats": {},
        "zone_metrics": {},
    }

    # Effective radius and reference geometry
    re_pix = estimate_effective_radius(phot_table)
    print(f"\nEffective radius (Re): {re_pix:.1f} px")

    ref_geom = derive_reference_geometry(phot_table, re_pix)
    print(f"Reference geometry (within 3 Re): x0={ref_geom['x0']:.1f}, "
          f"y0={ref_geom['y0']:.1f}, eps={ref_geom['eps']:.3f}, "
          f"pa={np.degrees(ref_geom['pa']):.1f} deg")

    # P00 QA figure
    if not skip_qa and not skip_model:
        print("\nGenerating P00 photutils QA figure ...")
        _generate_p00_qa(image, phot_table, output_dir, galaxy_label)

    # Numba JIT warmup
    print("\nWarming up numba JIT (dry run) ...")
    _warmup_jit(image, baseline)

    # Filter configs
    configs_to_run = CONFIGURATIONS
    if configs_filter:
        configs_to_run = [
            (cid, desc, ov) for cid, desc, ov in CONFIGURATIONS
            if cid in configs_filter
        ]
        print(f"\nRunning subset: {[c[0] for c in configs_to_run]}")

    # Run all isoster configs
    print(f"\nRunning {len(configs_to_run)} isoster configurations:\n")
    all_results = []
    timing_log: dict = {}

    for config_id, description, overrides in configs_to_run:
        # Inject reference geometry for S22/S23
        if config_id in NEEDS_REFERENCE_GEOMETRY:
            overrides = {**overrides}
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
            baseline=baseline,
            overrides=overrides,
            phot_table=phot_table,
            re_pix=re_pix,
            output_dir=output_dir,
            galaxy_label=galaxy_label,
            skip_qa=skip_qa,
            skip_model=skip_model,
        )
        all_results.append(result)
        timing_log[config_id] = {
            "wall_time": result["wall_time"],
            "n_isophotes": result["n_isophotes"],
            "stop_counts": result["stop_counts"],
        }

    # Save timing log
    timing_log["P00"] = {
        "wall_time": phot_result["wall_time"],
        "n_isophotes": phot_result["n_isophotes"],
        "stop_counts": phot_result["stop_counts"],
    }
    timing_path = output_dir / "timing_log.json"
    with open(timing_path, "w") as f:
        json.dump(timing_log, f, indent=2)
    print(f"\nTiming log saved: {timing_path}")

    # Summary figures and report (skip in quick mode)
    if not args.quick:
        print("\nGenerating summary figures ...")
        generate_summary_figure(all_results, phot_result, output_dir, galaxy_label)

        print("Generating markdown report ...")
        generate_markdown_report(all_results, phot_result, re_pix, output_dir, galaxy_label)

        if not skip_qa:
            compile_qa_gallery(output_dir)

    print("\n" + "=" * 70)
    print("Benchmark complete.")
    print(f"Output: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
