# ruff: noqa: E402
import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from unittest.mock import patch

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import isoster
from isoster.config import IsosterConfig
from isoster.model import build_isoster_model
from isoster.output_paths import resolve_output_directory
from isoster.plotting import (
    configure_qa_plot_style,
    normalize_pa_degrees,
    set_x_limits_with_right_margin,
)

# photutils imports
try:
    from photutils.isophote import Ellipse, EllipseGeometry
    PHOTUTILS_AVAILABLE = True
except ImportError:
    PHOTUTILS_AVAILABLE = False
    warnings.warn("photutils not available. Comparison will be limited to isoster (Normal vs Lazy).")

# ---------------------------------------------------------------------------
# Registry and Utils
# ---------------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"

GALAXY_REGISTRY = [
    {
        "name": "IC3370_mock2",
        "fits_path": DATA_DIR / "IC3370_mock2.fits",
        "center": (566.0, 566.0),
        "eps": 0.24,
        "pa": -0.49,
        "sma0": 6.0,
        "maxsma": 200.0,
    },
    {
        "name": "ngc3610",
        "fits_path": DATA_DIR / "ngc3610.fits",
        "center": None,
        "eps": None,
        "pa": None,
        "sma0": 10.0,
        "maxsma": 150.0,
    },
    {
        "name": "eso243-49",
        "fits_path": DATA_DIR / "eso243-49.fits",
        "center": None,
        "eps": None,
        "pa": None,
        "sma0": 10.0,
        "maxsma": 120.0,
    },
]

def estimate_initial_params(image, galaxy_info):
    info = dict(galaxy_info)
    ny, nx = image.shape
    if info["center"] is None:
        info["center"] = (nx / 2.0, ny / 2.0)
    if info["eps"] is None:
        info["eps"] = 0.2
    if info["pa"] is None:
        info["pa"] = 0.0
    return info

def run_isoster_benchmark(image, info, use_lazy):
    x0, y0 = info["center"]
    config = IsosterConfig(
        x0=x0, y0=y0, eps=info["eps"], pa=info["pa"],
        sma0=info["sma0"], minsma=1.0, maxsma=info["maxsma"],
        use_lazy_gradient=use_lazy,
        compute_errors=False,
        compute_deviations=True, # Enable deviations for better model and QA
        full_photometry=False,
        nclip=2, sclip=3.0
    )

    # Instrument extract_isophote_data to count calls
    with patch('isoster.fitting.extract_isophote_data', wraps=isoster.fitting.extract_isophote_data) as mock_extract:
        start_time = time.perf_counter()
        results = isoster.fit_image(image, None, config)
        runtime = time.perf_counter() - start_time
        call_count = mock_extract.call_count
        total_iters = sum([iso['niter'] for iso in results['isophotes']])

    return results, runtime, call_count, total_iters

def run_photutils_benchmark(image, info):
    if not PHOTUTILS_AVAILABLE:
        return None, 0, 0

    x0, y0 = info["center"]
    geometry = EllipseGeometry(x0=x0, y0=y0, sma=info["sma0"], eps=info["eps"], pa=info["pa"])
    ellipse = Ellipse(image, geometry)

    start_time = time.perf_counter()
    try:
        isolist = ellipse.fit_image(
            maxsma=info["maxsma"],
            minsma=1.0,
            step=0.1,
            linear=False,
            nclip=2,
            sclip=3.0
        )
        runtime = time.perf_counter() - start_time

        # Convert to dict format similar to isoster for easier comparison
        isophotes = []
        for iso in isolist:
            isophotes.append({
                'sma': iso.sma,
                'intens': iso.intens,
                'eps': iso.eps,
                'pa': iso.pa,
                'rms': iso.rms
            })
        return isophotes, runtime, 0
    except Exception as e:
        print(f"  Photutils failed: {e}")
        return None, 0, 0

def plot_comparison_qa(name, image, res_norm, res_lazy, res_photo, output_path,
                       calls_norm, calls_lazy, time_norm, time_lazy):
    """Generate high-quality comparison figure with 2D residuals and 1D profiles."""
    configure_qa_plot_style()

    # 1. Build models for residuals
    model_norm = build_isoster_model(image.shape, res_norm['isophotes'], use_harmonics=True)
    model_lazy = build_isoster_model(image.shape, res_lazy['isophotes'], use_harmonics=True)

    residual_norm = np.where(np.isfinite(image), image - model_norm, np.nan)
    residual_lazy = np.where(np.isfinite(image), image - model_lazy, np.nan)

    # Common scale for residuals
    all_res = np.concatenate([residual_norm[np.isfinite(residual_norm)],
                             residual_lazy[np.isfinite(residual_lazy)]])
    res_limit = float(np.clip(np.nanpercentile(np.abs(all_res), 99.0) if all_res.size else 1.0, 0.05, None))

    # 2. Figure Layout
    n_profile_rows = 5
    fig = plt.figure(figsize=(14.0, 12.0))
    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.5, 4.0], hspace=0.15)

    # Top: 2 residual panels
    top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.1)

    # Bottom: 5 profile rows
    profile_height_ratios = [2.0, 1.0, 1.0, 1.0, 1.0]
    bottom = gridspec.GridSpecFromSubplotSpec(n_profile_rows, 1, subplot_spec=outer[1],
                                             height_ratios=profile_height_ratios, hspace=0.0)

    fig.suptitle(f"Benchmark: {name} - Lazy Gradient Evaluation Improvement", fontsize=16, y=0.98)

    # Residual Plots
    for i, (res_map, title) in enumerate([(residual_norm, "Normal Residual"), (residual_lazy, "Lazy Residual")]):
        ax = fig.add_subplot(top[0, i])
        ax.imshow(res_map, origin="lower", cmap="coolwarm", vmin=-res_limit, vmax=res_limit, interpolation="nearest")
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    # Profile Plots
    ax_sb = fig.add_subplot(bottom[0])
    ax_diff = fig.add_subplot(bottom[1], sharex=ax_sb)
    ax_ba = fig.add_subplot(bottom[2], sharex=ax_sb)
    ax_pa = fig.add_subplot(bottom[3], sharex=ax_sb)
    ax_calls = fig.add_subplot(bottom[4], sharex=ax_sb)

    # Data extraction helper
    def _arr(isos, key): return np.array([r.get(key, np.nan) for r in isos])

    # Plotting loop
    conditions = [
        (res_norm['isophotes'], 'Normal', '#1f77b4', 'o'),
        (res_lazy['isophotes'], 'Lazy', '#d62728', 's')
    ]
    if res_photo:
        conditions.append((res_photo, 'Photutils', '#2ca02c', '^'))

    all_sma_pow = []

    for isos, label, col, mrk in conditions:
        sma = _arr(isos, 'sma')
        intens = _arr(isos, 'intens')
        eps = _arr(isos, 'eps')
        pa_rad = _arr(isos, 'pa')

        xax = sma ** 0.25
        all_sma_pow.append(xax)

        # SB
        ok = (sma > 0) & np.isfinite(intens) & (intens > 0)
        ax_sb.scatter(xax[ok], np.log10(intens[ok]), color=col, marker=mrk, s=15, alpha=0.6, label=label)

        # Axis ratio
        ax_ba.scatter(xax[ok], 1.0 - eps[ok], color=col, marker=mrk, s=15, alpha=0.6)

        # PA
        pa_deg = normalize_pa_degrees(np.degrees(pa_rad))
        ax_pa.scatter(xax[ok], pa_deg[ok], color=col, marker=mrk, s=15, alpha=0.6)

    # 4. Accuracy (Relative difference Lazy vs Normal)
    sma_lazy = _arr(res_lazy['isophotes'], 'sma')
    xax_lazy = sma_lazy ** 0.25
    intens_norm = _arr(res_norm['isophotes'], 'intens')
    intens_lazy = _arr(res_lazy['isophotes'], 'intens')

    if len(intens_norm) == len(intens_lazy):
        diff = (intens_lazy - intens_norm) / intens_norm
    else:
        intens_norm_interp = np.interp(sma_lazy, _arr(res_norm['isophotes'], 'sma'), intens_norm)
        diff = (intens_lazy - intens_norm_interp) / intens_norm_interp

    ax_diff.plot(xax_lazy, diff * 100, color='black', lw=1.0, alpha=0.8)
    ax_diff.axhline(0, color='red', linestyle='--', lw=0.8)

    # 5. Call reduction (visualized cumulatively if possible, but here we just show points)
    # Actually, we don't have per-isophote call counts easily, so let's just use it for labels

    # Axis formatting
    ax_sb.set_ylabel(r"$\log_{10}(I)$")
    ax_sb.set_title("Surface Brightness", fontsize=10, pad=2)
    ax_sb.legend(loc='upper right', frameon=True, fontsize=9)

    ax_diff.set_ylabel(r"$\Delta I/I$ [%]")
    ax_diff.set_title("Lazy vs Normal Difference", fontsize=10, pad=2)

    ax_ba.set_ylabel("b/a")
    ax_ba.set_title("Axis Ratio", fontsize=10, pad=2)

    ax_pa.set_ylabel("PA [deg]")
    ax_pa.set_title("Position Angle", fontsize=10, pad=2)

    ax_calls.set_xlabel(r"SMA$^{0.25}$ (pixel$^{0.25}$)")
    ax_calls.set_ylabel("Info")
    ax_calls.text(0.5, 0.5, f"Lazy Call Reduction: {(calls_norm-calls_lazy)/calls_norm*100:.1f}%\n"
                           f"Lazy Speedup vs Normal: {time_norm/time_lazy:.2f}x\n"
                           f"Normal Calls: {calls_norm}, Lazy Calls: {calls_lazy}",
                  transform=ax_calls.transAxes, ha='center', va='center', fontsize=12, fontweight='bold')
    ax_calls.set_yticks([])

    for ax in [ax_sb, ax_diff, ax_ba, ax_pa, ax_calls]:
        ax.grid(alpha=0.2)
        if ax != ax_calls: ax.tick_params(labelbottom=False)

    if all_sma_pow:
        xcat = np.concatenate(all_sma_pow)
        set_x_limits_with_right_margin(ax_calls, xcat)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=150)
    plt.close()

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Benchmark Lazy Gradient Evaluation')
    parser.add_argument('--output-dir', type=str, default=None, help='Explicit output directory')
    args = parser.parse_args()

    output_dir = resolve_output_directory("benchmarks_performance", "benchmark_lazy_gradient", explicit_output_directory=args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_results = []

    for galaxy in GALAXY_REGISTRY:
        name = galaxy["name"]
        print(f"Benchmarking {name}...")

        if not galaxy["fits_path"].exists():
            print(f"  File not found: {galaxy['fits_path']}")
            continue

        with fits.open(galaxy["fits_path"]) as hdul:
            image = hdul[0].data.astype(np.float64)
            if image.ndim == 3:
                image = image[0]

        info = estimate_initial_params(image, galaxy)

        # 1. Normal isoster
        print("  Running isoster (Normal)...")
        res_norm, time_norm, calls_norm, iters_norm = run_isoster_benchmark(image, info, use_lazy=False)

        # 2. Lazy isoster
        print("  Running isoster (Lazy)...")
        res_lazy, time_lazy, calls_lazy, iters_lazy = run_isoster_benchmark(image, info, use_lazy=True)

        # 3. Photutils
        print("  Running photutils...")
        res_photo, time_photo, _ = run_photutils_benchmark(image, info)

        # Compare
        call_reduction = (calls_norm - calls_lazy) / calls_norm * 100 if calls_norm > 0 else 0
        speedup_vs_norm = time_norm / time_lazy if time_lazy > 0 else 0

        print(f"  Normal: {time_norm:.3f}s, {calls_norm} calls, {iters_norm} iters, {len(res_norm['isophotes'])} isophotes")
        print(f"  Lazy:   {time_lazy:.3f}s, {calls_lazy} calls, {iters_lazy} iters, {len(res_lazy['isophotes'])} isophotes ({call_reduction:.1f}% reduction, {speedup_vs_norm:.2f}x speedup)")
        if time_photo > 0:
            speedup_vs_photo = time_photo / time_lazy
            print(f"  Photo:  {time_photo:.3f}s ({speedup_vs_photo:.2f}x speedup over photutils)")

        # Accuracy stats
        sma_lazy = np.array([iso['sma'] for iso in res_lazy['isophotes']])
        intens_norm = np.array([iso['intens'] for iso in res_norm['isophotes']])
        intens_lazy = np.array([iso['intens'] for iso in res_lazy['isophotes']])
        if len(intens_norm) != len(intens_lazy):
            intens_norm_interp = np.interp(sma_lazy, [iso['sma'] for iso in res_norm['isophotes']], intens_norm)
            diff = (intens_lazy - intens_norm_interp) / intens_norm_interp
        else:
            diff = (intens_lazy - intens_norm) / intens_norm

        max_diff_pct = np.max(np.abs(diff)) * 100
        median_diff_pct = np.median(np.abs(diff)) * 100
        rms_diff_pct = np.sqrt(np.mean(diff**2)) * 100
        print(f"  Accuracy: Intensity diff - Max: {max_diff_pct:.4f}%, Median: {median_diff_pct:.4f}%, RMS: {rms_diff_pct:.4f}%")

        # Save individual profiles and summary
        case_data = {
            "name": name,
            "normal": {"time": time_norm, "calls": calls_norm, "iters": iters_norm},
            "lazy": {"time": time_lazy, "calls": calls_lazy, "iters": iters_lazy},
            "photutils": {"time": time_photo},
            "call_reduction_pct": call_reduction,
            "speedup_vs_norm": speedup_vs_norm,
            "speedup_vs_photo": time_photo / time_lazy if time_lazy > 0 and time_photo > 0 else None,
            "accuracy": {"max_diff_pct": max_diff_pct, "median_diff_pct": median_diff_pct, "rms_diff_pct": rms_diff_pct}
        }
        summary_results.append(case_data)

        # Generate QA Plot
        plot_comparison_qa(name, image, res_norm, res_lazy, res_photo, output_dir / f"comparison_{name}.png",
                           calls_norm, calls_lazy, time_norm, time_lazy)

    # Write summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary_results, f, indent=2)

    print(f"\nBenchmark complete. Outputs saved to {output_dir}")

if __name__ == "__main__":
    main()
