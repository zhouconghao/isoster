#!/usr/bin/env python
"""
Benchmark: Explicit Noise Floor & Gradient Error Fix
isoster (Baseline) vs isoster (Noise Floor Fix) vs photutils.isophote

This benchmark evaluates the stability improvement and error bar correctness
introduced by the explicit noise floor (sigma_bg) and corrected geometric
error propagation.
"""

import sys
import time
import argparse
import os
import json
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import isoster
from isoster.config import IsosterConfig
from isoster.output_paths import resolve_output_directory
from isoster.model import build_isoster_model
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

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

def estimate_background(image: np.ndarray) -> float:
    """Estimate background noise sigma from image corners."""
    size = 50
    ny, nx = image.shape
    corners = [
        image[:size, :size],
        image[:size, -size:],
        image[-size:, :size],
        image[-size:, -size:],
    ]
    pixels = np.concatenate([c.ravel() for c in corners])
    return float(np.std(pixels))

def run_isoster_fit(image, sigma_bg=None, use_corrected_errors=True, maxsma=None):
    ny, nx = image.shape
    config = IsosterConfig(
        x0=nx/2.0, y0=ny/2.0, eps=0.2, pa=0.0,
        sma0=10.0, minsma=1.0, maxsma=maxsma,
        sigma_bg=sigma_bg,
        use_corrected_errors=use_corrected_errors,
        compute_errors=True,
        compute_deviations=True,
        nclip=2, sclip=3.0
    )
    results = isoster.fit_image(image, None, config)
    return results

def run_photutils_fit(image, maxsma=None):
    if not PHOTUTILS_AVAILABLE:
        return None
    ny, nx = image.shape
    geometry = EllipseGeometry(x0=nx/2.0, y0=ny/2.0, sma=10.0, eps=0.2, pa=0.0)
    ellipse = Ellipse(image, geometry)
    try:
        isolist = ellipse.fit_image(
            maxsma=maxsma,
            minsma=1.0,
            step=0.1,
            linear=False,
            nclip=2,
            sclip=3.0
        )
        # Convert to dict format
        isophotes = []
        for iso in isolist:
            isophotes.append({
                'sma': iso.sma,
                'intens': iso.intens,
                'eps': iso.eps,
                'pa': iso.pa,
                'eps_err': iso.ellip_err,
                'pa_err': iso.pa_err,
                'stop_code': 0 # photutils doesn't provide these easily
            })
        return {'isophotes': isophotes}
    except Exception:
        return None

def plot_noise_floor_qa(name, image, res_base, res_fix, res_photo, sigma_bg, output_path):
    configure_qa_plot_style()
    
    # Figure Layout
    fig = plt.figure(figsize=(14.0, 14.0))
    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.0, 4.0], hspace=0.1)
    
    # Top: 2 residual panels
    top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.1)
    
    # Bottom: 5 profile rows
    bottom = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=outer[1], 
                                             height_ratios=[2, 1, 1, 1, 1], hspace=0.0)
    
    fig.suptitle(f"Benchmark: {name} - Noise Floor and Error Fix", fontsize=16, y=0.98)
    
    # Build models for residuals
    model_base = build_isoster_model(image.shape, res_base['isophotes'], use_harmonics=True)
    model_fix = build_isoster_model(image.shape, res_fix['isophotes'], use_harmonics=True)
    
    residual_base = np.where(np.isfinite(image), image - model_base, np.nan)
    residual_fix = np.where(np.isfinite(image), image - model_fix, np.nan)
    
    all_res = np.concatenate([residual_base[np.isfinite(residual_base)], 
                             residual_fix[np.isfinite(residual_fix)]])
    res_limit = float(np.clip(np.nanpercentile(np.abs(all_res), 99.0) if all_res.size else 1.0, 0.05, None))

    for i, (res_map, title) in enumerate([(residual_base, "Baseline Residual"), (residual_fix, "Noise Floor Fix Residual")]):
        ax = fig.add_subplot(top[0, i])
        ax.imshow(res_map, origin="lower", cmap="coolwarm", vmin=-res_limit, vmax=res_limit, interpolation="nearest")
        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])

    # Profile Plots
    ax_sb = fig.add_subplot(bottom[0])
    ax_eps = fig.add_subplot(bottom[1], sharex=ax_sb)
    ax_pa = fig.add_subplot(bottom[2], sharex=ax_sb)
    ax_eps_err = fig.add_subplot(bottom[3], sharex=ax_sb)
    ax_pa_err = fig.add_subplot(bottom[4], sharex=ax_sb)
    
    def _arr(isos, key): return np.array([r.get(key, np.nan) for r in isos])
    
    conditions = [
        (res_base['isophotes'], 'Baseline', '#1f77b4', 'o'),
        (res_fix['isophotes'], 'Noise Floor Fix', '#d62728', 's')
    ]
    if res_photo:
        conditions.append((res_photo['isophotes'], 'Photutils', '#2ca02c', '^'))
        
    all_sma_pow = []
    for isos, label, col, mrk in conditions:
        sma = _arr(isos, 'sma')
        intens = _arr(isos, 'intens')
        eps = _arr(isos, 'eps')
        pa_rad = _arr(isos, 'pa')
        eps_err = _arr(isos, 'eps_err')
        pa_err_rad = _arr(isos, 'pa_err')
        
        xax = sma ** 0.25
        all_sma_pow.append(xax)
        
        ok = (sma > 0) & np.isfinite(intens) & (intens > 0)
        ax_sb.scatter(xax[ok], np.log10(intens[ok]), color=col, marker=mrk, s=15, alpha=0.6, label=label)
        
        ax_eps.errorbar(xax[ok], 1.0 - eps[ok], yerr=eps_err[ok], fmt=mrk, color=col, markersize=4, alpha=0.5, capsize=0)
        
        pa_deg = normalize_pa_degrees(np.degrees(pa_rad))
        pa_err_deg = np.degrees(pa_err_rad)
        ax_pa.errorbar(xax[ok], pa_deg[ok], yerr=pa_err_deg[ok], fmt=mrk, color=col, markersize=4, alpha=0.5, capsize=0)
        
        ax_eps_err.plot(xax[ok], eps_err[ok], color=col, lw=1.0, alpha=0.8)
        ax_pa_err.plot(xax[ok], pa_err_deg[ok], color=col, lw=1.0, alpha=0.8)

    # Mark sigma_bg level
    ax_sb.axhline(np.log10(sigma_bg), color='gray', linestyle='--', lw=1.0, label=r'$\sigma_{bg}$')
    
    # Formatting
    ax_sb.set_ylabel(r"$\log_{10}(I)$")
    ax_sb.legend(loc='upper right', fontsize=9)
    ax_eps.set_ylabel("b/a")
    ax_pa.set_ylabel("PA [deg]")
    ax_eps_err.set_ylabel(r"$\sigma_{eps}$")
    ax_pa_err.set_ylabel(r"$\sigma_{PA}$ [deg]")
    ax_pa_err.set_xlabel(r"SMA$^{0.25}$")
    
    for ax in [ax_sb, ax_eps, ax_pa, ax_eps_err]:
        ax.tick_params(labelbottom=False)
        ax.grid(alpha=0.2)
    ax_pa_err.grid(alpha=0.2)
    
    if all_sma_pow:
        xcat = np.concatenate(all_sma_pow)
        set_x_limits_with_right_margin(ax_pa_err, xcat)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Benchmark Noise Floor & Error Fix')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()
    
    output_dir = resolve_output_directory("benchmarks_performance", "benchmark_noise_floor", explicit_output_directory=args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mock_dir = Path("outputs/benchmark_noise_floor/mocks")
    galaxies = ["NGC1453_hsc_wide.fits", "NGC3585_hsc_wide.fits"]
    
    summary = []
    for gal_file in galaxies:
        fits_path = mock_dir / gal_file
        if not fits_path.exists():
            print(f"Skipping {gal_file}, not found.")
            continue
            
        name = gal_file.replace(".fits", "")
        print(f"Benchmarking {name}...")
        
        with fits.open(fits_path) as hdul:
            image = hdul[0].data.astype(np.float64)
            
        sigma_bg = estimate_background(image)
        print(f"  Estimated sigma_bg: {sigma_bg:.4f}")
        
        # 1. Baseline (no sigma_bg, no corrected errors)
        print("  Running Baseline isoster...")
        res_base = run_isoster_fit(image, sigma_bg=None, use_corrected_errors=False)
        
        # 2. Fix (with sigma_bg and corrected errors)
        print("  Running Noise Floor Fix isoster...")
        res_fix = run_isoster_fit(image, sigma_bg=sigma_bg, use_corrected_errors=True)
        
        # 3. Photutils
        print("  Running photutils...")
        res_photo = run_photutils_fit(image)
        
        # Generate QA
        plot_noise_floor_qa(name, image, res_base, res_fix, res_photo, sigma_bg, output_dir / f"comparison_{name}.png")
        
        summary.append({
            'name': name,
            'sigma_bg': sigma_bg,
            'n_isophotes_base': len(res_base['isophotes']),
            'n_isophotes_fix': len(res_fix['isophotes']),
            'avg_eps_err_base': float(np.nanmean([iso['eps_err'] for iso in res_base['isophotes'][-10:]])),
            'avg_eps_err_fix': float(np.nanmean([iso['eps_err'] for iso in res_fix['isophotes'][-10:]])),
            'eps_std_base': float(np.nanstd([iso['eps'] for iso in res_base['isophotes'][-10:]])),
            'eps_std_fix': float(np.nanstd([iso['eps'] for iso in res_fix['isophotes'][-10:]])),
            'pa_std_base': float(np.nanstd([iso['pa'] for iso in res_base['isophotes'][-10:]])),
            'pa_std_fix': float(np.nanstd([iso['pa'] for iso in res_fix['isophotes'][-10:]])),
        })
        
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nBenchmark complete. Outputs saved to {output_dir}")

if __name__ == "__main__":
    main()
