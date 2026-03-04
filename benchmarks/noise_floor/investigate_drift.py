#!/usr/bin/env python
"""
Investigation: Centroid Drift in Low Surface Brightness Outskirts
isoster (Baseline) vs isoster (Noise Floor Fix) vs photutils vs AutoProf

This script specifically focuses on quantifying and visualizing the radial
variation of the fitted centroid (x0, y0) in mock images where the truth
is a fixed center.
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
    derive_arcsinh_parameters,
    make_arcsinh_display_from_parameters,
    draw_isophote_overlays,
    set_axis_limits_from_finite_values,
    robust_limits,
)
from benchmarks.utils.autoprof_adapter import run_autoprof_fit

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

def run_isoster_fit(image, sigma_bg=None, maxsma=None):
    ny, nx = image.shape
    config = IsosterConfig(
        x0=nx/2.0, y0=ny/2.0, eps=0.2, pa=0.0,
        sma0=10.0, minsma=1.0, maxsma=maxsma,
        sigma_bg=sigma_bg,
        clip_max_shift=5.0,  # relaxed safeguard
        clip_max_eps=0.1,    # relaxed safeguard
        clip_max_pa=0.5,
        compute_errors=True,
        compute_deviations=True,
        nclip=2, sclip=3.0
    )
    start_time = time.perf_counter()
    results = isoster.fit_image(image, None, config)
    runtime = time.perf_counter() - start_time
    return results, runtime

def run_photutils_fit(image, maxsma=None):
    if not PHOTUTILS_AVAILABLE:
        return None, 0
    ny, nx = image.shape
    geometry = EllipseGeometry(x0=nx/2.0, y0=ny/2.0, sma=10.0, eps=0.2, pa=0.0)
    ellipse = Ellipse(image, geometry)
    start_time = time.perf_counter()
    try:
        isolist = ellipse.fit_image(
            maxsma=maxsma,
            minsma=1.0,
            step=0.1,
            linear=False,
            nclip=2,
            sclip=3.0
        )
        runtime = time.perf_counter() - start_time
        isophotes = []
        for iso in isolist:
            isophotes.append({
                'sma': iso.sma, 'intens': iso.intens, 'eps': iso.eps, 'pa': iso.pa,
                'x0': iso.x0, 'y0': iso.y0, 'eps_err': iso.ellip_err, 'pa_err': iso.pa_err,
                'x0_err': iso.x0_err, 'y0_err': iso.y0_err,
                'intens_err': iso.intens_err,
                'stop_code': 0
            })
        return {'isophotes': isophotes}, runtime
    except Exception:
        return None, 0

def plot_drift_qa(name, image, results_dict, sigma_bg, true_center, output_path):
    """
    Generate high-quality investigation figure focusing on Centroid Drift.
    Left: Mock Image + Residual maps.
    Right: 1-D profiles with scatter and error bars.
    """
    configure_qa_plot_style()
    
    fig = plt.figure(figsize=(16.0, 18.0))
    # Left: 4 image panels (Data, Base Res, Fix Res, Photutils Res)
    # Right: 6 profile panels (SB, SB Diff, dx, dy, b/a, PA)
    outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.8], wspace=0.2)
    
    left = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=outer[0], hspace=0.15)
    right = gridspec.GridSpecFromSubplotSpec(6, 1, subplot_spec=outer[1], 
                                             height_ratios=[2.0, 1.0, 1.0, 1.0, 1.0, 1.0], hspace=0.0)
    
    fig.suptitle(f"Centroid Drift Investigation: {name}", fontsize=20, y=0.98)
    
    # 1. Image panels
    ref_low, ref_high, ref_scale, ref_vmax = derive_arcsinh_parameters(image)
    img_disp, vmin, vmax = make_arcsinh_display_from_parameters(image, ref_low, ref_high, ref_scale, ref_vmax)
    
    ax_img = fig.add_subplot(left[0])
    ax_img.imshow(img_disp, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    ax_img.set_title("Data + Fitted Isophotes (Noise Fix)", fontsize=12)
    if results_dict.get('Noise Fix'):
        draw_isophote_overlays(ax_img, results_dict['Noise Fix']['isophotes'], step=max(1, len(results_dict['Noise Fix']['isophotes'])//15), edge_color='orangered')
    
    # Models and Residuals
    residuals = []
    res_titles = []
    
    if results_dict.get('Baseline'):
        m_base = build_isoster_model(image.shape, results_dict['Baseline']['isophotes'], use_harmonics=True)
        residuals.append(np.where(np.isfinite(image), image - m_base, np.nan))
        res_titles.append("Baseline Residual")
        
    if results_dict.get('Noise Fix'):
        m_fix = build_isoster_model(image.shape, results_dict['Noise Fix']['isophotes'], use_harmonics=True)
        residuals.append(np.where(np.isfinite(image), image - m_fix, np.nan))
        res_titles.append("Noise Fix Residual")
        
    if results_dict.get('Photutils') and results_dict['Photutils'] is not None:
        m_photo = build_isoster_model(image.shape, results_dict['Photutils']['isophotes'], use_harmonics=False)
        residuals.append(np.where(np.isfinite(image), image - m_photo, np.nan))
        res_titles.append("Photutils Residual")
        
    # Scale residuals
    if residuals:
        all_res = np.concatenate([r[np.isfinite(r)] for r in residuals])
        res_limit = float(np.clip(np.nanpercentile(np.abs(all_res), 99.0) if all_res.size else 1.0, 0.05, None))
    else:
        res_limit = 1.0
        
    for i, (res_map, title) in enumerate(zip(residuals, res_titles)):
        if i + 1 < 4:
            ax = fig.add_subplot(left[i + 1])
            ax.imshow(res_map, origin="lower", cmap="coolwarm", vmin=-res_limit, vmax=res_limit, interpolation="nearest")
            ax.set_title(title, fontsize=12)
            ax.set_xticks([]); ax.set_yticks([])

    # 2. Profiles
    ax_sb = fig.add_subplot(right[0])
    ax_sb_diff = fig.add_subplot(right[1], sharex=ax_sb)
    ax_dx = fig.add_subplot(right[2], sharex=ax_sb)
    ax_dy = fig.add_subplot(right[3], sharex=ax_sb)
    ax_ba = fig.add_subplot(right[4], sharex=ax_sb)
    ax_pa = fig.add_subplot(right[5], sharex=ax_sb)
    
    colors = {'Baseline': '#1f77b4', 'Noise Fix': '#d62728', 'Photutils': '#2ca02c', 'AutoProf': '#9467bd'}
    markers = {'Baseline': 'o', 'Noise Fix': 's', 'Photutils': '^', 'AutoProf': 'D'}
    
    all_xax = []
    
    def _arr(isos, key): return np.array([r.get(key, np.nan) for r in isos])
    
    # Pre-extract Fix for difference calculations
    if results_dict.get('Noise Fix'):
        sma_fix = _arr(results_dict['Noise Fix']['isophotes'], 'sma')
        intens_fix = _arr(results_dict['Noise Fix']['isophotes'], 'intens')
        xax_fix = sma_fix ** 0.25
    else:
        sma_fix = intens_fix = xax_fix = None
    
    for label, res in results_dict.items():
        if res is None: continue
        
        # Handle different output formats
        if isinstance(res, dict) and 'isophotes' in res:
            isos = res['isophotes']
            sma = _arr(isos, 'sma')
            intens = _arr(isos, 'intens')
            x0 = _arr(isos, 'x0')
            y0 = _arr(isos, 'y0')
            eps = _arr(isos, 'eps')
            pa = _arr(isos, 'pa')
            x0_err = _arr(isos, 'x0_err')
            y0_err = _arr(isos, 'y0_err')
            eps_err = _arr(isos, 'eps_err')
            pa_err = _arr(isos, 'pa_err')
            intens_err = _arr(isos, 'intens_err')
        else: # AutoProf format
            sma, intens = res['sma'], res['intens']
            x0, y0 = res.get('x0'), res.get('y0')
            eps, pa = res['eps'], res['pa']
            x0_err = res.get('x0_err', np.full_like(sma, np.nan))
            y0_err = res.get('y0_err', np.full_like(sma, np.nan))
            eps_err = res.get('eps_err', np.full_like(sma, np.nan))
            pa_err = res.get('pa_err', np.full_like(sma, np.nan))
            intens_err = res.get('intens_err', np.full_like(sma, np.nan))
            if x0 is None: x0 = np.full_like(sma, np.nan)
            if y0 is None: y0 = np.full_like(sma, np.nan)

        xax = sma ** 0.25
        all_xax.append(xax)
        ok = (sma > 0) & np.isfinite(intens) & (intens > 0)
        
        c = colors.get(label, 'gray')
        m = markers.get(label, 'o')
        
        # SB (convert intens_err to log10 err approx)
        sb_err = 0.434 * intens_err[ok] / intens[ok] if np.any(np.isfinite(intens_err[ok])) else None
        ax_sb.errorbar(xax[ok], np.log10(intens[ok]), yerr=sb_err, fmt=m, color=c, 
                       markersize=4, alpha=0.6, capsize=0, label=label)
        
        # SB Difference
        if label != 'Noise Fix' and sma_fix is not None:
            intens_interp = np.interp(sma_fix, sma, intens)
            diff = (intens_interp - intens_fix) / intens_fix
            ax_sb_diff.plot(xax_fix, diff * 100, color=c, lw=1.2, alpha=0.8)
        
        # Drift
        dx = x0 - true_center[0]
        dy = y0 - true_center[1]
        
        if np.any(np.isfinite(dx[ok])):
            ax_dx.errorbar(xax[ok], dx[ok], yerr=x0_err[ok], fmt=m, color=c, markersize=4, alpha=0.6, capsize=0)
            ax_dy.errorbar(xax[ok], dy[ok], yerr=y0_err[ok], fmt=m, color=c, markersize=4, alpha=0.6, capsize=0)
            
        # Axis Ratio / PA
        ax_ba.errorbar(xax[ok], 1.0 - eps[ok], yerr=eps_err[ok], fmt=m, color=c, markersize=4, alpha=0.6, capsize=0)
        ax_pa.errorbar(xax[ok], normalize_pa_degrees(np.degrees(pa[ok])), yerr=np.degrees(pa_err[ok]), 
                       fmt=m, color=c, markersize=4, alpha=0.6, capsize=0)

    # Reference lines
    ax_sb.axhline(np.log10(sigma_bg), color='gray', ls=':', label=r'$\sigma_{bg}$')
    for ax in [ax_sb_diff, ax_dx, ax_dy]:
        ax.axhline(0, color='gray', ls='--', lw=0.8)

    # Limits
    if results_dict.get('Noise Fix'):
        f_isos = results_dict['Noise Fix']['isophotes']
        set_axis_limits_from_finite_values(ax_sb, np.log10(_arr(f_isos, 'intens')), margin_fraction=0.1)
        set_axis_limits_from_finite_values(ax_ba, 1.0 - _arr(f_isos, 'eps'), margin_fraction=0.1, lower_clip=0.0, upper_clip=1.0)
        
        pa_vals = normalize_pa_degrees(np.degrees(_arr(f_isos, 'pa')))
        pa_low, pa_high = robust_limits(pa_vals[np.isfinite(pa_vals)], 5, 95)
        ax_pa.set_ylim(pa_low - 10, pa_high + 10)
        
        dx_f = _arr(f_isos, 'x0') - true_center[0]
        dy_f = _arr(f_isos, 'y0') - true_center[1]
        c_low, c_high = robust_limits(np.concatenate([dx_f[np.isfinite(dx_f)], dy_f[np.isfinite(dy_f)]]), 5, 95)
        ax_dx.set_ylim(c_low - 2, c_high + 2)
        ax_dy.set_ylim(c_low - 2, c_high + 2)
        
        if results_dict.get('Baseline'):
            diff = (np.interp(sma_fix, _arr(results_dict['Baseline']['isophotes'], 'sma'), _arr(results_dict['Baseline']['isophotes'], 'intens')) - intens_fix) / intens_fix * 100
            set_axis_limits_from_finite_values(ax_sb_diff, diff, margin_fraction=0.2)

    # Formatting
    ax_sb.set_ylabel(r"$\log_{10}(I)$")
    ax_sb.legend(loc='upper right', fontsize=10)
    ax_sb_diff.set_ylabel(r"$\Delta I/I$ [%]")
    ax_dx.set_ylabel(r"$\Delta$X [pix]")
    ax_dy.set_ylabel(r"$\Delta$Y [pix]")
    ax_ba.set_ylabel("b/a")
    ax_pa.set_ylabel("PA [deg]")
    ax_pa.set_xlabel(r"SMA$^{0.25}$")
    
    for ax in [ax_sb, ax_sb_diff, ax_dx, ax_dy, ax_ba]:
        ax.tick_params(labelbottom=False)
        ax.grid(alpha=0.2)
    ax_pa.grid(alpha=0.2)
    
    if all_xax:
        xcat = np.concatenate(all_xax)
        set_x_limits_with_right_margin(ax_pa, xcat)

    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.06, top=0.94)
    plt.savefig(output_path, dpi=150)
    plt.close()

def _arr(isos, key): return np.array([r.get(key, np.nan) for r in isos])

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Investigate Centroid Drift')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()
    
    output_dir = resolve_output_directory("benchmarks_performance", "investigate_drift", explicit_output_directory=args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = []
    mock_dirs = [
        Path("outputs/benchmark_noise_floor/drift_mocks"),
        Path("outputs/benchmark_noise_floor/drift_mocks_noisy")
    ]
    
    for mock_dir in mock_dirs:
        noise_label = "wide" if "noisy" not in str(mock_dir) else "noisy"
        print(f"\n--- Studying {noise_label} noise regime ---")
        
        for fits_path in mock_dir.glob("*.fits"):
            name = fits_path.stem
            print(f"Investigating {name}...")
            
            with fits.open(fits_path) as hdul:
                header = hdul[0].header
                image = hdul[0].data.astype(np.float64)
                pixel_scale = header.get('PIXSCALE', 0.168)
                zeropoint = header.get('MAGZERO', 27.0)
                
            ny, nx = image.shape
            true_center = (nx/2.0, ny/2.0)
            sigma_bg = estimate_background(image)
            
            # Fits
            print("  Running Baseline...")
            res_base, _ = run_isoster_fit(image, sigma_bg=None)
            print("  Running Noise Fix...")
            res_fix, _ = run_isoster_fit(image, sigma_bg=sigma_bg)
            print("  Running Photutils...")
            res_photo, _ = run_photutils_fit(image)
            print("  Running AutoProf...")
            res_auto = run_autoprof_fit(
                image_path=fits_path,
                output_dir=output_dir / "autoprof" / name,
                galaxy_name=name,
                pixel_scale=pixel_scale,
                zeropoint=zeropoint,
                center=true_center,
                eps=0.2, pa_rad_math=0.0,
                background=0.0, background_noise=sigma_bg
            )
            
            results_dict = {
                'Baseline': res_base,
                'Noise Fix': res_fix,
                'Photutils': res_photo,
                'AutoProf': res_auto
            }
            
            plot_drift_qa(f"{name}_{noise_label}", image, results_dict, sigma_bg, true_center, 
                          output_dir / f"drift_{name}_{noise_label}.png")

            # Collect stats
            gal_stats = {'name': name, 'noise_regime': noise_label, 'sigma_bg': sigma_bg}
            for label, res in results_dict.items():
                if res is None: continue
                if isinstance(res, dict) and 'isophotes' in res:
                    isos = res['isophotes']
                    sma = np.array([iso['sma'] for iso in isos])
                    intens = np.array([iso['intens'] for iso in isos])
                    x0 = np.array([iso['x0'] for iso in isos])
                    y0 = np.array([iso['y0'] for iso in isos])
                else:
                    sma, intens, x0, y0 = res['sma'], res['intens'], res.get('x0'), res.get('y0')
                
                if x0 is None or not np.any(np.isfinite(x0)):
                    gal_stats[label] = None
                    continue
                    
                dx, dy = x0 - true_center[0], y0 - true_center[1]
                dr = np.sqrt(dx**2 + dy**2)
                
                # Outskirt stats: I < 2 * sigma_bg
                out_mask = (intens < 2.0 * sigma_bg) & np.isfinite(dr)
                if np.any(out_mask):
                    gal_stats[label] = {
                        'max_drift_out': float(np.max(dr[out_mask])),
                        'mean_drift_out': float(np.mean(dr[out_mask])),
                        'max_dr': float(np.max(dr[np.isfinite(dr)]))
                    }
                else:
                    gal_stats[label] = {'max_dr': float(np.max(dr[np.isfinite(dr)])) if np.any(np.isfinite(dr)) else None}
            
            summary.append(gal_stats)

    with open(output_dir / "drift_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nInvestigation complete. Plots and summary saved to {output_dir}")

if __name__ == "__main__":
    main()