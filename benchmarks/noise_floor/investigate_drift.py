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
    from photutils.isophote import Ellipse, EllipseGeometry, build_ellipse_model
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
    # Truth center for mockgal is (nx-1)/2
    config = IsosterConfig(
        x0=(nx - 1) / 2.0, y0=(ny - 1) / 2.0, eps=0.2, pa=0.0,
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
    geometry = EllipseGeometry(x0=(nx - 1) / 2.0, y0=(ny - 1) / 2.0, sma=10.0, eps=0.2, pa=0.0)
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
        
        # Build native photutils model
        model = build_ellipse_model(image.shape, isolist)
        
        isophotes = []
        for iso in isolist:
            isophotes.append({
                'sma': iso.sma, 'intens': iso.intens, 'eps': iso.eps, 'pa': iso.pa,
                'x0': iso.x0, 'y0': iso.y0, 'eps_err': iso.ellip_err, 'pa_err': iso.pa_err,
                'x0_err': iso.x0_err, 'y0_err': iso.y0_err,
                'intens_err': iso.int_err,
                'stop_code': 0
            })
        return {'isophotes': isophotes, 'model': model}, runtime
    except Exception as e:
        print(f"Photutils failed: {e}")
        return None, 0

def get_huang2013_truth(header):
    """Extract multi-component truth from mock fits header."""
    ncomp = header.get('NCOMP', 0)
    truth = []
    for i in range(1, ncomp + 1):
        truth.append({
            'sma': header.get(f'RE_PX{i}', np.nan),
            'eps': header.get(f'ELLIP{i}', np.nan),
            'pa': np.radians(header.get(f'PA{i}', np.nan)),
            'label': f'Comp {i} Re'
        })
    return truth

def plot_drift_qa(name, image, results_dict, sigma_bg, true_center, output_path, truth_header=None):
    """
    Generate high-quality investigation figure focusing on Centroid Drift.
    Left: Mock Image + Residual maps for ALL methods.
    Right: 1-D profiles with scatter, error bars, and Huang2013 truth highlights.
    """
    configure_qa_plot_style()
    
    # 1. Models and Residuals collection
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
        
    if results_dict.get('AutoProf') and results_dict['AutoProf'] is not None:
        auto_res = results_dict['AutoProf']
        if 'model_fits_path' in auto_res and Path(auto_res['model_fits_path']).exists():
            with fits.open(auto_res['model_fits_path']) as hdul:
                m_auto_raw = hdul[1].data.astype(np.float64)
            m_auto = np.full(image.shape, np.nan)
            ny, nx = image.shape
            my, mx = m_auto_raw.shape
            cy, cx = int(true_center[1]), int(true_center[0])
            y_start = max(0, cy - my // 2)
            y_end = min(ny, y_start + my)
            x_start = max(0, cx - mx // 2)
            x_end = min(nx, x_start + mx)
            sy_start = 0 if y_start > 0 else (my // 2 - cy)
            sy_end = my if y_end < ny else (sy_start + (y_end - y_start))
            sx_start = 0 if x_start > 0 else (mx // 2 - cx)
            sx_end = mx if x_end < nx else (sx_start + (x_end - x_start))
            m_auto[y_start:y_end, x_start:x_end] = m_auto_raw[sy_start:sy_end, sx_start:sx_end]
            residuals.append(np.where(np.isfinite(image), image - m_auto, np.nan))
            res_titles.append("AutoProf Residual")
    elif results_dict.get('Photutils') and results_dict['Photutils'] is not None:
        if 'model' in results_dict['Photutils']:
            m_photo = results_dict['Photutils']['model']
            residuals.append(np.where(np.isfinite(image), image - m_photo, np.nan))
            res_titles.append("Photutils Residual")

    n_left_panels = 1 + len(residuals)
    fig = plt.figure(figsize=(16.0, 4.0 * max(n_left_panels, 5)))
    outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.8], wspace=0.2)
    
    left = gridspec.GridSpecFromSubplotSpec(n_left_panels, 1, subplot_spec=outer[0], hspace=0.15)
    right = gridspec.GridSpecFromSubplotSpec(6, 1, subplot_spec=outer[1], 
                                             height_ratios=[2.0, 1.0, 1.0, 1.0, 1.0, 1.0], hspace=0.0)
    
    fig.suptitle(f"Centroid Drift Investigation: {name}", fontsize=20, y=0.99)
    
    # Left Panels (Images)
    ref_low, ref_high, ref_scale, ref_vmax = derive_arcsinh_parameters(image)
    img_disp, vmin, vmax = make_arcsinh_display_from_parameters(image, ref_low, ref_high, ref_scale, ref_vmax)
    
    ax_img = fig.add_subplot(left[0])
    ax_img.imshow(img_disp, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    ax_img.set_title("Data + Fitted Isophotes", fontsize=12)
    
    # Overplot truth components if available
    truth_comps = get_huang2013_truth(truth_header) if truth_header else []
    for comp in truth_comps:
        draw_isophote_overlays(ax_img, [{
            'sma': comp['sma'], 'eps': comp['eps'], 'pa': comp['pa'], 
            'x0': true_center[0], 'y0': true_center[1]
        }], edge_color='white', line_width=2.0, alpha=0.8)
    
    if results_dict.get('Noise Fix'):
        draw_isophote_overlays(ax_img, results_dict['Noise Fix']['isophotes'], 
                               step=max(1, len(results_dict['Noise Fix']['isophotes'])//15), 
                               edge_color='orangered', line_width=0.8, alpha=0.5)
    
    if residuals:
        all_abs = np.abs(np.concatenate([r[np.isfinite(r)] for r in residuals]))
        res_limit = float(np.clip(np.nanpercentile(all_abs, 99.0) if all_abs.size else 1.0, 0.05, None))
        for i, (res_map, title) in enumerate(zip(residuals, res_titles), 1):
            ax = fig.add_subplot(left[i])
            ax.imshow(res_map, origin="lower", cmap="coolwarm", vmin=-res_limit, vmax=res_limit, interpolation="nearest")
            ax.set_title(title, fontsize=12)
            ax.set_xticks([]); ax.set_yticks([])

    # Right Panels (Profiles)
    ax_sb = fig.add_subplot(right[0])
    ax_sb_diff = fig.add_subplot(right[1], sharex=ax_sb)
    ax_dx = fig.add_subplot(right[2], sharex=ax_sb)
    ax_dy = fig.add_subplot(right[3], sharex=ax_sb)
    ax_ba = fig.add_subplot(right[4], sharex=ax_sb)
    ax_pa_panel = fig.add_subplot(right[5], sharex=ax_sb)
    
    colors = {'Baseline': '#1f77b4', 'Noise Fix': '#d62728', 'Photutils': '#2ca02c', 'AutoProf': '#9467bd'}
    markers = {'Baseline': 'o', 'Noise Fix': 's', 'Photutils': '^', 'AutoProf': 'D'}
    
    def _arr(isos, key): return np.array([r.get(key, np.nan) for r in isos])
    
    # Reference condition for differences
    ref_cond = 'Noise Fix' if results_dict.get('Noise Fix') else 'Baseline'
    sma_ref = _arr(results_dict[ref_cond]['isophotes'], 'sma')
    intens_ref = _arr(results_dict[ref_cond]['isophotes'], 'intens')
    xax_ref = sma_ref ** 0.25
    
    # Anchoring PA to the first truth component if possible
    pa_anchor = np.degrees(truth_comps[0]['pa']) if truth_comps else None

    # Storage for axis scaling
    all_sb, all_ba, all_pa, all_drift, all_diff, all_xax = [], [], [], [], [], []

    for label, res in results_dict.items():
        if res is None: continue
        if isinstance(res, dict) and 'isophotes' in res:
            isos = res['isophotes']
            sma, intens = _arr(isos, 'sma'), _arr(isos, 'intens')
            x0, y0 = _arr(isos, 'x0'), _arr(isos, 'y0')
            eps, pa_rad = _arr(isos, 'eps'), _arr(isos, 'pa')
            x0_err, y0_err = _arr(isos, 'x0_err'), _arr(isos, 'y0_err')
            eps_err, pa_err_rad = _arr(isos, 'eps_err'), _arr(isos, 'pa_err')
            intens_err = _arr(isos, 'intens_err')
        else: # AutoProf
            sma, intens = res['sma'], res['intens']
            x0 = res.get('x0', np.full_like(sma, true_center[0]))
            y0 = res.get('y0', np.full_like(sma, true_center[1]))
            eps, pa_rad = res['eps'], res['pa']
            x0_err = res.get('x0_err', np.full_like(sma, np.nan))
            y0_err = res.get('y0_err', np.full_like(sma, np.nan))
            eps_err, pa_err_rad = res['eps_err'], res['pa_err']
            intens_err = res.get('intens_err', np.full_like(sma, np.nan))

        xax = sma ** 0.25
        all_xax.append(xax)
        ok = (sma > 1.0) & np.isfinite(intens) & (intens > 0)
        c, m = colors.get(label, 'gray'), markers.get(label, 'o')
        
        # SB
        ax_sb.errorbar(xax[ok], np.log10(intens[ok]), yerr=0.434*intens_err[ok]/intens[ok] if np.any(np.isfinite(intens_err)) else None, 
                       fmt=m, color=c, markersize=4, alpha=0.6, capsize=0, label=label)
        all_sb.append(np.log10(intens[ok]))
        
        # SB Difference (scatter + errorbar)
        if label != ref_cond:
            valid_intens = np.isfinite(intens) & (intens > 0)
            if np.sum(valid_intens) > 1:
                intens_interp = np.interp(sma_ref, sma[valid_intens], intens[valid_intens], left=np.nan, right=np.nan)
                diff = (intens_interp - intens_ref) / intens_ref
                diff_err = np.nan
                if np.any(np.isfinite(intens_err[valid_intens])):
                    err_interp = np.interp(sma_ref, sma[valid_intens], intens_err[valid_intens], left=np.nan, right=np.nan)
                    diff_err = np.abs(diff) * (err_interp / intens_interp)
                ax_sb_diff.errorbar(xax_ref, diff * 100, yerr=diff_err * 100, fmt=m, color=c, markersize=4, alpha=0.6, capsize=0)
                all_diff.append(diff[np.isfinite(diff)] * 100)
            
        # Drift
        dx, dy = x0 - true_center[0], y0 - true_center[1]
        ax_dx.errorbar(xax[ok], dx[ok], yerr=x0_err[ok], fmt=m, color=c, markersize=4, alpha=0.6, capsize=0)
        ax_dy.errorbar(xax[ok], dy[ok], yerr=y0_err[ok], fmt=m, color=c, markersize=4, alpha=0.6, capsize=0)
        all_drift.extend([dx[ok], dy[ok]])
        
        # Axis Ratio
        ax_ba.errorbar(xax[ok], 1.0 - eps[ok], yerr=eps_err[ok], fmt=m, color=c, markersize=4, alpha=0.6, capsize=0)
        all_ba.append(1.0 - eps[ok])
        
        # PA
        pa_deg = normalize_pa_degrees(np.degrees(pa_rad[ok]), anchor=pa_anchor)
        ax_pa_panel.errorbar(xax[ok], pa_deg, yerr=np.degrees(pa_err_rad[ok]), fmt=m, color=c, markersize=4, alpha=0.6, capsize=0)
        all_pa.append(pa_deg)

    # Truth highlights on profiles
    for comp in truth_comps:
        xcomp = comp['sma'] ** 0.25
        ax_ba.scatter(xcomp, 1.0 - comp['eps'], s=150, marker='*', facecolor='gold', edgecolor='black', zorder=10, label='Truth' if comp==truth_comps[0] else None)
        ax_pa_panel.scatter(xcomp, normalize_pa_degrees(np.array([np.degrees(comp['pa'])]), anchor=pa_anchor), s=150, marker='*', facecolor='gold', edgecolor='black', zorder=10)

    # Reference lines
    ax_sb.axhline(np.log10(sigma_bg), color='gray', ls=':', label=r'$\sigma_{bg}$')
    for ax in [ax_sb_diff, ax_dx, ax_dy]: ax.axhline(0, color='gray', ls='--', lw=0.8)

    # Axis Limits (Global)
    if all_xax: set_x_limits_with_right_margin(ax_pa_panel, np.concatenate(all_xax))
    set_axis_limits_from_finite_values(ax_sb, np.concatenate(all_sb), margin_fraction=0.1)
    set_axis_limits_from_finite_values(ax_ba, np.concatenate(all_ba), margin_fraction=0.1, lower_clip=0.0, upper_clip=1.0)
    set_axis_limits_from_finite_values(ax_dx, np.concatenate(all_drift), margin_fraction=0.3)
    set_axis_limits_from_finite_values(ax_dy, np.concatenate(all_drift), margin_fraction=0.3)
    if all_diff: set_axis_limits_from_finite_values(ax_sb_diff, np.concatenate(all_diff), margin_fraction=0.2)
    
    if all_pa:
        p_vals = np.concatenate(all_pa)
        p_low, p_high = robust_limits(p_vals[np.isfinite(p_vals)], 2, 98)
        ax_pa_panel.set_ylim(p_low - 20, p_high + 20)

    # Formatting
    ax_sb.set_ylabel(r"$\log_{10}(I)$")
    ax_sb.legend(loc='upper right', fontsize=10)
    ax_sb_diff.set_ylabel(r"$\Delta I/I_{fix}$ [%]")
    ax_dx.set_ylabel(r"$\Delta$X [pix]")
    ax_dy.set_ylabel(r"$\Delta$Y [pix]")
    ax_ba.set_ylabel("b/a")
    ax_pa_panel.set_ylabel("PA [deg]")
    ax_pa_panel.set_xlabel(r"SMA$^{0.25}$ (pixel$^{0.25}$)")
    
    for ax in [ax_sb, ax_sb_diff, ax_dx, ax_dy, ax_ba]:
        ax.tick_params(labelbottom=False); ax.grid(alpha=0.2)
    ax_pa_panel.grid(alpha=0.2)

    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.06, top=0.94)
    plt.savefig(output_path, dpi=150); plt.close()

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
            true_center = ((nx - 1) / 2.0, (ny - 1) / 2.0)
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
            
            plot_drift_qa(f"{name}", image, results_dict, sigma_bg, true_center, 
                          output_dir / f"drift_{name}.png", truth_header=header)

            # Collect stats
            gal_stats = {'name': name, 'noise_regime': noise_label, 'sigma_bg': sigma_bg}
            for label, res in results_dict.items():
                if res is None: continue
                if isinstance(res, dict) and 'isophotes' in res:
                    isos = res['isophotes']
                    sma, intens = np.array([iso['sma'] for iso in isos]), np.array([iso['intens'] for iso in isos])
                    x0, y0 = np.array([iso['x0'] for iso in isos]), np.array([iso['y0'] for iso in isos])
                else:
                    sma, intens = res['sma'], res['intens']
                    x0 = res.get('x0', np.full_like(sma, true_center[0]))
                    y0 = res.get('y0', np.full_like(sma, true_center[1]))
                
                dx, dy = x0 - true_center[0], y0 - true_center[1]
                dr = np.sqrt(dx**2 + dy**2)
                out_mask = (intens < 2.0 * sigma_bg) & np.isfinite(dr)
                if np.any(out_mask):
                    gal_stats[label] = {'max_drift_out': float(np.max(dr[out_mask])), 'mean_drift_out': float(np.mean(dr[out_mask])), 'max_dr': float(np.max(dr[np.isfinite(dr)]))}
                else:
                    gal_stats[label] = {'max_dr': float(np.max(dr[np.isfinite(dr)])) if np.any(np.isfinite(dr)) else None}
            summary.append(gal_stats)

    with open(output_dir / "drift_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nInvestigation complete. Plots and summary saved to {output_dir}")

if __name__ == "__main__":
    main()
