#!/usr/bin/env python
# ruff: noqa: E402
"""
Benchmark: Explicit Noise Floor & Gradient Error Fix
isoster (Baseline) vs isoster (Noise Floor Fix) vs photutils vs AutoProf

This benchmark evaluates the stability improvement and error bar correctness
introduced by the explicit noise floor (sigma_bg) and corrected geometric
error propagation.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import isoster
from benchmarks.utils.autoprof_adapter import run_autoprof_fit
from isoster.config import IsosterConfig
from isoster.model import build_isoster_model
from isoster.output_paths import resolve_output_directory
from isoster.plotting import (
    configure_qa_plot_style,
    derive_arcsinh_parameters,
    draw_isophote_overlays,
    make_arcsinh_display_from_parameters,
    normalize_pa_degrees,
    robust_limits,
    set_axis_limits_from_finite_values,
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
        # Convert to dict format
        isophotes = []
        for iso in isolist:
            isophotes.append({
                'sma': iso.sma,
                'intens': iso.intens,
                'eps': iso.eps,
                'pa': iso.pa,
                'x0': iso.x0,
                'y0': iso.y0,
                'eps_err': iso.ellip_err,
                'pa_err': iso.pa_err,
                'stop_code': 0
            })
        return {'isophotes': isophotes}, runtime
    except Exception:
        return None, 0

def plot_noise_floor_qa(name, image, res_base, res_fix, res_photo, res_auto, sigma_bg, output_path):
    """Generate high-quality comparison figure with 2D panels and overhauled 1D profiles."""
    configure_qa_plot_style()

    # 1. Models and Residuals
    model_base = build_isoster_model(image.shape, res_base['isophotes'], use_harmonics=True)
    model_fix = build_isoster_model(image.shape, res_fix['isophotes'], use_harmonics=True)

    residual_base = np.where(np.isfinite(image), image - model_base, np.nan)
    residual_fix = np.where(np.isfinite(image), image - model_fix, np.nan)

    # 2. Figure Layout
    fig = plt.figure(figsize=(16.0, 16.0))
    # Left: 3 image panels
    # Right: 5 profile panels (SB, SB Diff, Centroids, b/a, PA)
    outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.8], wspace=0.2)

    left = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[0], hspace=0.15)
    right = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=outer[1],
                                             height_ratios=[2.2, 1.0, 1.0, 1.0, 1.0], hspace=0.0)

    fig.suptitle(f"Benchmark: {name} - Noise Floor and Error Fix Improvement", fontsize=18, y=0.98)

    # 3. Left Panels (Images)
    ref_low, ref_high, ref_scale, ref_vmax = derive_arcsinh_parameters(image)

    ax_img = fig.add_subplot(left[0])
    img_disp, vmin, vmax = make_arcsinh_display_from_parameters(image, ref_low, ref_high, ref_scale, ref_vmax)
    ax_img.imshow(img_disp, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    draw_isophote_overlays(ax_img, res_fix['isophotes'], step=max(1, len(res_fix['isophotes'])//15), edge_color='orangered')
    ax_img.set_title("Data + Fix Isophotes", fontsize=12)

    all_res = np.concatenate([residual_base[np.isfinite(residual_base)],
                             residual_fix[np.isfinite(residual_fix)]])
    res_limit = float(np.clip(np.nanpercentile(np.abs(all_res), 99.0) if all_res.size else 1.0, 0.05, None))

    for i, (res_map, title) in enumerate([(residual_base, "Baseline Residual"), (residual_fix, "Noise Floor Fix Residual")], 1):
        ax = fig.add_subplot(left[i])
        ax.imshow(res_map, origin="lower", cmap="coolwarm", vmin=-res_limit, vmax=res_limit, interpolation="nearest")
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    # 4. Right Panels (1D Profiles)
    ax_sb = fig.add_subplot(right[0])
    ax_sb_diff = fig.add_subplot(right[1], sharex=ax_sb)
    ax_cen = fig.add_subplot(right[2], sharex=ax_sb)
    ax_ba = fig.add_subplot(right[3], sharex=ax_sb)
    ax_pa = fig.add_subplot(right[4], sharex=ax_sb)

    def _arr(isos, key): return np.array([r.get(key, np.nan) for r in isos])

    conditions = [
        (res_base['isophotes'], 'Baseline', '#1f77b4', 'o'),
        (res_fix['isophotes'], 'Noise Floor Fix', '#d62728', 's')
    ]
    if res_photo:
        conditions.append((res_photo['isophotes'], 'Photutils', '#2ca02c', '^'))
    if res_auto:
        # Wrap AutoProf arrays into a list of dicts for _arr
        auto_isos = []
        for i in range(len(res_auto['sma'])):
            auto_isos.append({
                'sma': res_auto['sma'][i],
                'intens': res_auto['intens'][i],
                'eps': res_auto['eps'][i],
                'pa': res_auto['pa'][i],
                'x0': res_auto.get('x0', np.full_like(res_auto['sma'], np.nan))[i],
                'y0': res_auto.get('y0', np.full_like(res_auto['sma'], np.nan))[i],
                'eps_err': res_auto['eps_err'][i],
                'pa_err': res_auto['pa_err'][i]
            })
        conditions.append((auto_isos, 'AutoProf', '#9467bd', 'D'))

    all_sma_pow = []

    # Pre-extract Fix for difference calculations
    sma_fix = _arr(res_fix['isophotes'], 'sma')
    intens_fix = _arr(res_fix['isophotes'], 'intens')
    xax_fix = sma_fix ** 0.25

    for isos, label, col, mrk in conditions:
        sma = _arr(isos, 'sma')
        intens = _arr(isos, 'intens')
        eps = _arr(isos, 'eps')
        pa_rad = _arr(isos, 'pa')
        x0, y0 = _arr(isos, 'x0'), _arr(isos, 'y0')
        eps_err = _arr(isos, 'eps_err')
        pa_err_rad = _arr(isos, 'pa_err')

        xax = sma ** 0.25
        all_sma_pow.append(xax)
        ok = (sma > 0) & np.isfinite(intens) & (intens > 0)

        # SB
        ax_sb.scatter(xax[ok], np.log10(intens[ok]), color=col, marker=mrk, s=15, alpha=0.6, label=label)

        # Centroids (offsets relative to median of fix)
        med_x, med_y = np.nanmedian(_arr(res_fix['isophotes'], 'x0')), np.nanmedian(_arr(res_fix['isophotes'], 'y0'))
        if np.any(np.isfinite(x0[ok])):
            ax_cen.plot(xax[ok], x0[ok] - med_x, color=col, ls='-', lw=1.0, alpha=0.8)
            ax_cen.plot(xax[ok], y0[ok] - med_y, color=col, ls='--', lw=1.0, alpha=0.8)

        # Axis Ratio
        ax_ba.errorbar(xax[ok], 1.0 - eps[ok], yerr=eps_err[ok], fmt=mrk, color=col, markersize=4, alpha=0.4, capsize=0)

        # PA
        pa_deg = normalize_pa_degrees(np.degrees(pa_rad))
        pa_err_deg = np.degrees(pa_err_rad)
        ax_pa.errorbar(xax[ok], pa_deg[ok], yerr=pa_err_deg[ok], fmt=mrk, color=col, markersize=4, alpha=0.4, capsize=0)

        # SB Difference (relative to Fix)
        if label != 'Noise Floor Fix':
            # Interpolate to match Fix SMA grid
            intens_interp = np.interp(sma_fix, sma, intens)
            diff = (intens_interp - intens_fix) / intens_fix
            ax_sb_diff.plot(xax_fix, diff * 100, color=col, lw=1.2, alpha=0.8, label=f'{label} - Fix')

    # Mark sigma_bg level
    ax_sb.axhline(np.log10(sigma_bg), color='gray', linestyle=':', lw=1.0, label=r'$\sigma_{bg}$')
    ax_sb_diff.axhline(0, color='gray', ls='--', lw=0.8)
    ax_cen.axhline(0, color='gray', ls='--', lw=0.8)

    # Robust Y-limits
    set_axis_limits_from_finite_values(ax_sb, np.log10(_arr(res_fix['isophotes'], 'intens')), margin_fraction=0.1)
    set_axis_limits_from_finite_values(ax_ba, 1.0 - _arr(res_fix['isophotes'], 'eps'), margin_fraction=0.1, lower_clip=0.0, upper_clip=1.0)
    set_axis_limits_from_finite_values(ax_sb_diff, (np.interp(sma_fix, _arr(res_base['isophotes'], 'sma'), _arr(res_base['isophotes'], 'intens')) - intens_fix) / intens_fix * 100, margin_fraction=0.2)
    set_axis_limits_from_finite_values(ax_cen, np.concatenate([_arr(res_fix['isophotes'], 'x0') - np.nanmedian(_arr(res_fix['isophotes'], 'x0')), _arr(res_fix['isophotes'], 'y0') - np.nanmedian(_arr(res_fix['isophotes'], 'y0'))]), margin_fraction=0.2)

    pa_vals = normalize_pa_degrees(np.degrees(_arr(res_fix['isophotes'], 'pa')))
    pa_low, pa_high = robust_limits(pa_vals[np.isfinite(pa_vals)], 5, 95)
    ax_pa.set_ylim(pa_low - 10, pa_high + 10)

    # Formatting
    ax_sb.set_ylabel(r"$\log_{10}(I)$")
    ax_sb.legend(loc='upper right', fontsize=10)
    ax_sb_diff.set_ylabel(r"$\Delta I/I$ [%]")
    ax_cen.set_ylabel(r"$\Delta$X, $\Delta$Y [pix]")
    ax_ba.set_ylabel("b/a")
    ax_pa.set_ylabel("PA [deg]")
    ax_pa.set_xlabel(r"SMA$^{0.25}$ (pixel$^{0.25}$)")

    for ax in [ax_sb, ax_sb_diff, ax_cen, ax_ba]:
        ax.tick_params(labelbottom=False)
        ax.grid(alpha=0.2)
    ax_pa.grid(alpha=0.2)

    if all_sma_pow:
        xcat = np.concatenate(all_sma_pow)
        set_x_limits_with_right_margin(ax_pa, xcat)

    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.06, top=0.94)
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
            header = hdul[0].header
            image = hdul[0].data.astype(np.float64)
            pixel_scale = header.get('PIXSCALE', 0.168)
            zeropoint = header.get('MAGZERO', 27.0)

        sigma_bg = estimate_background(image)
        print(f"  Estimated sigma_bg: {sigma_bg:.4f}")

        # 1. Baseline (no sigma_bg, no corrected errors)
        print("  Running Baseline isoster...")
        res_base, time_base = run_isoster_fit(image, sigma_bg=None, use_corrected_errors=False)

        # 2. Fix (with sigma_bg and corrected errors)
        print("  Running Noise Floor Fix isoster...")
        res_fix, time_fix = run_isoster_fit(image, sigma_bg=sigma_bg, use_corrected_errors=True)

        # 3. Photutils
        print("  Running photutils...")
        res_photo, time_photo = run_photutils_fit(image)

        # 4. AutoProf
        print("  Running AutoProf...")
        ny, nx = image.shape
        res_auto = run_autoprof_fit(
            image_path=fits_path,
            output_dir=output_dir / "autoprof" / name,
            galaxy_name=name,
            pixel_scale=pixel_scale,
            zeropoint=zeropoint,
            center=(nx/2.0, ny/2.0),
            eps=0.2,
            pa_rad_math=0.0,
            background=0.0,
            background_noise=sigma_bg
        )
        time_auto = res_auto.get('runtime_s', 0) if res_auto else 0

        # Generate QA
        plot_noise_floor_qa(name, image, res_base, res_fix, res_photo, res_auto, sigma_bg, output_dir / f"comparison_{name}.png")

        summary.append({
            'name': name,
            'sigma_bg': sigma_bg,
            'time_base': time_base,
            'time_fix': time_fix,
            'time_photo': time_photo,
            'time_auto': time_auto,
            'speedup_fix_vs_base': time_base / time_fix if time_fix > 0 else None,
            'speedup_fix_vs_photo': time_photo / time_fix if time_fix > 0 and time_photo > 0 else None,
            'speedup_fix_vs_auto': time_auto / time_fix if time_fix > 0 and time_auto > 0 else None,
            'n_isophotes_base': len(res_base['isophotes']),
            'n_isophotes_fix': len(res_fix['isophotes']),
            'avg_eps_err_base': float(np.nanmean([iso['eps_err'] for iso in res_base['isophotes'][-10:]])),
            'avg_eps_err_fix': float(np.nanmean([iso['eps_err'] for iso in res_fix['isophotes'][-10:]])),
            'eps_std_base': float(np.nanstd([iso['eps'] for iso in res_base['isophotes'][-10:]])),
            'eps_std_fix': float(np.nanstd([iso['eps'] for iso in res_fix['isophotes'][-10:]])),
            'pa_std_base': float(np.nanstd([iso['pa'] for iso in res_base['isophotes'][-10:]])),
            'pa_std_fix': float(np.nanstd([iso['pa'] for iso in res_fix['isophotes'][-10:]])),
        })

        print(f"  Runtimes: Base={time_base:.3f}s, Fix={time_fix:.3f}s, Photo={time_photo:.3f}s, Auto={time_auto:.3f}s")

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBenchmark complete. Outputs saved to {output_dir}")

if __name__ == "__main__":
    main()
