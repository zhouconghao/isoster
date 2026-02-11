#!/usr/bin/env python3
"""
Standalone QA plotting script for Huang2013 mock galaxy tests.

This script loads saved results and generates QA figures comparing
photutils.isophote and isoster on mock galaxies.

Usage:
    python plot_qa_huang2013.py <data_file.npz> [--output figure.png] [--dpi 150]

The data file should contain:
    - mock_image: 2D mock galaxy image
    - iso_photutils: photutils isophote results (dict arrays)
    - iso_isoster: isoster isophote results (dict arrays)
    - true_intens_photutils: true intensity sampled along photutils ellipses
    - true_intens_isoster: true intensity sampled along isoster ellipses
    - true_cog_photutils: true curve of growth for photutils SMAs
    - true_cog_isoster: true curve of growth for isoster SMAs
    - metadata: dict with galaxy info, runtimes, etc.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.patches import Ellipse as MPLEllipse

# Stop code edge colors for markers
STOP_CODE_COLORS = {
    0: 'green',      # Converged
    1: 'orange',     # Flagged pixels
    2: 'gold',       # Minor issues
    3: 'coral',      # Few points
    -1: 'red',       # Gradient error
}

STOP_CODE_LABELS = {
    0: 'Stop=0',
    1: 'Stop=1',
    2: 'Stop=2',
    3: 'Stop=3',
    -1: 'Stop=-1',
}


def normalize_pa_array(pa_array, threshold=90.0):
    """Normalize PA array to remove jumps > threshold degrees."""
    if len(pa_array) == 0:
        return pa_array

    pa_norm = pa_array.copy()
    pa_norm = pa_norm % 180.0

    for i in range(1, len(pa_norm)):
        diff = pa_norm[i] - pa_norm[i-1]
        if abs(diff) > threshold:
            pa_norm[i:] = (pa_norm[i:] + 180.0) % 180.0

    return pa_norm


def plot_huang2013_qa(data_file, output_file=None, dpi=150):
    """
    Generate QA figure from saved data.

    Args:
        data_file: Path to .npz data file
        output_file: Path to output PNG (default: same name as data_file)
        dpi: Figure DPI
    """
    # Load data
    print(f"Loading data from {data_file}")
    data = np.load(data_file, allow_pickle=True)

    mock_image = data['mock_image']
    iso_photutils = data['iso_photutils'].item()  # Dict of arrays
    iso_isoster = data['iso_isoster'].item()
    true_intens_photutils = data['true_intens_photutils']
    true_intens_isoster = data['true_intens_isoster']
    true_cog_photutils = data['true_cog_photutils']
    true_cog_isoster = data['true_cog_isoster']
    model_photutils = data.get('model_photutils', None)
    model_isoster = data.get('model_isoster', None)
    metadata = data['metadata'].item()

    # Extract metadata
    galaxy_name = metadata['galaxy_name']
    redshift = metadata['redshift']
    pixel_scale = metadata['pixel_scale']
    psf_fwhm = metadata['psf_fwhm']
    runtime_photutils = metadata['runtime_photutils']
    runtime_isoster = metadata['runtime_isoster']
    speedup = metadata.get('speedup_factor', runtime_photutils / runtime_isoster)
    true_center = metadata.get('true_center', ((mock_image.shape[1]-1)/2, (mock_image.shape[0]-1)/2))

    # Conversion factor (approximate at z=0.3)
    kpc_per_arcsec = 5.26

    # Create figure with proper layout - wider to host panels better
    fig = plt.figure(figsize=(20, 16), dpi=dpi)

    # Use nested gridspecs for proper alignment
    # Main grid: 1 row, 2 columns
    main_gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.2], wspace=0.15)

    # Left column: 3 image panels (equal height, stacked vertically)
    left_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=main_gs[0], hspace=0.25)

    # Right column: 6 1D panels (SB taller, then 5 smaller panels sharing X-axis)
    right_gs = gridspec.GridSpecFromSubplotSpec(
        6, 1, subplot_spec=main_gs[1],
        height_ratios=[2, 1, 1, 1, 1, 1],  # SB is 2x taller
        hspace=0.0  # No gap for shared X-axis
    )

    # Title
    title_str = (
        f"{galaxy_name} Comparison: photutils vs isoster\n"
        f"z={redshift:.2f}, PSF={psf_fwhm:.2f}\", scale={pixel_scale:.2f}\"/pix | "
        f"photutils={runtime_photutils:.1f}s ({len(iso_photutils['sma'])} iso), "
        f"isoster={runtime_isoster:.1f}s ({len(iso_isoster['sma'])} iso), "
        f"speedup={speedup:.1f}×"
    )
    fig.suptitle(title_str, fontsize=12, fontweight='bold', y=0.995)

    # ===== LEFT COLUMN: 2D Images =====

    # Top: Input mock
    ax_input = fig.add_subplot(left_gs[0])
    vmin, vmax = np.percentile(mock_image[mock_image > 0], [0.5, 99.5])
    ax_input.imshow(mock_image, origin='lower', cmap='gray_r',
                    vmin=vmin, vmax=vmax, interpolation='nearest')
    ax_input.set_title('Mock Image', fontsize=10)
    ax_input.set_xlabel('X (pixels)', fontsize=9)
    ax_input.set_ylabel('Y (pixels)', fontsize=9)

    # Middle: photutils residual
    ax_res_p = fig.add_subplot(left_gs[1])
    if model_photutils is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            resid_p = 100.0 * (model_photutils - mock_image) / mock_image
        vlim = min(np.nanpercentile(np.abs(resid_p), 99), 2.0)
        im_p = ax_res_p.imshow(resid_p, origin='lower', cmap='RdBu_r',
                               vmin=-vlim, vmax=vlim, interpolation='nearest')
        plt.colorbar(im_p, ax=ax_res_p, fraction=0.046, pad=0.04, label='Residual (%)')

        # Overlay fitted isophotes (sparse)
        for i in range(0, len(iso_photutils['sma']), 10):
            if iso_photutils['sma'][i] > 3:
                ellipse = MPLEllipse(
                    (iso_photutils['x0'][i], iso_photutils['y0'][i]),
                    2 * iso_photutils['sma'][i],
                    2 * iso_photutils['sma'][i] * (1 - iso_photutils['eps'][i]),
                    angle=np.degrees(iso_photutils['pa'][i]),  # photutils uses radians
                    edgecolor='black', facecolor='none',
                    linewidth=0.5, alpha=0.5
                )
                ax_res_p.add_patch(ellipse)

    ax_res_p.set_title('photutils Residual', fontsize=10)
    ax_res_p.set_xlabel('X (pixels)', fontsize=9)
    ax_res_p.set_ylabel('Y (pixels)', fontsize=9)

    # Bottom: isoster residual
    ax_res_i = fig.add_subplot(left_gs[2])
    if model_isoster is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            resid_i = 100.0 * (model_isoster - mock_image) / mock_image
        vlim = min(np.nanpercentile(np.abs(resid_i), 99), 2.0)
        im_i = ax_res_i.imshow(resid_i, origin='lower', cmap='RdBu_r',
                              vmin=-vlim, vmax=vlim, interpolation='nearest')
        plt.colorbar(im_i, ax=ax_res_i, fraction=0.046, pad=0.04, label='Residual (%)')

        # Overlay fitted isophotes (sparse)
        for i in range(0, len(iso_isoster['sma']), 10):
            if iso_isoster['sma'][i] > 3:
                ellipse = MPLEllipse(
                    (iso_isoster['x0'][i], iso_isoster['y0'][i]),
                    2 * iso_isoster['sma'][i],
                    2 * iso_isoster['sma'][i] * (1 - iso_isoster['eps'][i]),
                    angle=np.degrees(iso_isoster['pa'][i]),  # isoster uses radians, convert
                    edgecolor='black', facecolor='none',
                    linewidth=0.5, alpha=0.5
                )
                ax_res_i.add_patch(ellipse)

    ax_res_i.set_title('isoster Residual', fontsize=10)
    ax_res_i.set_xlabel('X (pixels)', fontsize=9)
    ax_res_i.set_ylabel('Y (pixels)', fontsize=9)

    # ===== RIGHT COLUMN: 1D Profiles (sharing X-axis) =====

    # Convert SMA to kpc^0.25 for X-axis
    sma_kpc_p = iso_photutils['sma'] * pixel_scale * kpc_per_arcsec
    sma_kpc_i = iso_isoster['sma'] * pixel_scale * kpc_per_arcsec
    x_p = sma_kpc_p ** 0.25
    x_i = sma_kpc_i ** 0.25

    # Panel 1: Surface Brightness (tallest)
    ax_sb = fig.add_subplot(right_gs[0])

    # Plot true (dashed line)
    ax_sb.plot(x_p, true_intens_photutils, 'k--', linewidth=1.5,
              label='True', zorder=1)

    # Plot fitted with stop code edge colors
    # photutils: open circles
    for stop_code in [0, 1, 2, 3, -1]:
        mask_p = iso_photutils['stop_code'] == stop_code
        if np.any(mask_p):
            ax_sb.scatter(
                x_p[mask_p], iso_photutils['intens'][mask_p],
                marker='o', s=40, facecolors='none',
                edgecolors=STOP_CODE_COLORS.get(stop_code, 'gray'),
                linewidths=1.5, alpha=0.8,
                label=f"photutils ({STOP_CODE_LABELS[stop_code]})" if stop_code == 0 else None,
                zorder=3
            )

    # isoster: filled circles
    for stop_code in [0, 1, 2, 3, -1]:
        mask_i = iso_isoster['stop_code'] == stop_code
        if np.any(mask_i):
            ax_sb.scatter(
                x_i[mask_i], iso_isoster['intens'][mask_i],
                marker='o', s=40,
                c=STOP_CODE_COLORS.get(stop_code, 'gray'),
                alpha=0.6,
                label=f"isoster ({STOP_CODE_LABELS[stop_code]})" if stop_code == 0 else None,
                zorder=2
            )

    ax_sb.set_ylabel('Intensity', fontsize=10)
    ax_sb.set_yscale('log')
    ax_sb.legend(loc='upper right', fontsize=7, ncol=2)
    ax_sb.grid(True, alpha=0.3)
    ax_sb.set_xticklabels([])  # Remove X labels (shared axis)

    # Set Y limits based on converged points only
    valid_p = iso_photutils['intens'][iso_photutils['stop_code'] == 0]
    valid_i = iso_isoster['intens'][iso_isoster['stop_code'] == 0]
    if len(valid_p) > 0 or len(valid_i) > 0:
        all_valid = np.concatenate([valid_p, valid_i])
        all_valid = all_valid[all_valid > 0]
        if len(all_valid) > 0:
            ymin, ymax = np.percentile(all_valid, [1, 99])
            ax_sb.set_ylim(ymin * 0.5, ymax * 2.0)

    # Panel 2: Residuals
    ax_resid = fig.add_subplot(right_gs[1], sharex=ax_sb)

    with np.errstate(divide='ignore', invalid='ignore'):
        resid_p = 100.0 * (iso_photutils['intens'] - true_intens_photutils) / true_intens_photutils
        resid_i = 100.0 * (iso_isoster['intens'] - true_intens_isoster) / true_intens_isoster

    for stop_code in [0, 1, 2, 3, -1]:
        mask_p = iso_photutils['stop_code'] == stop_code
        if np.any(mask_p):
            ax_resid.scatter(
                x_p[mask_p], resid_p[mask_p],
                marker='o', s=30, facecolors='none',
                edgecolors=STOP_CODE_COLORS.get(stop_code, 'gray'),
                linewidths=1.2, alpha=0.8, zorder=3
            )

        mask_i = iso_isoster['stop_code'] == stop_code
        if np.any(mask_i):
            ax_resid.scatter(
                x_i[mask_i], resid_i[mask_i],
                marker='o', s=30,
                c=STOP_CODE_COLORS.get(stop_code, 'gray'),
                alpha=0.6, zorder=2
            )

    ax_resid.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax_resid.set_ylabel('Residual (%)', fontsize=10)
    ax_resid.grid(True, alpha=0.3)
    ax_resid.set_xticklabels([])

    # Set Y limits
    valid_resid_p = resid_p[iso_photutils['stop_code'] == 0]
    valid_resid_i = resid_i[iso_isoster['stop_code'] == 0]
    all_resid = np.concatenate([valid_resid_p, valid_resid_i])
    all_resid = all_resid[np.isfinite(all_resid)]
    if len(all_resid) > 0:
        ymax = np.percentile(np.abs(all_resid), 95)
        ax_resid.set_ylim(-ymax * 1.2, ymax * 1.2)

    # Panel 3: Curve of Growth
    ax_cog = fig.add_subplot(right_gs[2], sharex=ax_sb)

    # Plot true CoG (if available)
    if len(true_cog_photutils) > 0 and np.any(np.isfinite(true_cog_photutils)):
        ax_cog.plot(x_p, true_cog_photutils, 'k--', linewidth=1.5,
                   label='True', zorder=1)

    # Plot fitted CoG
    for stop_code in [0, 1, 2, 3, -1]:
        mask_p = iso_photutils['stop_code'] == stop_code
        if np.any(mask_p) and 'tflux_e' in iso_photutils:
            tflux_p = np.array([iso.get('tflux_e', np.nan) for iso in iso_photutils])
            valid = mask_p & np.isfinite(tflux_p)
            if np.any(valid):
                ax_cog.scatter(
                    x_p[valid], tflux_p[valid],
                    marker='o', s=30, facecolors='none',
                    edgecolors=STOP_CODE_COLORS.get(stop_code, 'gray'),
                    linewidths=1, alpha=0.8, zorder=3
                )

        mask_i = iso_isoster['stop_code'] == stop_code
        if np.any(mask_i):
            # Get tflux_e from isoster results
            tflux_i = true_cog_isoster  # Use computed true CoG for isoster
            valid = mask_i & (tflux_i > 0)
            if np.any(valid):
                ax_cog.scatter(
                    x_i[valid], tflux_i[valid],
                    marker='o', s=30,
                    c=STOP_CODE_COLORS.get(stop_code, 'gray'),
                    alpha=0.6, zorder=2
                )

    ax_cog.set_ylabel('Integrated Flux', fontsize=10)
    ax_cog.set_yscale('log')
    ax_cog.grid(True, alpha=0.3)
    ax_cog.set_xticklabels([])

    # Panel 4: Centroid
    ax_cent = fig.add_subplot(right_gs[3], sharex=ax_sb)

    # X centroid: circles
    for stop_code in [0, 1, 2, 3, -1]:
        mask_p = iso_photutils['stop_code'] == stop_code
        if np.any(mask_p):
            ax_cent.scatter(
                x_p[mask_p], iso_photutils['x0'][mask_p],
                marker='o', s=25, facecolors='none',
                edgecolors=STOP_CODE_COLORS.get(stop_code, 'gray'),
                linewidths=1, alpha=0.8, zorder=3
            )

        mask_i = iso_isoster['stop_code'] == stop_code
        if np.any(mask_i):
            ax_cent.scatter(
                x_i[mask_i], iso_isoster['x0'][mask_i],
                marker='o', s=25,
                c=STOP_CODE_COLORS.get(stop_code, 'gray'),
                alpha=0.6, zorder=2
            )

    # Y centroid: triangles
    for stop_code in [0, 1, 2, 3, -1]:
        mask_p = iso_photutils['stop_code'] == stop_code
        if np.any(mask_p):
            ax_cent.scatter(
                x_p[mask_p], iso_photutils['y0'][mask_p],
                marker='^', s=25, facecolors='none',
                edgecolors=STOP_CODE_COLORS.get(stop_code, 'gray'),
                linewidths=1, alpha=0.8, zorder=3
            )

        mask_i = iso_isoster['stop_code'] == stop_code
        if np.any(mask_i):
            ax_cent.scatter(
                x_i[mask_i], iso_isoster['y0'][mask_i],
                marker='^', s=25,
                c=STOP_CODE_COLORS.get(stop_code, 'gray'),
                alpha=0.6, zorder=2
            )

    # True center lines
    ax_cent.axhline(true_center[0], color='blue', linestyle='--',
                   linewidth=1, alpha=0.5, label=f'True X={true_center[0]:.1f}')
    ax_cent.axhline(true_center[1], color='red', linestyle='--',
                   linewidth=1, alpha=0.5, label=f'True Y={true_center[1]:.1f}')

    ax_cent.set_ylabel('Centroid (pix)', fontsize=10)
    ax_cent.legend(loc='best', fontsize=7)
    ax_cent.grid(True, alpha=0.3)
    ax_cent.set_xticklabels([])

    # Panel 5: Ellipticity
    ax_eps = fig.add_subplot(right_gs[4], sharex=ax_sb)

    for stop_code in [0, 1, 2, 3, -1]:
        mask_p = iso_photutils['stop_code'] == stop_code
        if np.any(mask_p):
            ax_eps.scatter(
                x_p[mask_p], iso_photutils['eps'][mask_p],
                marker='o', s=25, facecolors='none',
                edgecolors=STOP_CODE_COLORS.get(stop_code, 'gray'),
                linewidths=1, alpha=0.8, zorder=3
            )

        mask_i = iso_isoster['stop_code'] == stop_code
        if np.any(mask_i):
            ax_eps.scatter(
                x_i[mask_i], iso_isoster['eps'][mask_i],
                marker='o', s=25,
                c=STOP_CODE_COLORS.get(stop_code, 'gray'),
                alpha=0.6, zorder=2
            )

    ax_eps.set_ylabel('Ellipticity', fontsize=10)
    ax_eps.set_ylim(0, 1)
    ax_eps.grid(True, alpha=0.3)
    ax_eps.set_xticklabels([])

    # Panel 6: Position Angle
    ax_pa = fig.add_subplot(right_gs[5], sharex=ax_sb)

    # Normalize PAs (both photutils and isoster store PA in radians)
    pa_p = normalize_pa_array(np.degrees(iso_photutils['pa']))  # Convert from radians
    pa_i = normalize_pa_array(np.degrees(iso_isoster['pa']))  # Convert from radians

    for stop_code in [0, 1, 2, 3, -1]:
        mask_p = iso_photutils['stop_code'] == stop_code
        if np.any(mask_p):
            ax_pa.scatter(
                x_p[mask_p], pa_p[mask_p],
                marker='o', s=25, facecolors='none',
                edgecolors=STOP_CODE_COLORS.get(stop_code, 'gray'),
                linewidths=1, alpha=0.8, zorder=3
            )

        mask_i = iso_isoster['stop_code'] == stop_code
        if np.any(mask_i):
            ax_pa.scatter(
                x_i[mask_i], pa_i[mask_i],
                marker='o', s=25,
                c=STOP_CODE_COLORS.get(stop_code, 'gray'),
                alpha=0.6, zorder=2
            )

    ax_pa.set_ylabel('PA (deg)', fontsize=10)
    ax_pa.set_xlabel('SMA$^{0.25}$ (kpc$^{0.25}$)', fontsize=10)
    ax_pa.set_ylim(0, 180)
    ax_pa.grid(True, alpha=0.3)

    # Determine output path
    if output_file is None:
        output_file = Path(data_file).with_suffix('.png')

    # Save figure
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"Saved QA figure to {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate QA figure from saved Huang2013 test data'
    )
    parser.add_argument('data_file', type=str,
                       help='Path to .npz data file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output PNG path (default: same as data file)')
    parser.add_argument('--dpi', type=int, default=150,
                       help='Figure DPI (default: 150)')

    args = parser.parse_args()

    if not Path(args.data_file).exists():
        print(f"ERROR: Data file not found: {args.data_file}")
        sys.exit(1)

    plot_huang2013_qa(args.data_file, args.output, args.dpi)


if __name__ == '__main__':
    main()
