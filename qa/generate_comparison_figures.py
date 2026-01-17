"""
QA comparison: isoster vs photutils.isophote vs truth

Generate comprehensive QA figures comparing isoster and photutils results
against ground truth for challenging test cases.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

from isoster import fit_image
from isoster.config import IsosterConfig

try:
    from photutils.isophote import EllipseGeometry, Ellipse
    PHOTUTILS_AVAILABLE = True
except ImportError:
    PHOTUTILS_AVAILABLE = False
    print("Warning: photutils not available, skipping photutils comparison")


def create_sersic_model(R_e, n, I_e, eps, pa, noise_level=None, oversample=1, seed=42):
    """Create a centered 2D Sersic profile with optional noise.

    Per CLAUDE.md: Image half-size should be >= 10 * R_e (15x for better coverage)
    """
    half_size = max(int(15 * R_e), 150)
    shape = (2 * half_size, 2 * half_size)
    x0, y0 = half_size, half_size

    b_n = 1.9992 * n - 0.3271

    if oversample > 1:
        oversamp_shape = (shape[0] * oversample, shape[1] * oversample)
        y = np.arange(oversamp_shape[0]) / oversample
        x = np.arange(oversamp_shape[1]) / oversample
        yy, xx = np.meshgrid(y, x, indexing='ij')

        dx = xx - x0
        dy = yy - y0
        x_rot = dx * np.cos(pa) + dy * np.sin(pa)
        y_rot = -dx * np.sin(pa) + dy * np.cos(pa)
        r_ell = np.sqrt(x_rot**2 + (y_rot / (1 - eps))**2)

        image_oversamp = I_e * np.exp(-b_n * ((r_ell / R_e)**(1/n) - 1))

        image = np.zeros(shape)
        for i in range(oversample):
            for j in range(oversample):
                image += image_oversamp[i::oversample, j::oversample]
        image /= oversample**2
    else:
        y = np.arange(shape[0])
        x = np.arange(shape[1])
        yy, xx = np.meshgrid(y, x, indexing='ij')

        dx = xx - x0
        dy = yy - y0
        x_rot = dx * np.cos(pa) + dy * np.sin(pa)
        y_rot = -dx * np.sin(pa) + dy * np.cos(pa)
        r_ell = np.sqrt(x_rot**2 + (y_rot / (1 - eps))**2)

        image = I_e * np.exp(-b_n * ((r_ell / R_e)**(1/n) - 1))

    # Add noise if requested
    if noise_level is not None and noise_level > 0:
        rng = np.random.RandomState(seed)
        image += rng.normal(0, noise_level, image.shape)

    # Compute true profile for comparison
    r_true = np.linspace(0.1, 8 * R_e, 100)
    intens_true = I_e * np.exp(-b_n * ((r_true / R_e)**(1/n) - 1))

    return image, (x0, y0), shape, r_true, intens_true


def run_isoster(image, x0, y0, eps, pa, R_e):
    """Run isoster on image."""
    config = IsosterConfig(
        x0=x0, y0=y0,
        eps=eps, pa=pa,
        sma0=10.0, minsma=3.0, maxsma=8 * R_e,
        astep=0.15,
        minit=10, maxit=50,
        conver=0.05,
        maxgerr=1.0 if eps > 0.6 else 0.5,
        use_eccentric_anomaly=(eps > 0.3),
    )

    results = fit_image(image, None, config)
    isophotes = results['isophotes']

    return isophotes


def run_photutils(image, x0, y0, eps, pa, R_e):
    """Run photutils.isophote on image."""
    if not PHOTUTILS_AVAILABLE:
        return []

    sma0 = 10.0
    minsma = 3.0
    maxsma = 8 * R_e

    # Create geometry
    geometry = EllipseGeometry(x0, y0, sma0, eps, pa)

    # Create ellipse fitter
    ellipse = Ellipse(image, geometry)

    # Fit isophotes
    try:
        isolist = ellipse.fit_image()
    except Exception as e:
        print(f"photutils failed: {e}")
        return []

    # Convert to isoster format
    isophotes = []
    for iso in isolist:
        if iso.sma < minsma or iso.sma > maxsma:
            continue

        isophotes.append({
            'sma': iso.sma,
            'intens': iso.intens,
            'eps': iso.eps,
            'pa': iso.pa,
            'x0': iso.x0,
            'y0': iso.y0,
            'stop_code': 0 if iso.stop_code == 0 else -1,
            'niter': iso.niter,
        })

    return isophotes


def normalize_pa(pa):
    """Normalize PA to [0, π) range."""
    pa = np.asarray(pa)
    return pa % np.pi


def plot_qa_comparison(image, isophotes_iso, isophotes_phu, r_true, intens_true,
                       case_name, params, output_path):
    """Create comprehensive QA comparison figure.

    Per CLAUDE.md:
    - Vertical subplot layout sharing X-axis
    - PA subplot with Y-axis label
    - Angle normalization to [0, π)
    """
    # Extract isoster data
    sma_iso = np.array([iso['sma'] for iso in isophotes_iso])
    intens_iso = np.array([iso['intens'] for iso in isophotes_iso])
    eps_iso = np.array([iso['eps'] for iso in isophotes_iso])
    pa_iso = np.array([iso['pa'] for iso in isophotes_iso])
    stop_iso = np.array([iso['stop_code'] for iso in isophotes_iso])

    # Separate converged and failed
    converged_iso = (stop_iso == 0)
    failed_iso = ~converged_iso

    # Extract photutils data if available
    if isophotes_phu and len(isophotes_phu) > 0:
        sma_phu = np.array([iso['sma'] for iso in isophotes_phu])
        intens_phu = np.array([iso['intens'] for iso in isophotes_phu])
        eps_phu = np.array([iso['eps'] for iso in isophotes_phu])
        pa_phu = np.array([iso['pa'] for iso in isophotes_phu])
        stop_phu = np.array([iso['stop_code'] for iso in isophotes_phu])
        converged_phu = (stop_phu == 0)
        failed_phu = ~converged_phu
        has_photutils = True
    else:
        has_photutils = False

    # Normalize PA to [0, π)
    pa_iso_norm = normalize_pa(pa_iso)
    pa_true = normalize_pa(params['pa'])
    if has_photutils:
        pa_phu_norm = normalize_pa(pa_phu)

    # Create figure with vertical layout
    fig = plt.figure(figsize=(12, 14))
    gs = GridSpec(5, 1, figure=fig, hspace=0.0, height_ratios=[1, 1, 1, 1, 1])

    # Normalize radius by R_e
    R_e = params['R_e']
    r_norm = r_true / R_e
    sma_iso_norm = sma_iso / R_e
    if has_photutils:
        sma_phu_norm = sma_phu / R_e

    # 1. Intensity profile (log scale)
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(r_norm, intens_true, 'k-', lw=2, label='Truth', zorder=1)
    ax1.plot(sma_iso_norm[converged_iso], intens_iso[converged_iso], 'bo',
             markersize=4, label=f'isoster ({np.sum(converged_iso)}/{len(sma_iso)} converged)', zorder=3)
    if np.any(failed_iso):
        ax1.plot(sma_iso_norm[failed_iso], intens_iso[failed_iso], 'rx',
                 markersize=6, label=f'isoster failed ({np.sum(failed_iso)})', zorder=2)
    if has_photutils:
        ax1.plot(sma_phu_norm[converged_phu], intens_phu[converged_phu], 'g^',
                 markersize=4, alpha=0.7, label=f'photutils ({np.sum(converged_phu)}/{len(sma_phu)} converged)', zorder=3)
        if np.any(failed_phu):
            ax1.plot(sma_phu_norm[failed_phu], intens_phu[failed_phu], 'm+',
                     markersize=6, alpha=0.7, label=f'photutils failed ({np.sum(failed_phu)})', zorder=2)

    ax1.set_yscale('log')
    ax1.set_ylabel('Intensity', fontsize=11)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 8)
    ax1.set_xticklabels([])  # Share x-axis
    ax1.set_title(f'{case_name}: n={params["n"]}, Re={params["R_e"]}, eps={params["eps"]:.2f}, PA={np.degrees(params["pa"]):.1f}°, SNR={params["snr"]}',
                  fontsize=12, fontweight='bold')

    # 2. Fractional intensity residual
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    frac_resid_iso = 100.0 * (intens_iso - np.interp(sma_iso, r_true, intens_true)) / np.interp(sma_iso, r_true, intens_true)
    ax2.plot(sma_iso_norm[converged_iso], frac_resid_iso[converged_iso], 'bo', markersize=4, label='isoster', zorder=3)
    if np.any(failed_iso):
        ax2.plot(sma_iso_norm[failed_iso], frac_resid_iso[failed_iso], 'rx', markersize=6, zorder=2)
    if has_photutils:
        frac_resid_phu = 100.0 * (intens_phu - np.interp(sma_phu, r_true, intens_true)) / np.interp(sma_phu, r_true, intens_true)
        ax2.plot(sma_phu_norm[converged_phu], frac_resid_phu[converged_phu], 'g^', markersize=4, alpha=0.7, label='photutils', zorder=3)
        if np.any(failed_phu):
            ax2.plot(sma_phu_norm[failed_phu], frac_resid_phu[failed_phu], 'm+', markersize=6, alpha=0.7, zorder=2)

    ax2.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(1, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax2.axhline(-1, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax2.set_ylabel('Intensity Residual (%)', fontsize=11)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 8)
    ax2.set_ylim(-5, 5)
    ax2.set_xticklabels([])

    # 3. Ellipticity
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.axhline(params['eps'], color='k', linestyle='-', linewidth=2, label='Truth', zorder=1)
    ax3.plot(sma_iso_norm[converged_iso], eps_iso[converged_iso], 'bo', markersize=4, label='isoster', zorder=3)
    if np.any(failed_iso):
        ax3.plot(sma_iso_norm[failed_iso], eps_iso[failed_iso], 'rx', markersize=6, zorder=2)
    if has_photutils:
        ax3.plot(sma_phu_norm[converged_phu], eps_phu[converged_phu], 'g^', markersize=4, alpha=0.7, label='photutils', zorder=3)
        if np.any(failed_phu):
            ax3.plot(sma_phu_norm[failed_phu], eps_phu[failed_phu], 'm+', markersize=6, alpha=0.7, zorder=2)

    ax3.set_ylabel('Ellipticity', fontsize=11)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 8)
    ax3.set_ylim(0, 1)
    ax3.set_xticklabels([])

    # 4. Position Angle (normalized to [0, π))
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.axhline(np.degrees(pa_true), color='k', linestyle='-', linewidth=2, label='Truth', zorder=1)
    ax4.plot(sma_iso_norm[converged_iso], np.degrees(pa_iso_norm[converged_iso]), 'bo',
             markersize=4, label='isoster', zorder=3)
    if np.any(failed_iso):
        ax4.plot(sma_iso_norm[failed_iso], np.degrees(pa_iso_norm[failed_iso]), 'rx', markersize=6, zorder=2)
    if has_photutils:
        ax4.plot(sma_phu_norm[converged_phu], np.degrees(pa_phu_norm[converged_phu]), 'g^',
                 markersize=4, alpha=0.7, label='photutils', zorder=3)
        if np.any(failed_phu):
            ax4.plot(sma_phu_norm[failed_phu], np.degrees(pa_phu_norm[failed_phu]), 'm+', markersize=6, alpha=0.7, zorder=2)

    ax4.set_ylabel('PA (degrees)', fontsize=11)  # Y-axis label per CLAUDE.md
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 8)
    ax4.set_ylim(0, 180)
    ax4.set_xticklabels([])

    # 5. Residual image (2D)
    from isoster.model import build_isoster_model
    ax5 = fig.add_subplot(gs[4], sharex=ax1)

    # Build model from isoster results
    model_iso = build_isoster_model(image.shape, isophotes_iso)
    residual = image - model_iso

    # Compute radial profile of residual
    h, w = image.shape
    y, x = np.mgrid[:h, :w]
    x0, y0 = params['x0'], params['y0']
    r = np.sqrt((x - x0)**2 + (y - y0)**2)

    # Bin by radius
    r_bins = np.linspace(0, 8 * R_e, 50)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2 / R_e
    residual_profile = []

    for i in range(len(r_bins) - 1):
        mask = (r >= r_bins[i]) & (r < r_bins[i+1])
        if np.any(mask):
            residual_profile.append(np.median(residual[mask]))
        else:
            residual_profile.append(np.nan)

    ax5.plot(r_centers, residual_profile, 'b-', linewidth=2, label='Residual (median)')
    ax5.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax5.set_xlabel('Radius / Re', fontsize=11)
    ax5.set_ylabel('Residual (ADU)', fontsize=11)
    ax5.legend(loc='best', fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, 8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved QA figure to {output_path}")
    plt.close()

    # Compute statistics
    # Focus on 0.5-4 Re range
    mask_range = (sma_iso >= 0.5 * R_e) & (sma_iso <= 4.0 * R_e) & converged_iso
    if np.any(mask_range):
        frac_resid_range = frac_resid_iso[mask_range]
        stats = {
            'median_frac_resid': np.median(frac_resid_range),
            'max_abs_frac_resid': np.max(np.abs(frac_resid_range)),
            'convergence_rate': np.sum(converged_iso) / len(sma_iso) * 100,
        }
    else:
        stats = {
            'median_frac_resid': np.nan,
            'max_abs_frac_resid': np.nan,
            'convergence_rate': 0.0,
        }

    return stats


def run_qa_comparison():
    """Run QA comparison for difficult test cases."""
    output_dir = Path("qa_photutils_comparison")
    output_dir.mkdir(exist_ok=True)

    # Define difficult test cases
    test_cases = [
        # (name, n, R_e, eps, pa, snr, oversample)
        ("n1_eps07_high_ellipticity", 1.0, 20.0, 0.7, np.pi/3, None, 5),
        ("n1_eps04_snr100_noisy", 1.0, 20.0, 0.4, np.pi/4, 100, 5),
        ("n4_eps04_snr100_noisy", 4.0, 20.0, 0.4, np.pi/4, 100, 10),
    ]

    print("="*70)
    print("QA COMPARISON: isoster vs photutils vs truth")
    print("="*70)
    print()

    if not PHOTUTILS_AVAILABLE:
        print("WARNING: photutils not installed, comparison will be limited")
        print()

    all_stats = []

    for name, n, R_e, eps, pa, snr, oversample in test_cases:
        print(f"Running: {name}")
        print(f"  Parameters: n={n}, Re={R_e}, eps={eps:.1f}, PA={np.degrees(pa):.1f}°, SNR={snr}, oversample={oversample}")

        # Create mock image
        I_e = 1000.0
        noise_level = I_e / snr if snr is not None else None
        image, (x0, y0), shape, r_true, intens_true = create_sersic_model(
            R_e, n, I_e, eps, pa, noise_level, oversample
        )

        print(f"  Image shape: {shape}, center: ({x0}, {y0})")

        # Run isoster
        print(f"  Running isoster...")
        isophotes_iso = run_isoster(image, x0, y0, eps, pa, R_e)
        print(f"  isoster: {len(isophotes_iso)} isophotes fitted")

        # Run photutils
        isophotes_phu = []
        if PHOTUTILS_AVAILABLE:
            print(f"  Running photutils...")
            isophotes_phu = run_photutils(image, x0, y0, eps, pa, R_e)
            print(f"  photutils: {len(isophotes_phu)} isophotes fitted")
        else:
            print(f"  Skipping photutils (not installed)")

        # Generate QA figure
        params = {
            'n': n, 'R_e': R_e, 'eps': eps, 'pa': pa, 'snr': snr,
            'x0': x0, 'y0': y0
        }
        output_path = output_dir / f"{name}_qa.png"
        stats = plot_qa_comparison(image, isophotes_iso, isophotes_phu,
                                    r_true, intens_true, name, params, output_path)

        print(f"  Statistics (0.5-4 Re):")
        print(f"    Median fractional residual: {stats['median_frac_resid']:.2f}%")
        print(f"    Max abs fractional residual: {stats['max_abs_frac_resid']:.2f}%")
        print(f"    Convergence rate: {stats['convergence_rate']:.1f}%")
        print()

        all_stats.append({'name': name, **stats})

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()
    print(f"{'Test Case':<35} {'Med Resid (%)':<15} {'Max Resid (%)':<15} {'Conv Rate':<10}")
    print("-"*70)
    for s in all_stats:
        print(f"{s['name']:<35} {s['median_frac_resid']:>8.2f}       {s['max_abs_frac_resid']:>8.2f}       {s['convergence_rate']:>6.1f}%")
    print()
    print(f"QA figures saved to: {output_dir}/")


if __name__ == '__main__':
    run_qa_comparison()
