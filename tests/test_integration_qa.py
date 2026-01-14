"""
Integration tests for isoster with QA visualization.

These tests verify the main fit_image() entry point using synthetic Sersic models
and generate QA figures following CLAUDE.md guidelines.
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

from isoster.driver import fit_image
from isoster.config import IsosterConfig


def create_sersic_model(shape, x0, y0, I_e, R_e, n, eps, pa, oversample=1):
    """
    Create a 2D Sersic profile image with optional central oversampling.

    Args:
        shape: Image dimensions (h, w)
        x0, y0: Center coordinates
        I_e: Intensity at effective radius
        R_e: Effective radius (pixels)
        n: Sersic index
        eps: Ellipticity (0 to 1)
        pa: Position angle (radians)
        oversample: Oversampling factor for central region (pixels < 3*R_e^0.5)

    Returns:
        image: 2D Sersic profile
        true_profile: Function to compute true 1D profile at any radius
    """
    h, w = shape

    # Compute b_n parameter (approximation from Ciotti & Bertin 1999)
    b_n = 1.9992 * n - 0.3271

    if oversample > 1:
        # Create high-resolution grid for central region
        y_hr = np.linspace(0, h, h * oversample, endpoint=False) + 0.5/oversample
        x_hr = np.linspace(0, w, w * oversample, endpoint=False) + 0.5/oversample
        yy_hr, xx_hr = np.meshgrid(y_hr, x_hr, indexing='ij')

        # Rotate coordinates
        dx_hr = xx_hr - x0
        dy_hr = yy_hr - y0
        x_rot_hr = dx_hr * np.cos(pa) + dy_hr * np.sin(pa)
        y_rot_hr = -dx_hr * np.sin(pa) + dy_hr * np.cos(pa)

        # Elliptical radius
        r_hr = np.sqrt(x_rot_hr**2 + (y_rot_hr / (1 - eps))**2)

        # Sersic profile
        image_hr = I_e * np.exp(-b_n * ((r_hr / R_e)**(1/n) - 1))

        # Downsample to final resolution
        image = image_hr.reshape(h, oversample, w, oversample).mean(axis=(1, 3))
    else:
        # Standard resolution
        y = np.arange(h)
        x = np.arange(w)
        yy, xx = np.meshgrid(y, x, indexing='ij')

        # Rotate coordinates
        dx = xx - x0
        dy = yy - y0
        x_rot = dx * np.cos(pa) + dy * np.sin(pa)
        y_rot = -dx * np.sin(pa) + dy * np.cos(pa)

        # Elliptical radius
        r = np.sqrt(x_rot**2 + (y_rot / (1 - eps))**2)

        # Sersic profile
        image = I_e * np.exp(-b_n * ((r / R_e)**(1/n) - 1))

    # True 1D profile function
    def true_profile(sma):
        """Compute true Sersic intensity at given SMA."""
        return I_e * np.exp(-b_n * ((sma / R_e)**(1/n) - 1))

    return image, true_profile


def plot_qa_figure(image, results, true_profile, params, output_path):
    """
    Create comprehensive QA figure following CLAUDE.md guidelines.

    Args:
        image: Original image
        results: isoster fit results dict
        true_profile: Function for true 1D intensity profile
        params: Dict with model parameters (R_e, n, eps, pa, etc.)
        output_path: Path to save figure
    """
    isophotes = results['isophotes']

    # Extract data
    sma = np.array([iso['sma'] for iso in isophotes])
    intens = np.array([iso['intens'] for iso in isophotes])
    intens_err = np.array([iso['intens_err'] for iso in isophotes])
    x0 = np.array([iso['x0'] for iso in isophotes])
    y0 = np.array([iso['y0'] for iso in isophotes])
    eps = np.array([iso['eps'] for iso in isophotes])
    pa = np.array([iso['pa'] for iso in isophotes])
    stop_codes = np.array([iso['stop_code'] for iso in isophotes])

    # Normalize PA to avoid jumps > 90 degrees
    pa_normalized = pa.copy()
    for i in range(1, len(pa_normalized)):
        while pa_normalized[i] - pa_normalized[i-1] > np.pi/2:
            pa_normalized[i] -= np.pi
        while pa_normalized[i] - pa_normalized[i-1] < -np.pi/2:
            pa_normalized[i] += np.pi

    # Separate by stop code
    good = stop_codes == 0
    minor = stop_codes == 2
    flagged = stop_codes == 1
    failed = (stop_codes == -1) | (stop_codes == 3)

    # Compute true profiles
    true_intens = true_profile(sma)
    true_eps = np.full_like(sma, params['eps'])
    true_pa = np.full_like(sma, params['pa'])

    # Relative residuals
    rel_residual = (intens - true_intens) / true_intens

    # SMA^0.25 for X-axis (compress outer profile)
    x_axis = sma**0.25
    x_label = r'$\mathrm{SMA}^{0.25}$ (pixels$^{0.25}$)'

    # Create figure with GridSpec
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3,
                  height_ratios=[1, 1, 1])

    # ===== Top row: Images =====
    # Original image with isophotes
    ax_img = fig.add_subplot(gs[0, 0])
    im = ax_img.imshow(np.log10(image + 1), cmap='gray', origin='lower')
    plt.colorbar(im, ax=ax_img, label=r'$\log_{10}$(Intensity + 1)')

    # Plot selective isophotes (every 5th, color-coded by stop code)
    from matplotlib.patches import Ellipse as MPLEllipse
    for i, iso in enumerate(isophotes[::5]):
        color = {0: 'green', 1: 'orange', 2: 'yellow', -1: 'red', 3: 'red'}.get(iso['stop_code'], 'gray')
        ellipse = MPLEllipse((iso['x0'], iso['y0']),
                             2*iso['sma'], 2*iso['sma']*(1-iso['eps']),
                             angle=np.degrees(iso['pa']),
                             fill=False, edgecolor=color, linewidth=1, alpha=0.7)
        ax_img.add_patch(ellipse)
    ax_img.set_title('Original + Isophotes')
    ax_img.set_xlabel('X (pixels)')
    ax_img.set_ylabel('Y (pixels)')

    # Reconstruct model
    ax_model = fig.add_subplot(gs[0, 1])
    from isoster.model import build_ellipse_model
    model = build_ellipse_model(image.shape, isophotes)
    im = ax_model.imshow(np.log10(model + 1), cmap='gray', origin='lower')
    plt.colorbar(im, ax=ax_model, label=r'$\log_{10}$(Model + 1)')
    ax_model.set_title('Reconstructed Model')
    ax_model.set_xlabel('X (pixels)')

    # Residual
    ax_resid = fig.add_subplot(gs[0, 2])
    residual = image - model
    vmax = np.percentile(np.abs(residual), 99)
    im = ax_resid.imshow(residual, cmap='RdBu_r', origin='lower',
                         vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax_resid, label='Residual')
    ax_resid.set_title('Residual (Data - Model)')
    ax_resid.set_xlabel('X (pixels)')

    # ===== Middle row: Surface brightness (larger) =====
    ax_sb = fig.add_subplot(gs[1, :])  # Spans all 3 columns

    # Plot truth as dashed line
    ax_sb.plot(x_axis, np.log10(true_intens), 'k--', linewidth=2, label='True Sersic', zorder=10)

    # Plot data as scatter with errorbars (excluding errorbars from axis limits)
    if good.any():
        ax_sb.errorbar(x_axis[good], np.log10(intens[good]),
                       yerr=intens_err[good]/(intens[good]*np.log(10)),
                       fmt='o', color='green', markersize=4, capsize=2, alpha=0.7,
                       label='Converged (0)', zorder=3)
    if minor.any():
        ax_sb.errorbar(x_axis[minor], np.log10(intens[minor]),
                       yerr=intens_err[minor]/(intens[minor]*np.log(10)),
                       fmt='s', color='yellow', markersize=4, capsize=2, alpha=0.7,
                       label='Minor issues (2)', zorder=2)
    if flagged.any():
        ax_sb.scatter(x_axis[flagged], np.log10(intens[flagged]),
                      marker='^', color='orange', s=20, alpha=0.7,
                      label='Flagged (1)', zorder=1)
    if failed.any():
        ax_sb.scatter(x_axis[failed], np.log10(intens[failed]),
                      marker='x', color='red', s=30, alpha=0.7,
                      label='Failed (-1, 3)', zorder=0)

    ax_sb.set_xlabel(x_label)
    ax_sb.set_ylabel(r'$\log_{10}$(Intensity)')
    ax_sb.set_title(f'Surface Brightness Profile (n={params["n"]}, R_e={params["R_e"]:.1f}, eps={params["eps"]:.2f})')
    ax_sb.legend(loc='upper right', fontsize=8)
    ax_sb.grid(alpha=0.3)

    # Set Y-axis limits without including errorbars
    valid_intens = intens[(good | minor)]
    if len(valid_intens) > 0:
        y_min = np.log10(np.min(valid_intens)) - 0.5
        y_max = np.log10(np.max(valid_intens)) + 0.2
        ax_sb.set_ylim(y_min, y_max)

    # ===== Bottom row: Geometry profiles =====
    # Relative residual
    ax_res = fig.add_subplot(gs[2, 0])
    ax_res.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    if good.any():
        ax_res.scatter(x_axis[good], rel_residual[good] * 100,
                       color='green', s=15, alpha=0.7)
    if minor.any():
        ax_res.scatter(x_axis[minor], rel_residual[minor] * 100,
                       color='yellow', s=15, alpha=0.7)
    ax_res.set_xlabel(x_label)
    ax_res.set_ylabel('Relative Residual (%)')
    ax_res.set_title('(Isoster - Truth) / Truth')
    ax_res.grid(alpha=0.3)

    # Ellipticity
    ax_eps = fig.add_subplot(gs[2, 1])
    ax_eps.axhline(params['eps'], color='k', linestyle='--', linewidth=2, label='True', zorder=10)
    if good.any():
        ax_eps.scatter(x_axis[good], eps[good], color='green', s=15, alpha=0.7, label='Fitted')
    if minor.any():
        ax_eps.scatter(x_axis[minor], eps[minor], color='yellow', s=15, alpha=0.7)
    ax_eps.set_xlabel(x_label)
    ax_eps.set_ylabel('Ellipticity')
    ax_eps.set_title('Ellipticity Profile')
    ax_eps.legend(fontsize=8)
    ax_eps.grid(alpha=0.3)

    # Set Y-axis limits without errorbars
    valid_eps = eps[(good | minor)]
    if len(valid_eps) > 0:
        eps_range = np.max(valid_eps) - np.min(valid_eps)
        ax_eps.set_ylim(np.min(valid_eps) - 0.5*eps_range,
                        np.max(valid_eps) + 0.5*eps_range)

    # Position angle
    ax_pa = fig.add_subplot(gs[2, 2])
    ax_pa.axhline(params['pa'], color='k', linestyle='--', linewidth=2, label='True', zorder=10)
    if good.any():
        ax_pa.scatter(x_axis[good], pa_normalized[good], color='green', s=15, alpha=0.7, label='Fitted')
    if minor.any():
        ax_pa.scatter(x_axis[minor], pa_normalized[minor], color='yellow', s=15, alpha=0.7)
    ax_pa.set_xlabel(x_label)
    ax_pa.set_ylabel('Position Angle (rad)')
    ax_pa.set_title('Position Angle Profile')
    ax_pa.legend(fontsize=8)
    ax_pa.grid(alpha=0.3)

    # Set Y-axis limits without errorbars
    valid_pa = pa_normalized[(good | minor)]
    if len(valid_pa) > 0:
        pa_range = np.max(valid_pa) - np.min(valid_pa)
        ax_pa.set_ylim(np.min(valid_pa) - 0.5*pa_range,
                       np.max(valid_pa) + 0.5*pa_range)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"QA figure saved to: {output_path}")


# ============================================================================
# Integration Tests
# ============================================================================

def test_fit_image_basic():
    """Test basic fit_image() workflow with synthetic exponential galaxy."""
    # Create simple exponential disk (n=1)
    image, true_profile = create_sersic_model(
        shape=(100, 100),
        x0=50.0, y0=50.0,
        I_e=1000.0, R_e=15.0, n=1.0,
        eps=0.3, pa=0.5,
        oversample=1
    )

    # Add small noise
    rng = np.random.RandomState(42)
    image += rng.normal(0, 1, image.shape)

    # Configure fitting
    cfg = IsosterConfig(
        x0=50.0, y0=50.0,
        sma0=5.0, minsma=3.0, maxsma=40.0,
        astep=0.15,
        eps=0.2, pa=0.5,
        minit=5, maxit=30,
        conver=0.05
    )

    # Run fit
    results = fit_image(image, mask=None, config=cfg)

    # Basic assertions
    assert 'isophotes' in results
    assert 'config' in results
    assert len(results['isophotes']) > 0

    isophotes = results['isophotes']

    # Check first isophote structure
    iso0 = isophotes[0]
    assert 'sma' in iso0
    assert 'intens' in iso0
    assert 'stop_code' in iso0
    assert 'x0' in iso0 and 'y0' in iso0
    assert 'eps' in iso0 and 'pa' in iso0

    # Check convergence
    converged = [iso for iso in isophotes if iso['stop_code'] == 0]
    assert len(converged) > 5, f"Expected >5 converged isophotes, got {len(converged)}"

    # Check SMA ordering (should be monotonic)
    smas = [iso['sma'] for iso in isophotes]
    assert smas == sorted(smas), "SMAs should be sorted"

    print(f"✓ Basic test passed: {len(isophotes)} isophotes, {len(converged)} converged")


def test_sersic_n4_noiseless_with_qa():
    """
    Test fitting Sersic n=4 (de Vaucouleurs) noiseless profile with QA figure.

    High Sersic index requires higher oversampling in central region.
    """
    # Model parameters
    params = {
        'x0': 50.0, 'y0': 50.0,
        'I_e': 2000.0, 'R_e': 20.0, 'n': 4.0,
        'eps': 0.4, 'pa': np.pi/4
    }

    # Create model with central oversampling (n=4 needs higher oversample)
    image, true_profile = create_sersic_model(
        shape=(120, 120),
        oversample=10,  # High oversample for steep central profile
        **params
    )

    # Configure fitting
    cfg = IsosterConfig(
        x0=params['x0'], y0=params['y0'],
        sma0=10.0, minsma=3.0, maxsma=60.0,
        astep=0.12,
        eps=params['eps'], pa=params['pa'],
        minit=10, maxit=50,
        conver=0.03,  # Tighter convergence for noiseless
        fix_center=True,  # Fix center to avoid drift
    )

    # Run fit
    results = fit_image(image, mask=None, config=cfg)
    isophotes = results['isophotes']

    # Generate QA figure
    output_dir = Path('tests/qa_outputs')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'test_sersic_n4_noiseless.png'

    plot_qa_figure(image, results, true_profile, params, output_path)

    # Validation: Check accuracy in valid range
    # Per CLAUDE.md: Ignore region <3 pixels and where stop_code indicates problems
    sma = np.array([iso['sma'] for iso in isophotes])
    intens = np.array([iso['intens'] for iso in isophotes])
    stop_codes = np.array([iso['stop_code'] for iso in isophotes])

    # Valid range: 0.5*R_e to 8*R_e (per CLAUDE.md for noiseless)
    # Also require sma >= 3 pixels and stop_code == 0
    valid = (sma >= 3) & (sma >= 0.5 * params['R_e']) & \
            (sma <= 8 * params['R_e']) & (stop_codes == 0)

    if valid.sum() > 0:
        true_intens = true_profile(sma[valid])
        rel_diff = np.abs((intens[valid] - true_intens) / true_intens)

        max_rel_diff = np.max(rel_diff)
        median_rel_diff = np.median(rel_diff)

        print(f"✓ Sersic n=4 noiseless test:")
        print(f"  Valid range: {sma[valid].min():.1f} - {sma[valid].max():.1f} pixels")
        print(f"  Max relative difference: {max_rel_diff*100:.3f}%")
        print(f"  Median relative difference: {median_rel_diff*100:.3f}%")
        print(f"  Converged: {(stop_codes == 0).sum()} / {len(isophotes)}")

        # Strict accuracy target for noiseless: <1% error
        assert max_rel_diff < 0.01, \
            f"Max relative difference {max_rel_diff*100:.2f}% exceeds 1% threshold"
    else:
        pytest.skip("No valid isophotes in target range")


def test_sersic_n1_high_ellipticity_with_qa():
    """
    Test fitting high-ellipticity (eps=0.7) exponential disk with QA figure.

    High ellipticity may benefit from eccentric anomaly mode.
    """
    params = {
        'x0': 60.0, 'y0': 60.0,
        'I_e': 1500.0, 'R_e': 25.0, 'n': 1.0,
        'eps': 0.7, 'pa': np.pi/3
    }

    # Create model with moderate oversampling
    image, true_profile = create_sersic_model(
        shape=(140, 140),
        oversample=5,
        **params
    )

    # Add noise
    rng = np.random.RandomState(123)
    snr_at_re = 100
    noise_level = params['I_e'] / snr_at_re
    image += rng.normal(0, noise_level, image.shape)

    # Configure fitting with eccentric anomaly mode
    cfg = IsosterConfig(
        x0=params['x0'], y0=params['y0'],
        sma0=8.0, minsma=3.0, maxsma=80.0,
        astep=0.15,
        eps=params['eps'], pa=params['pa'],
        minit=10, maxit=50,
        conver=0.05,
        use_eccentric_anomaly=True,  # Better for high ellipticity
    )

    # Run fit
    results = fit_image(image, mask=None, config=cfg)
    isophotes = results['isophotes']

    # Generate QA figure
    output_dir = Path('tests/qa_outputs')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'test_sersic_n1_high_eps.png'

    plot_qa_figure(image, results, true_profile, params, output_path)

    # Validation: Check accuracy in valid range (with noise)
    # Per CLAUDE.md: 0.5*R_e to 5*R_e for noisy data
    sma = np.array([iso['sma'] for iso in isophotes])
    intens = np.array([iso['intens'] for iso in isophotes])
    eps_fit = np.array([iso['eps'] for iso in isophotes])
    stop_codes = np.array([iso['stop_code'] for iso in isophotes])

    valid = (sma >= 3) & (sma >= 0.5 * params['R_e']) & \
            (sma <= 5 * params['R_e']) & (stop_codes == 0)

    if valid.sum() > 0:
        true_intens = true_profile(sma[valid])
        rel_diff = np.abs((intens[valid] - true_intens) / true_intens)

        # Check ellipticity recovery
        eps_diff = np.abs(eps_fit[valid] - params['eps'])

        max_rel_diff = np.max(rel_diff)
        median_rel_diff = np.median(rel_diff)
        max_eps_diff = np.max(eps_diff)
        median_eps_diff = np.median(eps_diff)

        print(f"✓ High ellipticity test (eps=0.7, S/N=100):")
        print(f"  Valid range: {sma[valid].min():.1f} - {sma[valid].max():.1f} pixels")
        print(f"  Max intensity rel diff: {max_rel_diff*100:.3f}%")
        print(f"  Median intensity rel diff: {median_rel_diff*100:.3f}%")
        print(f"  Max eps diff: {max_eps_diff:.4f}")
        print(f"  Median eps diff: {median_eps_diff:.4f}")
        print(f"  Converged: {(stop_codes == 0).sum()} / {len(isophotes)}")

        # Relaxed accuracy for noisy data: <5% intensity error, <0.05 eps error
        assert max_rel_diff < 0.05, \
            f"Max intensity difference {max_rel_diff*100:.2f}% exceeds 5% threshold"
        assert max_eps_diff < 0.05, \
            f"Max ellipticity difference {max_eps_diff:.3f} exceeds 0.05 threshold"
    else:
        pytest.skip("No valid isophotes in target range")


def test_fit_image_with_mask():
    """Test fit_image() with masked regions."""
    # Create simple galaxy
    image, _ = create_sersic_model(
        shape=(100, 100),
        x0=50.0, y0=50.0,
        I_e=1000.0, R_e=15.0, n=1.0,
        eps=0.2, pa=0.0,
        oversample=1
    )

    # Create mask with a circular masked region
    y, x = np.mgrid[:100, :100]
    r = np.sqrt((x - 50)**2 + (y - 50)**2)
    mask = (r > 15) & (r < 20)  # Annular mask

    cfg = IsosterConfig(sma0=10, maxsma=35, fflag=0.4)  # More permissive fflag
    results = fit_image(image, mask=mask, config=cfg)

    # Should still get results
    assert len(results['isophotes']) > 0

    # Check stop code distribution
    stop_codes = [iso['stop_code'] for iso in results['isophotes']]
    converged_count = sum(1 for code in stop_codes if code == 0)
    flagged_count = sum(1 for code in stop_codes if code == 1)

    print(f"✓ Masked region test:")
    print(f"  Converged: {converged_count}, Flagged: {flagged_count}, Total: {len(results['isophotes'])}")

    # Just verify fitting completed successfully (doesn't crash with mask)
    assert len(results['isophotes']) > 5, "Should have multiple isophotes even with mask"


if __name__ == '__main__':
    # Run tests and generate QA figures
    pytest.main([__file__, '-v', '-s'])
