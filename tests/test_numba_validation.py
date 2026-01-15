"""
Validation tests for Numba optimization in ISOSTER.

These tests verify that the numba-accelerated code produces identical results
to the original numpy implementation, with QA visualization.
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

from isoster.driver import fit_image
from isoster.config import IsosterConfig
from isoster.numba_kernels import (
    NUMBA_AVAILABLE,
    _harmonic_model_numba, _harmonic_model_numpy,
    _ea_to_pa_numba, _ea_to_pa_numpy,
    _compute_ellipse_coords_numba, _compute_ellipse_coords_numpy,
    _build_harmonic_matrix_numba, _build_harmonic_matrix_numpy,
)


def create_sersic_model(R_e, n, I_e, eps, pa, oversample=5):
    """Create a centered 2D Sersic profile image."""
    half_size = max(int(15 * R_e), 150)
    shape = (2 * half_size, 2 * half_size)
    x0, y0 = half_size, half_size

    b_n = 1.9992 * n - 0.3271

    if oversample > 1:
        y_hr = np.linspace(0, shape[0], shape[0] * oversample, endpoint=False) + 0.5/oversample
        x_hr = np.linspace(0, shape[1], shape[1] * oversample, endpoint=False) + 0.5/oversample
        yy_hr, xx_hr = np.meshgrid(y_hr, x_hr, indexing='ij')

        dx_hr = xx_hr - x0
        dy_hr = yy_hr - y0
        x_rot_hr = dx_hr * np.cos(pa) + dy_hr * np.sin(pa)
        y_rot_hr = -dx_hr * np.sin(pa) + dy_hr * np.cos(pa)
        r_hr = np.sqrt(x_rot_hr**2 + (y_rot_hr / (1 - eps))**2)

        image_hr = I_e * np.exp(-b_n * ((r_hr / R_e)**(1/n) - 1))
        image = image_hr.reshape(shape[0], oversample, shape[1], oversample).mean(axis=(1, 3))
    else:
        y, x = np.mgrid[:shape[0], :shape[1]].astype(np.float64)
        dx = x - x0
        dy = y - y0
        x_rot = dx * np.cos(pa) + dy * np.sin(pa)
        y_rot = -dx * np.sin(pa) + dy * np.cos(pa)
        r = np.sqrt(x_rot**2 + (y_rot / (1 - eps))**2)
        image = I_e * np.exp(-b_n * ((r / R_e)**(1/n) - 1))

    def true_profile(sma):
        return I_e * np.exp(-b_n * ((sma / R_e)**(1/n) - 1))

    params = {
        'R_e': R_e, 'n': n, 'I_e': I_e, 'eps': eps, 'pa': pa,
        'x0': x0, 'y0': y0, 'shape': shape
    }

    return image, true_profile, params


class TestNumbaKernels:
    """Test individual numba kernels against numpy implementations."""

    def test_harmonic_model(self):
        """Verify harmonic_model numba vs numpy produces identical results."""
        phi = np.linspace(0, 2*np.pi, 100)
        coeffs = np.array([100.0, 1.0, 2.0, 0.5, 0.3])

        result_numba = _harmonic_model_numba(phi, coeffs)
        result_numpy = _harmonic_model_numpy(phi, coeffs)

        assert np.allclose(result_numba, result_numpy, rtol=1e-14, atol=1e-14), \
            f"Harmonic model max diff: {np.max(np.abs(result_numba - result_numpy)):.2e}"

    def test_ea_to_pa(self):
        """Verify EA to PA conversion numba vs numpy produces identical results."""
        psi = np.linspace(0, 2*np.pi, 100)
        eps = 0.4

        result_numba = _ea_to_pa_numba(psi, eps)
        result_numpy = _ea_to_pa_numpy(psi, eps)

        assert np.allclose(result_numba, result_numpy, rtol=1e-14, atol=1e-14), \
            f"EA to PA max diff: {np.max(np.abs(result_numba - result_numpy)):.2e}"

    def test_ellipse_coords(self):
        """Verify ellipse coordinate computation numba vs numpy produces identical results."""
        n_samples = 100
        sma, eps, pa, x0, y0 = 50.0, 0.3, 0.5, 200.0, 200.0

        # Test both EA and non-EA modes
        for use_ea in [True, False]:
            x1, y1, a1, p1 = _compute_ellipse_coords_numba(n_samples, sma, eps, pa, x0, y0, use_ea)
            x2, y2, a2, p2 = _compute_ellipse_coords_numpy(n_samples, sma, eps, pa, x0, y0, use_ea)

            assert np.allclose(x1, x2, rtol=1e-14, atol=1e-14), \
                f"Ellipse coords x max diff (use_ea={use_ea}): {np.max(np.abs(x1 - x2)):.2e}"
            assert np.allclose(y1, y2, rtol=1e-14, atol=1e-14), \
                f"Ellipse coords y max diff (use_ea={use_ea}): {np.max(np.abs(y1 - y2)):.2e}"
            assert np.allclose(a1, a2, rtol=1e-14, atol=1e-14), \
                f"Ellipse coords angles max diff (use_ea={use_ea}): {np.max(np.abs(a1 - a2)):.2e}"
            assert np.allclose(p1, p2, rtol=1e-14, atol=1e-14), \
                f"Ellipse coords phi max diff (use_ea={use_ea}): {np.max(np.abs(p1 - p2)):.2e}"

    def test_harmonic_matrix(self):
        """Verify harmonic matrix construction numba vs numpy produces identical results."""
        phi = np.linspace(0, 2*np.pi, 100)

        A_numba = _build_harmonic_matrix_numba(phi)
        A_numpy = _build_harmonic_matrix_numpy(phi)

        assert np.allclose(A_numba, A_numpy, rtol=1e-14, atol=1e-14), \
            f"Harmonic matrix max diff: {np.max(np.abs(A_numba - A_numpy)):.2e}"

    def test_circular_case(self):
        """Test coordinate computation for circular case (eps=0)."""
        n_samples = 100
        sma, eps, pa, x0, y0 = 50.0, 0.0, 0.0, 200.0, 200.0

        x1, y1, a1, p1 = _compute_ellipse_coords_numba(n_samples, sma, eps, pa, x0, y0, False)
        x2, y2, a2, p2 = _compute_ellipse_coords_numpy(n_samples, sma, eps, pa, x0, y0, False)

        assert np.allclose(x1, x2, rtol=1e-14, atol=1e-14)
        assert np.allclose(y1, y2, rtol=1e-14, atol=1e-14)

    def test_high_ellipticity(self):
        """Test coordinate computation for high ellipticity case."""
        n_samples = 100
        sma, eps, pa, x0, y0 = 50.0, 0.8, np.pi/3, 200.0, 200.0

        x1, y1, a1, p1 = _compute_ellipse_coords_numba(n_samples, sma, eps, pa, x0, y0, True)
        x2, y2, a2, p2 = _compute_ellipse_coords_numpy(n_samples, sma, eps, pa, x0, y0, True)

        assert np.allclose(x1, x2, rtol=1e-14, atol=1e-14)
        assert np.allclose(y1, y2, rtol=1e-14, atol=1e-14)


def test_numba_full_fit_accuracy():
    """
    Full integration test: verify numba-accelerated fitting produces accurate results.

    This test creates a Sersic model, fits it with numba-accelerated code,
    and verifies the results match the true profile.
    """
    # Create test image
    R_e, n, I_e = 25.0, 4.0, 1500.0
    eps, pa = 0.4, np.pi/4

    image, true_profile, params = create_sersic_model(
        R_e=R_e, n=n, I_e=I_e, eps=eps, pa=pa, oversample=10
    )

    # Configure fitting
    cfg = IsosterConfig(
        x0=params['x0'], y0=params['y0'],
        sma0=10.0, minsma=3.0, maxsma=min(100, params['x0']-10),
        astep=0.12,
        eps=eps, pa=pa,
        minit=10, maxit=50,
        conver=0.03,
        fix_center=True,
        use_eccentric_anomaly=True,
    )

    # Run fit with numba
    results = fit_image(image, mask=None, config=cfg)
    isophotes = results['isophotes']

    # Extract results
    sma = np.array([iso['sma'] for iso in isophotes])
    intens = np.array([iso['intens'] for iso in isophotes])
    stop_codes = np.array([iso['stop_code'] for iso in isophotes])

    # Validate accuracy
    valid = (sma >= 3) & (sma >= 0.5 * R_e) & (sma <= 5 * R_e) & (stop_codes == 0)

    if valid.sum() > 0:
        true_intens = true_profile(sma[valid])
        rel_diff = np.abs((intens[valid] - true_intens) / true_intens)

        max_rel_diff = np.max(rel_diff)
        median_rel_diff = np.median(rel_diff)

        print(f"\nNumba full fit accuracy test:")
        print(f"  Numba available: {NUMBA_AVAILABLE}")
        print(f"  Valid range: {sma[valid].min():.1f} - {sma[valid].max():.1f} pixels")
        print(f"  Max rel diff: {max_rel_diff*100:.3f}%")
        print(f"  Median rel diff: {median_rel_diff*100:.3f}%")
        print(f"  Converged: {(stop_codes == 0).sum()} / {len(isophotes)}")

        # Accuracy criterion: < 1% max difference for noiseless data
        assert max_rel_diff < 0.01, \
            f"Max relative difference {max_rel_diff*100:.2f}% exceeds 1% threshold"


def test_numba_qa_figure():
    """
    Generate QA figure demonstrating numba-accelerated fitting accuracy.

    Follows CLAUDE.md guidelines for QA figure layout.
    """
    # Create test image
    R_e, n, I_e = 30.0, 4.0, 2000.0
    eps, pa = 0.5, np.pi/5

    image, true_profile, params = create_sersic_model(
        R_e=R_e, n=n, I_e=I_e, eps=eps, pa=pa, oversample=12
    )

    # Add light noise
    rng = np.random.RandomState(42)
    snr_at_re = 200
    noise_level = I_e / snr_at_re
    image += rng.normal(0, noise_level, image.shape)

    # Configure fitting
    cfg = IsosterConfig(
        x0=params['x0'], y0=params['y0'],
        sma0=10.0, minsma=3.0, maxsma=min(150, params['x0']-10),
        astep=0.12,
        eps=eps, pa=pa,
        minit=10, maxit=50,
        conver=0.04,
        maxgerr=0.8,
        use_eccentric_anomaly=True,
    )

    # Run fit
    results = fit_image(image, mask=None, config=cfg)
    isophotes = results['isophotes']

    # Extract data
    sma = np.array([iso['sma'] for iso in isophotes])
    intens = np.array([iso['intens'] for iso in isophotes])
    intens_err = np.array([iso['intens_err'] for iso in isophotes])
    eps_fit = np.array([iso['eps'] for iso in isophotes])
    pa_fit = np.array([iso['pa'] for iso in isophotes])
    stop_codes = np.array([iso['stop_code'] for iso in isophotes])

    # Normalize PA
    pa_normalized = np.mod(pa_fit, np.pi)
    for i in range(1, len(pa_normalized)):
        diff = pa_normalized[i] - pa_normalized[i-1]
        if diff > np.pi/2:
            pa_normalized[i:] -= np.pi
        elif diff < -np.pi/2:
            pa_normalized[i:] += np.pi
    pa_normalized = np.mod(pa_normalized, np.pi)

    # Compute true values
    true_intens = true_profile(sma)
    rel_residual = (intens - true_intens) / true_intens

    # Stop code masks
    good = stop_codes == 0
    minor = stop_codes == 2
    failed = (stop_codes == -1) | (stop_codes == 3) | (stop_codes == 1)

    # X-axis: SMA^0.25
    x_axis = sma**0.25

    # Create figure
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(5, 2, figure=fig, hspace=0.05, wspace=0.25,
                  width_ratios=[1, 1], height_ratios=[2, 1, 1, 1, 1])

    # Left column: Images
    ax_img = fig.add_subplot(gs[0, 0])
    vmin, vmax = np.percentile(image[image > 0], [1, 99.9])
    im = ax_img.imshow(np.log10(np.clip(image, vmin, None)),
                       cmap='gray', origin='lower', vmin=np.log10(vmin), vmax=np.log10(vmax))
    plt.colorbar(im, ax=ax_img, label=r'$\log_{10}$(Intensity)')

    # Plot isophotes
    from matplotlib.patches import Ellipse as MPLEllipse
    for i, iso in enumerate(isophotes[::5]):
        color = {0: 'green', 1: 'orange', 2: 'yellow', -1: 'red', 3: 'red'}.get(iso['stop_code'], 'gray')
        ellipse = MPLEllipse((iso['x0'], iso['y0']),
                             2*iso['sma'], 2*iso['sma']*(1-iso['eps']),
                             angle=np.degrees(iso['pa']),
                             fill=False, edgecolor=color, linewidth=1.5, alpha=0.8)
        ax_img.add_patch(ellipse)

    ax_img.set_title(f"Numba-Accelerated Fit (n={n}, Re={R_e:.0f}, eps={eps:.1f})")
    ax_img.set_xlabel('X (pixels)')
    ax_img.set_ylabel('Y (pixels)')

    # Residual image
    ax_resid = fig.add_subplot(gs[1:, 0])
    from isoster.model import build_isoster_model
    model = build_isoster_model(image.shape, isophotes)
    residual = image - model
    vmax_res = np.percentile(np.abs(residual), 99)
    im = ax_resid.imshow(residual, cmap='RdBu_r', origin='lower',
                         vmin=-vmax_res, vmax=vmax_res)
    plt.colorbar(im, ax=ax_resid, label='Residual')
    ax_resid.set_title('Residual (Data - Model)')
    ax_resid.set_xlabel('X (pixels)')
    ax_resid.set_ylabel('Y (pixels)')

    # Right column: 1-D profiles
    # Row 1: Surface brightness
    ax_sb = fig.add_subplot(gs[0, 1])
    ax_sb.plot(x_axis, np.log10(true_intens), 'k--', linewidth=2, label='True Sersic', zorder=10)
    if good.any():
        ax_sb.errorbar(x_axis[good], np.log10(intens[good]),
                       yerr=intens_err[good]/(intens[good]*np.log(10)),
                       fmt='o', color='green', markersize=5, capsize=2, alpha=0.7,
                       label='Converged (0)', zorder=3)
    if failed.any():
        ax_sb.scatter(x_axis[failed], np.log10(intens[failed]),
                      marker='x', color='red', s=40, alpha=0.7,
                      label='Failed', zorder=1)

    ax_sb.set_ylabel(r'$\log_{10}$(Intensity)', fontsize=11)
    ax_sb.set_title(f'Surface Brightness Profile (Numba={NUMBA_AVAILABLE})')
    ax_sb.legend(loc='upper right', fontsize=9)
    ax_sb.grid(alpha=0.3)
    ax_sb.tick_params(labelbottom=False)

    # Row 2: Relative residual
    ax_res = fig.add_subplot(gs[1, 1], sharex=ax_sb)
    ax_res.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    if good.any():
        ax_res.scatter(x_axis[good], rel_residual[good] * 100,
                       color='green', s=20, alpha=0.7)
    ax_res.set_ylabel('Rel. Residual (%)', fontsize=10)
    ax_res.grid(alpha=0.3)
    ax_res.tick_params(labelbottom=False)

    # Row 3: Ellipticity
    ax_eps = fig.add_subplot(gs[2, 1], sharex=ax_sb)
    ax_eps.axhline(eps, color='k', linestyle='--', linewidth=2, label='True', zorder=10)
    if good.any():
        ax_eps.scatter(x_axis[good], eps_fit[good], color='green', s=20, alpha=0.7, label='Fitted')
    ax_eps.set_ylabel('Ellipticity', fontsize=10)
    ax_eps.legend(fontsize=8, loc='best')
    ax_eps.grid(alpha=0.3)
    ax_eps.tick_params(labelbottom=False)

    # Row 4: Position angle
    ax_pa = fig.add_subplot(gs[3, 1], sharex=ax_sb)
    ax_pa.axhline(pa, color='k', linestyle='--', linewidth=2, label='True', zorder=10)
    if good.any():
        ax_pa.scatter(x_axis[good], pa_normalized[good], color='green', s=20, alpha=0.7, label='Fitted')
    ax_pa.set_ylabel('Position Angle (rad)', fontsize=10)
    ax_pa.legend(fontsize=8, loc='best')
    ax_pa.grid(alpha=0.3)
    ax_pa.tick_params(labelbottom=False)

    # Row 5: Center offset
    ax_center = fig.add_subplot(gs[4, 1], sharex=ax_sb)
    x0_fit = np.array([iso['x0'] for iso in isophotes])
    y0_fit = np.array([iso['y0'] for iso in isophotes])
    center_offset = np.sqrt((x0_fit - params['x0'])**2 + (y0_fit - params['y0'])**2)
    if good.any():
        ax_center.scatter(x_axis[good], center_offset[good], color='green', s=20, alpha=0.7)
    ax_center.set_xlabel(r'$\mathrm{SMA}^{0.25}$ (pixels$^{0.25}$)', fontsize=11)
    ax_center.set_ylabel('Center Offset (pix)', fontsize=10)
    ax_center.grid(alpha=0.3)

    # Save figure
    output_dir = Path('tests/qa_outputs')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'test_numba_validation.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nQA figure saved to: {output_path}")

    # Validation
    valid = (sma >= 3) & (sma >= 0.5 * R_e) & (sma <= 5 * R_e) & good
    if valid.sum() > 0:
        max_rel_diff = np.max(np.abs(rel_residual[valid]))
        print(f"  Max rel diff in valid range: {max_rel_diff*100:.3f}%")
        print(f"  Converged: {good.sum()} / {len(isophotes)}")

        assert max_rel_diff < 0.02, \
            f"Max relative difference {max_rel_diff*100:.2f}% exceeds 2% threshold"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
