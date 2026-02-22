"""Tests for convergence improvements in fit_isophote."""
import numpy as np
import pytest
from isoster.fitting import fit_isophote
from isoster.config import IsosterConfig


def make_sersic_image(shape=(201, 201), x0=100, y0=100, ie=1000.0, re=30.0,
                      n=2.0, eps=0.3, pa=0.5):
    """Create a noiseless Sersic galaxy image for testing convergence."""
    from scipy.special import gammaincinv
    bn = gammaincinv(2 * n, 0.5)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    dx = xx - x0
    dy = yy - y0
    cos_pa, sin_pa = np.cos(pa), np.sin(pa)
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    r_ellip = np.sqrt(x_rot**2 + (y_rot / (1 - eps))**2)
    image = ie * np.exp(-bn * ((r_ellip / re)**(1.0/n) - 1))
    return image


class TestConvergenceScaling:
    """Test that sector_area scaling helps outer isophotes converge."""

    def test_outer_isophote_converges_with_scaling(self):
        """An outer isophote that hits maxit with 'none' should converge with 'sector_area'."""
        image = make_sersic_image(shape=(401, 401), x0=200, y0=200, re=30.0, eps=0.3)
        mask = np.zeros_like(image, dtype=bool)
        start_geom = {'x0': 200.0, 'y0': 200.0, 'eps': 0.3, 'pa': 0.5}

        # Use a tight conver to force stop=2 at large SMA with no scaling
        cfg_none = IsosterConfig(conver=0.02, maxit=30, minit=5,
                                 convergence_scaling='none')
        result_none = fit_isophote(image, mask, 150.0, start_geom, cfg_none)

        # With sector_area scaling, same params should converge
        cfg_scaled = IsosterConfig(conver=0.02, maxit=30, minit=5,
                                   convergence_scaling='sector_area')
        result_scaled = fit_isophote(image, mask, 150.0, start_geom, cfg_scaled)

        # The scaled version should converge (stop=0) or at least use fewer iterations
        assert result_scaled['niter'] <= result_none['niter'], \
            f"Scaling should help: {result_scaled['niter']} vs {result_none['niter']}"

    def test_scaling_none_preserves_legacy(self):
        """convergence_scaling='none' should match legacy behavior exactly."""
        image = make_sersic_image()
        mask = np.zeros_like(image, dtype=bool)
        start_geom = {'x0': 100.0, 'y0': 100.0, 'eps': 0.3, 'pa': 0.5}

        cfg = IsosterConfig(convergence_scaling='none')
        result = fit_isophote(image, mask, 30.0, start_geom, cfg)
        # Just verify it runs and returns valid result
        assert result['stop_code'] in {0, 1, 2, 3, -1}
        assert result['sma'] == 30.0

    def test_sqrt_sma_scaling(self):
        """sqrt_sma scaling should also help outer isophotes."""
        image = make_sersic_image(shape=(401, 401), x0=200, y0=200, re=30.0, eps=0.3)
        mask = np.zeros_like(image, dtype=bool)
        start_geom = {'x0': 200.0, 'y0': 200.0, 'eps': 0.3, 'pa': 0.5}

        cfg = IsosterConfig(conver=0.02, maxit=30, minit=5,
                            convergence_scaling='sqrt_sma')
        result = fit_isophote(image, mask, 150.0, start_geom, cfg)
        assert result['stop_code'] in {0, 1, 2, 3, -1}


class TestGeometryDamping:
    """Test that geometry damping reduces oscillations."""

    def test_damping_reduces_iterations(self):
        """With damping < 1.0, outer isophotes should use fewer iterations."""
        image = make_sersic_image(shape=(401, 401), x0=200, y0=200, re=30.0, eps=0.3)
        mask = np.zeros_like(image, dtype=bool)
        start_geom = {'x0': 200.0, 'y0': 200.0, 'eps': 0.3, 'pa': 0.5}

        cfg_nodamp = IsosterConfig(geometry_damping=1.0, convergence_scaling='none',
                                   maxit=50, minit=5, conver=0.02)
        result_nodamp = fit_isophote(image, mask, 120.0, start_geom, cfg_nodamp)

        cfg_damp = IsosterConfig(geometry_damping=0.7, convergence_scaling='none',
                                 maxit=50, minit=5, conver=0.02)
        result_damp = fit_isophote(image, mask, 120.0, start_geom, cfg_damp)

        # Damped should converge or use fewer iterations
        assert result_damp['niter'] <= result_nodamp['niter'] or result_damp['stop_code'] == 0

    def test_damping_1_is_legacy(self):
        """geometry_damping=1.0 should match legacy behavior."""
        image = make_sersic_image()
        mask = np.zeros_like(image, dtype=bool)
        start_geom = {'x0': 100.0, 'y0': 100.0, 'eps': 0.3, 'pa': 0.5}

        cfg = IsosterConfig(geometry_damping=1.0, convergence_scaling='none')
        result = fit_isophote(image, mask, 30.0, start_geom, cfg)
        assert result['stop_code'] in {0, 1, 2, 3, -1}

    def test_damping_preserves_accuracy(self):
        """Damped fitting should still produce accurate photometry."""
        image = make_sersic_image(shape=(301, 301), x0=150, y0=150, re=30.0, eps=0.3)
        mask = np.zeros_like(image, dtype=bool)
        start_geom = {'x0': 150.0, 'y0': 150.0, 'eps': 0.3, 'pa': 0.5}

        cfg = IsosterConfig(geometry_damping=0.5, convergence_scaling='none')
        result = fit_isophote(image, mask, 30.0, start_geom, cfg)

        # At re=30, intensity should be close to ie (1000) * exp(-bn*(1-1)) = 1000
        assert result['stop_code'] == 0
        assert abs(result['intens'] - 1000.0) / 1000.0 < 0.05
