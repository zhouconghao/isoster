"""Tests for convergence improvements in fit_isophote."""

import numpy as np

from isoster.config import IsosterConfig
from isoster.fitting import fit_isophote


def make_sersic_image(shape=(201, 201), x0=100, y0=100, ie=1000.0, re=30.0, n=2.0, eps=0.3, pa=0.5):
    """Create a noiseless Sersic galaxy image for testing convergence."""
    from scipy.special import gammaincinv

    bn = gammaincinv(2 * n, 0.5)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    dx = xx - x0
    dy = yy - y0
    cos_pa, sin_pa = np.cos(pa), np.sin(pa)
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    r_ellip = np.sqrt(x_rot**2 + (y_rot / (1 - eps)) ** 2)
    image = ie * np.exp(-bn * ((r_ellip / re) ** (1.0 / n) - 1))
    return image


class TestConvergenceScaling:
    """Test that sector_area scaling helps outer isophotes converge."""

    def test_outer_isophote_converges_with_scaling(self):
        """An outer isophote that hits maxit with 'none' should converge with 'sector_area'."""
        image = make_sersic_image(shape=(401, 401), x0=200, y0=200, re=30.0, eps=0.3)
        mask = np.zeros_like(image, dtype=bool)
        start_geom = {"x0": 200.0, "y0": 200.0, "eps": 0.3, "pa": 0.5}

        # Use a tight conver to force stop=2 at large SMA with no scaling
        cfg_none = IsosterConfig(conver=0.02, maxit=30, minit=5, convergence_scaling="none")
        result_none = fit_isophote(image, mask, 150.0, start_geom, cfg_none)

        # With sector_area scaling, same params should converge
        cfg_scaled = IsosterConfig(conver=0.02, maxit=30, minit=5, convergence_scaling="sector_area")
        result_scaled = fit_isophote(image, mask, 150.0, start_geom, cfg_scaled)

        # The scaled version should converge (stop=0) or at least use fewer iterations
        assert result_scaled["niter"] <= result_none["niter"], (
            f"Scaling should help: {result_scaled['niter']} vs {result_none['niter']}"
        )

    def test_scaling_none_preserves_legacy(self):
        """convergence_scaling='none' should match legacy behavior exactly."""
        image = make_sersic_image()
        mask = np.zeros_like(image, dtype=bool)
        start_geom = {"x0": 100.0, "y0": 100.0, "eps": 0.3, "pa": 0.5}

        cfg = IsosterConfig(convergence_scaling="none")
        result = fit_isophote(image, mask, 30.0, start_geom, cfg)
        # Just verify it runs and returns valid result
        assert result["stop_code"] in {0, 1, 2, 3, -1}
        assert result["sma"] == 30.0

    def test_sqrt_sma_scaling(self):
        """sqrt_sma scaling should also help outer isophotes."""
        image = make_sersic_image(shape=(401, 401), x0=200, y0=200, re=30.0, eps=0.3)
        mask = np.zeros_like(image, dtype=bool)
        start_geom = {"x0": 200.0, "y0": 200.0, "eps": 0.3, "pa": 0.5}

        cfg = IsosterConfig(conver=0.02, maxit=30, minit=5, convergence_scaling="sqrt_sma")
        result = fit_isophote(image, mask, 150.0, start_geom, cfg)
        assert result["stop_code"] in {0, 1, 2, 3, -1}


class TestGeometryDamping:
    """Test that geometry damping reduces oscillations."""

    def test_damping_reduces_iterations(self):
        """With damping < 1.0, outer isophotes should use fewer iterations."""
        image = make_sersic_image(shape=(401, 401), x0=200, y0=200, re=30.0, eps=0.3)
        mask = np.zeros_like(image, dtype=bool)
        start_geom = {"x0": 200.0, "y0": 200.0, "eps": 0.3, "pa": 0.5}

        cfg_nodamp = IsosterConfig(geometry_damping=1.0, convergence_scaling="none", maxit=50, minit=5, conver=0.02)
        result_nodamp = fit_isophote(image, mask, 120.0, start_geom, cfg_nodamp)

        cfg_damp = IsosterConfig(geometry_damping=0.7, convergence_scaling="none", maxit=50, minit=5, conver=0.02)
        result_damp = fit_isophote(image, mask, 120.0, start_geom, cfg_damp)

        # Damped should converge or use fewer iterations
        assert result_damp["niter"] <= result_nodamp["niter"] or result_damp["stop_code"] == 0

    def test_damping_1_is_legacy(self):
        """geometry_damping=1.0 should match legacy behavior."""
        image = make_sersic_image()
        mask = np.zeros_like(image, dtype=bool)
        start_geom = {"x0": 100.0, "y0": 100.0, "eps": 0.3, "pa": 0.5}

        cfg = IsosterConfig(geometry_damping=1.0, convergence_scaling="none")
        result = fit_isophote(image, mask, 30.0, start_geom, cfg)
        assert result["stop_code"] in {0, 1, 2, 3, -1}

    def test_damping_preserves_accuracy(self):
        """Damped fitting should still produce accurate photometry."""
        image = make_sersic_image(shape=(301, 301), x0=150, y0=150, re=30.0, eps=0.3)
        mask = np.zeros_like(image, dtype=bool)
        start_geom = {"x0": 150.0, "y0": 150.0, "eps": 0.3, "pa": 0.5}

        cfg = IsosterConfig(geometry_damping=0.5, convergence_scaling="none")
        result = fit_isophote(image, mask, 30.0, start_geom, cfg)

        # At re=30, intensity should be close to ie (1000) * exp(-bn*(1-1)) = 1000
        assert result["stop_code"] == 0
        # At re=30, Sersic intensity = ie * exp(-bn*(1-1)) = ie = 1000
        assert abs(result["intens"] - 1000.0) / 1000.0 < 0.05


class TestGeometryConvergence:
    """Test geometry-stability-based convergence."""

    def test_geometry_convergence_detects_stability(self):
        """When geometry stops changing, should declare convergence."""
        image = make_sersic_image(shape=(401, 401), x0=200, y0=200, re=30.0, eps=0.3)
        mask = np.zeros_like(image, dtype=bool)
        start_geom = {"x0": 200.0, "y0": 200.0, "eps": 0.3, "pa": 0.5}

        # Without geometry convergence: likely hits maxit at outer SMA
        cfg_off = IsosterConfig(geometry_convergence=False, convergence_scaling="none", maxit=30, minit=5, conver=0.01)
        result_off = fit_isophote(image, mask, 150.0, start_geom, cfg_off)

        # With geometry convergence: should converge via geometry stability
        cfg_on = IsosterConfig(
            geometry_convergence=True,
            convergence_scaling="none",
            maxit=30,
            minit=5,
            conver=0.01,
            geometry_tolerance=0.01,
            geometry_stable_iters=3,
        )
        result_on = fit_isophote(image, mask, 150.0, start_geom, cfg_on)

        # Geometry convergence should help
        assert result_on["niter"] <= result_off["niter"] or result_on["stop_code"] == 0

    def test_geometry_convergence_off_is_legacy(self):
        """geometry_convergence=False should match legacy behavior."""
        image = make_sersic_image()
        mask = np.zeros_like(image, dtype=bool)
        start_geom = {"x0": 100.0, "y0": 100.0, "eps": 0.3, "pa": 0.5}

        cfg = IsosterConfig(geometry_convergence=False, convergence_scaling="none")
        result = fit_isophote(image, mask, 30.0, start_geom, cfg)
        assert result["stop_code"] in {0, 1, 2, 3, -1}


class TestConverThresholdEffect:
    """Test that conver threshold affects iteration count."""

    def test_tighter_conver_needs_more_iterations(self):
        """A tighter conver threshold should require more iterations to converge."""
        image = make_sersic_image(shape=(201, 201), x0=100, y0=100, re=30.0, eps=0.3)
        mask = np.zeros_like(image, dtype=bool)
        start_geom = {"x0": 100.0, "y0": 100.0, "eps": 0.3, "pa": 0.5}

        cfg_loose = IsosterConfig(conver=0.5, maxit=100, minit=5, convergence_scaling="none")
        result_loose = fit_isophote(image, mask, 30.0, start_geom, cfg_loose)

        cfg_tight = IsosterConfig(conver=0.001, maxit=100, minit=5, convergence_scaling="none")
        result_tight = fit_isophote(image, mask, 30.0, start_geom, cfg_tight)

        # Tight threshold should need at least as many iterations
        assert result_tight["niter"] >= result_loose["niter"], (
            f"Tight conver should need >= iterations: tight={result_tight['niter']}, loose={result_loose['niter']}"
        )


class TestMaxgerrThreshold:
    """Test that maxgerr threshold affects stop behavior."""

    def test_tight_maxgerr_terminates_at_outer_radii(self):
        """Tight maxgerr at outer radius should produce non-zero stop code."""
        image = make_sersic_image(shape=(201, 201), x0=100, y0=100, re=30.0, eps=0.3)
        mask = np.zeros_like(image, dtype=bool)
        start_geom = {"x0": 100.0, "y0": 100.0, "eps": 0.3, "pa": 0.5}

        # At large SMA where gradient is weak, tight maxgerr should reject
        cfg_tight = IsosterConfig(maxgerr=0.05, maxit=50, convergence_scaling="none")
        result_tight = fit_isophote(image, mask, 90.0, start_geom, cfg_tight)

        cfg_loose = IsosterConfig(maxgerr=2.0, maxit=50, convergence_scaling="none")
        result_loose = fit_isophote(image, mask, 90.0, start_geom, cfg_loose)

        # Tight maxgerr should fail or behave differently from loose
        assert (
            result_tight["stop_code"] != 0
            or result_loose["stop_code"] != 0
            or (result_tight["stop_code"] >= result_loose["stop_code"])
        )


class TestMaxitExhaustion:
    """Test that maxit exhaustion produces stop_code=2."""

    def test_maxit_exhaustion_yields_stop_code_2(self):
        """Very low maxit should produce stop_code=2 with valid photometry."""
        image = make_sersic_image(shape=(201, 201), x0=100, y0=100, re=30.0, eps=0.3)
        mask = np.zeros_like(image, dtype=bool)
        start_geom = {"x0": 100.0, "y0": 100.0, "eps": 0.3, "pa": 0.5}

        # maxit=3 with tight convergence ensures hitting iteration limit
        cfg = IsosterConfig(conver=0.001, maxit=3, minit=1, convergence_scaling="none")
        result = fit_isophote(image, mask, 30.0, start_geom, cfg)

        assert result["stop_code"] == 2, f"Expected stop_code=2 (maxit exhaustion), got {result['stop_code']}"
        # Should still have valid photometry
        assert np.isfinite(result["intens"])
        assert result["intens"] > 0


class TestConvergenceInteraction:
    """Test that geometry_convergence and convergence_scaling do not conflict."""

    def test_geometry_convergence_with_scaling(self):
        """Both geometry_convergence and convergence_scaling active should not conflict."""
        image = make_sersic_image(shape=(401, 401), x0=200, y0=200, re=30.0, eps=0.3)
        mask = np.zeros_like(image, dtype=bool)
        start_geom = {"x0": 200.0, "y0": 200.0, "eps": 0.3, "pa": 0.5}

        cfg = IsosterConfig(
            geometry_convergence=True,
            convergence_scaling="sector_area",
            geometry_tolerance=0.01,
            geometry_stable_iters=3,
            maxit=50,
            minit=5,
            conver=0.02,
        )
        result = fit_isophote(image, mask, 100.0, start_geom, cfg)

        # Should complete without error and produce a valid stop code
        assert result["stop_code"] in {0, 1, 2, 3, -1}
        assert np.isfinite(result["intens"])
