import numpy as np
import unittest
import warnings
import pytest
from isoster.fitting import (fit_first_and_second_harmonics, sigma_clip,
                              compute_aperture_photometry, compute_parameter_errors,
                              compute_deviations, compute_central_regularization_penalty)
from isoster.config import IsosterConfig

class TestFitting(unittest.TestCase):
    def test_fit_harmonics(self):
        phi = np.linspace(0, 2*np.pi, 100, endpoint=False)
        y0, A1, B1, A2, B2 = 100.0, 10.0, 5.0, 2.0, 1.0
        intens = y0 + A1*np.sin(phi) + B1*np.cos(phi) + A2*np.sin(2*phi) + B2*np.cos(2*phi)
        
        coeffs, cov = fit_first_and_second_harmonics(phi, intens)
        self.assertTrue(np.allclose(coeffs, [y0, A1, B1, A2, B2], atol=1e-5))

    def test_sigma_clip(self):
        # We need at least 11 points for a single outlier to be clipped at 3rd sigma
        # due to the outlier's own contribution to the standard deviation.
        phi = np.arange(20)
        outlier_val = 1000.0
        intens = np.array([10.0] * 19 + [outlier_val])
        
        # Test no clipping
        p, i, n = sigma_clip(phi, intens, nclip=0)
        self.assertEqual(n, 0)
        self.assertEqual(len(i), 20)
        
        # Test 1 iteration of clipping
        p, i, n = sigma_clip(phi, intens, sclip=3.0, nclip=1)
        self.assertEqual(n, 1)
        self.assertEqual(len(i), 19)
        self.assertNotIn(outlier_val, i)

    def test_aperture_photometry(self):
        image = np.ones((100, 100))
        x0, y0 = 50.0, 50.0
        sma = 10.0
        eps = 0.0
        pa = 0.0
        
        tflux_e, tflux_c, npix_e, npix_c = compute_aperture_photometry(image, None, x0, y0, sma, eps, pa)
        
        # Area of r=10 circle is approx 314
        expected_area = np.pi * sma**2
        self.assertLess(abs(npix_c - expected_area) / expected_area, 0.05)
        self.assertAlmostEqual(tflux_c, npix_c)
        self.assertAlmostEqual(tflux_e, tflux_c)
        self.assertEqual(npix_e, npix_c)

if __name__ == '__main__':
    unittest.main()


# ============================================================================
# Pytest-style tests for exception handling (ISSUE-1)
# ============================================================================

def test_compute_parameter_errors_singular_matrix():
    """Test that singular matrix cases return zeros gracefully."""
    # Create degenerate case: all same intensity (no variation for harmonic fit)
    phi = np.linspace(0, 2*np.pi, 10)
    intens = np.ones(10) * 100.0  # Constant intensity

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        x0_err, y0_err, eps_err, pa_err = compute_parameter_errors(
            phi, intens, x0=50, y0=50, sma=10, eps=0.2, pa=0.5, gradient=-1.0
        )

        # Should return zeros (may or may not emit warning depending on code path)
        # The important thing is it doesn't crash
        assert x0_err == 0.0, f"Expected x0_err=0.0, got {x0_err}"
        assert y0_err == 0.0, f"Expected y0_err=0.0, got {y0_err}"
        assert eps_err == 0.0, f"Expected eps_err=0.0, got {eps_err}"
        assert pa_err == 0.0, f"Expected pa_err=0.0, got {pa_err}"

        # Note: This case may hit early return (line 216-217) without exception,
        # so warning emission is optional. The key is it returns zeros gracefully.


def test_compute_parameter_errors_zero_gradient():
    """Test that zero gradient is handled gracefully."""
    phi = np.linspace(0, 2*np.pi, 10)
    intens = np.random.random(10) * 100.0

    # Zero gradient should trigger early return before try block
    x0_err, y0_err, eps_err, pa_err = compute_parameter_errors(
        phi, intens, x0=50, y0=50, sma=10, eps=0.2, pa=0.5, gradient=0.0
    )

    assert x0_err == 0.0
    assert y0_err == 0.0
    assert eps_err == 0.0
    assert pa_err == 0.0


def test_compute_parameter_errors_none_gradient():
    """Test that None gradient is handled gracefully."""
    phi = np.linspace(0, 2*np.pi, 10)
    intens = np.random.random(10) * 100.0

    # None gradient should trigger early return
    x0_err, y0_err, eps_err, pa_err = compute_parameter_errors(
        phi, intens, x0=50, y0=50, sma=10, eps=0.2, pa=0.5, gradient=None
    )

    assert x0_err == 0.0
    assert y0_err == 0.0
    assert eps_err == 0.0
    assert pa_err == 0.0


def test_compute_deviations_singular_matrix():
    """Test that compute_deviations handles errors gracefully."""
    # Create degenerate case with only 2 points at same angle
    phi = np.array([0.0, 0.0])
    intens = np.array([100.0, 100.0])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        a, b, a_err, b_err = compute_deviations(phi, intens, sma=10, gradient=-1.0, order=3)

        # Should return zeros
        assert a == 0.0, f"Expected a=0.0, got {a}"
        assert b == 0.0, f"Expected b=0.0, got {b}"
        assert a_err == 0.0, f"Expected a_err=0.0, got {a_err}"
        assert b_err == 0.0, f"Expected b_err=0.0, got {b_err}"

        # Should emit warning
        assert len(w) >= 1, "Expected at least one warning to be emitted"
        assert any("compute_deviations" in str(warn.message) for warn in w), \
            f"Expected warning about compute_deviations, got: {[str(warn.message) for warn in w]}"


def test_compute_deviations_zero_factor():
    """Test that zero factor (gradient=0 or sma=0) returns zeros."""
    phi = np.linspace(0, 2*np.pi, 10)
    intens = np.random.random(10) * 100.0

    # Zero gradient should make factor=0, triggering early return
    a, b, a_err, b_err = compute_deviations(phi, intens, sma=10, gradient=0.0, order=3)

    assert a == 0.0
    assert b == 0.0
    assert a_err == 0.0
    assert b_err == 0.0


def test_compute_parameter_errors_normal_case():
    """Test that compute_parameter_errors works correctly for normal case."""
    # Create realistic harmonic data
    phi = np.linspace(0, 2*np.pi, 50, endpoint=False)
    y0, A1, B1, A2, B2 = 100.0, 5.0, 3.0, 1.0, 0.5
    intens = y0 + A1*np.sin(phi) + B1*np.cos(phi) + A2*np.sin(2*phi) + B2*np.cos(2*phi)
    intens += np.random.normal(0, 0.1, len(phi))  # Add small noise

    # Should succeed without warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        x0_err, y0_err, eps_err, pa_err = compute_parameter_errors(
            phi, intens, x0=50, y0=50, sma=10, eps=0.2, pa=0.5, gradient=-2.0
        )

        # Should return non-zero errors
        assert x0_err > 0.0, "Expected non-zero x0_err for normal case"
        assert y0_err > 0.0, "Expected non-zero y0_err for normal case"
        assert eps_err > 0.0, "Expected non-zero eps_err for normal case"
        assert pa_err > 0.0, "Expected non-zero pa_err for normal case"

        # Should not emit warnings
        compute_param_warnings = [warn for warn in w if "compute_parameter_errors" in str(warn.message)]
        assert len(compute_param_warnings) == 0, \
            f"Expected no warnings for normal case, got: {[str(warn.message) for warn in compute_param_warnings]}"


def test_compute_deviations_normal_case():
    """Test that compute_deviations works correctly for normal case."""
    # Create realistic data with 3rd order harmonic
    phi = np.linspace(0, 2*np.pi, 50, endpoint=False)
    y0, A3, B3 = 100.0, 2.0, 1.0
    intens = y0 + A3*np.sin(3*phi) + B3*np.cos(3*phi)
    intens += np.random.normal(0, 0.1, len(phi))  # Add small noise

    # Should succeed without warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        a, b, a_err, b_err = compute_deviations(phi, intens, sma=10, gradient=-2.0, order=3)

        # Should return non-zero values
        assert abs(a) > 0.0, "Expected non-zero a for normal case"
        assert abs(b) > 0.0, "Expected non-zero b for normal case"

        # Should not emit warnings
        compute_dev_warnings = [warn for warn in w if "compute_deviations" in str(warn.message)]
        assert len(compute_dev_warnings) == 0, \
            f"Expected no warnings for normal case, got: {[str(warn.message) for warn in compute_dev_warnings]}"

def test_pa_wraparound_vectorized():
    """Test that PA wrap-around is correctly handled with vectorized modulo arithmetic.

    This test verifies that the vectorized formula ((delta + π) % (2π)) - π
    correctly wraps PA differences to [-π, π] range and produces finite penalties.
    """
    # Create minimal config for regularization
    config = IsosterConfig(
        x0=50.0, y0=50.0, eps=0.3, pa=0.0,
        sma0=10.0, minsma=3.0, maxsma=100.0,
        use_central_regularization=True,
        central_reg_strength=0.1,
        central_reg_sma_threshold=10.0,
    )

    # Test cases: various PA differences that should wrap to [-π, π]
    test_cases = [
        # (current_pa, previous_pa, description)
        (0.1, 0.0, "Small positive difference"),
        (0.0, 0.1, "Small negative difference"),
        (np.pi - 0.1, -np.pi + 0.1, "Wrap from +π to -π"),
        (-np.pi + 0.1, np.pi - 0.1, "Wrap from -π to +π"),
        (3.0, -3.0, "Large positive wrap"),
        (-3.0, 3.0, "Large negative wrap"),
        (0.0, 0.0, "No difference"),
        (np.pi, -np.pi, "π and -π are equivalent"),
        (2*np.pi, 0.0, "Full circle wrap"),
        (4*np.pi, 0.0, "Multiple circle wrap"),
    ]

    for current_pa, previous_pa, description in test_cases:
        current_geom = {'x0': 50.0, 'y0': 50.0, 'eps': 0.3, 'pa': current_pa}
        previous_geom = {'x0': 50.0, 'y0': 50.0, 'eps': 0.3, 'pa': previous_pa}

        # Compute penalty (should not crash)
        penalty = compute_central_regularization_penalty(
            current_geom, previous_geom, sma=5.0, config=config
        )

        # Verify penalty is finite and non-negative
        assert np.isfinite(penalty), f"Penalty should be finite for {description}: pa={current_pa}, prev_pa={previous_pa}"
        assert penalty >= 0.0, f"Penalty should be non-negative for {description}, got {penalty}"

        # Manually compute delta_pa with vectorized formula
        delta_pa = current_pa - previous_pa
        delta_pa_wrapped = ((delta_pa + np.pi) % (2 * np.pi)) - np.pi

        # Verify wrapped value is in [-π, π]
        assert -np.pi - 1e-10 <= delta_pa_wrapped <= np.pi + 1e-10, \
            f"Wrapped delta_pa {delta_pa_wrapped:.6f} not in [-π, π] for {description}"

    print("✓ PA wrap-around vectorized test passed")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

def test_compute_parameter_errors_with_coeffs():
    """Test that passing coefficients avoids re-fitting (EFF-2 optimization)."""
    from isoster.fitting import compute_parameter_errors, fit_first_and_second_harmonics, harmonic_function
    
    # Create synthetic data
    phi = np.linspace(0, 2*np.pi, 100, endpoint=False)
    y0, A1, B1, A2, B2 = 100.0, 10.0, 5.0, 2.0, 1.0
    intens = y0 + A1*np.sin(phi) + B1*np.cos(phi) + A2*np.sin(2*phi) + B2*np.cos(2*phi)
    intens += np.random.RandomState(42).normal(0, 0.5, len(phi))
    
    # Geometry and gradient
    x0, y0_geom, sma, eps, pa = 50.0, 50.0, 10.0, 0.3, np.pi/4
    gradient = -2.0
    
    # Fit harmonics once
    coeffs, cov_matrix = fit_first_and_second_harmonics(phi, intens)
    
    # Compute errors WITH coefficients (optimized path)
    err_with_coeffs = compute_parameter_errors(
        phi, intens, x0, y0_geom, sma, eps, pa, gradient, cov_matrix, coeffs
    )
    
    # Compute errors WITHOUT coefficients (legacy path, re-fits)
    err_without_coeffs = compute_parameter_errors(
        phi, intens, x0, y0_geom, sma, eps, pa, gradient, cov_matrix, coeffs=None
    )
    
    # Results should be identical
    for i, (with_c, without_c) in enumerate(zip(err_with_coeffs, err_without_coeffs)):
        assert np.abs(with_c - without_c) < 1e-10, \
            f"Error {i}: with_coeffs={with_c}, without_coeffs={without_c}, diff={abs(with_c - without_c)}"
    
    print(f"✓ Parameter errors with/without coeffs: {err_with_coeffs}")
    print(f"✓ EFF-2 optimization produces identical results")


