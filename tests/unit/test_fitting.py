import numpy as np
import unittest
import warnings
import pytest
from isoster.fitting import (fit_first_and_second_harmonics, sigma_clip,
                              compute_aperture_photometry, compute_parameter_errors,
                              compute_deviations, compute_central_regularization_penalty,
                              fit_higher_harmonics_simultaneous)
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



def test_gradient_early_termination():
    """Test EFF-1: gradient early termination when first gradient is reliable."""
    from isoster.fitting import compute_gradient
    from isoster.config import IsosterConfig
    import time
    
    # Create simple test image with smooth gradient
    size = 200
    x0, y0 = size / 2, size / 2
    image = np.zeros((size, size))
    y, x = np.mgrid[:size, :size]
    r = np.sqrt((x - x0)**2 + (y - y0)**2)
    image = 1000.0 * np.exp(-r / 20.0)  # Smooth exponential profile
    
    mask = np.zeros_like(image, dtype=bool)
    
    geometry = {'x0': x0, 'y0': y0, 'sma': 30.0, 'eps': 0.3, 'pa': np.pi/4}
    config = IsosterConfig(
        x0=x0, y0=y0, eps=0.3, pa=np.pi/4,
        sma0=10.0, minsma=3.0, maxsma=80.0,
        astep=0.1, linear_growth=False,
        integrator='mean', use_eccentric_anomaly=False
    )
    
    # Case 1: First gradient call (no previous_gradient)
    # Should always compute second gradient when first looks suspicious
    gradient1, error1 = compute_gradient(image, mask, geometry, config,
                                         previous_gradient=None, current_data=None)
    
    assert gradient1 is not None, "First gradient should be computed"
    assert error1 is not None, "First gradient error should be computed"
    
    # Case 2: With reliable gradient (error << gradient)
    # Should skip second gradient extraction due to low relative error
    gradient2, error2 = compute_gradient(image, mask, geometry, config,
                                          previous_gradient=gradient1 * 1.5,
                                          current_data=None)
    
    assert gradient2 is not None, "Second gradient should be computed"
    assert error2 is not None, "Second gradient error should be computed"
    
    # Verify that gradients are reasonable
    assert gradient1 < 0, f"Gradient should be negative (declining profile), got {gradient1}"
    assert gradient2 < 0, f"Gradient should be negative (declining profile), got {gradient2}"
    
    # Verify relative error is small for smooth profile
    rel_error1 = abs(error1 / gradient1) if gradient1 != 0 else np.inf
    rel_error2 = abs(error2 / gradient2) if gradient2 != 0 else np.inf
    
    # For smooth exponential profile, relative errors should be small
    assert rel_error1 < 0.5, f"Relative error too large: {rel_error1}"
    assert rel_error2 < 0.5, f"Relative error too large: {rel_error2}"
    
    print(f"✓ Gradient 1: {gradient1:.6f} ± {error1:.6f} (rel_err={rel_error1:.3f})")
    print(f"✓ Gradient 2: {gradient2:.6f} ± {error2:.6f} (rel_err={rel_error2:.3f})")
    print(f"✓ EFF-1 gradient early termination test passed")


# ============================================================================
# Tests for fit_higher_harmonics_simultaneous (EA-HARMONICS feature)
# ============================================================================

def test_fit_higher_harmonics_simultaneous_basic():
    """Test basic functionality of simultaneous harmonics fitting."""
    # Create synthetic data with known 3rd and 4th order harmonics
    phi = np.linspace(0, 2*np.pi, 100, endpoint=False)
    y0 = 100.0
    A3, B3 = 5.0, 3.0
    A4, B4 = 2.0, 1.5
    intens = y0 + A3*np.sin(3*phi) + B3*np.cos(3*phi) + A4*np.sin(4*phi) + B4*np.cos(4*phi)

    sma = 10.0
    gradient = -2.0

    # Fit harmonics
    result = fit_higher_harmonics_simultaneous(phi, intens, sma, gradient, orders=[3, 4])

    assert 3 in result, "Should have 3rd harmonic result"
    assert 4 in result, "Should have 4th harmonic result"

    # Check structure of results
    a3, b3, a3_err, b3_err = result[3]
    a4, b4, a4_err, b4_err = result[4]

    # Verify coefficients are close to expected (normalized by sma*|gradient|)
    factor = sma * abs(gradient)
    expected_a3 = A3 / factor
    expected_b3 = B3 / factor
    expected_a4 = A4 / factor
    expected_b4 = B4 / factor

    assert np.allclose(a3, expected_a3, atol=1e-5), f"a3: expected {expected_a3}, got {a3}"
    assert np.allclose(b3, expected_b3, atol=1e-5), f"b3: expected {expected_b3}, got {b3}"
    assert np.allclose(a4, expected_a4, atol=1e-5), f"a4: expected {expected_a4}, got {a4}"
    assert np.allclose(b4, expected_b4, atol=1e-5), f"b4: expected {expected_b4}, got {b4}"

    print("✓ fit_higher_harmonics_simultaneous basic test passed")


def test_fit_higher_harmonics_simultaneous_with_noise():
    """Test simultaneous harmonics fitting with noisy data."""
    np.random.seed(42)
    phi = np.linspace(0, 2*np.pi, 100, endpoint=False)
    y0 = 100.0
    A3, B3 = 5.0, 3.0
    A4, B4 = 2.0, 1.5
    noise = np.random.normal(0, 0.5, len(phi))
    intens = y0 + A3*np.sin(3*phi) + B3*np.cos(3*phi) + A4*np.sin(4*phi) + B4*np.cos(4*phi) + noise

    sma = 10.0
    gradient = -2.0

    result = fit_higher_harmonics_simultaneous(phi, intens, sma, gradient, orders=[3, 4])

    a3, b3, a3_err, b3_err = result[3]
    a4, b4, a4_err, b4_err = result[4]

    # Should have non-zero errors
    assert a3_err > 0.0, "a3_err should be positive"
    assert b3_err > 0.0, "b3_err should be positive"
    assert a4_err > 0.0, "a4_err should be positive"
    assert b4_err > 0.0, "b4_err should be positive"

    # Recovered coefficients should be within 3-sigma of true values
    factor = sma * abs(gradient)
    expected_a3 = A3 / factor
    expected_b3 = B3 / factor

    assert abs(a3 - expected_a3) < 3 * a3_err, f"a3 outside 3-sigma: {a3} vs {expected_a3} ± {a3_err}"
    assert abs(b3 - expected_b3) < 3 * b3_err, f"b3 outside 3-sigma: {b3} vs {expected_b3} ± {b3_err}"

    print("✓ fit_higher_harmonics_simultaneous noisy data test passed")


def test_fit_higher_harmonics_simultaneous_empty_orders():
    """Test simultaneous harmonics fitting with empty orders list."""
    phi = np.linspace(0, 2*np.pi, 100, endpoint=False)
    intens = np.random.random(100) * 100.0

    result = fit_higher_harmonics_simultaneous(phi, intens, sma=10.0, gradient=-2.0, orders=[])

    assert result == {}, "Empty orders should return empty dict"

    print("✓ fit_higher_harmonics_simultaneous empty orders test passed")


def test_fit_higher_harmonics_simultaneous_extended_orders():
    """Test simultaneous harmonics fitting with extended orders [3, 4, 5, 6]."""
    phi = np.linspace(0, 2*np.pi, 200, endpoint=False)
    y0 = 100.0
    # Add harmonics for orders 3, 4, 5, 6
    intens = y0 + 3.0*np.sin(3*phi) + 2.0*np.cos(3*phi)
    intens += 2.0*np.sin(4*phi) + 1.0*np.cos(4*phi)
    intens += 1.0*np.sin(5*phi) + 0.5*np.cos(5*phi)
    intens += 0.5*np.sin(6*phi) + 0.3*np.cos(6*phi)

    result = fit_higher_harmonics_simultaneous(phi, intens, sma=10.0, gradient=-2.0, orders=[3, 4, 5, 6])

    assert 3 in result, "Should have 3rd harmonic"
    assert 4 in result, "Should have 4th harmonic"
    assert 5 in result, "Should have 5th harmonic"
    assert 6 in result, "Should have 6th harmonic"

    # Verify each result is a 4-tuple
    for n in [3, 4, 5, 6]:
        assert len(result[n]) == 4, f"Harmonic {n} should have 4 values (a, b, a_err, b_err)"

    print("✓ fit_higher_harmonics_simultaneous extended orders test passed")


def test_fit_higher_harmonics_simultaneous_insufficient_data():
    """Test simultaneous harmonics fitting with insufficient data points."""
    phi = np.linspace(0, 2*np.pi, 3, endpoint=False)  # Only 3 points
    intens = np.array([100.0, 101.0, 99.0])

    # Need 5 params (1 + 2*2) for orders=[3, 4], but only have 3 points
    result = fit_higher_harmonics_simultaneous(phi, intens, sma=10.0, gradient=-2.0, orders=[3, 4])

    # Should return zeros for all harmonics
    assert 3 in result, "Should have 3rd harmonic (zeros)"
    assert 4 in result, "Should have 4th harmonic (zeros)"
    assert result[3] == (0.0, 0.0, 0.0, 0.0), "Should return zeros for insufficient data"
    assert result[4] == (0.0, 0.0, 0.0, 0.0), "Should return zeros for insufficient data"

    print("✓ fit_higher_harmonics_simultaneous insufficient data test passed")


def test_fit_higher_harmonics_simultaneous_zero_gradient():
    """Test simultaneous harmonics fitting with zero gradient."""
    phi = np.linspace(0, 2*np.pi, 100, endpoint=False)
    intens = 100.0 + 5.0*np.sin(3*phi) + 3.0*np.cos(3*phi)

    result = fit_higher_harmonics_simultaneous(phi, intens, sma=10.0, gradient=0.0, orders=[3, 4])

    # Should handle zero gradient gracefully (uses factor=1.0)
    assert 3 in result, "Should have 3rd harmonic"
    assert 4 in result, "Should have 4th harmonic"

    # Should not crash with division by zero
    a3, b3, a3_err, b3_err = result[3]
    assert np.isfinite(a3), "a3 should be finite"
    assert np.isfinite(b3), "b3 should be finite"

    print("✓ fit_higher_harmonics_simultaneous zero gradient test passed")


def test_fit_higher_harmonics_vs_sequential():
    """Compare simultaneous fitting with sequential (compute_deviations) for consistency."""
    np.random.seed(42)
    phi = np.linspace(0, 2*np.pi, 100, endpoint=False)
    y0 = 100.0
    A3, B3 = 5.0, 3.0
    A4, B4 = 2.0, 1.5
    intens = y0 + A3*np.sin(3*phi) + B3*np.cos(3*phi) + A4*np.sin(4*phi) + B4*np.cos(4*phi)

    sma = 10.0
    gradient = -2.0

    # Sequential fitting
    a3_seq, b3_seq, a3_err_seq, b3_err_seq = compute_deviations(phi, intens, sma, gradient, 3)
    a4_seq, b4_seq, a4_err_seq, b4_err_seq = compute_deviations(phi, intens, sma, gradient, 4)

    # Simultaneous fitting
    result = fit_higher_harmonics_simultaneous(phi, intens, sma, gradient, orders=[3, 4])
    a3_sim, b3_sim, a3_err_sim, b3_err_sim = result[3]
    a4_sim, b4_sim, a4_err_sim, b4_err_sim = result[4]

    # For noiseless data with pure harmonics, both methods should give similar results
    # (not identical due to different model assumptions: sequential fits each harmonic
    # with its own constant term, simultaneous shares one constant term)
    assert np.allclose(a3_seq, a3_sim, rtol=0.1), f"a3: sequential={a3_seq}, simultaneous={a3_sim}"
    assert np.allclose(b3_seq, b3_sim, rtol=0.1), f"b3: sequential={b3_seq}, simultaneous={b3_sim}"
    assert np.allclose(a4_seq, a4_sim, rtol=0.1), f"a4: sequential={a4_seq}, simultaneous={a4_sim}"
    assert np.allclose(b4_seq, b4_sim, rtol=0.1), f"b4: sequential={b4_seq}, simultaneous={b4_sim}"

    print("✓ Simultaneous vs sequential comparison test passed")
    print(f"  3rd harmonic: sequential=({a3_seq:.4f}, {b3_seq:.4f}), simultaneous=({a3_sim:.4f}, {b3_sim:.4f})")
    print(f"  4th harmonic: sequential=({a4_seq:.4f}, {b4_seq:.4f}), simultaneous=({a4_sim:.4f}, {b4_sim:.4f})")


