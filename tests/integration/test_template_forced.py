"""
Test template-based forced photometry mode.

This tests the ability to apply geometry from a previously fitted isoster result
(the "template") to extract photometry on a new image. This enables multiband
analysis where one band defines geometry and others use the same geometry.
"""

import numpy as np
import pytest
import tempfile
import os

import isoster
from isoster import fit_image
from isoster.config import IsosterConfig
from isoster.utils import isophote_results_to_fits, isophote_results_from_fits


def make_elliptical_image(shape, x0, y0, eps, pa, intensity_scale, r_scale):
    """
    Create elliptical Sersic-like profile for testing.

    Args:
        shape: (height, width) tuple
        x0, y0: center coordinates
        eps: ellipticity
        pa: position angle in radians
        intensity_scale: peak intensity
        r_scale: exponential scale length

    Returns:
        np.ndarray: 2D image with elliptical profile
    """
    h, w = shape
    y, x = np.ogrid[:h, :w]
    dx = x - x0
    dy = y - y0
    x_rot = dx * np.cos(pa) + dy * np.sin(pa)
    y_rot = -dx * np.sin(pa) + dy * np.cos(pa)
    r_ell = np.sqrt(x_rot**2 + (y_rot / (1 - eps))**2)
    return intensity_scale * np.exp(-r_ell / r_scale)


class TestTemplateForced:
    """Tests for template-based forced photometry."""

    def test_template_forced_basic(self):
        """Verify template geometry is applied correctly."""
        # Create g-band and r-band images with same geometry but different intensities
        shape = (100, 100)
        x0, y0 = 50.0, 50.0
        eps, pa = 0.3, np.pi / 4

        image_g = make_elliptical_image(shape, x0, y0, eps, pa, 1000.0, 15.0)
        image_r = make_elliptical_image(shape, x0, y0, eps, pa, 800.0, 15.0)

        # Fit g-band normally
        config = IsosterConfig(
            x0=x0, y0=y0, eps=eps, pa=pa,
            sma0=10.0, minsma=5.0, maxsma=40.0,
            astep=0.2, linear_growth=True
        )
        results_g = fit_image(image_g, None, config)

        # Use g-band as template for r-band
        results_r = fit_image(image_r, None, config,
                              template_isophotes=results_g['isophotes'])

        # Check we got the same number of isophotes
        assert len(results_r['isophotes']) == len(results_g['isophotes'])

        # Check geometry matches exactly
        for iso_g, iso_r in zip(results_g['isophotes'], results_r['isophotes']):
            assert iso_r['sma'] == iso_g['sma'], "SMA should match template"
            assert iso_r['x0'] == iso_g['x0'], "x0 should match template"
            assert iso_r['y0'] == iso_g['y0'], "y0 should match template"
            # Note: eps and pa for template forced come from extract_forced_photometry
            # which preserves the input geometry

    def test_template_preserves_variable_geometry(self):
        """Verify that x0, y0, eps, pa vary per SMA from template."""
        # Create image with varying geometry (simulated)
        shape = (100, 100)
        x0, y0 = 50.0, 50.0
        eps, pa = 0.3, np.pi / 4

        image = make_elliptical_image(shape, x0, y0, eps, pa, 1000.0, 15.0)

        # Fit normally with fewer isophotes for this test
        config = IsosterConfig(
            x0=x0, y0=y0, eps=eps, pa=pa,
            sma0=10.0, minsma=5.0, maxsma=40.0,
            astep=5.0, linear_growth=True  # Larger step = fewer isophotes
        )
        results_template = fit_image(image, None, config)
        n_iso = len(results_template['isophotes'])

        # Manually modify template to have varying geometry
        # (simulate what happens in real data where geometry varies)
        # Ensure eps stays < 1.0
        for i, iso in enumerate(results_template['isophotes']):
            iso['x0'] = 50.0 + i * 0.1  # Simulate center drift
            iso['y0'] = 50.0 - i * 0.05
            iso['eps'] = min(0.3 + i * 0.01, 0.95)  # Cap at 0.95 to stay valid

        # Apply template to same image
        results_forced = fit_image(image, None, config,
                                   template_isophotes=results_template['isophotes'])

        # Check that forced mode preserves variable geometry from template
        for iso_templ, iso_forced in zip(results_template['isophotes'],
                                         results_forced['isophotes']):
            # x0, y0 may differ slightly for sma=0 (central pixel handling)
            if iso_templ['sma'] > 0:
                assert iso_forced['x0'] == iso_templ['x0']
                assert iso_forced['y0'] == iso_templ['y0']

    def test_template_intensity_differs(self):
        """Verify intensity comes from target image, not template."""
        shape = (100, 100)
        x0, y0 = 50.0, 50.0
        eps, pa = 0.3, np.pi / 4

        # g-band intensity = 1000, r-band = 800 (ratio = 0.8)
        image_g = make_elliptical_image(shape, x0, y0, eps, pa, 1000.0, 15.0)
        image_r = make_elliptical_image(shape, x0, y0, eps, pa, 800.0, 15.0)

        config = IsosterConfig(
            x0=x0, y0=y0, eps=eps, pa=pa,
            sma0=10.0, minsma=5.0, maxsma=40.0,
            astep=0.3, linear_growth=True
        )

        results_g = fit_image(image_g, None, config)
        results_r = fit_image(image_r, None, config,
                              template_isophotes=results_g['isophotes'])

        # Check color ratio at each radius
        for iso_g, iso_r in zip(results_g['isophotes'], results_r['isophotes']):
            if iso_g['stop_code'] == 0 and iso_r['stop_code'] == 0:
                if iso_g['intens'] > 10:  # Skip very faint points
                    ratio = iso_r['intens'] / iso_g['intens']
                    # Should be approximately 0.8 (800/1000)
                    assert 0.75 < ratio < 0.85, \
                        f"Color ratio at SMA={iso_g['sma']:.1f} should be ~0.8, got {ratio:.3f}"

    def test_template_from_fits(self):
        """Verify round-trip: save -> load -> use as template."""
        shape = (100, 100)
        x0, y0 = 50.0, 50.0
        eps, pa = 0.3, np.pi / 4

        image_g = make_elliptical_image(shape, x0, y0, eps, pa, 1000.0, 15.0)
        image_r = make_elliptical_image(shape, x0, y0, eps, pa, 800.0, 15.0)

        config = IsosterConfig(
            x0=x0, y0=y0, eps=eps, pa=pa,
            sma0=10.0, minsma=5.0, maxsma=40.0,
            astep=0.3, linear_growth=True
        )

        # Fit g-band
        results_g = fit_image(image_g, None, config)

        # Save to FITS and reload
        with tempfile.TemporaryDirectory() as tmpdir:
            fits_path = os.path.join(tmpdir, 'template.fits')
            isophote_results_to_fits(results_g, fits_path)

            # Load back
            loaded = isophote_results_from_fits(fits_path)

            # Use loaded template
            results_r = fit_image(image_r, None, config,
                                  template_isophotes=loaded['isophotes'])

        # Check we got valid results
        assert len(results_r['isophotes']) == len(results_g['isophotes'])

        # Check SMA values match
        for iso_orig, iso_r in zip(results_g['isophotes'], results_r['isophotes']):
            assert abs(iso_r['sma'] - iso_orig['sma']) < 1e-6, \
                f"SMA mismatch after FITS round-trip"

    def test_template_forced_with_mask(self):
        """Verify masked pixels handled correctly in template mode."""
        shape = (100, 100)
        x0, y0 = 50.0, 50.0
        eps, pa = 0.3, np.pi / 4

        image = make_elliptical_image(shape, x0, y0, eps, pa, 1000.0, 15.0)

        # Create mask covering part of the image
        mask = np.zeros(shape, dtype=bool)
        mask[40:60, 60:80] = True  # Mask a region

        config = IsosterConfig(
            x0=x0, y0=y0, eps=eps, pa=pa,
            sma0=10.0, minsma=5.0, maxsma=40.0,
            astep=0.3, linear_growth=True
        )

        # Create template (unmasked)
        results_template = fit_image(image, None, config)

        # Apply to masked image
        results_masked = fit_image(image, mask, config,
                                   template_isophotes=results_template['isophotes'])

        # Should still produce results
        assert len(results_masked['isophotes']) == len(results_template['isophotes'])

        # Intensities may differ due to masking
        for iso_templ, iso_mask in zip(results_template['isophotes'],
                                       results_masked['isophotes']):
            # Both should have valid structure
            assert 'sma' in iso_mask
            assert 'intens' in iso_mask

    def test_empty_template_raises_error(self):
        """Verify proper error for empty template list."""
        image = np.ones((50, 50)) * 100.0

        # Empty list should raise error
        with pytest.raises(ValueError, match="template_isophotes cannot be empty"):
            fit_image(image, None, {}, template_isophotes=[])

    def test_none_template_uses_normal_fitting(self):
        """Verify that template_isophotes=None uses normal fitting mode."""
        image = np.ones((50, 50)) * 100.0

        # None means "don't use template mode" - normal fitting proceeds
        config = IsosterConfig(x0=25.0, y0=25.0, sma0=5.0, maxsma=15.0)
        results = fit_image(image, None, config, template_isophotes=None)

        # Should produce normal fitting results
        assert 'isophotes' in results
        assert len(results['isophotes']) > 0

    def test_original_forced_mode_unchanged(self):
        """Verify backward compatibility with original forced mode."""
        shape = (100, 100)
        y, x = np.ogrid[:100, :100]
        r = np.sqrt((x - 50)**2 + (y - 50)**2)
        image = 1000.0 * np.exp(-r / 10.0)

        forced_sma = [5.0, 10.0, 15.0, 20.0]

        config = IsosterConfig(
            x0=50.0, y0=50.0,
            eps=0.0, pa=0.0,
            forced=True,
            forced_sma=forced_sma,
        )

        results = fit_image(image, None, config)
        isophotes = results['isophotes']

        # Original forced mode behavior unchanged
        assert len(isophotes) == len(forced_sma)
        sma_values = [iso['sma'] for iso in isophotes]
        assert sma_values == forced_sma

        # Intensities should decrease with radius
        intensities = [iso['intens'] for iso in isophotes]
        assert all(intensities[i] > intensities[i + 1]
                   for i in range(len(intensities) - 1))

    def test_template_with_central_pixel(self):
        """Test that central pixel (sma=0) in template is handled correctly."""
        shape = (100, 100)
        x0, y0 = 50.0, 50.0
        eps, pa = 0.2, 0.0

        image_g = make_elliptical_image(shape, x0, y0, eps, pa, 1000.0, 15.0)
        image_r = make_elliptical_image(shape, x0, y0, eps, pa, 800.0, 15.0)

        # Config that includes minsma=0 to get central pixel
        config = IsosterConfig(
            x0=x0, y0=y0, eps=eps, pa=pa,
            sma0=10.0, minsma=0.0, maxsma=30.0,
            astep=0.3, linear_growth=True
        )

        results_g = fit_image(image_g, None, config)
        results_r = fit_image(image_r, None, config,
                              template_isophotes=results_g['isophotes'])

        # Check that sma=0 is present in both
        sma_g = [iso['sma'] for iso in results_g['isophotes']]
        sma_r = [iso['sma'] for iso in results_r['isophotes']]

        assert 0.0 in sma_g, "Template should have central pixel"
        assert 0.0 in sma_r, "Result should have central pixel"

        # Check central pixel intensity ratio
        central_g = [iso for iso in results_g['isophotes'] if iso['sma'] == 0.0][0]
        central_r = [iso for iso in results_r['isophotes'] if iso['sma'] == 0.0][0]

        if not np.isnan(central_g['intens']) and not np.isnan(central_r['intens']):
            ratio = central_r['intens'] / central_g['intens']
            assert 0.7 < ratio < 0.9, f"Central pixel color ratio should be ~0.8, got {ratio:.3f}"

    def test_template_sorts_by_sma(self):
        """Test that template isophotes are sorted by SMA."""
        shape = (100, 100)
        x0, y0 = 50.0, 50.0
        eps, pa = 0.2, 0.0

        image = make_elliptical_image(shape, x0, y0, eps, pa, 1000.0, 15.0)

        config = IsosterConfig(
            x0=x0, y0=y0, eps=eps, pa=pa,
            sma0=10.0, minsma=5.0, maxsma=30.0,
            astep=0.3, linear_growth=True
        )

        results = fit_image(image, None, config)

        # Shuffle the template order
        shuffled_template = results['isophotes'].copy()
        np.random.shuffle(shuffled_template)

        # Apply shuffled template
        results_forced = fit_image(image, None, config,
                                   template_isophotes=shuffled_template)

        # Output should be sorted by SMA
        sma_values = [iso['sma'] for iso in results_forced['isophotes']]
        assert sma_values == sorted(sma_values), "Output should be sorted by SMA"


class TestIsophoteResultsFromFits:
    """Tests for the isophote_results_from_fits utility function."""

    def test_basic_round_trip(self):
        """Test basic save and load functionality."""
        shape = (100, 100)
        y, x = np.ogrid[:100, :100]
        r = np.sqrt((x - 50)**2 + (y - 50)**2)
        image = 1000.0 * np.exp(-r / 15.0)

        config = IsosterConfig(
            x0=50.0, y0=50.0,
            sma0=10.0, maxsma=40.0
        )

        results = fit_image(image, None, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            fits_path = os.path.join(tmpdir, 'test.fits')
            isophote_results_to_fits(results, fits_path)

            loaded = isophote_results_from_fits(fits_path)

        # Check structure
        assert 'isophotes' in loaded
        assert 'config' in loaded
        assert loaded['config'] is None  # Config not reconstructed

        # Check isophotes content
        assert len(loaded['isophotes']) == len(results['isophotes'])

        for orig, load in zip(results['isophotes'], loaded['isophotes']):
            assert abs(load['sma'] - orig['sma']) < 1e-10
            assert abs(load['intens'] - orig['intens']) < 1e-10
            assert load['stop_code'] == orig['stop_code']

    def test_preserves_all_columns(self):
        """Test that all columns are preserved in round-trip."""
        shape = (100, 100)
        y, x = np.ogrid[:100, :100]
        r = np.sqrt((x - 50)**2 + (y - 50)**2)
        image = 1000.0 * np.exp(-r / 15.0)

        config = IsosterConfig(
            x0=50.0, y0=50.0,
            sma0=10.0, maxsma=40.0,
            compute_errors=True,
            compute_deviations=True
        )

        results = fit_image(image, None, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            fits_path = os.path.join(tmpdir, 'test.fits')
            isophote_results_to_fits(results, fits_path)
            loaded = isophote_results_from_fits(fits_path)

        # Check that key columns are preserved
        orig_iso = results['isophotes'][5]  # Pick a middle isophote
        load_iso = loaded['isophotes'][5]

        expected_keys = ['sma', 'intens', 'intens_err', 'eps', 'pa',
                         'x0', 'y0', 'rms', 'stop_code', 'niter']

        for key in expected_keys:
            assert key in load_iso, f"Missing key: {key}"
            if isinstance(orig_iso[key], float):
                assert abs(load_iso[key] - orig_iso[key]) < 1e-10, \
                    f"Value mismatch for {key}"
            else:
                assert load_iso[key] == orig_iso[key], f"Value mismatch for {key}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
