import unittest

import numpy as np

from isoster.config import IsosterConfig
from isoster.driver import fit_image
from isoster.sampling import extract_isophote_data, get_elliptical_coordinates


class TestSampling(unittest.TestCase):
    def test_get_elliptical_coordinates(self):
        # Test circular
        x0, y0 = 50.0, 50.0
        pa, eps = 0.0, 0.0

        # Point at (60, 50) should have sma=10, phi=0
        sma, phi = get_elliptical_coordinates(60.0, 50.0, x0, y0, pa, eps)
        self.assertAlmostEqual(sma, 10.0)
        self.assertAlmostEqual(phi, 0.0)

        # Point at (50, 60) should have sma=10, phi=pi/2
        sma, phi = get_elliptical_coordinates(50.0, 60.0, x0, y0, pa, eps)
        self.assertAlmostEqual(sma, 10.0)
        self.assertAlmostEqual(phi, np.pi / 2)

        # Test elliptical (eps=0.5 -> b=a/2)
        eps = 0.5
        # Point at (50, 55) should have sma=10
        sma, phi = get_elliptical_coordinates(50.0, 55.0, x0, y0, pa, eps)
        self.assertAlmostEqual(sma, 10.0)
        self.assertAlmostEqual(phi, np.pi / 2)

    def test_extract_isophote_data(self):
        image = np.zeros((100, 100))
        image[45:55, 45:55] = 10.0

        x0, y0 = 50.0, 50.0
        sma = 2.0
        eps = 0.0
        pa = 0.0

        # extract_isophote_data returns IsophoteData namedtuple with (angles, phi, intens, radii)
        data = extract_isophote_data(image, None, x0, y0, sma, eps, pa)

        # At sma=2, n_samples = max(64, int(2*pi*2)) = 64; most land on bright square
        self.assertGreaterEqual(len(data.angles), 12)
        self.assertTrue(np.allclose(data.intens, 10.0))
        self.assertTrue(np.all(data.radii == sma))

        # Test out of bounds
        data = extract_isophote_data(image, None, -10, -10, sma, eps, pa)
        self.assertEqual(len(data.angles), 0)
        self.assertEqual(len(data.intens), 0)

    def test_eccentric_anomaly_uniform_spacing(self):
        """EA sampling produces a different phi distribution at high eps."""
        image = np.ones((200, 200)) * 100.0
        x0, y0, sma, eps, pa = 100.0, 100.0, 40.0, 0.6, 0.0

        data_phi = extract_isophote_data(image, None, x0, y0, sma, eps, pa, use_eccentric_anomaly=False)
        data_ea = extract_isophote_data(image, None, x0, y0, sma, eps, pa, use_eccentric_anomaly=True)

        # Both should return samples
        self.assertGreater(len(data_phi.angles), 0)
        self.assertGreater(len(data_ea.angles), 0)

        # EA mode redistributes samples along the ellipse arc
        self.assertFalse(
            np.allclose(data_phi.phi, data_ea.phi),
            "EA mode should produce different phi distribution than standard",
        )

    def test_mask_excludes_samples(self):
        """Masked pixels should be excluded from sampled data."""
        image = np.ones((100, 100)) * 100.0
        mask = np.zeros((100, 100), dtype=bool)
        # Mask a quadrant that the isophote passes through
        mask[50:, 50:] = True

        x0, y0, sma, eps, pa = 50.0, 50.0, 20.0, 0.0, 0.0

        data_no_mask = extract_isophote_data(image, None, x0, y0, sma, eps, pa)
        data_masked = extract_isophote_data(image, mask, x0, y0, sma, eps, pa)

        # Masked version should have fewer samples
        self.assertLess(
            len(data_masked.angles),
            len(data_no_mask.angles),
            "Masking should reduce sample count",
        )
        # But still have some samples (3 quadrants unmasked)
        self.assertGreater(len(data_masked.angles), 0)

    def test_linear_step_produces_even_spacing(self):
        """linear_growth=True should produce evenly spaced SMA values."""
        image = np.zeros((100, 100))
        y, x = np.ogrid[:100, :100]
        r = np.sqrt((x - 50) ** 2 + (y - 50) ** 2)
        image = 1000.0 * np.exp(-r / 15.0)

        config = IsosterConfig(
            x0=50.0,
            y0=50.0,
            sma0=10.0,
            minsma=5.0,
            maxsma=30.0,
            astep=2.0,
            linear_growth=True,
            eps=0.0,
            pa=0.0,
            fix_center=True,
            fix_eps=True,
            fix_pa=True,
        )
        result = fit_image(image, config=config)
        sma_values = sorted([iso["sma"] for iso in result["isophotes"] if iso["sma"] > 0])

        if len(sma_values) >= 3:
            # Linear growth: consecutive differences should be constant (=astep)
            diffs = np.diff(sma_values)
            self.assertTrue(
                np.allclose(diffs, 2.0, atol=0.01),
                f"Linear step diffs should be ~2.0, got {diffs}",
            )

    def test_geometric_step_produces_ratio_spacing(self):
        """linear_growth=False (default) should produce geometrically spaced SMA."""
        image = np.zeros((100, 100))
        y, x = np.ogrid[:100, :100]
        r = np.sqrt((x - 50) ** 2 + (y - 50) ** 2)
        image = 1000.0 * np.exp(-r / 15.0)

        config = IsosterConfig(
            x0=50.0,
            y0=50.0,
            sma0=10.0,
            minsma=5.0,
            maxsma=30.0,
            astep=0.2,
            linear_growth=False,
            eps=0.0,
            pa=0.0,
            fix_center=True,
            fix_eps=True,
            fix_pa=True,
        )
        result = fit_image(image, config=config)
        sma_values = sorted([iso["sma"] for iso in result["isophotes"] if iso["sma"] >= 10.0])

        if len(sma_values) >= 3:
            # Geometric growth: consecutive ratios should be constant (=1+astep)
            ratios = [sma_values[i + 1] / sma_values[i] for i in range(len(sma_values) - 1)]
            expected_ratio = 1.2
            self.assertTrue(
                np.allclose(ratios, expected_ratio, rtol=0.01),
                f"Geometric step ratios should be ~{expected_ratio}, got {ratios}",
            )

    def test_large_sma_no_crash(self):
        """SMA larger than half-image should not crash, may return fewer points."""
        image = np.ones((50, 50)) * 100.0
        x0, y0, sma, eps, pa = 25.0, 25.0, 40.0, 0.0, 0.0

        # Should not raise
        data = extract_isophote_data(image, None, x0, y0, sma, eps, pa)
        self.assertIsInstance(data.angles, np.ndarray)

    def test_sma_zero_still_samples(self):
        """SMA=0 still produces samples (central pixel photometry via sampling)."""
        image = np.ones((100, 100)) * 100.0
        data = extract_isophote_data(image, None, 50.0, 50.0, 0.0, 0.0, 0.0)
        # At sma=0 all samples collapse to the center; function still returns them
        self.assertGreater(len(data.angles), 0)


if __name__ == "__main__":
    unittest.main()
