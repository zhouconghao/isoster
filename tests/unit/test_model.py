import numpy as np
import unittest
from isoster.model import build_isoster_model


class TestModel(unittest.TestCase):
    def test_build_isoster_model_basic(self):
        """Test basic model building with single isophote."""
        shape = (100, 100)
        # Create a single isophote
        iso = {
            'x0': 50.0, 'y0': 50.0, 'sma': 10.0, 'eps': 0.0, 'pa': 0.0,
            'intens': 100.0
        }

        model = build_isoster_model(shape, [iso])

        # At sma=10, intensity should be 100
        r = np.sqrt((np.arange(100) - 50.0)**2 + (50.0 - 50.0)**2)
        idx_at_sma = np.argmin(np.abs(r - 10.0))  # Find pixel closest to sma=10
        self.assertAlmostEqual(model[50, idx_at_sma], 100.0, places=1)

        # Outside sma, should use boundary value (or fill)
        # With single isophote, pixels beyond sma=10 get fill value (0.0)
        self.assertEqual(model[50, 70], 0.0)

    def test_build_isoster_model_interpolation(self):
        """Test radial interpolation between isophotes."""
        shape = (100, 100)
        isos = [
            {'x0': 50.0, 'y0': 50.0, 'sma': 10.0, 'eps': 0.0, 'pa': 0.0, 'intens': 100.0},
            {'x0': 50.0, 'y0': 50.0, 'sma': 20.0, 'eps': 0.0, 'pa': 0.0, 'intens': 50.0}
        ]

        model = build_isoster_model(shape, isos)

        # At sma=10, intensity should be ~100
        self.assertAlmostEqual(model[50, 60], 100.0, places=1)  # r=10

        # At sma=20, intensity should be ~50
        self.assertAlmostEqual(model[50, 70], 50.0, places=1)  # r=20

        # At sma=15 (halfway), intensity should be interpolated (~75 for linear)
        val_mid = model[50, 65]  # r=15
        self.assertGreater(val_mid, 50.0)  # Should be > 50
        self.assertLess(val_mid, 100.0)    # Should be < 100
        self.assertAlmostEqual(val_mid, 75.0, delta=5.0)  # ~75 for linear interp

        # Outside sma=20, should be fill value
        self.assertEqual(model[50, 80], 0.0)


    def test_pa_interpolation_wrap_boundary(self):
        """Regression test for I5: PA interpolation across 0/180 boundary.

        When PA wraps from ~170 to ~10 degrees (pi-boundary), linear interpolation
        without unwrapping produces ~90 degrees. With np.unwrap(period=pi), the
        interpolation correctly goes through 180 (=0).
        """
        shape = (100, 100)
        # Three isophotes with PA wrapping: 170 -> 175 -> 5 degrees (in radians)
        isos = [
            {'x0': 50.0, 'y0': 50.0, 'sma': 10.0, 'eps': 0.3,
             'pa': np.radians(170.0), 'intens': 100.0},
            {'x0': 50.0, 'y0': 50.0, 'sma': 20.0, 'eps': 0.3,
             'pa': np.radians(175.0), 'intens': 80.0},
            {'x0': 50.0, 'y0': 50.0, 'sma': 30.0, 'eps': 0.3,
             'pa': np.radians(5.0), 'intens': 60.0},
        ]

        model = build_isoster_model(shape, isos)

        # The model should not have artifacts from PA interpolation errors.
        # Without unwrap, PA at sma~25 would be ~90 degrees (midpoint of 175 and 5)
        # instead of ~180/0 degrees. This causes a twist artifact visible as
        # asymmetric intensity along the minor axis.
        #
        # Verify by checking that the model is roughly symmetric about the expected PA.
        # At sma~25 the PA should be near 180 (=0) degrees, not 90 degrees.
        # A pixel at (50, 75) is on the major axis for PA~0/180; for PA~90 it would
        # be on the minor axis, producing very different intensities.
        val_along_major = model[50, 75]  # r~25, along PA~0 major axis
        val_along_minor = model[75, 50]  # r~25, along PA~90 direction

        # With correct PA interpolation (~0/180 deg), val_along_major should be
        # noticeably higher than val_along_minor because eps=0.3 means the ellipse
        # is elongated along the PA direction.
        # With buggy interpolation (~90 deg), the roles reverse.
        self.assertGreater(val_along_major, val_along_minor,
                           "PA interpolation should go through 180/0, not through 90")

    def test_nan_isophotes_filtered_before_interpolation(self):
        """Regression test for I4: NaN intensities should be filtered out.

        Isophotes with stop_code != 0 may carry NaN intensity. Without filtering,
        interp1d propagates NaN across the whole model.
        """
        shape = (100, 100)
        isos = [
            {'x0': 50.0, 'y0': 50.0, 'sma': 10.0, 'eps': 0.0, 'pa': 0.0, 'intens': 100.0},
            {'x0': 50.0, 'y0': 50.0, 'sma': 20.0, 'eps': 0.0, 'pa': 0.0, 'intens': np.nan},
            {'x0': 50.0, 'y0': 50.0, 'sma': 30.0, 'eps': 0.0, 'pa': 0.0, 'intens': 60.0},
        ]

        model = build_isoster_model(shape, isos)

        # The NaN isophote at sma=20 should be excluded; model should not contain NaN
        # in the region covered by the valid isophotes (sma 10-30)
        val_at_10 = model[50, 60]  # r~10
        val_at_30 = model[50, 80]  # r~30
        self.assertTrue(np.isfinite(val_at_10), f"Model at sma=10 should be finite, got {val_at_10}")
        self.assertTrue(np.isfinite(val_at_30), f"Model at sma=30 should be finite, got {val_at_30}")

    def test_nan_geometry_filtered_before_interpolation(self):
        """Isophotes with NaN geometry columns should be excluded."""
        shape = (100, 100)
        isos = [
            {'x0': 50.0, 'y0': 50.0, 'sma': 10.0, 'eps': 0.0, 'pa': 0.0, 'intens': 100.0},
            {'x0': np.nan, 'y0': 50.0, 'sma': 20.0, 'eps': 0.0, 'pa': 0.0, 'intens': 80.0},
            {'x0': 50.0, 'y0': 50.0, 'sma': 30.0, 'eps': 0.0, 'pa': 0.0, 'intens': 60.0},
        ]

        model = build_isoster_model(shape, isos)

        val_at_10 = model[50, 60]
        self.assertTrue(np.isfinite(val_at_10), f"Model at sma=10 should be finite, got {val_at_10}")

    def test_nan_harmonic_coefficients_treated_as_zero(self):
        """NaN harmonic coefficients should be replaced with 0, not poison the model."""
        shape = (100, 100)
        isos = [
            {'x0': 50.0, 'y0': 50.0, 'sma': 10.0, 'eps': 0.0, 'pa': 0.0,
             'intens': 100.0, 'a3': 0.01, 'b3': 0.01, 'a4': 0.01, 'b4': 0.01},
            {'x0': 50.0, 'y0': 50.0, 'sma': 20.0, 'eps': 0.0, 'pa': 0.0,
             'intens': 80.0, 'a3': np.nan, 'b3': np.inf, 'a4': 0.01, 'b4': 0.01},
            {'x0': 50.0, 'y0': 50.0, 'sma': 30.0, 'eps': 0.0, 'pa': 0.0,
             'intens': 60.0, 'a3': 0.01, 'b3': 0.01, 'a4': 0.01, 'b4': 0.01},
        ]

        model = build_isoster_model(shape, isos, use_harmonics=True, harmonic_orders=[3, 4])

        # Model should be entirely finite despite NaN/Inf harmonic coefficients
        n_nonfinite = np.sum(~np.isfinite(model))
        self.assertEqual(n_nonfinite, 0,
                         f"Model has {n_nonfinite} non-finite pixels from NaN harmonics")


if __name__ == '__main__':
    unittest.main()
