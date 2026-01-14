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


if __name__ == '__main__':
    unittest.main()
