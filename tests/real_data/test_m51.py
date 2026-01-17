"""
Tests using real M51 galaxy data.

These tests verify ISOSTER works correctly on actual astronomical images.
They are marked with @pytest.mark.real_data and skipped by default in CI.

To run: pytest tests/real_data/ -m real_data -v
"""

from pathlib import Path

import numpy as np
import pytest

from isoster import fit_image
from isoster.config import IsosterConfig


# Path to M51 data
M51_PATH = Path(__file__).parent.parent.parent / "examples" / "data" / "m51" / "M51.fits"


@pytest.mark.real_data
class TestM51:
    """Tests using real M51 galaxy data."""

    @pytest.fixture
    def m51_image(self):
        """Load M51 FITS image."""
        if not M51_PATH.exists():
            pytest.skip(f"M51 data not found at {M51_PATH}")

        from astropy.io import fits
        with fits.open(M51_PATH) as hdul:
            image = hdul[0].data.astype(np.float64)
        return image

    def test_m51_basic_fit(self, m51_image):
        """
        Basic isophote fitting on M51.

        Verifies that:
        1. fit_image completes without error
        2. Multiple isophotes are extracted
        3. Most isophotes converge (stop_code = 0)
        """
        # Get image center
        y0, x0 = np.array(m51_image.shape) / 2

        config = IsosterConfig(
            x0=x0,
            y0=y0,
            sma0=10.0,
            minsma=5.0,
            maxsma=min(x0, y0) - 20,  # Stay within image
            astep=0.15,
            eps=0.3,
            pa=0.0,
            minit=10,
            maxit=50,
            conver=0.05,
        )

        results = fit_image(m51_image, mask=None, config=config)
        isophotes = results['isophotes']

        # Basic checks
        assert len(isophotes) > 10, f"Expected >10 isophotes, got {len(isophotes)}"

        # Check convergence rate
        stop_codes = [iso['stop_code'] for iso in isophotes]
        converged = sum(1 for sc in stop_codes if sc == 0)
        convergence_rate = converged / len(isophotes)

        assert convergence_rate > 0.5, \
            f"Low convergence rate: {convergence_rate:.1%} ({converged}/{len(isophotes)})"

        print(f"\nM51 fit results:")
        print(f"  Total isophotes: {len(isophotes)}")
        print(f"  Converged: {converged} ({convergence_rate:.1%})")
        print(f"  SMA range: {isophotes[0]['sma']:.1f} - {isophotes[-1]['sma']:.1f} pixels")

    def test_m51_with_mask(self, m51_image):
        """
        Isophote fitting on M51 with a simple mask.

        Tests that masking (e.g., for stars) works correctly.
        """
        y0, x0 = np.array(m51_image.shape) / 2

        # Create a simple mask (mask out very bright pixels as "stars")
        threshold = np.percentile(m51_image, 99.5)
        mask = m51_image > threshold

        config = IsosterConfig(
            x0=x0,
            y0=y0,
            sma0=10.0,
            minsma=5.0,
            maxsma=min(x0, y0) - 20,
            astep=0.15,
            eps=0.3,
            pa=0.0,
            minit=10,
            maxit=50,
            conver=0.05,
        )

        results = fit_image(m51_image, mask=mask, config=config)
        isophotes = results['isophotes']

        assert len(isophotes) > 5, f"Expected >5 isophotes with mask, got {len(isophotes)}"

        print(f"\nM51 with mask results:")
        print(f"  Masked pixels: {mask.sum()} ({100*mask.sum()/mask.size:.2f}%)")
        print(f"  Total isophotes: {len(isophotes)}")
