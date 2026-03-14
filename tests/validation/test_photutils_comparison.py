"""
Comparison tests between isoster and photutils.isophote.

This module validates that isoster produces results comparable to photutils.isophote
on identical mock galaxy images. Key metrics:
- Intensity profiles: <1% median difference in 0.5-4 Re
- Ellipticity: <0.01 median difference
- Position angle: <5° median difference
"""

import numpy as np
import pytest

from isoster import fit_image
from isoster.config import IsosterConfig
from tests.fixtures import create_sersic_model as _create_sersic_model


def compute_profile_agreement(iso_isoster, iso_photutils, R_e):
    """Compute agreement metrics between isoster and photutils profiles.

    Focus on 0.5-4 Re range where both should be most reliable.

    Returns:
        dict with keys:
            - intens_median_diff: median fractional intensity difference (%)
            - intens_max_diff: max fractional intensity difference (%)
            - eps_median_diff: median absolute ellipticity difference
            - eps_max_diff: max absolute ellipticity difference
            - pa_median_diff: median absolute PA difference (degrees)
            - pa_max_diff: max absolute PA difference (degrees)
    """
    # Extract isoster data
    sma_iso = np.array([iso["sma"] for iso in iso_isoster])
    intens_iso = np.array([iso["intens"] for iso in iso_isoster])
    eps_iso = np.array([iso["eps"] for iso in iso_isoster])
    pa_iso = np.array([iso["pa"] for iso in iso_isoster])
    stop_iso = np.array([iso["stop_code"] for iso in iso_isoster])

    # Extract photutils data
    sma_phu = np.array([iso["sma"] for iso in iso_photutils])
    intens_phu = np.array([iso["intens"] for iso in iso_photutils])
    eps_phu = np.array([iso["eps"] for iso in iso_photutils])
    pa_phu = np.array([iso["pa"] for iso in iso_photutils])
    stop_phu = np.array([iso["stop_code"] for iso in iso_photutils])

    # Focus on 0.5-4 Re range and converged isophotes
    mask_iso = (sma_iso >= 0.5 * R_e) & (sma_iso <= 4.0 * R_e) & (stop_iso == 0)
    mask_phu = (sma_phu >= 0.5 * R_e) & (sma_phu <= 4.0 * R_e) & (stop_phu == 0)

    # Interpolate photutils to isoster SMA grid
    from scipy.interpolate import interp1d

    if np.sum(mask_phu) < 3:
        # Not enough photutils points for interpolation
        return {
            "intens_median_diff": np.nan,
            "intens_max_diff": np.nan,
            "eps_median_diff": np.nan,
            "eps_max_diff": np.nan,
            "pa_median_diff": np.nan,
            "pa_max_diff": np.nan,
        }

    # Create interpolators for photutils data
    intens_phu_interp = interp1d(
        sma_phu[mask_phu], intens_phu[mask_phu], kind="linear", bounds_error=False, fill_value=np.nan
    )
    eps_phu_interp = interp1d(
        sma_phu[mask_phu], eps_phu[mask_phu], kind="linear", bounds_error=False, fill_value=np.nan
    )
    pa_phu_interp = interp1d(sma_phu[mask_phu], pa_phu[mask_phu], kind="linear", bounds_error=False, fill_value=np.nan)

    # Evaluate at isoster SMA values
    sma_common = sma_iso[mask_iso]
    intens_iso_common = intens_iso[mask_iso]
    eps_iso_common = eps_iso[mask_iso]
    pa_iso_common = pa_iso[mask_iso]

    intens_phu_common = intens_phu_interp(sma_common)
    eps_phu_common = eps_phu_interp(sma_common)
    pa_phu_common = pa_phu_interp(sma_common)

    # Remove NaN values (from extrapolation)
    valid = np.isfinite(intens_phu_common) & np.isfinite(eps_phu_common) & np.isfinite(pa_phu_common)

    if np.sum(valid) == 0:
        return {
            "intens_median_diff": np.nan,
            "intens_max_diff": np.nan,
            "eps_median_diff": np.nan,
            "eps_max_diff": np.nan,
            "pa_median_diff": np.nan,
            "pa_max_diff": np.nan,
        }

    # Compute differences
    intens_frac_diff = 100.0 * np.abs(intens_iso_common[valid] - intens_phu_common[valid]) / intens_phu_common[valid]
    eps_diff = np.abs(eps_iso_common[valid] - eps_phu_common[valid])

    # PA difference (handle wrap-around)
    pa_diff = np.abs(pa_iso_common[valid] - pa_phu_common[valid])
    pa_diff = np.minimum(pa_diff, 2 * np.pi - pa_diff)  # Wrap to [0, π]
    pa_diff_deg = np.degrees(pa_diff)

    return {
        "intens_median_diff": np.median(intens_frac_diff),
        "intens_max_diff": np.max(intens_frac_diff),
        "eps_median_diff": np.median(eps_diff),
        "eps_max_diff": np.max(eps_diff),
        "pa_median_diff": np.median(pa_diff_deg),
        "pa_max_diff": np.max(pa_diff_deg),
    }


def run_photutils_isophote(image, x0, y0, eps, pa, sma0=10.0, minsma=3.0, maxsma=None):
    """Run photutils.isophote on an image with default settings.

    Returns list of isophote dicts in isoster format.
    """
    try:
        from photutils.isophote import Ellipse, EllipseGeometry
    except ImportError:
        pytest.skip("photutils not installed")

    if maxsma is None:
        maxsma = min(image.shape) / 2 - 10

    # Create geometry
    geometry = EllipseGeometry(x0, y0, sma0, eps, pa)

    # Create ellipse fitter
    ellipse = Ellipse(image, geometry)

    # Fit isophotes
    try:
        isolist = ellipse.fit_image()
    except Exception as e:
        # photutils can fail on some images
        print(f"photutils failed: {e}")
        return []

    # Convert to isoster format
    isophotes = []
    for iso in isolist:
        if iso.sma < minsma or iso.sma > maxsma:
            continue

        isophotes.append(
            {
                "sma": iso.sma,
                "intens": iso.intens,
                "eps": iso.eps,
                "pa": iso.pa,
                "x0": iso.x0,
                "y0": iso.y0,
                "stop_code": 0 if iso.stop_code == 0 else -1,  # Map to isoster codes
                "niter": iso.niter,
            }
        )

    return isophotes


class TestPhotutilsComparison:
    """Compare isoster and photutils.isophote on identical mock images."""

    @pytest.mark.parametrize(
        "n,R_e,eps,pa,snr",
        [
            # Circular cases
            (1.0, 20.0, 0.0, 0.0, None),
            (4.0, 20.0, 0.0, 0.0, None),
            # Moderate ellipticity
            (1.0, 20.0, 0.4, np.pi / 4, None),
            (4.0, 20.0, 0.4, np.pi / 4, None),
            # High ellipticity
            (1.0, 20.0, 0.7, np.pi / 3, None),
            # Noisy cases
            (1.0, 20.0, 0.4, np.pi / 4, 100),
            (4.0, 20.0, 0.4, np.pi / 4, 100),
        ],
    )
    def test_isoster_vs_photutils(self, n, R_e, eps, pa, snr):
        """Compare isoster and photutils on Sersic profile."""
        # Create mock image
        I_e = 1000.0
        oversample = 5 if snr is None else 1

        image, _, params = _create_sersic_model(
            R_e,
            n,
            I_e,
            eps,
            pa,
            oversample=oversample,
            noise_snr=snr,
            seed=42,
        )
        x0, y0, shape = params["x0"], params["y0"], params["shape"]

        # Run isoster
        config = IsosterConfig(
            x0=x0,
            y0=y0,
            eps=eps,
            pa=pa,
            sma0=10.0,
            minsma=3.0,
            maxsma=8 * R_e,
            astep=0.15,
            minit=10,
            maxit=50,
            conver=0.05,
            maxgerr=1.0 if eps > 0.6 else 0.5,
            use_eccentric_anomaly=(eps > 0.3),
        )

        results_iso = fit_image(image, None, config)
        isophotes_iso = results_iso["isophotes"]

        # Run photutils
        isophotes_phu = run_photutils_isophote(image, x0, y0, eps, pa, sma0=10.0, minsma=3.0, maxsma=8 * R_e)

        if len(isophotes_phu) == 0:
            pytest.skip("photutils failed to fit isophotes")

        # Compute agreement
        agreement = compute_profile_agreement(isophotes_iso, isophotes_phu, R_e)

        # Print results for debugging
        print(f"\nTest: n={n}, Re={R_e}, eps={eps:.1f}, SNR={snr}")
        print(f"  Intensity: median={agreement['intens_median_diff']:.2f}%, max={agreement['intens_max_diff']:.2f}%")
        print(f"  Ellipticity: median={agreement['eps_median_diff']:.4f}, max={agreement['eps_max_diff']:.4f}")
        print(f"  PA: median={agreement['pa_median_diff']:.2f}°, max={agreement['pa_max_diff']:.2f}°")

        # Acceptance criteria (per EFFICIENCY_OPTIMIZATION_PLAN.md)
        if not np.isnan(agreement["intens_median_diff"]):
            assert agreement["intens_median_diff"] < 1.0, (
                f"Intensity median difference {agreement['intens_median_diff']:.2f}% >= 1%"
            )

        if not np.isnan(agreement["eps_median_diff"]):
            assert agreement["eps_median_diff"] < 0.01, (
                f"Ellipticity median difference {agreement['eps_median_diff']:.4f} >= 0.01"
            )

        # PA comparison only meaningful for non-circular cases (eps > 0.1)
        if eps > 0.1 and not np.isnan(agreement["pa_median_diff"]):
            assert agreement["pa_median_diff"] < 5.0, f"PA median difference {agreement['pa_median_diff']:.2f}° >= 5°"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
