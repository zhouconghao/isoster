"""
Pytest configuration and shared fixtures for ISOSTER tests.

This module provides:
- Shared fixtures for test images (Sersic models)
- Pytest markers for test categorization
- Common utilities used across test modules
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add tests directory to path for fixtures import
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

from fixtures.sersic_factory import (
    create_sersic_model,
    SersicFactory,
)


# =============================================================================
# Pytest Markers
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "real_data: tests that use real astronomical data (skipped in CI)"
    )
    config.addinivalue_line(
        "markers", "slow: tests that take longer than usual"
    )
    config.addinivalue_line(
        "markers", "benchmark: performance benchmark tests"
    )


# =============================================================================
# Sersic Model Fixtures
# =============================================================================

@pytest.fixture
def sersic_factory():
    """
    Factory for creating Sersic test images with configurable parameters.

    Usage:
        def test_example(sersic_factory):
            image, profile, params = sersic_factory.create(n=4, eps=0.3)
    """
    return SersicFactory()


@pytest.fixture
def sersic_image_n4():
    """
    Standard n=4 Sersic (de Vaucouleurs) test image.

    Returns:
        Tuple of (image, true_profile, params)
    """
    return create_sersic_model(
        R_e=20.0,
        n=4.0,
        I_e=1000.0,
        eps=0.4,
        pa=np.pi / 4,
        oversample=10,
    )


@pytest.fixture
def sersic_image_n1():
    """
    Standard n=1 Sersic (exponential) test image.

    Returns:
        Tuple of (image, true_profile, params)
    """
    return create_sersic_model(
        R_e=25.0,
        n=1.0,
        I_e=1500.0,
        eps=0.3,
        pa=np.pi / 6,
        oversample=5,
    )


@pytest.fixture
def sersic_image_circular():
    """
    Circular (eps=0) Sersic test image.

    Returns:
        Tuple of (image, true_profile, params)
    """
    return create_sersic_model(
        R_e=20.0,
        n=4.0,
        I_e=1000.0,
        eps=0.0,
        pa=0.0,
        oversample=5,
    )


@pytest.fixture
def sersic_image_high_eps():
    """
    High ellipticity (eps=0.7) Sersic test image.

    Returns:
        Tuple of (image, true_profile, params)
    """
    return create_sersic_model(
        R_e=25.0,
        n=1.0,
        I_e=2000.0,
        eps=0.7,
        pa=np.pi / 3,
        oversample=12,
    )


@pytest.fixture
def sersic_image_noisy():
    """
    Sersic test image with Gaussian noise (SNR=100 at R_e).

    Returns:
        Tuple of (image, true_profile, params)
    """
    return create_sersic_model(
        R_e=20.0,
        n=4.0,
        I_e=1000.0,
        eps=0.4,
        pa=np.pi / 4,
        oversample=10,
        noise_snr=100.0,
        seed=42,
    )


# =============================================================================
# Utility Fixtures
# =============================================================================

@pytest.fixture
def default_isoster_config():
    """
    Default IsosterConfig for testing.

    Returns a config dict that can be used with IsosterConfig(**config).
    """
    return {
        'sma0': 10.0,
        'minsma': 3.0,
        'maxsma': 100.0,
        'astep': 0.1,
        'eps': 0.3,
        'pa': 0.785,  # π/4
        'minit': 10,
        'maxit': 50,
        'conver': 0.05,
    }


@pytest.fixture
def output_dir(tmp_path):
    """
    Temporary output directory for test artifacts.

    Use this for tests that generate files during execution.
    """
    out = tmp_path / "test_output"
    out.mkdir()
    return out
