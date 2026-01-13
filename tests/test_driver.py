"""
Tests for isoster.driver module.
"""

import numpy as np
import pytest
from isoster.driver import fit_central_pixel


def test_central_pixel_rounding_basic():
    """Test that central pixel uses proper rounding, not truncation."""
    # Create a simple test image with unique values
    image = np.arange(100).reshape(10, 10).astype(float)

    # Test case 1: x0=5.6, y0=5.6 should round to [6, 6] not truncate to [5, 5]
    # image[6, 6] = 66, image[5, 5] = 55
    result = fit_central_pixel(image, None, x0=5.6, y0=5.6)
    expected_val = image[6, 6]  # Should be 66
    assert result['intens'] == expected_val, \
        f"For (5.6, 5.6): Expected image[6,6]={expected_val}, got {result['intens']}"
    assert result['stop_code'] == 0
    assert result['valid'] == True


def test_central_pixel_rounding_down():
    """Test rounding down case."""
    image = np.arange(100).reshape(10, 10).astype(float)

    # Test case 2: x0=5.4, y0=5.4 should round to [5, 5]
    result = fit_central_pixel(image, None, x0=5.4, y0=5.4)
    expected_val = image[5, 5]  # Should be 55
    assert result['intens'] == expected_val, \
        f"For (5.4, 5.4): Expected image[5,5]={expected_val}, got {result['intens']}"


def test_central_pixel_rounding_half():
    """Test rounding at exactly .5 (should round to even, but np.round rounds half up)."""
    image = np.arange(100).reshape(10, 10).astype(float)

    # Test case 3: x0=5.5, y0=5.5 should round to [6, 6]
    # np.round uses banker's rounding (round half to even) by default
    result = fit_central_pixel(image, None, x0=5.5, y0=5.5)
    expected_y = int(np.round(5.5))  # 6
    expected_x = int(np.round(5.5))  # 6
    expected_val = image[expected_y, expected_x]
    assert result['intens'] == expected_val, \
        f"For (5.5, 5.5): Expected image[{expected_y},{expected_x}]={expected_val}, got {result['intens']}"


def test_central_pixel_rounding_high():
    """Test rounding with values close to next integer."""
    image = np.arange(100).reshape(10, 10).astype(float)

    # Test case 4: x0=5.9, y0=5.9 should definitely round to [6, 6] not truncate to [5, 5]
    result = fit_central_pixel(image, None, x0=5.9, y0=5.9)
    expected_val = image[6, 6]  # Should be 66
    assert result['intens'] == expected_val, \
        f"For (5.9, 5.9): Expected image[6,6]={expected_val}, got {result['intens']}"


def test_central_pixel_rounding_low():
    """Test rounding with values close to current integer."""
    image = np.arange(100).reshape(10, 10).astype(float)

    # Test case 5: x0=5.1, y0=5.1 should round to [5, 5]
    result = fit_central_pixel(image, None, x0=5.1, y0=5.1)
    expected_val = image[5, 5]  # Should be 55
    assert result['intens'] == expected_val, \
        f"For (5.1, 5.1): Expected image[5,5]={expected_val}, got {result['intens']}"


def test_central_pixel_rounding_with_mask():
    """Test that mask also uses proper rounding."""
    image = np.ones((10, 10)) * 100.0
    mask = np.zeros((10, 10), dtype=bool)

    # Mask pixel [6, 6] but not [5, 5]
    mask[6, 6] = True

    # x0=5.6, y0=5.6 should round to [6, 6] which is masked
    result = fit_central_pixel(image, mask, x0=5.6, y0=5.6)
    assert result['valid'] == False, \
        "Coordinate (5.6, 5.6) should round to masked pixel [6, 6]"
    assert result['stop_code'] == -1
    assert np.isnan(result['intens'])

    # x0=5.4, y0=5.4 should round to [5, 5] which is not masked
    result = fit_central_pixel(image, mask, x0=5.4, y0=5.4)
    assert result['valid'] == True, \
        "Coordinate (5.4, 5.4) should round to unmasked pixel [5, 5]"
    assert result['stop_code'] == 0
    assert result['intens'] == 100.0


def test_central_pixel_asymmetric_rounding():
    """Test that x and y round independently."""
    image = np.arange(100).reshape(10, 10).astype(float)

    # x0=5.3 (rounds to 5), y0=5.7 (rounds to 6)
    result = fit_central_pixel(image, None, x0=5.3, y0=5.7)
    expected_val = image[6, 5]  # y=6, x=5 => 65
    assert result['intens'] == expected_val, \
        f"For (5.3, 5.7): Expected image[6,5]={expected_val}, got {result['intens']}"


@pytest.mark.parametrize("x0,y0,expected_y,expected_x", [
    (5.1, 5.1, 5, 5),
    (5.4, 5.4, 5, 5),
    (5.5, 5.5, 6, 6),  # np.round behavior
    (5.6, 5.6, 6, 6),
    (5.9, 5.9, 6, 6),
    (4.1, 6.9, 7, 4),  # Asymmetric
    (7.2, 3.8, 4, 7),  # Asymmetric
])
def test_central_pixel_rounding_parametrized(x0, y0, expected_y, expected_x):
    """Parametrized test for various rounding scenarios."""
    image = np.arange(100).reshape(10, 10).astype(float)
    result = fit_central_pixel(image, None, x0=x0, y0=y0)
    expected_val = image[expected_y, expected_x]
    assert result['intens'] == expected_val, \
        f"For ({x0}, {y0}): Expected image[{expected_y},{expected_x}]={expected_val}, got {result['intens']}"
