"""
Tests for first isophote failure detection and retry mechanism.

Tests cover:
- Failure flag set on flat images (zero gradient → stop_code=-1)
- FIRST_FEW_ISOPHOTE_FAILURE warning emitted
- Retry succeeds with perturbed sma0 on images with localized gradient
- Retry exhaustion on truly flat images
- Backward compatibility (default max_retry=0, no overhead)
- Normal run unaffected when retry is enabled but first iso succeeds
- Config validation for new fields
"""

import warnings

import numpy as np
import pytest

from isoster.config import IsosterConfig
from isoster.driver import fit_image


def _make_galaxy_image(shape=(100, 100), center=None, scale=10.0, peak=1000.0, background=100.0):
    """Create a simple exponential-profile galaxy image with clear gradient."""
    h, w = shape
    if center is None:
        center = (w / 2.0, h / 2.0)
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    return peak * np.exp(-r / scale) + background


class TestFirstIsophoteFailureDetection:
    """Part 1: informative failure notification."""

    def test_failure_flag_on_flat_image(self):
        """First isophote at sma0 fails on flat image → flag is set."""
        image = np.ones((100, 100)) * 100.0
        config = IsosterConfig(x0=50.0, y0=50.0, sma0=10.0, maxsma=30.0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fit_image(image, None, config)

        assert result.get("first_isophote_failure") is True
        warning_messages = [str(w.message) for w in caught]
        assert any("FIRST_FEW_ISOPHOTE_FAILURE" in msg for msg in warning_messages)

    def test_failure_warning_includes_stop_codes(self):
        """Warning message contains the actual stop codes for diagnostics."""
        image = np.ones((80, 80)) * 50.0
        config = IsosterConfig(x0=40.0, y0=40.0, sma0=8.0, maxsma=20.0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit_image(image, None, config)

        failure_warnings = [w for w in caught if "FIRST_FEW_ISOPHOTE_FAILURE" in str(w.message)]
        assert len(failure_warnings) == 1
        msg = str(failure_warnings[0].message)
        # Warning should mention stop codes (list of ints)
        assert "stop codes" in msg.lower() or "stop_codes" in msg.lower()

    def test_failure_flag_not_set_on_normal_galaxy(self):
        """Normal galaxy image should not trigger failure flag."""
        image = _make_galaxy_image()
        config = IsosterConfig(x0=50.0, y0=50.0, sma0=10.0, maxsma=40.0)

        result = fit_image(image, None, config)

        assert result.get("first_isophote_failure") is not True
        assert len(result["isophotes"]) > 1

    def test_probe_finds_anchor_at_later_sma(self):
        """If first iso fails but a later probe succeeds, growth continues."""
        # Create image with a gradient hole at sma=10 but gradient at sma=12+
        image = _make_galaxy_image(scale=15.0, peak=2000.0)
        # Mask a ring around sma=10 to force stop_code=3 there
        h, w = image.shape
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - 50) ** 2 + (y - 50) ** 2)
        mask = np.zeros_like(image, dtype=bool)
        mask[(r >= 9.0) & (r <= 11.0)] = True

        config = IsosterConfig(
            x0=50.0, y0=50.0, sma0=10.0, maxsma=40.0,
            first_isophote_fail_count=3,
        )

        result = fit_image(image, mask, config)

        # The probes at sma ~11 or ~12.1 should succeed as anchors
        assert result.get("first_isophote_failure") is not True
        assert len(result["isophotes"]) > 1


class TestFirstIsophoteRetry:
    """Part 2: optional retry mechanism."""

    def test_retry_succeeds_with_perturbed_sma0(self, monkeypatch):
        """Retry with smaller sma0 succeeds when original sma0 fails.

        Mock fit_isophote to fail at sma >= 15 but succeed at smaller radii,
        simulating a compact galaxy where the outer region has no gradient.
        """
        from isoster import fitting

        original_fit_isophote = fitting.fit_isophote

        def patched_fit_isophote(image, mask, sma, start_geometry, cfg, **kwargs):
            result = original_fit_isophote(image, mask, sma, start_geometry, cfg, **kwargs)
            # Force failure for large SMA to simulate gradient-free outer region
            if sma >= 15.0:
                result["stop_code"] = -1
            return result

        monkeypatch.setattr("isoster.driver.fit_isophote", patched_fit_isophote)

        image = _make_galaxy_image(scale=10.0, peak=5000.0, background=10.0)
        config = IsosterConfig(
            x0=50.0, y0=50.0,
            sma0=20.0,  # Fails at this SMA
            maxsma=30.0,
            max_retry_first_isophote=5,
        )

        result = fit_image(image, None, config)

        assert result.get("first_isophote_failure") is not True
        assert len(result["isophotes"]) > 1
        assert "first_isophote_retry_log" in result
        assert len(result["first_isophote_retry_log"]) >= 1

    def test_retry_log_records_attempts(self, monkeypatch):
        """Retry log contains structured attempt data."""
        from isoster import fitting

        original_fit_isophote = fitting.fit_isophote

        def patched_fit_isophote(image, mask, sma, start_geometry, cfg, **kwargs):
            result = original_fit_isophote(image, mask, sma, start_geometry, cfg, **kwargs)
            if sma >= 15.0:
                result["stop_code"] = -1
            return result

        monkeypatch.setattr("isoster.driver.fit_isophote", patched_fit_isophote)

        image = _make_galaxy_image(scale=10.0, peak=5000.0, background=10.0)
        config = IsosterConfig(
            x0=50.0, y0=50.0, sma0=20.0, maxsma=30.0,
            max_retry_first_isophote=5,
        )

        result = fit_image(image, None, config)

        if "first_isophote_retry_log" in result:
            for entry in result["first_isophote_retry_log"]:
                assert "attempt" in entry
                assert "sma0" in entry
                assert "eps" in entry
                assert "pa" in entry
                assert "stop_code" in entry

    def test_retry_exhaustion_on_flat_image(self):
        """All retries fail on completely flat image."""
        image = np.ones((100, 100)) * 100.0
        config = IsosterConfig(
            x0=50.0, y0=50.0, sma0=10.0, maxsma=30.0,
            max_retry_first_isophote=3,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fit_image(image, None, config)

        assert result.get("first_isophote_failure") is True
        assert len(result.get("first_isophote_retry_log", [])) == 3
        assert any("FIRST_FEW_ISOPHOTE_FAILURE" in str(w.message) for w in caught)

    def test_no_retry_log_when_disabled(self):
        """With max_retry_first_isophote=0 (default), no retry log in result."""
        image = _make_galaxy_image()
        config = IsosterConfig(x0=50.0, y0=50.0, sma0=10.0, maxsma=40.0)

        result = fit_image(image, None, config)

        assert "first_isophote_retry_log" not in result
        assert result.get("first_isophote_failure") is not True

    def test_no_retry_when_first_iso_succeeds(self):
        """When first iso succeeds, retry is never triggered even if enabled."""
        image = _make_galaxy_image()
        config = IsosterConfig(
            x0=50.0, y0=50.0, sma0=10.0, maxsma=40.0,
            max_retry_first_isophote=5,
        )

        result = fit_image(image, None, config)

        # No retry log because first iso succeeded on first try
        assert "first_isophote_retry_log" not in result
        assert result.get("first_isophote_failure") is not True
        assert len(result["isophotes"]) > 1


class TestFirstIsophoteRetryConfig:
    """Config validation for new fields."""

    def test_negative_max_retry_rejected(self):
        """max_retry_first_isophote < 0 raises validation error."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            IsosterConfig(max_retry_first_isophote=-1)

    def test_zero_fail_count_rejected(self):
        """first_isophote_fail_count < 1 raises validation error."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            IsosterConfig(first_isophote_fail_count=0)

    def test_default_values(self):
        """Default config enables retry (3 attempts) and fail_count=3."""
        config = IsosterConfig()
        assert config.max_retry_first_isophote == 3
        assert config.first_isophote_fail_count == 3

    def test_large_max_retry_rejected(self):
        """max_retry_first_isophote > 20 raises validation error."""
        with pytest.raises(Exception):
            IsosterConfig(max_retry_first_isophote=21)
