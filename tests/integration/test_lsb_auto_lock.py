"""
Tests for the Automatic LSB Geometry Lock feature.

Covers:
- Backward compatibility (lsb_auto_lock=False is untouched).
- No trigger on a clean, high-S/N galaxy profile.
- Trigger on a truncated-into-noise profile, with anchor before the streak.
- Locked geometry (x0, y0, eps, pa) exactly constant after transition.
- Inward growth unaffected by the lock.
- Config validation: lsb_auto_lock + fix_center conflict.
- debug=False auto-enable with UserWarning.
- Forced-photometry guard: the feature must warn and no-op when a template
  is provided.
"""

import warnings

import numpy as np
import pytest
from pydantic import ValidationError

from isoster.config import IsosterConfig
from isoster.driver import fit_image


def _make_exponential_galaxy(
    shape=(200, 200),
    center=None,
    scale=15.0,
    peak=5000.0,
    background=0.0,
    rng_seed=0,
    noise_sigma=0.0,
):
    """Simple exponential profile galaxy image with optional Gaussian noise."""
    h, w = shape
    if center is None:
        center = (w / 2.0, h / 2.0)
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    image = peak * np.exp(-r / scale) + background
    if noise_sigma > 0.0:
        rng = np.random.default_rng(rng_seed)
        image = image + rng.normal(0.0, noise_sigma, size=image.shape)
    return image


def _make_truncated_galaxy(
    shape=(200, 200),
    center=None,
    scale=10.0,
    peak=8000.0,
    truncate_r=40.0,
    background=100.0,
    noise_sigma=30.0,
    rng_seed=42,
):
    """Bright galaxy inside ``truncate_r`` that fades abruptly into noisy
    background outside, so outward growth slides into the LSB regime."""
    h, w = shape
    if center is None:
        center = (w / 2.0, h / 2.0)
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    galaxy = peak * np.exp(-r / scale)
    galaxy[r > truncate_r] = 0.0
    rng = np.random.default_rng(rng_seed)
    noise = rng.normal(background, noise_sigma, size=(h, w))
    return galaxy + noise


class TestBackwardCompat:
    def test_default_is_disabled(self):
        """Default cfg leaves hybrid mode off and adds no hybrid keys."""
        cfg = IsosterConfig(x0=100, y0=100, sma0=10.0, maxsma=60.0)
        assert cfg.lsb_auto_lock is False

        image = _make_exponential_galaxy()
        result = fit_image(image, None, cfg)

        assert "lsb_auto_lock" not in result
        assert "lsb_auto_lock_sma" not in result
        assert "lsb_auto_lock_count" not in result
        for iso in result["isophotes"]:
            assert "lsb_locked" not in iso
            assert "lsb_auto_lock_anchor" not in iso


class TestNoTriggerOnCleanProfile:
    def test_clean_galaxy_never_triggers(self):
        """High-S/N exponential galaxy fits should stay in free mode."""
        image = _make_exponential_galaxy(scale=20.0, peak=10000.0, noise_sigma=0.0)
        cfg = IsosterConfig(
            x0=100.0,
            y0=100.0,
            sma0=10.0,
            maxsma=50.0,
            astep=0.2,
            lsb_auto_lock=True,
            debug=True,
        )
        result = fit_image(image, None, cfg)

        assert result["lsb_auto_lock"] is True
        assert result["lsb_auto_lock_sma"] is None
        assert result["lsb_auto_lock_count"] == 0
        assert all(iso.get("lsb_locked", False) is False for iso in result["isophotes"])


class TestTriggerOnTruncatedProfile:
    def _run(self):
        image = _make_truncated_galaxy()
        cfg = IsosterConfig(
            x0=100.0,
            y0=100.0,
            sma0=8.0,
            maxsma=90.0,
            astep=0.15,
            lsb_auto_lock=True,
            lsb_auto_lock_debounce=2,
            lsb_auto_lock_maxgerr=0.3,
            debug=True,
        )
        return fit_image(image, None, cfg)

    def test_transition_sma_is_finite(self):
        result = self._run()
        assert result["lsb_auto_lock"] is True
        transition = result["lsb_auto_lock_sma"]
        assert transition is not None
        assert np.isfinite(transition)
        assert result["lsb_auto_lock_count"] >= 1

    def test_lsb_auto_lock_anchor_marker_present(self):
        result = self._run()
        marked = [iso for iso in result["isophotes"] if iso.get("lsb_auto_lock_anchor") is True]
        assert len(marked) == 1

    def test_all_locked_isos_flagged(self):
        result = self._run()
        locked_count = sum(1 for iso in result["isophotes"] if iso.get("lsb_locked", False))
        assert locked_count == result["lsb_auto_lock_count"]
        assert locked_count >= 1


class TestLockedGeometryConstant:
    def test_locked_geometry_is_exactly_constant(self):
        image = _make_truncated_galaxy()
        cfg = IsosterConfig(
            x0=100.0,
            y0=100.0,
            sma0=8.0,
            maxsma=90.0,
            astep=0.15,
            lsb_auto_lock=True,
            lsb_auto_lock_debounce=2,
            debug=True,
        )
        result = fit_image(image, None, cfg)
        if result["lsb_auto_lock_sma"] is None:
            pytest.skip("lock did not commit in this synthetic run")

        locked_isos = [iso for iso in result["isophotes"] if iso.get("lsb_locked")]
        assert len(locked_isos) >= 1
        x0s = np.array([iso["x0"] for iso in locked_isos])
        y0s = np.array([iso["y0"] for iso in locked_isos])
        eps = np.array([iso["eps"] for iso in locked_isos])
        pas = np.array([iso["pa"] for iso in locked_isos])
        assert np.all(x0s == x0s[0])
        assert np.all(y0s == y0s[0])
        assert np.all(eps == eps[0])
        assert np.all(pas == pas[0])


class TestInwardGrowthUnaffected:
    def test_inward_isophotes_do_not_carry_hybrid_keys(self):
        image = _make_exponential_galaxy(scale=20.0, peak=10000.0)
        cfg = IsosterConfig(
            x0=100.0,
            y0=100.0,
            sma0=15.0,
            minsma=0.0,
            maxsma=50.0,
            astep=0.2,
            lsb_auto_lock=True,
            debug=True,
        )
        result = fit_image(image, None, cfg)

        # Find isophotes strictly inside sma0 (the inward grown ones). The
        # anchor isophote at sma0 is marked lsb_locked=False; further-in
        # isophotes should not carry the key at all, because the inward
        # loop does not participate in hybrid mode.
        inward_isos = [iso for iso in result["isophotes"] if 0.0 < iso["sma"] < 15.0]
        assert len(inward_isos) >= 1
        for iso in inward_isos:
            assert "lsb_locked" not in iso
            assert "lsb_auto_lock_anchor" not in iso


class TestConfigValidation:
    def test_hybrid_with_fix_center_raises(self):
        with pytest.raises(ValidationError, match="lsb_auto_lock"):
            IsosterConfig(lsb_auto_lock=True, fix_center=True, debug=True)

    def test_hybrid_with_fix_pa_raises(self):
        with pytest.raises(ValidationError, match="lsb_auto_lock"):
            IsosterConfig(lsb_auto_lock=True, fix_pa=True, debug=True)

    def test_hybrid_with_fix_eps_raises(self):
        with pytest.raises(ValidationError, match="lsb_auto_lock"):
            IsosterConfig(lsb_auto_lock=True, fix_eps=True, debug=True)


class TestDebugAutoEnable:
    def test_debug_false_emits_warning_and_runs(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = IsosterConfig(
                x0=100.0,
                y0=100.0,
                sma0=10.0,
                maxsma=50.0,
                astep=0.2,
                lsb_auto_lock=True,
                debug=False,
            )
        assert any("lsb_auto_lock" in str(w.message) and "debug" in str(w.message) for w in caught)

        image = _make_exponential_galaxy(scale=20.0, peak=10000.0)
        result = fit_image(image, None, cfg)
        # The fit should still run and auto-lock metadata should be present.
        assert result["lsb_auto_lock"] is True
        # Outward isophotes should carry the lsb_locked key even though
        # the caller did not set debug=True themselves.
        outward = [iso for iso in result["isophotes"] if iso["sma"] >= 10.0]
        assert any("lsb_locked" in iso for iso in outward)


class TestForcedPhotometryGuard:
    """lsb_auto_lock must emit a UserWarning when combined with template-based
    forced photometry, because the feature is not wired into the forced path."""

    def test_forced_photometry_warns_and_runs(self):
        image = _make_exponential_galaxy(scale=20.0, peak=10000.0)

        template_cfg = IsosterConfig(x0=100.0, y0=100.0, sma0=10.0, maxsma=40.0, astep=0.25)
        template_result = fit_image(image, None, template_cfg)
        template = template_result["isophotes"]

        cfg = IsosterConfig(
            x0=100.0,
            y0=100.0,
            sma0=10.0,
            maxsma=40.0,
            astep=0.25,
            lsb_auto_lock=True,
            debug=True,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fit_image(image, None, cfg, template=template)

        # The feature should be silently inactive in the forced path, but a
        # UserWarning must have been emitted.
        assert any("lsb_auto_lock" in str(w.message) and "forced photometry" in str(w.message) for w in caught), (
            "expected a forced-photometry guard warning for lsb_auto_lock"
        )
        assert "lsb_auto_lock" not in result
        assert "lsb_auto_lock_sma" not in result
