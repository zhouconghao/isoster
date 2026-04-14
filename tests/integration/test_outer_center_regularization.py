"""
Tests for the outer-region center regularization feature.

Covers:
- Backward compatibility (use_outer_center_regularization=False is untouched,
  including the inward-first loop reorder that is always active).
- Clean exponential galaxy is not over-constrained at any strength value.
- Contaminated galaxy has reduced pre-reference center drift with the feature on.
- Inner reference fallback when no inward isophotes exist.
- Config validation (warnings for onset < sma0, minsma >= sma0, and
  fix_center combined with the feature).
- LSB auto-lock still fires when both features are active.
- Forced-photometry guard: the feature must warn and no-op when a template
  is provided.
"""

import warnings

import numpy as np
import pytest

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
    """Simple exponential profile galaxy image with optional Gaussian noise.

    Duplicated from test_lsb_auto_lock.py because tests/integration has no
    __init__.py and thus does not support intra-package imports.
    """
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
    background outside, so outward growth slides into the LSB regime.

    Duplicated from test_lsb_auto_lock.py (see _make_exponential_galaxy note).
    """
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


def _make_contaminated_truncated_galaxy(
    shape=(200, 200),
    center=None,
    scale=10.0,
    peak=8000.0,
    truncate_r=40.0,
    background=100.0,
    noise_sigma=30.0,
    contaminant_center=(150.0, 100.0),
    contaminant_flux=3000.0,
    contaminant_scale=8.0,
    rng_seed=123,
):
    """Truncated galaxy with a bright off-center point source as contaminant.

    The point source is placed well outside truncate_r so it only affects the
    LSB outer isophotes once the fit starts to drift.
    """
    h, w = shape
    if center is None:
        center = (w / 2.0, h / 2.0)
    y, x = np.ogrid[:h, :w]
    r_gal = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    galaxy = peak * np.exp(-r_gal / scale)
    galaxy[r_gal > truncate_r] = 0.0

    r_cont = np.sqrt((x - contaminant_center[0]) ** 2 + (y - contaminant_center[1]) ** 2)
    contaminant = contaminant_flux * np.exp(-r_cont / contaminant_scale)

    rng = np.random.default_rng(rng_seed)
    noise = rng.normal(background, noise_sigma, size=(h, w))
    return galaxy + contaminant + noise


def _pre_lock_outward_isophotes(result):
    """Return outward isophotes excluding any hybrid-lock-frozen tail."""
    sma0 = result["config"].sma0
    out = []
    for iso in result["isophotes"]:
        if iso.get("sma", 0.0) < sma0:
            continue
        if iso.get("lsb_locked", False):
            continue
        out.append(iso)
    return out


def _combined_drift(isophotes, x0_ref, y0_ref):
    if not isophotes:
        return 0.0, 0.0, 0.0
    dx = np.array([iso["x0"] - x0_ref for iso in isophotes])
    dy = np.array([iso["y0"] - y0_ref for iso in isophotes])
    return float(np.max(np.abs(dx))), float(np.max(np.abs(dy))), float(np.sqrt(np.max(np.abs(dx)) ** 2 + np.max(np.abs(dy)) ** 2))


class TestBackwardCompatibility:
    """With the feature off, nothing should observably change. This also
    catches regressions from the inward-first loop reorder."""

    def test_default_is_disabled(self):
        cfg = IsosterConfig(x0=100, y0=100, sma0=10.0, maxsma=60.0)
        assert cfg.use_outer_center_regularization is False

        image = _make_exponential_galaxy()
        result = fit_image(image, None, cfg)

        assert "use_outer_center_regularization" not in result
        assert "outer_reg_x0_ref" not in result
        assert "outer_reg_y0_ref" not in result

    def test_inward_first_reorder_preserves_inward_results(self):
        """The inward loop's outputs must match between old and new driver
        order. Since the old order is gone, we check that the inward isophotes
        are independent of the feature flag (feature off vs on).
        """
        image = _make_exponential_galaxy(scale=20.0, peak=10000.0, noise_sigma=0.0)
        base = dict(x0=100.0, y0=100.0, sma0=10.0, minsma=2.0, maxsma=80.0, astep=0.2)

        result_off = fit_image(image, None, IsosterConfig(**base))
        result_on = fit_image(
            image,
            None,
            IsosterConfig(
                **base,
                use_outer_center_regularization=True,
                outer_reg_strength=0.0,  # strength=0 => penalty is zero
            ),
        )

        inward_off = [iso for iso in result_off["isophotes"] if iso["sma"] < 10.0]
        inward_on = [iso for iso in result_on["isophotes"] if iso["sma"] < 10.0]
        assert len(inward_off) == len(inward_on)
        for a, b in zip(inward_off, inward_on):
            assert a["x0"] == pytest.approx(b["x0"])
            assert a["y0"] == pytest.approx(b["y0"])
            assert a["eps"] == pytest.approx(b["eps"])


class TestCleanGalaxyNoOverConstraint:
    """The feature must not introduce artificial drift on a clean galaxy."""

    @pytest.mark.parametrize("strength", [0.5, 2.0, 8.0])
    def test_clean_galaxy_center_stays_stable(self, strength):
        image = _make_exponential_galaxy(scale=20.0, peak=10000.0, noise_sigma=0.0)
        cfg = IsosterConfig(
            x0=100.0,
            y0=100.0,
            sma0=10.0,
            minsma=2.0,
            maxsma=60.0,
            astep=0.25,
            use_outer_center_regularization=True,
            outer_reg_sma_onset=25.0,
            outer_reg_sma_width=8.0,
            outer_reg_strength=strength,
        )
        result = fit_image(image, None, cfg)

        assert result["use_outer_center_regularization"] is True
        x0_ref = result["outer_reg_x0_ref"]
        y0_ref = result["outer_reg_y0_ref"]

        # Reference should sit near the true center (within 0.5 px).
        assert abs(x0_ref - 100.0) < 0.5
        assert abs(y0_ref - 100.0) < 0.5

        # Every outward isophote should sit within 1 px of the reference —
        # a clean noiseless galaxy should never drift.
        outward = _pre_lock_outward_isophotes(result)
        assert len(outward) > 0
        max_dx = max(abs(iso["x0"] - x0_ref) for iso in outward)
        max_dy = max(abs(iso["y0"] - y0_ref) for iso in outward)
        assert max_dx < 1.0, f"clean galaxy drifted {max_dx} px in x at strength={strength}"
        assert max_dy < 1.0, f"clean galaxy drifted {max_dy} px in y at strength={strength}"


class TestContaminatedGalaxyDriftReduced:
    """Pre-reference center drift should drop with the feature on."""

    def test_contaminated_galaxy_drift_decreases(self):
        image = _make_contaminated_truncated_galaxy()
        base = dict(
            x0=100.0,
            y0=100.0,
            sma0=8.0,
            minsma=2.0,
            maxsma=90.0,
            astep=0.15,
        )

        cfg_off = IsosterConfig(**base)
        cfg_on = IsosterConfig(
            **base,
            use_outer_center_regularization=True,
            outer_reg_sma_onset=30.0,
            outer_reg_sma_width=8.0,
            outer_reg_strength=5.0,
        )

        result_off = fit_image(image, None, cfg_off)
        result_on = fit_image(image, None, cfg_on)

        # Reference used for both metrics: the feature-on reference centroid
        # (this is the "ground truth" inner center either driver builds).
        x0_ref = result_on["outer_reg_x0_ref"]
        y0_ref = result_on["outer_reg_y0_ref"]

        out_off = _pre_lock_outward_isophotes(result_off)
        out_on = _pre_lock_outward_isophotes(result_on)

        _, _, drift_off = _combined_drift(out_off, x0_ref, y0_ref)
        _, _, drift_on = _combined_drift(out_on, x0_ref, y0_ref)

        assert drift_on < drift_off, (
            f"Feature-on drift {drift_on:.2f} px should be less than "
            f"feature-off drift {drift_off:.2f} px on contaminated galaxy"
        )


class TestInnerReferenceFallback:
    """When there are no inward isophotes, the reference must fall back to
    the anchor isophote center without error."""

    def test_fallback_to_anchor_when_no_inward(self):
        image = _make_exponential_galaxy(scale=20.0, peak=10000.0, noise_sigma=0.0)
        cfg = IsosterConfig(
            x0=100.0,
            y0=100.0,
            sma0=10.0,
            minsma=10.0,  # no inward growth
            maxsma=40.0,
            astep=0.25,
            use_outer_center_regularization=True,
            outer_reg_strength=1.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fit_image(image, None, cfg)

        assert result["use_outer_center_regularization"] is True
        anchor = next(iso for iso in result["isophotes"] if iso["sma"] == 10.0)
        assert result["outer_reg_x0_ref"] == pytest.approx(anchor["x0"])
        assert result["outer_reg_y0_ref"] == pytest.approx(anchor["y0"])


class TestConfigValidation:
    """Validator warnings for the outer-center-regularization fields."""

    def test_onset_below_sma0_warns(self):
        with pytest.warns(UserWarning, match="outer_reg_sma_onset"):
            IsosterConfig(
                x0=100,
                y0=100,
                sma0=50.0,
                maxsma=200.0,
                use_outer_center_regularization=True,
                outer_reg_sma_onset=10.0,
            )

    def test_minsma_ge_sma0_warns(self):
        with pytest.warns(UserWarning, match="inward isophotes to build the reference"):
            IsosterConfig(
                x0=100,
                y0=100,
                sma0=10.0,
                minsma=10.0,
                maxsma=60.0,
                use_outer_center_regularization=True,
                outer_reg_sma_onset=30.0,
            )

    def test_fix_center_warns(self):
        with pytest.warns(UserWarning, match="use_outer_center_regularization"):
            IsosterConfig(
                x0=100,
                y0=100,
                sma0=10.0,
                maxsma=60.0,
                use_outer_center_regularization=True,
                fix_center=True,
            )


class TestLsbAutoLockStillFires:
    """With both use_outer_center_regularization and lsb_auto_lock enabled,
    the auto-lock must still commit at roughly the same sma on the existing
    truncated fixture."""

    def test_lock_still_fires_with_outer_reg(self):
        image = _make_truncated_galaxy()
        base = dict(
            x0=100.0,
            y0=100.0,
            sma0=8.0,
            minsma=2.0,
            maxsma=90.0,
            astep=0.15,
            lsb_auto_lock=True,
            lsb_auto_lock_debounce=2,
            lsb_auto_lock_maxgerr=0.3,
            debug=True,
        )
        result_lock = fit_image(image, None, IsosterConfig(**base))
        result_both = fit_image(
            image,
            None,
            IsosterConfig(
                **base,
                use_outer_center_regularization=True,
                outer_reg_sma_onset=20.0,
                outer_reg_sma_width=6.0,
                outer_reg_strength=2.0,
            ),
        )

        assert result_lock["lsb_auto_lock_sma"] is not None, "baseline lock should fire"
        assert result_both["lsb_auto_lock_sma"] is not None, "lock should still fire with outer_reg on"

        sma_lock = result_lock["lsb_auto_lock_sma"]
        sma_both = result_both["lsb_auto_lock_sma"]

        # Allow up to 2 growth steps difference.
        ratio = sma_both / sma_lock
        assert 1.0 / (1.15 ** 2) <= ratio <= (1.15 ** 2), (
            f"lock sma shifted too far: baseline={sma_lock:.1f}, with outer_reg={sma_both:.1f}"
        )


class TestForcedPhotometryGuard:
    """use_outer_center_regularization must warn and no-op in forced mode."""

    def test_forced_photometry_warns_and_runs(self):
        image = _make_exponential_galaxy(scale=20.0, peak=10000.0)

        template_cfg = IsosterConfig(
            x0=100.0, y0=100.0, sma0=10.0, maxsma=40.0, astep=0.25
        )
        template_result = fit_image(image, None, template_cfg)
        template = template_result["isophotes"]

        cfg = IsosterConfig(
            x0=100.0,
            y0=100.0,
            sma0=10.0,
            maxsma=40.0,
            astep=0.25,
            use_outer_center_regularization=True,
            outer_reg_strength=2.0,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fit_image(image, None, cfg, template=template)

        assert any(
            "use_outer_center_regularization" in str(w.message)
            and "forced photometry" in str(w.message)
            for w in caught
        ), "expected a forced-photometry guard warning for outer_center_regularization"
        assert "use_outer_center_regularization" not in result
        assert "outer_reg_x0_ref" not in result
        assert "outer_reg_y0_ref" not in result
