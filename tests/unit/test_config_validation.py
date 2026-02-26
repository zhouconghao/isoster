"""Tests for config validation warnings and errors (Phase 23, V1-V11)."""

import warnings

import numpy as np
import pytest

from isoster.config import IsosterConfig


# ---------------------------------------------------------------------------
# V1: isofit_mode is a no-op without simultaneous_harmonics
# ---------------------------------------------------------------------------

def test_warn_isofit_mode_without_simultaneous():
    """V1: isofit_mode='original' should warn when simultaneous_harmonics=False."""
    with pytest.warns(UserWarning, match="isofit_mode has no effect"):
        IsosterConfig(isofit_mode='original', simultaneous_harmonics=False)


def test_no_warn_isofit_mode_with_simultaneous():
    """V1 negative: no warning when simultaneous_harmonics=True."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        IsosterConfig(isofit_mode='original', simultaneous_harmonics=True)


# ---------------------------------------------------------------------------
# V2: maxsma < sma0
# ---------------------------------------------------------------------------

def test_warn_maxsma_less_than_sma0():
    """V2: maxsma < sma0 should warn about limited output."""
    with pytest.warns(UserWarning, match="maxsma.*< sma0"):
        IsosterConfig(sma0=10.0, maxsma=5.0)


# ---------------------------------------------------------------------------
# V3: minsma >= sma0
# ---------------------------------------------------------------------------

def test_warn_minsma_ge_sma0():
    """V3: minsma >= sma0 should warn that inward loop won't run."""
    with pytest.warns(UserWarning, match="minsma.*>= sma0.*inward loop"):
        IsosterConfig(minsma=10.0, sma0=10.0)


# ---------------------------------------------------------------------------
# V4: simultaneous geometry mode with high damping
# ---------------------------------------------------------------------------

def test_warn_simultaneous_high_damping():
    """V4: geometry_update_mode='simultaneous' + damping > 0.7 should warn."""
    with pytest.warns(UserWarning, match="geometry_damping > 0.7.*simultaneous"):
        IsosterConfig(geometry_update_mode='simultaneous', geometry_damping=0.9)


def test_no_warn_simultaneous_low_damping():
    """V4 negative: no warning when damping <= 0.7."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        IsosterConfig(geometry_update_mode='simultaneous', geometry_damping=0.5)


# ---------------------------------------------------------------------------
# V5: forced=True silently drops params
# ---------------------------------------------------------------------------

def test_warn_forced_drops_params():
    """V5: forced=True with active features should warn about ignored params."""
    with pytest.warns(UserWarning, match="forced=True ignores"):
        IsosterConfig(
            forced=True, forced_sma=[5.0, 10.0],
            compute_deviations=True, compute_errors=True
        )


def test_no_warn_forced_defaults():
    """V5 negative: forced=True with default compute_errors=True should still warn."""
    with pytest.warns(UserWarning, match="forced=True ignores.*compute_errors"):
        IsosterConfig(forced=True, forced_sma=[5.0, 10.0])


# ---------------------------------------------------------------------------
# V6: template_isophotes + forced=True (integration test, needs driver)
# ---------------------------------------------------------------------------

def test_warn_template_and_forced():
    """V6: template_isophotes + forced=True should warn that forced is skipped."""
    from isoster.driver import fit_image

    image = np.random.default_rng(42).normal(100, 10, (64, 64))
    template = [
        {'sma': 0, 'x0': 32.0, 'y0': 32.0, 'eps': 0.2, 'pa': 0.0,
         'intens': 100.0},
        {'sma': 5.0, 'x0': 32.0, 'y0': 32.0, 'eps': 0.2, 'pa': 0.0,
         'intens': 90.0},
    ]
    config = IsosterConfig(forced=True, forced_sma=[5.0, 10.0])

    with pytest.warns(UserWarning, match="template_isophotes takes priority"):
        fit_image(image, config=config, template_isophotes=template)


# ---------------------------------------------------------------------------
# V7: invalid central_reg_weights keys
# ---------------------------------------------------------------------------

def test_error_bad_central_reg_keys():
    """V7: unknown keys in central_reg_weights should raise ValueError."""
    with pytest.raises(ValueError, match="central_reg_weights contains unknown keys"):
        IsosterConfig(central_reg_weights={'Eps': 1.0, 'pa': 1.0})


def test_valid_central_reg_keys():
    """V7 negative: valid keys should pass without error."""
    cfg = IsosterConfig(central_reg_weights={'eps': 2.0, 'pa': 0.5, 'center': 1.0})
    assert cfg.central_reg_weights['eps'] == 2.0


# ---------------------------------------------------------------------------
# V8: no harmonic keys when compute_deviations=False and
#     simultaneous_harmonics=False
# ---------------------------------------------------------------------------

def test_no_harmonic_keys_when_disabled():
    """V8: harmonic keys should not appear when both deviations and ISOFIT off."""
    from isoster.fitting import fit_isophote

    image = np.random.default_rng(42).normal(100, 10, (64, 64))
    config = IsosterConfig(
        compute_deviations=False,
        simultaneous_harmonics=False,
        maxit=5, minit=1,
    )
    geometry = {'x0': 32.0, 'y0': 32.0, 'eps': 0.2, 'pa': 0.0}
    result = fit_isophote(image, None, 10.0, geometry, config)

    # Should NOT have a3, b3, a4, b4 keys
    for key in ['a3', 'b3', 'a4', 'b4', 'a3_err', 'b3_err', 'a4_err', 'b4_err']:
        assert key not in result, f"unexpected harmonic key '{key}' in result"


def test_harmonic_keys_present_when_deviations_enabled():
    """V8 negative: harmonic keys present when compute_deviations=True."""
    from isoster.fitting import fit_isophote

    image = np.random.default_rng(42).normal(100, 10, (64, 64))
    config = IsosterConfig(
        compute_deviations=True,
        simultaneous_harmonics=False,
        maxit=5, minit=1,
    )
    geometry = {'x0': 32.0, 'y0': 32.0, 'eps': 0.2, 'pa': 0.0}
    result = fit_isophote(image, None, 10.0, geometry, config)

    for key in ['a3', 'b3', 'a4', 'b4']:
        assert key in result, f"expected harmonic key '{key}' in result"


# ---------------------------------------------------------------------------
# V9: template forced mode respects debug flag
# ---------------------------------------------------------------------------

def test_template_forced_respects_debug():
    """V9: template forced mode should pass debug flag to fit_central_pixel."""
    from isoster.driver import fit_image

    image = np.random.default_rng(42).normal(100, 10, (64, 64))
    template = [
        {'sma': 0, 'x0': 32.0, 'y0': 32.0, 'eps': 0.2, 'pa': 0.0,
         'intens': 100.0},
        {'sma': 5.0, 'x0': 32.0, 'y0': 32.0, 'eps': 0.2, 'pa': 0.0,
         'intens': 90.0},
    ]
    config = IsosterConfig(debug=True)
    result = fit_image(image, config=config, template_isophotes=template)

    # Central pixel (sma=0) should have debug fields when debug=True
    central = [iso for iso in result['isophotes'] if iso['sma'] == 0][0]
    assert 'ndata' in central, "debug=True should produce 'ndata' key in central pixel"


# ---------------------------------------------------------------------------
# V10: geometry convergence can never trigger
# ---------------------------------------------------------------------------

def test_warn_geometry_convergence_impossible():
    """V10: maxit < minit + geometry_stable_iters should warn."""
    with pytest.warns(UserWarning, match="geometry convergence can never trigger"):
        IsosterConfig(
            geometry_convergence=True,
            maxit=12, minit=10, geometry_stable_iters=5
        )


def test_no_warn_geometry_convergence_possible():
    """V10 negative: no warning when maxit is sufficient."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        IsosterConfig(
            geometry_convergence=True,
            maxit=50, minit=10, geometry_stable_iters=3
        )


# ---------------------------------------------------------------------------
# V11: lsb_sma_threshold with non-adaptive integrator
# ---------------------------------------------------------------------------

def test_warn_lsb_threshold_without_adaptive():
    """V11: lsb_sma_threshold with non-adaptive integrator should warn."""
    with pytest.warns(UserWarning, match="lsb_sma_threshold is set but integrator="):
        IsosterConfig(integrator='mean', lsb_sma_threshold=50.0)


def test_no_warn_lsb_threshold_with_adaptive():
    """V11 negative: no warning when integrator='adaptive'."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        IsosterConfig(integrator='adaptive', lsb_sma_threshold=50.0)


# ---------------------------------------------------------------------------
# R26-T3: invalid convergence_scaling values rejected by pattern
# ---------------------------------------------------------------------------

def test_error_invalid_convergence_scaling():
    """R26-T3: invalid convergence_scaling values should raise ValidationError."""
    from pydantic import ValidationError

    for bad_value in ['linear', 'SECTOR_AREA', 'sma', '']:
        with pytest.raises(ValidationError, match="convergence_scaling"):
            IsosterConfig(convergence_scaling=bad_value)


def test_valid_convergence_scaling_values():
    """R26-T3 negative: all valid convergence_scaling values accepted."""
    for good_value in ['none', 'sector_area', 'sqrt_sma']:
        cfg = IsosterConfig(convergence_scaling=good_value)
        assert cfg.convergence_scaling == good_value
