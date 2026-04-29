"""Tests for ``isoster.multiband.config_mb.IsosterConfigMB``."""

import warnings

import pytest
from pydantic import ValidationError

from isoster.multiband import IsosterConfigMB


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_minimal_construction():
    """A two-band config with required fields builds and exposes defaults."""
    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g")
    assert cfg.bands == ["g", "r"]
    assert cfg.reference_band == "g"
    assert cfg.harmonic_combination == "joint"
    assert cfg.band_weights is None
    assert cfg.resolved_band_weights() == {"g": 1.0, "r": 1.0}
    assert cfg.integrator == "mean"
    assert cfg.variance_mode is None


def test_resolved_band_weights_dict_form():
    cfg = IsosterConfigMB(
        bands=["g", "r", "i"],
        reference_band="i",
        band_weights={"g": 0.5, "r": 1.0, "i": 2.0},
    )
    assert cfg.resolved_band_weights() == {"g": 0.5, "r": 1.0, "i": 2.0}


def test_resolved_band_weights_list_form():
    cfg = IsosterConfigMB(
        bands=["g", "r", "i"],
        reference_band="i",
        band_weights=[0.5, 1.0, 2.0],
    )
    assert cfg.resolved_band_weights() == {"g": 0.5, "r": 1.0, "i": 2.0}


def test_underscore_band_names_accepted():
    """``HSC_G`` style names with underscores are accepted verbatim."""
    cfg = IsosterConfigMB(bands=["HSC_G", "HSC_R"], reference_band="HSC_G")
    assert cfg.bands == ["HSC_G", "HSC_R"]


def test_harmonic_combination_ref():
    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="r", harmonic_combination="ref")
    assert cfg.harmonic_combination == "ref"


# ---------------------------------------------------------------------------
# Band-name validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_band",
    [
        "HSC-G",   # hyphen rejected
        "1g",      # leading digit rejected
        "g r",     # whitespace rejected
        "g.r",     # dot rejected
        "",        # empty rejected
        "g+",      # special char rejected
    ],
)
def test_band_name_regex_rejected(bad_band):
    with pytest.raises(ValidationError) as exc_info:
        IsosterConfigMB(bands=[bad_band], reference_band=bad_band)
    assert "regex" in str(exc_info.value).lower() or "match" in str(exc_info.value).lower() or bad_band in str(
        exc_info.value
    )


def test_band_duplicates_rejected():
    with pytest.raises(ValidationError) as exc_info:
        IsosterConfigMB(bands=["g", "g"], reference_band="g")
    assert "duplicate" in str(exc_info.value).lower()


def test_empty_bands_rejected():
    with pytest.raises(ValidationError):
        IsosterConfigMB(bands=[], reference_band="g")


# ---------------------------------------------------------------------------
# reference_band membership
# ---------------------------------------------------------------------------


def test_reference_band_must_be_in_bands():
    with pytest.raises(ValidationError) as exc_info:
        IsosterConfigMB(bands=["g", "r"], reference_band="i")
    msg = str(exc_info.value)
    assert "reference_band" in msg
    assert "'i'" in msg or "i" in msg


# ---------------------------------------------------------------------------
# band_weights validation
# ---------------------------------------------------------------------------


def test_band_weights_dict_missing_key_rejected():
    with pytest.raises(ValidationError) as exc_info:
        IsosterConfigMB(
            bands=["g", "r", "i"],
            reference_band="g",
            band_weights={"g": 1.0, "r": 1.0},  # missing 'i'
        )
    assert "missing keys" in str(exc_info.value)


def test_band_weights_dict_extra_key_rejected():
    with pytest.raises(ValidationError) as exc_info:
        IsosterConfigMB(
            bands=["g", "r"],
            reference_band="g",
            band_weights={"g": 1.0, "r": 1.0, "z": 1.0},
        )
    assert "not in `bands`" in str(exc_info.value)


def test_band_weights_list_wrong_length_rejected():
    with pytest.raises(ValidationError) as exc_info:
        IsosterConfigMB(
            bands=["g", "r", "i"],
            reference_band="g",
            band_weights=[1.0, 1.0],
        )
    assert "length" in str(exc_info.value)


@pytest.mark.parametrize("bad_w", [0.0, -1.0, float("inf"), float("nan")])
def test_band_weights_must_be_positive_finite(bad_w):
    with pytest.raises(ValidationError):
        IsosterConfigMB(
            bands=["g", "r"],
            reference_band="g",
            band_weights={"g": 1.0, "r": bad_w},
        )


# ---------------------------------------------------------------------------
# Integrator restriction
# ---------------------------------------------------------------------------


def test_integrator_adaptive_rejected():
    with pytest.raises(ValidationError):
        IsosterConfigMB(bands=["g"], reference_band="g", integrator="adaptive")


def test_integrator_mean_and_median_accepted():
    for integ in ("mean", "median"):
        cfg = IsosterConfigMB(bands=["g"], reference_band="g", integrator=integ)
        assert cfg.integrator == integ


# ---------------------------------------------------------------------------
# SMA / iteration consistency (copied from single-band semantics)
# ---------------------------------------------------------------------------


def test_maxsma_below_minsma_rejected():
    with pytest.raises(ValidationError) as exc_info:
        IsosterConfigMB(
            bands=["g"], reference_band="g", minsma=10.0, maxsma=5.0,
        )
    assert "maxsma" in str(exc_info.value)


def test_minit_above_maxit_rejected():
    with pytest.raises(ValidationError) as exc_info:
        IsosterConfigMB(bands=["g"], reference_band="g", minit=10, maxit=5)
    assert "minit" in str(exc_info.value)


def test_minsma_above_sma0_emits_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        IsosterConfigMB(bands=["g"], reference_band="g", sma0=5.0, minsma=10.0)
    assert any("inward loop will not run" in str(item.message) for item in w)


def test_maxsma_below_sma0_emits_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        IsosterConfigMB(bands=["g"], reference_band="g", sma0=10.0, maxsma=5.0)
    assert any("only one isophote" in str(item.message) for item in w)
