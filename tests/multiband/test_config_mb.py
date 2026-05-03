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


# ---------------------------------------------------------------------------
# D9 backport — loose validity
# ---------------------------------------------------------------------------


def test_loose_validity_defaults_off_with_neutral_thresholds():
    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g")
    assert cfg.loose_validity is False
    assert cfg.loose_validity_min_per_band_count == 6
    assert cfg.loose_validity_min_per_band_frac == 0.2
    assert cfg.loose_validity_band_normalization == "none"


def test_loose_validity_band_normalization_requires_loose_validity():
    """`per_band_count` is meaningless under shared validity (N_b all equal)."""
    with pytest.raises(ValidationError) as exc_info:
        IsosterConfigMB(
            bands=["g", "r"], reference_band="g",
            loose_validity=False,
            loose_validity_band_normalization="per_band_count",
        )
    assert "loose_validity=True" in str(exc_info.value)


def test_loose_validity_band_normalization_accepted_with_loose_validity():
    cfg = IsosterConfigMB(
        bands=["g", "r"], reference_band="g",
        loose_validity=True,
        loose_validity_band_normalization="per_band_count",
    )
    assert cfg.loose_validity is True
    assert cfg.loose_validity_band_normalization == "per_band_count"


def test_loose_validity_compatible_with_ring_mean_intercept():
    cfg = IsosterConfigMB(
        bands=["g", "r"], reference_band="g",
        loose_validity=True,
        fit_per_band_intens_jointly=False,
    )
    assert cfg.loose_validity
    assert cfg.fit_per_band_intens_jointly is False


def test_loose_validity_compatible_with_ref_mode():
    cfg = IsosterConfigMB(
        bands=["g", "r"], reference_band="g",
        loose_validity=True,
        harmonic_combination="ref",
    )
    assert cfg.loose_validity and cfg.harmonic_combination == "ref"


# ---------------------------------------------------------------------------
# Section 6: multiband_higher_harmonics enum + harmonic_orders
# ---------------------------------------------------------------------------


def test_higher_harmonics_default_independent():
    """Default value reproduces Stage-1 behavior."""
    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g")
    assert cfg.multiband_higher_harmonics == "independent"
    assert cfg.harmonic_orders == [3, 4]


@pytest.mark.parametrize(
    "value",
    ["independent", "shared", "simultaneous_in_loop", "simultaneous_original"],
)
def test_higher_harmonics_enum_values(value):
    """All four enum values construct successfully."""
    with warnings.catch_warnings():
        # simultaneous_* emit an experimental UserWarning; not relevant here.
        warnings.simplefilter("ignore", UserWarning)
        cfg = IsosterConfigMB(
            bands=["g", "r"], reference_band="g",
            multiband_higher_harmonics=value,
        )
    assert cfg.multiband_higher_harmonics == value


def test_higher_harmonics_invalid_value_rejected():
    with pytest.raises(ValidationError):
        IsosterConfigMB(
            bands=["g", "r"], reference_band="g",
            multiband_higher_harmonics="bogus",
        )


@pytest.mark.parametrize("value", ["shared", "simultaneous_in_loop", "simultaneous_original"])
def test_higher_harmonics_ref_mode_incompatible(value):
    """All non-independent modes hard-error with harmonic_combination='ref'."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ValidationError) as exc_info:
            IsosterConfigMB(
                bands=["g", "r"], reference_band="g",
                multiband_higher_harmonics=value,
                harmonic_combination="ref",
            )
    assert "ref" in str(exc_info.value)
    assert "incompatible" in str(exc_info.value)


def test_higher_harmonics_independent_compatible_with_ref():
    """Default 'independent' mode does NOT clash with ref-mode."""
    cfg = IsosterConfigMB(
        bands=["g", "r"], reference_band="g",
        multiband_higher_harmonics="independent",
        harmonic_combination="ref",
    )
    assert cfg.multiband_higher_harmonics == "independent"


@pytest.mark.parametrize("value", ["simultaneous_in_loop", "simultaneous_original"])
def test_higher_harmonics_simultaneous_warns(value):
    """simultaneous_* modes emit a UserWarning at construction."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        IsosterConfigMB(
            bands=["g", "r"], reference_band="g",
            multiband_higher_harmonics=value,
        )
    msgs = [str(w.message) for w in captured if issubclass(w.category, UserWarning)]
    assert any("experimental" in m and value in m for m in msgs)


def test_higher_harmonics_shared_does_not_warn():
    """shared mode is the recommended new feature; no experimental warning."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        IsosterConfigMB(
            bands=["g", "r"], reference_band="g",
            multiband_higher_harmonics="shared",
        )
    msgs = [str(w.message) for w in captured if issubclass(w.category, UserWarning)]
    assert not any("experimental" in m for m in msgs)


def test_harmonic_orders_default():
    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g")
    assert cfg.harmonic_orders == [3, 4]


def test_harmonic_orders_custom_list():
    cfg = IsosterConfigMB(
        bands=["g", "r"], reference_band="g",
        harmonic_orders=[3, 4, 5, 6],
    )
    assert cfg.harmonic_orders == [3, 4, 5, 6]


def test_harmonic_orders_unique_sorted():
    """Out-of-order input is unique-sorted on construction."""
    cfg = IsosterConfigMB(
        bands=["g", "r"], reference_band="g",
        harmonic_orders=[5, 3, 4],
    )
    assert cfg.harmonic_orders == [3, 4, 5]


def test_harmonic_orders_empty_rejected():
    with pytest.raises(ValidationError):
        IsosterConfigMB(
            bands=["g", "r"], reference_band="g",
            harmonic_orders=[],
        )


@pytest.mark.parametrize("bad", [[1, 2], [2], [0, 3], [-1, 3]])
def test_harmonic_orders_below_three_rejected(bad):
    with pytest.raises(ValidationError):
        IsosterConfigMB(
            bands=["g", "r"], reference_band="g",
            harmonic_orders=bad,
        )


def test_harmonic_orders_duplicates_rejected():
    with pytest.raises(ValidationError) as exc_info:
        IsosterConfigMB(
            bands=["g", "r"], reference_band="g",
            harmonic_orders=[3, 3, 4],
        )
    assert "duplicate" in str(exc_info.value).lower()


@pytest.mark.parametrize(
    "value",
    ["shared", "simultaneous_in_loop", "simultaneous_original"],
)
def test_higher_harmonics_compatible_with_ring_mean_intercept(value):
    """All non-independent modes silently allow fit_per_band_intens_jointly=False."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cfg = IsosterConfigMB(
            bands=["g", "r"], reference_band="g",
            multiband_higher_harmonics=value,
            fit_per_band_intens_jointly=False,
        )
    assert cfg.multiband_higher_harmonics == value
    assert cfg.fit_per_band_intens_jointly is False


@pytest.mark.parametrize(
    "value",
    ["shared", "simultaneous_in_loop", "simultaneous_original"],
)
def test_higher_harmonics_compatible_with_loose_validity(value):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cfg = IsosterConfigMB(
            bands=["g", "r"], reference_band="g",
            multiband_higher_harmonics=value,
            loose_validity=True,
        )
    assert cfg.multiband_higher_harmonics == value
    assert cfg.loose_validity is True


# ---------------------------------------------------------------------------
# Stage-3 Stage-B: outer-region regularization (damping mode)
# ---------------------------------------------------------------------------


def test_outer_reg_default_off_with_neutral_fields():
    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g")
    assert cfg.use_outer_center_regularization is False
    assert cfg.outer_reg_mode == "damping"
    assert cfg.outer_reg_sma_onset == 50.0
    assert cfg.outer_reg_strength == 2.0
    assert cfg.outer_reg_weights == {"center": 1.0, "eps": 1.0, "pa": 1.0}
    assert cfg.outer_reg_sma_width is None
    assert cfg.outer_reg_ref_sma_factor == 2.0


def test_outer_reg_solver_value_rejected_until_stage_e():
    """Stage B ships ``damping`` only; ``solver`` must fail Literal check."""
    with pytest.raises(ValidationError):
        IsosterConfigMB(
            bands=["g", "r"], reference_band="g",
            use_outer_center_regularization=True,
            outer_reg_mode="solver",  # type: ignore[arg-type]
        )


def test_outer_reg_unknown_axis_key_rejected():
    with pytest.raises(ValidationError) as exc_info:
        IsosterConfigMB(
            bands=["g", "r"], reference_band="g",
            use_outer_center_regularization=True,
            outer_reg_weights={"center": 1.0, "spin": 1.0},
        )
    assert "spin" in str(exc_info.value)


def test_outer_reg_onset_below_sma0_emits_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        IsosterConfigMB(
            bands=["g", "r"], reference_band="g",
            use_outer_center_regularization=True,
            sma0=20.0, outer_reg_sma_onset=10.0,
        )
    msgs = [str(item.message) for item in w]
    assert any(
        "outer_reg_sma_onset" in m and "sma0" in m for m in msgs
    ), msgs


def test_outer_reg_minsma_above_sma0_emits_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        IsosterConfigMB(
            bands=["g", "r"], reference_band="g",
            use_outer_center_regularization=True,
            sma0=10.0, minsma=20.0,
        )
    msgs = [str(item.message) for item in w]
    assert any("inward isophotes" in m for m in msgs), msgs


def test_outer_reg_all_zero_weights_emits_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        IsosterConfigMB(
            bands=["g", "r"], reference_band="g",
            use_outer_center_regularization=True,
            outer_reg_weights={"center": 0.0, "eps": 0.0, "pa": 0.0},
        )
    msgs = [str(item.message) for item in w]
    assert any("identically zero" in m for m in msgs), msgs


def test_outer_reg_auto_enables_geometry_convergence():
    """Auto-enable: feature on with default geometry_convergence=False
    flips the field to True after the warning fires."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = IsosterConfigMB(
            bands=["g", "r"], reference_band="g",
            use_outer_center_regularization=True,
        )
    msgs = [str(item.message) for item in w]
    assert any("auto-enables" in m and "geometry_convergence" in m for m in msgs)
    assert cfg.geometry_convergence is True


def test_outer_reg_no_geom_conv_warning_when_already_enabled():
    """If user already set geometry_convergence=True, no auto-enable warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = IsosterConfigMB(
            bands=["g", "r"], reference_band="g",
            use_outer_center_regularization=True,
            geometry_convergence=True,
        )
    msgs = [str(item.message) for item in w]
    assert not any("auto-enables" in m for m in msgs), msgs
    assert cfg.geometry_convergence is True


@pytest.mark.parametrize(
    "fixed_field, axis",
    [("fix_center", "center"), ("fix_eps", "eps"), ("fix_pa", "pa")],
)
def test_outer_reg_fix_axis_warning(fixed_field, axis):
    """fix_<axis>=True with positive outer_reg_weights[<axis>] warns."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        IsosterConfigMB(
            bands=["g", "r"], reference_band="g",
            use_outer_center_regularization=True,
            **{fixed_field: True},
        )
    msgs = [str(item.message) for item in w]
    assert any(
        f"outer_reg_weights[{axis!r}]" in m and fixed_field in m for m in msgs
    ), msgs


def test_outer_reg_off_does_not_emit_warnings():
    """Sanity: feature disabled means no outer_reg warnings, even if
    sma_onset etc. would otherwise trip them."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = IsosterConfigMB(
            bands=["g", "r"], reference_band="g",
            use_outer_center_regularization=False,
            sma0=20.0, outer_reg_sma_onset=5.0,  # would warn if feature on
            outer_reg_weights={"center": 0.0, "eps": 0.0, "pa": 0.0},
        )
    msgs = [str(item.message) for item in w]
    for kw in ("outer_reg", "auto-enables"):
        assert not any(kw in m for m in msgs), msgs
    assert cfg.use_outer_center_regularization is False
    assert cfg.geometry_convergence is False  # NOT auto-enabled
