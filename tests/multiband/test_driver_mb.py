"""Tests for ``isoster.multiband.driver_mb.fit_image_multiband``."""

import warnings

import numpy as np
import pytest

from isoster.multiband import IsosterConfigMB, fit_image_multiband


# ---------------------------------------------------------------------------
# Synthetic galaxy fixture (Sersic with shared geometry across bands)
# ---------------------------------------------------------------------------


def _planted_galaxy(
    h: int = 192, w: int = 192,
    x0: float = 96.0, y0: float = 96.0,
    eps: float = 0.3, pa: float = 0.5,
    re: float = 25.0, n_sersic: float = 1.5,
    amplitude: float = 1.0, noise_sigma: float = 0.005,
    seed: int = 0,
) -> np.ndarray:
    """Sersic in elliptical coordinates: truth (x0, y0, eps, pa)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    dx = x - x0
    dy = y - y0
    cos_pa, sin_pa = np.cos(pa), np.sin(pa)
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    r = np.sqrt(x_rot**2 + (y_rot / (1.0 - eps)) ** 2)
    bn = 2.0 * n_sersic - 0.327
    img = amplitude * np.exp(-bn * ((r / re) ** (1.0 / n_sersic) - 1.0))
    img += rng.normal(0.0, noise_sigma, size=img.shape)
    return img


@pytest.fixture
def planted_two_band():
    """Two bands with shared geometry, different per-band amplitudes."""
    img_g = _planted_galaxy(amplitude=100.0, noise_sigma=0.05, seed=1)
    img_r = _planted_galaxy(amplitude=200.0, noise_sigma=0.05, seed=2)
    return img_g, img_r


# ---------------------------------------------------------------------------
# B=1 fallback: delegation to single-band fit_image
# ---------------------------------------------------------------------------


def test_b1_delegates_to_single_band(planted_two_band):
    img, _ = planted_two_band
    cfg = IsosterConfigMB(
        bands=["g"], reference_band="g",
        sma0=15.0, eps=0.2, pa=0.4,
        astep=0.2, maxsma=60.0, debug=True, nclip=0,
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = fit_image_multiband([img], None, cfg)
    assert any("delegating to" in str(w.message) for w in captured)
    # Single-band schema: no `intens_g` column (legacy uses bare `intens`).
    iso0 = result["isophotes"][0]
    assert "intens" in iso0
    assert "intens_g" not in iso0
    # Multi-band top-level keys are absent.
    assert "multiband" not in result
    assert "bands" not in result


def test_b1_delegation_unwraps_variance_maps_list(planted_two_band):
    """Regression for B17: a length-1 variance_maps list must be unwrapped
    to a single ndarray when delegating to single-band ``fit_image``.

    The single-band sampler accepts only ``variance_map`` (singular,
    ndarray); passing a list would raise. The multi-band B=1 fallback
    must therefore unwrap the list before delegation.
    """
    img, _ = planted_two_band
    var = np.full_like(img, 0.25, dtype=np.float64)
    cfg = IsosterConfigMB(
        bands=["g"], reference_band="g",
        sma0=15.0, eps=0.2, pa=0.4,
        astep=0.2, maxsma=60.0, debug=True, nclip=0,
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = fit_image_multiband([img], None, cfg, variance_maps=[var])
    assert any("delegating to" in str(w.message) for w in captured)
    iso = next(
        (i for i in result["isophotes"] if float(i.get("sma", 0.0)) > 0.0),
        None,
    )
    assert iso is not None, "Expected at least one non-central isophote"
    assert "intens" in iso
    assert float(iso.get("intens_err", 0.0)) > 0.0  # WLS propagated errors


def test_b1_delegation_unwraps_variance_maps_tuple(planted_two_band):
    """Length-1 tuples are unwrapped just like lists."""
    img, _ = planted_two_band
    var = np.full_like(img, 0.25, dtype=np.float64)
    cfg = IsosterConfigMB(
        bands=["g"], reference_band="g",
        sma0=15.0, eps=0.2, pa=0.4,
        astep=0.2, maxsma=60.0, debug=True, nclip=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_image_multiband([img], None, cfg, variance_maps=(var,))
    iso = next(
        (i for i in result["isophotes"] if float(i.get("sma", 0.0)) > 0.0),
        None,
    )
    assert iso is not None
    assert float(iso.get("intens_err", 0.0)) > 0.0


def test_b1_delegation_unwraps_masks_list(planted_two_band):
    """Length-1 mask list is unwrapped to a single ndarray."""
    img, _ = planted_two_band
    mask = np.zeros_like(img, dtype=bool)
    cfg = IsosterConfigMB(
        bands=["g"], reference_band="g",
        sma0=15.0, eps=0.2, pa=0.4,
        astep=0.2, maxsma=60.0, debug=True, nclip=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_image_multiband([img], [mask], cfg)
    assert "isophotes" in result and len(result["isophotes"]) > 0


# ---------------------------------------------------------------------------
# B>=2: end-to-end joint fit on planted galaxy
# ---------------------------------------------------------------------------


def test_two_band_end_to_end_recovers_geometry(planted_two_band):
    img_g, img_r = planted_two_band
    cfg = IsosterConfigMB(
        bands=["g", "r"], reference_band="g",
        sma0=15.0, eps=0.2, pa=0.4,
        astep=0.15, maxsma=60.0,
        debug=True, compute_deviations=True, nclip=0,
    )
    result = fit_image_multiband([img_g, img_r], None, cfg)

    assert result["multiband"] is True
    assert result["bands"] == ["g", "r"]
    assert result["reference_band"] == "g"
    assert result["harmonic_combination"] == "joint"
    assert result["band_weights"] == {"g": 1.0, "r": 1.0}
    assert result["variance_mode"] == "ols"

    isophotes = result["isophotes"]
    # Filter to acceptable stop codes (0 or 2) at SMA in the well-fit
    # mid-radius range (avoid the unconstrained outermost rings).
    mids = [
        iso for iso in isophotes
        if iso["valid"]
        and iso["stop_code"] in (0, 2)
        and 15.0 <= iso["sma"] <= 40.0
    ]
    assert len(mids) >= 5
    # Average geometry on these isophotes recovers the truth within
    # the Q17 quality bars.
    x0_avg = np.mean([iso["x0"] for iso in mids])
    y0_avg = np.mean([iso["y0"] for iso in mids])
    eps_avg = np.mean([iso["eps"] for iso in mids])
    pa_avg = np.mean([iso["pa"] for iso in mids])
    assert abs(x0_avg - 96.0) < 0.5
    assert abs(y0_avg - 96.0) < 0.5
    assert abs(eps_avg - 0.3) < 0.02
    pa_diff = abs((pa_avg - 0.5 + np.pi / 2) % np.pi - np.pi / 2)
    assert pa_diff < np.deg2rad(1.0)

    # Per-band intensity columns populated for every isophote.
    for iso in isophotes:
        assert "intens_g" in iso
        assert "intens_r" in iso


def test_two_band_with_variance_maps_runs(planted_two_band):
    img_g, img_r = planted_two_band
    var = np.full_like(img_g, 0.05**2)
    cfg = IsosterConfigMB(
        bands=["g", "r"], reference_band="g",
        sma0=15.0, eps=0.2, pa=0.4,
        astep=0.2, maxsma=50.0, debug=True, nclip=0,
    )
    result = fit_image_multiband(
        [img_g, img_r], None, cfg, variance_maps=[var, var.copy()],
    )
    assert result["variance_mode"] == "wls"
    valid_count = sum(1 for iso in result["isophotes"] if iso["valid"])
    assert valid_count > 0


def test_band_weights_passthrough(planted_two_band):
    img_g, img_r = planted_two_band
    cfg = IsosterConfigMB(
        bands=["g", "r"], reference_band="g",
        band_weights={"g": 2.0, "r": 0.5},
        sma0=15.0, astep=0.2, maxsma=40.0, nclip=0,
    )
    result = fit_image_multiband([img_g, img_r], None, cfg)
    assert result["band_weights"] == {"g": 2.0, "r": 0.5}


def test_ref_mode_runs(planted_two_band):
    img_g, img_r = planted_two_band
    cfg = IsosterConfigMB(
        bands=["g", "r"], reference_band="g",
        harmonic_combination="ref",
        sma0=15.0, astep=0.2, maxsma=40.0, debug=True, nclip=0,
    )
    result = fit_image_multiband([img_g, img_r], None, cfg)
    assert result["harmonic_combination"] == "ref"
    assert any(iso["valid"] for iso in result["isophotes"])


# ---------------------------------------------------------------------------
# Validation paths
# ---------------------------------------------------------------------------


def test_missing_config_raises():
    img = _planted_galaxy()
    with pytest.raises(ValueError, match="config is required"):
        fit_image_multiband([img], None, None)


def test_image_count_mismatch_with_bands_rejected(planted_two_band):
    img_g, img_r = planted_two_band
    cfg = IsosterConfigMB(bands=["g", "r", "i"], reference_band="g")
    with pytest.raises(ValueError, match="does not match"):
        fit_image_multiband([img_g, img_r], None, cfg)


def test_image_shape_mismatch_rejected():
    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g")
    img1 = np.zeros((100, 100), dtype=np.float64)
    img2 = np.zeros((50, 50), dtype=np.float64)
    with pytest.raises(ValueError, match="shape"):
        fit_image_multiband([img1, img2], None, cfg)


def test_variance_maps_count_mismatch_rejected(planted_two_band):
    img_g, img_r = planted_two_band
    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g")
    var = np.ones_like(img_g)
    with pytest.raises(ValueError, match="does not match"):
        fit_image_multiband(
            [img_g, img_r], None, cfg, variance_maps=[var, var, var],
        )


def test_variance_maps_tuple_count_mismatch_rejected(planted_two_band):
    """Regression for B1: tuples must be validated, not silently accepted."""
    img_g, img_r = planted_two_band
    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g")
    var = np.ones_like(img_g)
    with pytest.raises(ValueError, match="does not match"):
        fit_image_multiband(
            [img_g, img_r], None, cfg, variance_maps=(var, var, var),
        )


def test_variance_maps_non_sequence_rejected(planted_two_band):
    """Non-ndarray non-sequence inputs are explicitly rejected."""
    img_g, img_r = planted_two_band
    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g")
    with pytest.raises(TypeError, match="non-sequence"):
        fit_image_multiband(
            [img_g, img_r], None, cfg, variance_maps=42,  # type: ignore[arg-type]
        )


def test_masks_count_mismatch_rejected(planted_two_band):
    """Regression for B1: per-band mask sequence length is checked."""
    img_g, img_r = planted_two_band
    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g")
    bad_mask = np.zeros_like(img_g, dtype=bool)
    with pytest.raises(ValueError, match="does not match"):
        fit_image_multiband([img_g, img_r], [bad_mask, bad_mask, bad_mask], cfg)


def test_masks_tuple_count_mismatch_rejected(planted_two_band):
    """Tuples are accepted and length-checked just like lists."""
    img_g, img_r = planted_two_band
    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g")
    bad_mask = np.zeros_like(img_g, dtype=bool)
    with pytest.raises(ValueError, match="does not match"):
        fit_image_multiband([img_g, img_r], (bad_mask, bad_mask, bad_mask), cfg)


def test_first_isophote_failure_warning():
    """Sma0 entirely off-image triggers FIRST_FEW_ISOPHOTE_FAILURE."""
    cfg = IsosterConfigMB(
        bands=["g", "r"], reference_band="g",
        x0=10.0, y0=10.0, sma0=200.0, maxsma=300.0, astep=0.2, nclip=0,
        max_retry_first_isophote=0,
    )
    img = np.zeros((128, 128), dtype=np.float64)  # featureless image
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = fit_image_multiband([img, img], None, cfg)
    assert any("FIRST_FEW_ISOPHOTE_FAILURE" in str(w.message) for w in captured)
    assert result.get("first_isophote_failure") is True
