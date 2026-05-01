"""Tests for ``isoster.multiband.utils_mb``."""

import numpy as np
import pytest
from astropy.io import fits

from isoster.multiband import (
    IsosterConfigMB,
    fit_image_multiband,
    isophote_results_mb_from_fits,
    isophote_results_mb_to_astropy_table,
    isophote_results_mb_to_fits,
    load_bands_from_hdus,
)


# ---------------------------------------------------------------------------
# Helpers (lifted from test_driver_mb to avoid cross-fixture coupling)
# ---------------------------------------------------------------------------


def _planted(amp: float = 100.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = w = 192
    x0 = y0 = 96.0
    eps, pa = 0.3, 0.5
    re = 25.0
    n = 1.5
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    dx, dy = x - x0, y - y0
    cos_pa, sin_pa = np.cos(pa), np.sin(pa)
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    r = np.sqrt(x_rot**2 + (y_rot / (1.0 - eps)) ** 2)
    bn = 2.0 * n - 0.327
    img = amp * np.exp(-bn * ((r / re) ** (1.0 / n) - 1.0))
    img += rng.normal(0.0, 0.05, size=img.shape)
    return img


def _two_band_fit() -> dict:
    cfg = IsosterConfigMB(
        bands=["g", "r"], reference_band="g",
        sma0=15.0, eps=0.2, pa=0.4,
        astep=0.2, maxsma=50.0,
        debug=True, compute_deviations=True, nclip=0,
    )
    return fit_image_multiband([_planted(100.0, 1), _planted(200.0, 2)], None, cfg)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_to_table_has_per_band_and_shared_columns():
    result = _two_band_fit()
    tbl = isophote_results_mb_to_astropy_table(result)
    cols = set(tbl.colnames)
    # Shared
    for shared in ("sma", "x0", "y0", "eps", "pa", "stop_code", "niter", "valid"):
        assert shared in cols, f"missing shared column {shared}"
    # Per-band suffixes
    for b in ("g", "r"):
        for col in (f"intens_{b}", f"intens_err_{b}", f"rms_{b}",
                    f"a3_{b}", f"b3_{b}", f"a4_{b}", f"b4_{b}"):
            assert col in cols, f"missing per-band column {col}"


def test_fits_roundtrip_preserves_per_band_columns(tmp_path):
    result = _two_band_fit()
    fname = tmp_path / "mb_result.fits"
    isophote_results_mb_to_fits(result, fname)
    loaded = isophote_results_mb_from_fits(fname)

    assert loaded["multiband"] is True
    assert loaded["bands"] == ["g", "r"]
    assert loaded["reference_band"] == "g"
    assert loaded["harmonic_combination"] == "joint"
    assert loaded["variance_mode"] == "ols"
    assert loaded["band_weights"] == {"g": 1.0, "r": 1.0}

    # Same number of isophotes preserved.
    assert len(loaded["isophotes"]) == len(result["isophotes"])
    # Compare per-row per-band intensity columns within float tolerance.
    for orig, restored in zip(result["isophotes"], loaded["isophotes"]):
        for col in ("sma", "x0", "y0", "eps", "pa", "intens_g", "intens_r"):
            if col in orig and orig[col] is not None:
                if isinstance(orig[col], float) and np.isnan(orig[col]):
                    assert np.isnan(float(restored[col]))
                else:
                    np.testing.assert_allclose(
                        float(restored[col]), float(orig[col]), atol=1e-9, rtol=0,
                    )


def test_fits_roundtrip_loose_validity_preserves_n_valid_columns(tmp_path):
    """D9 backport: per-band ``n_valid_<b>`` columns and the top-level
    ``loose_validity`` key round-trip through Schema 1 FITS."""
    img_g = _planted(100.0, 1)
    img_r = _planted(200.0, 2)
    cfg = IsosterConfigMB(
        bands=["g", "r"], reference_band="g",
        sma0=15.0, eps=0.2, pa=0.4, astep=0.2, maxsma=60.0,
        debug=True, nclip=0, loose_validity=True,
    )
    result = fit_image_multiband([img_g, img_r], None, cfg)
    fname = tmp_path / "mb_loose.fits"
    isophote_results_mb_to_fits(result, fname)
    loaded = isophote_results_mb_from_fits(fname)

    assert loaded["loose_validity"] is True
    cols = list(loaded["isophotes"][0].keys())
    for b in ("g", "r"):
        assert f"n_valid_{b}" in cols, f"missing n_valid_{b} after round-trip"
    # Numerics survive: each per-band count matches original.
    for orig, restored in zip(result["isophotes"], loaded["isophotes"]):
        for b in ("g", "r"):
            key = f"n_valid_{b}"
            if key in orig:
                assert int(restored[key]) == int(orig[key])


def test_fits_roundtrip_records_multiband_keys_in_primary_header(tmp_path):
    result = _two_band_fit()
    fname = tmp_path / "mb_result.fits"
    isophote_results_mb_to_fits(result, fname)
    with fits.open(fname) as hdulist:
        primary = hdulist[0]
        assert primary.header["MULTIBND"] is True
        assert primary.header["BANDS"] == "g,r"
        assert primary.header["REFBAND"] == "g"
        assert primary.header["HARMCMB"] == "joint"
        assert primary.header["VARMODE"] == "ols"


def test_fits_roundtrip_with_variance_maps_records_wls(tmp_path):
    cfg = IsosterConfigMB(
        bands=["g", "r"], reference_band="g",
        sma0=15.0, astep=0.2, maxsma=40.0, debug=True, nclip=0,
    )
    img_g = _planted(100.0, 1)
    img_r = _planted(200.0, 2)
    var = np.full_like(img_g, 0.05**2)
    result = fit_image_multiband(
        [img_g, img_r], None, cfg, variance_maps=[var, var.copy()],
    )
    fname = tmp_path / "mb_result_wls.fits"
    isophote_results_mb_to_fits(result, fname)
    loaded = isophote_results_mb_from_fits(fname)
    assert loaded["variance_mode"] == "wls"


# ---------------------------------------------------------------------------
# load_bands_from_hdus
# ---------------------------------------------------------------------------


def test_load_bands_from_hdus_basic():
    hdus = [
        fits.PrimaryHDU(np.ones((32, 32))),
        fits.PrimaryHDU(np.ones((32, 32)) * 2),
    ]
    hdus[0].header["FILTER"] = "HSC-G"
    hdus[1].header["FILTER"] = "HSC-R"
    images, masks, var_maps, bands = load_bands_from_hdus(hdus)
    assert bands == ["g", "r"]
    assert len(images) == 2
    np.testing.assert_array_equal(images[1], np.ones((32, 32)) * 2)
    assert masks == [None, None]
    assert var_maps is None


def test_load_bands_from_hdus_handles_underscore_filters():
    hdus = [fits.PrimaryHDU(np.ones((8, 8)))]
    hdus[0].header["FILTER"] = "z_band"
    _, _, _, bands = load_bands_from_hdus(hdus)
    assert bands == ["z_band"]


def test_load_bands_from_hdus_missing_filter_rejected():
    hdus = [fits.PrimaryHDU(np.ones((8, 8)))]
    with pytest.raises(KeyError, match="FILTER"):
        load_bands_from_hdus(hdus)
