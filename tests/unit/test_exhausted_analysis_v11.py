"""Focused tests for exhausted benchmark v1.1 evaluation helpers."""

from __future__ import annotations

import numpy as np
from astropy.io import fits
from astropy.table import Table


def test_profile_io_honours_pa_units_aliases_and_sentinel_filter(tmp_path):
    from benchmarks.exhausted.analysis.profile_io import (
        profile_summary,
        read_eps,
        read_pa_in_radians,
        valid_isophote_mask,
    )

    path = tmp_path / "profile.fits"
    table = Table(
        {
            "sma": [0.0, 3.0, 5.0, 7.0],
            "pa": [0.0, 30.0, 35.0, 33.0],
            "ellipticity": [0.0, 0.2, 0.23, 0.21],
            "x0": [10.0, 10.0, 10.2, 10.4],
            "y0": [10.0, 10.0, 9.9, 9.8],
            "stop_code": [0, 0, 1, 0],
        }
    )
    table["pa"].unit = "deg"
    table.write(path, overwrite=True)

    loaded = Table.read(path)
    np.testing.assert_allclose(read_pa_in_radians(loaded), np.deg2rad([0.0, 30.0, 35.0, 33.0]))
    np.testing.assert_allclose(read_eps(loaded), [0.0, 0.2, 0.23, 0.21])
    np.testing.assert_array_equal(valid_isophote_mask(loaded), [False, True, True, True])

    summary = profile_summary(path)
    assert summary["n_iso"] == 3
    assert summary["n_stop_0"] == 2
    assert summary["n_stop_1"] == 1
    assert summary["min_sma_pix"] == 3.0
    assert summary["max_dpa_deg"] == 5.0
    assert np.isclose(summary["max_deps"], 0.03)


def test_residual_metrics_noise_contract_and_mask_invariance(tmp_path):
    from benchmarks.exhausted.analysis.residual_metrics import metrics_from_residual

    rng = np.random.default_rng(123)
    image = np.full((80, 80), 10.0)
    residual = rng.normal(0.0, 2.0, image.shape)
    model = image - residual
    model_path = tmp_path / "model.fits"
    fits.PrimaryHDU(data=model).writeto(model_path)

    mask = np.zeros(image.shape, dtype=bool)
    mask[5:15, 5:15] = True
    image_with_masked_spike = image.copy()
    image_with_masked_spike[mask] = 1.0e9
    model_with_masked_spike = model.copy()
    model_with_masked_spike[mask] = -1.0e9
    model_spike_path = tmp_path / "model_spike.fits"
    fits.PrimaryHDU(data=model_with_masked_spike).writeto(model_spike_path)

    kwargs = {
        "mask": mask,
        "x0": 40.0,
        "y0": 40.0,
        "eps": 0.2,
        "pa_rad": 0.0,
        "R_ref_pix": 20.0,
        "maxsma_pix": 35.0,
        "r_inner_floor_pix": 2.0,
    }
    base = metrics_from_residual(image, model_path, **kwargs)
    spiked = metrics_from_residual(image_with_masked_spike, model_spike_path, **kwargs)

    assert np.isclose(base["abs_resid_over_sigma_mid"], np.sqrt(2.0 / np.pi), rtol=0.25)
    for key in (
        "F_ref",
        "resid_rms_inner",
        "resid_rms_mid",
        "resid_rms_outer",
        "sigma_inner",
        "sigma_mid",
        "sigma_outer",
        "abs_resid_over_sigma_mid",
    ):
        assert np.isclose(base[key], spiked[key], equal_nan=True)


def test_residual_and_azimuthal_metrics_empty_zones_are_nan(tmp_path):
    from benchmarks.exhausted.analysis.azimuthal_metrics import azimuthal_metrics
    from benchmarks.exhausted.analysis.residual_metrics import metrics_from_residual

    image = np.ones((20, 20))
    model_path = tmp_path / "model.fits"
    fits.PrimaryHDU(data=image).writeto(model_path)
    mask = np.ones_like(image, dtype=bool)

    metrics = metrics_from_residual(
        image,
        model_path,
        mask=mask,
        x0=10.0,
        y0=10.0,
        eps=0.0,
        pa_rad=0.0,
        R_ref_pix=5.0,
        maxsma_pix=10.0,
    )
    assert np.isnan(metrics["resid_rms_inner"])
    assert np.isnan(metrics["abs_resid_over_sigma_outer"])

    residual = np.full_like(image, np.nan, dtype=float)
    azim = azimuthal_metrics(
        residual,
        x0=10.0,
        y0=10.0,
        eps=0.0,
        pa_rad=0.0,
        R_ref_pix=5.0,
        maxsma_pix=10.0,
    )
    assert np.isnan(azim["azim_A1_inner"])
    assert np.isnan(azim["quadrant_imbalance_outer"])


def test_v11_quality_flags_keep_portable_names():
    from benchmarks.exhausted.analysis.quality_flags import evaluate_flags

    row = {
        "status": "ok",
        "n_iso": 3,
        "first_isophote_retry_attempts": 1,
        "frac_stop_nonzero": 0.7,
        "combined_drift_pix": 6.0,
        "max_dpa_deg": 35.0,
        "max_deps": 0.25,
        "abs_resid_over_sigma_inner": 6.0,
        "abs_resid_over_sigma_outer": 3.0,
    }
    flags = evaluate_flags(row)
    for expected in (
        "FEW_ISOPHOTES",
        "FIRST_ISOPHOTE_RETRY",
        "HIGH_NONZERO_STOP_FRAC",
        "LARGE_DRIFT",
        "LARGE_DPA",
        "LARGE_DEPS",
        "INNER_RESID_LARGE",
        "OUTER_RESID_LARGE",
    ):
        assert expected in flags["flags"]
    assert flags["flag_severity_max"] == 1.5
