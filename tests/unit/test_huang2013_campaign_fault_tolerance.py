"""Regression tests for Huang2013 campaign fault-tolerance status checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from astropy.io import fits
from astropy.table import Table
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
HUANG2013_DIR = REPO_ROOT / "examples" / "huang2013"
if str(HUANG2013_DIR) not in sys.path:
    sys.path.insert(0, str(HUANG2013_DIR))

import run_huang2013_campaign as campaign  # noqa: E402
import run_huang2013_profile_extraction as profile_extraction  # noqa: E402
import run_huang2013_qa_afterburner as qa_afterburner  # noqa: E402
import run_huang2013_real_mock_demo as real_mock_demo  # noqa: E402


def _build_method_paths(base_dir: Path) -> dict[str, Path]:
    """Build synthetic method artifact paths under a temp directory."""
    return {
        "profile_fits": base_dir / "mock_photutils_baseline_profile.fits",
        "profile_ecsv": base_dir / "mock_photutils_baseline_profile.ecsv",
        "runtime_profile": base_dir / "mock_photutils_baseline_runtime-profile.txt",
        "run_json": base_dir / "mock_photutils_baseline_run.json",
    }


def test_reusable_success_requires_minimum_isophote_count(tmp_path: Path) -> None:
    """Reusable-success checks must reject low isophote-count summaries."""
    paths = _build_method_paths(tmp_path)
    for artifact_path in paths.values():
        artifact_path.write_text("placeholder\n")

    failed_payload = {
        "status": "success",
        "table_summary": {"isophote_count": 2},
    }
    paths["run_json"].write_text(json.dumps(failed_payload))
    assert profile_extraction.is_reusable_success(paths) is False
    assert campaign.is_reusable_success(paths) is False

    success_payload = {
        "status": "success",
        "table_summary": {"isophote_count": 3},
    }
    paths["run_json"].write_text(json.dumps(success_payload))
    assert profile_extraction.is_reusable_success(paths) is True
    assert campaign.is_reusable_success(paths) is True


def test_empty_photutils_profile_is_marked_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty photutils extraction output must fail and persist failed run JSON."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    def fake_runtime_wrapper(func, image, fit_config):  # noqa: ANN001
        return [], {"wall_time_seconds": 0.01, "cpu_time_seconds": 0.01}, "runtime-profile\n"

    monkeypatch.setattr(profile_extraction, "run_with_runtime_profile", fake_runtime_wrapper)

    arguments = SimpleNamespace(
        update=False,
        sma0=None,
        minsma=1.0,
        astep=0.1,
        photutils_nclip=2,
        photutils_sclip=3.0,
        photutils_integrmode="bilinear",
        verbose=False,
        save_log=False,
        use_eccentric_anomaly=False,
        isoster_config_json=None,
        cog_subpixels=9,
    )

    with pytest.raises(RuntimeError, match="insufficient profile table"):
        profile_extraction.run_single_method(
            method_name="photutils",
            image=np.ones((20, 20), dtype=float),
            base_geometry={
                "x0": 10.0,
                "y0": 10.0,
                "eps": 0.2,
                "pa_deg": 30.0,
                "sma0": 3.0,
            },
            arguments=arguments,
            redshift=0.2,
            pixel_scale_arcsec=0.168,
            zeropoint_mag=27.0,
            max_sma=15.0,
            maxgerr=0.5,
            output_dir=output_dir,
            prefix="ESO185-G054_mock3",
            config_tag="baseline",
        )

    run_json_path = output_dir / "ESO185-G054_mock3_photutils_baseline_run.json"
    assert run_json_path.exists()
    payload = json.loads(run_json_path.read_text())
    assert payload["status"] == "failed"
    assert payload["table_summary"]["isophote_count"] == 0
    assert "insufficient profile table" in payload["error"]
    assert "min_required=3" in payload["error"]

    profile_fits_path = output_dir / "ESO185-G054_mock3_photutils_baseline_profile.fits"
    profile_ecsv_path = output_dir / "ESO185-G054_mock3_photutils_baseline_profile.ecsv"
    assert not profile_fits_path.exists()
    assert not profile_ecsv_path.exists()


def test_negative_error_value_fails_extraction_with_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative uncertainty values must fail extraction and persist warning metadata."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    def fake_runtime_wrapper(func, image, fit_config):  # noqa: ANN001
        isophotes = [
            {
                "sma": 3.0,
                "intens": 2.0,
                "intens_err": 0.2,
                "eps": 0.2,
                "pa": np.deg2rad(30.0),
                "x0": 10.0,
                "y0": 10.0,
                "x0_err": -0.1,
                "y0_err": 0.1,
                "stop_code": 0,
                "tflux_e": 10.0,
            },
            {
                "sma": 4.0,
                "intens": 1.8,
                "intens_err": 0.2,
                "eps": 0.2,
                "pa": np.deg2rad(30.0),
                "x0": 10.0,
                "y0": 10.0,
                "x0_err": 0.1,
                "y0_err": 0.1,
                "stop_code": 0,
                "tflux_e": 12.0,
            },
            {
                "sma": 5.0,
                "intens": 1.5,
                "intens_err": 0.2,
                "eps": 0.2,
                "pa": np.deg2rad(30.0),
                "x0": 10.0,
                "y0": 10.0,
                "x0_err": 0.1,
                "y0_err": 0.1,
                "stop_code": 0,
                "tflux_e": 14.0,
            },
        ]
        return isophotes, {"wall_time_seconds": 0.01, "cpu_time_seconds": 0.01}, "runtime-profile\n"

    monkeypatch.setattr(profile_extraction, "run_with_runtime_profile", fake_runtime_wrapper)

    arguments = SimpleNamespace(
        update=False,
        sma0=None,
        minsma=1.0,
        astep=0.1,
        photutils_nclip=2,
        photutils_sclip=3.0,
        photutils_integrmode="bilinear",
        verbose=False,
        save_log=False,
        use_eccentric_anomaly=False,
        isoster_config_json=None,
        cog_subpixels=9,
    )

    with pytest.raises(RuntimeError, match="error-column validation"):
        profile_extraction.run_single_method(
            method_name="photutils",
            image=np.ones((20, 20), dtype=float),
            base_geometry={
                "x0": 10.0,
                "y0": 10.0,
                "eps": 0.2,
                "pa_deg": 30.0,
                "sma0": 3.0,
            },
            arguments=arguments,
            redshift=0.2,
            pixel_scale_arcsec=0.168,
            zeropoint_mag=27.0,
            max_sma=15.0,
            maxgerr=0.5,
            output_dir=output_dir,
            prefix="ESO185-G054_mock3",
            config_tag="baseline",
        )

    run_json_path = output_dir / "ESO185-G054_mock3_photutils_baseline_run.json"
    assert run_json_path.exists()
    payload = json.loads(run_json_path.read_text())
    assert payload["status"] == "failed"
    assert isinstance(payload["warnings"], list)
    assert payload["warnings"]
    assert "negative error values" in payload["warnings"][0]


def test_prepare_profile_table_masks_sb_for_nonpositive_intensity() -> None:
    """Surface brightness and its uncertainty should be undefined for non-positive intensity."""
    isophotes = [
        {
            "sma": 3.0,
            "intens": 1.0,
            "intens_err": 0.1,
            "eps": 0.2,
            "pa": np.deg2rad(20.0),
            "x0": 10.0,
            "y0": 10.0,
            "stop_code": 0,
            "tflux_e": 8.0,
        },
        {
            "sma": 4.0,
            "intens": -0.1,
            "intens_err": 0.2,
            "eps": 0.2,
            "pa": np.deg2rad(20.0),
            "x0": 10.0,
            "y0": 10.0,
            "stop_code": 0,
            "tflux_e": 9.0,
        },
    ]
    profile_table = real_mock_demo.prepare_profile_table(
        isophotes=isophotes,
        image=np.ones((21, 21), dtype=float),
        redshift=0.2,
        pixel_scale_arcsec=0.168,
        zeropoint_mag=27.0,
        cog_subpixels=9,
        method_name="photutils",
    )

    assert np.isfinite(float(profile_table["sb_mag_arcsec2"][0]))
    assert np.isfinite(float(profile_table["sb_err_mag"][0]))
    assert np.isnan(float(profile_table["sb_mag_arcsec2"][1]))
    assert np.isnan(float(profile_table["sb_err_mag"][1]))


def test_validate_profile_table_for_qa_rejects_empty_and_missing_columns() -> None:
    """QA validation must reject empty and malformed tables before plotting."""
    empty_table = Table()
    is_valid, reason = qa_afterburner.validate_profile_table_for_qa(empty_table)
    assert is_valid is False
    assert reason == "empty_profile_table"

    malformed_table = Table()
    malformed_table["sma"] = np.array([1.0, 2.0])
    is_valid, reason = qa_afterburner.validate_profile_table_for_qa(malformed_table)
    assert is_valid is False
    assert isinstance(reason, str)
    assert reason.startswith("missing_required_columns:")


def test_build_comparison_isophote_count_warning_threshold() -> None:
    """Comparison warning should trigger at absolute count difference >= 10."""
    no_warning = qa_afterburner.build_comparison_isophote_count_warning(
        {"isophote_count": 40},
        {"isophote_count": 49},
    )
    assert no_warning is None

    warning = qa_afterburner.build_comparison_isophote_count_warning(
        {"isophote_count": 30},
        {"isophote_count": 45},
    )
    assert isinstance(warning, dict)
    assert warning["code"] == "isophote_count_mismatch"
    assert warning["absolute_difference"] == 15


def test_infer_initial_geometry_applies_huang2013_pa_offset() -> None:
    """Initial geometry should apply the Huang2013-specific PA offset."""
    header = fits.Header()
    header["NCOMP"] = 1
    header["ELLIP1"] = 0.3
    header["PA1"] = 30.0
    header["APPMAG1"] = 16.0
    header["RE_PX1"] = 12.0

    geometry = real_mock_demo.infer_initial_geometry(header, image_shape=(100, 100))
    assert geometry["pa_deg"] == pytest.approx(-60.0)


def test_infer_initial_geometry_uses_fixed_six_pixel_initial_sma() -> None:
    """Initial SMA should use fixed 6 px default regardless of RE_PX1."""
    header = fits.Header()
    header["NCOMP"] = 1
    header["RE_PX1"] = 24.0
    geometry = real_mock_demo.infer_initial_geometry(header, image_shape=(100, 100))
    assert geometry["sma0"] == pytest.approx(6.0)
