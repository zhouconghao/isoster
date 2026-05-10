"""Tests for artifact-level exhausted-campaign metric refresh."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table


def test_enumerate_galaxy_dirs_filters_scenario_and_only(tmp_path):
    from benchmarks.exhausted.campaigns.refresh_model_evaluation import enumerate_galaxy_dirs

    root = tmp_path / "campaigns"
    wanted = root / "s4g_clean_z005" / "s4g" / "NGC1433__clean_z005"
    other_galaxy = root / "s4g_clean_z005" / "s4g" / "NGC0275__clean_z005"
    other_scenario = root / "s4g_wide_z005" / "s4g" / "NGC1433__wide_z005"
    ignored_analysis = root / "_analysis" / "s4g" / "NGC9999__clean_z005"
    for path in (wanted, other_galaxy, other_scenario, ignored_analysis):
        path.mkdir(parents=True)

    found = enumerate_galaxy_dirs(
        root,
        "s4g",
        scenarios={"s4g_clean_z005"},
        only={"NGC1433__clean_z005"},
    )

    assert found == [wanted]


def test_refresh_galaxy_writes_v11_inventory_and_record(tmp_path):
    from benchmarks.exhausted.analysis.inventory import read_inventory
    from benchmarks.exhausted.campaigns.refresh_model_evaluation import RefreshOptions, refresh_galaxy

    galaxy_dir = _make_campaign_galaxy(tmp_path, tool="isoster", arm_id="ref_default")
    stats, rows_by_tool = refresh_galaxy(galaxy_dir, RefreshOptions(write=True, tools=("isoster",)))

    assert stats.galaxies == 1
    assert stats.arms_seen == 1
    assert stats.arms_refreshed == 1
    row = rows_by_tool["isoster"][0]
    assert row["status"] == "ok"
    assert np.isfinite(row["R_ref_used_pix"])
    assert np.isfinite(row["abs_resid_over_sigma_mid"])
    assert "F_ref" in row
    assert row["n_iso_ref_used"] == 6

    inventory = read_inventory(galaxy_dir / "isoster" / "inventory.fits")
    assert "R_ref_used_pix" in inventory.colnames
    assert "abs_resid_over_sigma_outer" in inventory.colnames
    assert int(inventory["n_iso_ref_used"][0]) == 6

    record = json.loads((galaxy_dir / "isoster" / "arms" / "ref_default" / "run_record.json").read_text())
    assert record["evaluation_refresh"]["schema"] == "v1.1"
    assert "R_ref_used_pix" in record["metrics"]
    assert "azim_A1_mid" in record["metrics"]
    assert record["metrics"]["n_iso_ref_used"] == 6
    assert np.isfinite(record["metrics"]["composite_score"])
    assert record["metrics"]["image_sigma_adu"] == 0.01


def test_refresh_photutils_harmonic_model_rewrites_model(tmp_path):
    from benchmarks.exhausted.campaigns.refresh_model_evaluation import RefreshOptions, refresh_galaxy

    galaxy_dir = _make_campaign_galaxy(tmp_path, tool="photutils", arm_id="baseline_median", with_harmonics=True)
    model_path = galaxy_dir / "photutils" / "arms" / "baseline_median" / "model.fits"
    with fits.open(model_path) as hdul:
        before = np.asarray(hdul[0].data, dtype=float)

    stats, rows_by_tool = refresh_galaxy(
        galaxy_dir,
        RefreshOptions(
            write=True,
            tools=("photutils",),
            refresh_photutils_harmonic_models=True,
        ),
    )

    assert stats.arms_refreshed == 1
    assert rows_by_tool["photutils"][0]["model_refresh"] == "profile_harmonics_on"
    with fits.open(model_path) as hdul:
        after = np.asarray(hdul[0].data, dtype=float)
        assert hdul[0].header["MODSRC"] == "profile_harmonics_on"
    assert not np.allclose(before, after)


def _make_campaign_galaxy(
    tmp_path: Path,
    *,
    tool: str,
    arm_id: str,
    with_harmonics: bool = False,
) -> Path:
    root = tmp_path / "campaigns" / "s4g_clean_z005" / "s4g" / "NGC1433__clean_z005"
    arm_dir = root / tool / "arms" / arm_id
    arm_dir.mkdir(parents=True)

    image = _synthetic_image()
    image_path = tmp_path / "source.fits"
    fits.PrimaryHDU(data=image.astype(np.float32)).writeto(image_path)

    manifest = {
        "galaxy_id": "NGC1433/clean_z005",
        "dataset": "s4g",
        "pixel_scale_arcsec": 0.168,
        "sb_zeropoint": 27.0,
        "effective_Re_pix": 9.0,
        "image_sigma": {"image_sigma_adu": 0.01, "sigma_method": "test"},
        "initial_geometry": {
            "x0": 20.0,
            "y0": 20.0,
            "eps": 0.25,
            "pa": 0.2,
            "sma0": 3.0,
            "maxsma": 18.0,
        },
        "extra": {"fits_path": str(image_path)},
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest))

    profile_path = arm_dir / "profile.fits"
    _write_profile(profile_path, with_harmonics=with_harmonics)

    from benchmarks.exhausted.campaigns.refresh_model_evaluation import _load_profile_as_isophotes
    from isoster import build_isoster_model

    isophotes = _load_profile_as_isophotes(profile_path, tool=tool)
    model = build_isoster_model(image.shape, isophotes, use_harmonics=False)
    fits.HDUList(
        [
            fits.PrimaryHDU(data=model.astype(np.float32), header=fits.Header({"EXTNAME": "MODEL"})),
            fits.ImageHDU(data=(image - model).astype(np.float32), header=fits.Header({"EXTNAME": "RESIDUAL"})),
        ]
    ).writeto(arm_dir / "model.fits")

    (arm_dir / "config.yaml").write_text(f"tool: {tool}\narm_id: {arm_id}\n")
    (arm_dir / "run_record.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "wall_time_fit_s": 0.5,
                "wall_time_total_s": 0.8,
                "metrics": {"legacy_metric": 1.0},
            }
        )
    )
    return root


def _synthetic_image() -> np.ndarray:
    yy, xx = np.mgrid[0:40, 0:40]
    radius = np.hypot(xx - 20.0, yy - 20.0)
    return 5.0 * np.exp(-radius / 8.0) + 0.05


def _write_profile(path: Path, *, with_harmonics: bool) -> None:
    sma = np.array([0.0, 3.0, 5.0, 7.0, 9.0, 12.0, 15.0])
    intens = 5.0 * np.exp(-sma / 8.0) + 0.05
    table_data = {
        "sma": sma,
        "intens": intens,
        "intens_err": np.full_like(sma, 0.01),
        "eps": np.array([0.0, 0.2, 0.22, 0.23, 0.24, 0.25, 0.25]),
        "pa": np.full_like(sma, 0.2),
        "x0": np.full_like(sma, 20.0),
        "y0": np.full_like(sma, 20.0),
        "stop_code": np.zeros_like(sma, dtype=int),
        "grad": np.full_like(sma, -0.1),
    }
    if with_harmonics:
        table_data["a3"] = np.array([0.0, 0.02, 0.03, 0.02, 0.01, 0.02, 0.01])
        table_data["b3"] = np.array([0.0, -0.01, -0.02, -0.01, 0.01, -0.01, 0.0])
        table_data["a4"] = np.array([0.0, 0.01, 0.02, 0.01, 0.0, -0.01, 0.0])
        table_data["b4"] = np.array([0.0, 0.03, 0.02, 0.01, 0.02, 0.01, 0.0])
    Table(table_data).write(path, overwrite=True)
