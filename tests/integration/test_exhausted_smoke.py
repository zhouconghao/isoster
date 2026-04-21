"""End-to-end smoke test for the exhausted benchmark orchestrator.

Runs the real campaign pipeline on a synthetic Sersic galaxy with two
isoster arms. Verifies inventory schema, profile monotonicity, QA
rendering, and ``skip_existing`` short-circuit behavior.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits

from tests.fixtures import create_sersic_model


def _write_campaign(
    tmp_path: Path,
    image: np.ndarray,
    arms_yaml: Path,
) -> Path:
    """Materialize the synthetic FITS + campaign + arms YAMLs under tmp_path."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    fits_path = data_dir / "smoke_galaxy.fits"
    fits.PrimaryHDU(data=image.astype(np.float32)).writeto(fits_path, overwrite=True)

    output_root = tmp_path / "out"
    campaign_path = tmp_path / "campaign.yaml"
    campaign_yaml = {
        "campaign_name": "smoke",
        "output_root": str(output_root),
        "tools": {
            "isoster": {
                "enabled": True,
                "arms_file": str(arms_yaml),
            }
        },
        "isoster_harmonic_sweeps": [],
        "datasets": {
            "local_fits": {
                "enabled": True,
                "adapter": "local_fits",
                "root": str(data_dir),
                "pixel_scale_arcsec": 0.168,
                "sb_zeropoint": 27.0,
                "files": [
                    {
                        "path": "smoke_galaxy.fits",
                        "galaxy_id": "smoke_galaxy",
                        "sma0": 4.0,
                    }
                ],
            }
        },
        "qa": {
            "per_galaxy_qa": True,
            "cross_arm_overlay": False,
            "cross_tool_comparison": False,
            "summary_grids": False,
            "residual_models": False,
        },
        "execution": {
            "max_parallel_galaxies": 1,
            "skip_existing": True,
            "dry_run": False,
            "fail_fast": False,
        },
        "summary": {
            "per_galaxy_inventory": True,
            "per_galaxy_cross_arm_table": True,
            "per_tool_cross_arm_table": True,
            "cross_tool_table": False,
            "composite_score_weights": {
                "resid_inner": 1.0,
                "resid_mid": 1.0,
                "resid_outer": 2.0,
                "centroid_drift": 1.0,
                "centroid_tol_pix": 2.0,
                "n_stop_m1": 2.0,
                "frac_stop_nonzero": 5.0,
                "n_iso_completeness": 3.0,
                "wall_time": 0.1,
            },
        },
    }
    with campaign_path.open("w") as handle:
        yaml.safe_dump(campaign_yaml, handle, sort_keys=False)
    return campaign_path


def _write_arms_yaml(tmp_path: Path) -> Path:
    """Minimal 2-arm isoster roster: one reference, one sclip variant."""
    arms_path = tmp_path / "isoster_arms.yaml"
    arms = {
        "arms": {
            "ref_default": {},
            "sclip_off": {"nclip": 0},
        }
    }
    with arms_path.open("w") as handle:
        yaml.safe_dump(arms, handle, sort_keys=False)
    return arms_path


def test_exhausted_smoke_end_to_end(tmp_path):
    """Two-arm × one-galaxy campaign should produce two ok rows, and a
    second invocation with ``skip_existing`` should fit nothing."""
    image, _true_profile, _params = create_sersic_model(
        R_e=8.0,
        n=2.0,
        I_e=1.0,
        eps=0.2,
        pa=0.0,
        size_factor=6.0,
        min_half_size=60,
        seed=42,
    )
    arms_yaml = _write_arms_yaml(tmp_path)
    campaign_yaml = _write_campaign(tmp_path, image, arms_yaml)

    from benchmarks.exhausted.orchestrator.config_loader import load_campaign
    from benchmarks.exhausted.orchestrator.runner import run_campaign

    plan = load_campaign(campaign_yaml)
    summary = run_campaign(plan)

    assert summary.total_requested == 2
    assert summary.total_ok == 2
    assert summary.total_failed == 0
    assert summary.total_skipped_existing == 0

    dataset_dir = Path(plan.output_root) / plan.campaign_name / "local_fits"
    inv_path = dataset_dir / "smoke_galaxy" / "isoster" / "inventory.fits"
    assert inv_path.is_file(), f"missing inventory.fits: {inv_path}"
    with fits.open(inv_path) as hdul:
        rows = hdul[1].data
    assert len(rows) == 2
    statuses = [str(s).strip() for s in rows["status"]]
    assert statuses.count("ok") == 2, f"expected 2 ok, got {statuses}"

    for arm_id in ("ref_default", "sclip_off"):
        arm_dir = dataset_dir / "smoke_galaxy" / "isoster" / "arms" / arm_id
        profile_path = arm_dir / "profile.fits"
        qa_path = arm_dir / "qa.png"
        assert profile_path.is_file()
        assert qa_path.is_file() and qa_path.stat().st_size > 0

        with fits.open(profile_path) as hdul:
            sma = np.asarray(hdul[1].data["sma"], dtype=float)
        assert sma.size >= 2, f"{arm_id}: too few isophotes"
        assert np.all(np.diff(sma) > 0), f"{arm_id}: sma not strictly monotonic"

    # Second run: every arm must short-circuit via skip_existing.
    plan = load_campaign(campaign_yaml)
    summary2 = run_campaign(plan)
    assert summary2.total_requested == 2
    assert summary2.total_skipped_existing == 2
    assert summary2.total_ran == 0
    assert summary2.total_failed == 0
