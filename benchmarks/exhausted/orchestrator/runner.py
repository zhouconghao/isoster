"""Sequential campaign runner.

Phase B ships a straight-line ``for galaxy: for arm: fit_one`` driver
with ``skip_existing`` semantics. Parallelism is deferred to Phase E so
we can iterate quickly on output layout, QA figures, and metrics first.

Public entry points:
- :func:`run_campaign`   — top-level, iterates all datasets and tools
- :func:`run_dataset`    — one dataset, one tool (``isoster`` in Phase B)
- :func:`write_inventory` — per-(dataset, tool) inventory FITS
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from astropy.table import Table

from ..adapters.base import GalaxyBundle, safe_galaxy_id
from ..fitters.isoster_fitter import INVENTORY_COLUMNS, run_one_arm
from .config_loader import CampaignPlan, DatasetPlan, ToolPlan, load_campaign


@dataclass
class RunSummary:
    total_requested: int = 0
    total_ran: int = 0
    total_skipped_existing: int = 0
    total_skipped_arm: int = 0
    total_failed: int = 0
    total_ok: int = 0


def run_campaign(plan: CampaignPlan) -> RunSummary:
    """Execute the whole campaign described by ``plan``. Phase B: sequential."""
    campaign_dir = plan.output_root / plan.campaign_name
    campaign_dir.mkdir(parents=True, exist_ok=True)

    _write_campaign_snapshot(plan, campaign_dir)
    _write_environment_snapshot(campaign_dir)

    summary = RunSummary()
    for ds_plan in plan.datasets.values():
        if not ds_plan.enabled:
            print(f"[skip] dataset '{ds_plan.name}' disabled")
            continue
        ds_summary = run_dataset(plan, ds_plan, campaign_dir)
        summary.total_requested += ds_summary.total_requested
        summary.total_ran += ds_summary.total_ran
        summary.total_skipped_existing += ds_summary.total_skipped_existing
        summary.total_skipped_arm += ds_summary.total_skipped_arm
        summary.total_failed += ds_summary.total_failed
        summary.total_ok += ds_summary.total_ok

    return summary


def run_dataset(
    plan: CampaignPlan, ds_plan: DatasetPlan, campaign_dir: Path
) -> RunSummary:
    """Run every enabled tool × arm matrix on one dataset, sequentially."""
    summary = RunSummary()
    dataset_dir = campaign_dir / ds_plan.name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    galaxy_ids = ds_plan.adapter.list_galaxies()
    if ds_plan.select:
        allow = set(ds_plan.select)
        galaxy_ids = [g for g in galaxy_ids if g in allow]

    print(f"=== dataset '{ds_plan.name}' ({len(galaxy_ids)} galaxies) ===")
    for index, galaxy_id in enumerate(galaxy_ids, start=1):
        print(f"  [{index:>3d}/{len(galaxy_ids)}] {galaxy_id}")
        try:
            bundle = ds_plan.adapter.load_galaxy(galaxy_id)
        except Exception as exc:  # noqa: BLE001 - record and continue
            print(f"      ERROR loading galaxy: {exc}")
            continue

        galaxy_dir = dataset_dir / safe_galaxy_id(galaxy_id)
        galaxy_dir.mkdir(parents=True, exist_ok=True)
        _write_manifest(bundle, galaxy_dir)

        for tool_name, tool_plan in plan.tools.items():
            if not tool_plan.enabled:
                continue
            if tool_name != "isoster":
                # Phase B: only isoster fitter exists.
                print(f"      [skip] tool '{tool_name}' — Phase B supports isoster only")
                continue
            tool_summary = _run_tool_on_galaxy(plan, tool_plan, bundle, galaxy_dir)
            summary.total_requested += tool_summary.total_requested
            summary.total_ran += tool_summary.total_ran
            summary.total_skipped_existing += tool_summary.total_skipped_existing
            summary.total_skipped_arm += tool_summary.total_skipped_arm
            summary.total_failed += tool_summary.total_failed
            summary.total_ok += tool_summary.total_ok

    return summary


def _run_tool_on_galaxy(
    plan: CampaignPlan,
    tool_plan: ToolPlan,
    bundle: GalaxyBundle,
    galaxy_dir: Path,
) -> RunSummary:
    tool_dir = galaxy_dir / tool_plan.name
    arms_dir = tool_dir / "arms"
    arms_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    summary = RunSummary()

    for arm_id, arm_delta in tool_plan.arms.items():
        arm_dir = arms_dir / _safe_arm_id(arm_id)
        profile_path = arm_dir / "profile.fits"
        summary.total_requested += 1

        if plan.execution["skip_existing"] and profile_path.is_file():
            print(f"        [skip] {arm_id} (profile.fits exists)")
            summary.total_skipped_existing += 1
            rows.append(_load_cached_row(arm_dir, tool_plan.name, bundle.metadata.galaxy_id, arm_id))
            continue

        print(f"        [run ] {arm_id}")
        write_qa = plan.qa.get("per_galaxy_qa", True)
        write_model_fits = plan.qa.get("residual_models", True)
        start = time.perf_counter()
        row = run_one_arm(
            bundle,
            arm_id,
            arm_delta,
            arm_dir,
            write_qa=write_qa,
            write_model_fits=write_model_fits,
        )
        elapsed = time.perf_counter() - start
        status = row.get("status", "failed")
        if status == "skipped":
            summary.total_skipped_arm += 1
            print(f"             skipped: {row.get('error_msg', '')}")
        elif status == "ok":
            summary.total_ran += 1
            summary.total_ok += 1
            drift = row.get("combined_drift_pix")
            drift_s = f"{drift:.3f}" if isinstance(drift, (int, float)) and np.isfinite(drift) else "nan"
            print(f"             ok  ({elapsed:5.2f}s)  n_iso={row.get('n_iso')} drift={drift_s}px")
        else:
            summary.total_ran += 1
            summary.total_failed += 1
            if plan.execution.get("fail_fast"):
                raise RuntimeError(
                    f"Arm {arm_id} on {bundle.metadata.galaxy_id} returned status={status!r}: "
                    f"{row.get('error_msg')}"
                )
            print(f"             FAIL: {row.get('error_msg', '')}")
        rows.append(row)

    # Per-galaxy-per-tool inventory.
    inv_path = tool_dir / "inventory.fits"
    write_inventory(rows, inv_path)
    return summary


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _safe_arm_id(arm_id: str) -> str:
    return arm_id.replace(":", "_").replace("/", "_")


def write_inventory(rows: list[dict[str, Any]], path: Path) -> None:
    """Write an inventory FITS table with one row per arm."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {col: [row.get(col, _default_for(col)) for row in rows] for col in INVENTORY_COLUMNS}
    table = Table(ordered)
    table.write(path, overwrite=True)


def _default_for(col: str) -> Any:
    numeric_cols = {
        "wall_time_fit_s",
        "wall_time_total_s",
        "frac_stop_nonzero",
        "combined_drift_pix",
        "spline_rms_center",
        "max_dpa_deg",
        "max_deps",
        "outer_gerr_median",
        "outward_drift_x",
        "outward_drift_y",
        "locked_drift_x",
        "locked_drift_y",
    }
    int_cols = {"n_iso", "n_stop_0", "n_stop_1", "n_stop_2", "n_stop_m1", "n_locked"}
    if col in numeric_cols:
        return float("nan")
    if col in int_cols:
        return 0
    return ""


def _load_cached_row(
    arm_dir: Path, tool: str, galaxy_id: str, arm_id: str
) -> dict[str, Any]:
    """Rebuild an inventory row from an existing run_record.json."""
    record_path = arm_dir / "run_record.json"
    row: dict[str, Any] = {col: _default_for(col) for col in INVENTORY_COLUMNS}
    row.update({"galaxy_id": galaxy_id, "tool": tool, "arm_id": arm_id, "status": "cached"})
    if not record_path.is_file():
        row["status"] = "cached_missing_record"
        return row
    with record_path.open() as handle:
        record = json.load(handle)
    row["status"] = record.get("status", "cached")
    row["wall_time_fit_s"] = float(record.get("wall_time_fit_s", float("nan")))
    row["wall_time_total_s"] = float(record.get("wall_time_total_s", float("nan")))
    metrics = record.get("metrics") or {}
    for key in INVENTORY_COLUMNS:
        if key in metrics:
            row[key] = metrics[key]
    # Preserve output paths
    row["profile_path"] = str(arm_dir / "profile.fits")
    model_fits = arm_dir / "model.fits"
    if model_fits.is_file():
        row["model_path"] = str(model_fits)
    qa_png = arm_dir / "qa.png"
    if qa_png.is_file():
        row["qa_path"] = str(qa_png)
    row["config_path"] = str(arm_dir / "config.yaml")
    return row


def _write_manifest(bundle: GalaxyBundle, galaxy_dir: Path) -> None:
    """Write MANIFEST.json summarizing the galaxy bundle for downstream tools."""
    md = bundle.metadata
    manifest = {
        "galaxy_id": md.galaxy_id,
        "dataset": md.dataset,
        "pixel_scale_arcsec": md.pixel_scale_arcsec,
        "sb_zeropoint": md.sb_zeropoint,
        "effective_Re_pix": md.effective_Re_pix,
        "redshift": md.redshift,
        "image_shape": list(np.asarray(bundle.image).shape),
        "has_variance": bundle.variance is not None,
        "has_mask": bundle.mask is not None,
        "initial_geometry": bundle.initial_geometry,
        "extra": md.extra,
    }
    with (galaxy_dir / "MANIFEST.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, default=_json_default)


def _write_campaign_snapshot(plan: CampaignPlan, campaign_dir: Path) -> None:
    with (campaign_dir / "campaign.yaml").open("w") as handle:
        yaml.safe_dump(plan.raw, handle, sort_keys=False)


def _write_environment_snapshot(campaign_dir: Path) -> None:
    env = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy": _module_version("numpy"),
        "scipy": _module_version("scipy"),
        "astropy": _module_version("astropy"),
        "photutils": _module_version("photutils"),
        "isoster": _module_version("isoster"),
        "git_sha": _git_sha_short(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with (campaign_dir / "environment.json").open("w") as handle:
        json.dump(env, handle, indent=2)


def _module_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except ImportError:
        return None
    return getattr(module, "__version__", None)


def _git_sha_short() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # noqa: BLE001 - best-effort only
        return None


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def run_from_yaml(yaml_path: Path) -> RunSummary:
    """Convenience: load the campaign YAML and run it."""
    plan = load_campaign(yaml_path)
    return run_campaign(plan)
