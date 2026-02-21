"""Shared naming and manifest helpers for the Huang2013 campaign workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_test_name(mock_id: int) -> str:
    """Return canonical test folder name from mock ID."""
    return f"mock{mock_id}"


def build_case_prefix(galaxy_name: str, mock_id: int) -> str:
    """Return canonical case prefix used by all Huang2013 stage artifacts."""
    return f"{galaxy_name}_{build_test_name(mock_id)}"


def build_case_output_dir(output_root: Path, galaxy_name: str, mock_id: int) -> Path:
    """Return canonical output folder for one case: <root>/<GALAXY>/<TEST>/."""
    return output_root / galaxy_name / build_test_name(mock_id)


def build_method_stem(prefix: str, method_name: str, config_tag: str) -> str:
    """Return canonical per-method stem used by extraction and QA outputs."""
    return f"{prefix}_{method_name}_{config_tag}"


def build_method_artifact_paths(
    output_dir: Path,
    prefix: str,
    method_name: str,
    config_tag: str,
    include_method_log: bool = False,
) -> dict[str, Path]:
    """Return canonical artifact paths for one method run."""
    method_stem = build_method_stem(prefix, method_name, config_tag)
    paths = {
        "profile_fits": output_dir / f"{method_stem}_profile.fits",
        "profile_ecsv": output_dir / f"{method_stem}_profile.ecsv",
        "runtime_profile": output_dir / f"{method_stem}_runtime-profile.txt",
        "run_json": output_dir / f"{method_stem}_run.json",
    }
    if include_method_log:
        paths["method_log"] = output_dir / f"{prefix}_{method_name}.log"
    return paths


def build_profiles_manifest_path(output_dir: Path, prefix: str) -> Path:
    """Return canonical extraction-manifest filename for one case."""
    return output_dir / f"{prefix}_profiles_manifest.json"


def build_qa_manifest_path(output_dir: Path, prefix: str, manifest_suffix: str = "") -> Path:
    """Return canonical QA-manifest filename for one case."""
    return output_dir / f"{prefix}_qa_manifest{manifest_suffix}.json"


def read_json_dict_if_exists(path: Path | None) -> dict[str, Any] | None:
    """Read JSON payload from path when it exists and is a dictionary."""
    if path is None or not path.exists():
        return None

    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None

    return payload if isinstance(payload, dict) else None


def load_method_statuses_from_profiles_manifest(manifest_path: Path | None) -> dict[str, str]:
    """Read per-method status values from extraction manifest."""
    manifest_payload = read_json_dict_if_exists(manifest_path)
    if manifest_payload is None:
        return {}

    method_runs = manifest_payload.get("method_runs", {})
    if not isinstance(method_runs, dict):
        return {}

    statuses: dict[str, str] = {}
    for method_name, method_payload in method_runs.items():
        if not isinstance(method_payload, dict):
            continue
        status_value = method_payload.get("status")
        if isinstance(status_value, str):
            statuses[method_name] = status_value
    return statuses
