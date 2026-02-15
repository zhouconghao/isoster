#!/usr/bin/env python3
"""Run Huang2013 extraction + QA afterburner across many galaxies/mocks.

This campaign runner is fault-tolerant:
- A failure in one method/image does not stop the overall campaign.
- Missing per-method outputs are tolerated by the QA afterburner stage.
- Final aggregate statistics are written as JSON + Markdown.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_TAG = "baseline"


def sanitize_label(label: str) -> str:
    """Return filesystem-safe configuration labels."""
    label = label.strip().lower()
    safe_chars = []
    for character in label:
        if character.isalnum() or character in {"-", "_"}:
            safe_chars.append(character)
        else:
            safe_chars.append("-")
    sanitized = "".join(safe_chars).strip("-")
    return sanitized or DEFAULT_CONFIG_TAG

DEFAULT_HUANG_ROOT = Path("/Users/mac/work/hsc/huang2013")
DEFAULT_SUMMARY_DIR = Path("outputs/huang2013_campaign")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for campaign execution."""
    parser = argparse.ArgumentParser(
        description="Run Huang2013 profile extraction + QA afterburner over a galaxy/mock campaign.",
    )
    parser.add_argument(
        "--huang-root",
        type=Path,
        default=DEFAULT_HUANG_ROOT,
        help="Root directory containing galaxy subfolders and mock FITS inputs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output root for per-galaxy artifacts. Default: write into each galaxy folder under --huang-root.",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=DEFAULT_SUMMARY_DIR,
        help="Directory for campaign-level summary JSON/Markdown outputs.",
    )
    parser.add_argument(
        "--galaxies",
        nargs="*",
        default=None,
        help="Optional explicit galaxy list. Default: auto-discover all subfolders in --huang-root.",
    )
    parser.add_argument(
        "--mock-ids",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4],
        help="Mock IDs to process for each galaxy.",
    )
    parser.add_argument("--method", choices=["photutils", "isoster", "both"], default="both", help="Methods to run.")
    parser.add_argument("--config-tag", default=DEFAULT_CONFIG_TAG, help="Configuration tag propagated to extraction/QA scripts.")
    parser.add_argument("--qa-dpi", type=int, default=180, help="QA figure DPI for afterburner.")
    parser.add_argument("--isophote-overlay-step", type=int, default=10, help="Overlay every Nth isophote in QA figures.")
    parser.add_argument("--skip-comparison", action="store_true", help="Skip cross-method comparison QA figure.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of galaxy/mock cases for smoke runs.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned cases/commands without executing them.")

    parser.add_argument("--verbose", action="store_true", help="Emit detailed stage start/end logs.")
    parser.add_argument("--save-log", action="store_true", help="Save per-stage stdout/stderr logs to files.")
    parser.add_argument("--max-runtime-seconds", type=int, default=900, help="Per-stage timeout in seconds (default: 900).")
    parser.add_argument("--continue-from", default=None, help="Resume from this galaxy name (inclusive), e.g. NGC4767.")
    parser.add_argument(
        "--continue-from-case",
        default=None,
        help="Resume from this case (inclusive), e.g. NGC4767_mock1.",
    )
    parser.add_argument("--update", action="store_true", help="Force rerun even when outputs already exist.")
    return parser.parse_args()


def discover_galaxies(huang_root: Path, explicit_galaxies: list[str] | None) -> list[str]:
    """Resolve list of galaxies to process."""
    if explicit_galaxies:
        return sorted(set(explicit_galaxies))

    if not huang_root.exists():
        return []

    galaxies = [entry.name for entry in huang_root.iterdir() if entry.is_dir()]
    return sorted(galaxies)


def build_requested_methods(method_option: str) -> list[str]:
    """Resolve method names for counters."""
    if method_option == "both":
        return ["photutils", "isoster"]
    return [method_option]


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    """Load JSON payload if available and valid."""
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None

    return payload if isinstance(payload, dict) else None


def write_stage_log(log_path: Path, payload: dict[str, Any], enabled: bool) -> None:
    """Persist stage log payload if enabled."""
    if not enabled:
        return
    log_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def normalize_stream_value(value: str | bytes | None) -> str:
    """Normalize subprocess stream payload to UTF-8 text."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def execute_command(
    command: list[str],
    stage_name: str,
    timeout_seconds: int | None,
    verbose: bool,
    save_log: bool,
    log_path: Path | None,
) -> dict[str, Any]:
    """Run command with timeout and optional log persistence."""
    start_time = time.time()

    if verbose:
        print(f"[CAMPAIGN] START stage={stage_name} timeout_seconds={timeout_seconds}")

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        elapsed_seconds = time.time() - start_time
        status = "success" if completed.returncode == 0 else "failed"
        payload = {
            "status": status,
            "timed_out": False,
            "return_code": int(completed.returncode),
            "elapsed_seconds": elapsed_seconds,
            "command": command,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except subprocess.TimeoutExpired as error:
        elapsed_seconds = time.time() - start_time
        payload = {
            "status": "timeout",
            "timed_out": True,
            "return_code": None,
            "elapsed_seconds": elapsed_seconds,
            "command": command,
            "stdout": normalize_stream_value(error.stdout),
            "stderr": normalize_stream_value(error.stderr),
            "error": f"TimeoutExpired: stage exceeded {timeout_seconds} seconds",
        }

    if log_path is not None:
        write_stage_log(log_path, payload, save_log)

    if verbose:
        if payload["status"] == "success":
            print(f"[CAMPAIGN] END stage={stage_name} status=success elapsed={payload['elapsed_seconds']:.2f}s")
        elif payload["status"] == "failed":
            print(
                f"[CAMPAIGN] END stage={stage_name} status=failed return_code={payload['return_code']} "
                f"elapsed={payload['elapsed_seconds']:.2f}s"
            )
        else:
            print(f"[CAMPAIGN] END stage={stage_name} status=timeout elapsed={payload['elapsed_seconds']:.2f}s")

    return payload


def method_artifact_paths(output_dir: Path, prefix: str, method_name: str, config_tag: str) -> dict[str, Path]:
    """Return expected method artifact paths."""
    sanitized_tag = sanitize_label(config_tag)
    stem = f"{prefix}_{method_name}_{sanitized_tag}"
    return {
        "profile_fits": output_dir / f"{stem}_profile.fits",
        "profile_ecsv": output_dir / f"{stem}_profile.ecsv",
        "runtime_profile": output_dir / f"{stem}_runtime-profile.txt",
        "run_json": output_dir / f"{stem}_run.json",
    }


def is_reusable_success(paths: dict[str, Path]) -> bool:
    """Return true when existing outputs are complete and marked successful."""
    required_keys = ["profile_fits", "profile_ecsv", "runtime_profile", "run_json"]
    if not all(paths[key].exists() for key in required_keys):
        return False

    payload = read_json_if_exists(paths["run_json"])
    if payload is None:
        return False

    payload_status = payload.get("status")
    if payload_status is not None and payload_status != "success":
        return False

    return True


def parse_continue_case(case_text: str) -> tuple[str, int] | None:
    """Parse case label like NGC4767_mock1 into (galaxy, mock_id)."""
    if "_mock" not in case_text:
        return None
    galaxy_name, mock_text = case_text.rsplit("_mock", maxsplit=1)
    if not mock_text.isdigit():
        return None
    return galaxy_name, int(mock_text)


def keep_case(
    galaxy_name: str,
    mock_id: int,
    continue_from_galaxy: str | None,
    continue_from_case: tuple[str, int] | None,
) -> bool:
    """Return true when case should be processed under continue-from settings."""
    if continue_from_case is not None:
        target_galaxy, target_mock = continue_from_case
        if galaxy_name < target_galaxy:
            return False
        if galaxy_name == target_galaxy and mock_id < target_mock:
            return False
        return True

    if continue_from_galaxy is not None:
        return galaxy_name >= continue_from_galaxy

    return True


def write_markdown_summary(summary_path: Path, payload: dict[str, Any]) -> None:
    """Write a concise human-readable campaign summary markdown."""
    lines = [
        "# Huang2013 Campaign Summary",
        "",
        "## Overview",
        "",
        f"- Galaxy count: `{payload['galaxy_count']}`",
        f"- Requested cases: `{payload['requested_case_count']}`",
        f"- Processed cases: `{payload['processed_case_count']}`",
        f"- Missing-input cases: `{payload['missing_input_count']}`",
        "",
        "## Stage Outcomes",
        "",
        f"- Extraction invocation failures: `{payload['extraction_invocation_failures']}`",
        f"- Extraction timeouts: `{payload['extraction_timeouts']}`",
        f"- QA invocation failures: `{payload['qa_invocation_failures']}`",
        f"- QA timeouts: `{payload['qa_timeouts']}`",
        f"- Comparison QA generated: `{payload['comparison_qa_generated']}`",
        "",
        "## Method Outcomes",
        "",
    ]

    for method_name in ["photutils", "isoster"]:
        if method_name not in payload["method_counters"]:
            continue
        method_counter = payload["method_counters"][method_name]
        lines.extend(
            [
                f"### {method_name}",
                "",
                f"- Success: `{method_counter['success']}`",
                f"- Failed: `{method_counter['failed']}`",
                f"- Timeout: `{method_counter['timeout']}`",
                f"- Skipped existing: `{method_counter['skipped_existing']}`",
                f"- Unknown: `{method_counter['unknown']}`",
                "",
            ]
        )

    summary_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    """Run the Huang2013 campaign."""
    arguments = parse_arguments()

    profile_script = Path(__file__).with_name("run_huang2013_profile_extraction.py")
    qa_script = Path(__file__).with_name("run_huang2013_qa_afterburner.py")

    galaxies = discover_galaxies(arguments.huang_root, arguments.galaxies)
    requested_methods = build_requested_methods(arguments.method)
    timeout_seconds = None if arguments.max_runtime_seconds <= 0 else int(arguments.max_runtime_seconds)

    continue_from_case = None
    continue_from_galaxy = None
    if arguments.continue_from_case is not None:
        continue_from_case = parse_continue_case(arguments.continue_from_case)
        if continue_from_case is None:
            raise ValueError("--continue-from-case must look like '<GALAXY>_mock<ID>'")
    elif arguments.continue_from is not None:
        parsed = parse_continue_case(arguments.continue_from)
        if parsed is None:
            continue_from_galaxy = arguments.continue_from
        else:
            continue_from_case = parsed

    requested_cases: list[tuple[str, int]] = []
    for galaxy_name in galaxies:
        for mock_id in arguments.mock_ids:
            if keep_case(galaxy_name, mock_id, continue_from_galaxy, continue_from_case):
                requested_cases.append((galaxy_name, mock_id))

    if arguments.limit is not None:
        requested_cases = requested_cases[: max(arguments.limit, 0)]

    arguments.summary_dir.mkdir(parents=True, exist_ok=True)

    method_counters = {
        method_name: {
            "success": 0,
            "failed": 0,
            "timeout": 0,
            "skipped_existing": 0,
            "unknown": 0,
        }
        for method_name in requested_methods
    }

    results_by_case: list[dict[str, Any]] = []
    missing_input_count = 0
    extraction_invocation_failures = 0
    extraction_timeouts = 0
    qa_invocation_failures = 0
    qa_timeouts = 0
    comparison_qa_generated = 0

    for galaxy_name, mock_id in requested_cases:
        prefix = f"{galaxy_name}_mock{mock_id}"
        input_fits = arguments.huang_root / galaxy_name / f"{prefix}.fits"
        output_dir = (
            arguments.output_root / galaxy_name
            if arguments.output_root is not None
            else arguments.huang_root / galaxy_name
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        case_payload: dict[str, Any] = {
            "galaxy": galaxy_name,
            "mock_id": mock_id,
            "prefix": prefix,
            "input_fits": str(input_fits),
            "output_dir": str(output_dir),
            "method_stages": {},
            "qa_stage": None,
        }

        if not input_fits.exists():
            missing_input_count += 1
            case_payload["status"] = "missing_input"
            results_by_case.append(case_payload)
            if arguments.verbose:
                print(f"[CAMPAIGN] SKIP prefix={prefix} reason=missing_input")
            continue

        if arguments.verbose:
            print(f"[CAMPAIGN] CASE start prefix={prefix}")

        if arguments.dry_run:
            case_payload["status"] = "dry_run"
            results_by_case.append(case_payload)
            continue

        # Extraction stage per method
        for method_name in requested_methods:
            stage_name = f"{prefix}:extract:{method_name}"
            method_log_path = output_dir / f"{prefix}_{method_name}.log"

            paths = method_artifact_paths(output_dir, prefix, method_name, arguments.config_tag)
            if not arguments.update and is_reusable_success(paths):
                method_counters[method_name]["skipped_existing"] += 1
                case_payload["method_stages"][method_name] = {
                    "status": "skipped_existing",
                    "reason": "artifacts_already_available",
                    "artifacts": {key: str(value) for key, value in paths.items()},
                }
                if arguments.verbose:
                    print(f"[CAMPAIGN] SKIP stage={stage_name} reason=artifacts_already_available")
                continue

            extract_command = [
                sys.executable,
                str(profile_script),
                "--galaxy",
                galaxy_name,
                "--mock-id",
                str(mock_id),
                "--method",
                method_name,
                "--config-tag",
                arguments.config_tag,
                "--huang-root",
                str(arguments.huang_root),
                "--output-dir",
                str(output_dir),
            ]
            if arguments.verbose:
                extract_command.append("--verbose")
            if arguments.update:
                extract_command.append("--update")

            stage_result = execute_command(
                command=extract_command,
                stage_name=stage_name,
                timeout_seconds=timeout_seconds,
                verbose=arguments.verbose,
                save_log=arguments.save_log,
                log_path=method_log_path,
            )

            case_payload["method_stages"][method_name] = stage_result

            if stage_result["status"] == "timeout":
                method_counters[method_name]["timeout"] += 1
                extraction_timeouts += 1
                continue

            if stage_result["status"] == "failed":
                method_counters[method_name]["failed"] += 1
                extraction_invocation_failures += 1
                continue

            profiles_manifest_path = output_dir / f"{prefix}_profiles_manifest.json"
            profiles_manifest = read_json_if_exists(profiles_manifest_path)
            if profiles_manifest is None:
                method_counters[method_name]["unknown"] += 1
                continue

            method_runs_payload = profiles_manifest.get("method_runs", {})
            method_payload = method_runs_payload.get(method_name, {}) if isinstance(method_runs_payload, dict) else {}
            method_status = method_payload.get("status") if isinstance(method_payload, dict) else None
            skipped_existing = bool(method_payload.get("skipped_existing")) if isinstance(method_payload, dict) else False

            if method_status == "success":
                if skipped_existing:
                    method_counters[method_name]["skipped_existing"] += 1
                else:
                    method_counters[method_name]["success"] += 1
            elif method_status == "failed":
                method_counters[method_name]["failed"] += 1
            else:
                method_counters[method_name]["unknown"] += 1

        # QA stage
        qa_manifest_path = output_dir / f"{prefix}_qa_manifest.json"
        qa_log_path = output_dir / f"{prefix}_qa.log"

        if not arguments.update and qa_manifest_path.exists():
            case_payload["qa_stage"] = {
                "status": "skipped_existing",
                "reason": "qa_manifest_already_available",
                "qa_manifest": str(qa_manifest_path),
            }
            if arguments.verbose:
                print(f"[CAMPAIGN] SKIP stage={prefix}:qa reason=qa_manifest_already_available")
        else:
            qa_command = [
                sys.executable,
                str(qa_script),
                "--galaxy",
                galaxy_name,
                "--mock-id",
                str(mock_id),
                "--method",
                arguments.method,
                "--config-tag",
                arguments.config_tag,
                "--huang-root",
                str(arguments.huang_root),
                "--output-dir",
                str(output_dir),
                "--qa-dpi",
                str(arguments.qa_dpi),
                "--isophote-overlay-step",
                str(arguments.isophote_overlay_step),
                "--profiles-manifest",
                str(output_dir / f"{prefix}_profiles_manifest.json"),
            ]
            if arguments.skip_comparison:
                qa_command.append("--skip-comparison")
            if arguments.verbose:
                qa_command.append("--verbose")

            qa_result = execute_command(
                command=qa_command,
                stage_name=f"{prefix}:qa",
                timeout_seconds=timeout_seconds,
                verbose=arguments.verbose,
                save_log=arguments.save_log,
                log_path=qa_log_path,
            )
            case_payload["qa_stage"] = qa_result

            if qa_result["status"] == "timeout":
                qa_timeouts += 1
            elif qa_result["status"] == "failed":
                qa_invocation_failures += 1

        qa_manifest = read_json_if_exists(qa_manifest_path)
        if qa_manifest is not None and qa_manifest.get("comparison_qa"):
            comparison_qa_generated += 1

        case_payload["status"] = "processed"
        case_payload["profiles_manifest"] = str(output_dir / f"{prefix}_profiles_manifest.json")
        case_payload["qa_manifest"] = str(qa_manifest_path)
        results_by_case.append(case_payload)

        if arguments.verbose:
            print(f"[CAMPAIGN] CASE end prefix={prefix}")

    summary_payload = {
        "huang_root": str(arguments.huang_root),
        "output_root": str(arguments.output_root) if arguments.output_root is not None else None,
        "summary_dir": str(arguments.summary_dir),
        "config_tag": arguments.config_tag,
        "method": arguments.method,
        "mock_ids": arguments.mock_ids,
        "continue_from": arguments.continue_from,
        "continue_from_case": arguments.continue_from_case,
        "max_runtime_seconds": timeout_seconds,
        "update": bool(arguments.update),
        "galaxy_count": len(galaxies),
        "requested_case_count": len(requested_cases),
        "processed_case_count": len(results_by_case),
        "missing_input_count": missing_input_count,
        "extraction_invocation_failures": extraction_invocation_failures,
        "extraction_timeouts": extraction_timeouts,
        "qa_invocation_failures": qa_invocation_failures,
        "qa_timeouts": qa_timeouts,
        "comparison_qa_generated": comparison_qa_generated,
        "method_counters": method_counters,
        "cases": results_by_case,
    }

    summary_json_path = arguments.summary_dir / "huang2013_campaign_summary.json"
    summary_markdown_path = arguments.summary_dir / "huang2013_campaign_summary.md"
    summary_json_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n")
    write_markdown_summary(summary_markdown_path, summary_payload)

    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary MD: {summary_markdown_path}")
    for method_name, method_counter in method_counters.items():
        print(
            f"Method {method_name}: success={method_counter['success']} "
            f"failed={method_counter['failed']} timeout={method_counter['timeout']} "
            f"skipped_existing={method_counter['skipped_existing']} unknown={method_counter['unknown']}"
        )


if __name__ == "__main__":
    main()
