#!/usr/bin/env python3
"""Run Huang2013 1-D profile extraction only (no QA plotting).

This script executes photutils and/or isoster independently and writes:
- profile FITS and ECSV tables
- per-method runtime profile text
- per-method run JSON
- extraction manifest JSON

QA figures and comparison report are intentionally excluded and should be
produced later by the QA afterburner script.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from run_huang2013_real_mock_demo import (
    DEFAULT_CONFIG_TAG,
    DEFAULT_PIXEL_SCALE_ARCSEC,
    DEFAULT_REDSHIFT,
    DEFAULT_ZEROPOINT,
    collect_software_versions,
    compute_sha256,
    dump_json,
    get_header_value,
    infer_initial_geometry,
    prepare_profile_table,
    read_mock_image,
    run_isoster_fit,
    run_photutils_fit,
    run_with_runtime_profile,
    sanitize_label,
    summarize_negative_error_values,
    summarize_table,
)

MIN_SUCCESS_ISOPHOTE_COUNT = 3
MAX_FIT_ATTEMPTS = 5
SMA0_RETRY_INCREMENT_PIX = 2.0
ASTEP_RETRY_INCREMENT = 0.02
MAXSMA_RETRY_DECAY_FACTOR = 0.95


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for profile extraction."""
    parser = argparse.ArgumentParser(
        description="Run Huang2013 profile extraction (photutils/isoster) without QA plotting.",
    )

    parser.add_argument("--galaxy", required=True, help="Galaxy folder/name, e.g., IC2597")
    parser.add_argument("--mock-id", type=int, default=1, help="Mock ID suffix in FITS name")
    parser.add_argument(
        "--huang-root",
        type=Path,
        default=Path("/Users/mac/work/hsc/huang2013"),
        help="Root directory containing galaxy subfolders.",
    )
    parser.add_argument(
        "--input-fits",
        type=Path,
        default=None,
        help="Explicit input FITS path. Overrides --huang-root/--galaxy/--mock-id.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for extracted profiles. Defaults to FITS folder.",
    )

    parser.add_argument(
        "--method",
        choices=["photutils", "isoster", "both"],
        default="both",
        help="Method to run.",
    )
    parser.add_argument(
        "--config-tag",
        default=DEFAULT_CONFIG_TAG,
        help="Suffix tag to distinguish configuration variants.",
    )

    parser.add_argument("--redshift", type=float, default=None, help="Override redshift")
    parser.add_argument(
        "--pixel-scale",
        type=float,
        default=None,
        help="Override pixel scale [arcsec/pix]. Default: FITS header PIXSCALE.",
    )
    parser.add_argument("--zeropoint", type=float, default=None, help="Override magnitude zeropoint")
    parser.add_argument("--psf-fwhm", type=float, default=None, help="Override PSF FWHM [arcsec]")

    parser.add_argument("--sma0", type=float, default=None, help="Initial SMA [pix]")
    parser.add_argument("--minsma", type=float, default=1.0, help="Minimum SMA [pix]")
    parser.add_argument("--maxsma", type=float, default=None, help="Maximum SMA [pix]")
    parser.add_argument("--astep", type=float, default=0.1, help="SMA growth step")
    parser.add_argument("--maxgerr", type=float, default=None, help="Gradient error threshold")

    parser.add_argument("--photutils-nclip", type=int, default=2, help="photutils nclip")
    parser.add_argument("--photutils-sclip", type=float, default=3.0, help="photutils sclip")
    parser.add_argument(
        "--photutils-integrmode",
        choices=["bilinear", "nearest_neighbor", "mean"],
        default="bilinear",
        help="photutils integration mode",
    )

    parser.add_argument(
        "--use-eccentric-anomaly",
        action="store_true",
        help="Enable isoster eccentric-anomaly sampling. Default: disabled.",
    )
    parser.add_argument(
        "--isoster-config-json",
        type=Path,
        default=None,
        help="JSON file with isoster config overrides.",
    )

    parser.add_argument(
        "--cog-subpixels",
        type=int,
        default=9,
        help="Subpixel factor for true CoG aperture photometry.",
    )

    parser.add_argument("--verbose", action="store_true", help="Emit stage and method progress logs.")
    parser.add_argument("--save-log", action="store_true", help="Write per-method logs to output directory.")
    parser.add_argument("--update", action="store_true", help="Force rerun and overwrite existing method outputs.")

    return parser.parse_args()


def build_input_path(arguments: argparse.Namespace) -> Path:
    """Resolve the input FITS path."""
    if arguments.input_fits is not None:
        return arguments.input_fits
    return arguments.huang_root / arguments.galaxy / f"{arguments.galaxy}_mock{arguments.mock_id}.fits"


def infer_default_maxsma(header, image_shape: tuple[int, int]) -> float:
    """Infer robust default maxsma from FITS header component scales."""
    image_limit = 0.48 * min(image_shape)

    component_count = int(get_header_value(header, "NCOMP", 0))
    component_re_pixels = []
    for index in range(1, component_count + 1):
        key_name = f"RE_PX{index}"
        if key_name in header:
            value = float(header[key_name])
            if np.isfinite(value) and value > 0.0:
                component_re_pixels.append(value)

    if component_re_pixels:
        max_component_re = max(component_re_pixels)
        science_limit = max(30.0, 8.0 * max_component_re)
    else:
        science_limit = 0.3 * min(image_shape)

    return float(min(image_limit, science_limit))


def method_artifact_paths(output_dir: Path, prefix: str, method_name: str, config_tag: str) -> dict[str, Path]:
    """Return per-method artifact paths."""
    stem = f"{prefix}_{method_name}_{config_tag}"
    return {
        "profile_fits": output_dir / f"{stem}_profile.fits",
        "profile_ecsv": output_dir / f"{stem}_profile.ecsv",
        "runtime_profile": output_dir / f"{stem}_runtime-profile.txt",
        "run_json": output_dir / f"{stem}_run.json",
        "method_log": output_dir / f"{prefix}_{method_name}.log",
    }


def is_reusable_success(paths: dict[str, Path]) -> bool:
    """Return true when existing outputs represent a reusable successful run."""
    required_keys = ["profile_fits", "profile_ecsv", "runtime_profile", "run_json"]
    if not all(paths[key].exists() for key in required_keys):
        return False

    try:
        payload = json.loads(paths["run_json"].read_text())
    except json.JSONDecodeError:
        return False

    payload_status = payload.get("status")
    if payload_status is not None and payload_status != "success":
        return False

    table_summary = payload.get("table_summary", {})
    if not isinstance(table_summary, dict):
        return False

    isophote_count = table_summary.get("isophote_count")
    if not isinstance(isophote_count, int) or isophote_count < MIN_SUCCESS_ISOPHOTE_COUNT:
        return False

    return True


def load_existing_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load existing manifest if present, otherwise empty dictionary."""
    if not manifest_path.exists():
        return {}

    try:
        payload = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return {}

    if not isinstance(payload, dict):
        return {}

    return payload


def write_method_log(log_path: Path, lines: list[str], enabled: bool) -> None:
    """Write method log lines when logging is enabled."""
    if not enabled:
        return
    log_path.write_text("\n".join(lines) + "\n")


def build_retry_attempt_config(base_fit_config: dict[str, Any], attempt_index: int) -> dict[str, Any]:
    """Build per-attempt fit config using fixed Huang2013 retry increments."""
    if attempt_index < 1:
        raise ValueError(f"attempt_index must be >= 1, got {attempt_index}")

    attempt_config = dict(base_fit_config)
    retry_offset = attempt_index - 1
    attempt_config["sma0"] = float(base_fit_config["sma0"] + SMA0_RETRY_INCREMENT_PIX * retry_offset)
    attempt_config["astep"] = float(base_fit_config["astep"] + ASTEP_RETRY_INCREMENT * retry_offset)
    attempt_config["maxsma"] = float(base_fit_config["maxsma"] * (MAXSMA_RETRY_DECAY_FACTOR**retry_offset))
    return attempt_config


def write_failed_run_payload(
    run_json_path: Path,
    prefix: str,
    method_name: str,
    config_tag: str,
    fit_config: dict[str, Any] | None,
    fit_retry_log: list[dict[str, Any]],
    runtime_info: dict[str, Any] | None,
    table_summary: dict[str, Any] | None,
    outputs: dict[str, Path],
    error_message: str,
    warning_messages: list[str] | None = None,
    validation: dict[str, Any] | None = None,
) -> None:
    """Persist failed per-method run payload for downstream status checks."""
    payload = {
        "prefix": prefix,
        "method": method_name,
        "config_tag": config_tag,
        "status": "failed",
        "runtime": runtime_info,
        "fit_config": fit_config,
        "table_summary": table_summary,
        "fit_retry_log": fit_retry_log,
        "attempt_count": len(fit_retry_log),
        "max_attempts": MAX_FIT_ATTEMPTS,
        "error": error_message,
        "warnings": warning_messages if warning_messages else [],
        "outputs": {
            "profile_fits": str(outputs["profile_fits"]),
            "profile_ecsv": str(outputs["profile_ecsv"]),
            "runtime_profile": str(outputs["runtime_profile"]),
        },
    }
    if validation is not None:
        payload["validation"] = validation
    dump_json(run_json_path, payload)


def format_negative_error_summary(negative_error_entries: list[dict[str, Any]]) -> str:
    """Format concise summary text for negative-error validation failures."""
    formatted_entries = []
    for entry in negative_error_entries:
        formatted_entries.append(
            (
                f"{entry['column']}(count={entry['count']},"
                f"min={entry['minimum']:.6g},first_index={entry['first_index']})"
            )
        )
    return "; ".join(formatted_entries)


def run_single_method(
    method_name: str,
    image: np.ndarray,
    base_geometry: dict[str, float],
    arguments: argparse.Namespace,
    redshift: float,
    pixel_scale_arcsec: float,
    zeropoint_mag: float,
    max_sma: float,
    maxgerr: float,
    output_dir: Path,
    prefix: str,
    config_tag: str,
) -> dict[str, Any]:
    """Execute one method and persist profile extraction artifacts."""
    paths = method_artifact_paths(output_dir, prefix, method_name, config_tag)
    log_lines = [
        f"method={method_name}",
        f"prefix={prefix}",
        f"start_time_unix={time.time():.6f}",
    ]

    if not arguments.update and is_reusable_success(paths):
        log_lines.append("status=skipped_existing")
        write_method_log(paths["method_log"], log_lines, arguments.save_log)
        return {
            "method": method_name,
            "status": "success",
            "skipped_existing": True,
            "run_json": str(paths["run_json"]),
            "profile_fits": str(paths["profile_fits"]),
            "profile_ecsv": str(paths["profile_ecsv"]),
            "runtime_profile": str(paths["runtime_profile"]),
        }

    fit_retry_log: list[dict[str, Any]] = []
    if method_name == "photutils":
        base_fit_config = {
            "x0": base_geometry["x0"],
            "y0": base_geometry["y0"],
            "eps": base_geometry["eps"],
            "pa_deg": base_geometry["pa_deg"],
            "sma0": float(arguments.sma0 if arguments.sma0 is not None else base_geometry["sma0"]),
            "minsma": float(arguments.minsma),
            "maxsma": float(max_sma),
            "astep": float(arguments.astep),
            "maxgerr": float(maxgerr),
            "nclip": int(arguments.photutils_nclip),
            "sclip": float(arguments.photutils_sclip),
            "integrmode": arguments.photutils_integrmode,
        }
    else:
        base_fit_config = {
            "x0": base_geometry["x0"],
            "y0": base_geometry["y0"],
            "eps": base_geometry["eps"],
            "pa": float(np.deg2rad(base_geometry["pa_deg"])),
            "sma0": float(arguments.sma0 if arguments.sma0 is not None else base_geometry["sma0"]),
            "minsma": float(arguments.minsma),
            "maxsma": float(max_sma),
            "astep": float(arguments.astep),
            "maxgerr": float(maxgerr),
            "nclip": 2,
            "sclip": 3.0,
            "conver": 0.05,
            "minit": 10,
            "maxit": 100,
            "compute_errors": True,
            "compute_deviations": True,
            "full_photometry": True,
            "compute_cog": True,
            "use_eccentric_anomaly": bool(arguments.use_eccentric_anomaly),
            "simultaneous_harmonics": False,
            "harmonic_orders": [3, 4],
            "permissive_geometry": False,
            "use_central_regularization": False,
        }
        if arguments.isoster_config_json is not None:
            overrides = json.loads(arguments.isoster_config_json.read_text())
            if "pa_deg" in overrides and "pa" not in overrides:
                overrides["pa"] = float(np.deg2rad(float(overrides.pop("pa_deg"))))
            base_fit_config.update(overrides)

    successful_profile_table: Any | None = None
    successful_runtime_info: dict[str, Any] | None = None
    successful_runtime_profile_text = ""
    successful_fit_config: dict[str, Any] | None = None
    successful_table_summary: dict[str, Any] | None = None

    last_runtime_info: dict[str, Any] | None = None
    last_table_summary: dict[str, Any] | None = None
    last_fit_config: dict[str, Any] | None = None
    last_warning_messages: list[str] = []
    last_validation_payload: dict[str, Any] | None = None
    last_error_message = "fit did not start"

    for attempt_index in range(1, MAX_FIT_ATTEMPTS + 1):
        attempt_config = build_retry_attempt_config(base_fit_config, attempt_index)
        if arguments.verbose:
            print(
                f"[EXTRACT] START method={method_name} attempt={attempt_index} "
                f"sma0={attempt_config['sma0']:.3f} astep={attempt_config['astep']:.3f} "
                f"maxsma={attempt_config['maxsma']:.3f}"
            )

        try:
            if method_name == "photutils":
                isophotes, runtime_info, runtime_profile_text = run_with_runtime_profile(
                    run_photutils_fit,
                    image,
                    attempt_config,
                )
                fit_config = attempt_config
            else:
                fit_output, runtime_info, runtime_profile_text = run_with_runtime_profile(
                    run_isoster_fit,
                    image,
                    attempt_config,
                )
                isophotes = fit_output[0]
                fit_config = fit_output[1]
        except Exception as error:
            last_error_message = f"{type(error).__name__}: {error}"
            fit_retry_log.append(
                {
                    "attempt": attempt_index,
                    "status": "failed",
                    "failure_reason": "fit_exception",
                    "sma0": attempt_config["sma0"],
                    "astep": attempt_config["astep"],
                    "maxsma": attempt_config["maxsma"],
                    "error": last_error_message,
                }
            )
            if arguments.verbose:
                print(
                    f"[EXTRACT] END method={method_name} attempt={attempt_index} "
                    f"status=failed error={last_error_message}"
                )
            continue

        profile_table = prepare_profile_table(
            isophotes=isophotes,
            image=image,
            redshift=redshift,
            pixel_scale_arcsec=pixel_scale_arcsec,
            zeropoint_mag=zeropoint_mag,
            cog_subpixels=int(arguments.cog_subpixels),
            method_name=method_name,
        )
        table_summary = summarize_table(profile_table)
        isophote_count = int(table_summary.get("isophote_count", 0))

        negative_error_entries = summarize_negative_error_values(profile_table)
        if negative_error_entries:
            negative_error_summary = format_negative_error_summary(negative_error_entries)
            warning_message = f"{method_name} extraction found negative error values: {negative_error_summary}"
            last_error_message = f"{method_name} extraction failed error-column validation: {negative_error_summary}"
            last_runtime_info = runtime_info
            last_table_summary = table_summary
            last_fit_config = fit_config
            last_warning_messages = [warning_message]
            last_validation_payload = {"negative_error_entries": negative_error_entries}

            fit_retry_log.append(
                {
                    "attempt": attempt_index,
                    "status": "failed",
                    "failure_reason": "negative_error_validation",
                    "sma0": attempt_config["sma0"],
                    "astep": attempt_config["astep"],
                    "maxsma": attempt_config["maxsma"],
                    "isophote_count": isophote_count,
                    "error": last_error_message,
                    "warning": warning_message,
                }
            )
            if arguments.verbose:
                print(
                    f"[EXTRACT] END method={method_name} attempt={attempt_index} "
                    f"status=failed error={last_error_message}"
                )
            continue

        if isophote_count < MIN_SUCCESS_ISOPHOTE_COUNT:
            last_error_message = (
                f"{method_name} extraction produced insufficient profile table "
                f"(isophote_count={isophote_count}, min_required={MIN_SUCCESS_ISOPHOTE_COUNT})"
            )
            last_runtime_info = runtime_info
            last_table_summary = table_summary
            last_fit_config = fit_config
            last_warning_messages = []
            last_validation_payload = None

            fit_retry_log.append(
                {
                    "attempt": attempt_index,
                    "status": "failed",
                    "failure_reason": "insufficient_profile_table",
                    "sma0": attempt_config["sma0"],
                    "astep": attempt_config["astep"],
                    "maxsma": attempt_config["maxsma"],
                    "isophote_count": isophote_count,
                    "error": last_error_message,
                }
            )
            if arguments.verbose:
                print(
                    f"[EXTRACT] END method={method_name} attempt={attempt_index} "
                    f"status=failed error={last_error_message}"
                )
            continue

        successful_profile_table = profile_table
        successful_runtime_info = runtime_info
        successful_runtime_profile_text = runtime_profile_text
        successful_fit_config = fit_config
        successful_table_summary = table_summary
        fit_retry_log.append(
            {
                "attempt": attempt_index,
                "status": "success",
                "sma0": attempt_config["sma0"],
                "astep": attempt_config["astep"],
                "maxsma": attempt_config["maxsma"],
                "isophote_count": isophote_count,
            }
        )
        if arguments.verbose:
            print(f"[EXTRACT] END method={method_name} attempt={attempt_index} status=success")
        break

    if (
        successful_profile_table is None
        or successful_runtime_info is None
        or successful_fit_config is None
        or successful_table_summary is None
    ):
        failure_message = (
            f"{method_name} extraction failed after {MAX_FIT_ATTEMPTS} attempts: "
            f"{last_error_message}"
        )
        write_failed_run_payload(
            run_json_path=paths["run_json"],
            prefix=prefix,
            method_name=method_name,
            config_tag=config_tag,
            fit_config=last_fit_config,
            fit_retry_log=fit_retry_log,
            runtime_info=last_runtime_info,
            table_summary=last_table_summary,
            outputs=paths,
            error_message=failure_message,
            warning_messages=last_warning_messages,
            validation=last_validation_payload,
        )
        log_lines.extend([f"warning={warning}" for warning in last_warning_messages])
        log_lines.append("status=failed")
        log_lines.append(f"error={failure_message}")
        log_lines.append(f"attempt_count={len(fit_retry_log)}")
        log_lines.append(f"fit_retry_log={json.dumps(fit_retry_log, sort_keys=True)}")
        write_method_log(paths["method_log"], log_lines, arguments.save_log)
        raise RuntimeError(failure_message)

    successful_profile_table.write(paths["profile_fits"], overwrite=True)
    successful_profile_table.write(paths["profile_ecsv"], format="ascii.ecsv", overwrite=True)
    paths["runtime_profile"].write_text(successful_runtime_profile_text)

    run_payload = {
        "prefix": prefix,
        "method": method_name,
        "config_tag": config_tag,
        "status": "success",
        "runtime": successful_runtime_info,
        "fit_config": successful_fit_config,
        "table_summary": successful_table_summary,
        "fit_retry_log": fit_retry_log,
        "attempt_count": len(fit_retry_log),
        "max_attempts": MAX_FIT_ATTEMPTS,
        "outputs": {
            "profile_fits": str(paths["profile_fits"]),
            "profile_ecsv": str(paths["profile_ecsv"]),
            "runtime_profile": str(paths["runtime_profile"]),
        },
    }
    dump_json(paths["run_json"], run_payload)

    log_lines.append("status=success")
    log_lines.append(f"runtime_wall_seconds={successful_runtime_info['wall_time_seconds']:.6f}")
    log_lines.append(f"runtime_cpu_seconds={successful_runtime_info['cpu_time_seconds']:.6f}")
    log_lines.append(f"attempt_count={len(fit_retry_log)}")
    log_lines.append(f"fit_retry_log={json.dumps(fit_retry_log, sort_keys=True)}")
    write_method_log(paths["method_log"], log_lines, arguments.save_log)

    return {
        "method": method_name,
        "status": "success",
        "runtime": successful_runtime_info,
        "run_json": str(paths["run_json"]),
        "profile_fits": str(paths["profile_fits"]),
        "profile_ecsv": str(paths["profile_ecsv"]),
        "runtime_profile": str(paths["runtime_profile"]),
        "fit_config": successful_fit_config,
        "fit_retry_log": fit_retry_log,
        "table_summary": run_payload["table_summary"],
        "attempt_count": len(fit_retry_log),
        "skipped_existing": False,
    }


def main() -> None:
    """Run profile extraction workflow."""
    arguments = parse_arguments()

    input_fits = build_input_path(arguments)
    if not input_fits.exists():
        raise FileNotFoundError(f"Input FITS does not exist: {input_fits}")

    output_dir = arguments.output_dir if arguments.output_dir is not None else input_fits.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    image, header = read_mock_image(input_fits)
    base_geometry = infer_initial_geometry(header, image.shape)

    redshift = float(
        arguments.redshift if arguments.redshift is not None else get_header_value(header, "REDSHIFT", DEFAULT_REDSHIFT)
    )
    pixel_scale_arcsec = float(
        arguments.pixel_scale
        if arguments.pixel_scale is not None
        else get_header_value(header, "PIXSCALE", DEFAULT_PIXEL_SCALE_ARCSEC)
    )
    zeropoint_mag = float(
        arguments.zeropoint if arguments.zeropoint is not None else get_header_value(header, "MAGZERO", DEFAULT_ZEROPOINT)
    )
    psf_fwhm_arcsec = float(
        arguments.psf_fwhm if arguments.psf_fwhm is not None else get_header_value(header, "PSFFWHM", np.nan)
    )

    max_sma = (
        float(arguments.maxsma)
        if arguments.maxsma is not None
        else infer_default_maxsma(header, image.shape)
    )

    maxgerr = arguments.maxgerr
    if maxgerr is None:
        maxgerr = 1.0 if base_geometry["eps"] > 0.55 else 0.5

    prefix = f"{arguments.galaxy}_mock{arguments.mock_id}"
    config_tag = sanitize_label(arguments.config_tag)

    methods_to_run = [arguments.method] if arguments.method in {"photutils", "isoster"} else ["photutils", "isoster"]

    manifest_path = output_dir / f"{prefix}_profiles_manifest.json"
    existing_manifest = load_existing_manifest(manifest_path)
    existing_method_runs = existing_manifest.get("method_runs", {}) if isinstance(existing_manifest.get("method_runs", {}), dict) else {}

    method_runs: dict[str, dict[str, Any]] = dict(existing_method_runs)
    for method_name in methods_to_run:
        if arguments.verbose:
            print(f"[EXTRACT] START method={method_name} prefix={prefix}")

        try:
            run_result = run_single_method(
                method_name=method_name,
                image=image,
                base_geometry=base_geometry,
                arguments=arguments,
                redshift=redshift,
                pixel_scale_arcsec=pixel_scale_arcsec,
                zeropoint_mag=zeropoint_mag,
                max_sma=max_sma,
                maxgerr=float(maxgerr),
                output_dir=output_dir,
                prefix=prefix,
                config_tag=config_tag,
            )
            method_runs[method_name] = run_result
            if arguments.verbose:
                status_text = "skipped_existing" if run_result.get("skipped_existing") else "success"
                print(f"[EXTRACT] END method={method_name} status={status_text}")
        except Exception as error:
            error_message = f"{type(error).__name__}: {error}"
            failed_run_path = method_artifact_paths(output_dir, prefix, method_name, config_tag)["run_json"]
            failed_payload = load_existing_manifest(failed_run_path)
            run_warnings = failed_payload.get("warnings", []) if isinstance(failed_payload, dict) else []
            if not isinstance(run_warnings, list):
                run_warnings = []
            method_runs[method_name] = {
                "method": method_name,
                "status": "failed",
                "error": error_message,
                "warnings": run_warnings,
            }
            if arguments.verbose:
                print(f"[EXTRACT] END method={method_name} status=failed error={error_message}")

            if arguments.save_log:
                method_log = method_artifact_paths(output_dir, prefix, method_name, config_tag)["method_log"]
                failure_lines = [
                    f"method={method_name}",
                    f"prefix={prefix}",
                    f"start_time_unix={time.time():.6f}",
                    "status=failed",
                    f"error={error_message}",
                ]
                method_log.write_text("\n".join(failure_lines) + "\n")

    successful_methods = sorted(
        [name for name, payload in method_runs.items() if payload.get("status") == "success"]
    )
    failed_methods = sorted(
        [name for name, payload in method_runs.items() if payload.get("status") == "failed"]
    )
    warning_entries: list[dict[str, Any]] = []
    for method_name, method_payload in method_runs.items():
        method_warnings = method_payload.get("warnings", [])
        if not isinstance(method_warnings, list):
            continue
        for warning_message in method_warnings:
            warning_entries.append(
                {
                    "method": method_name,
                    "severity": "warning",
                    "message": str(warning_message),
                }
            )

    manifest_payload = {
        "prefix": prefix,
        "input_fits": str(input_fits),
        "input_sha256": compute_sha256(input_fits),
        "output_dir": str(output_dir),
        "redshift": redshift,
        "pixel_scale_arcsec": pixel_scale_arcsec,
        "zeropoint_mag": zeropoint_mag,
        "psf_fwhm_arcsec": psf_fwhm_arcsec,
        "cog_subpixels": int(arguments.cog_subpixels),
        "software_versions": collect_software_versions(),
        "run_summary": {
            "requested_methods": methods_to_run,
            "successful_methods": successful_methods,
            "failed_methods": failed_methods,
            "warning_count": len(warning_entries),
        },
        "method_runs": method_runs,
        "warnings": warning_entries,
    }
    dump_json(manifest_path, manifest_payload)

    print(f"Input FITS: {input_fits}")
    print(f"Output directory: {output_dir}")
    print(f"Profile manifest: {manifest_path}")
    print(f"Successful methods: {successful_methods if successful_methods else 'none'}")
    if failed_methods:
        print(f"Failed methods: {failed_methods}")
    if warning_entries:
        print(f"Warnings: {len(warning_entries)}")


if __name__ == "__main__":
    main()
