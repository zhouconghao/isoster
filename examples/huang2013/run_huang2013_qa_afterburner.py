#!/usr/bin/env python3
"""Generate Huang2013 QA/report artifacts from saved profile extraction outputs.

This script is an afterburner stage. It does not run fitting; it reads saved
profile FITS/JSON products and creates:
- per-method QA figures
- cross-method comparison QA figure
- markdown report
- QA manifest JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from astropy.table import Table

from huang2013_campaign_contract import (
    build_case_prefix,
    build_case_output_dir,
    build_method_artifact_paths,
    build_method_stem,
    build_profiles_manifest_path,
    build_qa_manifest_path,
    load_method_statuses_from_profiles_manifest,
    read_json_dict_if_exists,
)
from run_huang2013_real_mock_demo import (
    DEFAULT_CONFIG_TAG,
    DEFAULT_PIXEL_SCALE_ARCSEC,
    DEFAULT_REDSHIFT,
    DEFAULT_ZEROPOINT,
    build_comparison_qa_figure,
    build_method_qa_figure,
    build_model_image,
    collect_software_versions,
    compute_sha256,
    dump_json,
    get_header_value,
    harmonize_method_cog_columns,
    read_mock_image,
    sanitize_label,
    summarize_table,
    write_markdown_report,
)

ISOPHOTE_COUNT_MISMATCH_WARNING_THRESHOLD = 10


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for QA afterburner."""
    parser = argparse.ArgumentParser(
        description="Generate Huang2013 QA/report artifacts from saved profile products.",
    )

    parser.add_argument(
        "--galaxy", required=True, help="Galaxy folder/name, e.g., IC2597"
    )
    parser.add_argument(
        "--mock-id", type=int, default=1, help="Mock ID suffix in FITS name"
    )
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
        help="Directory containing profile outputs and destination for QA products. Default: <huang-root>/<GALAXY>/<TEST>/.",
    )

    parser.add_argument(
        "--method",
        choices=["photutils", "isoster", "both"],
        default="both",
        help="Methods to render in QA.",
    )
    parser.add_argument(
        "--config-tag",
        default=DEFAULT_CONFIG_TAG,
        help="Shared config tag used for both methods unless method-specific tag is set.",
    )
    parser.add_argument(
        "--photutils-tag", default=None, help="Config tag for photutils artifacts"
    )
    parser.add_argument(
        "--isoster-tag", default=None, help="Config tag for isoster artifacts"
    )

    parser.add_argument(
        "--photutils-profile-fits",
        type=Path,
        default=None,
        help="Explicit photutils profile FITS",
    )
    parser.add_argument(
        "--isoster-profile-fits",
        type=Path,
        default=None,
        help="Explicit isoster profile FITS",
    )
    parser.add_argument(
        "--photutils-run-json",
        type=Path,
        default=None,
        help="Explicit photutils run JSON",
    )
    parser.add_argument(
        "--isoster-run-json", type=Path, default=None, help="Explicit isoster run JSON"
    )

    parser.add_argument(
        "--profiles-manifest",
        type=Path,
        default=None,
        help="Extraction manifest path. Default: <prefix>_profiles_manifest.json in output dir.",
    )
    parser.add_argument(
        "--ignore-extraction-status",
        action="store_true",
        help="Ignore extraction manifest method statuses when deciding QA eligibility.",
    )

    parser.add_argument(
        "--redshift", type=float, default=None, help="Override redshift"
    )
    parser.add_argument(
        "--pixel-scale",
        type=float,
        default=None,
        help="Override pixel scale [arcsec/pix]. Default: FITS header PIXSCALE.",
    )
    parser.add_argument(
        "--zeropoint", type=float, default=None, help="Override magnitude zeropoint"
    )
    parser.add_argument(
        "--psf-fwhm", type=float, default=None, help="Override PSF FWHM [arcsec]"
    )

    parser.add_argument(
        "--isophote-overlay-step",
        type=int,
        default=10,
        help="Overlay every Nth isophote",
    )
    parser.add_argument("--qa-dpi", type=int, default=180, help="DPI for QA figures")
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip cross-method comparison figure",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Emit QA stage progress logs."
    )

    return parser.parse_args()


def build_input_path(arguments: argparse.Namespace) -> Path:
    """Resolve the input FITS path."""
    if arguments.input_fits is not None:
        return arguments.input_fits
    return (
        arguments.huang_root
        / arguments.galaxy
        / f"{arguments.galaxy}_mock{arguments.mock_id}.fits"
    )


def load_runtime_from_run_json(run_json_path: Path | None) -> dict[str, float]:
    """Load runtime from per-method run JSON or return NaNs."""
    default_runtime = {
        "wall_time_seconds": float("nan"),
        "cpu_time_seconds": float("nan"),
    }
    if run_json_path is None or not run_json_path.exists():
        return default_runtime

    try:
        payload = json.loads(run_json_path.read_text())
    except json.JSONDecodeError:
        return default_runtime

    runtime = payload.get("runtime", {})
    if not isinstance(runtime, dict):
        runtime = {}
    return {
        "wall_time_seconds": float(runtime.get("wall_time_seconds", float("nan"))),
        "cpu_time_seconds": float(runtime.get("cpu_time_seconds", float("nan"))),
    }


def resolve_method_paths(
    method_name: str,
    output_dir: Path,
    prefix: str,
    tag: str,
    explicit_profile_fits: Path | None,
    explicit_run_json: Path | None,
) -> tuple[Path, Path | None]:
    """Resolve profile FITS and run JSON path for a method."""
    method_paths = build_method_artifact_paths(output_dir, prefix, method_name, tag)
    if explicit_profile_fits is not None:
        profile_fits = explicit_profile_fits
    else:
        profile_fits = method_paths["profile_fits"]

    if explicit_run_json is not None:
        run_json = explicit_run_json
    else:
        run_json = method_paths["run_json"]
        if not run_json.exists():
            run_json = None

    return profile_fits, run_json


def load_extraction_method_statuses(manifest_path: Path | None) -> dict[str, str]:
    """Load extraction method status mapping from profiles manifest."""
    return load_method_statuses_from_profiles_manifest(manifest_path)


def validate_profile_table_for_qa(profile_table: Table) -> tuple[bool, str | None]:
    """Validate profile table minimum requirements for QA rendering."""
    if len(profile_table) == 0:
        return False, "empty_profile_table"

    required_columns = {
        "x_kpc_quarter",
        "sma",
        "stop_code",
        "intens",
        "eps",
        "pa",
        "x0",
        "y0",
    }
    missing_columns = sorted(required_columns - set(profile_table.colnames))
    if missing_columns:
        missing_text = ",".join(missing_columns)
        return False, f"missing_required_columns:{missing_text}"

    return True, None


def read_json_if_exists(file_path: Path | None) -> dict | None:
    """Read JSON dictionary payload when available and valid."""
    return read_json_dict_if_exists(file_path)


def build_comparison_isophote_count_warning(
    summary_photutils: dict | None,
    summary_isoster: dict | None,
) -> dict | None:
    """Build warning payload for large inter-method isophote count differences."""
    if summary_photutils is None or summary_isoster is None:
        return None

    photutils_count = int(summary_photutils.get("isophote_count", 0))
    isoster_count = int(summary_isoster.get("isophote_count", 0))
    absolute_difference = abs(photutils_count - isoster_count)
    if absolute_difference < ISOPHOTE_COUNT_MISMATCH_WARNING_THRESHOLD:
        return None

    return {
        "code": "isophote_count_mismatch",
        "severity": "warning",
        "message": (
            "Large method isophote-count difference detected: "
            f"|photutils({photutils_count}) - isoster({isoster_count})| = {absolute_difference} "
            f">= {ISOPHOTE_COUNT_MISMATCH_WARNING_THRESHOLD}"
        ),
        "photutils_isophote_count": photutils_count,
        "isoster_isophote_count": isoster_count,
        "absolute_difference": absolute_difference,
        "threshold": ISOPHOTE_COUNT_MISMATCH_WARNING_THRESHOLD,
    }


def collect_artifact_availability_warnings(
    artifact_entries: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Collect missing/empty artifact warnings from expected artifact records."""
    warnings: list[dict[str, str]] = []
    seen_entries: set[tuple[str, str]] = set()

    for artifact_entry in artifact_entries:
        artifact_role = artifact_entry["artifact_role"]
        artifact_path = Path(artifact_entry["artifact_path"])
        entry_key = (artifact_role, str(artifact_path))
        if entry_key in seen_entries:
            continue
        seen_entries.add(entry_key)

        if not artifact_path.exists():
            warnings.append(
                {
                    "code": "artifact_missing",
                    "severity": "warning",
                    "artifact_role": artifact_role,
                    "artifact_path": str(artifact_path),
                    "message": f"Expected artifact missing: {artifact_role}",
                }
            )
            continue

        if artifact_path.is_file() and artifact_path.stat().st_size == 0:
            warnings.append(
                {
                    "code": "artifact_empty",
                    "severity": "warning",
                    "artifact_role": artifact_role,
                    "artifact_path": str(artifact_path),
                    "message": f"Expected artifact exists but is empty: {artifact_role}",
                }
            )

    return warnings


def prepare_case_output_dir(output_dir: Path, verbose: bool, prefix: str) -> None:
    """Ensure case output directory exists and emit create/skip telemetry."""
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(
                f"Case output path exists but is not a directory: {output_dir}"
            )
        if verbose:
            print(
                f"[QA] SKIP stage={prefix}:mkdir reason=output_dir_exists path={output_dir}"
            )
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"[QA] END stage={prefix}:mkdir status=created path={output_dir}")


def main() -> None:
    """Run the QA afterburner."""
    arguments = parse_arguments()

    input_fits = build_input_path(arguments)
    if not input_fits.exists():
        raise FileNotFoundError(f"Input FITS does not exist: {input_fits}")

    prefix = build_case_prefix(arguments.galaxy, arguments.mock_id)
    default_output_dir = build_case_output_dir(
        arguments.huang_root, arguments.galaxy, arguments.mock_id
    )
    output_dir = (
        arguments.output_dir if arguments.output_dir is not None else default_output_dir
    )
    prepare_case_output_dir(output_dir, arguments.verbose, prefix)

    image, header = read_mock_image(input_fits)

    redshift = float(
        arguments.redshift
        if arguments.redshift is not None
        else get_header_value(header, "REDSHIFT", DEFAULT_REDSHIFT)
    )
    pixel_scale_arcsec = float(
        arguments.pixel_scale
        if arguments.pixel_scale is not None
        else get_header_value(header, "PIXSCALE", DEFAULT_PIXEL_SCALE_ARCSEC)
    )
    zeropoint_mag = float(
        arguments.zeropoint
        if arguments.zeropoint is not None
        else get_header_value(header, "MAGZERO", DEFAULT_ZEROPOINT)
    )
    psf_fwhm_arcsec = float(
        arguments.psf_fwhm
        if arguments.psf_fwhm is not None
        else get_header_value(header, "PSFFWHM", np.nan)
    )

    shared_tag = sanitize_label(arguments.config_tag)
    photutils_tag = (
        sanitize_label(arguments.photutils_tag)
        if arguments.photutils_tag is not None
        else shared_tag
    )
    isoster_tag = (
        sanitize_label(arguments.isoster_tag)
        if arguments.isoster_tag is not None
        else shared_tag
    )

    inferred_profiles_manifest = build_profiles_manifest_path(output_dir, prefix)
    profiles_manifest_path = (
        arguments.profiles_manifest
        if arguments.profiles_manifest is not None
        else inferred_profiles_manifest
    )
    extraction_status_by_method = load_extraction_method_statuses(
        profiles_manifest_path
    )

    methods_to_render = (
        [arguments.method]
        if arguments.method in {"photutils", "isoster"}
        else ["photutils", "isoster"]
    )

    method_tables: dict[str, Table] = {}
    method_models: dict[str, np.ndarray] = {}
    method_runtime: dict[str, dict[str, float]] = {}
    method_outputs: dict[str, dict[str, str | None]] = {}
    method_skips: list[dict[str, str]] = []
    method_failures: list[dict[str, str]] = []
    expected_artifact_entries: list[dict[str, str]] = [
        {
            "artifact_role": "profiles_manifest",
            "artifact_path": str(profiles_manifest_path),
        }
    ]

    for method_name in methods_to_render:
        extraction_status = extraction_status_by_method.get(method_name)
        if (
            not arguments.ignore_extraction_status
            and extraction_status is not None
            and extraction_status != "success"
        ):
            method_skips.append(
                {
                    "method": method_name,
                    "reason": f"extraction_status_{extraction_status}",
                    "profiles_manifest": str(profiles_manifest_path),
                }
            )
            if arguments.verbose:
                print(
                    f"[QA] SKIP method={method_name} reason=extraction_status_{extraction_status}"
                )
            continue

        method_tag = photutils_tag if method_name == "photutils" else isoster_tag
        method_stem = build_method_stem(prefix, method_name, method_tag)
        method_paths = build_method_artifact_paths(
            output_dir, prefix, method_name, method_tag
        )
        expected_artifact_entries.extend(
            [
                {
                    "artifact_role": f"{method_name}_profile_fits",
                    "artifact_path": str(method_paths["profile_fits"]),
                },
                {
                    "artifact_role": f"{method_name}_profile_ecsv",
                    "artifact_path": str(method_paths["profile_ecsv"]),
                },
                {
                    "artifact_role": f"{method_name}_runtime_profile",
                    "artifact_path": str(method_paths["runtime_profile"]),
                },
                {
                    "artifact_role": f"{method_name}_run_json",
                    "artifact_path": str(method_paths["run_json"]),
                },
                {
                    "artifact_role": f"{method_name}_qa_figure",
                    "artifact_path": str(output_dir / f"{method_stem}_qa.png"),
                },
            ]
        )
        explicit_profile = (
            arguments.photutils_profile_fits
            if method_name == "photutils"
            else arguments.isoster_profile_fits
        )
        explicit_run_json = (
            arguments.photutils_run_json
            if method_name == "photutils"
            else arguments.isoster_run_json
        )

        profile_fits_path, run_json_path = resolve_method_paths(
            method_name=method_name,
            output_dir=output_dir,
            prefix=prefix,
            tag=method_tag,
            explicit_profile_fits=explicit_profile,
            explicit_run_json=explicit_run_json,
        )

        if not profile_fits_path.exists():
            method_skips.append(
                {
                    "method": method_name,
                    "reason": "missing_profile_fits",
                    "profile_fits": str(profile_fits_path),
                }
            )
            if arguments.verbose:
                print(f"[QA] SKIP method={method_name} reason=missing_profile_fits")
            continue

        if arguments.verbose:
            print(f"[QA] START method={method_name}")

        try:
            profile_table = Table.read(profile_fits_path)
            harmonize_method_cog_columns(profile_table, method_name=method_name)
            is_valid_table, invalid_reason = validate_profile_table_for_qa(
                profile_table
            )
            if not is_valid_table:
                method_skips.append(
                    {
                        "method": method_name,
                        "reason": str(invalid_reason),
                        "profile_fits": str(profile_fits_path),
                    }
                )
                if arguments.verbose:
                    print(f"[QA] SKIP method={method_name} reason={invalid_reason}")
                continue

            model_image = build_model_image(
                image.shape,
                profile_table,
                method_name=method_name,
            )
            runtime_info = load_runtime_from_run_json(run_json_path)

            qa_path = output_dir / f"{prefix}_{method_name}_{method_tag}_qa.png"
            build_method_qa_figure(
                image=image,
                profile_table=profile_table,
                model_image=model_image,
                output_path=qa_path,
                method_name=method_name,
                galaxy_name=arguments.galaxy,
                mock_id=arguments.mock_id,
                pixel_scale_arcsec=pixel_scale_arcsec,
                redshift=redshift,
                runtime_metadata=runtime_info,
                overlay_step=arguments.isophote_overlay_step,
                dpi=arguments.qa_dpi,
            )
        except Exception as error:
            method_failures.append(
                {
                    "method": method_name,
                    "reason": "qa_generation_failed",
                    "error": f"{type(error).__name__}: {error}",
                    "profile_fits": str(profile_fits_path),
                }
            )
            if arguments.verbose:
                print(
                    f"[QA] END method={method_name} status=failed error={type(error).__name__}: {error}"
                )
            continue

        method_tables[method_name] = profile_table
        method_models[method_name] = model_image
        method_runtime[method_name] = runtime_info
        method_outputs[method_name] = {
            "status": "success",
            "profile_fits": str(profile_fits_path),
            "run_json": str(run_json_path) if run_json_path is not None else None,
            "qa_figure": str(qa_path),
        }

        if arguments.verbose:
            print(f"[QA] END method={method_name} status=success")

    comparison_summary = None
    comparison_qa_path = None

    if (
        not arguments.skip_comparison
        and "photutils" in method_tables
        and "isoster" in method_tables
    ):
        comparison_suffix = (
            f"{photutils_tag}_vs_{isoster_tag}"
            if photutils_tag != isoster_tag
            else shared_tag
        )
        comparison_qa_path = output_dir / f"{prefix}_compare_{comparison_suffix}_qa.png"

        if arguments.verbose:
            print("[QA] START comparison")

        comparison_summary = build_comparison_qa_figure(
            image=image,
            photutils_table=method_tables["photutils"],
            isoster_table=method_tables["isoster"],
            photutils_model=method_models["photutils"],
            isoster_model=method_models["isoster"],
            output_path=comparison_qa_path,
            galaxy_name=arguments.galaxy,
            mock_id=arguments.mock_id,
            pixel_scale_arcsec=pixel_scale_arcsec,
            redshift=redshift,
            runtime_photutils=method_runtime["photutils"],
            runtime_isoster=method_runtime["isoster"],
            overlay_step=arguments.isophote_overlay_step,
            dpi=arguments.qa_dpi,
        )

        if arguments.verbose:
            print("[QA] END comparison status=success")

    summary_photutils = (
        summarize_table(method_tables["photutils"])
        if "photutils" in method_tables
        else None
    )
    summary_isoster = (
        summarize_table(method_tables["isoster"])
        if "isoster" in method_tables
        else None
    )
    warning_entries: list[dict] = []

    comparison_warning = build_comparison_isophote_count_warning(
        summary_photutils, summary_isoster
    )
    if comparison_warning is not None:
        warning_entries.append(comparison_warning)
        if arguments.verbose:
            print(
                "[QA] WARN "
                f"code={comparison_warning['code']} diff={comparison_warning['absolute_difference']}"
            )

    report_suffix = ""
    if photutils_tag != isoster_tag:
        report_suffix = f"_{photutils_tag}_vs_{isoster_tag}"

    report_path = output_dir / f"{prefix}_report{report_suffix}.md"
    manifest_suffix = ""
    if photutils_tag != isoster_tag:
        manifest_suffix = f"_{photutils_tag}_vs_{isoster_tag}"
    manifest_path = build_qa_manifest_path(output_dir, prefix, manifest_suffix)

    for method_name, method_payload in method_outputs.items():
        for field_name in ["profile_fits", "run_json", "qa_figure"]:
            field_value = method_payload.get(field_name)
            if isinstance(field_value, str):
                expected_artifact_entries.append(
                    {
                        "artifact_role": f"{method_name}_{field_name}",
                        "artifact_path": field_value,
                    }
                )

        run_json_path_text = method_payload.get("run_json")
        run_json_payload = (
            read_json_if_exists(Path(run_json_path_text))
            if isinstance(run_json_path_text, str)
            else None
        )
        run_outputs = (
            run_json_payload.get("outputs", {})
            if isinstance(run_json_payload, dict)
            else {}
        )
        if isinstance(run_outputs, dict):
            for field_name in ["profile_fits", "profile_ecsv", "runtime_profile"]:
                field_value = run_outputs.get(field_name)
                if isinstance(field_value, str):
                    expected_artifact_entries.append(
                        {
                            "artifact_role": f"{method_name}_{field_name}",
                            "artifact_path": field_value,
                        }
                    )
        run_warnings = (
            run_json_payload.get("warnings", [])
            if isinstance(run_json_payload, dict)
            else []
        )
        if isinstance(run_warnings, list):
            for warning_text in run_warnings:
                warning_entries.append(
                    {
                        "code": "extraction_warning",
                        "severity": "warning",
                        "method": method_name,
                        "message": str(warning_text),
                    }
                )

    if comparison_qa_path is not None:
        expected_artifact_entries.append(
            {
                "artifact_role": "comparison_qa",
                "artifact_path": str(comparison_qa_path),
            }
        )
    elif not arguments.skip_comparison and set(methods_to_render) == {
        "photutils",
        "isoster",
    }:
        comparison_suffix = (
            f"{photutils_tag}_vs_{isoster_tag}"
            if photutils_tag != isoster_tag
            else shared_tag
        )
        expected_artifact_entries.append(
            {
                "artifact_role": "comparison_qa",
                "artifact_path": str(
                    output_dir / f"{prefix}_compare_{comparison_suffix}_qa.png"
                ),
            }
        )

    warning_entries.extend(
        collect_artifact_availability_warnings(expected_artifact_entries)
    )
    run_metadata = {
        "prefix": prefix,
        "input_fits": str(input_fits),
        "input_sha256": compute_sha256(input_fits),
        "redshift": redshift,
        "pixel_scale_arcsec": pixel_scale_arcsec,
        "zeropoint_mag": zeropoint_mag,
        "psf_fwhm_arcsec": psf_fwhm_arcsec,
        "software_versions": collect_software_versions(),
        "profiles_manifest": str(profiles_manifest_path),
        "extraction_status_by_method": extraction_status_by_method,
        "method_runs": {
            method_name: {
                "runtime": method_runtime[method_name],
                "profile_fits": method_outputs[method_name]["profile_fits"],
                "run_json": method_outputs[method_name]["run_json"],
                "qa_figure": method_outputs[method_name]["qa_figure"],
            }
            for method_name in method_outputs
        },
        "method_skips": method_skips,
        "method_failures": method_failures,
        "warnings": warning_entries,
    }

    write_markdown_report(
        report_path=report_path,
        prefix=prefix,
        input_path=input_fits,
        run_metadata=run_metadata,
        summary_photutils=summary_photutils,
        summary_isoster=summary_isoster,
        comparison_summary=comparison_summary,
    )

    manifest_payload = {
        "prefix": prefix,
        "input_fits": str(input_fits),
        "output_dir": str(output_dir),
        "report": str(report_path),
        "comparison_qa": str(comparison_qa_path)
        if comparison_qa_path is not None
        else None,
        "comparison_summary": comparison_summary,
        "method_outputs": method_outputs,
        "method_skips": method_skips,
        "method_failures": method_failures,
        "warnings": warning_entries,
        "method_summary": {
            "requested_methods": methods_to_render,
            "successful_methods": sorted(method_outputs.keys()),
            "skipped_methods": sorted([entry["method"] for entry in method_skips]),
            "failed_methods": sorted([entry["method"] for entry in method_failures]),
        },
        "run_metadata": run_metadata,
    }
    dump_json(manifest_path, manifest_payload)

    post_write_warnings = collect_artifact_availability_warnings(
        [
            {
                "artifact_role": "report_markdown",
                "artifact_path": str(report_path),
            },
            {
                "artifact_role": "qa_manifest",
                "artifact_path": str(manifest_path),
            },
        ]
    )
    if post_write_warnings:
        warning_entries.extend(post_write_warnings)
        run_metadata["warnings"] = warning_entries
        manifest_payload["warnings"] = warning_entries
        manifest_payload["run_metadata"] = run_metadata
        write_markdown_report(
            report_path=report_path,
            prefix=prefix,
            input_path=input_fits,
            run_metadata=run_metadata,
            summary_photutils=summary_photutils,
            summary_isoster=summary_isoster,
            comparison_summary=comparison_summary,
        )
        dump_json(manifest_path, manifest_payload)

    print(f"Input FITS: {input_fits}")
    print(f"Output directory: {output_dir}")
    print(f"QA manifest: {manifest_path}")
    successful_methods = sorted(method_outputs.keys())
    print(
        f"Successful QA methods: {successful_methods if successful_methods else 'none'}"
    )
    if method_skips:
        print(f"Skipped methods: {[entry['method'] for entry in method_skips]}")
    if method_failures:
        print(f"Failed methods: {[entry['method'] for entry in method_failures]}")
    if comparison_qa_path is not None:
        print(f"Comparison QA: {comparison_qa_path}")
    if warning_entries:
        print(f"Warnings: {len(warning_entries)}")


if __name__ == "__main__":
    main()
