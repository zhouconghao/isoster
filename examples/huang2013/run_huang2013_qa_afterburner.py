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
    read_mock_image,
    sanitize_label,
    summarize_table,
    write_markdown_report,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for QA afterburner."""
    parser = argparse.ArgumentParser(
        description="Generate Huang2013 QA/report artifacts from saved profile products.",
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
        help="Directory containing profile outputs and destination for QA products.",
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
    parser.add_argument("--photutils-tag", default=None, help="Config tag for photutils artifacts")
    parser.add_argument("--isoster-tag", default=None, help="Config tag for isoster artifacts")

    parser.add_argument("--photutils-profile-fits", type=Path, default=None, help="Explicit photutils profile FITS")
    parser.add_argument("--isoster-profile-fits", type=Path, default=None, help="Explicit isoster profile FITS")
    parser.add_argument("--photutils-run-json", type=Path, default=None, help="Explicit photutils run JSON")
    parser.add_argument("--isoster-run-json", type=Path, default=None, help="Explicit isoster run JSON")

    parser.add_argument("--redshift", type=float, default=None, help="Override redshift")
    parser.add_argument(
        "--pixel-scale",
        type=float,
        default=None,
        help="Override pixel scale [arcsec/pix]. Default: FITS header PIXSCALE.",
    )
    parser.add_argument("--zeropoint", type=float, default=None, help="Override magnitude zeropoint")
    parser.add_argument("--psf-fwhm", type=float, default=None, help="Override PSF FWHM [arcsec]")

    parser.add_argument("--isophote-overlay-step", type=int, default=10, help="Overlay every Nth isophote")
    parser.add_argument("--qa-dpi", type=int, default=180, help="DPI for QA figures")
    parser.add_argument("--skip-comparison", action="store_true", help="Skip cross-method comparison figure")

    return parser.parse_args()


def build_input_path(arguments: argparse.Namespace) -> Path:
    """Resolve the input FITS path."""
    if arguments.input_fits is not None:
        return arguments.input_fits
    return arguments.huang_root / arguments.galaxy / f"{arguments.galaxy}_mock{arguments.mock_id}.fits"


def load_runtime_from_run_json(run_json_path: Path | None) -> dict[str, float]:
    """Load runtime from per-method run JSON or return NaNs."""
    default_runtime = {"wall_time_seconds": float("nan"), "cpu_time_seconds": float("nan")}
    if run_json_path is None or not run_json_path.exists():
        return default_runtime

    payload = json.loads(run_json_path.read_text())
    runtime = payload.get("runtime", {})
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
    if explicit_profile_fits is not None:
        profile_fits = explicit_profile_fits
    else:
        profile_fits = output_dir / f"{prefix}_{method_name}_{tag}_profile.fits"

    if explicit_run_json is not None:
        run_json = explicit_run_json
    else:
        run_json = output_dir / f"{prefix}_{method_name}_{tag}_run.json"
        if not run_json.exists():
            run_json = None

    return profile_fits, run_json


def main() -> None:
    """Run the QA afterburner."""
    arguments = parse_arguments()

    input_fits = build_input_path(arguments)
    if not input_fits.exists():
        raise FileNotFoundError(f"Input FITS does not exist: {input_fits}")

    output_dir = arguments.output_dir if arguments.output_dir is not None else input_fits.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    image, header = read_mock_image(input_fits)

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

    prefix = f"{arguments.galaxy}_mock{arguments.mock_id}"

    shared_tag = sanitize_label(arguments.config_tag)
    photutils_tag = sanitize_label(arguments.photutils_tag) if arguments.photutils_tag is not None else shared_tag
    isoster_tag = sanitize_label(arguments.isoster_tag) if arguments.isoster_tag is not None else shared_tag

    methods_to_render = [arguments.method] if arguments.method in {"photutils", "isoster"} else ["photutils", "isoster"]

    method_tables: dict[str, Table] = {}
    method_models: dict[str, np.ndarray] = {}
    method_runtime: dict[str, dict[str, float]] = {}
    method_outputs: dict[str, dict[str, str]] = {}

    for method_name in methods_to_render:
        method_tag = photutils_tag if method_name == "photutils" else isoster_tag
        explicit_profile = arguments.photutils_profile_fits if method_name == "photutils" else arguments.isoster_profile_fits
        explicit_run_json = arguments.photutils_run_json if method_name == "photutils" else arguments.isoster_run_json

        profile_fits_path, run_json_path = resolve_method_paths(
            method_name=method_name,
            output_dir=output_dir,
            prefix=prefix,
            tag=method_tag,
            explicit_profile_fits=explicit_profile,
            explicit_run_json=explicit_run_json,
        )

        if not profile_fits_path.exists():
            raise FileNotFoundError(f"Missing {method_name} profile FITS: {profile_fits_path}")

        profile_table = Table.read(profile_fits_path)
        model_image = build_model_image(image.shape, profile_table)
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

        method_tables[method_name] = profile_table
        method_models[method_name] = model_image
        method_runtime[method_name] = runtime_info
        method_outputs[method_name] = {
            "profile_fits": str(profile_fits_path),
            "run_json": str(run_json_path) if run_json_path is not None else None,
            "qa_figure": str(qa_path),
        }

    comparison_summary = None
    comparison_qa_path = None

    if (
        not arguments.skip_comparison
        and "photutils" in method_tables
        and "isoster" in method_tables
    ):
        comparison_suffix = f"{photutils_tag}_vs_{isoster_tag}" if photutils_tag != isoster_tag else shared_tag
        comparison_qa_path = output_dir / f"{prefix}_compare_{comparison_suffix}_qa.png"

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

    summary_photutils = summarize_table(method_tables["photutils"]) if "photutils" in method_tables else None
    summary_isoster = summarize_table(method_tables["isoster"]) if "isoster" in method_tables else None

    report_suffix = ""
    if photutils_tag != isoster_tag:
        report_suffix = f"_{photutils_tag}_vs_{isoster_tag}"

    report_path = output_dir / f"{prefix}_report{report_suffix}.md"
    run_metadata = {
        "prefix": prefix,
        "input_fits": str(input_fits),
        "input_sha256": compute_sha256(input_fits),
        "redshift": redshift,
        "pixel_scale_arcsec": pixel_scale_arcsec,
        "zeropoint_mag": zeropoint_mag,
        "psf_fwhm_arcsec": psf_fwhm_arcsec,
        "software_versions": collect_software_versions(),
        "method_runs": {
            method_name: {
                "runtime": method_runtime[method_name],
                "profile_fits": method_outputs[method_name]["profile_fits"],
                "run_json": method_outputs[method_name]["run_json"],
                "qa_figure": method_outputs[method_name]["qa_figure"],
            }
            for method_name in method_outputs
        },
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

    manifest_suffix = ""
    if photutils_tag != isoster_tag:
        manifest_suffix = f"_{photutils_tag}_vs_{isoster_tag}"

    manifest_path = output_dir / f"{prefix}_qa_manifest{manifest_suffix}.json"
    manifest_payload = {
        "prefix": prefix,
        "input_fits": str(input_fits),
        "output_dir": str(output_dir),
        "report": str(report_path),
        "comparison_qa": str(comparison_qa_path) if comparison_qa_path is not None else None,
        "comparison_summary": comparison_summary,
        "method_outputs": method_outputs,
        "run_metadata": run_metadata,
    }
    dump_json(manifest_path, manifest_payload)

    print(f"Input FITS: {input_fits}")
    print(f"Output directory: {output_dir}")
    print(f"QA manifest: {manifest_path}")
    if comparison_qa_path is not None:
        print(f"Comparison QA: {comparison_qa_path}")


if __name__ == "__main__":
    main()
