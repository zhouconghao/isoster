#!/usr/bin/env python3
"""Run reproducible Huang2013 mock comparisons for photutils and isoster.

This script is designed for external Huang2013 mock images stored under:
    /Users/mac/work/hsc/huang2013/<GALAXY>/<GALAXY>_mock<id>.fits

It can run photutils and isoster independently, save method-specific profile
products, create QA figures, and optionally build a cross-method comparison
figure from saved artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from astropy.table import Table
from huang2013_campaign_contract import build_case_output_dir, build_case_prefix
from huang2013_shared import (
    DEFAULT_CONFIG_TAG,
    DEFAULT_HUANG_ROOT,
    DEFAULT_PIXEL_SCALE_ARCSEC,
    DEFAULT_REDSHIFT,
    DEFAULT_ZEROPOINT,
    build_comparison_qa_figure,
    build_input_path,
    build_method_qa_figure,
    build_model_image,
    collect_software_versions,
    compute_sha256,
    dump_json,
    get_header_value,
    harmonize_method_cog_columns,
    infer_initial_geometry,
    prepare_profile_table,
    read_mock_image,
    run_isoster_fit,
    run_photutils_fit,
    run_with_runtime_profile,
    sanitize_label,
    summarize_table,
    write_markdown_report,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run IC2597-style Huang2013 real-mock comparison workflow.",
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
        default=DEFAULT_HUANG_ROOT,
        help="Root directory containing galaxy subfolders.",
    )
    parser.add_argument(
        "--input-fits",
        type=Path,
        default=None,
        help="Explicit FITS file. Overrides --huang-root/--galaxy/--mock-id.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <huang-root>/<GALAXY>/<TEST>/.",
    )
    parser.add_argument(
        "--method",
        choices=["photutils", "isoster", "both"],
        default="both",
        help="Method to run in this invocation.",
    )
    parser.add_argument(
        "--config-tag",
        default=DEFAULT_CONFIG_TAG,
        help="Tag used in artifact names to distinguish configurations.",
    )

    parser.add_argument(
        "--redshift", type=float, default=None, help="Override redshift"
    )
    parser.add_argument("--pixel-scale", type=float, default=None, help="Arcsec/pixel")
    parser.add_argument(
        "--zeropoint", type=float, default=None, help="Magnitude zeropoint"
    )
    parser.add_argument(
        "--psf-fwhm", type=float, default=None, help="PSF FWHM in arcsec"
    )

    parser.add_argument(
        "--sma0", type=float, default=None, help="Initial SMA in pixels"
    )
    parser.add_argument(
        "--minsma", type=float, default=1.0, help="Minimum fitted SMA in pixels"
    )
    parser.add_argument(
        "--maxsma", type=float, default=None, help="Maximum fitted SMA in pixels"
    )
    parser.add_argument("--astep", type=float, default=0.1, help="SMA growth step")
    parser.add_argument(
        "--maxgerr", type=float, default=None, help="Gradient error threshold"
    )

    parser.add_argument(
        "--photutils-nclip", type=int, default=2, help="photutils nclip"
    )
    parser.add_argument(
        "--photutils-sclip", type=float, default=3.0, help="photutils sclip"
    )
    parser.add_argument(
        "--photutils-integrmode",
        default="bilinear",
        choices=["bilinear", "nearest_neighbor", "mean"],
        help="photutils integration mode",
    )

    parser.add_argument(
        "--use-eccentric-anomaly",
        action="store_true",
        help="Enable isoster eccentric-anomaly sampling.",
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
        default=32,
        help="Subpixel factor for true CoG aperture photometry.",
    )
    parser.add_argument(
        "--isophote-overlay-step",
        type=int,
        default=10,
        help="Overlay every Nth isophote in QA panels.",
    )
    parser.add_argument("--qa-dpi", type=int, default=180, help="DPI for QA figures")
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip the cross-method comparison figure.",
    )

    parser.add_argument(
        "--photutils-profile-fits",
        type=Path,
        default=None,
        help="Existing photutils profile FITS for comparison-only use.",
    )
    parser.add_argument(
        "--isoster-profile-fits",
        type=Path,
        default=None,
        help="Existing isoster profile FITS for comparison-only use.",
    )

    return parser.parse_args()


def main() -> None:
    """Main entrypoint for the Huang2013 real-mock demo workflow."""
    args = parse_arguments()

    input_fits = build_input_path(args)
    if not input_fits.exists():
        raise FileNotFoundError(f"Input FITS does not exist: {input_fits}")

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else build_case_output_dir(args.huang_root, args.galaxy, args.mock_id)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    image, header = read_mock_image(input_fits)

    inferred = infer_initial_geometry(header, image.shape)
    redshift = float(
        args.redshift
        if args.redshift is not None
        else get_header_value(header, "REDSHIFT", DEFAULT_REDSHIFT)
    )
    pixel_scale_arcsec = float(
        args.pixel_scale
        if args.pixel_scale is not None
        else get_header_value(header, "PIXSCALE", DEFAULT_PIXEL_SCALE_ARCSEC)
    )
    zeropoint_mag = float(
        args.zeropoint
        if args.zeropoint is not None
        else get_header_value(header, "MAGZERO", DEFAULT_ZEROPOINT)
    )
    psf_fwhm_arcsec = float(
        args.psf_fwhm
        if args.psf_fwhm is not None
        else get_header_value(header, "PSFFWHM", np.nan)
    )

    max_sma = args.maxsma
    if max_sma is None:
        max_sma = 0.48 * min(image.shape)

    maxgerr = args.maxgerr
    if maxgerr is None:
        maxgerr = 1.0 if inferred["eps"] > 0.55 else 0.5

    config_tag = sanitize_label(args.config_tag)
    prefix = build_case_prefix(args.galaxy, args.mock_id)

    run_metadata: dict[str, Any] = {
        "prefix": prefix,
        "input_fits": str(input_fits),
        "input_sha256": compute_sha256(input_fits),
        "redshift": redshift,
        "pixel_scale_arcsec": pixel_scale_arcsec,
        "zeropoint_mag": zeropoint_mag,
        "psf_fwhm_arcsec": psf_fwhm_arcsec,
        "cog_subpixels": args.cog_subpixels,
        "software_versions": collect_software_versions(),
        "method_runs": {},
    }

    method_profile_paths: dict[str, Path] = {}
    method_tables: dict[str, Table] = {}
    method_runtime: dict[str, dict[str, Any]] = {}

    methods_to_run = (
        [args.method]
        if args.method in {"photutils", "isoster"}
        else ["photutils", "isoster"]
    )

    for method_name in methods_to_run:
        stem = f"{prefix}_{method_name}_{config_tag}"
        profile_fits_path = output_dir / f"{stem}_profile.fits"
        profile_ecsv_path = output_dir / f"{stem}_profile.ecsv"
        runtime_profile_path = output_dir / f"{stem}_runtime-profile.txt"
        method_json_path = output_dir / f"{stem}_run.json"
        method_qa_path = output_dir / f"{stem}_qa.png"

        if method_name == "photutils":
            photutils_config = {
                "x0": inferred["x0"],
                "y0": inferred["y0"],
                "eps": inferred["eps"],
                "pa_deg": inferred["pa_deg"],
                "sma0": float(args.sma0 if args.sma0 is not None else inferred["sma0"]),
                "minsma": float(args.minsma),
                "maxsma": float(max_sma),
                "astep": float(args.astep),
                "maxgerr": float(maxgerr),
                "nclip": int(args.photutils_nclip),
                "sclip": float(args.photutils_sclip),
                "integrmode": args.photutils_integrmode,
            }
            isophotes, runtime_info, profile_text = run_with_runtime_profile(
                run_photutils_fit,
                image,
                photutils_config,
            )
            validated_config = photutils_config
        else:
            isoster_config = {
                "x0": inferred["x0"],
                "y0": inferred["y0"],
                "eps": inferred["eps"],
                "pa": float(np.deg2rad(inferred["pa_deg"])),
                "sma0": float(args.sma0 if args.sma0 is not None else inferred["sma0"]),
                "minsma": float(args.minsma),
                "maxsma": float(max_sma),
                "astep": float(args.astep),
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
                "use_eccentric_anomaly": bool(args.use_eccentric_anomaly),
                "simultaneous_harmonics": False,
                "harmonic_orders": [3, 4],
                "permissive_geometry": False,
                "use_central_regularization": False,
            }

            if args.isoster_config_json is not None:
                user_overrides = json.loads(args.isoster_config_json.read_text())
                if "pa_deg" in user_overrides and "pa" not in user_overrides:
                    user_overrides["pa"] = float(
                        np.deg2rad(float(user_overrides.pop("pa_deg")))
                    )
                isoster_config.update(user_overrides)

            fit_output, runtime_info, profile_text = run_with_runtime_profile(
                run_isoster_fit,
                image,
                isoster_config,
            )
            isophotes = fit_output[0]
            validated_config = fit_output[1]

        profile_table = prepare_profile_table(
            isophotes=isophotes,
            image=image,
            redshift=redshift,
            pixel_scale_arcsec=pixel_scale_arcsec,
            zeropoint_mag=zeropoint_mag,
            cog_subpixels=args.cog_subpixels,
            method_name=method_name,
        )

        model_image = build_model_image(
            image.shape,
            profile_table,
            method_name=method_name,
        )
        build_method_qa_figure(
            image=image,
            profile_table=profile_table,
            model_image=model_image,
            output_path=method_qa_path,
            method_name=method_name,
            galaxy_name=args.galaxy,
            mock_id=args.mock_id,
            pixel_scale_arcsec=pixel_scale_arcsec,
            redshift=redshift,
            runtime_metadata=runtime_info,
            overlay_step=args.isophote_overlay_step,
            dpi=args.qa_dpi,
        )

        profile_table.write(profile_fits_path, overwrite=True)
        profile_table.write(profile_ecsv_path, format="ascii.ecsv", overwrite=True)
        runtime_profile_path.write_text(profile_text)

        method_payload = {
            "prefix": prefix,
            "method": method_name,
            "config_tag": config_tag,
            "input_fits": str(input_fits),
            "runtime": runtime_info,
            "fit_config": validated_config,
            "table_summary": summarize_table(profile_table),
            "outputs": {
                "profile_fits": str(profile_fits_path),
                "profile_ecsv": str(profile_ecsv_path),
                "runtime_profile": str(runtime_profile_path),
                "qa_figure": str(method_qa_path),
            },
        }
        dump_json(method_json_path, method_payload)

        run_metadata["method_runs"][method_name] = {
            "runtime": runtime_info,
            "run_json": str(method_json_path),
            "qa_figure": str(method_qa_path),
            "profile_fits": str(profile_fits_path),
            "profile_ecsv": str(profile_ecsv_path),
            "runtime_profile": str(runtime_profile_path),
            "fit_config": validated_config,
        }

        method_profile_paths[method_name] = profile_fits_path
        method_tables[method_name] = profile_table
        method_runtime[method_name] = runtime_info

    if (
        "photutils" not in method_profile_paths
        and args.photutils_profile_fits is not None
    ):
        method_profile_paths["photutils"] = args.photutils_profile_fits
        method_tables["photutils"] = Table.read(args.photutils_profile_fits)
        harmonize_method_cog_columns(
            method_tables["photutils"], method_name="photutils"
        )
        method_runtime["photutils"] = {
            "wall_time_seconds": np.nan,
            "cpu_time_seconds": np.nan,
        }

    if "isoster" not in method_profile_paths and args.isoster_profile_fits is not None:
        method_profile_paths["isoster"] = args.isoster_profile_fits
        method_tables["isoster"] = Table.read(args.isoster_profile_fits)
        harmonize_method_cog_columns(method_tables["isoster"], method_name="isoster")
        method_runtime["isoster"] = {
            "wall_time_seconds": np.nan,
            "cpu_time_seconds": np.nan,
        }

    comparison_summary = None
    comparison_figure_path = None

    if (
        not args.skip_comparison
        and "photutils" in method_tables
        and "isoster" in method_tables
    ):
        comparison_figure_path = output_dir / f"{prefix}_compare_{config_tag}_qa.png"

        photutils_model = build_model_image(
            image.shape,
            method_tables["photutils"],
            method_name="photutils",
        )
        isoster_model = build_model_image(
            image.shape,
            method_tables["isoster"],
            method_name="isoster",
        )

        comparison_summary = build_comparison_qa_figure(
            image=image,
            photutils_table=method_tables["photutils"],
            isoster_table=method_tables["isoster"],
            photutils_model=photutils_model,
            isoster_model=isoster_model,
            output_path=comparison_figure_path,
            galaxy_name=args.galaxy,
            mock_id=args.mock_id,
            pixel_scale_arcsec=pixel_scale_arcsec,
            redshift=redshift,
            runtime_photutils=method_runtime["photutils"],
            runtime_isoster=method_runtime["isoster"],
            overlay_step=args.isophote_overlay_step,
            dpi=args.qa_dpi,
        )

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

    report_path = output_dir / f"{prefix}_report.md"
    write_markdown_report(
        report_path=report_path,
        prefix=prefix,
        input_path=input_fits,
        run_metadata=run_metadata,
        summary_photutils=summary_photutils,
        summary_isoster=summary_isoster,
        comparison_summary=comparison_summary,
    )

    manifest_path = output_dir / f"{prefix}_manifest.json"
    manifest_payload = {
        "prefix": prefix,
        "input_fits": str(input_fits),
        "output_dir": str(output_dir),
        "method_profile_paths": {
            key: str(value) for key, value in method_profile_paths.items()
        },
        "comparison_figure": str(comparison_figure_path)
        if comparison_figure_path is not None
        else None,
        "report": str(report_path),
        "run_metadata": run_metadata,
        "comparison_summary": comparison_summary,
    }
    dump_json(manifest_path, manifest_payload)

    print(f"Input FITS: {input_fits}")
    print(f"Output directory: {output_dir}")
    print(f"Manifest: {manifest_path}")
    if comparison_figure_path is not None:
        print(f"Comparison QA: {comparison_figure_path}")


if __name__ == "__main__":
    main()
