#!/usr/bin/env python
"""Run isoster baseline fits on registered galaxies with photutils/autoprof comparison.

Per-galaxy artifacts:
    results.json, REPORT.md, fit_configs.json,
    isoster/  (profile.fits, profile.ecsv, model.fits)
    photutils/  (profile.fits, profile.ecsv, model.fits)  [if available]
    autoprof/  (profile.ecsv, model.fits)  [if available]
    figures/  (qa_comparison.png)

Aggregate artifacts:
    results.json, REPORT.md

Usage:
    uv run python benchmarks/benchmark_baseline/run_baseline.py
    uv run python benchmarks/benchmark_baseline/run_baseline.py --quick
    uv run python benchmarks/benchmark_baseline/run_baseline.py --galaxy eso243-49
    uv run python benchmarks/benchmark_baseline/run_baseline.py --output /tmp/baseline
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure matplotlib cache directories before import
if "XDG_CACHE_HOME" not in os.environ:
    _xdg = PROJECT_ROOT / "outputs" / "tmp" / "xdg-cache"
    _xdg.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(_xdg)
if "MPLCONFIGDIR" not in os.environ:
    _mpl = PROJECT_ROOT / "outputs" / "tmp" / "mplconfig"
    _mpl.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl)

import matplotlib
matplotlib.rcParams["text.usetex"] = False

from isoster.output_paths import resolve_output_directory
from benchmarks.utils.run_metadata import collect_environment_metadata, write_json
from benchmarks.benchmark_baseline.baseline_shared import (
    GALAXY_REGISTRY,
    build_autoprof_config,
    build_isoster_model_image,
    build_photutils_config,
    build_photutils_model_image,
    check_autoprof_available,
    check_photutils_available,
    estimate_background,
    get_galaxy,
    load_autoprof_model_image,
    load_galaxy_image,
    make_comparison_qa_figure,
    resolve_geometry,
    run_autoprof_fit,
    run_isoster_fit,
    run_photutils_fit,
    save_autoprof_profile_ecsv,
    save_fit_configs,
    save_model_fits,
    save_profile_ecsv,
    save_profile_fits,
)


QUICK_GALAXY = "IC3370_mock2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isoster baseline fits with photutils/autoprof comparison.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help=f"Smoke test: run {QUICK_GALAXY} only.",
    )
    parser.add_argument(
        "--galaxy", type=str, default=None,
        help="Run a specific galaxy by name.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Override output directory.",
    )
    parser.add_argument(
        "--no-photutils", action="store_true",
        help="Skip photutils comparison.",
    )
    parser.add_argument(
        "--no-autoprof", action="store_true",
        help="Skip autoprof comparison.",
    )
    return parser.parse_args()


def run_galaxy(
    galaxy_entry: dict,
    output_dir: Path,
    run_photutils: bool = True,
    run_autoprof_flag: bool = True,
) -> dict:
    """Run fits for a single galaxy with all available methods."""
    name = galaxy_entry["name"]
    galaxy_dir = output_dir / name
    figures_dir = galaxy_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    print(f"  Loading {galaxy_entry['fits_path'].name} ...")
    image = load_galaxy_image(
        galaxy_entry["fits_path"],
        cube_index=galaxy_entry.get("cube_index"),
    )
    print(f"  Image shape: {image.shape}")

    # Resolve shared geometry (excluded from timing for all methods)
    geometry = resolve_geometry(galaxy_entry, image)
    config_overrides = galaxy_entry["config_overrides"]
    pixel_scale = galaxy_entry.get("pixel_scale", 1.0)
    zeropoint = galaxy_entry.get("zeropoint", 22.5)

    # Collect per-method data for QA and results
    methods_data: dict[str, dict] = {}
    fit_configs: dict[str, dict] = {}
    method_results: dict[str, dict] = {}

    # --- ISOSTER ---
    print(f"  [isoster] Fitting ...")
    iso_result = run_isoster_fit(image, geometry, config_overrides)
    iso_isophotes = iso_result["isophotes"]
    iso_config = iso_result["config"]
    print(
        f"    {iso_result['isophote_count']} isophotes, "
        f"stop codes: {iso_result['stop_code_counts']}, "
        f"wall={iso_result['runtime']['wall_time_seconds']:.2f}s"
    )

    # Save isoster artifacts
    iso_dir = galaxy_dir / "isoster"
    iso_dir.mkdir(parents=True, exist_ok=True)
    save_profile_fits(iso_isophotes, iso_dir / "profile.fits")
    save_profile_ecsv(iso_isophotes, iso_dir / "profile.ecsv")

    iso_model = build_isoster_model_image(image.shape, iso_isophotes)
    if iso_model is not None:
        save_model_fits(iso_model, iso_dir / "model.fits")

    fit_configs["isoster"] = iso_config.model_dump()
    methods_data["isoster"] = {
        "isophotes": iso_isophotes,
        "model": iso_model,
        "runtime": iso_result["runtime"],
    }
    method_results["isoster"] = {
        "isophote_count": iso_result["isophote_count"],
        "stop_code_counts": iso_result["stop_code_counts"],
        "runtime": iso_result["runtime"],
    }

    # --- PHOTUTILS ---
    if run_photutils:
        print(f"  [photutils] Fitting ...")
        phot_config = build_photutils_config(geometry, config_overrides)
        fit_configs["photutils"] = phot_config
        phot_result = run_photutils_fit(image, phot_config)

        if phot_result is not None:
            phot_isophotes = phot_result["isophotes"]
            print(
                f"    {phot_result['isophote_count']} isophotes, "
                f"stop codes: {phot_result['stop_code_counts']}, "
                f"wall={phot_result['runtime']['wall_time_seconds']:.2f}s"
            )

            phot_dir = galaxy_dir / "photutils"
            phot_dir.mkdir(parents=True, exist_ok=True)
            save_profile_fits(phot_isophotes, phot_dir / "profile.fits")
            save_profile_ecsv(phot_isophotes, phot_dir / "profile.ecsv")

            phot_model = build_photutils_model_image(image.shape, phot_isophotes)
            if phot_model is not None:
                save_model_fits(phot_model, phot_dir / "model.fits")

            methods_data["photutils"] = {
                "isophotes": phot_isophotes,
                "model": phot_model,
                "runtime": phot_result["runtime"],
            }
            method_results["photutils"] = {
                "isophote_count": phot_result["isophote_count"],
                "stop_code_counts": phot_result["stop_code_counts"],
                "runtime": phot_result["runtime"],
            }
        else:
            print(f"    photutils not available or failed")

    # --- AUTOPROF ---
    if run_autoprof_flag:
        print(f"  [autoprof] Fitting ...")
        background, background_noise = estimate_background(image)
        ap_config = build_autoprof_config(
            geometry, config_overrides,
            pixel_scale, zeropoint,
            background, background_noise,
        )
        fit_configs["autoprof"] = ap_config
        ap_result = run_autoprof_fit(
            image, galaxy_dir, name, ap_config,
        )

        if ap_result is not None:
            ap_profile = ap_result["profile"]
            print(
                f"    {ap_result['isophote_count']} isophotes, "
                f"wall={ap_result['runtime']['wall_time_seconds']:.2f}s"
            )

            ap_dir = galaxy_dir / "autoprof"
            ap_dir.mkdir(parents=True, exist_ok=True)
            save_autoprof_profile_ecsv(ap_profile, ap_dir / "profile.ecsv")

            ap_model = load_autoprof_model_image(ap_result)
            if ap_model is not None:
                save_model_fits(ap_model, ap_dir / "model.fits")

            methods_data["autoprof"] = {
                "profile": ap_profile,
                "model": ap_model,
                "runtime": ap_result["runtime"],
            }
            method_results["autoprof"] = {
                "isophote_count": ap_result["isophote_count"],
                "runtime": ap_result["runtime"],
            }
        else:
            print(f"    AutoProf not available or failed")

    # Save fit configurations JSON
    save_fit_configs(fit_configs, galaxy_dir / "fit_configs.json")

    # Generate comparison QA figure
    print(f"  Generating comparison QA figure ...")
    make_comparison_qa_figure(
        image, methods_data, name,
        figures_dir / "qa_comparison.png",
    )

    # Build per-galaxy record
    run_record = {
        "galaxy": name,
        "status": "success" if iso_result["isophote_count"] > 0 else "failure",
        "image_shape": list(image.shape),
        "methods": method_results,
    }

    # Per-galaxy results.json
    galaxy_results_payload = {
        **run_record,
        "environment": collect_environment_metadata(project_root=PROJECT_ROOT),
    }
    write_json(galaxy_dir / "results.json", galaxy_results_payload)

    # Per-galaxy REPORT.md
    write_galaxy_report(run_record, galaxy_dir)

    return run_record


def write_galaxy_report(run_record: dict, galaxy_dir: Path) -> None:
    """Write a per-galaxy REPORT.md with multi-method results."""
    name = run_record["galaxy"]
    lines = [
        f"# Baseline Report: {name}",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## Results",
        "",
        "| Method | Isophotes | Wall Time (s) | Stop Codes |",
        "|--------|-----------|---------------|------------|",
    ]
    for method_name, mresult in run_record["methods"].items():
        codes = mresult.get("stop_code_counts", {})
        codes_str = ", ".join(f"{k}:{v}" for k, v in sorted(codes.items()))
        lines.append(
            f"| {method_name} | {mresult['isophote_count']} "
            f"| {mresult['runtime']['wall_time_seconds']:.2f} "
            f"| {codes_str} |"
        )
    lines.extend([
        "",
        f"- **Image shape**: {run_record['image_shape']}",
        "",
        "## Artifacts",
        "",
        "- `fit_configs.json` — configurations for all methods (reusable as input)",
        "- `<method>/profile.fits`, `<method>/profile.ecsv` — isophote profile",
        "- `<method>/model.fits` — 2D reconstructed model image",
        "- `figures/qa_comparison.png` — multi-method comparison QA figure",
    ])
    report_path = galaxy_dir / "REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_aggregate_report(
    galaxy_results: list[dict],
    output_dir: Path,
    wall_total: float,
    photutils_available: bool,
    autoprof_available: bool,
) -> None:
    """Write aggregate REPORT.md summarizing all galaxies."""
    # Build header for methods table
    all_methods = set()
    for result in galaxy_results:
        all_methods.update(result.get("methods", {}).keys())
    method_order = [m for m in ["isoster", "photutils", "autoprof"] if m in all_methods]

    lines = [
        "# Baseline Benchmark Report",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## Method Availability",
        "",
        f"- isoster: available",
        f"- photutils: {'available' if photutils_available else 'NOT AVAILABLE'}",
        f"- AutoProf: {'available' if autoprof_available else 'NOT AVAILABLE'}",
        "",
    ]

    # Per-method summary tables
    for method_name in method_order:
        label = {"isoster": "isoster", "photutils": "photutils",
                 "autoprof": "AutoProf"}.get(method_name, method_name)
        lines.extend([
            f"## {label}",
            "",
            "| Galaxy | Isophotes | Wall Time (s) | Stop Codes |",
            "|--------|-----------|---------------|------------|",
        ])
        for result in galaxy_results:
            mresult = result.get("methods", {}).get(method_name)
            if mresult is None:
                lines.append(f"| {result['galaxy']} | - | - | - |")
                continue
            codes = mresult.get("stop_code_counts", {})
            codes_str = ", ".join(f"{k}:{v}" for k, v in sorted(codes.items()))
            lines.append(
                f"| {result['galaxy']} | {mresult['isophote_count']} "
                f"| {mresult['runtime']['wall_time_seconds']:.2f} "
                f"| {codes_str} |"
            )
        lines.append("")

    lines.extend([
        f"**Total wall time**: {wall_total:.2f}s",
        "",
        "## Artifacts",
        "",
        "Per galaxy (`<galaxy>/`):",
        "- `results.json` — multi-method results with environment metadata",
        "- `REPORT.md` — per-galaxy report",
        "- `fit_configs.json` — all method configs (reusable as input)",
        "- `<method>/profile.fits`, `<method>/profile.ecsv` — isophote profiles",
        "- `<method>/model.fits` — 2D model images",
        "- `figures/qa_comparison.png` — multi-method comparison QA figure",
        "",
        "Aggregate (this directory):",
        "- `results.json` — combined results for all galaxies",
        "- `REPORT.md` — this file",
    ])
    report_path = output_dir / "REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()

    # Determine which galaxies to run
    if args.galaxy:
        galaxies = [get_galaxy(args.galaxy)]
    elif args.quick:
        galaxies = [get_galaxy(QUICK_GALAXY)]
    else:
        galaxies = list(GALAXY_REGISTRY)

    galaxy_names = ", ".join(g["name"] for g in galaxies)

    # Check method availability
    photutils_available = (not args.no_photutils) and check_photutils_available()
    autoprof_available = (not args.no_autoprof) and check_autoprof_available()

    print(f"Baseline benchmark: {galaxy_names}")
    print(f"  photutils: {'YES' if photutils_available else 'NO (skipped)'}")
    print(f"  AutoProf:  {'YES' if autoprof_available else 'NO (skipped)'}")

    output_dir = resolve_output_directory(
        "benchmark_baseline",
        explicit_output_directory=args.output,
    )
    print(f"Output: {output_dir}")

    wall_start = time.perf_counter()
    galaxy_results = []

    for galaxy_entry in galaxies:
        print(f"\n--- {galaxy_entry['name']} ---")
        try:
            result = run_galaxy(
                galaxy_entry, output_dir,
                run_photutils=photutils_available,
                run_autoprof_flag=autoprof_available,
            )
            galaxy_results.append(result)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"  ERROR: {exc}")
            galaxy_results.append({
                "galaxy": galaxy_entry["name"],
                "status": "error",
                "error": str(exc),
                "image_shape": [],
                "methods": {},
            })

    wall_total = time.perf_counter() - wall_start

    # Aggregate results.json
    results_payload = {
        "galaxies": galaxy_results,
        "total_wall_time_seconds": round(wall_total, 4),
        "photutils_available": photutils_available,
        "autoprof_available": autoprof_available,
        "environment": collect_environment_metadata(project_root=PROJECT_ROOT),
    }
    write_json(output_dir / "results.json", results_payload)

    # Aggregate REPORT.md
    write_aggregate_report(
        galaxy_results, output_dir, wall_total,
        photutils_available, autoprof_available,
    )

    # Console summary
    print(f"\n{'='*60}")
    print(f"Baseline benchmark complete in {wall_total:.2f}s")
    for result in galaxy_results:
        status_icon = "OK" if result["status"] == "success" else "FAIL"
        methods_summary = []
        for method_name, mresult in result.get("methods", {}).items():
            methods_summary.append(
                f"{method_name}={mresult['isophote_count']}"
            )
        methods_str = ", ".join(methods_summary) if methods_summary else "no results"
        print(f"  [{status_icon}] {result['galaxy']}: {methods_str}")
    print(f"Output: {output_dir}")

    # Return non-zero if isoster failed for any galaxy
    if any(r["status"] != "success" for r in galaxy_results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
