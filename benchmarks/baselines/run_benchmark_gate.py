"""Run baseline-locked benchmark gate with efficiency and quantitative QA metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "XDG_CACHE_HOME" not in os.environ:
    xdg_cache_dir = PROJECT_ROOT / "outputs" / "tmp" / "xdg-cache"
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache_dir)
if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = PROJECT_ROOT / "outputs" / "tmp" / "mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

from benchmarks.baselines.collect_phase4_profile_baseline import collect_baseline_metrics  # noqa: E402
from benchmarks.performance.bench_efficiency import run_benchmark_suite  # noqa: E402
from benchmarks.utils.run_metadata import collect_environment_metadata, write_json  # noqa: E402
from benchmarks.utils.sersic_model import create_sersic_image_vectorized  # noqa: E402
from examples.huang2013.run_huang2013_real_mock_demo import (  # noqa: E402
    DEFAULT_PIXEL_SCALE_ARCSEC,
    DEFAULT_REDSHIFT,
    DEFAULT_ZEROPOINT,
    build_method_qa_figure,
    prepare_profile_table,
)
from isoster import fit_image  # noqa: E402
from isoster.config import IsosterConfig  # noqa: E402
from isoster.model import build_isoster_model  # noqa: E402
from isoster.output_paths import resolve_output_directory  # noqa: E402


DEFAULT_EFFICIENCY_LOCK_PATH = Path("benchmarks/baselines/efficiency_thresholds_2026-02-14.json")
DEFAULT_PROFILE_LOCK_PATH = Path("benchmarks/baselines/phase4_profile_thresholds_2026-02-11.json")
DEFAULT_GATE_DEFAULTS_PATH = Path("benchmarks/baselines/benchmark_gate_defaults.json")

TWO_D_CAVEAT = (
    "2-D residual metrics are system-level diagnostics (profile extraction + model reconstruction combined); "
    "they are reported quantitatively but do not isolate extraction-only behavior."
)


def get_phase4_scenarios() -> List[dict]:
    """Return scenario set used for baseline 1-D and 2-D QA reporting."""
    return [
        {
            "name": "sersic_n4_noiseless",
            "n": 4.0,
            "R_e": 20.0,
            "I_e": 2000.0,
            "eps": 0.4,
            "pa": float(np.pi / 4.0),
            "oversample": 10,
            "noise_snr": None,
            "config": {
                "sma0": 10.0,
                "minsma": 3.0,
                "maxsma": 80.0,
                "astep": 0.12,
                "eps": 0.4,
                "pa": float(np.pi / 4.0),
                "minit": 10,
                "maxit": 50,
                "conver": 0.03,
                "fix_center": True,
            },
        },
        {
            "name": "sersic_n1_high_eps_noise",
            "n": 1.0,
            "R_e": 25.0,
            "I_e": 1500.0,
            "eps": 0.7,
            "pa": float(np.pi / 3.0),
            "oversample": 5,
            "noise_snr": 100.0,
            "noise_seed": 123,
            "config": {
                "sma0": 8.0,
                "minsma": 3.0,
                "maxsma": 100.0,
                "astep": 0.15,
                "eps": 0.7,
                "pa": float(np.pi / 3.0),
                "minit": 10,
                "maxit": 50,
                "conver": 0.05,
                "maxgerr": 1.0,
                "use_eccentric_anomaly": True,
            },
        },
        {
            "name": "sersic_n4_extreme_eps_noise",
            "n": 4.0,
            "R_e": 20.0,
            "I_e": 2000.0,
            "eps": 0.6,
            "pa": float(np.pi / 6.0),
            "oversample": 15,
            "noise_snr": 80.0,
            "noise_seed": 456,
            "config": {
                "sma0": 10.0,
                "minsma": 3.0,
                "maxsma": 80.0,
                "astep": 0.15,
                "eps": 0.6,
                "pa": float(np.pi / 6.0),
                "minit": 10,
                "maxit": 60,
                "conver": 0.08,
                "maxgerr": 1.2,
                "use_eccentric_anomaly": True,
            },
        },
    ]


def load_json(path: Path) -> Dict[str, object]:
    """Load a JSON object from path."""
    with path.open("r", encoding="utf-8") as file_pointer:
        return json.load(file_pointer)


def load_gate_defaults(path: Path) -> Dict[str, object]:
    """Load optional gate defaults from JSON; return empty dict when missing."""
    if not path.exists():
        return {}

    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Gate defaults must be a JSON object: {path}")
    return payload


def resolve_float_default(
    cli_value: float | None,
    config_payload: Dict[str, object],
    config_key: str,
    fallback_value: float,
    minimum_value: float = 0.0,
) -> float:
    """Resolve a float setting from CLI override, config file, then fallback default."""
    raw_value = cli_value if cli_value is not None else config_payload.get(config_key, fallback_value)
    resolved = float(raw_value)
    if (not np.isfinite(resolved)) or (resolved < minimum_value):
        raise ValueError(
            f"Invalid {config_key}={raw_value}; expected finite float >= {minimum_value}."
        )
    return resolved


def resolve_int_default(
    cli_value: int | None,
    config_payload: Dict[str, object],
    config_key: str,
    fallback_value: int,
    minimum_value: int = 1,
) -> int:
    """Resolve an int setting from CLI override, config file, then fallback default."""
    raw_value = cli_value if cli_value is not None else config_payload.get(config_key, fallback_value)
    resolved = int(raw_value)
    if resolved < minimum_value:
        raise ValueError(
            f"Invalid {config_key}={raw_value}; expected integer >= {minimum_value}."
        )
    return resolved


def resolve_str_default(
    cli_value: str | None,
    config_payload: Dict[str, object],
    config_key: str,
    fallback_value: str,
) -> str:
    """Resolve a string setting from CLI override, config file, then fallback default."""
    raw_value = cli_value if cli_value is not None else config_payload.get(config_key, fallback_value)
    resolved = str(raw_value).strip()
    if not resolved:
        raise ValueError(f"Invalid {config_key}; expected non-empty string.")
    return resolved


def evaluate_efficiency_gate(
    efficiency_payload: Dict[str, object],
    efficiency_lock_payload: Dict[str, object],
    require_all_locked_cases: bool,
    runtime_jitter_sigma: float,
    runtime_jitter_floor_seconds: float,
) -> Dict[str, object]:
    """Evaluate efficiency results against locked thresholds with measured jitter tolerance."""
    current_cases = {
        str(case["name"]): case
        for case in efficiency_payload.get("cases", [])
    }

    rows = []
    missing_locked_cases = []
    for locked in efficiency_lock_payload.get("cases", []):
        case_name = str(locked["name"])
        current = current_cases.get(case_name)
        if current is None:
            missing_locked_cases.append(case_name)
            continue

        current_std_time = float(current.get("std_time", 0.0))
        if (not np.isfinite(current_std_time)) or (current_std_time < 0.0):
            current_std_time = 0.0

        jitter_tolerance_seconds = max(
            float(runtime_jitter_floor_seconds),
            float(runtime_jitter_sigma) * current_std_time,
        )
        adjusted_mean_time_threshold = float(locked["max_mean_time_threshold"]) + jitter_tolerance_seconds

        mean_time_ok = float(current["mean_time"]) <= adjusted_mean_time_threshold
        convergence_ok = float(current["convergence_rate"]) >= float(
            locked["minimum_convergence_rate_threshold"]
        )
        converged_ok = int(current["n_converged"]) >= int(locked["minimum_converged_count_threshold"])
        isophote_ok = int(current["n_isophotes"]) >= int(locked["minimum_isophote_count_threshold"])

        rows.append(
            {
                "name": case_name,
                "mean_time_current": float(current["mean_time"]),
                "mean_time_threshold": float(locked["max_mean_time_threshold"]),
                "mean_time_threshold_adjusted": float(adjusted_mean_time_threshold),
                "std_time_current": float(current_std_time),
                "time_jitter_tolerance_seconds": float(jitter_tolerance_seconds),
                "convergence_rate_current": float(current["convergence_rate"]),
                "convergence_rate_threshold": float(locked["minimum_convergence_rate_threshold"]),
                "converged_count_current": int(current["n_converged"]),
                "converged_count_threshold": int(locked["minimum_converged_count_threshold"]),
                "isophote_count_current": int(current["n_isophotes"]),
                "isophote_count_threshold": int(locked["minimum_isophote_count_threshold"]),
                "mean_time_ok": bool(mean_time_ok),
                "convergence_rate_ok": bool(convergence_ok),
                "converged_count_ok": bool(converged_ok),
                "isophote_count_ok": bool(isophote_ok),
                "pass": bool(mean_time_ok and convergence_ok and converged_ok and isophote_ok),
            }
        )

    all_rows_pass = bool(rows) and all(row["pass"] for row in rows)
    missing_ok = (not require_all_locked_cases) or (len(missing_locked_cases) == 0)
    return {
        "pass": bool(all_rows_pass and missing_ok),
        "rows": rows,
        "missing_locked_cases": missing_locked_cases,
        "require_all_locked_cases": bool(require_all_locked_cases),
        "runtime_jitter_sigma": float(runtime_jitter_sigma),
        "runtime_jitter_floor_seconds": float(runtime_jitter_floor_seconds),
    }


def evaluate_profile_1d_gate(
    profile_payload: Dict[str, object],
    profile_lock_payload: Dict[str, object],
) -> Dict[str, object]:
    """Evaluate 1-D profile metrics against locked thresholds."""
    current_cases = {
        str(case["name"]): case
        for case in profile_payload.get("cases", [])
    }

    rows = []
    missing_locked_cases = []
    for locked in profile_lock_payload.get("cases", []):
        case_name = str(locked["name"])
        current = current_cases.get(case_name)
        if current is None:
            missing_locked_cases.append(case_name)
            continue

        valid_count_ok = int(current["valid_point_count"]) >= int(locked["minimum_valid_point_count"])
        median_ok = float(current["median_abs_delta_intensity"]) <= float(
            locked["median_abs_delta_intensity_threshold"]
        )
        max_ok = float(current["max_abs_delta_intensity"]) <= float(
            locked["max_abs_delta_intensity_threshold"]
        )

        rows.append(
            {
                "name": case_name,
                "valid_point_count_current": int(current["valid_point_count"]),
                "valid_point_count_threshold": int(locked["minimum_valid_point_count"]),
                "median_abs_delta_intensity_current": float(current["median_abs_delta_intensity"]),
                "median_abs_delta_intensity_threshold": float(
                    locked["median_abs_delta_intensity_threshold"]
                ),
                "max_abs_delta_intensity_current": float(current["max_abs_delta_intensity"]),
                "max_abs_delta_intensity_threshold": float(locked["max_abs_delta_intensity_threshold"]),
                "valid_count_ok": bool(valid_count_ok),
                "median_ok": bool(median_ok),
                "max_ok": bool(max_ok),
                "pass": bool(valid_count_ok and median_ok and max_ok),
                "stop_code_distribution": current.get("stop_code_distribution", {}),
            }
        )

    return {
        "pass": bool(rows) and all(row["pass"] for row in rows) and len(missing_locked_cases) == 0,
        "rows": rows,
        "missing_locked_cases": missing_locked_cases,
    }


def compute_two_d_band_metrics(
    image: np.ndarray,
    model: np.ndarray,
    center_x: float,
    center_y: float,
    effective_radius: float,
    noise_sigma: float | None,
) -> Dict[str, object]:
    """Compute 2-D residual diagnostics in canonical radial bands."""
    y_grid, x_grid = np.mgrid[: image.shape[0], : image.shape[1]].astype(np.float64)
    radius = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)

    safe_data = np.where(np.abs(image) > 1e-12, image, np.nan)
    fractional_residual = 100.0 * (model - image) / safe_data
    fractional_abs_residual = 100.0 * np.abs(model - image) / safe_data

    chi_square_map = None
    if noise_sigma is not None and noise_sigma > 0:
        chi_square_map = ((model - image) / noise_sigma) ** 2.0

    radial_bands = [
        ("lt_0p5_re", 0.0, 0.5),
        ("0p5_to_4_re", 0.5, 4.0),
        ("4_to_8_re", 4.0, 8.0),
    ]

    band_metrics: Dict[str, object] = {}
    for band_name, min_re, max_re in radial_bands:
        mask = (radius >= (min_re * effective_radius)) & (radius < (max_re * effective_radius))
        finite_mask = mask & np.isfinite(fractional_residual) & np.isfinite(fractional_abs_residual)

        if int(finite_mask.sum()) == 0:
            band_metrics[band_name] = {
                "pixel_count": 0,
                "median_fractional_residual": float("nan"),
                "median_fractional_absolute_residual": float("nan"),
                "max_abs_fractional_residual": float("nan"),
                "integrated_chi_square": float("nan"),
            }
            continue

        integrated_chi_square = float("nan")
        if chi_square_map is not None:
            chi_mask = finite_mask & np.isfinite(chi_square_map)
            if int(chi_mask.sum()) > 0:
                integrated_chi_square = float(np.sum(chi_square_map[chi_mask]))

        band_metrics[band_name] = {
            "pixel_count": int(finite_mask.sum()),
            "median_fractional_residual": float(np.median(fractional_residual[finite_mask])),
            "median_fractional_absolute_residual": float(
                np.median(fractional_abs_residual[finite_mask])
            ),
            "max_abs_fractional_residual": float(
                np.max(np.abs(fractional_residual[finite_mask]))
            ),
            "integrated_chi_square": integrated_chi_square,
        }

    return band_metrics


def collect_two_d_system_metrics(
    output_directory: Path,
    qa_output_subdir: str,
    qa_overlay_step: int,
    qa_dpi: int,
) -> Dict[str, object]:
    """Collect quantitative 2-D system-level diagnostics for baseline scenarios."""
    qa_output_directory = output_directory / qa_output_subdir
    qa_output_directory.mkdir(parents=True, exist_ok=True)

    case_rows = []
    for case_index, case in enumerate(get_phase4_scenarios(), start=1):
        half_size = max(int(15 * case["R_e"]), 150)
        shape = (2 * half_size, 2 * half_size)
        center = (float(half_size), float(half_size))

        image, _ = create_sersic_image_vectorized(
            n=case["n"],
            R_e=case["R_e"],
            I_e=case["I_e"],
            eps=case["eps"],
            pa=case["pa"],
            shape=shape,
            center=center,
            oversample=case["oversample"],
        )

        noise_sigma = None
        noise_snr = case.get("noise_snr")
        if noise_snr is not None:
            rng = np.random.default_rng(case["noise_seed"])
            noise_sigma = float(case["I_e"] / noise_snr)
            image = image + rng.normal(0.0, noise_sigma, image.shape)

        config_values = dict(case["config"])
        config_values.setdefault("x0", center[0])
        config_values.setdefault("y0", center[1])
        config = IsosterConfig(**config_values)

        start_wall = time.perf_counter()
        start_cpu = time.process_time()
        fit_results = fit_image(image, mask=None, config=config)
        runtime_metadata = {
            "wall_time_seconds": float(time.perf_counter() - start_wall),
            "cpu_time_seconds": float(time.process_time() - start_cpu),
        }

        isophotes = fit_results["isophotes"]
        model = build_isoster_model(image.shape, isophotes)
        profile_table = prepare_profile_table(
            isophotes=isophotes,
            image=image,
            redshift=float(DEFAULT_REDSHIFT),
            pixel_scale_arcsec=float(DEFAULT_PIXEL_SCALE_ARCSEC),
            zeropoint_mag=float(DEFAULT_ZEROPOINT),
            cog_subpixels=32,
            method_name="isoster",
        )

        case_name = str(case["name"])
        qa_profile_fits_path = qa_output_directory / f"{case_name}_isoster_phase4_profile.fits"
        qa_figure_path = qa_output_directory / f"{case_name}_isoster_phase4_qa.png"

        profile_table.write(qa_profile_fits_path, overwrite=True)
        build_method_qa_figure(
            image=image,
            profile_table=profile_table,
            model_image=model,
            output_path=qa_figure_path,
            method_name="isoster",
            galaxy_name="phase4",
            mock_id=case_index,
            pixel_scale_arcsec=float(DEFAULT_PIXEL_SCALE_ARCSEC),
            redshift=float(DEFAULT_REDSHIFT),
            runtime_metadata=runtime_metadata,
            overlay_step=max(1, int(qa_overlay_step)),
            dpi=max(60, int(qa_dpi)),
        )

        stop_codes = [int(iso["stop_code"]) for iso in isophotes]
        stop_code_distribution = {
            str(code): int(count)
            for code, count in sorted(Counter(stop_codes).items())
        }

        case_rows.append(
            {
                "name": case_name,
                "noise_sigma": noise_sigma,
                "isophote_count": int(len(isophotes)),
                "stop_code_distribution": stop_code_distribution,
                "runtime": runtime_metadata,
                "band_metrics": compute_two_d_band_metrics(
                    image=image,
                    model=model,
                    center_x=center[0],
                    center_y=center[1],
                    effective_radius=float(case["R_e"]),
                    noise_sigma=noise_sigma,
                ),
                "qa_artifacts": {
                    "status": "generated",
                    "qa_style": "huang2013_basic_method_qa",
                    "qa_figure_path": str(qa_figure_path),
                    "profile_fits_path": str(qa_profile_fits_path),
                },
            }
        )

    return {
        "caveat": TWO_D_CAVEAT,
        "qa_style_default": "IC2597 Huang2013 basic-QA method style (build_method_qa_figure)",
        "qa_output_directory": str(qa_output_directory),
        "qa_overlay_step": int(max(1, qa_overlay_step)),
        "qa_dpi": int(max(60, qa_dpi)),
        "cases": case_rows,
    }


def write_rows_csv(path: Path, rows: List[Dict[str, object]], field_names: List[str]) -> None:
    """Write flat row dictionaries to CSV."""
    with path.open("w", newline="", encoding="utf-8") as file_pointer:
        writer = csv.DictWriter(file_pointer, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in field_names})


def flatten_two_d_rows(two_d_payload: Dict[str, object]) -> List[Dict[str, object]]:
    """Flatten nested 2-D metrics for CSV output."""
    flattened = []
    for case in two_d_payload.get("cases", []):
        for band_name, metrics in case.get("band_metrics", {}).items():
            flattened.append(
                {
                    "name": case["name"],
                    "band": band_name,
                    "pixel_count": metrics["pixel_count"],
                    "median_fractional_residual": metrics["median_fractional_residual"],
                    "median_fractional_absolute_residual": metrics[
                        "median_fractional_absolute_residual"
                    ],
                    "max_abs_fractional_residual": metrics["max_abs_fractional_residual"],
                    "integrated_chi_square": metrics["integrated_chi_square"],
                }
            )
    return flattened


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run baseline-locked benchmark gate with efficiency + QA metrics."
    )
    parser.add_argument("--output", type=str, default=None, help="Optional explicit output directory.")
    parser.add_argument("--quick", action="store_true", help="Use quick efficiency benchmark cases.")
    parser.add_argument("--n-runs", type=int, default=1, help="Efficiency benchmark repetitions per case.")
    parser.add_argument(
        "--efficiency-lock",
        type=str,
        default=str(DEFAULT_EFFICIENCY_LOCK_PATH),
        help="Efficiency lock JSON path.",
    )
    parser.add_argument(
        "--profile-lock",
        type=str,
        default=str(DEFAULT_PROFILE_LOCK_PATH),
        help="1-D profile lock JSON path.",
    )
    parser.add_argument(
        "--gate-defaults",
        type=str,
        default=str(DEFAULT_GATE_DEFAULTS_PATH),
        help="Optional JSON file for gate default settings.",
    )
    parser.add_argument(
        "--require-all-locked-cases",
        action="store_true",
        help="Fail when any locked efficiency case is missing from current run.",
    )
    parser.add_argument(
        "--efficiency-time-jitter-sigma",
        type=float,
        default=None,
        help="Per-case runtime jitter multiplier applied to measured std_time. Uses gate-defaults file when omitted.",
    )
    parser.add_argument(
        "--efficiency-time-jitter-floor-seconds",
        type=float,
        default=None,
        help="Minimum absolute runtime tolerance (seconds) for efficiency gate. Uses gate-defaults file when omitted.",
    )
    parser.add_argument(
        "--qa-overlay-step",
        type=int,
        default=None,
        help="Overlay every Nth isophote in generated QA figures. Uses gate-defaults file when omitted.",
    )
    parser.add_argument(
        "--qa-dpi",
        type=int,
        default=None,
        help="DPI for generated QA figures. Uses gate-defaults file when omitted.",
    )
    parser.add_argument(
        "--qa-output-subdir",
        type=str,
        default=None,
        help="Subdirectory under gate output root reserved for QA figure artifacts. Uses gate-defaults file when omitted.",
    )
    args = parser.parse_args()

    gate_defaults_path = Path(args.gate_defaults)
    gate_defaults_payload = load_gate_defaults(gate_defaults_path)

    resolved_efficiency_time_jitter_sigma = resolve_float_default(
        cli_value=args.efficiency_time_jitter_sigma,
        config_payload=gate_defaults_payload,
        config_key="efficiency_time_jitter_sigma",
        fallback_value=3.0,
        minimum_value=0.0,
    )
    resolved_efficiency_time_jitter_floor_seconds = resolve_float_default(
        cli_value=args.efficiency_time_jitter_floor_seconds,
        config_payload=gate_defaults_payload,
        config_key="efficiency_time_jitter_floor_seconds",
        fallback_value=0.002,
        minimum_value=0.0,
    )
    resolved_qa_overlay_step = resolve_int_default(
        cli_value=args.qa_overlay_step,
        config_payload=gate_defaults_payload,
        config_key="qa_overlay_step",
        fallback_value=10,
        minimum_value=1,
    )
    resolved_qa_dpi = resolve_int_default(
        cli_value=args.qa_dpi,
        config_payload=gate_defaults_payload,
        config_key="qa_dpi",
        fallback_value=180,
        minimum_value=60,
    )
    resolved_qa_output_subdir = resolve_str_default(
        cli_value=args.qa_output_subdir,
        config_payload=gate_defaults_payload,
        config_key="qa_output_subdir",
        fallback_value="qa_figures",
    )

    efficiency_lock_path = Path(args.efficiency_lock)
    profile_lock_path = Path(args.profile_lock)
    if not efficiency_lock_path.exists():
        raise FileNotFoundError(f"Efficiency lock file not found: {efficiency_lock_path}")
    if not profile_lock_path.exists():
        raise FileNotFoundError(f"Profile lock file not found: {profile_lock_path}")

    output_directory = resolve_output_directory(
        "benchmarks_performance",
        "benchmark_gate",
        explicit_output_directory=args.output,
    )

    efficiency_payload = run_benchmark_suite(n_runs=max(1, args.n_runs), quick=bool(args.quick))
    profile_payload = collect_baseline_metrics()
    two_d_payload = collect_two_d_system_metrics(
        output_directory=output_directory,
        qa_output_subdir=resolved_qa_output_subdir,
        qa_overlay_step=resolved_qa_overlay_step,
        qa_dpi=resolved_qa_dpi,
    )

    efficiency_lock_payload = load_json(efficiency_lock_path)
    profile_lock_payload = load_json(profile_lock_path)

    efficiency_gate = evaluate_efficiency_gate(
        efficiency_payload=efficiency_payload,
        efficiency_lock_payload=efficiency_lock_payload,
        require_all_locked_cases=bool(args.require_all_locked_cases),
        runtime_jitter_sigma=resolved_efficiency_time_jitter_sigma,
        runtime_jitter_floor_seconds=resolved_efficiency_time_jitter_floor_seconds,
    )
    profile_gate = evaluate_profile_1d_gate(
        profile_payload=profile_payload,
        profile_lock_payload=profile_lock_payload,
    )

    overall_pass = bool(efficiency_gate["pass"] and profile_gate["pass"])
    report_payload = {
        "environment": collect_environment_metadata(project_root=PROJECT_ROOT),
        "overall_pass": overall_pass,
        "gate_policy": {
            "efficiency": "locked thresholds",
            "profile_1d": "locked thresholds",
            "profile_2d": "report-only system-level diagnostics",
            "profile_2d_caveat": TWO_D_CAVEAT,
            "qa_style_default": "IC2597 Huang2013 basic-QA method style",
            "efficiency_runtime_jitter_sigma": resolved_efficiency_time_jitter_sigma,
            "efficiency_runtime_jitter_floor_seconds": resolved_efficiency_time_jitter_floor_seconds,
            "gate_defaults_path": str(gate_defaults_path),
            "gate_defaults_loaded": bool(gate_defaults_payload),
        },
        "qa_generation": {
            "status": "enabled",
            "mode": "built_in",
            "qa_output_directory": two_d_payload.get("qa_output_directory"),
            "qa_overlay_step": two_d_payload.get("qa_overlay_step"),
            "qa_dpi": two_d_payload.get("qa_dpi"),
        },
        "efficiency": {
            "lock_file": str(efficiency_lock_path),
            "current_metrics": efficiency_payload,
            "evaluation": efficiency_gate,
        },
        "profile_1d": {
            "lock_file": str(profile_lock_path),
            "current_metrics": profile_payload,
            "evaluation": profile_gate,
        },
        "profile_2d_system_level": two_d_payload,
    }

    report_json_path = output_directory / "benchmark_gate_report.json"
    efficiency_csv_path = output_directory / "benchmark_gate_efficiency.csv"
    profile_csv_path = output_directory / "benchmark_gate_profile_1d.csv"
    two_d_csv_path = output_directory / "benchmark_gate_profile_2d_system.csv"

    write_json(report_json_path, report_payload)
    write_rows_csv(
        efficiency_csv_path,
        efficiency_gate["rows"],
        [
            "name",
            "mean_time_current",
            "mean_time_threshold",
            "mean_time_threshold_adjusted",
            "std_time_current",
            "time_jitter_tolerance_seconds",
            "convergence_rate_current",
            "convergence_rate_threshold",
            "converged_count_current",
            "converged_count_threshold",
            "isophote_count_current",
            "isophote_count_threshold",
            "mean_time_ok",
            "convergence_rate_ok",
            "converged_count_ok",
            "isophote_count_ok",
            "pass",
        ],
    )
    write_rows_csv(
        profile_csv_path,
        profile_gate["rows"],
        [
            "name",
            "valid_point_count_current",
            "valid_point_count_threshold",
            "median_abs_delta_intensity_current",
            "median_abs_delta_intensity_threshold",
            "max_abs_delta_intensity_current",
            "max_abs_delta_intensity_threshold",
            "valid_count_ok",
            "median_ok",
            "max_ok",
            "pass",
        ],
    )
    write_rows_csv(
        two_d_csv_path,
        flatten_two_d_rows(two_d_payload),
        [
            "name",
            "band",
            "pixel_count",
            "median_fractional_residual",
            "median_fractional_absolute_residual",
            "max_abs_fractional_residual",
            "integrated_chi_square",
        ],
    )

    print(f"Benchmark gate report JSON: {report_json_path}")
    print(f"Efficiency gate CSV: {efficiency_csv_path}")
    print(f"1-D profile gate CSV: {profile_csv_path}")
    print(f"2-D system-level metrics CSV: {two_d_csv_path}")
    print(f"Gate pass: {overall_pass}")
    if efficiency_gate["missing_locked_cases"]:
        print(f"Missing locked efficiency cases: {efficiency_gate['missing_locked_cases']}")
    if profile_gate["missing_locked_cases"]:
        print(f"Missing locked 1-D cases: {profile_gate['missing_locked_cases']}")
    print(f"2-D caveat: {TWO_D_CAVEAT}")
    print(
        "QA generation: "
        f"style=huang2013_basic_method_qa, output_dir={two_d_payload['qa_output_directory']}, "
        f"overlay_step={two_d_payload['qa_overlay_step']}, dpi={two_d_payload['qa_dpi']}"
    )

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
