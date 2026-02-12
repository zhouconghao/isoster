"""Run focused numba profiling drills for flagged/high-variability benchmark cases."""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import sys
import time
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.utils.run_metadata import collect_environment_metadata, write_json  # noqa: E402
from isoster import fit_image  # noqa: E402
from isoster.config import IsosterConfig  # noqa: E402
from isoster.output_paths import resolve_output_directory  # noqa: E402

DEFAULT_SOURCE_PATHS = [
    Path("outputs/benchmarks_performance/bench_numba_speedup/numba_benchmark_results.json"),
    Path("outputs/benchmarks_performance/bench_numba_speedup_scale2/numba_benchmark_results.json"),
]
DEFAULT_FOCUS_CASE = "n1_medium_eps07__scale2"


def create_test_image(
    sersic_index: float,
    effective_radius: float,
    intensity_at_effective_radius: float,
    ellipticity: float,
    position_angle: float,
    oversample: int,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Create a synthetic Sersic image matching benchmark case settings."""
    half_size = int(15 * effective_radius)
    shape = (2 * half_size, 2 * half_size)
    center_x, center_y = half_size, half_size

    b_n = 1.9992 * sersic_index - 0.3271
    if oversample > 1:
        oversampled_shape = (shape[0] * oversample, shape[1] * oversample)
        y_coordinates = np.arange(oversampled_shape[0]) / oversample
        x_coordinates = np.arange(oversampled_shape[1]) / oversample
        y_grid, x_grid = np.meshgrid(y_coordinates, x_coordinates, indexing="ij")
    else:
        y_grid, x_grid = np.mgrid[: shape[0], : shape[1]].astype(np.float64)

    dx = x_grid - center_x
    dy = y_grid - center_y
    cos_position_angle = np.cos(position_angle)
    sin_position_angle = np.sin(position_angle)
    x_rot = dx * cos_position_angle + dy * sin_position_angle
    y_rot = -dx * sin_position_angle + dy * cos_position_angle
    radial_distance = np.sqrt(x_rot**2 + (y_rot / (1 - ellipticity)) ** 2)
    radial_distance = np.maximum(radial_distance, 0.1)

    image_full = intensity_at_effective_radius * np.exp(
        -b_n * ((radial_distance / effective_radius) ** (1 / sersic_index) - 1)
    )
    if oversample <= 1:
        return image_full, (center_x, center_y)

    image = np.zeros(shape, dtype=np.float64)
    for index_y in range(oversample):
        for index_x in range(oversample):
            image += image_full[index_y::oversample, index_x::oversample]
    image /= oversample**2
    return image, (center_x, center_y)


def build_config(
    center_x: int,
    center_y: int,
    effective_radius: float,
    ellipticity: float,
    position_angle: float,
) -> IsosterConfig:
    """Build an isophote fitting config consistent with speedup benchmark cases."""
    return IsosterConfig(
        x0=center_x,
        y0=center_y,
        eps=ellipticity,
        pa=position_angle,
        sma0=10.0,
        minsma=3.0,
        maxsma=8 * effective_radius,
        astep=0.15,
        minit=10,
        maxit=50,
        conver=0.05,
        maxgerr=1.0 if ellipticity > 0.6 else 0.5,
        use_eccentric_anomaly=(ellipticity > 0.3),
    )


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as file_pointer:
        return json.load(file_pointer)


def _collect_flag_reasons(payload: Dict[str, object]) -> Dict[str, List[str]]:
    summary = payload.get("summary", {})
    case_to_reasons: Dict[str, set] = defaultdict(set)

    for case_name in summary.get("high_variability_cases", []):
        case_to_reasons[case_name].add("high_variability")
    for case_name in summary.get("slowdown_cases_steady_state", []):
        case_to_reasons[case_name].add("slowdown_steady_state")
    for case_name in summary.get("slowdown_cases_mean", []):
        case_to_reasons[case_name].add("slowdown_mean")

    return {case_name: sorted(reasons) for case_name, reasons in case_to_reasons.items()}


def _build_case_index(payload: Dict[str, object], source_path: Path) -> Dict[str, Dict[str, object]]:
    diagnostics_rows = {
        row["name"]: row
        for row in payload.get("case_diagnostics", [])
    }
    flag_reasons = _collect_flag_reasons(payload)

    case_index: Dict[str, Dict[str, object]] = {}
    for case in payload.get("numba_results", []):
        case_name = case["name"]
        case_index[case_name] = {
            "name": case_name,
            "n": float(case["n"]),
            "R_e": float(case["R_e"]),
            "eps": float(case["eps"]),
            "pa": float(case["pa"]),
            "oversample": int(case["oversample"]),
            "n_runs_source": int(case.get("n_runs", payload.get("n_runs", 1))),
            "source_path": str(source_path),
            "source_diagnostics": diagnostics_rows.get(case_name, {}),
            "flag_reasons": flag_reasons.get(case_name, []),
        }
    return case_index


def load_case_catalog(source_paths: Iterable[Path]) -> Dict[str, Dict[str, object]]:
    """Load and merge benchmark case metadata from benchmark result JSON files."""
    catalog: Dict[str, Dict[str, object]] = {}
    for source_path in source_paths:
        payload = _load_json(source_path)
        source_index = _build_case_index(payload=payload, source_path=source_path)
        for case_name, entry in source_index.items():
            # Keep the first observed entry for deterministic case metadata.
            if case_name not in catalog:
                catalog[case_name] = entry
            elif entry.get("flag_reasons"):
                merged_reasons = sorted(set(catalog[case_name].get("flag_reasons", [])) | set(entry["flag_reasons"]))
                catalog[case_name]["flag_reasons"] = merged_reasons
    return catalog


def choose_drill_cases(
    catalog: Dict[str, Dict[str, object]],
    focus_case: str,
    explicit_cases: List[str],
) -> Tuple[List[str], List[str]]:
    """Choose drill cases and return (selected, missing_requested)."""
    missing_requested: List[str] = []
    if explicit_cases:
        ordered_requested = []
        for case_name in explicit_cases:
            if case_name in catalog and case_name not in ordered_requested:
                ordered_requested.append(case_name)
            elif case_name not in catalog:
                missing_requested.append(case_name)

        if focus_case in catalog and focus_case not in ordered_requested:
            ordered_requested.insert(0, focus_case)
        return ordered_requested, missing_requested

    flagged_case_names = [name for name, item in catalog.items() if item.get("flag_reasons")]
    flagged_case_names.sort()
    if focus_case in catalog and focus_case not in flagged_case_names:
        flagged_case_names.insert(0, focus_case)
    elif focus_case in flagged_case_names:
        flagged_case_names.remove(focus_case)
        flagged_case_names.insert(0, focus_case)
    elif focus_case not in catalog:
        missing_requested.append(focus_case)
    return flagged_case_names, missing_requested


def summarize_times(samples: List[float]) -> Dict[str, float]:
    """Return deterministic timing statistics."""
    if len(samples) == 0:
        return {
            "count": 0,
            "mean_seconds": float("nan"),
            "median_seconds": float("nan"),
            "std_seconds": float("nan"),
            "min_seconds": float("nan"),
            "max_seconds": float("nan"),
            "p90_seconds": float("nan"),
            "coefficient_of_variation": float("nan"),
        }

    data = np.array(samples, dtype=np.float64)
    mean_seconds = float(np.mean(data))
    std_seconds = float(np.std(data))
    return {
        "count": int(len(samples)),
        "mean_seconds": mean_seconds,
        "median_seconds": float(np.median(data)),
        "std_seconds": std_seconds,
        "min_seconds": float(np.min(data)),
        "max_seconds": float(np.max(data)),
        "p90_seconds": float(np.percentile(data, 90.0)),
        "coefficient_of_variation": float(std_seconds / mean_seconds) if mean_seconds > 0 else 0.0,
    }


def format_stats_text(profiler: cProfile.Profile, sort_by: str, limit: int) -> str:
    """Render a textual pstats report."""
    output_buffer = StringIO()
    pstats.Stats(profiler, stream=output_buffer).sort_stats(sort_by).print_stats(limit)
    return output_buffer.getvalue()


def extract_top_rows(profiler: cProfile.Profile, sort_by: str, limit: int) -> List[Dict[str, object]]:
    """Extract top pstats rows for machine-readable artifact output."""
    stats = pstats.Stats(profiler)
    rows = []
    for function_key, data in stats.stats.items():
        primitive_calls, total_calls, total_time, cumulative_time, _ = data
        file_name, line_number, function_name = function_key
        rows.append(
            {
                "file": file_name,
                "line": int(line_number),
                "function": function_name,
                "primitive_calls": int(primitive_calls),
                "total_calls": int(total_calls),
                "total_time_seconds": float(total_time),
                "cumulative_time_seconds": float(cumulative_time),
            }
        )

    key_name = "cumulative_time_seconds" if sort_by == "cumulative" else "total_time_seconds"
    rows.sort(key=lambda row: row[key_name], reverse=True)
    return rows[:limit]


def run_case_drill(
    case: Dict[str, object],
    timing_runs: int,
    profile_runs: int,
    top_n: int,
    output_directory: Path,
) -> Dict[str, object]:
    """Run one focused numba drill and persist artifacts."""
    image, (center_x, center_y) = create_test_image(
        sersic_index=float(case["n"]),
        effective_radius=float(case["R_e"]),
        intensity_at_effective_radius=1000.0,
        ellipticity=float(case["eps"]),
        position_angle=float(case["pa"]),
        oversample=int(case["oversample"]),
    )
    config = build_config(
        center_x=center_x,
        center_y=center_y,
        effective_radius=float(case["R_e"]),
        ellipticity=float(case["eps"]),
        position_angle=float(case["pa"]),
    )

    probe_start = time.perf_counter()
    fit_image(image, None, config)
    probe_seconds = time.perf_counter() - probe_start

    steady_state_times: List[float] = []
    steady_state_results = None
    for _ in range(max(1, timing_runs)):
        start_time = time.perf_counter()
        steady_state_results = fit_image(image, None, config)
        steady_state_times.append(float(time.perf_counter() - start_time))

    profiler = cProfile.Profile()
    for _ in range(max(1, profile_runs)):
        profiler.enable()
        steady_state_results = fit_image(image, None, config)
        profiler.disable()

    output_directory.mkdir(parents=True, exist_ok=True)
    profile_path = output_directory / "case_profile.prof"
    profiler.dump_stats(profile_path)
    (output_directory / "top_cumulative.txt").write_text(
        format_stats_text(profiler, sort_by="cumulative", limit=top_n),
        encoding="utf-8",
    )
    (output_directory / "top_tottime.txt").write_text(
        format_stats_text(profiler, sort_by="tottime", limit=top_n),
        encoding="utf-8",
    )

    isophotes = steady_state_results["isophotes"] if steady_state_results else []
    stop_code_distribution: Dict[str, int] = defaultdict(int)
    for isophote in isophotes:
        stop_code_distribution[str(isophote["stop_code"])] += 1

    steady_summary = summarize_times(steady_state_times)
    drill_payload = {
        "case": case,
        "probe_seconds": float(probe_seconds),
        "steady_state_times_seconds": steady_state_times,
        "steady_state_summary": steady_summary,
        "probe_minus_steady_median_seconds": float(probe_seconds - steady_summary["median_seconds"]),
        "n_isophotes_last_run": int(len(isophotes)),
        "n_converged_last_run": int(sum(1 for iso in isophotes if iso["stop_code"] == 0)),
        "stop_code_distribution_last_run": dict(sorted(stop_code_distribution.items())),
        "profile_top_cumulative": extract_top_rows(profiler, sort_by="cumulative", limit=top_n),
        "profile_top_tottime": extract_top_rows(profiler, sort_by="tottime", limit=top_n),
        "profile_artifact": str(profile_path),
    }
    write_json(output_directory / "case_drill_summary.json", drill_payload)
    return drill_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run focused numba profiling drills for flagged benchmark cases.")
    parser.add_argument(
        "--source-json",
        action="append",
        default=[],
        help="Benchmark result JSON path. Repeatable; defaults to scale1 and scale2 speedup outputs.",
    )
    parser.add_argument(
        "--focus-case",
        default=DEFAULT_FOCUS_CASE,
        help=f"Case to force-prioritize if present. Default: {DEFAULT_FOCUS_CASE}",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Explicit case name to drill. Repeat to run multiple cases.",
    )
    parser.add_argument(
        "--timing-runs",
        type=int,
        default=0,
        help="Steady-state timing repetitions per case. 0 uses each case source n_runs.",
    )
    parser.add_argument("--profile-runs", type=int, default=1, help="Profile repetitions per case.")
    parser.add_argument("--top-n", type=int, default=20, help="Top rows in profiler text/JSON summaries.")
    parser.add_argument("--skip-warmup", action="store_true", help="Skip explicit numba warmup.")
    parser.add_argument("--output", default=None, help="Explicit output directory.")
    args = parser.parse_args()

    source_paths = [Path(path) for path in args.source_json] if args.source_json else list(DEFAULT_SOURCE_PATHS)
    missing_source_paths = [path for path in source_paths if not path.exists()]
    if missing_source_paths:
        missing_label = ", ".join(str(path) for path in missing_source_paths)
        raise FileNotFoundError(f"Missing source benchmark JSON file(s): {missing_label}")

    catalog = load_case_catalog(source_paths=source_paths)
    selected_cases, missing_requested = choose_drill_cases(
        catalog=catalog,
        focus_case=str(args.focus_case),
        explicit_cases=list(args.case),
    )
    if len(selected_cases) == 0:
        raise RuntimeError("No cases selected for numba drills. Provide --case or valid source JSON files.")

    output_directory = resolve_output_directory(
        "benchmarks_profiling",
        "profile_numba_flagged_cases",
        explicit_output_directory=args.output,
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    warmup_seconds = 0.0
    numba_available = None
    if not args.skip_warmup:
        from isoster.numba_kernels import NUMBA_AVAILABLE, warmup_numba

        numba_available = bool(NUMBA_AVAILABLE)
        if numba_available:
            warmup_start = time.perf_counter()
            warmup_numba()
            warmup_seconds = float(time.perf_counter() - warmup_start)
    else:
        from isoster.numba_kernels import NUMBA_AVAILABLE

        numba_available = bool(NUMBA_AVAILABLE)

    case_results = []
    failed_cases = []
    for case_name in selected_cases:
        case_entry = catalog[case_name]
        case_output_directory = output_directory / case_name
        timing_runs = int(args.timing_runs) if args.timing_runs > 0 else int(case_entry["n_runs_source"])
        try:
            print(
                f"Running case drill: {case_name} "
                f"(timing_runs={max(1, timing_runs)}, profile_runs={max(1, int(args.profile_runs))})"
            )
            case_payload = run_case_drill(
                case=case_entry,
                timing_runs=max(1, timing_runs),
                profile_runs=max(1, int(args.profile_runs)),
                top_n=max(1, int(args.top_n)),
                output_directory=case_output_directory,
            )
            case_results.append(case_payload)
        except Exception as error:
            failed_cases.append({"name": case_name, "error": str(error)})

    case_rows = []
    for row in case_results:
        source_diagnostics = row["case"].get("source_diagnostics", {})
        steady_summary = row["steady_state_summary"]
        case_rows.append(
            {
                "name": row["case"]["name"],
                "flag_reasons": row["case"].get("flag_reasons", []),
                "source_speedup_steady_state": source_diagnostics.get("speedup_steady_state"),
                "source_numba_cv": source_diagnostics.get("numba_cv"),
                "source_no_numba_cv": source_diagnostics.get("no_numba_cv"),
                "drill_probe_seconds": row["probe_seconds"],
                "drill_steady_state_mean_seconds": steady_summary["mean_seconds"],
                "drill_steady_state_cv": steady_summary["coefficient_of_variation"],
                "probe_minus_steady_median_seconds": row["probe_minus_steady_median_seconds"],
            }
        )

    case_rows.sort(
        key=lambda row: (
            -(row["drill_steady_state_cv"] if np.isfinite(row["drill_steady_state_cv"]) else -1.0),
            row["name"],
        )
    )
    summary_payload = {
        "environment": collect_environment_metadata(project_root=PROJECT_ROOT),
        "source_json_paths": [str(path) for path in source_paths],
        "selected_cases": selected_cases,
        "missing_requested_cases": missing_requested,
        "failed_cases": failed_cases,
        "numba_available": numba_available,
        "numba_warmup_seconds": warmup_seconds,
        "drill_case_count": int(len(case_results)),
        "case_rows_by_descending_cv": case_rows,
    }
    write_json(output_directory / "numba_case_drill_summary.json", summary_payload)

    print("Numba case drills complete.")
    print(f"- Output directory: {output_directory}")
    print(f"- Selected cases: {len(selected_cases)}")
    print(f"- Successful drills: {len(case_results)}")
    print(f"- Failed drills: {len(failed_cases)}")
    if missing_requested:
        print(f"- Missing requested cases: {', '.join(missing_requested)}")

    return 0 if len(failed_cases) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
