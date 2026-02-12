"""Profile hot paths in isoster and persist profiler artifacts."""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
from io import StringIO
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.utils.run_metadata import collect_environment_metadata, write_json  # noqa: E402
from isoster import fit_image  # noqa: E402
from isoster.config import IsosterConfig  # noqa: E402
from isoster.output_paths import resolve_output_directory  # noqa: E402


def create_test_image(
    sersic_index: float = 4.0,
    effective_radius: float = 30.0,
    intensity_at_effective_radius: float = 1000.0,
    ellipticity: float = 0.4,
    position_angle: float = np.pi / 4,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[float, float, float]]:
    """Create a synthetic Sersic image for profiling."""
    half_size = int(15 * effective_radius)
    shape = (2 * half_size, 2 * half_size)
    x0, y0 = half_size, half_size

    b_n = 1.9992 * sersic_index - 0.3271
    y_grid, x_grid = np.mgrid[: shape[0], : shape[1]].astype(np.float64)
    dx = x_grid - x0
    dy = y_grid - y0

    cos_pa = np.cos(position_angle)
    sin_pa = np.sin(position_angle)
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    radial_distance = np.sqrt(x_rot**2 + (y_rot / (1 - ellipticity)) ** 2)
    radial_distance = np.maximum(radial_distance, 0.1)

    image = intensity_at_effective_radius * np.exp(
        -b_n * ((radial_distance / effective_radius) ** (1 / sersic_index) - 1)
    )
    return image, (x0, y0), (effective_radius, ellipticity, position_angle)


def build_config(x0: int, y0: int, effective_radius: float, ellipticity: float, position_angle: float) -> IsosterConfig:
    """Create a profiling configuration."""
    return IsosterConfig(
        x0=x0,
        y0=y0,
        eps=ellipticity,
        pa=position_angle,
        sma0=10.0,
        minsma=3.0,
        maxsma=8 * effective_radius,
        astep=0.15,
        minit=10,
        maxit=50,
        conver=0.05,
        use_eccentric_anomaly=True,
    )


def extract_top_rows(profiler: cProfile.Profile, sort_by: str, limit: int) -> List[Dict[str, object]]:
    """Extract top function rows from pstats for JSON output."""
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


def format_stats_text(profiler: cProfile.Profile, sort_by: str, limit: int, filter_pattern: str | None = None) -> str:
    """Render textual pstats report."""
    output_buffer = StringIO()
    stats = pstats.Stats(profiler, stream=output_buffer)
    stats.sort_stats(sort_by)
    if filter_pattern:
        stats.print_stats(filter_pattern, limit)
    else:
        stats.print_stats(limit)
    return output_buffer.getvalue()


def run_profile_runs(n_runs: int) -> Tuple[cProfile.Profile, Dict[str, object]]:
    """Run profiling for one or more executions and return aggregate profiler."""
    image, (x0, y0), (effective_radius, ellipticity, position_angle) = create_test_image()
    config = build_config(x0=x0, y0=y0, effective_radius=effective_radius, ellipticity=ellipticity, position_angle=position_angle)

    profiler = cProfile.Profile()
    last_results = None
    for _ in range(n_runs):
        profiler.enable()
        last_results = fit_image(image, None, config)
        profiler.disable()

    n_isophotes = len(last_results["isophotes"]) if last_results is not None else 0
    n_converged = sum(1 for iso in last_results["isophotes"] if iso["stop_code"] == 0) if last_results else 0
    run_summary = {
        "n_runs": int(n_runs),
        "n_isophotes_last_run": int(n_isophotes),
        "n_converged_last_run": int(n_converged),
    }
    return profiler, run_summary


def persist_profile_artifacts(
    output_directory: Path,
    profiler: cProfile.Profile,
    run_summary: Dict[str, object],
    top_n: int,
) -> Dict[str, str]:
    """Write .prof, text summaries, and JSON hotspot summaries."""
    output_directory.mkdir(parents=True, exist_ok=True)

    profile_path = output_directory / "profile_hotpaths.prof"
    profiler.dump_stats(profile_path)

    cumulative_text = format_stats_text(profiler, sort_by="cumulative", limit=top_n)
    tottime_text = format_stats_text(profiler, sort_by="tottime", limit=top_n)
    isoster_text = format_stats_text(profiler, sort_by="tottime", limit=top_n, filter_pattern="isoster")

    (output_directory / "top_cumulative.txt").write_text(cumulative_text, encoding="utf-8")
    (output_directory / "top_tottime.txt").write_text(tottime_text, encoding="utf-8")
    (output_directory / "top_isoster_tottime.txt").write_text(isoster_text, encoding="utf-8")

    summary_payload = {
        "environment": collect_environment_metadata(project_root=Path(__file__).resolve().parents[2]),
        "run_summary": run_summary,
        "top_cumulative": extract_top_rows(profiler, sort_by="cumulative", limit=top_n),
        "top_tottime": extract_top_rows(profiler, sort_by="tottime", limit=top_n),
    }
    json_path = output_directory / "hotpaths_summary.json"
    write_json(json_path, summary_payload)

    return {
        "profile_path": str(profile_path),
        "cumulative_text_path": str(output_directory / "top_cumulative.txt"),
        "tottime_text_path": str(output_directory / "top_tottime.txt"),
        "isoster_text_path": str(output_directory / "top_isoster_tottime.txt"),
        "summary_json_path": str(json_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile ISOSTER hot paths and persist artifacts.")
    parser.add_argument("--multi", type=int, default=1, help="Number of repeated runs to aggregate.")
    parser.add_argument("--top-n", type=int, default=30, help="Top rows to keep in reports.")
    parser.add_argument("--output", default=None, help="Explicit output directory.")
    args = parser.parse_args()

    n_runs = max(1, args.multi)
    profiler, run_summary = run_profile_runs(n_runs=n_runs)
    output_directory = resolve_output_directory(
        "benchmarks_profiling",
        "profile_hotpaths",
        explicit_output_directory=args.output,
    )

    artifact_paths = persist_profile_artifacts(
        output_directory=output_directory,
        profiler=profiler,
        run_summary=run_summary,
        top_n=max(1, args.top_n),
    )

    print("Profile artifacts written:")
    for key, value in artifact_paths.items():
        print(f"- {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
