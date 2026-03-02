"""Profile optimized isophote fitting on a synthetic Sersic image."""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
from io import StringIO
from pathlib import Path

import numpy as np
from scipy.special import gammaincinv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.utils.run_metadata import collect_environment_metadata, write_json  # noqa: E402
from isoster import fit_image, IsosterConfig  # noqa: E402
from isoster.output_paths import resolve_output_directory  # noqa: E402


def create_sersic_image(
    shape=(500, 500),
    x0=250,
    y0=250,
    eps=0.3,
    pa=1.0,
    n=4.0,
    effective_radius=50,
    intensity_at_effective_radius=1000.0,
    noise_sigma=1.0,
):
    """Create a synthetic 2-D Sersic profile image."""
    y_grid, x_grid = np.mgrid[: shape[0], : shape[1]]
    dx = x_grid - x0
    dy = y_grid - y0

    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    elliptical_radius = np.sqrt(x_rot**2 + (y_rot / (1 - eps)) ** 2)
    elliptical_radius = np.maximum(elliptical_radius, 1e-6)

    b_n = gammaincinv(2.0 * n, 0.5)
    image = intensity_at_effective_radius * np.exp(
        -b_n * ((elliptical_radius / effective_radius) ** (1.0 / n) - 1.0)
    )

    if noise_sigma > 0:
        rng = np.random.default_rng(42)
        image += rng.normal(0, noise_sigma, shape)
    return image


def run_profile(profile_repetitions: int) -> tuple[cProfile.Profile, dict]:
    """Run cProfile on fit_image for one or multiple repetitions."""
    image = create_sersic_image()
    center_x, center_y = 250, 250
    config = IsosterConfig(
        x0=center_x,
        y0=center_y,
        sma0=10.0,
        eps=0.1,
        pa=0.0,
        minsma=0.0,
        maxsma=image.shape[0] / 2,
        astep=0.1,
        maxit=50,
        conver=0.05,
        fix_center=False,
        fix_eps=False,
        fix_pa=False,
        compute_errors=True,
        compute_deviations=True,
    )

    profiler = cProfile.Profile()
    last_results = None
    for _ in range(profile_repetitions):
        profiler.enable()
        last_results = fit_image(image, None, config)
        profiler.disable()

    isophotes = last_results["isophotes"] if last_results is not None else []
    return profiler, {
        "profile_repetitions": int(profile_repetitions),
        "n_isophotes_last_run": int(len(isophotes)),
        "n_converged_last_run": int(sum(1 for iso in isophotes if iso["stop_code"] == 0)),
    }


def format_pstats_text(profiler: cProfile.Profile, sort_key: str, top_n: int) -> str:
    """Render pstats report to text."""
    output_buffer = StringIO()
    stats = pstats.Stats(profiler, stream=output_buffer).sort_stats(sort_key)
    stats.print_stats(top_n)
    return output_buffer.getvalue()


def persist_artifacts(output_directory: Path, profiler: cProfile.Profile, run_summary: dict, top_n: int) -> None:
    """Persist .prof and textual/json summaries."""
    output_directory.mkdir(parents=True, exist_ok=True)
    profiler_path = output_directory / "profile_isophote.prof"
    profiler.dump_stats(profiler_path)

    cumulative_text = format_pstats_text(profiler, "cumulative", top_n)
    tottime_text = format_pstats_text(profiler, "tottime", top_n)
    (output_directory / "top_cumulative.txt").write_text(cumulative_text, encoding="utf-8")
    (output_directory / "top_tottime.txt").write_text(tottime_text, encoding="utf-8")

    summary_payload = {
        "environment": collect_environment_metadata(project_root=Path(__file__).resolve().parents[2]),
        "run_summary": run_summary,
        "profiler_artifact": str(profiler_path),
    }
    write_json(output_directory / "profile_summary.json", summary_payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile isophote fitting and persist profiler artifacts.")
    parser.add_argument("--repetitions", type=int, default=1, help="Number of repeated profile runs.")
    parser.add_argument("--top-n", type=int, default=30, help="Top rows in text summaries.")
    parser.add_argument("--output", default=None, help="Explicit output directory.")
    args = parser.parse_args()

    profiler, run_summary = run_profile(profile_repetitions=max(1, args.repetitions))
    output_directory = resolve_output_directory(
        "benchmarks_profiling",
        "profile_isophote",
        explicit_output_directory=args.output,
    )
    persist_artifacts(output_directory, profiler, run_summary, top_n=max(1, args.top_n))

    print(f"Profile artifacts written under: {output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
