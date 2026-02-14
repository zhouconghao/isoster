"""Efficiency benchmark with standardized output artifacts and metadata."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.utils.run_metadata import collect_environment_metadata, write_json  # noqa: E402
from benchmarks.utils.sersic_model import compute_bn  # noqa: E402
from isoster import fit_image  # noqa: E402
from isoster.config import IsosterConfig  # noqa: E402
from isoster.output_paths import resolve_output_directory  # noqa: E402


def create_sersic_model(
    effective_radius: float,
    sersic_index: float,
    intensity_at_effective_radius: float,
    ellipticity: float,
    position_angle: float,
    noise_level: float | None = None,
    oversample: int = 1,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """Create a centered 2-D Sersic profile image with optional Gaussian noise."""
    half_size = max(int(15 * effective_radius), 150)
    shape = (2 * half_size, 2 * half_size)
    x0, y0 = half_size, half_size

    b_n = compute_bn(sersic_index)
    if oversample > 1:
        oversampled_shape = (shape[0] * oversample, shape[1] * oversample)
        y = np.arange(oversampled_shape[0]) / oversample
        x = np.arange(oversampled_shape[1]) / oversample
        y_grid, x_grid = np.meshgrid(y, x, indexing="ij")
    else:
        y = np.arange(shape[0])
        x = np.arange(shape[1])
        y_grid, x_grid = np.meshgrid(y, x, indexing="ij")

    dx = x_grid - x0
    dy = y_grid - y0
    x_rot = dx * np.cos(position_angle) + dy * np.sin(position_angle)
    y_rot = -dx * np.sin(position_angle) + dy * np.cos(position_angle)
    radial_distance = np.sqrt(x_rot**2 + (y_rot / (1 - ellipticity)) ** 2)
    oversampled_image = intensity_at_effective_radius * np.exp(
        -b_n * ((radial_distance / effective_radius) ** (1 / sersic_index) - 1)
    )

    if oversample > 1:
        image = np.zeros(shape)
        for y_offset in range(oversample):
            for x_offset in range(oversample):
                image += oversampled_image[y_offset::oversample, x_offset::oversample]
        image /= oversample**2
    else:
        image = oversampled_image

    if noise_level is not None and noise_level > 0:
        rng = np.random.default_rng(42)
        image += rng.normal(0, noise_level, image.shape)

    return image, (x0, y0), shape


def benchmark_case(name: str, image: np.ndarray, config: IsosterConfig, n_runs: int) -> Dict[str, object]:
    """Benchmark a single case over multiple runs."""
    times: List[float] = []
    last_results = None
    for _ in range(n_runs):
        start = time.perf_counter()
        last_results = fit_image(image, None, config)
        elapsed = time.perf_counter() - start
        times.append(float(elapsed))

    isophotes = last_results["isophotes"] if last_results is not None else []
    n_isophotes = len(isophotes)
    n_converged = sum(1 for iso in isophotes if iso["stop_code"] == 0)

    return {
        "name": name,
        "times": times,
        "mean_time": float(np.mean(times)),
        "std_time": float(np.std(times)),
        "min_time": float(np.min(times)),
        "n_isophotes": int(n_isophotes),
        "n_converged": int(n_converged),
        "convergence_rate": float(n_converged / n_isophotes) if n_isophotes > 0 else 0.0,
    }


def get_test_cases(quick: bool) -> List[Tuple[str, float, float, float, float, float | None, int]]:
    """Return benchmark case definitions."""
    if quick:
        return [
            ("n1_small_circular", 1.0, 10.0, 0.0, 0.0, None, 5),
            ("n1_medium_eps04", 1.0, 20.0, 0.4, np.pi / 4, None, 5),
            ("n4_medium_snr100", 4.0, 20.0, 0.4, np.pi / 4, 100, 10),
        ]

    return [
        ("n1_small_circular", 1.0, 10.0, 0.0, 0.0, None, 5),
        ("n4_small_circular", 4.0, 10.0, 0.0, 0.0, None, 10),
        ("n1_medium_circular", 1.0, 20.0, 0.0, 0.0, None, 5),
        ("n1_medium_eps04", 1.0, 20.0, 0.4, np.pi / 4, None, 5),
        ("n4_medium_eps04", 4.0, 20.0, 0.4, np.pi / 4, None, 10),
        ("n1_medium_eps07", 1.0, 20.0, 0.7, np.pi / 3, None, 5),
        ("n4_medium_eps06", 4.0, 20.0, 0.6, np.pi / 4, None, 15),
        ("n1_medium_snr100", 1.0, 20.0, 0.4, np.pi / 4, 100, 5),
        ("n4_medium_snr100", 4.0, 20.0, 0.4, np.pi / 4, 100, 10),
    ]


def run_benchmark_suite(n_runs: int, quick: bool) -> Dict[str, object]:
    """Run all benchmark cases and return structured payload."""
    test_cases = get_test_cases(quick=quick)
    case_results: List[Dict[str, object]] = []

    for test_name, n, effective_radius, ellipticity, position_angle, signal_to_noise_ratio, oversample in test_cases:
        print(
            f"Running {test_name}: n={n}, Re={effective_radius}, eps={ellipticity}, "
            f"PA={position_angle:.2f}, SNR={signal_to_noise_ratio}, oversample={oversample}"
        )

        intensity_at_effective_radius = 1000.0
        noise_level = (
            intensity_at_effective_radius / signal_to_noise_ratio
            if signal_to_noise_ratio is not None
            else None
        )
        image, (x0, y0), _ = create_sersic_model(
            effective_radius=effective_radius,
            sersic_index=n,
            intensity_at_effective_radius=intensity_at_effective_radius,
            ellipticity=ellipticity,
            position_angle=position_angle,
            noise_level=noise_level,
            oversample=oversample,
        )

        config = IsosterConfig(
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
            maxgerr=1.0 if ellipticity > 0.6 else 0.5,
            use_eccentric_anomaly=(ellipticity > 0.3),
        )

        result = benchmark_case(test_name, image, config, n_runs=n_runs)
        case_results.append(result)
        print(
            f"  mean={result['mean_time']:.3f}s std={result['std_time']:.3f}s "
            f"converged={result['n_converged']}/{result['n_isophotes']}"
        )

    mean_times = [result["mean_time"] for result in case_results]
    convergence_rates = [result["convergence_rate"] for result in case_results]
    summary = {
        "n_cases": len(case_results),
        "n_runs_per_case": int(n_runs),
        "quick_mode": bool(quick),
        "total_mean_runtime_seconds": float(np.sum(mean_times)),
        "mean_case_runtime_seconds": float(np.mean(mean_times)) if mean_times else 0.0,
        "median_case_runtime_seconds": float(np.median(mean_times)) if mean_times else 0.0,
        "minimum_convergence_rate": float(np.min(convergence_rates)) if convergence_rates else 0.0,
    }

    return {"summary": summary, "cases": case_results}


def save_case_csv(path: Path, case_results: List[Dict[str, object]]) -> None:
    """Persist per-case benchmark summary as CSV."""
    field_names = [
        "name",
        "mean_time",
        "std_time",
        "min_time",
        "n_isophotes",
        "n_converged",
        "convergence_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as file_pointer:
        writer = csv.DictWriter(file_pointer, fieldnames=field_names)
        writer.writeheader()
        for result in case_results:
            writer.writerow({name: result[name] for name in field_names})


def save_summary_plots(output_directory: Path, case_results: List[Dict[str, object]]) -> None:
    """Write benchmark diagnostic plots."""
    case_names = [result["name"] for result in case_results]
    mean_times = [result["mean_time"] for result in case_results]
    convergence_rates = [result["convergence_rate"] for result in case_results]

    figure, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    axes[0].bar(case_names, mean_times, color="steelblue", alpha=0.8)
    axes[0].set_ylabel("Mean Runtime (s)")
    axes[0].set_title("Efficiency Benchmark Runtime Summary")
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(case_names, convergence_rates, color="seagreen", alpha=0.8)
    axes[1].set_ylabel("Convergence Rate")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].tick_params(axis="x", rotation=45)

    figure.tight_layout()
    figure.savefig(output_directory / "efficiency_benchmark_summary.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    flagged_cases = [
        result
        for result in case_results
        if result["convergence_rate"] < 0.9
        or result["mean_time"] > (np.median(mean_times) * 1.5 if mean_times else 0.0)
    ]
    if not flagged_cases:
        return

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.axis("off")
    report_lines = ["Flagged Cases (low convergence or high runtime):"]
    for result in flagged_cases:
        report_lines.append(
            f"- {result['name']}: mean={result['mean_time']:.3f}s, "
            f"convergence={100.0 * result['convergence_rate']:.1f}%"
        )
    axis.text(0.01, 0.98, "\n".join(report_lines), va="top", family="monospace")
    figure.tight_layout()
    figure.savefig(output_directory / "efficiency_flagged_cases.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ISOSTER efficiency benchmark suite.")
    parser.add_argument("--output", "-o", default=None, help="Explicit output directory.")
    parser.add_argument("--quick", action="store_true", help="Run reduced-case quick benchmark.")
    parser.add_argument("--n-runs", type=int, default=3, help="Number of runs per case.")
    args = parser.parse_args()

    output_directory = resolve_output_directory(
        "benchmarks_performance",
        "bench_efficiency",
        explicit_output_directory=args.output,
    )
    output_payload = run_benchmark_suite(n_runs=max(1, args.n_runs), quick=args.quick)
    output_payload["environment"] = collect_environment_metadata(project_root=Path(__file__).resolve().parents[2])

    json_path = output_directory / "efficiency_benchmark_results.json"
    csv_path = output_directory / "efficiency_benchmark_results.csv"
    write_json(json_path, output_payload)
    save_case_csv(csv_path, output_payload["cases"])
    save_summary_plots(output_directory, output_payload["cases"])

    print(f"JSON summary written to: {json_path}")
    print(f"CSV summary written to: {csv_path}")
    print(f"Plots written under: {output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
