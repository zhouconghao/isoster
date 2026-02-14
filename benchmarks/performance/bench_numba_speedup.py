"""Numba speedup benchmark with detailed runtime diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.utils.run_metadata import collect_environment_metadata, write_json  # noqa: E402
from benchmarks.utils.sersic_model import compute_bn  # noqa: E402
from isoster.output_paths import resolve_output_directory  # noqa: E402


def get_base_test_cases() -> List[Dict[str, float]]:
    """Return baseline benchmark cases."""
    return [
        {"name": "n1_small_circular", "n": 1.0, "R_e": 10.0, "eps": 0.0, "pa": 0.0, "oversample": 5},
        {"name": "n4_small_circular", "n": 4.0, "R_e": 10.0, "eps": 0.0, "pa": 0.0, "oversample": 10},
        {"name": "n1_medium_eps04", "n": 1.0, "R_e": 20.0, "eps": 0.4, "pa": float(np.pi / 4), "oversample": 5},
        {"name": "n4_medium_eps04", "n": 4.0, "R_e": 20.0, "eps": 0.4, "pa": float(np.pi / 4), "oversample": 10},
        {"name": "n1_medium_eps07", "n": 1.0, "R_e": 20.0, "eps": 0.7, "pa": float(np.pi / 3), "oversample": 5},
        {"name": "n4_medium_eps06", "n": 4.0, "R_e": 20.0, "eps": 0.6, "pa": float(np.pi / 4), "oversample": 15},
        {"name": "n1_large_circular", "n": 1.0, "R_e": 40.0, "eps": 0.0, "pa": 0.0, "oversample": 5},
        {"name": "n4_large_eps04", "n": 4.0, "R_e": 40.0, "eps": 0.4, "pa": float(np.pi / 4), "oversample": 10},
        {"name": "n1_large_eps06", "n": 1.0, "R_e": 40.0, "eps": 0.6, "pa": float(np.pi / 6), "oversample": 5},
    ]


def scale_test_cases(scale_factor: float) -> List[Dict[str, float]]:
    """Scale radius-sensitive case parameters while preserving morphology."""
    cases = []
    for case in get_base_test_cases():
        scaled = dict(case)
        scaled["R_e"] = float(case["R_e"] * scale_factor)
        suffix = f"scale{scale_factor:g}".replace(".", "p")
        scaled["name"] = f"{case['name']}__{suffix}"
        cases.append(scaled)
    return cases


def create_test_image(
    n: float = 4.0,
    R_e: float = 30.0,
    I_e: float = 1000.0,
    eps: float = 0.4,
    pa: float = float(np.pi / 4),
    oversample: int = 5,
):
    """Create a Sersic image for benchmarking."""
    half_size = int(15 * R_e)
    shape = (2 * half_size, 2 * half_size)
    x0, y0 = half_size, half_size

    b_n = compute_bn(n)
    if oversample > 1:
        oversampled_shape = (shape[0] * oversample, shape[1] * oversample)
        y = np.arange(oversampled_shape[0]) / oversample
        x = np.arange(oversampled_shape[1]) / oversample
        y_grid, x_grid = np.meshgrid(y, x, indexing="ij")
    else:
        y_grid, x_grid = np.mgrid[: shape[0], : shape[1]].astype(np.float64)

    dx = x_grid - x0
    dy = y_grid - y0
    cos_pa, sin_pa = np.cos(pa), np.sin(pa)
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    r_ell = np.sqrt(x_rot**2 + (y_rot / (1 - eps)) ** 2)
    r_ell = np.maximum(r_ell, 0.1)

    image_full = I_e * np.exp(-b_n * ((r_ell / R_e) ** (1 / n) - 1))
    if oversample > 1:
        image = np.zeros(shape)
        for i in range(oversample):
            for j in range(oversample):
                image += image_full[i::oversample, j::oversample]
        image /= oversample**2
    else:
        image = image_full

    return image, (x0, y0)


def summarize_timing_samples(samples: List[float]) -> Dict[str, float]:
    """Compute first-run and steady-state timing diagnostics."""
    if len(samples) == 0:
        return {
            "mean_time": float("nan"),
            "std_time": float("nan"),
            "first_run_time": float("nan"),
            "steady_state_mean_time": float("nan"),
            "steady_state_std_time": float("nan"),
            "coefficient_of_variation": float("nan"),
        }

    samples_array = np.array(samples, dtype=np.float64)
    steady_state = samples_array[1:] if len(samples_array) > 1 else samples_array
    steady_mean = float(np.mean(steady_state))
    steady_std = float(np.std(steady_state))
    return {
        "mean_time": float(np.mean(samples_array)),
        "std_time": float(np.std(samples_array)),
        "first_run_time": float(samples_array[0]),
        "steady_state_mean_time": steady_mean,
        "steady_state_std_time": steady_std,
        "coefficient_of_variation": float(steady_std / steady_mean) if steady_mean > 0 else 0.0,
    }


def run_benchmark_cases(
    test_cases: List[Dict[str, float]],
    n_runs: int,
    disable_numba: bool,
    include_warmup: bool,
    verbose: bool,
):
    """Execute benchmark cases in current process."""
    from isoster import fit_image
    from isoster.config import IsosterConfig

    if disable_numba:
        os.environ["NUMBA_DISABLE_JIT"] = "1"
        numba_available = False
        warmup_seconds = 0.0
    else:
        from isoster.numba_kernels import NUMBA_AVAILABLE, warmup_numba

        numba_available = bool(NUMBA_AVAILABLE)
        if include_warmup and numba_available:
            warmup_start = time.perf_counter()
            warmup_numba()
            warmup_seconds = time.perf_counter() - warmup_start
        else:
            warmup_seconds = 0.0

    results = []
    for case in test_cases:
        image, (x0, y0) = create_test_image(
            n=case["n"],
            R_e=case["R_e"],
            I_e=1000.0,
            eps=case["eps"],
            pa=case["pa"],
            oversample=int(case["oversample"]),
        )

        config = IsosterConfig(
            x0=x0,
            y0=y0,
            eps=case["eps"],
            pa=case["pa"],
            sma0=10.0,
            minsma=3.0,
            maxsma=8 * case["R_e"],
            astep=0.15,
            minit=10,
            maxit=50,
            conver=0.05,
            maxgerr=1.0 if case["eps"] > 0.6 else 0.5,
            use_eccentric_anomaly=(case["eps"] > 0.3),
        )

        timing_samples: List[float] = []
        fit_results = None
        for _ in range(n_runs):
            start = time.perf_counter()
            fit_results = fit_image(image, None, config)
            elapsed = time.perf_counter() - start
            timing_samples.append(float(elapsed))

        isophotes = fit_results["isophotes"] if fit_results is not None else []
        profile = {
            "sma": [iso["sma"] for iso in isophotes],
            "intens": [iso["intens"] for iso in isophotes],
            "eps": [iso["eps"] for iso in isophotes],
            "pa": [iso["pa"] for iso in isophotes],
        }
        timing_summary = summarize_timing_samples(timing_samples)
        result = {
            "name": case["name"],
            "n": case["n"],
            "R_e": case["R_e"],
            "eps": case["eps"],
            "pa": case["pa"],
            "oversample": int(case["oversample"]),
            "n_runs": int(n_runs),
            "times": timing_samples,
            **timing_summary,
            "n_isophotes": int(len(isophotes)),
            "n_converged": int(sum(1 for iso in isophotes if iso["stop_code"] == 0)),
            "profile": profile,
        }
        results.append(result)

        if verbose:
            print(
                f"  {case['name']}: mean={result['mean_time']:.3f}s "
                f"steady={result['steady_state_mean_time']:.3f}s cv={result['coefficient_of_variation']:.3f}"
            )

    return results, numba_available, warmup_seconds


def run_benchmark_without_numba_subprocess(test_cases: List[Dict[str, float]], n_runs: int):
    """Run benchmark in a subprocess with NUMBA_DISABLE_JIT=1."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-no-numba",
        "--cases-json",
        json.dumps(test_cases),
        "--n-runs",
        str(n_runs),
    ]
    environment = os.environ.copy()
    environment["NUMBA_DISABLE_JIT"] = "1"
    result = subprocess.run(command, capture_output=True, text=True, env=environment)
    if result.returncode != 0:
        raise RuntimeError(f"No-numba worker failed: {result.stderr}")
    payload = json.loads(result.stdout)
    return payload["results"]


def validate_results(numba_results, no_numba_results):
    """Validate numerical equivalence between numba and no-numba outputs."""
    validation = {"all_pass": True, "details": []}
    for with_numba, without_numba in zip(numba_results, no_numba_results):
        intens_numba = np.array(with_numba["profile"]["intens"])
        intens_no_numba = np.array(without_numba["profile"]["intens"])
        eps_numba = np.array(with_numba["profile"]["eps"])
        eps_no_numba = np.array(without_numba["profile"]["eps"])
        sma_numba = np.array(with_numba["profile"]["sma"])
        sma_no_numba = np.array(without_numba["profile"]["sma"])

        count_match = with_numba["n_isophotes"] == without_numba["n_isophotes"]
        converged_match = with_numba["n_converged"] == without_numba["n_converged"]
        sma_match = np.allclose(sma_numba, sma_no_numba, rtol=1e-10, atol=1e-12)

        valid_intensity = ~np.isnan(intens_numba) & ~np.isnan(intens_no_numba)
        if np.any(valid_intensity):
            relative_difference = np.abs(intens_numba[valid_intensity] - intens_no_numba[valid_intensity]) / np.abs(
                intens_no_numba[valid_intensity]
            )
            intensity_match = float(np.max(relative_difference)) < 1e-3
            max_relative_intensity_difference = float(np.max(relative_difference))
        else:
            intensity_match = True
            max_relative_intensity_difference = 0.0

        valid_eps = ~np.isnan(eps_numba) & ~np.isnan(eps_no_numba)
        if np.any(valid_eps):
            eps_match = np.allclose(eps_numba[valid_eps], eps_no_numba[valid_eps], atol=1e-4)
            max_eps_difference = float(np.max(np.abs(eps_numba[valid_eps] - eps_no_numba[valid_eps])))
        else:
            eps_match = True
            max_eps_difference = 0.0

        is_valid = bool(count_match and converged_match and sma_match and intensity_match and eps_match)
        validation["all_pass"] = bool(validation["all_pass"] and is_valid)
        validation["details"].append(
            {
                "name": with_numba["name"],
                "pass": is_valid,
                "count_match": bool(count_match),
                "converged_match": bool(converged_match),
                "sma_match": bool(sma_match),
                "intens_match": bool(intensity_match),
                "eps_match": bool(eps_match),
                "max_rel_intens_diff": max_relative_intensity_difference,
                "max_eps_diff": max_eps_difference,
            }
        )
    return validation


def compute_case_diagnostics(numba_results, no_numba_results):
    """Compute per-case speedup diagnostics."""
    diagnostics = []
    for with_numba, without_numba in zip(numba_results, no_numba_results):
        speedup_mean = without_numba["mean_time"] / with_numba["mean_time"]
        speedup_steady = without_numba["steady_state_mean_time"] / with_numba["steady_state_mean_time"]
        numba_compile_overhead = with_numba["first_run_time"] - with_numba["steady_state_mean_time"]
        no_numba_first_run_delta = without_numba["first_run_time"] - without_numba["steady_state_mean_time"]
        diagnostics.append(
            {
                "name": with_numba["name"],
                "speedup_mean": float(speedup_mean),
                "speedup_steady_state": float(speedup_steady),
                "numba_compile_overhead_seconds": float(numba_compile_overhead),
                "no_numba_first_run_delta_seconds": float(no_numba_first_run_delta),
                "numba_cv": float(with_numba["coefficient_of_variation"]),
                "no_numba_cv": float(without_numba["coefficient_of_variation"]),
                "numba_mean_time": float(with_numba["mean_time"]),
                "no_numba_mean_time": float(without_numba["mean_time"]),
            }
        )
    return diagnostics


def summarize_diagnostics(case_diagnostics):
    """Build aggregate summary from per-case diagnostics."""
    speedup_mean = np.array([row["speedup_mean"] for row in case_diagnostics], dtype=np.float64)
    speedup_steady = np.array([row["speedup_steady_state"] for row in case_diagnostics], dtype=np.float64)
    slowdown_mean = [row["name"] for row in case_diagnostics if row["speedup_mean"] < 1.0]
    slowdown_steady = [row["name"] for row in case_diagnostics if row["speedup_steady_state"] < 1.0]
    high_variability = [
        row["name"]
        for row in case_diagnostics
        if row["numba_cv"] > 0.1 or row["no_numba_cv"] > 0.1
    ]

    return {
        "mean_speedup": float(np.mean(speedup_mean)),
        "median_speedup": float(np.median(speedup_mean)),
        "min_speedup": float(np.min(speedup_mean)),
        "max_speedup": float(np.max(speedup_mean)),
        "geometric_mean_speedup": float(np.exp(np.mean(np.log(speedup_mean)))),
        "mean_steady_state_speedup": float(np.mean(speedup_steady)),
        "median_steady_state_speedup": float(np.median(speedup_steady)),
        "min_steady_state_speedup": float(np.min(speedup_steady)),
        "max_steady_state_speedup": float(np.max(speedup_steady)),
        "slowdown_cases_mean": slowdown_mean,
        "slowdown_cases_steady_state": slowdown_steady,
        "high_variability_cases": high_variability,
    }


def write_case_csv(output_path, case_diagnostics, validation_details):
    """Write case-level diagnostics CSV."""
    validation_map = {row["name"]: row for row in validation_details}
    field_names = [
        "name",
        "numba_mean_time",
        "no_numba_mean_time",
        "speedup_mean",
        "speedup_steady_state",
        "numba_compile_overhead_seconds",
        "no_numba_first_run_delta_seconds",
        "numba_cv",
        "no_numba_cv",
        "validation_pass",
        "max_rel_intens_diff",
        "max_eps_diff",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as file_pointer:
        writer = csv.DictWriter(file_pointer, fieldnames=field_names)
        writer.writeheader()
        for row in case_diagnostics:
            validation_row = validation_map[row["name"]]
            writer.writerow(
                {
                    **{key: row[key] for key in field_names if key in row},
                    "validation_pass": validation_row["pass"],
                    "max_rel_intens_diff": validation_row["max_rel_intens_diff"],
                    "max_eps_diff": validation_row["max_eps_diff"],
                }
            )


def generate_speedup_plot(output_path, case_diagnostics):
    """Persist mean/steady-state speedup diagnostic plots."""
    names = [row["name"] for row in case_diagnostics]
    speedup_mean = [row["speedup_mean"] for row in case_diagnostics]
    speedup_steady = [row["speedup_steady_state"] for row in case_diagnostics]
    compile_overhead = [row["numba_compile_overhead_seconds"] for row in case_diagnostics]

    figure, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    x_index = np.arange(len(names))

    axes[0].bar(x_index, speedup_mean, color="steelblue", alpha=0.85, label="Mean speedup")
    axes[0].axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="No speedup")
    axes[0].set_ylabel("Mean Speedup")
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(x_index, speedup_steady, color="seagreen", alpha=0.85, label="Steady-state speedup")
    axes[1].axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="No speedup")
    axes[1].set_ylabel("Steady-State Speedup")
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")

    axes[2].bar(x_index, compile_overhead, color="darkorange", alpha=0.85, label="Numba first-run overhead")
    axes[2].axhline(0.0, color="black", linestyle="-", linewidth=1.0)
    axes[2].set_ylabel("Overhead (s)")
    axes[2].set_xlabel("Benchmark Case")
    axes[2].legend()
    axes[2].grid(alpha=0.3, axis="y")
    axes[2].set_xticks(x_index)
    axes[2].set_xticklabels(names, rotation=45, ha="right")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def run_worker_mode(args):
    """Execute no-numba worker mode and write compact JSON to stdout."""
    os.environ["NUMBA_DISABLE_JIT"] = "1"
    cases = json.loads(args.cases_json)
    results, _, _ = run_benchmark_cases(
        test_cases=cases,
        n_runs=args.n_runs,
        disable_numba=True,
        include_warmup=False,
        verbose=False,
    )
    print(json.dumps({"results": results}))
    return 0


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description="Benchmark numba speedup with enhanced diagnostics.")
    parser.add_argument("--output", "-o", default=None, help="Output directory override.")
    parser.add_argument("--n-runs", type=int, default=5, help="Timing repetitions per case.")
    parser.add_argument("--scale-factor", type=float, default=1.0, help="Scale factor applied to R_e in all test cases.")
    parser.add_argument("--skip-numba-warmup", action="store_true", help="Disable explicit numba warmup.")

    # Hidden worker mode used by subprocess no-numba execution.
    parser.add_argument("--worker-no-numba", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cases-json", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_no_numba:
        return run_worker_mode(args)

    n_runs = max(1, int(args.n_runs))
    scale_factor = max(0.1, float(args.scale_factor))
    test_cases = scale_test_cases(scale_factor=scale_factor)

    print("=" * 78)
    print("ISOSTER NUMBA PERFORMANCE BENCHMARK (DIAGNOSTIC MODE)")
    print("=" * 78)
    print(f"n_runs={n_runs}, scale_factor={scale_factor}")
    print()

    print("Running benchmark WITH numba...")
    numba_results, numba_available, warmup_seconds = run_benchmark_cases(
        test_cases=test_cases,
        n_runs=n_runs,
        disable_numba=False,
        include_warmup=not args.skip_numba_warmup,
        verbose=True,
    )

    print()
    print("Running benchmark WITHOUT numba (NUMBA_DISABLE_JIT=1)...")
    no_numba_results = run_benchmark_without_numba_subprocess(test_cases=test_cases, n_runs=n_runs)

    print()
    print("Validating numerical results...")
    validation = validate_results(numba_results, no_numba_results)
    diagnostics = compute_case_diagnostics(numba_results, no_numba_results)
    summary = summarize_diagnostics(diagnostics)

    print()
    print("-" * 78)
    print(f"{'Case':<34} {'mean':>8} {'steady':>8} {'valid':>8}")
    print("-" * 78)
    validation_by_name = {row["name"]: row for row in validation["details"]}
    for row in diagnostics:
        valid_flag = "✓" if validation_by_name[row["name"]]["pass"] else "✗"
        print(
            f"{row['name']:<34} {row['speedup_mean']:>8.2f}x "
            f"{row['speedup_steady_state']:>8.2f}x {valid_flag:>8}"
        )
    print("-" * 78)
    print(f"Numba available: {numba_available}")
    print(f"Numba warmup time: {warmup_seconds:.4f}s")
    print(f"Mean speedup: {summary['mean_speedup']:.2f}x (median {summary['median_speedup']:.2f}x)")
    print(
        f"Steady-state mean speedup: {summary['mean_steady_state_speedup']:.2f}x "
        f"(median {summary['median_steady_state_speedup']:.2f}x)"
    )
    print(f"Slowdown cases (mean): {len(summary['slowdown_cases_mean'])}")
    print(f"Slowdown cases (steady-state): {len(summary['slowdown_cases_steady_state'])}")
    print(f"High variability cases: {len(summary['high_variability_cases'])}")
    print()

    output_dir = resolve_output_directory(
        "benchmarks_performance",
        "bench_numba_speedup",
        explicit_output_directory=args.output,
    )
    payload = {
        "environment": collect_environment_metadata(project_root=PROJECT_ROOT),
        "numba_available": numba_available,
        "numba_warmup_seconds": float(warmup_seconds),
        "n_runs": int(n_runs),
        "scale_factor": float(scale_factor),
        "numba_results": numba_results,
        "no_numba_results": no_numba_results,
        "validation": validation,
        "case_diagnostics": diagnostics,
        "summary": summary,
    }
    json_path = output_dir / "numba_benchmark_results.json"
    csv_path = output_dir / "numba_benchmark_results.csv"
    plot_path = output_dir / "numba_speedup.png"

    write_json(json_path, payload)
    write_case_csv(csv_path, diagnostics, validation["details"])
    generate_speedup_plot(plot_path, diagnostics)

    print(f"Results saved to: {json_path}")
    print(f"CSV summary saved to: {csv_path}")
    print(f"Speedup diagnostics plot saved to: {plot_path}")

    return 0 if validation["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
