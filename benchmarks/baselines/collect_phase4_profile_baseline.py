"""
Collect Phase 4 baseline profile metrics for integration-test scenarios.

This script runs representative Sersic scenarios, computes quantitative 1-D
profile residual metrics, and writes a machine-readable JSON artifact for
threshold calibration.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.utils.run_metadata import collect_environment_metadata  # noqa: E402
from benchmarks.utils.sersic_model import create_sersic_image_vectorized, sersic_1d  # noqa: E402
from isoster import fit_image  # noqa: E402
from isoster.config import IsosterConfig  # noqa: E402
from isoster.output_paths import resolve_output_directory  # noqa: E402


def run_baseline_case(case_definition: dict) -> dict:
    """Run one baseline case and return measured metrics."""
    half_size = max(int(15 * case_definition["R_e"]), 150)
    shape = (2 * half_size, 2 * half_size)
    center = (float(half_size), float(half_size))

    image, _ = create_sersic_image_vectorized(
        n=case_definition["n"],
        R_e=case_definition["R_e"],
        I_e=case_definition["I_e"],
        eps=case_definition["eps"],
        pa=case_definition["pa"],
        shape=shape,
        center=center,
        oversample=case_definition["oversample"],
    )

    noise_snr = case_definition.get("noise_snr")
    if noise_snr is not None:
        rng = np.random.default_rng(case_definition["noise_seed"])
        noise_sigma = case_definition["I_e"] / noise_snr
        image = image + rng.normal(0.0, noise_sigma, image.shape)

    config_values = dict(case_definition["config"])
    config_values.setdefault("x0", center[0])
    config_values.setdefault("y0", center[1])
    config = IsosterConfig(**config_values)
    results = fit_image(image, mask=None, config=config)
    isophotes = results["isophotes"]

    sma = np.array([iso["sma"] for iso in isophotes], dtype=np.float64)
    intens = np.array([iso["intens"] for iso in isophotes], dtype=np.float64)
    stop_codes = np.array([iso["stop_code"] for iso in isophotes], dtype=np.int64)

    valid = (
        (sma >= case_definition["metric_sma_min"])
        & (sma <= case_definition["metric_sma_max"])
        & (stop_codes == 0)
    )

    valid_count = int(valid.sum())
    if valid_count <= 0:
        raise RuntimeError(
            f"{case_definition['name']}: no valid converged points in metric window "
            f"[{case_definition['metric_sma_min']:.2f}, {case_definition['metric_sma_max']:.2f}]"
        )

    true_intensity = sersic_1d(
        sma[valid],
        I_e=case_definition["I_e"],
        R_e=case_definition["R_e"],
        n=case_definition["n"],
    )
    delta_intensity = (intens[valid] - true_intensity) / true_intensity
    abs_delta_intensity = np.abs(delta_intensity)

    stop_code_distribution = {
        str(code): int(count)
        for code, count in sorted(Counter(stop_codes.tolist()).items())
    }

    return {
        "name": case_definition["name"],
        "parameters": {
            "n": case_definition["n"],
            "R_e": case_definition["R_e"],
            "I_e": case_definition["I_e"],
            "eps": case_definition["eps"],
            "pa": case_definition["pa"],
            "oversample": case_definition["oversample"],
            "noise_snr": noise_snr,
        },
        "metric_window": {
            "sma_min": float(case_definition["metric_sma_min"]),
            "sma_max": float(case_definition["metric_sma_max"]),
        },
        "valid_point_count": valid_count,
        "isophote_count": int(len(isophotes)),
        "converged_count": int((stop_codes == 0).sum()),
        "median_abs_delta_intensity": float(np.median(abs_delta_intensity)),
        "max_abs_delta_intensity": float(np.max(abs_delta_intensity)),
        "stop_code_distribution": stop_code_distribution,
    }


def collect_baseline_metrics() -> dict:
    """Run all baseline scenarios and return collected metrics."""
    scenarios = [
        {
            "name": "sersic_n4_noiseless",
            "n": 4.0,
            "R_e": 20.0,
            "I_e": 2000.0,
            "eps": 0.4,
            "pa": float(np.pi / 4.0),
            "oversample": 10,
            "noise_snr": None,
            "metric_sma_min": max(3.0, 0.5 * 20.0),
            "metric_sma_max": 8.0 * 20.0,
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
            "metric_sma_min": max(3.0, 0.5 * 25.0),
            "metric_sma_max": 5.0 * 25.0,
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
            "metric_sma_min": max(5.0, 0.5 * 20.0),
            "metric_sma_max": 5.0 * 20.0,
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

    case_metrics = [run_baseline_case(case) for case in scenarios]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": collect_environment_metadata(project_root=PROJECT_ROOT),
        "cases": case_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect Phase 4 baseline profile metrics.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional explicit output directory.",
    )
    args = parser.parse_args()

    output_directory = resolve_output_directory(
        "tests_integration",
        "baseline_metrics",
        explicit_output_directory=args.output,
    )
    output_path = output_directory / "phase4_profile_baseline_metrics.json"

    baseline_metrics = collect_baseline_metrics()
    with output_path.open("w", encoding="utf-8") as file_pointer:
        json.dump(baseline_metrics, file_pointer, indent=2, sort_keys=True)

    print(f"Saved baseline metrics to: {output_path}")
    for case in baseline_metrics["cases"]:
        print(
            f"- {case['name']}: valid={case['valid_point_count']}, "
            f"median|dI|={case['median_abs_delta_intensity']:.6f}, "
            f"max|dI|={case['max_abs_delta_intensity']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
