"""Lock efficiency thresholds directly from measured baseline benchmark metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

DEFAULT_INPUT_PATH = Path("outputs/benchmark_performance/bench_efficiency/efficiency_benchmark_results.json")
DEFAULT_OUTPUT_PATH = Path("benchmarks/baselines/efficiency_thresholds.json")


def _load_cases(payload: object) -> List[Dict[str, object]]:
    """Normalize supported efficiency baseline payloads to a case list."""
    if isinstance(payload, list):
        return [dict(case) for case in payload]
    if isinstance(payload, dict) and "cases" in payload:
        return [dict(case) for case in payload.get("cases", [])]
    raise ValueError("Unsupported efficiency baseline format; expected list or {'cases': [...]} payload")


def lock_efficiency_thresholds(input_path: Path) -> dict:
    """Create locked efficiency thresholds from measured benchmark outputs."""
    with input_path.open("r", encoding="utf-8") as file_pointer:
        payload = json.load(file_pointer)

    cases = _load_cases(payload)
    locked_cases = []
    for case in cases:
        time_samples = [float(value) for value in case.get("times", [])]
        if not time_samples:
            time_samples = [float(case["mean_time"])]

        locked_cases.append(
            {
                "name": str(case["name"]),
                "max_mean_time_threshold": float(max(time_samples)),
                "minimum_convergence_rate_threshold": float(case["convergence_rate"]),
                "minimum_converged_count_threshold": int(case["n_converged"]),
                "minimum_isophote_count_threshold": int(case["n_isophotes"]),
            }
        )

    return {
        "policy": (
            "Thresholds are locked directly from measured efficiency baselines with no synthetic margins. "
            "Runtime threshold uses the measured maximum sampled runtime for each case."
        ),
        "source_baseline_path": str(input_path),
        "cases": locked_cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Lock efficiency thresholds from measured benchmark baselines.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_PATH), help="Baseline efficiency JSON path.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Output threshold JSON path.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Efficiency baseline file not found: {input_path}")

    threshold_payload = lock_efficiency_thresholds(input_path=input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_pointer:
        json.dump(threshold_payload, file_pointer, indent=2, sort_keys=True)

    print(f"Locked efficiency thresholds written to: {output_path}")
    for case in threshold_payload["cases"]:
        print(
            f"- {case['name']}: mean_time<={case['max_mean_time_threshold']:.6f}s, "
            f"conv_rate>={case['minimum_convergence_rate_threshold']:.6f}, "
            f"n_conv>={case['minimum_converged_count_threshold']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
