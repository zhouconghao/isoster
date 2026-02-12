"""Lock Phase 4 thresholds directly from measured baseline metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_INPUT_PATH = Path("outputs/tests_integration/baseline_metrics/phase4_profile_baseline_metrics.json")
DEFAULT_OUTPUT_PATH = Path("benchmarks/baselines/phase4_profile_thresholds_2026-02-11.json")


def lock_thresholds(input_path: Path) -> dict:
    """Create threshold payload from measured baseline metrics."""
    with input_path.open("r", encoding="utf-8") as file_pointer:
        baseline_payload = json.load(file_pointer)

    threshold_cases = []
    for case in baseline_payload.get("cases", []):
        threshold_cases.append(
            {
                "name": case["name"],
                "metric_window": case["metric_window"],
                "locked_from_run_generated_at_utc": baseline_payload.get("generated_at_utc"),
                "minimum_valid_point_count": int(case["valid_point_count"]),
                "max_abs_delta_intensity_threshold": float(case["max_abs_delta_intensity"]),
                "median_abs_delta_intensity_threshold": float(case["median_abs_delta_intensity"]),
            }
        )

    return {
        "policy": "Thresholds are locked directly from measured baselines; no synthetic margins added.",
        "source_baseline_path": str(input_path),
        "source_environment": baseline_payload.get("environment", {}),
        "cases": threshold_cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Lock Phase 4 thresholds from baseline metrics.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_PATH), help="Baseline metrics JSON path.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Threshold output JSON path.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Baseline metrics file not found: {input_path}")

    threshold_payload = lock_thresholds(input_path=input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_pointer:
        json.dump(threshold_payload, file_pointer, indent=2, sort_keys=True)

    print(f"Locked thresholds written to: {output_path}")
    for case in threshold_payload["cases"]:
        print(
            f"- {case['name']}: min_valid={case['minimum_valid_point_count']}, "
            f"max|dI|<={case['max_abs_delta_intensity_threshold']:.6f}, "
            f"median|dI|<={case['median_abs_delta_intensity_threshold']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
