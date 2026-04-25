"""Driver for running the exhausted benchmark over the scenario mock grids.

Wraps ``benchmarks.exhausted.orchestrator`` so routine sweeps over the
Huang2013 and S4G-like mock matrices (depth in {clean, wide, deep},
redshift tag in {005, 020, 035, 050}) can be launched with a compact
CLI rather than hand-written campaign YAML. A campaign YAML is still
assembled internally and written under the campaign output directory,
so every run is reproducible from the frozen snapshot.

Example invocations (two-galaxy smoke, then a full depth sweep)::

    uv run python -m benchmarks.exhausted.campaigns.run_mock_campaigns \\
        --dataset huang2013 --depth wide --redshift 020 \\
        --select IC1459 NGC1600 --tools isoster,autoprof \\
        --max-parallel 2 --dry-run

    uv run python -m benchmarks.exhausted.campaigns.run_mock_campaigns \\
        --dataset huang2013 --depth all --redshift 020 \\
        --select IC1459 --tools isoster,photutils,autoprof \\
        --max-parallel 4

Defaults: huang2013 / depth=wide / redshift=020 /
tools=isoster+photutils+autoprof (in that order) / max-parallel=4 /
skip-existing=true. A full 3-tool x 837-scenario x 33-arm run is
opt-in (pass --depth all --redshift all and expect ~28 000 fits for
huang2013 alone). Override --tools for fast iteration, e.g.
``--tools isoster`` or ``--tools isoster,autoprof``.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..adapters.huang2013_scenarios import (
    SCENARIO_DEPTHS,
    SCENARIO_REDSHIFT_TAGS,
)
from ..orchestrator.cli import _print_dataset_matrix, _print_header, _print_tools
from ..orchestrator.config_loader import load_campaign
from ..orchestrator.runner import run_campaign

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "benchmarks" / "exhausted" / "configs"

DATASET_DEFAULTS = {
    "huang2013": {
        "adapter": "huang2013_scenarios",
        "root": "/Volumes/galaxy/isophote/huang2013",
    },
    "s4g": {
        "adapter": "s4g_scenarios",
        "root": "/Volumes/galaxy/isophote/s4g_mock",
    },
}

TOOL_ARMS_FILES = {
    "isoster": "benchmarks/exhausted/configs/isoster_arms.yaml",
    "photutils": "benchmarks/exhausted/configs/photutils_arms.yaml",
    "autoprof": "benchmarks/exhausted/configs/autoprof_arms.yaml",
}

DEFAULT_OUTPUT_ROOT = "outputs/benchmark_exhausted"
DEFAULT_AUTOPROF_VENV = "~/.venvs/autoprof_venv/bin/python"
DEFAULT_AUTOPROF_TIMEOUT = 300


@dataclass
class CliArgs:
    datasets: list[str]
    depths: list[str]
    redshift_tags: list[str]
    selects: list[str]
    tools: list[str]
    max_parallel: int
    skip_existing: bool
    campaign_name: str | None
    dry_run: bool
    write_yaml: Path | None
    output_root: str
    autoprof_venv: str
    autoprof_timeout: int
    root_huang2013: str
    root_s4g: str
    harmonic_sweeps: list[list[int]]


def _parse_depths(value: str) -> list[str]:
    if value == "all":
        return list(SCENARIO_DEPTHS)
    items = [v.strip() for v in value.split(",") if v.strip()]
    bad = [v for v in items if v not in SCENARIO_DEPTHS]
    if bad:
        raise argparse.ArgumentTypeError(
            f"unknown depth tag(s) {bad}; valid = {SCENARIO_DEPTHS + ('all',)}"
        )
    return items


def _parse_redshifts(value: str) -> list[str]:
    if value == "all":
        return list(SCENARIO_REDSHIFT_TAGS)
    items = [v.strip() for v in value.split(",") if v.strip()]
    bad = [v for v in items if v not in SCENARIO_REDSHIFT_TAGS]
    if bad:
        raise argparse.ArgumentTypeError(
            f"unknown redshift tag(s) {bad}; valid = {SCENARIO_REDSHIFT_TAGS + ('all',)}"
        )
    return items


def _parse_tools(value: str) -> list[str]:
    if value == "all":
        return ["isoster", "photutils", "autoprof"]
    items = [v.strip() for v in value.split(",") if v.strip()]
    bad = [v for v in items if v not in TOOL_ARMS_FILES]
    if bad:
        raise argparse.ArgumentTypeError(
            f"unknown tool(s) {bad}; valid = {list(TOOL_ARMS_FILES) + ['all']}"
        )
    return items


def _parse_datasets(value: str) -> list[str]:
    if value == "both":
        return ["huang2013", "s4g"]
    if value not in DATASET_DEFAULTS:
        raise argparse.ArgumentTypeError(
            f"unknown dataset {value!r}; valid = {list(DATASET_DEFAULTS) + ['both']}"
        )
    return [value]


def _parse_harmonic_sweeps(value: str) -> list[list[int]]:
    """Accept ``5_6,5_6_7_8`` or the empty string (= no expansion)."""
    if not value.strip():
        return []
    sweeps: list[list[int]] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            sweeps.append([int(n) for n in chunk.split("_")])
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"harmonic sweep '{chunk}' must be underscore-separated ints"
            ) from exc
    return sweeps


def parse_args(argv: list[str] | None = None) -> CliArgs:
    parser = argparse.ArgumentParser(
        prog="run_mock_campaigns",
        description=(
            "Run the exhausted benchmark over Huang2013 / S4G scenario mocks."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=_parse_datasets,
        default=["huang2013"],
        help="which dataset to run: huang2013 | s4g | both (default: huang2013)",
    )
    parser.add_argument(
        "--depth",
        type=_parse_depths,
        default=["wide"],
        help=(
            "depth tag(s): clean | wide | deep | all | comma-list "
            "(default: wide)"
        ),
    )
    parser.add_argument(
        "--redshift",
        type=_parse_redshifts,
        default=["020"],
        help=(
            "redshift tag(s): 005 | 010 | 020 | 035 | 050 | all | comma-list "
            "(default: 020). huang2013 has {005, 020, 035, 050}; newer "
            "S4G-like regens have {005, 010}."
        ),
    )
    parser.add_argument(
        "--select",
        nargs="*",
        default=None,
        help=(
            "optional list of galaxy names (e.g. IC1459 NGC1600). The driver "
            "expands to <name>/<depth>_z<zzz> for every selected scenario "
            "and writes that as the campaign YAML select: list. If omitted, "
            "all galaxies the adapter enumerates (subject to depth/redshift "
            "filters) are run."
        ),
    )
    parser.add_argument(
        "--tools",
        type=_parse_tools,
        default=["isoster", "photutils", "autoprof"],
        help=(
            "tools to enable (runs in isoster -> photutils -> autoprof "
            "order regardless of argument order): comma-list of "
            "{isoster, photutils, autoprof} or 'all'. Default: all three."
        ),
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=4,
        help="execution.max_parallel_galaxies (default: 4)",
    )
    parser.add_argument(
        "--skip-existing",
        choices=("true", "false"),
        default="true",
        help="skip arms whose profile.fits already exists (default: true)",
    )
    parser.add_argument(
        "--campaign-name",
        default=None,
        help=(
            "campaign subdirectory name. Default: "
            "<dataset>_<depths>_z<redshifts>[_<select-fp>]."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planned fit matrix and exit without running any fits.",
    )
    parser.add_argument(
        "--write-yaml",
        type=Path,
        default=None,
        help="also persist the generated campaign YAML at this path.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help=f"campaign output_root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--autoprof-venv",
        default=DEFAULT_AUTOPROF_VENV,
        help=f"autoprof venv python (default: {DEFAULT_AUTOPROF_VENV})",
    )
    parser.add_argument(
        "--autoprof-timeout",
        type=int,
        default=DEFAULT_AUTOPROF_TIMEOUT,
        help=f"autoprof per-arm timeout in seconds (default: {DEFAULT_AUTOPROF_TIMEOUT})",
    )
    parser.add_argument(
        "--root-huang2013",
        default=DATASET_DEFAULTS["huang2013"]["root"],
        help="override huang2013 data root",
    )
    parser.add_argument(
        "--root-s4g",
        default=DATASET_DEFAULTS["s4g"]["root"],
        help="override s4g data root",
    )
    parser.add_argument(
        "--harmonic-sweeps",
        type=_parse_harmonic_sweeps,
        default=[[5, 6]],
        help=(
            "isoster_harmonic_sweeps as comma-separated underscore lists "
            "(default: '5_6'; pass '' to disable)"
        ),
    )

    ns = parser.parse_args(argv)

    return CliArgs(
        datasets=ns.dataset,
        depths=ns.depth,
        redshift_tags=ns.redshift,
        selects=list(ns.select) if ns.select else [],
        tools=ns.tools,
        max_parallel=ns.max_parallel,
        skip_existing=(ns.skip_existing == "true"),
        campaign_name=ns.campaign_name,
        dry_run=ns.dry_run,
        write_yaml=ns.write_yaml,
        output_root=ns.output_root,
        autoprof_venv=ns.autoprof_venv,
        autoprof_timeout=ns.autoprof_timeout,
        root_huang2013=ns.root_huang2013,
        root_s4g=ns.root_s4g,
        harmonic_sweeps=ns.harmonic_sweeps,
    )


def _default_campaign_name(args: CliArgs) -> str:
    ds_part = "_".join(args.datasets)
    depth_part = "".join(sorted(args.depths))
    z_part = "z" + "".join(sorted(args.redshift_tags))
    parts = [ds_part, depth_part, z_part]
    if args.selects:
        fingerprint = "+".join(args.selects) if len(args.selects) <= 3 else (
            f"{args.selects[0]}+{len(args.selects) - 1}more"
        )
        parts.append(fingerprint)
    return "mock_" + "_".join(parts)


def build_campaign_dict(args: CliArgs) -> dict[str, Any]:
    """Assemble a campaign YAML dict from the parsed CLI args."""
    roots = {
        "huang2013": args.root_huang2013,
        "s4g": args.root_s4g,
    }

    datasets: dict[str, Any] = {}
    for ds_name in args.datasets:
        entry: dict[str, Any] = {
            "enabled": True,
            "adapter": DATASET_DEFAULTS[ds_name]["adapter"],
            "root": roots[ds_name],
            "depths": list(args.depths),
            "redshift_tags": list(args.redshift_tags),
        }
        if args.selects:
            entry["select"] = [
                f"{galaxy}/{depth}_z{z}"
                for galaxy in args.selects
                for depth in args.depths
                for z in args.redshift_tags
            ]
        datasets[ds_name] = entry

    tools: dict[str, Any] = {}
    for tool_name, arms_file in TOOL_ARMS_FILES.items():
        tool_entry: dict[str, Any] = {
            "enabled": tool_name in args.tools,
            "arms_file": arms_file,
        }
        if tool_name == "autoprof":
            tool_entry["venv_python"] = args.autoprof_venv
            tool_entry["timeout"] = args.autoprof_timeout
        tools[tool_name] = tool_entry

    campaign_name = args.campaign_name or _default_campaign_name(args)

    return {
        "campaign_name": campaign_name,
        "output_root": args.output_root,
        "tools": tools,
        "isoster_harmonic_sweeps": args.harmonic_sweeps,
        "datasets": datasets,
        "qa": {
            "per_galaxy_qa": True,
            "cross_arm_overlay": True,
            "cross_tool_comparison": True,
            "summary_grids": False,
            "residual_models": True,
        },
        "execution": {
            "max_parallel_galaxies": args.max_parallel,
            "skip_existing": args.skip_existing,
            "dry_run": False,
            "fail_fast": False,
        },
        "summary": {
            "per_galaxy_inventory": True,
            "per_tool_cross_arm_table": True,
            "cross_tool_table": True,
            "composite_score_weights": {
                "resid_inner": 1.0,
                "resid_mid": 1.0,
                "resid_outer": 2.0,
                "centroid_drift": 1.0,
                "centroid_tol_pix": 2.0,
                "n_stop_m1": 2.0,
                "frac_stop_nonzero": 5.0,
                "n_iso_completeness": 3.0,
                "wall_time": 0.1,
            },
            "cross_tool_score_weights": {
                "resid_inner": 1.0,
                "resid_mid": 1.0,
                "resid_outer": 2.0,
                "wall_time": 0.1,
            },
        },
    }


def _write_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, indent=2)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Sanity: complain early if any dataset root is missing. Keeps the
    # failure mode at CLI parse time rather than after the YAML loads.
    missing: list[str] = []
    for ds in args.datasets:
        root = args.root_huang2013 if ds == "huang2013" else args.root_s4g
        if not Path(root).expanduser().is_dir():
            missing.append(f"{ds}={root}")
    if missing:
        print(
            "error: dataset root directory not found:\n  "
            + "\n  ".join(missing)
            + "\nOverride with --root-huang2013 / --root-s4g.",
            file=sys.stderr,
        )
        return 2

    campaign_dict = build_campaign_dict(args)

    if args.write_yaml is not None:
        _write_yaml(campaign_dict, args.write_yaml)
        print(f"Wrote generated campaign YAML: {args.write_yaml}")

    # Write a tempfile so load_campaign can read from disk; the runner
    # snapshots this under <output_root>/<campaign_name>/campaign.yaml.
    tmp_path: Path
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="mock_campaign_"
    ) as handle:
        yaml.safe_dump(campaign_dict, handle, sort_keys=False, indent=2)
        tmp_path = Path(handle.name)

    try:
        plan = load_campaign(tmp_path)
    except Exception as exc:  # noqa: BLE001 - surface any config error
        print(f"error: campaign load failed: {exc}", file=sys.stderr)
        return 2

    _print_header(plan)
    _print_tools(plan)

    if args.dry_run:
        total = 0
        for ds_plan in plan.datasets.values():
            if not ds_plan.enabled:
                continue
            total += _print_dataset_matrix(plan, ds_plan, max_galaxies=10)
        print("-" * 72)
        print(f"Total planned fits across all datasets and tools: {total}")
        print(f"(generated YAML: {tmp_path})")
        return 0

    summary = run_campaign(plan)
    print("=" * 72)
    print(f"Total requested:        {summary.total_requested}")
    print(f"  Ran:                  {summary.total_ran}")
    print(f"  Skipped (cached):     {summary.total_skipped_existing}")
    print(f"  Skipped (arm):        {summary.total_skipped_arm}")
    print(f"  Ok:                   {summary.total_ok}")
    print(f"  Failed:               {summary.total_failed}")
    print(f"(generated YAML: {tmp_path})")
    return 0 if summary.total_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
