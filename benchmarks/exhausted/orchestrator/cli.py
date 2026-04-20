"""Top-level campaign CLI.

Usage::

    uv run python -m benchmarks.exhausted.orchestrator.cli dry-run <campaign.yaml>

Phase A implements only ``dry-run``, which loads the campaign, resolves
each adapter, and prints the planned ``(dataset, galaxy, tool, arm)``
matrix without running any fits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config_loader import CampaignPlan, DatasetPlan, load_campaign
from .runner import run_campaign


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="isoster-campaign",
        description="Exhausted benchmark campaign driver (Phase A: dry-run only).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    dry = subparsers.add_parser(
        "dry-run",
        help="Load the campaign YAML, resolve adapters, print the matrix.",
    )
    dry.add_argument("campaign_yaml", type=Path, help="Path to the campaign YAML.")
    dry.add_argument(
        "--max-galaxies",
        type=int,
        default=None,
        help="Cap galaxies printed per dataset (does not affect the count).",
    )

    run = subparsers.add_parser(
        "run",
        help="Run the campaign (Phase B: isoster-only, sequential).",
    )
    run.add_argument("campaign_yaml", type=Path)

    args = parser.parse_args(argv)

    if args.command == "dry-run":
        return _cmd_dry_run(args.campaign_yaml, args.max_galaxies)
    if args.command == "run":
        return _cmd_run(args.campaign_yaml)
    return 2


def _cmd_run(yaml_path: Path) -> int:
    plan = load_campaign(yaml_path)
    _print_header(plan)
    _print_tools(plan)
    summary = run_campaign(plan)
    print("=" * 72)
    print(f"Total requested:        {summary.total_requested}")
    print(f"  Ran:                  {summary.total_ran}")
    print(f"  Skipped (cached):     {summary.total_skipped_existing}")
    print(f"  Skipped (arm):        {summary.total_skipped_arm}")
    print(f"  Ok:                   {summary.total_ok}")
    print(f"  Failed:               {summary.total_failed}")
    return 0 if summary.total_failed == 0 else 1


def _cmd_dry_run(yaml_path: Path, max_galaxies: int | None) -> int:
    plan = load_campaign(yaml_path)
    _print_header(plan)
    _print_tools(plan)
    total_fits = 0
    for ds_plan in plan.datasets.values():
        if not ds_plan.enabled:
            print(f"[skip] dataset '{ds_plan.name}' disabled in YAML")
            continue
        ds_fits = _print_dataset_matrix(plan, ds_plan, max_galaxies)
        total_fits += ds_fits
    print("-" * 72)
    print(f"Total planned fits across all datasets and tools: {total_fits}")
    return 0


def _print_header(plan: CampaignPlan) -> None:
    print("=" * 72)
    print(f"Campaign:     {plan.campaign_name}")
    print(f"Output root:  {plan.output_root}")
    print(f"Harm sweeps:  {plan.isoster_harmonic_sweeps}")
    print(f"Execution:    {plan.execution}")
    print("=" * 72)


def _print_tools(plan: CampaignPlan) -> None:
    print("Tools:")
    for tool in plan.tools.values():
        flag = "ENABLED " if tool.enabled else "disabled"
        print(f"  [{flag}] {tool.name:10s}  arms_file={tool.arms_file}")
        if tool.enabled:
            arms_preview = list(tool.arms.keys())
            print(f"            {len(arms_preview)} arms: {arms_preview}")
    print()


def _print_dataset_matrix(
    plan: CampaignPlan, ds_plan: DatasetPlan, max_galaxies: int | None
) -> int:
    adapter = ds_plan.adapter
    galaxy_ids = adapter.list_galaxies()
    if ds_plan.select:
        allow = set(ds_plan.select)
        galaxy_ids = [g for g in galaxy_ids if g in allow]

    enabled_tools = [t for t in plan.tools.values() if t.enabled]
    total_arms_per_galaxy = sum(len(t.arms) for t in enabled_tools)
    total_fits = len(galaxy_ids) * total_arms_per_galaxy

    print(f"Dataset: {ds_plan.name}  (adapter={ds_plan.adapter_name})")
    print(f"  root:     {getattr(adapter, 'root', '?')}")
    print(f"  galaxies: {len(galaxy_ids)}")
    print(f"  arms/galaxy: {total_arms_per_galaxy}")
    print(f"  planned fits: {total_fits}")

    preview = galaxy_ids if max_galaxies is None else galaxy_ids[:max_galaxies]
    if preview:
        print("  first galaxies:")
        for gid in preview:
            print(f"    - {gid}")
        if max_galaxies is not None and len(galaxy_ids) > max_galaxies:
            print(f"    ... and {len(galaxy_ids) - max_galaxies} more")
    print()
    return total_fits


if __name__ == "__main__":
    raise SystemExit(main())
