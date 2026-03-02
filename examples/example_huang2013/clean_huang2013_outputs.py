#!/usr/bin/env python3
"""Clean Huang2013 generated artifacts while preserving mock inputs.

Preserved files in each galaxy folder:
- mock FITS inputs matching <GALAXY>_<TEST>.fits (single suffix token only)
- mosaic image <GALAXY>_mosaic.png

Supported cleanup scopes:
- one case: --galaxy <GALAXY> --test-name <TEST_NAME>
- one galaxy (all tests): --galaxy <GALAXY>
- all galaxies: --all-galaxies
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re


DEFAULT_HUANG_ROOT = Path("/Users/mac/work/hsc/huang2013")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Clean Huang2013 generated files and keep mock FITS/mosaic inputs.",
    )
    parser.add_argument(
        "--huang-root",
        type=Path,
        default=DEFAULT_HUANG_ROOT,
        help="Root directory containing per-galaxy folders.",
    )
    parser.add_argument(
        "--galaxy",
        default=None,
        help="Galaxy folder name (for single-galaxy cleanup).",
    )
    parser.add_argument(
        "--test-name",
        default=None,
        help="Single test name under a galaxy, e.g. mock1.",
    )
    parser.add_argument(
        "--all-galaxies",
        action="store_true",
        help="Clean all galaxy folders under --huang-root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would be removed without deleting.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file action details.",
    )
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    """Validate argument combinations."""
    if arguments.all_galaxies and arguments.galaxy is not None:
        raise ValueError("Use either --all-galaxies or --galaxy, not both.")
    if not arguments.all_galaxies and arguments.galaxy is None:
        raise ValueError("Set --galaxy for single-galaxy cleanup, or use --all-galaxies.")
    if arguments.test_name is not None and arguments.galaxy is None:
        raise ValueError("--test-name requires --galaxy.")


def discover_galaxies(huang_root: Path) -> list[str]:
    """Discover galaxy folder names under root."""
    if not huang_root.exists():
        return []
    return sorted(path.name for path in huang_root.iterdir() if path.is_dir())


def is_preserved_mock_fits(file_name: str, galaxy_name: str) -> bool:
    """Return True when file is an input mock FITS like <GALAXY>_<TEST>.fits."""
    pattern = rf"^{re.escape(galaxy_name)}_[^_]+\.fits$"
    return re.match(pattern, file_name) is not None


def collect_cleanup_targets(
    galaxy_dir: Path,
    galaxy_name: str,
    test_name: str | None,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Collect removable and preserved files for one galaxy directory."""
    removable_paths: list[Path] = []
    preserved_paths: list[Path] = []
    removable_directories: list[Path] = []

    mosaic_name = f"{galaxy_name}_mosaic.png"
    single_test_input_name = f"{galaxy_name}_{test_name}.fits" if test_name is not None else None
    single_test_prefix = f"{galaxy_name}_{test_name}_" if test_name is not None else None
    galaxy_prefix = f"{galaxy_name}_"

    for path in sorted(galaxy_dir.iterdir()):
        if path.is_dir():
            if test_name is not None:
                if path.name != test_name:
                    continue
                for subpath in sorted(path.rglob("*")):
                    if subpath.is_file():
                        removable_paths.append(subpath)
                removable_directories.append(path)
                continue

            if not re.match(r"^mock\d+$", path.name):
                continue
            for subpath in sorted(path.rglob("*")):
                if subpath.is_file():
                    removable_paths.append(subpath)
            removable_directories.append(path)
            continue

        file_name = path.name
        if file_name == mosaic_name:
            preserved_paths.append(path)
            continue

        if test_name is not None:
            if file_name == single_test_input_name:
                preserved_paths.append(path)
                continue
            if file_name.startswith(single_test_prefix):
                removable_paths.append(path)
            continue

        if is_preserved_mock_fits(file_name, galaxy_name):
            preserved_paths.append(path)
            continue
        if file_name.startswith(galaxy_prefix):
            removable_paths.append(path)

    return removable_paths, preserved_paths, removable_directories


def clean_paths(
    removable_paths: list[Path],
    dry_run: bool,
) -> tuple[int, list[tuple[Path, str]]]:
    """Delete file paths (or simulate deletion in dry-run mode)."""
    removed_count = 0
    failed_removals: list[tuple[Path, str]] = []

    for path in removable_paths:
        if dry_run:
            removed_count += 1
            continue
        try:
            path.unlink()
            removed_count += 1
        except OSError as error:
            failed_removals.append((path, str(error)))

    return removed_count, failed_removals


def remove_empty_directories(
    removable_directories: list[Path],
    dry_run: bool,
) -> tuple[int, list[tuple[Path, str]]]:
    """Remove empty test directories after file cleanup."""
    removed_count = 0
    failed_removals: list[tuple[Path, str]] = []

    for directory_path in sorted(removable_directories, reverse=True):
        if not directory_path.exists():
            continue
        if any(directory_path.iterdir()):
            continue
        if dry_run:
            removed_count += 1
            continue
        try:
            directory_path.rmdir()
            removed_count += 1
        except OSError as error:
            failed_removals.append((directory_path, str(error)))

    return removed_count, failed_removals


def main() -> None:
    """Execute cleanup workflow."""
    arguments = parse_arguments()
    validate_arguments(arguments)

    huang_root = arguments.huang_root
    if not huang_root.exists():
        raise FileNotFoundError(f"Huang root does not exist: {huang_root}")

    if arguments.all_galaxies:
        galaxy_names = discover_galaxies(huang_root)
    else:
        galaxy_names = [str(arguments.galaxy)]

    if not galaxy_names:
        print("No galaxy directories found.")
        return

    total_candidates = 0
    total_removed = 0
    total_removed_directories = 0
    total_preserved = 0
    total_missing_galaxy_dirs = 0
    all_failures: list[tuple[Path, str]] = []

    for galaxy_name in galaxy_names:
        galaxy_dir = huang_root / galaxy_name
        if not galaxy_dir.exists() or not galaxy_dir.is_dir():
            total_missing_galaxy_dirs += 1
            print(f"[CLEANUP] SKIP galaxy={galaxy_name} reason=missing_directory")
            continue

        removable_paths, preserved_paths, removable_directories = collect_cleanup_targets(
            galaxy_dir=galaxy_dir,
            galaxy_name=galaxy_name,
            test_name=arguments.test_name,
        )

        removed_count, file_failures = clean_paths(removable_paths, arguments.dry_run)
        removed_directories_count, directory_failures = remove_empty_directories(
            removable_directories,
            arguments.dry_run,
        )
        failures = file_failures + directory_failures
        all_failures.extend(failures)

        total_candidates += len(removable_paths)
        total_removed += removed_count
        total_removed_directories += removed_directories_count
        total_preserved += len(preserved_paths)

        mode_label = "single-test" if arguments.test_name is not None else "all-tests"
        action_label = "would_remove" if arguments.dry_run else "removed"
        print(
            f"[CLEANUP] galaxy={galaxy_name} mode={mode_label} "
            f"{action_label}={removed_count} removed_dirs={removed_directories_count} preserved={len(preserved_paths)}"
        )

        if arguments.verbose:
            for path in removable_paths:
                status = "DRY-RUN remove" if arguments.dry_run else "REMOVE"
                print(f"[CLEANUP] {status} {path}")
            for path in preserved_paths:
                print(f"[CLEANUP] KEEP {path}")
            for failed_path, message in failures:
                print(f"[CLEANUP] ERROR failed_to_remove={failed_path} reason={message}")

    print(f"Huang root: {huang_root}")
    print(f"Scope: {'all-galaxies' if arguments.all_galaxies else arguments.galaxy}")
    if arguments.test_name is not None:
        print(f"Test name: {arguments.test_name}")
    print(f"Dry run: {arguments.dry_run}")
    print(f"Candidates: {total_candidates}")
    print(f"Removed: {total_removed}")
    print(f"Removed directories: {total_removed_directories}")
    print(f"Preserved: {total_preserved}")
    if total_missing_galaxy_dirs > 0:
        print(f"Missing galaxy directories: {total_missing_galaxy_dirs}")
    if all_failures:
        print(f"Removal failures: {len(all_failures)}")
        raise RuntimeError("One or more files could not be removed.")


if __name__ == "__main__":
    main()
