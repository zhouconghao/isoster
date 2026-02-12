"""Copy models_config_batch template YAML files into a new output run directory."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.utils.run_metadata import collect_environment_metadata, write_json  # noqa: E402
from isoster.output_paths import resolve_output_directory  # noqa: E402

TEMPLATE_SOURCE_DIRECTORY = PROJECT_ROOT / "examples" / "mockgal" / "models_config_batch"
TEMPLATE_FILE_NAMES = ("galaxies.yaml", "image_config.yaml")


def copy_templates(destination_directory: Path, overwrite: bool, dry_run: bool) -> Dict[str, object]:
    """Copy template YAML files to destination and return a manifest payload."""
    if not TEMPLATE_SOURCE_DIRECTORY.exists():
        raise FileNotFoundError(f"Template source directory does not exist: {TEMPLATE_SOURCE_DIRECTORY}")

    copied_files: List[Dict[str, str]] = []
    skipped_files: List[Dict[str, str]] = []
    destination_directory.mkdir(parents=True, exist_ok=True)

    for file_name in TEMPLATE_FILE_NAMES:
        source_path = TEMPLATE_SOURCE_DIRECTORY / file_name
        if not source_path.exists():
            raise FileNotFoundError(f"Template file missing: {source_path}")

        destination_path = destination_directory / file_name
        if destination_path.exists() and not overwrite:
            skipped_files.append(
                {
                    "file_name": file_name,
                    "reason": "destination_exists",
                    "destination_path": str(destination_path),
                }
            )
            continue

        if not dry_run:
            shutil.copy2(source_path, destination_path)
        copied_files.append(
            {
                "file_name": file_name,
                "source_path": str(source_path),
                "destination_path": str(destination_path),
            }
        )

    return {
        "source_directory": str(TEMPLATE_SOURCE_DIRECTORY),
        "destination_directory": str(destination_directory),
        "copied_files": copied_files,
        "skipped_files": skipped_files,
        "overwrite": bool(overwrite),
        "dry_run": bool(dry_run),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scaffold models_config_batch template files into an outputs/ run directory."
    )
    parser.add_argument("--output", default=None, help="Explicit output directory override.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite destination files if they already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print and persist manifest without copying files.")
    args = parser.parse_args()

    output_directory = resolve_output_directory(
        "benchmarks_performance",
        "mockgal_models_config_batch_templates",
        explicit_output_directory=args.output,
    )

    manifest = {
        "environment": collect_environment_metadata(project_root=PROJECT_ROOT),
        "template_copy": copy_templates(
            destination_directory=output_directory,
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
        ),
    }
    manifest_path = output_directory / "template_scaffold_manifest.json"
    write_json(manifest_path, manifest)

    print(f"Template scaffold manifest written to: {manifest_path}")
    print(f"Destination directory: {output_directory}")
    print(f"Copied files: {len(manifest['template_copy']['copied_files'])}")
    print(f"Skipped files: {len(manifest['template_copy']['skipped_files'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
