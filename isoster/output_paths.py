"""Utilities for consistent generated-artifact output paths."""

from __future__ import annotations

import os
from pathlib import Path

OUTPUT_ROOT_ENV_VAR = "ISOSTER_OUTPUT_ROOT"


def get_output_root(output_root: str | Path | None = None) -> Path:
    """Return the output root directory.

    Priority:
    1. Explicit ``output_root`` argument
    2. ``ISOSTER_OUTPUT_ROOT`` environment variable
    3. ``outputs`` directory under current working directory
    """
    if output_root is not None:
        return Path(output_root)

    env_output_root = os.getenv(OUTPUT_ROOT_ENV_VAR)
    if env_output_root:
        return Path(env_output_root)

    return Path("outputs")


def resolve_output_directory(
    category_name: str,
    run_name: str | None = None,
    output_root: str | Path | None = None,
    explicit_output_directory: str | Path | None = None,
) -> Path:
    """Resolve and create a standardized output directory.

    Parameters
    ----------
    category_name
        High-level output category such as ``tests_integration``.
    run_name
        Optional subfolder for a specific script or scenario.
    output_root
        Optional output root path override.
    explicit_output_directory
        Optional explicit full output directory. When set, this takes priority.
    """
    if explicit_output_directory is not None:
        output_directory = Path(explicit_output_directory)
    else:
        output_directory = get_output_root(output_root) / category_name
        if run_name:
            output_directory = output_directory / run_name

    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory
