"""Shared benchmark/profiling metadata utilities."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import scipy


def get_git_sha(project_root: Optional[Path] = None) -> str:
    """Return the current git SHA for metadata recording."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            capture_output=True,
            check=True,
        )
    except Exception:
        return "unknown"

    return result.stdout.strip()


def _optional_module_version(module_name: str) -> Optional[str]:
    """Return module version if import succeeds, else None."""
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def collect_environment_metadata(
    project_root: Optional[Path] = None,
    extra_env_keys: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    """Collect standard machine/environment metadata for benchmark artifacts."""
    keys = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_DISABLE_JIT"]
    if extra_env_keys is not None:
        keys.extend(list(extra_env_keys))

    unique_keys = sorted(set(keys))
    selected_environment = {key: os.getenv(key, "") for key in unique_keys}

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_sha(project_root=project_root),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": _optional_module_version("numba"),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "environment_variables": selected_environment,
    }


def write_json(path: Path, payload: Dict[str, object]) -> None:
    """Write JSON payload with deterministic key ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_pointer:
        json.dump(payload, file_pointer, indent=2, sort_keys=True)
