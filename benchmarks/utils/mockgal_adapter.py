"""Optional adapter for generating high-fidelity mocks via external mockgal.py."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.utils.run_metadata import (  # noqa: E402
    collect_environment_metadata,
    get_git_sha,
    write_json,
)
from isoster.output_paths import resolve_output_directory  # noqa: E402


# Original fallback paths used before the env-var override was added.
# Set MOCKGAL_PATH / MOCKGAL_LIBPROFIT_BUILD to point at a different
# checkout without editing this file.
_MOCKGAL_FALLBACK_PATH = Path(
    "/Users/mac/Dropbox/work/project/otters/isophote_test/mockgal.py"
)
_LIBPROFIT_BUILD_FALLBACK_PATH = Path(
    "/Users/mac/Dropbox/work/project/otters/isophote_test/libprofit/build"
)
DEFAULT_MOCKGAL_PATH = Path(os.environ.get("MOCKGAL_PATH", str(_MOCKGAL_FALLBACK_PATH)))
DEFAULT_LIBPROFIT_BUILD_PATH = Path(
    os.environ.get(
        "MOCKGAL_LIBPROFIT_BUILD", str(_LIBPROFIT_BUILD_FALLBACK_PATH)
    )
)

PRESETS = {
    "single_noiseless_truth": {
        "description": (
            "Single-component noiseless truth image for analytic/algorithmic checks "
            "(no PSF, no sky, no noise)."
        ),
        "argument_template": [
            "--single",
            "--name",
            "{name}",
            "-z",
            "{redshift}",
            "--r-eff",
            "{r_eff_kpc}",
            "--abs-mag",
            "{abs_mag}",
            "--sersic-n",
            "{sersic_n}",
            "--ellip",
            "{ellip}",
            "--pa",
            "{pa_deg}",
            "--engine",
            "{engine}",
            "--output",
            "{output_dir}",
        ],
        "default_values": {
            "name": "mock_truth",
            "redshift": 0.01,
            "r_eff_kpc": 1.0,
            "abs_mag": -20.0,
            "sersic_n": 4.0,
            "ellip": 0.0,
            "pa_deg": 0.0,
            "engine": "libprofit",
        },
    },
    "single_psf_noise_sblimit": {
        "description": (
            "Single-component PSF-convolved and sky/noise-controlled image using "
            "surface-brightness controls for realistic mock diagnostics."
        ),
        "argument_template": [
            "--single",
            "--name",
            "{name}",
            "-z",
            "{redshift}",
            "--r-eff",
            "{r_eff_kpc}",
            "--abs-mag",
            "{abs_mag}",
            "--sersic-n",
            "{sersic_n}",
            "--ellip",
            "{ellip}",
            "--pa",
            "{pa_deg}",
            "--psf",
            "--psf-type",
            "moffat",
            "--psf-fwhm",
            "{psf_fwhm}",
            "--moffat-beta",
            "{moffat_beta}",
            "--sky-sb-value",
            "{sky_sb_value}",
            "--sky-sb-limit",
            "{sky_sb_limit}",
            "--gain",
            "{gain}",
            "--seed",
            "{seed}",
            "--engine",
            "{engine}",
            "--output",
            "{output_dir}",
        ],
        "default_values": {
            "name": "mock_psf_noise",
            "redshift": 0.01,
            "r_eff_kpc": 1.0,
            "abs_mag": -20.0,
            "sersic_n": 4.0,
            "ellip": 0.2,
            "pa_deg": 20.0,
            "psf_fwhm": 1.0,
            "moffat_beta": 4.765,
            "sky_sb_value": 22.0,
            "sky_sb_limit": 27.0,
            "gain": 4.0,
            "seed": 42,
            "engine": "libprofit",
        },
    },
    "models_config_batch": {
        "description": (
            "Batch generation from explicit model+config files with worker control."
        ),
        "argument_template": [
            "--models",
            "{models_file}",
            "--config",
            "{config_file}",
            "--workers",
            "{workers}",
            "--output",
            "{output_dir}",
            "--format",
            "{output_format}",
            "--engine",
            "{engine}",
        ],
        "default_values": {
            "models_file": "examples/mockgal/models_config_batch/galaxies.yaml",
            "config_file": "examples/mockgal/models_config_batch/image_config.yaml",
            "workers": 1,
            "output_format": "fits",
            "engine": "libprofit",
        },
    },
}


def parse_preset_value(raw_value: str):
    """Parse preset value and preserve simple string fallback."""
    try:
        return json.loads(raw_value)
    except Exception:
        return raw_value


def parse_key_value_pairs(items: List[str]) -> Dict[str, object]:
    """Parse KEY=VALUE pairs for preset overrides."""
    parsed = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --preset-value format: {item} (expected KEY=VALUE)")
        key, value = item.split("=", 1)
        parsed[key.strip()] = parse_preset_value(value.strip())
    return parsed


def resolve_preset_values(
    preset_name: str,
    output_directory: Path,
    preset_overrides: Dict[str, object],
) -> Dict[str, object]:
    """Resolve preset values with defaults and caller overrides."""
    preset = PRESETS[preset_name]
    resolved = dict(preset["default_values"])
    resolved.update(preset_overrides)
    resolved["output_dir"] = str(output_directory)
    return resolved


def render_preset_arguments(preset_name: str, resolved_values: Dict[str, object]) -> List[str]:
    """Render preset argument template to a flat argument list."""
    tokens = []
    for token in PRESETS[preset_name]["argument_template"]:
        if token.startswith("{") and token.endswith("}"):
            key = token[1:-1]
            if key not in resolved_values:
                raise KeyError(f"Missing preset value: {key}")
            value = resolved_values[key]
            if isinstance(value, (list, tuple)):
                tokens.extend([str(item) for item in value])
            else:
                tokens.append(str(value))
        else:
            tokens.append(token)
    return tokens


def parse_passthrough_arguments(passthrough_arguments: str) -> List[str]:
    """Parse raw passthrough argument string into shell-safe tokens."""
    if not passthrough_arguments:
        return []
    return shlex.split(passthrough_arguments)


def _extract_last_option_value(tokens: List[str], option_name: str) -> Optional[str]:
    """Extract last value for a --key value or --key=value option."""
    extracted_value: Optional[str] = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == option_name:
            if index + 1 >= len(tokens):
                raise ValueError(f"Missing value for {option_name}")
            extracted_value = tokens[index + 1]
            index += 2
            continue
        if token.startswith(f"{option_name}="):
            extracted_value = token.split("=", 1)[1]
        index += 1
    return extracted_value


def ensure_engine_is_libprofit(
    resolved_values: Dict[str, object],
    passthrough_tokens: List[str],
) -> str:
    """Enforce libprofit engine in presets and passthrough arguments."""
    preset_engine = resolved_values.get("engine")
    if preset_engine is not None and str(preset_engine).strip() != "libprofit":
        raise ValueError(
            f"Preset engine override is not allowed: expected 'libprofit', got '{preset_engine}'"
        )

    passthrough_engine = _extract_last_option_value(passthrough_tokens, "--engine")
    if passthrough_engine is not None and passthrough_engine.strip() != "libprofit":
        raise ValueError(
            f"Pass-through --engine must be 'libprofit', got '{passthrough_engine}'"
        )

    if "engine" in resolved_values:
        resolved_values["engine"] = "libprofit"
    return "libprofit"


def _resolve_profit_cli_path(path_value: Optional[str]) -> Optional[Path]:
    """Resolve potential profit-cli path from binary or directory input."""
    if not path_value:
        return None

    candidate = Path(path_value).expanduser()
    if candidate.is_dir():
        candidate = candidate / "profit-cli"

    if candidate.exists() and os.access(candidate, os.X_OK):
        return candidate.resolve()
    return None


def find_profit_cli_path(custom_path: Optional[str] = None) -> Optional[Path]:
    """Locate an executable profit-cli path using common resolution order."""
    ordered_candidates = [
        _resolve_profit_cli_path(custom_path),
        _resolve_profit_cli_path(os.getenv("LIBPROFIT_PATH")),
        _resolve_profit_cli_path(os.getenv("PROFIT_CLI_PATH")),
        _resolve_profit_cli_path(str(DEFAULT_LIBPROFIT_BUILD_PATH)),
    ]
    for candidate in ordered_candidates:
        if candidate is not None:
            return candidate

    from_path = shutil.which("profit-cli")
    if from_path:
        return Path(from_path).resolve()

    fallback_paths = [
        Path(__file__).resolve().parent / "libprofit" / "build" / "profit-cli",
        Path.home() / "libprofit" / "build" / "profit-cli",
        Path("/usr/local/bin/profit-cli"),
        Path("/opt/homebrew/bin/profit-cli"),
    ]
    for candidate in fallback_paths:
        resolved = _resolve_profit_cli_path(str(candidate))
        if resolved is not None:
            return resolved
    return None


def get_mockgal_git_sha(mockgal_script_path: Path) -> str:
    """Return git SHA for external mockgal.py checkout when available."""
    if not mockgal_script_path.exists():
        return "unknown"
    return get_git_sha(project_root=mockgal_script_path.parent)


def run_engine_preflight(passthrough_tokens: List[str]) -> Tuple[bool, Optional[Path], str]:
    """Validate libprofit runtime requirements before invoking mockgal."""
    profit_cli_override = _extract_last_option_value(passthrough_tokens, "--profit-cli")
    profit_cli_path = find_profit_cli_path(custom_path=profit_cli_override)
    if profit_cli_path is None:
        message = (
            "Preflight failed: required engine 'libprofit' is unavailable because "
            "profit-cli was not found. Set --profit-cli or LIBPROFIT_PATH/PROFIT_CLI_PATH."
        )
        return False, None, message
    return True, profit_cli_path, "libprofit preflight passed."


def build_command(
    mockgal_script_path: Path,
    preset_arguments: List[str],
    passthrough_tokens: List[str],
    enforced_engine: str,
) -> List[str]:
    """Build subprocess command for mockgal execution."""
    command = [sys.executable, str(mockgal_script_path)]
    command.extend(preset_arguments)
    if _extract_last_option_value(preset_arguments, "--engine") is None and _extract_last_option_value(
        passthrough_tokens, "--engine"
    ) is None:
        command.extend(["--engine", enforced_engine])
    command.extend(passthrough_tokens)
    return command


def run_mockgal_adapter(
    mockgal_script_path: Path,
    preset_name: str | None,
    preset_values: Dict[str, object],
    passthrough_arguments: str,
    dry_run: bool,
    output_directory: Path,
) -> dict:
    """Run adapter command and persist metadata."""
    output_directory.mkdir(parents=True, exist_ok=True)
    preset_arguments: List[str] = []
    resolved_values: Dict[str, object] = {}
    if preset_name:
        resolved_values = resolve_preset_values(
            preset_name=preset_name,
            output_directory=output_directory,
            preset_overrides=preset_values,
        )
        preset_arguments = render_preset_arguments(preset_name=preset_name, resolved_values=resolved_values)

    passthrough_tokens = parse_passthrough_arguments(passthrough_arguments)
    enforced_engine = ensure_engine_is_libprofit(
        resolved_values=resolved_values,
        passthrough_tokens=passthrough_tokens,
    )
    preflight_ok, profit_cli_path, preflight_message = run_engine_preflight(
        passthrough_tokens=passthrough_tokens
    )

    metadata = {
        "environment": collect_environment_metadata(project_root=PROJECT_ROOT),
        "mockgal_script_path": str(mockgal_script_path),
        "mockgal_git_sha": get_mockgal_git_sha(mockgal_script_path),
        "preset_name": preset_name,
        "engine": enforced_engine,
        "profit_cli_path": str(profit_cli_path) if profit_cli_path else None,
        "preflight_passed": bool(preflight_ok),
        "preflight_message": preflight_message,
        "resolved_preset_values": resolved_values,
        "preset_arguments": preset_arguments,
        "passthrough_arguments": passthrough_arguments,
        "passthrough_tokens": passthrough_tokens,
        "dry_run": bool(dry_run),
    }

    if not mockgal_script_path.exists():
        metadata["status"] = "skipped_missing_script"
        metadata["message"] = f"mockgal.py not found at {mockgal_script_path}"
        write_json(output_directory / "mockgal_adapter_run.json", metadata)
        return metadata

    if not preflight_ok:
        metadata["status"] = "failed_preflight"
        metadata["message"] = preflight_message
        write_json(output_directory / "mockgal_adapter_run.json", metadata)
        return metadata

    command = build_command(
        mockgal_script_path=mockgal_script_path,
        preset_arguments=preset_arguments,
        passthrough_tokens=passthrough_tokens,
        enforced_engine=enforced_engine,
    )
    metadata["command"] = command

    if dry_run:
        metadata["status"] = "dry_run"
        metadata["message"] = "Command prepared but not executed."
        write_json(output_directory / "mockgal_adapter_run.json", metadata)
        return metadata

    completed_process = subprocess.run(command, text=True, capture_output=True)
    metadata["status"] = "success" if completed_process.returncode == 0 else "failed"
    metadata["return_code"] = int(completed_process.returncode)
    metadata["stdout"] = completed_process.stdout[-8000:]
    metadata["stderr"] = completed_process.stderr[-8000:]
    write_json(output_directory / "mockgal_adapter_run.json", metadata)
    return metadata


def print_presets() -> None:
    """Print available preset names and descriptions."""
    print("Available presets:")
    for name in sorted(PRESETS):
        print(f"- {name}: {PRESETS[name]['description']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Optional mockgal.py adapter for high-fidelity mock generation.")
    parser.add_argument("--mockgal-script", default=str(DEFAULT_MOCKGAL_PATH), help="Path to external mockgal.py script.")
    parser.add_argument("--mockgal-args", default="", help="Additional pass-through argument string for mockgal.py.")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default=None, help="Named science-ready argument preset.")
    parser.add_argument(
        "--preset-value",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override preset value(s). Value can be JSON for lists.",
    )
    parser.add_argument(
        "--preset-values-file",
        default=None,
        help="JSON file containing preset value overrides.",
    )
    parser.add_argument("--list-presets", action="store_true", help="List available presets and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Do not execute mockgal.py; only persist run metadata.")
    parser.add_argument("--output", default=None, help="Explicit output directory.")
    args = parser.parse_args()

    if args.list_presets:
        print_presets()
        return 0

    preset_values = parse_key_value_pairs(args.preset_value)
    if args.preset_values_file:
        with open(args.preset_values_file, "r", encoding="utf-8") as file_pointer:
            file_values = json.load(file_pointer)
        preset_values.update(file_values)

    output_directory = resolve_output_directory(
        "benchmarks_performance",
        "mockgal_adapter",
        explicit_output_directory=args.output,
    )

    try:
        metadata = run_mockgal_adapter(
            mockgal_script_path=Path(args.mockgal_script),
            preset_name=args.preset,
            preset_values=preset_values,
            passthrough_arguments=args.mockgal_args,
            dry_run=args.dry_run,
            output_directory=output_directory,
        )
    except ValueError as error:
        metadata_path = output_directory / "mockgal_adapter_run.json"
        write_json(
            metadata_path,
            {
                "status": "failed_validation",
                "message": str(error),
                "preset_name": args.preset,
                "passthrough_arguments": args.mockgal_args,
            },
        )
        print("mockgal adapter status: failed_validation")
        print(f"Metadata written to: {metadata_path}")
        print(str(error))
        return 1

    print(f"mockgal adapter status: {metadata['status']}")
    print(f"Metadata written to: {output_directory / 'mockgal_adapter_run.json'}")
    if "message" in metadata:
        print(metadata["message"])
    if metadata["status"] in {"failed_preflight", "failed"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
