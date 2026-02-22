#!/usr/bin/env python3
"""Run reproducible Huang2013 mock comparisons for photutils and isoster.

This script is designed for external Huang2013 mock images stored under:
    /Users/mac/work/hsc/huang2013/<GALAXY>/<GALAXY>_mock<id>.fits

It can run photutils and isoster independently, save method-specific profile
products, create QA figures, and optionally build a cross-method comparison
figure from saved artifacts.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import platform
import pstats
import shutil
import time
import warnings
from pathlib import Path
from typing import Any

import astropy
import isoster
import matplotlib.pyplot as plt
import numpy as np
import photutils
from astropy import units as units
from astropy.cosmology import Planck18
from astropy.io import fits
from astropy.table import Table
from huang2013_campaign_contract import build_case_output_dir, build_case_prefix
from isoster.cog import compute_cog
from isoster.config import IsosterConfig
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse as mpl_ellipse
from photutils.aperture import EllipticalAperture, aperture_photometry
from photutils.isophote import Ellipse, EllipseGeometry, build_ellipse_model


DEFAULT_HUANG_ROOT = Path("/Users/mac/work/hsc/huang2013")
DEFAULT_PIXEL_SCALE_ARCSEC = 0.176
DEFAULT_ZEROPOINT = 27.0
DEFAULT_REDSHIFT = 0.05
DEFAULT_CONFIG_TAG = "baseline"
HUANG2013_PA_OFFSET_DEG = -90.0
DEFAULT_INITIAL_SMA_PIX = 6.0

STOP_CODE_STYLES = {
    0: {"color": "#1f77b4", "marker": "o", "label": "stop=0"},
    1: {"color": "#ff7f0e", "marker": "s", "label": "stop=1"},
    2: {"color": "#2ca02c", "marker": "^", "label": "stop=2"},
    3: {"color": "#d62728", "marker": "D", "label": "stop=3"},
    4: {"color": "#8c564b", "marker": "P", "label": "stop=4"},
    5: {"color": "#17becf", "marker": "v", "label": "stop=5"},
    -1: {"color": "#9467bd", "marker": "X", "label": "stop=-1"},
}

MONOCHROME_STOP_MARKERS = {
    0: "o",
    1: "s",
    2: "^",
    3: "D",
    4: "P",
    5: "v",
    -1: "X",
}

MONOCHROME_STOP_COLORS = {
    0: "#111111",
    1: "#1f3b73",
    2: "#1b5e20",
    3: "#7f1d1d",
    4: "#8b5a2b",
    5: "#006d77",
    -1: "#4b2e83",
}


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run IC2597-style Huang2013 real-mock comparison workflow.",
    )
    parser.add_argument(
        "--galaxy", required=True, help="Galaxy folder/name, e.g., IC2597"
    )
    parser.add_argument(
        "--mock-id", type=int, default=1, help="Mock ID suffix in FITS name"
    )
    parser.add_argument(
        "--huang-root",
        type=Path,
        default=DEFAULT_HUANG_ROOT,
        help="Root directory containing galaxy subfolders.",
    )
    parser.add_argument(
        "--input-fits",
        type=Path,
        default=None,
        help="Explicit FITS file. Overrides --huang-root/--galaxy/--mock-id.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <huang-root>/<GALAXY>/<TEST>/.",
    )
    parser.add_argument(
        "--method",
        choices=["photutils", "isoster", "both"],
        default="both",
        help="Method to run in this invocation.",
    )
    parser.add_argument(
        "--config-tag",
        default=DEFAULT_CONFIG_TAG,
        help="Tag used in artifact names to distinguish configurations.",
    )

    parser.add_argument(
        "--redshift", type=float, default=None, help="Override redshift"
    )
    parser.add_argument("--pixel-scale", type=float, default=None, help="Arcsec/pixel")
    parser.add_argument(
        "--zeropoint", type=float, default=None, help="Magnitude zeropoint"
    )
    parser.add_argument(
        "--psf-fwhm", type=float, default=None, help="PSF FWHM in arcsec"
    )

    parser.add_argument(
        "--sma0", type=float, default=None, help="Initial SMA in pixels"
    )
    parser.add_argument(
        "--minsma", type=float, default=1.0, help="Minimum fitted SMA in pixels"
    )
    parser.add_argument(
        "--maxsma", type=float, default=None, help="Maximum fitted SMA in pixels"
    )
    parser.add_argument("--astep", type=float, default=0.1, help="SMA growth step")
    parser.add_argument(
        "--maxgerr", type=float, default=None, help="Gradient error threshold"
    )

    parser.add_argument(
        "--photutils-nclip", type=int, default=2, help="photutils nclip"
    )
    parser.add_argument(
        "--photutils-sclip", type=float, default=3.0, help="photutils sclip"
    )
    parser.add_argument(
        "--photutils-integrmode",
        default="bilinear",
        choices=["bilinear", "nearest_neighbor", "mean"],
        help="photutils integration mode",
    )

    parser.add_argument(
        "--use-eccentric-anomaly",
        action="store_true",
        help="Enable isoster eccentric-anomaly sampling.",
    )
    parser.add_argument(
        "--isoster-config-json",
        type=Path,
        default=None,
        help="JSON file with isoster config overrides.",
    )

    parser.add_argument(
        "--cog-subpixels",
        type=int,
        default=32,
        help="Subpixel factor for true CoG aperture photometry.",
    )
    parser.add_argument(
        "--isophote-overlay-step",
        type=int,
        default=10,
        help="Overlay every Nth isophote in QA panels.",
    )
    parser.add_argument("--qa-dpi", type=int, default=180, help="DPI for QA figures")
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip the cross-method comparison figure.",
    )

    parser.add_argument(
        "--photutils-profile-fits",
        type=Path,
        default=None,
        help="Existing photutils profile FITS for comparison-only use.",
    )
    parser.add_argument(
        "--isoster-profile-fits",
        type=Path,
        default=None,
        help="Existing isoster profile FITS for comparison-only use.",
    )

    return parser.parse_args()


def sanitize_label(label: str) -> str:
    """Return filesystem-safe configuration labels."""
    label = label.strip().lower()
    safe_chars = []
    for character in label:
        if character.isalnum() or character in {"-", "_"}:
            safe_chars.append(character)
        else:
            safe_chars.append("-")
    sanitized = "".join(safe_chars).strip("-")
    return sanitized or DEFAULT_CONFIG_TAG


def build_input_path(args: argparse.Namespace) -> Path:
    """Resolve the input FITS path from arguments."""
    if args.input_fits is not None:
        return args.input_fits
    return args.huang_root / args.galaxy / f"{args.galaxy}_mock{args.mock_id}.fits"


def read_mock_image(mock_path: Path) -> tuple[np.ndarray, fits.Header]:
    """Read FITS image and header as float64."""
    with fits.open(mock_path) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
        header = hdul[0].header.copy()
    if data.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {data.shape}")
    return data, header


def get_header_value(header: fits.Header, key: str, default: Any) -> Any:
    """Read a header value with default fallback."""
    if key in header:
        return header[key]
    return default


def compute_kpc_per_arcsec(redshift: float) -> float:
    """Convert angular scale to kpc/arcsec using Planck18 cosmology."""
    scale = Planck18.kpc_proper_per_arcmin(redshift)
    return scale.to_value(units.kpc / units.arcsec)


def normalize_pa_degrees(pa_degrees: np.ndarray) -> np.ndarray:
    """Normalize PA values while preserving continuity across 180-degree periodicity."""
    pa = np.asarray(pa_degrees, dtype=float)
    output = np.full(pa.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(pa)
    if not np.any(finite_mask):
        return output

    wrapped = np.mod(pa[finite_mask], 180.0)
    doubled_rad = np.deg2rad(2.0 * wrapped)
    unwrapped = np.unwrap(doubled_rad)
    normalized = np.rad2deg(0.5 * unwrapped)
    output[finite_mask] = np.mod(normalized, 180.0)
    return output


def style_for_stop_code(stop_code: int, monochrome: bool = False) -> dict[str, str]:
    """Return plotting style for stop code."""
    if monochrome:
        marker = MONOCHROME_STOP_MARKERS.get(stop_code, "o")
        color = MONOCHROME_STOP_COLORS.get(stop_code, "#374151")
        return {"color": color, "marker": marker, "label": f"stop={stop_code}"}
    if stop_code in STOP_CODE_STYLES:
        return STOP_CODE_STYLES[stop_code]
    return {"color": "#7f7f7f", "marker": "o", "label": f"stop={stop_code}"}


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 checksum for reproducibility metadata."""
    digest = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_initial_geometry(
    header: fits.Header, image_shape: tuple[int, int]
) -> dict[str, float]:
    """Infer center and initial geometric parameters from FITS header."""
    center_x = (image_shape[1] - 1) / 2.0
    center_y = (image_shape[0] - 1) / 2.0

    component_count = int(get_header_value(header, "NCOMP", 0))
    eps_values = []
    pa_values = []
    weight_values = []

    for index in range(1, component_count + 1):
        eps_key = f"ELLIP{index}"
        pa_key = f"PA{index}"
        magnitude_key = f"APPMAG{index}"

        if eps_key not in header or pa_key not in header:
            continue

        eps_value = float(header[eps_key])
        pa_value = float(header[pa_key])

        if magnitude_key in header:
            flux_weight = 10.0 ** (-0.4 * float(header[magnitude_key]))
        else:
            flux_weight = 1.0

        eps_values.append(eps_value)
        pa_values.append(pa_value)
        weight_values.append(flux_weight)

    if eps_values:
        weights = np.asarray(weight_values, dtype=float)
        weights /= np.sum(weights)
        eps_initial = float(np.sum(np.asarray(eps_values) * weights))

        pa_rad = np.deg2rad(np.asarray(pa_values, dtype=float))
        pa_initial = np.rad2deg(
            np.arctan2(
                np.sum(weights * np.sin(2.0 * pa_rad)),
                np.sum(weights * np.cos(2.0 * pa_rad)),
            )
            / 2.0
        )
    else:
        eps_initial = float(get_header_value(header, "ELLIP1", 0.2))
        pa_initial = float(get_header_value(header, "PA1", 0.0))

    # Huang2013 mocks use a PA convention offset from the image-space convention
    # expected by photutils/isoster initialization in this workflow.
    pa_initial = float(pa_initial + HUANG2013_PA_OFFSET_DEG)

    sma_initial = float(DEFAULT_INITIAL_SMA_PIX)
    sma_initial = max(sma_initial, 3.0)

    return {
        "x0": center_x,
        "y0": center_y,
        "eps": float(np.clip(eps_initial, 0.0, 0.95)),
        "pa_deg": float(pa_initial),
        "sma0": sma_initial,
    }


def run_with_runtime_profile(
    function_to_run, *args, **kwargs
) -> tuple[Any, dict[str, Any], str]:
    """Run callable with wall/cpu timing and cProfile summary."""
    profiler = cProfile.Profile()
    start_wall = time.perf_counter()
    start_cpu = time.process_time()

    profiler.enable()
    try:
        output = function_to_run(*args, **kwargs)
    finally:
        profiler.disable()

    elapsed_wall = time.perf_counter() - start_wall
    elapsed_cpu = time.process_time() - start_cpu

    summary_stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=summary_stream).sort_stats("cumulative")
    stats.print_stats(20)

    runtime_metadata = {
        "wall_time_seconds": elapsed_wall,
        "cpu_time_seconds": elapsed_cpu,
    }
    return output, runtime_metadata, summary_stream.getvalue()


def convert_photutils_isolist(isolist) -> list[dict[str, Any]]:
    """Convert photutils isolist objects into serializable dictionaries."""
    attribute_map = {
        "sma": "sma",
        "intens": "intens",
        "intens_err": "int_err",
        "eps": "eps",
        "ellip_err": "ellip_err",
        "pa": "pa",
        "pa_err": "pa_err",
        "x0": "x0",
        "x0_err": "x0_err",
        "y0": "y0",
        "y0_err": "y0_err",
        "grad": "grad",
        "grad_error": "grad_error",
        "rms": "rms",
        "pix_stddev": "pix_stddev",
        "stop_code": "stop_code",
        "ndata": "ndata",
        "nflag": "nflag",
        "niter": "niter",
        "a1": "a1",
        "b1": "b1",
        "a2": "a2",
        "b2": "b2",
        "a3": "a3",
        "b3": "b3",
        "a4": "a4",
        "b4": "b4",
        "a1_err": "a1_err",
        "b1_err": "b1_err",
        "a2_err": "a2_err",
        "b2_err": "b2_err",
        "a3_err": "a3_err",
        "b3_err": "b3_err",
        "a4_err": "a4_err",
        "b4_err": "b4_err",
        "tflux_e": "tflux_e",
        "tflux_c": "tflux_c",
        "npix_e": "npix_e",
        "npix_c": "npix_c",
    }

    isophotes: list[dict[str, Any]] = []
    for iso in isolist:
        row: dict[str, Any] = {}
        for output_key, source_key in attribute_map.items():
            value = getattr(iso, source_key, np.nan)
            if value is None:
                row[output_key] = np.nan
            else:
                row[output_key] = value
        isophotes.append(row)

    return isophotes


def run_photutils_fit(
    image: np.ndarray, fit_config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Run photutils.isophote fitting."""
    geometry = EllipseGeometry(
        x0=fit_config["x0"],
        y0=fit_config["y0"],
        sma=fit_config["sma0"],
        eps=fit_config["eps"],
        pa=np.deg2rad(fit_config["pa_deg"]),
    )
    ellipse = Ellipse(image, geometry)

    fit_kwargs = {
        "step": fit_config["astep"],
        "minsma": fit_config["minsma"],
        "maxgerr": fit_config["maxgerr"],
        "nclip": fit_config["nclip"],
        "sclip": fit_config["sclip"],
        "integrmode": fit_config["integrmode"],
    }
    if fit_config["maxsma"] is not None:
        fit_kwargs["maxsma"] = fit_config["maxsma"]

    isolist = ellipse.fit_image(**fit_kwargs)
    return convert_photutils_isolist(isolist)


def run_isoster_fit(
    image: np.ndarray, config_dict: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run isoster fitting and return isophotes plus validated config dictionary."""
    config = IsosterConfig(**config_dict)
    results = isoster.fit_image(image, None, config)
    return results["isophotes"], config.model_dump()


def ensure_numeric_column(
    table: Table, column_name: str, default_value: float = np.nan
) -> None:
    """Ensure table has the requested numeric column."""
    if column_name not in table.colnames:
        table[column_name] = np.full(len(table), default_value, dtype=float)


def is_error_column_name(column_name: str) -> bool:
    """Return True when a profile-table column represents an uncertainty/error."""
    normalized_name = column_name.lower()
    return (
        normalized_name.endswith("_err")
        or normalized_name.endswith("_error")
        or "_err_" in normalized_name
    )


def summarize_negative_error_values(profile_table: Table) -> list[dict[str, Any]]:
    """Summarize columns containing negative error values."""
    negative_entries: list[dict[str, Any]] = []
    for column_name in profile_table.colnames:
        if not is_error_column_name(column_name):
            continue

        try:
            column_values = np.asarray(profile_table[column_name], dtype=float)
        except (TypeError, ValueError):
            continue

        negative_mask = column_values < 0.0
        if not np.any(negative_mask):
            continue

        negative_indices = np.flatnonzero(negative_mask)
        minimum_value = float(np.nanmin(column_values[negative_mask]))
        negative_entries.append(
            {
                "column": column_name,
                "count": int(negative_mask.sum()),
                "minimum": minimum_value,
                "first_index": int(negative_indices[0]),
            }
        )

    return negative_entries


def compute_true_curve_of_growth(
    image: np.ndarray,
    profile_table: Table,
    subpixels: int,
) -> np.ndarray:
    """Compute true curve-of-growth using fitted geometry on noiseless image."""
    curve_of_growth = np.full(len(profile_table), np.nan, dtype=float)

    for index in range(len(profile_table)):
        sma = float(profile_table["sma"][index])
        eps = float(profile_table["eps"][index])
        x0 = float(profile_table["x0"][index])
        y0 = float(profile_table["y0"][index])
        pa = float(profile_table["pa"][index])

        if not np.isfinite(sma) or not np.isfinite(eps) or sma <= 0.0:
            continue

        axis_ratio = 1.0 - eps
        if axis_ratio <= 0.0:
            continue

        aperture = EllipticalAperture((x0, y0), a=sma, b=sma * axis_ratio, theta=pa)
        photometry = aperture_photometry(
            image,
            aperture,
            method="subpixel",
            subpixels=subpixels,
        )
        curve_of_growth[index] = float(photometry["aperture_sum"][0])

    return curve_of_growth


def ensure_isoster_style_cog_columns(profile_table: Table) -> None:
    """Populate CoG columns using the shared `isoster.cog.compute_cog` recipe."""
    if len(profile_table) == 0:
        return

    required_columns = {"sma", "eps", "intens", "x0", "y0"}
    if not required_columns.issubset(profile_table.colnames):
        return

    if "cog" in profile_table.colnames:
        existing_cog = np.asarray(profile_table["cog"], dtype=float)
        if np.any(np.isfinite(existing_cog)):
            return

    isophote_rows: list[dict[str, float]] = []
    try:
        for index in range(len(profile_table)):
            isophote_rows.append(
                {
                    "sma": float(profile_table["sma"][index]),
                    "eps": float(profile_table["eps"][index]),
                    "intens": float(profile_table["intens"][index]),
                    "x0": float(profile_table["x0"][index]),
                    "y0": float(profile_table["y0"][index]),
                }
            )
    except (TypeError, ValueError):
        return

    cog_results = compute_cog(isophote_rows)
    profile_table["cog"] = np.asarray(cog_results["cog"], dtype=float)
    profile_table["cog_annulus"] = np.asarray(cog_results["cog_annulus"], dtype=float)
    profile_table["area_annulus"] = np.asarray(cog_results["area_annulus"], dtype=float)
    profile_table["flag_cross"] = np.asarray(cog_results["flag_cross"], dtype=bool)
    profile_table["flag_negative_area"] = np.asarray(
        cog_results["flag_negative_area"],
        dtype=bool,
    )


def harmonize_method_cog_columns(
    profile_table: Table, method_name: str | None = None
) -> None:
    """Populate method CoG column using method-aware source priority."""
    if len(profile_table) == 0:
        return

    ensure_isoster_style_cog_columns(profile_table)

    resolved_method_name = None
    if isinstance(method_name, str) and method_name:
        resolved_method_name = method_name.strip().lower()
    elif isinstance(profile_table.meta.get("METHOD"), str):
        resolved_method_name = profile_table.meta["METHOD"].strip().lower()

    if resolved_method_name in {"isoster", "photutils"}:
        source_priority = ["cog", "method_cog_flux", "tflux_e", "true_cog_flux"]
    else:
        source_priority = ["method_cog_flux", "cog", "tflux_e", "true_cog_flux"]

    merged_method_cog = np.full(len(profile_table), np.nan, dtype=float)
    for column_name in source_priority:
        if column_name not in profile_table.colnames:
            continue
        source_values = np.asarray(profile_table[column_name], dtype=float)
        fill_mask = ~np.isfinite(merged_method_cog) & np.isfinite(source_values)
        merged_method_cog[fill_mask] = source_values[fill_mask]

    profile_table["method_cog_flux"] = merged_method_cog

    if "true_cog_flux" in profile_table.colnames:
        true_cog_flux = np.asarray(profile_table["true_cog_flux"], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            profile_table["cog_rel_diff"] = (
                merged_method_cog - true_cog_flux
            ) / true_cog_flux


def prepare_profile_table(
    isophotes: list[dict[str, Any]],
    image: np.ndarray,
    redshift: float,
    pixel_scale_arcsec: float,
    zeropoint_mag: float,
    cog_subpixels: int,
    method_name: str,
) -> Table:
    """Create profile table with derived columns and true CoG appendices."""
    table = Table(rows=isophotes)
    if len(table) == 0:
        return table

    ensure_numeric_column(table, "intens_err")
    ensure_numeric_column(table, "stop_code", default_value=0.0)
    ensure_numeric_column(table, "tflux_e")

    center_x = (image.shape[1] - 1) / 2.0
    center_y = (image.shape[0] - 1) / 2.0

    sma = np.asarray(table["sma"], dtype=float)
    intensity = np.asarray(table["intens"], dtype=float)
    intensity_error = np.asarray(table["intens_err"], dtype=float)
    eps = np.asarray(table["eps"], dtype=float)
    pa_radians = np.asarray(table["pa"], dtype=float)

    kpc_per_arcsec = compute_kpc_per_arcsec(redshift)
    sma_arcsec = sma * pixel_scale_arcsec
    sma_kpc = sma_arcsec * kpc_per_arcsec

    with np.errstate(invalid="ignore"):
        x_axis_kpc_quarter = np.power(np.clip(sma_kpc, 0.0, np.inf), 0.25)

    pa_deg_raw = np.rad2deg(pa_radians)
    pa_deg_normalized = normalize_pa_degrees(pa_deg_raw)

    axis_ratio = 1.0 - eps
    x0_offset = np.asarray(table["x0"], dtype=float) - center_x
    y0_offset = np.asarray(table["y0"], dtype=float) - center_y
    centroid_offset = np.hypot(x0_offset, y0_offset)

    surface_brightness = np.full(len(table), np.nan, dtype=float)
    surface_brightness_error = np.full(len(table), np.nan, dtype=float)
    valid_intensity = np.isfinite(intensity) & (intensity > 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        surface_brightness[valid_intensity] = zeropoint_mag - 2.5 * np.log10(
            intensity[valid_intensity] / (pixel_scale_arcsec**2)
        )
        surface_brightness_error[valid_intensity] = (2.5 / np.log(10.0)) * (
            intensity_error[valid_intensity] / intensity[valid_intensity]
        )

    table["axis_ratio"] = axis_ratio
    table["sma_arcsec"] = sma_arcsec
    table["sma_kpc"] = sma_kpc
    table["x_kpc_quarter"] = x_axis_kpc_quarter
    table["pa_deg_raw"] = pa_deg_raw
    table["pa_deg_norm"] = pa_deg_normalized
    table["x0_offset_pix"] = x0_offset
    table["y0_offset_pix"] = y0_offset
    table["centroid_offset_pix"] = centroid_offset
    table["sb_mag_arcsec2"] = surface_brightness
    table["sb_err_mag"] = surface_brightness_error

    true_curve_of_growth = compute_true_curve_of_growth(
        image, table, subpixels=cog_subpixels
    )
    table["true_cog_flux"] = true_curve_of_growth

    harmonize_method_cog_columns(table, method_name=method_name)

    table.meta["METHOD"] = method_name
    table.meta["REDSHIFT"] = redshift
    table.meta["PIXSCALE"] = pixel_scale_arcsec
    table.meta["MAGZERO"] = zeropoint_mag
    table.meta["KPCASC"] = kpc_per_arcsec
    table.meta["COGSUBPX"] = cog_subpixels

    return table


def table_to_isophote_dicts(profile_table: Table) -> list[dict[str, Any]]:
    """Convert profile table rows back to dictionaries for model reconstruction."""
    isophotes: list[dict[str, Any]] = []
    for index in range(len(profile_table)):
        row_dict: dict[str, Any] = {}
        for column_name in profile_table.colnames:
            value = profile_table[column_name][index]
            if isinstance(value, np.generic):
                row_dict[column_name] = value.item()
            else:
                row_dict[column_name] = value
        isophotes.append(row_dict)
    return isophotes


class _PhotutilsModelSmaNode:
    """Minimal node object exposing `sma` for photutils model builder."""

    def __init__(self, sma_value: float) -> None:
        self.sma = float(sma_value)


class _PhotutilsModelIsolistAdapter:
    """Duck-typed adapter matching the `build_ellipse_model` access pattern."""

    def __init__(self, model_columns: dict[str, np.ndarray]) -> None:
        self.sma = model_columns["sma"]
        self.intens = model_columns["intens"]
        self.eps = model_columns["eps"]
        self.pa = model_columns["pa"]
        self.x0 = model_columns["x0"]
        self.y0 = model_columns["y0"]
        self.grad = model_columns["grad"]
        self.a3 = model_columns["a3"]
        self.b3 = model_columns["b3"]
        self.a4 = model_columns["a4"]
        self.b4 = model_columns["b4"]
        self._nodes = [_PhotutilsModelSmaNode(value) for value in self.sma]

    def __len__(self) -> int:
        return len(self._nodes)

    def __getitem__(self, index: int) -> _PhotutilsModelSmaNode:
        return self._nodes[index]


def extract_photutils_model_columns(profile_table: Table) -> dict[str, np.ndarray]:
    """Extract and sanitize profile columns required by `build_ellipse_model`."""
    required_columns = ["sma", "intens", "eps", "pa", "x0", "y0", "grad"]
    missing_columns = [
        name for name in required_columns if name not in profile_table.colnames
    ]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing photutils model columns: {missing_text}")

    model_columns: dict[str, np.ndarray] = {}
    for column_name in required_columns:
        model_columns[column_name] = np.asarray(profile_table[column_name], dtype=float)

    for harmonic_name in ["a3", "b3", "a4", "b4"]:
        if harmonic_name in profile_table.colnames:
            model_columns[harmonic_name] = np.asarray(
                profile_table[harmonic_name], dtype=float
            )
        else:
            model_columns[harmonic_name] = np.zeros(len(profile_table), dtype=float)

    valid_mask = model_columns["sma"] > 0.0
    for column_name in model_columns:
        valid_mask &= np.isfinite(model_columns[column_name])

    if int(np.sum(valid_mask)) < 6:
        raise ValueError(
            "Insufficient valid photutils rows for 2-D model reconstruction"
        )

    sort_indices = np.argsort(model_columns["sma"][valid_mask])
    for column_name in model_columns:
        model_columns[column_name] = model_columns[column_name][valid_mask][
            sort_indices
        ]

    unique_sma_values, unique_indices = np.unique(
        model_columns["sma"], return_index=True
    )
    for column_name in model_columns:
        model_columns[column_name] = model_columns[column_name][unique_indices]
    model_columns["sma"] = unique_sma_values

    if model_columns["sma"].size < 6:
        raise ValueError(
            "Insufficient unique photutils SMA rows for 2-D model reconstruction"
        )

    return model_columns


def build_photutils_model_image(
    image_shape: tuple[int, int],
    profile_table: Table,
) -> np.ndarray:
    """Build 2-D model using photutils' native `build_ellipse_model`."""
    model_columns = extract_photutils_model_columns(profile_table)
    isolist_adapter = _PhotutilsModelIsolistAdapter(model_columns)
    return build_ellipse_model(
        image_shape,
        isolist_adapter,
        fill=0.0,
        high_harmonics=True,
        sma_interval=0.1,
    )


def extract_isoster_model_rows(
    profile_table: Table,
    harmonic_orders: list[int] | None = None,
) -> tuple[list[dict[str, float]], dict[str, int]]:
    """Extract and sanitize isoster rows required by `build_isoster_model`."""
    if harmonic_orders is None:
        harmonic_orders = [3, 4]

    required_columns = ["sma", "intens", "eps", "pa", "x0", "y0"]
    missing_columns = [
        name for name in required_columns if name not in profile_table.colnames
    ]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing isoster model columns: {missing_text}")

    model_columns: dict[str, np.ndarray] = {}
    for column_name in required_columns:
        model_columns[column_name] = np.asarray(profile_table[column_name], dtype=float)

    valid_mask = model_columns["sma"] > 0.0
    for column_name in required_columns:
        valid_mask &= np.isfinite(model_columns[column_name])

    input_row_count = len(profile_table)
    valid_row_count = int(np.sum(valid_mask))
    if valid_row_count < 6:
        raise ValueError("Insufficient valid isoster rows for 2-D model reconstruction")

    sort_indices = np.argsort(model_columns["sma"][valid_mask])
    for column_name in required_columns:
        model_columns[column_name] = model_columns[column_name][valid_mask][
            sort_indices
        ]

    unique_sma_values, unique_indices = np.unique(
        model_columns["sma"], return_index=True
    )
    for column_name in required_columns:
        model_columns[column_name] = model_columns[column_name][unique_indices]
    model_columns["sma"] = unique_sma_values

    unique_row_count = int(model_columns["sma"].size)
    if unique_row_count < 6:
        raise ValueError(
            "Insufficient unique isoster SMA rows for 2-D model reconstruction"
        )

    harmonic_columns: dict[str, np.ndarray] = {}
    for harmonic_order in harmonic_orders:
        for harmonic_prefix in ["a", "b"]:
            harmonic_name = f"{harmonic_prefix}{harmonic_order}"
            if harmonic_name in profile_table.colnames:
                harmonic_values = np.asarray(profile_table[harmonic_name], dtype=float)
                harmonic_values = harmonic_values[valid_mask][sort_indices][
                    unique_indices
                ]
                harmonic_columns[harmonic_name] = np.where(
                    np.isfinite(harmonic_values),
                    harmonic_values,
                    0.0,
                )
            else:
                harmonic_columns[harmonic_name] = np.zeros(
                    unique_row_count, dtype=float
                )

    isophote_rows: list[dict[str, float]] = []
    for index in range(unique_row_count):
        row = {
            "sma": float(model_columns["sma"][index]),
            "intens": float(model_columns["intens"][index]),
            "eps": float(model_columns["eps"][index]),
            "pa": float(model_columns["pa"][index]),
            "x0": float(model_columns["x0"][index]),
            "y0": float(model_columns["y0"][index]),
        }
        for harmonic_name, harmonic_values in harmonic_columns.items():
            row[harmonic_name] = float(harmonic_values[index])
        isophote_rows.append(row)

    summary = {
        "input_row_count": int(input_row_count),
        "valid_row_count": int(valid_row_count),
        "invalid_row_count": int(input_row_count - valid_row_count),
        "unique_row_count": int(unique_row_count),
        "duplicate_row_count": int(valid_row_count - unique_row_count),
    }
    return isophote_rows, summary


def build_model_image(
    image_shape: tuple[int, int],
    profile_table: Table,
    method_name: str,
) -> np.ndarray:
    """Reconstruct 2-D model image using the method-specific builder."""
    normalized_method = method_name.strip().lower()
    if normalized_method == "photutils":
        return build_photutils_model_image(image_shape, profile_table)

    harmonic_orders = [3, 4]
    isophotes, filter_summary = extract_isoster_model_rows(
        profile_table, harmonic_orders=harmonic_orders
    )
    if (
        filter_summary["invalid_row_count"] > 0
        or filter_summary["duplicate_row_count"] > 0
    ):
        warnings.warn(
            (
                "Filtered isoster model rows before 2-D reconstruction: "
                f"invalid_rows={filter_summary['invalid_row_count']} "
                f"duplicate_rows={filter_summary['duplicate_row_count']} "
                f"valid_rows={filter_summary['valid_row_count']} "
                f"unique_rows={filter_summary['unique_row_count']}"
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    model_image = isoster.build_isoster_model(
        image_shape,
        isophotes,
        use_harmonics=True,
        harmonic_orders=harmonic_orders,
    )

    finite_mask = np.isfinite(model_image)
    if not np.all(finite_mask):
        non_finite_count = int(model_image.size - np.count_nonzero(finite_mask))
        warnings.warn(
            (
                "Non-finite values detected in isoster 2-D model output after filtering; "
                f"replacing with 0.0 (count={non_finite_count})."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        model_image = np.where(finite_mask, model_image, 0.0)

    return model_image


def compute_fractional_residual_percent(
    image: np.ndarray, model: np.ndarray
) -> np.ndarray:
    """Compute residual map as 100 * (model - data) / data."""
    residual = np.full(image.shape, np.nan, dtype=float)
    valid = np.isfinite(image) & (np.abs(image) > 0.0)
    residual[valid] = 100.0 * (model[valid] - image[valid]) / image[valid]
    return residual


def draw_isophote_overlays(
    axis,
    profile_table: Table,
    step: int,
    line_width: float = 1.0,
    alpha: float = 0.7,
    edge_color: str | None = None,
) -> None:
    """Overlay selective isophotes on an image axis."""
    if len(profile_table) == 0:
        return

    step = max(step, 1)
    for index in range(0, len(profile_table), step):
        sma = float(profile_table["sma"][index])
        if not np.isfinite(sma) or sma <= 1.0:
            continue

        x0 = float(profile_table["x0"][index])
        y0 = float(profile_table["y0"][index])
        eps = float(profile_table["eps"][index])
        pa_rad = float(profile_table["pa"][index])
        stop_code = int(profile_table["stop_code"][index])

        style = style_for_stop_code(stop_code)
        color = edge_color if edge_color is not None else style["color"]

        ellipse = mpl_ellipse(
            (x0, y0),
            2.0 * sma,
            2.0 * sma * (1.0 - eps),
            angle=np.rad2deg(pa_rad),
            fill=False,
            linewidth=line_width,
            alpha=alpha,
            edgecolor=color,
        )
        axis.add_patch(ellipse)


def robust_limits(
    values: np.ndarray, lower_percentile: float = 5.0, upper_percentile: float = 95.0
) -> tuple[float, float]:
    """Compute robust plotting limits from finite values."""
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return -1.0, 1.0
    low = float(np.nanpercentile(finite_values, lower_percentile))
    high = float(np.nanpercentile(finite_values, upper_percentile))
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        delta = max(abs(low), 1.0)
        return low - delta, high + delta
    return low, high


def set_axis_limits_from_finite_values(
    axis,
    values: np.ndarray,
    invert: bool = False,
    margin_fraction: float = 0.08,
    min_margin: float = 0.05,
    lower_clip: float | None = None,
    upper_clip: float | None = None,
) -> None:
    """Set y-axis limits from finite data values with a comfortable margin."""
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return

    value_low = float(np.nanmin(finite_values))
    value_high = float(np.nanmax(finite_values))
    if value_high <= value_low:
        margin = max(min_margin, 0.05 * max(abs(value_low), 1.0))
    else:
        margin = max(min_margin, margin_fraction * (value_high - value_low))

    limit_low = value_low - margin
    limit_high = value_high + margin
    if lower_clip is not None:
        limit_low = max(limit_low, lower_clip)
    if upper_clip is not None:
        limit_high = min(limit_high, upper_clip)

    if limit_high <= limit_low:
        midpoint = 0.5 * (value_low + value_high)
        half_width = max(min_margin, 0.02)
        limit_low = midpoint - half_width
        limit_high = midpoint + half_width
        if lower_clip is not None:
            limit_low = max(limit_low, lower_clip)
        if upper_clip is not None:
            limit_high = min(limit_high, upper_clip)
        if limit_high <= limit_low:
            return

    if invert:
        axis.set_ylim(limit_high, limit_low)
    else:
        axis.set_ylim(limit_low, limit_high)


def set_x_limits_with_right_margin(
    axis,
    x_values: np.ndarray,
    min_margin: float = 0.02,
    margin_fraction: float = 0.03,
) -> None:
    """Set x-axis limits with a small right-edge margin for readability."""
    finite_values = np.asarray(x_values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return

    x_low = float(np.nanmin(finite_values))
    x_high = float(np.nanmax(finite_values))
    width = max(x_high - x_low, 0.0)
    right_margin = max(min_margin, margin_fraction * max(width, 1.0))
    axis.set_xlim(x_low, x_high + right_margin)


def configure_qa_plot_style() -> None:
    """Apply shared plotting style for QA figures."""
    latex_available = (
        platform.system() != "Windows" and shutil.which("latex") is not None
    )
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "text.usetex": latex_available,
            "axes.labelsize": 15,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 14,
            "axes.unicode_minus": False,
        }
    )


def latex_safe_text(label_text: str) -> str:
    """Escape percent signs when LaTeX text rendering is active."""
    if plt.rcParams.get("text.usetex", False):
        return label_text.replace("%", r"\%")
    return label_text


def derive_arcsinh_parameters(
    image_values: np.ndarray,
    lower_percentile: float = 0.05,
    upper_percentile: float = 99.99,
    scale_percentile: float = 70.0,
) -> tuple[float, float, float, float]:
    """Derive arcsinh display parameters from image statistics."""
    finite_values = image_values[np.isfinite(image_values)]
    if finite_values.size == 0:
        return 0.0, 1.0, 1.0, 1.0

    low = float(np.nanpercentile(finite_values, lower_percentile))
    high = float(np.nanpercentile(finite_values, upper_percentile))
    if not np.isfinite(high) or high <= low:
        high = low + 1.0

    clipped = np.clip(image_values, low, high)
    shifted = np.clip(clipped - low, 0.0, None)
    positive = shifted[np.isfinite(shifted) & (shifted > 0.0)]
    scale = (
        float(np.nanpercentile(positive, scale_percentile)) if positive.size else 1.0
    )
    scale = max(scale, 1e-12)

    display = np.arcsinh(shifted / scale)
    finite_display = display[np.isfinite(display)]
    vmax = float(np.nanpercentile(finite_display, 99.8)) if finite_display.size else 1.0
    vmax = max(vmax, 1e-6)

    return low, high, scale, vmax


def make_arcsinh_display_from_parameters(
    image_values: np.ndarray,
    low: float,
    high: float,
    scale: float,
    vmax: float,
) -> tuple[np.ndarray, float, float]:
    """Build arcsinh display map from explicit scaling parameters."""
    clipped = np.clip(image_values, low, high)
    shifted = np.clip(clipped - low, 0.0, None)
    display = np.arcsinh(shifted / max(scale, 1e-12))
    return display, 0.0, max(vmax, 1e-6)


def make_arcsinh_display(
    image_values: np.ndarray,
    lower_percentile: float = 0.05,
    upper_percentile: float = 99.99,
    scale_percentile: float = 70.0,
) -> tuple[np.ndarray, float, float, float]:
    """Build arcsinh-scaled display map and limits for low-SB-friendly rendering."""
    low, high, scale, vmax = derive_arcsinh_parameters(
        image_values,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        scale_percentile=scale_percentile,
    )
    display, vmin, vmax = make_arcsinh_display_from_parameters(
        image_values,
        low=low,
        high=high,
        scale=scale,
        vmax=vmax,
    )
    return display, vmin, vmax, scale


def plot_profile_by_stop_code(
    axis,
    x_values: np.ndarray,
    y_values: np.ndarray,
    stop_codes: np.ndarray,
    y_errors: np.ndarray | None = None,
    marker_face: str = "filled",
    label_prefix: str = "",
    marker_size: float = 26.0,
    monochrome: bool = False,
) -> None:
    """Scatter/errorbar profile points split by stop codes."""
    unique_stop_codes = sorted(
        {int(code) for code in stop_codes[np.isfinite(stop_codes)]},
        key=lambda code: (
            (0, 0) if code == 0 else (1, code) if code > 0 else (2, abs(code))
        ),
    )
    for stop_code in unique_stop_codes:
        mask = stop_codes == stop_code
        if not np.any(mask):
            continue
        style = style_for_stop_code(stop_code, monochrome=monochrome)
        if monochrome and stop_code != 0:
            face_color = "none"
        else:
            face_color = style["color"] if marker_face == "filled" else "none"

        if y_errors is not None:
            sanitized_errors = np.asarray(y_errors[mask], dtype=float)
            sanitized_errors[sanitized_errors < 0.0] = np.nan
            axis.errorbar(
                x_values[mask],
                y_values[mask],
                yerr=sanitized_errors,
                fmt=style["marker"],
                markersize=4.5,
                color=style["color"],
                mfc=face_color,
                mec=style["color"],
                capsize=1.8,
                linewidth=0.7,
                alpha=0.85,
                label=f"{label_prefix}{style['label']}",
            )
        else:
            axis.scatter(
                x_values[mask],
                y_values[mask],
                s=marker_size,
                marker=style["marker"],
                facecolors=face_color,
                edgecolors=style["color"],
                linewidths=0.9,
                alpha=0.9,
                label=f"{label_prefix}{style['label']}",
            )


def build_method_qa_figure(
    image: np.ndarray,
    profile_table: Table,
    model_image: np.ndarray,
    output_path: Path,
    method_name: str,
    galaxy_name: str,
    mock_id: int,
    pixel_scale_arcsec: float,
    redshift: float,
    runtime_metadata: dict[str, Any],
    overlay_step: int,
    dpi: int,
) -> None:
    """Build per-method QA figure with 2D and 1D summaries."""
    x_values = np.asarray(profile_table["x_kpc_quarter"], dtype=float)
    sma = np.asarray(profile_table["sma"], dtype=float)
    stop_codes = np.asarray(profile_table["stop_code"], dtype=int)
    valid_radius_mask = np.isfinite(x_values) & (sma > 1.0)

    residual = compute_fractional_residual_percent(image, model_image)
    configure_qa_plot_style()

    figure = plt.figure(figsize=(13.6, 11.0), dpi=dpi)
    outer = gridspec.GridSpec(
        1, 2, figure=figure, width_ratios=[1.0, 2.01], wspace=0.27
    )
    left = gridspec.GridSpecFromSubplotSpec(
        3,
        2,
        subplot_spec=outer[0],
        width_ratios=[1.0, 0.04],
        hspace=0.10,
        wspace=-0.20,
    )
    right = gridspec.GridSpecFromSubplotSpec(
        5,
        1,
        subplot_spec=outer[1],
        height_ratios=[2.2, 1.0, 1.0, 1.0, 1.2],
        hspace=0.0,
    )

    wall_time = runtime_metadata["wall_time_seconds"]
    cpu_time = runtime_metadata["cpu_time_seconds"]
    run_title = (
        f"{galaxy_name}_mock{mock_id} | {method_name} | "
        f"z={redshift:.3f}, | "
        f"wall={wall_time:.2f}s cpu={cpu_time:.2f}s"
    )
    figure.suptitle(run_title, fontsize=20, y=0.989)

    reference_low, reference_high, reference_scale, reference_vmax = (
        derive_arcsinh_parameters(image)
    )

    axis_original = figure.add_subplot(left[0, 0])
    original_display, original_vmin, original_vmax = (
        make_arcsinh_display_from_parameters(
            image,
            low=reference_low,
            high=reference_high,
            scale=reference_scale,
            vmax=reference_vmax,
        )
    )
    original_handle = axis_original.imshow(
        original_display,
        origin="lower",
        cmap="viridis",
        vmin=original_vmin,
        vmax=original_vmax,
        interpolation="none",
    )
    draw_isophote_overlays(
        axis_original,
        profile_table,
        step=overlay_step,
        line_width=1.2,
        alpha=0.8,
        edge_color="orangered",
    )
    axis_original.text(
        0.15,
        0.9,
        "Data",
        fontsize=18,
        color="w",
        transform=axis_original.transAxes,
        ha="center",
        va="center",
        weight="bold",
        alpha=0.85,
    )

    axis_original_colorbar = figure.add_subplot(left[0, 1])
    colorbar_original = figure.colorbar(original_handle, cax=axis_original_colorbar)
    colorbar_original.set_label(r"arcsinh((data - p0.5) / scale)")

    axis_model = figure.add_subplot(left[1, 0])
    model_display, model_vmin, model_vmax = make_arcsinh_display_from_parameters(
        model_image,
        low=reference_low,
        high=reference_high,
        scale=reference_scale,
        vmax=reference_vmax,
    )
    model_handle = axis_model.imshow(
        model_display,
        origin="lower",
        cmap="viridis",
        vmin=model_vmin,
        vmax=model_vmax,
        interpolation="none",
    )
    axis_model.text(
        0.15,
        0.9,
        "Model",
        fontsize=18,
        color="w",
        transform=axis_model.transAxes,
        ha="center",
        va="center",
        weight="bold",
        alpha=0.85,
    )

    axis_model_colorbar = figure.add_subplot(left[1, 1])
    colorbar_model = figure.colorbar(model_handle, cax=axis_model_colorbar)
    colorbar_model.set_label(r"arcsinh((model - p0.5) / scale)")

    axis_residual = figure.add_subplot(left[2, 0])
    abs_residual = np.abs(residual[np.isfinite(residual)])
    residual_limit = np.nanpercentile(abs_residual, 99.0) if abs_residual.size else 1.0
    residual_limit = float(np.clip(residual_limit, 0.05, 8.0))
    residual_handle = axis_residual.imshow(
        residual,
        origin="lower",
        cmap="coolwarm",
        vmin=-residual_limit,
        vmax=residual_limit,
        interpolation="nearest",
    )
    axis_residual.text(
        0.18,
        0.9,
        "Residual",
        fontsize=18,
        color="k",
        transform=axis_residual.transAxes,
        ha="center",
        va="center",
        weight="bold",
        alpha=0.85,
    )

    axis_residual_colorbar = figure.add_subplot(left[2, 1])
    colorbar_residual = figure.colorbar(residual_handle, cax=axis_residual_colorbar)
    colorbar_residual.set_label(latex_safe_text("(model - data) / data [%]"))

    axis_surface_brightness = figure.add_subplot(right[0])
    axis_centroid = figure.add_subplot(right[1], sharex=axis_surface_brightness)
    axis_axis_ratio = figure.add_subplot(right[2], sharex=axis_surface_brightness)
    axis_position_angle = figure.add_subplot(right[3], sharex=axis_surface_brightness)
    axis_cog = figure.add_subplot(right[4], sharex=axis_surface_brightness)

    plot_mask = valid_radius_mask & np.isfinite(profile_table["sb_mag_arcsec2"])
    plot_profile_by_stop_code(
        axis_surface_brightness,
        x_values[plot_mask],
        np.asarray(profile_table["sb_mag_arcsec2"], dtype=float)[plot_mask],
        stop_codes[plot_mask],
        y_errors=np.asarray(profile_table["sb_err_mag"], dtype=float)[plot_mask],
        marker_face="filled",
        monochrome=True,
    )
    axis_surface_brightness.set_ylabel(r"$\mu$ [mag arcsec$^{-2}$]")
    axis_surface_brightness.grid(alpha=0.25)
    axis_surface_brightness.invert_yaxis()
    axis_surface_brightness.set_title("Surface brightness profile")

    sb_values = np.asarray(profile_table["sb_mag_arcsec2"], dtype=float)
    sb_valid_for_limits = plot_mask & np.isfinite(sb_values)
    if np.any(sb_valid_for_limits):
        set_axis_limits_from_finite_values(
            axis_surface_brightness,
            sb_values[sb_valid_for_limits],
            invert=True,
            margin_fraction=0.06,
            min_margin=0.2,
        )

    x0_error = np.full(len(profile_table), np.nan, dtype=float)
    y0_error = np.full(len(profile_table), np.nan, dtype=float)
    if "x0_err" in profile_table.colnames:
        x0_error = np.asarray(profile_table["x0_err"], dtype=float)
    if "y0_err" in profile_table.colnames:
        y0_error = np.asarray(profile_table["y0_err"], dtype=float)

    centroid_x = np.asarray(profile_table["x0_offset_pix"], dtype=float)
    centroid_y = np.asarray(profile_table["y0_offset_pix"], dtype=float)
    plot_profile_by_stop_code(
        axis_centroid,
        x_values[valid_radius_mask],
        centroid_x[valid_radius_mask],
        stop_codes[valid_radius_mask],
        y_errors=x0_error[valid_radius_mask],
        marker_face="filled",
        label_prefix="dx ",
        monochrome=True,
    )
    plot_profile_by_stop_code(
        axis_centroid,
        x_values[valid_radius_mask],
        centroid_y[valid_radius_mask],
        stop_codes[valid_radius_mask],
        y_errors=y0_error[valid_radius_mask],
        marker_face="open",
        label_prefix="dy ",
        monochrome=True,
    )
    axis_centroid.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    axis_centroid.set_ylabel("center offset [pix]")
    axis_centroid.grid(alpha=0.25)
    centroid_legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            color="black",
            markerfacecolor="black",
            markersize=4.8,
            label="X",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            color="black",
            markerfacecolor="none",
            markersize=4.8,
            label="Y",
        ),
    ]
    axis_centroid.legend(
        handles=centroid_legend_handles,
        loc="upper right",
        frameon=True,
        fontsize=14,
        ncol=2,
    )

    axis_ratio = np.asarray(profile_table["axis_ratio"], dtype=float)
    axis_ratio_error = np.full(len(profile_table), np.nan, dtype=float)
    if "ellip_err" in profile_table.colnames:
        axis_ratio_error = np.abs(np.asarray(profile_table["ellip_err"], dtype=float))
    elif "eps_err" in profile_table.colnames:
        axis_ratio_error = np.abs(np.asarray(profile_table["eps_err"], dtype=float))

    plot_profile_by_stop_code(
        axis_axis_ratio,
        x_values[valid_radius_mask],
        axis_ratio[valid_radius_mask],
        stop_codes[valid_radius_mask],
        y_errors=axis_ratio_error[valid_radius_mask],
        marker_face="filled",
        monochrome=True,
    )
    axis_axis_ratio.set_ylabel("axis ratio")
    axis_axis_ratio.grid(alpha=0.25)
    axis_ratio_for_limits = axis_ratio[valid_radius_mask & np.isfinite(axis_ratio)]
    set_axis_limits_from_finite_values(
        axis_axis_ratio,
        axis_ratio_for_limits,
        invert=False,
        margin_fraction=0.08,
        min_margin=0.03,
        lower_clip=0.0,
        upper_clip=1.0,
    )

    pa_norm = np.asarray(profile_table["pa_deg_norm"], dtype=float)
    pa_error_degrees = np.full(len(profile_table), np.nan, dtype=float)
    if "pa_err" in profile_table.colnames:
        pa_error_degrees = np.rad2deg(np.asarray(profile_table["pa_err"], dtype=float))

    plot_profile_by_stop_code(
        axis_position_angle,
        x_values[valid_radius_mask],
        pa_norm[valid_radius_mask],
        stop_codes[valid_radius_mask],
        y_errors=pa_error_degrees[valid_radius_mask],
        marker_face="filled",
        monochrome=True,
    )
    axis_position_angle.set_ylabel("PA [deg]")
    axis_position_angle.grid(alpha=0.25)

    pa_for_limits = pa_norm[
        valid_radius_mask & np.isfinite(pa_norm) & (stop_codes == 0)
    ]
    if pa_for_limits.size > 1:
        pa_low, pa_high = robust_limits(pa_for_limits, 3, 97)
        margin = max(3.0, 0.08 * (pa_high - pa_low + 1e-6))
        axis_position_angle.set_ylim(pa_low - margin, pa_high + margin)

    true_cog = np.asarray(profile_table["true_cog_flux"], dtype=float)
    fitted_cog = np.asarray(profile_table["method_cog_flux"], dtype=float)

    axis_cog.plot(
        x_values[valid_radius_mask],
        true_cog[valid_radius_mask],
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="true CoG",
    )
    plot_profile_by_stop_code(
        axis_cog,
        x_values[valid_radius_mask],
        fitted_cog[valid_radius_mask],
        stop_codes[valid_radius_mask],
        marker_face="filled",
        label_prefix="fit ",
        monochrome=True,
    )
    axis_cog.set_ylabel("CoG flux")
    axis_cog.set_xlabel(r"$R^{1/4}$ [kpc$^{1/4}$]")
    axis_cog.grid(alpha=0.25)

    finite_cog = fitted_cog[
        valid_radius_mask & np.isfinite(fitted_cog) & (fitted_cog > 0)
    ]
    if finite_cog.size > 0 and np.all(finite_cog > 0):
        axis_cog.set_yscale("log")

    if np.any(valid_radius_mask):
        x_valid = x_values[valid_radius_mask]
        set_x_limits_with_right_margin(axis_cog, x_valid)

    for axis in [
        axis_surface_brightness,
        axis_centroid,
        axis_axis_ratio,
        axis_position_angle,
    ]:
        axis.tick_params(labelbottom=False)

    handles, labels = axis_surface_brightness.get_legend_handles_labels()
    if handles:
        axis_surface_brightness.legend(
            handles[:8],
            labels[:8],
            loc="upper right",
            fontsize=14,
            ncol=1,
        )

    figure.subplots_adjust(left=0.025, right=0.992, bottom=0.05, top=0.940, wspace=0.18)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def interpolate_column(
    reference_sma: np.ndarray,
    reference_values: np.ndarray,
    target_sma: np.ndarray,
) -> np.ndarray:
    """Interpolate reference values onto a target SMA grid."""
    result = np.full(target_sma.shape, np.nan, dtype=float)

    finite_mask = np.isfinite(reference_sma) & np.isfinite(reference_values)
    if np.sum(finite_mask) < 2:
        return result

    sorted_indices = np.argsort(reference_sma[finite_mask])
    x_values = reference_sma[finite_mask][sorted_indices]
    y_values = reference_values[finite_mask][sorted_indices]

    target_mask = (
        np.isfinite(target_sma)
        & (target_sma >= x_values[0])
        & (target_sma <= x_values[-1])
    )
    result[target_mask] = np.interp(target_sma[target_mask], x_values, y_values)
    return result


def build_comparison_qa_figure(
    image: np.ndarray,
    photutils_table: Table,
    isoster_table: Table,
    photutils_model: np.ndarray,
    isoster_model: np.ndarray,
    output_path: Path,
    galaxy_name: str,
    mock_id: int,
    pixel_scale_arcsec: float,
    redshift: float,
    runtime_photutils: dict[str, Any],
    runtime_isoster: dict[str, Any],
    overlay_step: int,
    dpi: int,
) -> dict[str, float]:
    """Build cross-method comparison QA figure and return summary metrics."""
    x_photutils = np.asarray(photutils_table["x_kpc_quarter"], dtype=float)
    x_isoster = np.asarray(isoster_table["x_kpc_quarter"], dtype=float)

    sma_photutils = np.asarray(photutils_table["sma"], dtype=float)
    sma_isoster = np.asarray(isoster_table["sma"], dtype=float)

    valid_photutils = np.isfinite(x_photutils) & (sma_photutils > 1.0)
    valid_isoster = np.isfinite(x_isoster) & (sma_isoster > 1.0)

    residual_photutils = compute_fractional_residual_percent(image, photutils_model)
    residual_isoster = compute_fractional_residual_percent(image, isoster_model)
    configure_qa_plot_style()

    figure = plt.figure(figsize=(13.6, 11.0), dpi=dpi)
    outer = gridspec.GridSpec(
        1, 2, figure=figure, width_ratios=[1.0, 2.01], wspace=0.27
    )
    left = gridspec.GridSpecFromSubplotSpec(
        3,
        2,
        subplot_spec=outer[0],
        width_ratios=[1.0, 0.04],
        hspace=0.10,
        wspace=-0.20,
    )
    right = gridspec.GridSpecFromSubplotSpec(
        6,
        1,
        subplot_spec=outer[1],
        height_ratios=[2.0, 1.0, 1.0, 1.0, 1.0, 1.2],
        hspace=0.0,
    )

    title = (
        f"{galaxy_name}_mock{mock_id} | photutils vs isoster | "
        f"z={redshift:.3f}, | "
        f"photutils={runtime_photutils['wall_time_seconds']:.2f}s, "
        f"isoster={runtime_isoster['wall_time_seconds']:.2f}s"
    )
    figure.suptitle(title, fontsize=20, y=0.989)

    axis_original = figure.add_subplot(left[0, 0])
    image_display, image_vmin, image_vmax, _ = make_arcsinh_display(image)
    handle_original = axis_original.imshow(
        image_display,
        origin="lower",
        cmap="viridis",
        vmin=image_vmin,
        vmax=image_vmax,
        interpolation="none",
    )
    draw_isophote_overlays(
        axis_original,
        photutils_table,
        step=overlay_step,
        line_width=1.1,
        alpha=0.9,
        edge_color="orangered",
    )
    draw_isophote_overlays(
        axis_original,
        isoster_table,
        step=overlay_step,
        line_width=0.9,
        alpha=0.85,
        edge_color="w",
    )
    axis_original.text(
        0.15,
        0.9,
        "Data",
        fontsize=18,
        color="w",
        transform=axis_original.transAxes,
        ha="center",
        va="center",
        weight="bold",
        alpha=0.85,
    )

    axis_original_colorbar = figure.add_subplot(left[0, 1])
    colorbar_original = figure.colorbar(handle_original, cax=axis_original_colorbar)
    colorbar_original.set_label(r"arcsinh((data - p0.5) / scale)")

    abs_residual_photutils = np.abs(residual_photutils[np.isfinite(residual_photutils)])
    residual_limit_photutils = (
        np.nanpercentile(abs_residual_photutils, 99.0)
        if abs_residual_photutils.size
        else 1.0
    )
    residual_limit_photutils = float(np.clip(residual_limit_photutils, 0.05, 8.0))

    axis_residual_photutils = figure.add_subplot(left[1, 0])
    handle_residual_photutils = axis_residual_photutils.imshow(
        residual_photutils,
        origin="lower",
        cmap="coolwarm",
        vmin=-residual_limit_photutils,
        vmax=residual_limit_photutils,
        interpolation="nearest",
    )
    axis_residual_photutils.text(
        0.20,
        0.9,
        "photutils",
        fontsize=18,
        color="k",
        transform=axis_residual_photutils.transAxes,
        ha="center",
        va="center",
        weight="bold",
        alpha=0.85,
    )

    axis_photutils_colorbar = figure.add_subplot(left[1, 1])
    colorbar_photutils = figure.colorbar(
        handle_residual_photutils, cax=axis_photutils_colorbar
    )
    colorbar_photutils.set_label(latex_safe_text("(model - data) / data [%]"))

    abs_residual_isoster = np.abs(residual_isoster[np.isfinite(residual_isoster)])
    residual_limit_isoster = (
        np.nanpercentile(abs_residual_isoster, 99.0)
        if abs_residual_isoster.size
        else 1.0
    )
    residual_limit_isoster = float(np.clip(residual_limit_isoster, 0.05, 8.0))

    axis_residual_isoster = figure.add_subplot(left[2, 0])
    handle_residual_isoster = axis_residual_isoster.imshow(
        residual_isoster,
        origin="lower",
        cmap="coolwarm",
        vmin=-residual_limit_isoster,
        vmax=residual_limit_isoster,
        interpolation="nearest",
    )
    axis_residual_isoster.text(
        0.17,
        0.9,
        "isoster",
        fontsize=18,
        color="k",
        transform=axis_residual_isoster.transAxes,
        ha="center",
        va="center",
        weight="bold",
        alpha=0.85,
    )

    axis_isoster_colorbar = figure.add_subplot(left[2, 1])
    colorbar_isoster = figure.colorbar(
        handle_residual_isoster, cax=axis_isoster_colorbar
    )
    colorbar_isoster.set_label(latex_safe_text("(model - data) / data [%]"))

    for axis in [axis_original, axis_residual_photutils, axis_residual_isoster]:
        axis.set_xlabel("x [pixel]")
        axis.set_ylabel("y [pixel]")

    axis_sb = figure.add_subplot(right[0])
    axis_sb_relative = figure.add_subplot(right[1], sharex=axis_sb)
    axis_centroid = figure.add_subplot(right[2], sharex=axis_sb)
    axis_axis_ratio = figure.add_subplot(right[3], sharex=axis_sb)
    axis_pa = figure.add_subplot(right[4], sharex=axis_sb)
    axis_cog = figure.add_subplot(right[5], sharex=axis_sb)

    sb_photutils = np.asarray(photutils_table["sb_mag_arcsec2"], dtype=float)
    sb_isoster = np.asarray(isoster_table["sb_mag_arcsec2"], dtype=float)
    sb_error_photutils = np.asarray(photutils_table["sb_err_mag"], dtype=float)
    sb_error_isoster = np.asarray(isoster_table["sb_err_mag"], dtype=float)

    valid_photutils_sb = (
        valid_photutils
        & np.isfinite(sb_photutils)
        & np.isfinite(sb_error_photutils)
        & (sb_error_photutils >= 0.0)
    )
    valid_isoster_sb = (
        valid_isoster
        & np.isfinite(sb_isoster)
        & np.isfinite(sb_error_isoster)
        & (sb_error_isoster >= 0.0)
    )

    axis_sb.errorbar(
        x_photutils[valid_photutils_sb],
        sb_photutils[valid_photutils_sb],
        yerr=sb_error_photutils[valid_photutils_sb],
        fmt="o",
        mfc="none",
        mec="black",
        color="black",
        markersize=4,
        capsize=1.8,
        linewidth=0.7,
        alpha=0.8,
        label="photutils",
    )
    axis_sb.errorbar(
        x_isoster[valid_isoster_sb],
        sb_isoster[valid_isoster_sb],
        yerr=sb_error_isoster[valid_isoster_sb],
        fmt="^",
        mfc="black",
        mec="black",
        color="black",
        markersize=3.8,
        capsize=1.8,
        linewidth=0.7,
        alpha=0.8,
        label="isoster",
    )
    axis_sb.set_ylabel(r"$\mu$ [mag arcsec$^{-2}$]")
    axis_sb.set_title("Surface brightness")
    axis_sb.invert_yaxis()
    axis_sb.grid(alpha=0.25)
    axis_sb.legend(loc="upper right", fontsize=14)

    sb_values_for_limits = np.concatenate(
        [
            sb_photutils[valid_photutils & np.isfinite(sb_photutils)],
            sb_isoster[valid_isoster & np.isfinite(sb_isoster)],
        ]
    )
    set_axis_limits_from_finite_values(
        axis_sb,
        sb_values_for_limits,
        invert=True,
        margin_fraction=0.06,
        min_margin=0.2,
    )

    intensity_photutils = np.asarray(photutils_table["intens"], dtype=float)
    intensity_isoster = np.asarray(isoster_table["intens"], dtype=float)
    intensity_photutils_interp = interpolate_column(
        sma_photutils, intensity_photutils, sma_isoster
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        relative_sb_intensity_diff = (
            100.0
            * (intensity_isoster - intensity_photutils_interp)
            / intensity_photutils_interp
        )

    valid_relative = valid_isoster & np.isfinite(relative_sb_intensity_diff)
    axis_sb_relative.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axis_sb_relative.scatter(
        x_isoster[valid_relative],
        relative_sb_intensity_diff[valid_relative],
        s=22,
        marker="s",
        facecolors="black",
        edgecolors="black",
        alpha=0.85,
    )
    axis_sb_relative.set_ylabel(latex_safe_text("dI/I_phot [%]"))
    axis_sb_relative.grid(alpha=0.25)
    relative_values_for_limits = relative_sb_intensity_diff[
        valid_relative & np.isfinite(relative_sb_intensity_diff)
    ]
    if relative_values_for_limits.size > 1:
        relative_low, relative_high = robust_limits(relative_values_for_limits, 3, 97)
        relative_amplitude = max(abs(relative_low), abs(relative_high))
        relative_margin = max(0.8, 0.12 * max(relative_amplitude, 1.0))
        axis_sb_relative.set_ylim(
            -(relative_amplitude + relative_margin),
            relative_amplitude + relative_margin,
        )

    x0_phot = np.asarray(photutils_table["x0_offset_pix"], dtype=float)
    y0_phot = np.asarray(photutils_table["y0_offset_pix"], dtype=float)
    x0_iso = np.asarray(isoster_table["x0_offset_pix"], dtype=float)
    y0_iso = np.asarray(isoster_table["y0_offset_pix"], dtype=float)

    x0_err_phot = (
        np.asarray(photutils_table["x0_err"], dtype=float)
        if "x0_err" in photutils_table.colnames
        else np.full(len(photutils_table), np.nan, dtype=float)
    )
    y0_err_phot = (
        np.asarray(photutils_table["y0_err"], dtype=float)
        if "y0_err" in photutils_table.colnames
        else np.full(len(photutils_table), np.nan, dtype=float)
    )
    x0_err_iso = (
        np.asarray(isoster_table["x0_err"], dtype=float)
        if "x0_err" in isoster_table.colnames
        else np.full(len(isoster_table), np.nan, dtype=float)
    )
    y0_err_iso = (
        np.asarray(isoster_table["y0_err"], dtype=float)
        if "y0_err" in isoster_table.colnames
        else np.full(len(isoster_table), np.nan, dtype=float)
    )

    axis_centroid.errorbar(
        x_photutils[valid_photutils],
        x0_phot[valid_photutils],
        yerr=x0_err_phot[valid_photutils],
        fmt="o",
        mfc="black",
        mec="black",
        color="black",
        markersize=3.8,
        capsize=1.6,
        linewidth=0.7,
        alpha=0.85,
        label="photutils dx",
    )
    axis_centroid.errorbar(
        x_photutils[valid_photutils],
        y0_phot[valid_photutils],
        yerr=y0_err_phot[valid_photutils],
        fmt="o",
        mfc="none",
        mec="black",
        color="black",
        markersize=3.8,
        capsize=1.6,
        linewidth=0.7,
        alpha=0.85,
        label="photutils dy",
    )
    axis_centroid.errorbar(
        x_isoster[valid_isoster],
        x0_iso[valid_isoster],
        yerr=x0_err_iso[valid_isoster],
        fmt="^",
        mfc="black",
        mec="black",
        color="black",
        markersize=3.7,
        capsize=1.5,
        linewidth=0.7,
        alpha=0.85,
        label="isoster dx",
    )
    axis_centroid.errorbar(
        x_isoster[valid_isoster],
        y0_iso[valid_isoster],
        yerr=y0_err_iso[valid_isoster],
        fmt="^",
        mfc="none",
        mec="black",
        color="black",
        markersize=3.7,
        capsize=1.5,
        linewidth=0.7,
        alpha=0.85,
        label="isoster dy",
    )
    axis_centroid.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    axis_centroid.set_ylabel("center offset [pix]")
    axis_centroid.grid(alpha=0.25)
    axis_centroid.legend(loc="best", fontsize=10, ncol=2)
    centroid_values_for_limits = np.concatenate(
        [
            x0_phot[valid_photutils & np.isfinite(x0_phot)],
            y0_phot[valid_photutils & np.isfinite(y0_phot)],
            x0_iso[valid_isoster & np.isfinite(x0_iso)],
            y0_iso[valid_isoster & np.isfinite(y0_iso)],
        ]
    )
    if centroid_values_for_limits.size > 1:
        centroid_low, centroid_high = robust_limits(centroid_values_for_limits, 3, 97)
        centroid_margin = max(0.5, 0.12 * (centroid_high - centroid_low + 1e-6))
        axis_centroid.set_ylim(
            centroid_low - centroid_margin, centroid_high + centroid_margin
        )

    axis_ratio_photutils = np.asarray(photutils_table["axis_ratio"], dtype=float)
    axis_ratio_isoster = np.asarray(isoster_table["axis_ratio"], dtype=float)
    axis_ratio_error_phot = (
        np.asarray(photutils_table["ellip_err"], dtype=float)
        if "ellip_err" in photutils_table.colnames
        else np.asarray(photutils_table["eps_err"], dtype=float)
        if "eps_err" in photutils_table.colnames
        else np.full(len(photutils_table), np.nan, dtype=float)
    )
    axis_ratio_error_iso = (
        np.asarray(isoster_table["ellip_err"], dtype=float)
        if "ellip_err" in isoster_table.colnames
        else np.asarray(isoster_table["eps_err"], dtype=float)
        if "eps_err" in isoster_table.colnames
        else np.full(len(isoster_table), np.nan, dtype=float)
    )
    axis_axis_ratio.errorbar(
        x_photutils[valid_photutils],
        axis_ratio_photutils[valid_photutils],
        yerr=np.abs(axis_ratio_error_phot[valid_photutils]),
        fmt="o",
        mfc="none",
        mec="black",
        color="black",
        markersize=3.8,
        capsize=1.6,
        linewidth=0.7,
        alpha=0.85,
    )
    axis_axis_ratio.errorbar(
        x_isoster[valid_isoster],
        axis_ratio_isoster[valid_isoster],
        yerr=np.abs(axis_ratio_error_iso[valid_isoster]),
        fmt="^",
        mfc="black",
        mec="black",
        color="black",
        markersize=3.7,
        capsize=1.5,
        linewidth=0.7,
        alpha=0.85,
    )
    axis_axis_ratio.set_ylabel("axis ratio")
    axis_axis_ratio.grid(alpha=0.25)
    axis_ratio_for_limits = np.concatenate(
        [
            axis_ratio_photutils[valid_photutils & np.isfinite(axis_ratio_photutils)],
            axis_ratio_isoster[valid_isoster & np.isfinite(axis_ratio_isoster)],
        ]
    )
    set_axis_limits_from_finite_values(
        axis_axis_ratio,
        axis_ratio_for_limits,
        invert=False,
        margin_fraction=0.08,
        min_margin=0.03,
        lower_clip=0.0,
        upper_clip=1.0,
    )

    pa_photutils = np.asarray(photutils_table["pa_deg_norm"], dtype=float)
    pa_isoster = np.asarray(isoster_table["pa_deg_norm"], dtype=float)
    axis_pa.scatter(
        x_photutils[valid_photutils],
        pa_photutils[valid_photutils],
        s=22,
        marker="o",
        facecolors="none",
        edgecolors="black",
        alpha=0.8,
    )
    axis_pa.scatter(
        x_isoster[valid_isoster],
        pa_isoster[valid_isoster],
        s=20,
        marker="^",
        facecolors="black",
        edgecolors="black",
        alpha=0.8,
    )
    axis_pa.set_ylabel("PA [deg]")
    axis_pa.grid(alpha=0.25)
    pa_values_for_limits = np.concatenate(
        [
            pa_photutils[valid_photutils & np.isfinite(pa_photutils)],
            pa_isoster[valid_isoster & np.isfinite(pa_isoster)],
        ]
    )
    if pa_values_for_limits.size > 1:
        pa_low, pa_high = robust_limits(pa_values_for_limits, 3, 97)
        pa_margin = max(3.0, 0.08 * (pa_high - pa_low + 1e-6))
        axis_pa.set_ylim(pa_low - pa_margin, pa_high + pa_margin)

    cog_photutils = np.asarray(photutils_table["method_cog_flux"], dtype=float)
    cog_isoster = np.asarray(isoster_table["method_cog_flux"], dtype=float)
    axis_cog.scatter(
        x_photutils[valid_photutils],
        cog_photutils[valid_photutils],
        s=22,
        marker="o",
        facecolors="none",
        edgecolors="black",
        alpha=0.8,
        label="photutils fit CoG",
    )
    axis_cog.scatter(
        x_isoster[valid_isoster],
        cog_isoster[valid_isoster],
        s=20,
        marker="^",
        facecolors="black",
        edgecolors="black",
        alpha=0.8,
        label="isoster fit CoG",
    )

    true_cog_photutils = np.asarray(photutils_table["true_cog_flux"], dtype=float)
    true_cog_isoster = np.asarray(isoster_table["true_cog_flux"], dtype=float)
    axis_cog.plot(
        x_photutils[valid_photutils],
        true_cog_photutils[valid_photutils],
        linestyle="--",
        color="black",
        linewidth=1.0,
        alpha=0.8,
    )
    axis_cog.plot(
        x_isoster[valid_isoster],
        true_cog_isoster[valid_isoster],
        linestyle=":",
        color="black",
        linewidth=1.1,
        alpha=0.85,
    )

    axis_cog.set_ylabel("CoG flux")
    axis_cog.set_xlabel(r"$R^{1/4}$ [kpc$^{1/4}$]")
    axis_cog.grid(alpha=0.25)

    if np.any((cog_photutils > 0) & np.isfinite(cog_photutils)) and np.any(
        (cog_isoster > 0) & np.isfinite(cog_isoster)
    ):
        axis_cog.set_yscale("log")

    x_limits = []
    if np.any(valid_photutils):
        x_limits.extend(
            [
                np.nanmin(x_photutils[valid_photutils]),
                np.nanmax(x_photutils[valid_photutils]),
            ]
        )
    if np.any(valid_isoster):
        x_limits.extend(
            [np.nanmin(x_isoster[valid_isoster]), np.nanmax(x_isoster[valid_isoster])]
        )
    if x_limits:
        set_x_limits_with_right_margin(axis_cog, np.asarray(x_limits, dtype=float))

    for axis in [axis_sb, axis_sb_relative, axis_centroid, axis_axis_ratio, axis_pa]:
        axis.tick_params(labelbottom=False)

    figure.subplots_adjust(left=0.025, right=0.992, bottom=0.05, top=0.940, wspace=0.18)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)

    valid_relative_for_summary = relative_sb_intensity_diff[
        np.isfinite(relative_sb_intensity_diff)
    ]
    if valid_relative_for_summary.size:
        median_abs_rel_sb = float(np.nanmedian(np.abs(valid_relative_for_summary)))
        max_abs_rel_sb = float(np.nanmax(np.abs(valid_relative_for_summary)))
    else:
        median_abs_rel_sb = float("nan")
        max_abs_rel_sb = float("nan")

    return {
        "median_abs_relative_sb_percent": median_abs_rel_sb,
        "max_abs_relative_sb_percent": max_abs_rel_sb,
    }


def summarize_table(profile_table: Table) -> dict[str, Any]:
    """Build concise summary metrics for report output."""
    if len(profile_table) == 0:
        return {
            "isophote_count": 0,
            "converged_count": 0,
            "stop_code_counts": {},
            "median_abs_cog_rel_diff": float("nan"),
        }

    stop_codes = np.asarray(profile_table["stop_code"], dtype=int)
    unique_stop, counts = np.unique(stop_codes, return_counts=True)
    stop_summary = {
        str(int(code)): int(count) for code, count in zip(unique_stop, counts)
    }

    cog_diff = np.asarray(profile_table["cog_rel_diff"], dtype=float)
    finite_cog_diff = cog_diff[np.isfinite(cog_diff)]
    median_abs_cog_rel_diff = (
        float(np.nanmedian(np.abs(finite_cog_diff)))
        if finite_cog_diff.size
        else float("nan")
    )

    return {
        "isophote_count": int(len(profile_table)),
        "converged_count": int(np.sum(stop_codes == 0)),
        "stop_code_counts": stop_summary,
        "median_abs_cog_rel_diff": median_abs_cog_rel_diff,
    }


def dump_json(file_path: Path, payload: dict[str, Any]) -> None:
    """Serialize JSON with deterministic formatting."""
    file_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_markdown_report(
    report_path: Path,
    prefix: str,
    input_path: Path,
    run_metadata: dict[str, Any],
    summary_photutils: dict[str, Any] | None,
    summary_isoster: dict[str, Any] | None,
    comparison_summary: dict[str, Any] | None,
) -> None:
    """Write concise report for the current demo run."""
    lines = [
        f"# {prefix} demo report",
        "",
        "## Input",
        "",
        f"- FITS: `{input_path}`",
        f"- SHA256: `{run_metadata['input_sha256']}`",
        f"- Redshift: `{run_metadata['redshift']}`",
        f"- Pixel scale [arcsec/pix]: `{run_metadata['pixel_scale_arcsec']}`",
        f"- Zeropoint [mag]: `{run_metadata['zeropoint_mag']}`",
        "",
        "## Runtime",
        "",
    ]

    for method_name in ["photutils", "isoster"]:
        if method_name in run_metadata["method_runs"]:
            runtime = run_metadata["method_runs"][method_name]["runtime"]
            lines.append(
                f"- {method_name}: wall=`{runtime['wall_time_seconds']:.3f}` s, "
                f"cpu=`{runtime['cpu_time_seconds']:.3f}` s"
            )

    lines.extend(["", "## Method Summaries", ""])

    if summary_photutils is not None:
        lines.append("### photutils")
        lines.append("")
        lines.append(f"- Isophotes: `{summary_photutils['isophote_count']}`")
        lines.append(f"- Converged: `{summary_photutils['converged_count']}`")
        lines.append(f"- Stop codes: `{summary_photutils['stop_code_counts']}`")
        lines.append(
            "- Median |CoG relative difference|: "
            f"`{summary_photutils['median_abs_cog_rel_diff']:.6g}`"
        )
        lines.append("")

    if summary_isoster is not None:
        lines.append("### isoster")
        lines.append("")
        lines.append(f"- Isophotes: `{summary_isoster['isophote_count']}`")
        lines.append(f"- Converged: `{summary_isoster['converged_count']}`")
        lines.append(f"- Stop codes: `{summary_isoster['stop_code_counts']}`")
        lines.append(
            "- Median |CoG relative difference|: "
            f"`{summary_isoster['median_abs_cog_rel_diff']:.6g}`"
        )
        lines.append("")

    if comparison_summary is not None:
        lines.append("## Cross-method Summary")
        lines.append("")
        lines.append(
            "- Median |relative surface-brightness difference| [%]: "
            f"`{comparison_summary['median_abs_relative_sb_percent']:.6g}`"
        )
        lines.append(
            "- Max |relative surface-brightness difference| [%]: "
            f"`{comparison_summary['max_abs_relative_sb_percent']:.6g}`"
        )
        lines.append("")

    warning_entries = run_metadata.get("warnings", [])
    if isinstance(warning_entries, list) and warning_entries:
        lines.append("## Warnings")
        lines.append("")
        for warning_entry in warning_entries:
            if isinstance(warning_entry, dict):
                message = warning_entry.get("message", str(warning_entry))
            else:
                message = str(warning_entry)
            lines.append(f"- {message}")
        lines.append("")

    report_path.write_text("\n".join(lines) + "\n")


def collect_software_versions() -> dict[str, str]:
    """Collect software version metadata."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "astropy": astropy.__version__,
        "photutils": photutils.__version__,
        "isoster": getattr(isoster, "__version__", "unknown"),
    }


def main() -> None:
    """Main entrypoint for the Huang2013 real-mock demo workflow."""
    args = parse_arguments()

    input_fits = build_input_path(args)
    if not input_fits.exists():
        raise FileNotFoundError(f"Input FITS does not exist: {input_fits}")

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else build_case_output_dir(args.huang_root, args.galaxy, args.mock_id)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    image, header = read_mock_image(input_fits)

    inferred = infer_initial_geometry(header, image.shape)
    redshift = float(
        args.redshift
        if args.redshift is not None
        else get_header_value(header, "REDSHIFT", DEFAULT_REDSHIFT)
    )
    pixel_scale_arcsec = float(
        args.pixel_scale
        if args.pixel_scale is not None
        else get_header_value(header, "PIXSCALE", DEFAULT_PIXEL_SCALE_ARCSEC)
    )
    zeropoint_mag = float(
        args.zeropoint
        if args.zeropoint is not None
        else get_header_value(header, "MAGZERO", DEFAULT_ZEROPOINT)
    )
    psf_fwhm_arcsec = float(
        args.psf_fwhm
        if args.psf_fwhm is not None
        else get_header_value(header, "PSFFWHM", np.nan)
    )

    max_sma = args.maxsma
    if max_sma is None:
        max_sma = 0.48 * min(image.shape)

    maxgerr = args.maxgerr
    if maxgerr is None:
        maxgerr = 1.0 if inferred["eps"] > 0.55 else 0.5

    config_tag = sanitize_label(args.config_tag)
    prefix = build_case_prefix(args.galaxy, args.mock_id)

    run_metadata: dict[str, Any] = {
        "prefix": prefix,
        "input_fits": str(input_fits),
        "input_sha256": compute_sha256(input_fits),
        "redshift": redshift,
        "pixel_scale_arcsec": pixel_scale_arcsec,
        "zeropoint_mag": zeropoint_mag,
        "psf_fwhm_arcsec": psf_fwhm_arcsec,
        "cog_subpixels": args.cog_subpixels,
        "software_versions": collect_software_versions(),
        "method_runs": {},
    }

    method_profile_paths: dict[str, Path] = {}
    method_tables: dict[str, Table] = {}
    method_runtime: dict[str, dict[str, Any]] = {}

    methods_to_run = (
        [args.method]
        if args.method in {"photutils", "isoster"}
        else ["photutils", "isoster"]
    )

    for method_name in methods_to_run:
        stem = f"{prefix}_{method_name}_{config_tag}"
        profile_fits_path = output_dir / f"{stem}_profile.fits"
        profile_ecsv_path = output_dir / f"{stem}_profile.ecsv"
        runtime_profile_path = output_dir / f"{stem}_runtime-profile.txt"
        method_json_path = output_dir / f"{stem}_run.json"
        method_qa_path = output_dir / f"{stem}_qa.png"

        if method_name == "photutils":
            photutils_config = {
                "x0": inferred["x0"],
                "y0": inferred["y0"],
                "eps": inferred["eps"],
                "pa_deg": inferred["pa_deg"],
                "sma0": float(args.sma0 if args.sma0 is not None else inferred["sma0"]),
                "minsma": float(args.minsma),
                "maxsma": float(max_sma),
                "astep": float(args.astep),
                "maxgerr": float(maxgerr),
                "nclip": int(args.photutils_nclip),
                "sclip": float(args.photutils_sclip),
                "integrmode": args.photutils_integrmode,
            }
            isophotes, runtime_info, profile_text = run_with_runtime_profile(
                run_photutils_fit,
                image,
                photutils_config,
            )
            validated_config = photutils_config
        else:
            isoster_config = {
                "x0": inferred["x0"],
                "y0": inferred["y0"],
                "eps": inferred["eps"],
                "pa": float(np.deg2rad(inferred["pa_deg"])),
                "sma0": float(args.sma0 if args.sma0 is not None else inferred["sma0"]),
                "minsma": float(args.minsma),
                "maxsma": float(max_sma),
                "astep": float(args.astep),
                "maxgerr": float(maxgerr),
                "nclip": 2,
                "sclip": 3.0,
                "conver": 0.05,
                "minit": 10,
                "maxit": 100,
                "compute_errors": True,
                "compute_deviations": True,
                "full_photometry": True,
                "compute_cog": True,
                "use_eccentric_anomaly": bool(args.use_eccentric_anomaly),
                "simultaneous_harmonics": False,
                "harmonic_orders": [3, 4],
                "permissive_geometry": False,
                "use_central_regularization": False,
            }

            if args.isoster_config_json is not None:
                user_overrides = json.loads(args.isoster_config_json.read_text())
                if "pa_deg" in user_overrides and "pa" not in user_overrides:
                    user_overrides["pa"] = float(
                        np.deg2rad(float(user_overrides.pop("pa_deg")))
                    )
                isoster_config.update(user_overrides)

            fit_output, runtime_info, profile_text = run_with_runtime_profile(
                run_isoster_fit,
                image,
                isoster_config,
            )
            isophotes = fit_output[0]
            validated_config = fit_output[1]

        profile_table = prepare_profile_table(
            isophotes=isophotes,
            image=image,
            redshift=redshift,
            pixel_scale_arcsec=pixel_scale_arcsec,
            zeropoint_mag=zeropoint_mag,
            cog_subpixels=args.cog_subpixels,
            method_name=method_name,
        )

        model_image = build_model_image(
            image.shape,
            profile_table,
            method_name=method_name,
        )
        build_method_qa_figure(
            image=image,
            profile_table=profile_table,
            model_image=model_image,
            output_path=method_qa_path,
            method_name=method_name,
            galaxy_name=args.galaxy,
            mock_id=args.mock_id,
            pixel_scale_arcsec=pixel_scale_arcsec,
            redshift=redshift,
            runtime_metadata=runtime_info,
            overlay_step=args.isophote_overlay_step,
            dpi=args.qa_dpi,
        )

        profile_table.write(profile_fits_path, overwrite=True)
        profile_table.write(profile_ecsv_path, format="ascii.ecsv", overwrite=True)
        runtime_profile_path.write_text(profile_text)

        method_payload = {
            "prefix": prefix,
            "method": method_name,
            "config_tag": config_tag,
            "input_fits": str(input_fits),
            "runtime": runtime_info,
            "fit_config": validated_config,
            "table_summary": summarize_table(profile_table),
            "outputs": {
                "profile_fits": str(profile_fits_path),
                "profile_ecsv": str(profile_ecsv_path),
                "runtime_profile": str(runtime_profile_path),
                "qa_figure": str(method_qa_path),
            },
        }
        dump_json(method_json_path, method_payload)

        run_metadata["method_runs"][method_name] = {
            "runtime": runtime_info,
            "run_json": str(method_json_path),
            "qa_figure": str(method_qa_path),
            "profile_fits": str(profile_fits_path),
            "profile_ecsv": str(profile_ecsv_path),
            "runtime_profile": str(runtime_profile_path),
            "fit_config": validated_config,
        }

        method_profile_paths[method_name] = profile_fits_path
        method_tables[method_name] = profile_table
        method_runtime[method_name] = runtime_info

    if (
        "photutils" not in method_profile_paths
        and args.photutils_profile_fits is not None
    ):
        method_profile_paths["photutils"] = args.photutils_profile_fits
        method_tables["photutils"] = Table.read(args.photutils_profile_fits)
        harmonize_method_cog_columns(
            method_tables["photutils"], method_name="photutils"
        )
        method_runtime["photutils"] = {
            "wall_time_seconds": np.nan,
            "cpu_time_seconds": np.nan,
        }

    if "isoster" not in method_profile_paths and args.isoster_profile_fits is not None:
        method_profile_paths["isoster"] = args.isoster_profile_fits
        method_tables["isoster"] = Table.read(args.isoster_profile_fits)
        harmonize_method_cog_columns(method_tables["isoster"], method_name="isoster")
        method_runtime["isoster"] = {
            "wall_time_seconds": np.nan,
            "cpu_time_seconds": np.nan,
        }

    comparison_summary = None
    comparison_figure_path = None

    if (
        not args.skip_comparison
        and "photutils" in method_tables
        and "isoster" in method_tables
    ):
        comparison_figure_path = output_dir / f"{prefix}_compare_{config_tag}_qa.png"

        photutils_model = build_model_image(
            image.shape,
            method_tables["photutils"],
            method_name="photutils",
        )
        isoster_model = build_model_image(
            image.shape,
            method_tables["isoster"],
            method_name="isoster",
        )

        comparison_summary = build_comparison_qa_figure(
            image=image,
            photutils_table=method_tables["photutils"],
            isoster_table=method_tables["isoster"],
            photutils_model=photutils_model,
            isoster_model=isoster_model,
            output_path=comparison_figure_path,
            galaxy_name=args.galaxy,
            mock_id=args.mock_id,
            pixel_scale_arcsec=pixel_scale_arcsec,
            redshift=redshift,
            runtime_photutils=method_runtime["photutils"],
            runtime_isoster=method_runtime["isoster"],
            overlay_step=args.isophote_overlay_step,
            dpi=args.qa_dpi,
        )

    summary_photutils = (
        summarize_table(method_tables["photutils"])
        if "photutils" in method_tables
        else None
    )
    summary_isoster = (
        summarize_table(method_tables["isoster"])
        if "isoster" in method_tables
        else None
    )

    report_path = output_dir / f"{prefix}_report.md"
    write_markdown_report(
        report_path=report_path,
        prefix=prefix,
        input_path=input_fits,
        run_metadata=run_metadata,
        summary_photutils=summary_photutils,
        summary_isoster=summary_isoster,
        comparison_summary=comparison_summary,
    )

    manifest_path = output_dir / f"{prefix}_manifest.json"
    manifest_payload = {
        "prefix": prefix,
        "input_fits": str(input_fits),
        "output_dir": str(output_dir),
        "method_profile_paths": {
            key: str(value) for key, value in method_profile_paths.items()
        },
        "comparison_figure": str(comparison_figure_path)
        if comparison_figure_path is not None
        else None,
        "report": str(report_path),
        "run_metadata": run_metadata,
        "comparison_summary": comparison_summary,
    }
    dump_json(manifest_path, manifest_payload)

    print(f"Input FITS: {input_fits}")
    print(f"Output directory: {output_dir}")
    print(f"Manifest: {manifest_path}")
    if comparison_figure_path is not None:
        print(f"Comparison QA: {comparison_figure_path}")


if __name__ == "__main__":
    main()
