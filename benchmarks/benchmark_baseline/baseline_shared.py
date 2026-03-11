"""Shared helpers for baseline benchmark: galaxy registry, data loading, QA figures.

Supports isoster, photutils, and autoprof comparison runs with timed core
fitting, per-method 2D model reconstruction, and multi-method QA figures.
"""

from __future__ import annotations

import json
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.table import Table

import isoster
from isoster.config import IsosterConfig
from isoster.model import build_isoster_model
from isoster.plotting import (
    compute_fractional_residual_percent,
    configure_qa_plot_style,
    derive_arcsinh_parameters,
    draw_isophote_overlays,
    make_arcsinh_display_from_parameters,
    normalize_pa_degrees,
    robust_limits,
    set_axis_limits_from_finite_values,
    set_x_limits_with_right_margin,
    style_for_stop_code,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Method style constants for QA figures
# ---------------------------------------------------------------------------

METHOD_STYLES = {
    "isoster": {
        "color": "#1f77b4",
        "marker": "o",
        "marker_face": "filled",
        "overlay_color": "white",
        "overlay_width": 1.0,
        "label": "isoster",
    },
    "photutils": {
        "color": "#d62728",
        "marker": "s",
        "marker_face": "none",
        "overlay_color": "orangered",
        "overlay_width": 1.1,
        "label": "photutils",
    },
    "autoprof": {
        "color": "#2ca02c",
        "marker": "^",
        "marker_face": "filled",
        "overlay_color": "lime",
        "overlay_width": 0.9,
        "label": "AutoProf",
    },
}

# ---------------------------------------------------------------------------
# Galaxy registry
# ---------------------------------------------------------------------------

# Each entry: name, fits_path, cube_index (None for 2D), geometry overrides,
# config overrides, pixel_scale, zeropoint.

GALAXY_REGISTRY: list[dict[str, Any]] = [
    {
        "name": "eso243-49",
        "fits_path": DATA_DIR / "eso243-49.fits",
        "cube_index": 0,
        "geometry": None,  # estimate at runtime
        "config_overrides": {
            "sma0": 10.0,
            "minsma": 1.0,
            "maxsma": 118.0,
            "astep": 0.1,
            "minit": 10,
            "maxit": 50,
            "conver": 0.05,
        },
        "pixel_scale": 1.0,
        "zeropoint": 22.5,
    },
    {
        "name": "IC3370_mock2",
        "fits_path": DATA_DIR / "IC3370_mock2.fits",
        "cube_index": None,
        "geometry": {"x0": 566.0, "y0": 566.0, "eps": 0.239, "pa": -0.489},
        "config_overrides": {
            "sma0": 6.0,
            "minsma": 1.0,
            "maxsma": 283.0,
            "astep": 0.1,
            "minit": 10,
            "maxit": 50,
            "conver": 0.05,
        },
        "pixel_scale": 1.0,
        "zeropoint": 22.5,
    },
    {
        "name": "ngc3610",
        "fits_path": DATA_DIR / "ngc3610.fits",
        "cube_index": 0,
        "geometry": None,  # estimate at runtime
        "config_overrides": {
            "sma0": 5.0,
            "minsma": 1.0,
            "maxsma": 118.0,
            "astep": 0.1,
            "minit": 10,
            "maxit": 50,
            "conver": 0.05,
        },
        "pixel_scale": 1.0,
        "zeropoint": 22.5,
    },
]


def get_galaxy(name: str) -> dict[str, Any]:
    """Look up a galaxy entry by name (case-insensitive)."""
    key = name.lower().replace("-", "").replace("_", "")
    for entry in GALAXY_REGISTRY:
        if entry["name"].lower().replace("-", "").replace("_", "") == key:
            return entry
    available = ", ".join(e["name"] for e in GALAXY_REGISTRY)
    raise ValueError(f"Unknown galaxy '{name}'. Available: {available}")


# ---------------------------------------------------------------------------
# Data loading and geometry estimation
# ---------------------------------------------------------------------------


def load_galaxy_image(fits_path: Path, cube_index: int | None = None) -> np.ndarray:
    """Load FITS image as float64.  Handles 2D and 3D cubes."""
    with fits.open(fits_path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim == 2:
                return np.asarray(hdu.data, dtype=np.float64)
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim == 3:
                idx = cube_index if cube_index is not None else 0
                return np.asarray(hdu.data[idx], dtype=np.float64)
    raise ValueError(f"No usable image data in {fits_path}")


def estimate_moment_geometry(image: np.ndarray, percentile: float = 80.0) -> dict:
    """Estimate center, ellipticity, and PA from image moments."""
    threshold = np.nanpercentile(image, percentile)
    weights = np.clip(image - threshold, 0, None)
    total = np.nansum(weights)
    if total <= 0:
        h, w = image.shape
        return {"x0": w / 2.0, "y0": h / 2.0, "eps": 0.2, "pa": 0.0}

    yy, xx = np.mgrid[: image.shape[0], : image.shape[1]].astype(float)
    x0 = float(np.nansum(weights * xx) / total)
    y0 = float(np.nansum(weights * yy) / total)
    dx, dy = xx - x0, yy - y0
    mxx = float(np.nansum(weights * dx**2) / total)
    myy = float(np.nansum(weights * dy**2) / total)
    mxy = float(np.nansum(weights * dx * dy) / total)
    trace = mxx + myy
    det = mxx * myy - mxy**2
    disc = max(trace**2 / 4 - det, 0)
    lam1 = trace / 2 + np.sqrt(disc)
    lam2 = trace / 2 - np.sqrt(disc)
    axis_ratio = np.sqrt(max(lam2, 0) / max(lam1, 1e-12))
    eps = float(np.clip(1 - axis_ratio, 0.05, 0.95))
    pa = float(0.5 * np.arctan2(2 * mxy, mxx - myy))
    return {"x0": x0, "y0": y0, "eps": eps, "pa": pa}


def estimate_background(image: np.ndarray, box_size: int = 50) -> tuple[float, float]:
    """Estimate background level and noise from image corners."""
    h, w = image.shape
    bs = min(box_size, h // 4, w // 4)
    corners = np.concatenate([
        image[:bs, :bs].ravel(),
        image[:bs, -bs:].ravel(),
        image[-bs:, :bs].ravel(),
        image[-bs:, -bs:].ravel(),
    ])
    corners = corners[np.isfinite(corners)]
    if corners.size == 0:
        return 0.0, 1.0
    return float(np.median(corners)), float(np.std(corners))


def resolve_geometry(galaxy_entry: dict[str, Any], image: np.ndarray) -> dict:
    """Resolve geometry from registry entry or estimate from image."""
    if galaxy_entry["geometry"] is not None:
        return dict(galaxy_entry["geometry"])
    return estimate_moment_geometry(image)


# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------


def check_photutils_available() -> bool:
    """Return True if photutils.isophote is importable."""
    try:
        from photutils.isophote import Ellipse, EllipseGeometry  # noqa: F401
        return True
    except ImportError:
        return False


def check_autoprof_available() -> bool:
    """Return True if AutoProf is available via the subprocess adapter."""
    try:
        from benchmarks.utils.autoprof_adapter import check_autoprof_available as _check
        return _check()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Isoster fitting (timed core only)
# ---------------------------------------------------------------------------


def run_isoster_fit(
    image: np.ndarray,
    geometry: dict,
    config_overrides: dict[str, Any],
) -> dict[str, Any]:
    """Run isoster fit.  Times only the core fit_image() call."""
    config_kwargs = dict(geometry)
    config_kwargs.update(config_overrides)
    config = IsosterConfig(**config_kwargs)

    # Time only the core fitting
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    result = isoster.fit_image(image, None, config)
    wall_time = time.perf_counter() - wall_start
    cpu_time = time.process_time() - cpu_start

    isophotes = result["isophotes"]
    code_counts = dict(Counter(iso["stop_code"] for iso in isophotes))

    return {
        "method": "isoster",
        "isophotes": isophotes,
        "config": config,
        "runtime": {
            "wall_time_seconds": round(wall_time, 4),
            "cpu_time_seconds": round(cpu_time, 4),
        },
        "isophote_count": len(isophotes),
        "stop_code_counts": {str(k): v for k, v in sorted(code_counts.items())},
    }


# ---------------------------------------------------------------------------
# Photutils fitting (timed core only)
# ---------------------------------------------------------------------------


def build_photutils_config(geometry: dict, config_overrides: dict) -> dict[str, Any]:
    """Build photutils-compatible config dict from geometry and overrides."""
    return {
        "x0": geometry["x0"],
        "y0": geometry["y0"],
        "eps": geometry["eps"],
        "pa": geometry["pa"],
        "sma0": config_overrides["sma0"],
        "minsma": config_overrides.get("minsma", 1.0),
        "maxsma": config_overrides.get("maxsma", None),
        "step": config_overrides.get("astep", 0.1),
        "maxgerr": 0.5,
        "nclip": 0,
        "sclip": 3.0,
        "integrmode": "bilinear",
    }


def _convert_photutils_isolist(isolist) -> list[dict[str, Any]]:
    """Convert photutils isolist to serializable list of dicts."""
    attribute_map = {
        "sma": "sma", "intens": "intens", "intens_err": "int_err",
        "eps": "eps", "ellip_err": "ellip_err",
        "pa": "pa", "pa_err": "pa_err",
        "x0": "x0", "x0_err": "x0_err", "y0": "y0", "y0_err": "y0_err",
        "grad": "grad", "grad_error": "grad_error",
        "rms": "rms", "pix_stddev": "pix_stddev",
        "stop_code": "stop_code", "ndata": "ndata", "nflag": "nflag",
        "niter": "niter",
        "a3": "a3", "b3": "b3", "a4": "a4", "b4": "b4",
        "a3_err": "a3_err", "b3_err": "b3_err",
        "a4_err": "a4_err", "b4_err": "b4_err",
    }
    isophotes: list[dict[str, Any]] = []
    for iso in isolist:
        row: dict[str, Any] = {}
        for output_key, source_key in attribute_map.items():
            value = getattr(iso, source_key, np.nan)
            row[output_key] = np.nan if value is None else value
        isophotes.append(row)
    return isophotes


def run_photutils_fit(
    image: np.ndarray,
    photutils_config: dict[str, Any],
) -> dict[str, Any] | None:
    """Run photutils.isophote fit.  Times only the core fit_image() call."""
    if not check_photutils_available():
        return None

    from photutils.isophote import Ellipse, EllipseGeometry

    # Setup (excluded from timing)
    geometry = EllipseGeometry(
        x0=photutils_config["x0"],
        y0=photutils_config["y0"],
        sma=photutils_config["sma0"],
        eps=photutils_config["eps"],
        pa=photutils_config["pa"],
    )
    ellipse = Ellipse(image, geometry)

    fit_kwargs = {
        "step": photutils_config["step"],
        "minsma": photutils_config["minsma"],
        "maxgerr": photutils_config["maxgerr"],
        "nclip": photutils_config["nclip"],
        "sclip": photutils_config["sclip"],
        "integrmode": photutils_config["integrmode"],
    }
    if photutils_config["maxsma"] is not None:
        fit_kwargs["maxsma"] = photutils_config["maxsma"]

    # Time only the core fitting
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        isolist = ellipse.fit_image(**fit_kwargs)
    wall_time = time.perf_counter() - wall_start
    cpu_time = time.process_time() - cpu_start

    isophotes = _convert_photutils_isolist(isolist)
    code_counts = dict(Counter(iso["stop_code"] for iso in isophotes))

    return {
        "method": "photutils",
        "isophotes": isophotes,
        "isolist": isolist,
        "runtime": {
            "wall_time_seconds": round(wall_time, 4),
            "cpu_time_seconds": round(cpu_time, 4),
        },
        "isophote_count": len(isophotes),
        "stop_code_counts": {str(k): v for k, v in sorted(code_counts.items())},
    }


# ---------------------------------------------------------------------------
# AutoProf fitting
# ---------------------------------------------------------------------------


def build_autoprof_config(
    geometry: dict,
    config_overrides: dict,
    pixel_scale: float,
    zeropoint: float,
    background: float,
    background_noise: float,
) -> dict[str, Any]:
    """Build autoprof-compatible config dict."""
    return {
        "pixel_scale": pixel_scale,
        "zeropoint": zeropoint,
        "center": [geometry["x0"], geometry["y0"]],
        "eps": geometry["eps"],
        "pa_rad_math": geometry["pa"],
        "background": background,
        "background_noise": background_noise,
    }


def prepare_2d_fits_for_autoprof(
    image: np.ndarray,
    output_dir: Path,
    galaxy_name: str,
) -> Path:
    """Write a 2D FITS file for AutoProf (handles 3D cubes transparently)."""
    out_path = output_dir / f"{galaxy_name}_2d.fits"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU(data=image.astype(np.float32))
    hdu.writeto(out_path, overwrite=True)
    return out_path


def run_autoprof_fit(
    image: np.ndarray,
    output_dir: Path,
    galaxy_name: str,
    autoprof_config: dict[str, Any],
) -> dict[str, Any] | None:
    """Run AutoProf via subprocess adapter.  Returns profile dict or None."""
    if not check_autoprof_available():
        return None

    from benchmarks.utils.autoprof_adapter import run_autoprof_fit as _run_autoprof

    autoprof_output_dir = output_dir / "autoprof_workdir"
    autoprof_output_dir.mkdir(parents=True, exist_ok=True)

    # AutoProf needs a 2D FITS file on disk
    fits_2d_path = prepare_2d_fits_for_autoprof(
        image, autoprof_output_dir, galaxy_name,
    )

    result = _run_autoprof(
        image_path=fits_2d_path,
        output_dir=autoprof_output_dir,
        galaxy_name=galaxy_name,
        pixel_scale=autoprof_config["pixel_scale"],
        zeropoint=autoprof_config["zeropoint"],
        center=tuple(autoprof_config["center"]),
        eps=autoprof_config["eps"],
        pa_rad_math=autoprof_config["pa_rad_math"],
        background=autoprof_config["background"],
        background_noise=autoprof_config["background_noise"],
        run_ellipse_model=True,
    )

    if result is None:
        return None

    return {
        "method": "autoprof",
        "profile": result,
        "runtime": {
            "wall_time_seconds": round(result.get("runtime_s", 0.0), 4),
            "cpu_time_seconds": 0.0,  # not available from subprocess
        },
        "isophote_count": result.get("n_isophotes", 0),
    }


# ---------------------------------------------------------------------------
# Model reconstruction
# ---------------------------------------------------------------------------


def build_isoster_model_image(
    image_shape: tuple[int, int],
    isophotes: list[dict],
) -> np.ndarray | None:
    """Build 2D model using isoster's native model builder."""
    if not isophotes:
        return None
    try:
        return build_isoster_model(image_shape, isophotes)
    except Exception as exc:
        print(f"    isoster model build failed: {exc}")
        return None


def build_photutils_model_image(
    image_shape: tuple[int, int],
    isophotes: list[dict],
) -> np.ndarray | None:
    """Build 2D model using photutils' native build_ellipse_model."""
    if not isophotes:
        return None

    try:
        from photutils.isophote import build_ellipse_model
    except ImportError:
        return None

    # Build the adapter (duck-typed isolist for photutils model builder)
    required = ["sma", "intens", "eps", "pa", "x0", "y0", "grad"]
    harmonic_keys = ["a3", "b3", "a4", "b4"]

    # Filter to valid rows
    valid_rows = []
    for iso in isophotes:
        if iso.get("sma", 0) <= 0:
            continue
        if not all(np.isfinite(iso.get(k, np.nan)) for k in required):
            continue
        valid_rows.append(iso)

    if len(valid_rows) < 6:
        print(f"    photutils model: insufficient valid rows ({len(valid_rows)})")
        return None

    # Sort by sma and deduplicate
    valid_rows.sort(key=lambda r: r["sma"])
    seen_sma = set()
    unique_rows = []
    for row in valid_rows:
        if row["sma"] not in seen_sma:
            seen_sma.add(row["sma"])
            unique_rows.append(row)

    if len(unique_rows) < 6:
        return None

    # Build column arrays
    columns = {}
    for key in required + harmonic_keys:
        columns[key] = np.array(
            [r.get(key, 0.0) for r in unique_rows], dtype=float
        )

    class _SmaNode:
        def __init__(self, sma_value):
            self.sma = float(sma_value)

    class _IsolistAdapter:
        def __init__(self, cols):
            for k, v in cols.items():
                setattr(self, k, v)
            self._nodes = [_SmaNode(s) for s in cols["sma"]]

        def __len__(self):
            return len(self._nodes)

        def __getitem__(self, index):
            return self._nodes[index]

    try:
        adapter = _IsolistAdapter(columns)
        return build_ellipse_model(
            image_shape, adapter, fill=0.0,
            high_harmonics=True, sma_interval=0.1,
        )
    except Exception as exc:
        print(f"    photutils model build failed: {exc}")
        return None


def load_autoprof_model_image(
    autoprof_result: dict[str, Any],
) -> np.ndarray | None:
    """Load AutoProf's native 2D model from its FITS output."""
    profile = autoprof_result.get("profile", {})
    model_path = profile.get("model_fits_path")
    if not model_path or not Path(model_path).exists():
        return None

    try:
        with fits.open(model_path) as hdul:
            for hdu in hdul:
                if hdu.data is not None and hdu.data.ndim == 2:
                    return np.asarray(hdu.data, dtype=np.float64)
    except Exception as exc:
        print(f"    AutoProf model load failed: {exc}")
    return None


# ---------------------------------------------------------------------------
# Artifact I/O
# ---------------------------------------------------------------------------


def save_profile_fits(isophotes: list[dict], output_path: Path) -> None:
    """Save isophote profile as a FITS table."""
    if not isophotes:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Table().write(output_path, format="fits", overwrite=True)
        return

    keys = [
        k for k in isophotes[0]
        if isinstance(isophotes[0][k], (int, float, np.integer, np.floating))
    ]
    table = Table()
    for key in keys:
        table[key] = [iso.get(key, np.nan) for iso in isophotes]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.write(output_path, format="fits", overwrite=True)


def save_profile_ecsv(isophotes: list[dict], output_path: Path) -> None:
    """Save isophote profile as an ECSV table."""
    if not isophotes:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Table().write(output_path, format="ascii.ecsv", overwrite=True)
        return

    keys = [
        k for k in isophotes[0]
        if isinstance(isophotes[0][k], (int, float, np.integer, np.floating))
    ]
    table = Table()
    for key in keys:
        table[key] = [iso.get(key, np.nan) for iso in isophotes]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.write(output_path, format="ascii.ecsv", overwrite=True)


def save_autoprof_profile_ecsv(profile: dict, output_path: Path) -> None:
    """Save autoprof profile arrays as an ECSV table."""
    table = Table()
    for key in ["sma", "intens", "eps", "pa", "intens_err", "eps_err", "pa_err"]:
        if key in profile and isinstance(profile[key], np.ndarray):
            table[key] = profile[key]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.write(output_path, format="ascii.ecsv", overwrite=True)


def save_model_fits(model: np.ndarray, output_path: Path) -> None:
    """Save 2D model image as a FITS file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU(data=model.astype(np.float32))
    hdu.writeto(output_path, overwrite=True)


def save_fit_configs(configs: dict[str, Any], output_path: Path) -> None:
    """Save all method fit configurations as JSON."""

    def _serialize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(configs, fp, indent=2, sort_keys=True, default=_serialize)


# ---------------------------------------------------------------------------
# QA figures
# ---------------------------------------------------------------------------


def _extract_profile_arrays(isophotes: list[dict]) -> dict[str, np.ndarray]:
    """Extract standard profile arrays from isophote list."""
    sma = np.array([iso["sma"] for iso in isophotes])
    return {
        "sma": sma,
        "x_axis": sma**0.25,
        "intens": np.array([iso["intens"] for iso in isophotes]),
        "eps": np.array([iso["eps"] for iso in isophotes]),
        "pa": np.array([iso["pa"] for iso in isophotes]),
        "x0": np.array([iso["x0"] for iso in isophotes]),
        "y0": np.array([iso["y0"] for iso in isophotes]),
        "stop_codes": np.array([iso["stop_code"] for iso in isophotes]),
        "rms": np.array([iso.get("rms", np.nan) for iso in isophotes]),
    }


def _extract_autoprof_profile_arrays(profile: dict) -> dict[str, np.ndarray]:
    """Extract profile arrays from autoprof result dict."""
    sma = profile["sma"]
    return {
        "sma": sma,
        "x_axis": sma**0.25,
        "intens": profile["intens"],
        "eps": profile["eps"],
        "pa": profile["pa"],
        "intens_err": profile.get("intens_err", np.full_like(sma, np.nan)),
    }


def make_comparison_qa_figure(
    image: np.ndarray,
    methods_data: dict[str, dict[str, Any]],
    galaxy_name: str,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Build multi-method comparison QA figure.

    Parameters
    ----------
    image : np.ndarray
        Original galaxy image.
    methods_data : dict
        Keys are method names ('isoster', 'photutils', 'autoprof').
        Values are dicts with keys:
        - 'isophotes' or 'profile': list[dict] or autoprof profile dict
        - 'model': np.ndarray or None (2D reconstructed model)
        - 'runtime': dict with wall_time_seconds
    galaxy_name : str
        Galaxy identifier for the title.
    output_path : Path
        Output path for the figure.
    dpi : int
        Figure resolution.
    """
    import matplotlib
    matplotlib.rcParams["text.usetex"] = False
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    configure_qa_plot_style()

    available_methods = list(methods_data.keys())
    n_models = sum(
        1 for m in available_methods
        if methods_data[m].get("model") is not None
    )
    # Left column: original image + residual images (one per method with model)
    n_left_rows = 1 + max(n_models, 1)
    # Right column: SB, relative diff, eps, PA, centroid
    n_right_rows = 5

    fig = plt.figure(figsize=(15, max(12, n_left_rows * 3)), dpi=dpi)
    outer = gridspec.GridSpec(
        1, 2, figure=fig, width_ratios=[1.0, 1.8], wspace=0.25,
    )
    left = gridspec.GridSpecFromSubplotSpec(
        n_left_rows, 2, subplot_spec=outer[0],
        width_ratios=[1.0, 0.04], hspace=0.12, wspace=0.05,
    )
    right = gridspec.GridSpecFromSubplotSpec(
        n_right_rows, 1, subplot_spec=outer[1],
        height_ratios=[2.0, 1.0, 1.0, 1.0, 1.0],
        hspace=0.0,
    )

    # Build title with runtime info
    runtime_parts = []
    for method_name in available_methods:
        rt = methods_data[method_name].get("runtime", {})
        wall = rt.get("wall_time_seconds", 0.0)
        style = METHOD_STYLES.get(method_name, {})
        label = style.get("label", method_name)
        runtime_parts.append(f"{label}={wall:.2f}s")
    title = f"{galaxy_name} | {', '.join(runtime_parts)}"
    fig.suptitle(title, fontsize=16, y=0.995)

    # --- Left column: Image + Residual images ---
    # Row 0: Original image with overlays from all methods
    ax_img = fig.add_subplot(left[0, 0])
    low, high, scale, vmax = derive_arcsinh_parameters(image)
    display, _, disp_vmax = make_arcsinh_display_from_parameters(
        image, low, high, scale, vmax,
    )
    handle_img = ax_img.imshow(
        display, cmap="viridis", origin="lower", vmin=0, vmax=disp_vmax,
        interpolation="none",
    )
    ax_cbar = fig.add_subplot(left[0, 1])
    fig.colorbar(handle_img, cax=ax_cbar)

    # Draw isophote overlays for each method
    for method_name in available_methods:
        mdata = methods_data[method_name]
        style = METHOD_STYLES.get(method_name, {})
        isos = mdata.get("isophotes", [])
        if isos:
            draw_isophote_overlays(
                ax_img, isos, step=5,
                line_width=style.get("overlay_width", 1.0),
                alpha=0.8,
                edge_color=style.get("overlay_color", "white"),
            )
    ax_img.set_title("Data", fontsize=12)
    ax_img.set_xlabel("x (pixels)")
    ax_img.set_ylabel("y (pixels)")

    # Residual images for each method with a model
    residual_row = 1
    for method_name in available_methods:
        mdata = methods_data[method_name]
        model = mdata.get("model")
        if model is None:
            continue

        style = METHOD_STYLES.get(method_name, {})
        label = style.get("label", method_name)
        residual_pct = compute_fractional_residual_percent(image, model)

        abs_vals = np.abs(residual_pct[np.isfinite(residual_pct)])
        res_limit = float(np.clip(
            np.nanpercentile(abs_vals, 99.0) if abs_vals.size else 1.0,
            0.05, 8.0,
        ))

        ax_res = fig.add_subplot(left[residual_row, 0])
        handle_res = ax_res.imshow(
            residual_pct, origin="lower", cmap="coolwarm",
            vmin=-res_limit, vmax=res_limit, interpolation="nearest",
        )
        ax_res.set_title(f"{label} residual", fontsize=11)
        ax_res.set_xlabel("x (pixels)")
        ax_res.set_ylabel("y (pixels)")

        ax_res_cbar = fig.add_subplot(left[residual_row, 1])
        cbar = fig.colorbar(handle_res, cax=ax_res_cbar)
        cbar.set_label("(model-data)/data [%]", fontsize=8)

        residual_row += 1

    # --- Right column: 1D profiles ---
    # Collect profile arrays per method
    profiles: dict[str, dict[str, np.ndarray]] = {}
    for method_name in available_methods:
        mdata = methods_data[method_name]
        if method_name == "autoprof":
            profile = mdata.get("profile")
            if profile is not None:
                profiles[method_name] = _extract_autoprof_profile_arrays(profile)
        else:
            isos = mdata.get("isophotes", [])
            if isos:
                profiles[method_name] = _extract_profile_arrays(isos)

    if not profiles:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return

    # Determine shared x-axis range
    all_x = np.concatenate([p["x_axis"] for p in profiles.values()])

    # Panel 0: Surface brightness (larger)
    ax_sb = fig.add_subplot(right[0])
    for method_name, prof in profiles.items():
        style = METHOD_STYLES.get(method_name, {})
        x = prof["x_axis"]
        y = np.log10(np.clip(prof["intens"], 1e-30, None))
        mfc = style["color"] if style.get("marker_face") == "filled" else "none"
        ax_sb.scatter(
            x, y, color=style["color"], marker=style["marker"],
            facecolors=mfc, edgecolors=style["color"],
            s=22, alpha=0.7, label=style.get("label", method_name), zorder=3,
        )
    ax_sb.set_ylabel(r"$\log_{10}$(Intensity)")
    ax_sb.set_title("Surface Brightness", fontsize=12)
    ax_sb.grid(alpha=0.25)
    ax_sb.legend(loc="upper right", fontsize=10)
    ax_sb.tick_params(labelbottom=False)
    set_x_limits_with_right_margin(ax_sb, all_x)

    # Panel 1: Relative SB difference (isoster as reference)
    ax_diff = fig.add_subplot(right[1], sharex=ax_sb)
    if "isoster" in profiles:
        ref_sma = profiles["isoster"]["sma"]
        ref_intens = profiles["isoster"]["intens"]
        for method_name, prof in profiles.items():
            if method_name == "isoster":
                continue
            style = METHOD_STYLES.get(method_name, {})
            # Interpolate reference to this method's SMA grid
            from scipy.interpolate import interp1d
            valid_ref = np.isfinite(ref_intens) & (ref_intens > 0)
            if valid_ref.sum() < 2:
                continue
            interp_func = interp1d(
                ref_sma[valid_ref], ref_intens[valid_ref],
                kind="linear", bounds_error=False, fill_value=np.nan,
            )
            ref_at_method = interp_func(prof["sma"])
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_diff = 100.0 * (prof["intens"] - ref_at_method) / ref_at_method
            valid = np.isfinite(rel_diff)
            mfc = style["color"] if style.get("marker_face") == "filled" else "none"
            ax_diff.scatter(
                prof["x_axis"][valid], rel_diff[valid],
                color=style["color"], marker=style["marker"],
                facecolors=mfc, edgecolors=style["color"],
                s=18, alpha=0.7, label=style.get("label", method_name), zorder=3,
            )
        ax_diff.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax_diff.set_ylabel("dI/I_isoster [%]")
    ax_diff.grid(alpha=0.25)
    ax_diff.tick_params(labelbottom=False)
    if len(profiles) > 1:
        ax_diff.legend(loc="best", fontsize=9)

    # Panel 2: Ellipticity
    ax_eps = fig.add_subplot(right[2], sharex=ax_sb)
    for method_name, prof in profiles.items():
        style = METHOD_STYLES.get(method_name, {})
        x = prof["x_axis"]
        y = prof["eps"]
        mfc = style["color"] if style.get("marker_face") == "filled" else "none"
        ax_eps.scatter(
            x, y, color=style["color"], marker=style["marker"],
            facecolors=mfc, edgecolors=style["color"],
            s=18, alpha=0.7, zorder=3,
        )
    ax_eps.set_ylabel("Ellipticity")
    ax_eps.grid(alpha=0.25)
    ax_eps.tick_params(labelbottom=False)

    # Panel 3: PA (normalized)
    ax_pa = fig.add_subplot(right[3], sharex=ax_sb)
    for method_name, prof in profiles.items():
        style = METHOD_STYLES.get(method_name, {})
        x = prof["x_axis"]
        pa_deg = np.degrees(prof["pa"])
        pa_norm = normalize_pa_degrees(pa_deg)
        mfc = style["color"] if style.get("marker_face") == "filled" else "none"
        ax_pa.scatter(
            x, pa_norm, color=style["color"], marker=style["marker"],
            facecolors=mfc, edgecolors=style["color"],
            s=18, alpha=0.7, zorder=3,
        )
    ax_pa.set_ylabel("PA (deg)")
    ax_pa.grid(alpha=0.25)
    ax_pa.tick_params(labelbottom=False)

    # Panel 4: Center offset (only for methods with x0/y0)
    ax_cen = fig.add_subplot(right[4], sharex=ax_sb)
    for method_name, prof in profiles.items():
        if "x0" not in prof:
            continue
        style = METHOD_STYLES.get(method_name, {})
        x = prof["x_axis"]
        x0_med = np.nanmedian(prof["x0"])
        y0_med = np.nanmedian(prof["y0"])
        offset = np.sqrt((prof["x0"] - x0_med)**2 + (prof["y0"] - y0_med)**2)
        mfc = style["color"] if style.get("marker_face") == "filled" else "none"
        ax_cen.scatter(
            x, offset, color=style["color"], marker=style["marker"],
            facecolors=mfc, edgecolors=style["color"],
            s=18, alpha=0.7, zorder=3,
        )
    ax_cen.set_ylabel("Center Offset (pix)")
    ax_cen.set_xlabel(r"SMA$^{0.25}$ (pix$^{0.25}$)")
    ax_cen.grid(alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
