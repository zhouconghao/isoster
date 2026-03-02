#!/usr/bin/env python
"""
Real Galaxy Benchmark: isoster vs AutoProf

Compares isophote fitting accuracy and performance between isoster and AutoProf
on real galaxy FITS images.  Both tools receive identical inputs (center,
ellipticity, PA, background) so the comparison is fair.

AutoProf's automatic background/PSF/center steps are bypassed via ap_set_*
parameters.  See docs/lessons-autoprof.md for convention details.

Usage:
    # Quick smoke test (IC3370 only)
    uv run python benchmarks/performance/bench_vs_autoprof.py --quick --plots

    # Full benchmark (all galaxies)
    uv run python benchmarks/performance/bench_vs_autoprof.py --plots

    # Skip AutoProf (replot from cached isoster results)
    uv run python benchmarks/performance/bench_vs_autoprof.py --skip-autoprof --plots
"""

import sys
import csv
import time
import json
import argparse
import os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Isolate matplotlib cache to avoid polluting user env
if "XDG_CACHE_HOME" not in os.environ:
    xdg_cache_dir = PROJECT_ROOT / "outputs" / "tmp" / "xdg-cache"
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache_dir)
if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = PROJECT_ROOT / "outputs" / "tmp" / "mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402
from matplotlib.patches import Ellipse as MplEllipse  # noqa: E402
from astropy.io import fits as afits  # noqa: E402
import isoster  # noqa: E402
from isoster.config import IsosterConfig  # noqa: E402
from isoster.model import build_isoster_model  # noqa: E402
from isoster.output_paths import resolve_output_directory  # noqa: E402
from benchmarks.utils.run_metadata import (  # noqa: E402
    collect_environment_metadata,
    write_json,
)
from benchmarks.utils.autoprof_adapter import (  # noqa: E402
    check_autoprof_available,
    run_autoprof_fit,
)

# ---------------------------------------------------------------------------
# Galaxy registry
# ---------------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"

# IC3370_mock2 values from benchmarks/ic3370_exhausted/config_registry.py
GALAXY_REGISTRY = [
    {
        "name": "IC3370_mock2",
        "fits_path": DATA_DIR / "IC3370_mock2.fits",
        "pixel_scale": 0.168,
        "zeropoint": 27.0,
        "center": (566.0, 566.0),
        "eps": 0.23877770323309452,
        "pa": -0.48862880623300525,
        "background": None,  # estimated at runtime
        "background_noise": None,
        "sma0": 6.0,
        "maxsma": 283.0,
    },
    {
        "name": "eso243-49",
        "fits_path": DATA_DIR / "eso243-49.fits",
        "pixel_scale": 0.25,
        "zeropoint": 22.5,
        "center": None,  # estimated at runtime
        "eps": None,
        "pa": None,
        "background": None,
        "background_noise": None,
        "sma0": 10.0,
        "maxsma": None,
    },
    {
        "name": "ngc3610",
        "fits_path": DATA_DIR / "ngc3610.fits",
        "pixel_scale": 1.0,
        "zeropoint": 22.5,
        "center": None,
        "eps": None,
        "pa": None,
        "background": None,
        "background_noise": None,
        "sma0": 5.0,
        "maxsma": None,
    },
]

QUICK_GALAXIES = ["IC3370_mock2"]


def resolve_benchmark_output_directory(output_dir=None):
    """Return output directory for this benchmark run."""
    return resolve_output_directory(
        "benchmarks_performance",
        "bench_vs_autoprof",
        explicit_output_directory=output_dir,
    )


# ---------------------------------------------------------------------------
# Parameter estimation
# ---------------------------------------------------------------------------

def estimate_background(image: np.ndarray) -> tuple[float, float]:
    """Estimate background level and noise from image corners.

    Samples a 50x50 pixel region from each corner and computes the
    median (background) and standard deviation (noise).

    Returns
    -------
    background : float
        Median pixel value in the corner regions.
    background_noise : float
        Standard deviation of pixel values in the corner regions.
    """
    size = 50
    ny, nx = image.shape
    corners = [
        image[:size, :size],
        image[:size, -size:],
        image[-size:, :size],
        image[-size:, -size:],
    ]
    pixels = np.concatenate([c.ravel() for c in corners])
    return float(np.median(pixels)), float(np.std(pixels))


def estimate_initial_params(image: np.ndarray, galaxy_info: dict) -> dict:
    """Estimate missing initial parameters for a galaxy.

    Runs a quick isoster fit with default guesses to obtain center,
    ellipticity, and PA.  Background is estimated from image corners.

    Parameters
    ----------
    image : np.ndarray
        Galaxy image.
    galaxy_info : dict
        Galaxy registry entry (may have None values for unknown params).

    Returns
    -------
    dict
        Updated galaxy_info with all parameters filled in.
    """
    info = dict(galaxy_info)

    # Background estimation
    if info["background"] is None or info["background_noise"] is None:
        bg, bg_noise = estimate_background(image)
        if info["background"] is None:
            info["background"] = bg
        if info["background_noise"] is None:
            info["background_noise"] = bg_noise
        print(f"  Estimated background: {info['background']:.2f} +/- {info['background_noise']:.4f}")

    # Center, eps, pa estimation via quick isoster fit
    if info["center"] is None or info["eps"] is None or info["pa"] is None:
        print("  Running preliminary isoster fit to estimate geometry...")
        ny, nx = image.shape
        cx, cy = nx / 2.0, ny / 2.0
        prelim_config = {
            "x0": cx,
            "y0": cy,
            "eps": 0.2,
            "pa": 0.0,
            "sma0": info["sma0"],
            "minsma": 1.0,
            "maxsma": info["maxsma"] or min(nx, ny) / 2 - 20,
            "astep": 0.2,  # coarse step for speed
            "maxit": 30,
            "compute_errors": False,
            "compute_deviations": False,
        }
        try:
            result = isoster.fit_image(image, None, prelim_config)
            isophotes = result["isophotes"]
            if isophotes:
                # Use median geometry from the middle third of the profile
                n_iso = len(isophotes)
                mid_start = n_iso // 3
                mid_end = 2 * n_iso // 3
                mid_isos = isophotes[mid_start:mid_end] if mid_end > mid_start else isophotes
                if info["center"] is None:
                    median_x0 = float(np.median([iso["x0"] for iso in mid_isos]))
                    median_y0 = float(np.median([iso["y0"] for iso in mid_isos]))
                    info["center"] = (median_x0, median_y0)
                if info["eps"] is None:
                    info["eps"] = float(np.median([iso["eps"] for iso in mid_isos]))
                if info["pa"] is None:
                    info["pa"] = float(np.median([iso["pa"] for iso in mid_isos]))

                print(f"  Estimated center: ({info['center'][0]:.1f}, {info['center'][1]:.1f})")
                print(f"  Estimated eps: {info['eps']:.3f}")
                print(f"  Estimated PA: {info['pa']:.3f} rad ({np.degrees(info['pa']):.1f} deg)")
        except Exception as exc:
            print(f"  Preliminary fit failed: {exc}")
            # Fallback to image center with round guess
            ny, nx = image.shape
            if info["center"] is None:
                info["center"] = (nx / 2.0, ny / 2.0)
            if info["eps"] is None:
                info["eps"] = 0.2
            if info["pa"] is None:
                info["pa"] = 0.0

    # maxsma default
    if info["maxsma"] is None:
        ny, nx = image.shape
        info["maxsma"] = min(nx, ny) / 2 - 20

    return info


# ---------------------------------------------------------------------------
# Isoster runner
# ---------------------------------------------------------------------------

def warmup_jit(image: np.ndarray):
    """Warm up Numba JIT with a small throwaway fit."""
    ny, nx = image.shape
    tiny_config = {
        "x0": nx / 2.0,
        "y0": ny / 2.0,
        "eps": 0.2,
        "pa": 0.0,
        "sma0": 10.0,
        "minsma": 5.0,
        "maxsma": 30.0,
        "maxit": 5,
        "compute_errors": False,
        "compute_deviations": False,
    }
    try:
        isoster.fit_image(image, None, tiny_config)
    except Exception:
        pass


def run_isoster_benchmark(image: np.ndarray, params: dict) -> dict:
    """Run isoster fitting with timing.

    Parameters
    ----------
    image : np.ndarray
        Galaxy image.
    params : dict
        Galaxy info dict with center, eps, pa, sma0, maxsma.

    Returns
    -------
    dict
        Profile dict with sma, intens, eps, pa, runtime_s, n_isophotes,
        stop_codes.
    """
    x0, y0 = params["center"]
    config = {
        "x0": x0,
        "y0": y0,
        "eps": params["eps"],
        "pa": params["pa"],
        "sma0": params["sma0"],
        "minsma": 1.0,
        "maxsma": params["maxsma"],
        "astep": 0.1,
        "linear_growth": False,
        "compute_errors": True,
        "compute_deviations": False,
        "nclip": 2,
        "sclip": 3.0,
        "convergence_scaling": "sector_area",
        "geometry_damping": 0.7,
    }

    t0 = time.perf_counter()
    result = isoster.fit_image(image, None, config)
    elapsed = time.perf_counter() - t0

    isophotes = result["isophotes"]
    stop_codes = {}
    for iso in isophotes:
        sc = iso.get("stop_code", 0)
        stop_codes[sc] = stop_codes.get(sc, 0) + 1

    return {
        "sma": np.array([iso["sma"] for iso in isophotes]),
        "intens": np.array([iso["intens"] for iso in isophotes]),
        "eps": np.array([iso["eps"] for iso in isophotes]),
        "pa": np.array([iso["pa"] for iso in isophotes]),
        "intens_err": np.array([iso.get("intens_err", np.nan) for iso in isophotes]),
        "stop_code": np.array([iso.get("stop_code", 0) for iso in isophotes], dtype=int),
        "runtime_s": elapsed,
        "n_isophotes": len(isophotes),
        "stop_codes": stop_codes,
        "isophotes": isophotes,  # raw list needed for build_isoster_model
    }


# ---------------------------------------------------------------------------
# AutoProf runner
# ---------------------------------------------------------------------------

def run_autoprof_benchmark(
    galaxy_info: dict, params: dict, output_dir: Path,
    fits_path_override: Path | None = None,
) -> dict | None:
    """Run AutoProf fitting with timing.

    Parameters
    ----------
    galaxy_info : dict
        Galaxy registry entry with fits_path, pixel_scale, zeropoint.
    params : dict
        Filled-in parameter dict (center, eps, pa, background, etc.).
    output_dir : Path
        Output directory for AutoProf files.
    fits_path_override : Path, optional
        If set, use this FITS file instead of galaxy_info["fits_path"].
        Used for multi-band images that need a 2D extraction.

    Returns
    -------
    dict or None
        Profile dict (same format as isoster), or None on failure.
    """
    ap_output = output_dir / f"autoprof_{galaxy_info['name']}"
    image_path = fits_path_override or galaxy_info["fits_path"]
    return run_autoprof_fit(
        image_path=image_path,
        output_dir=ap_output,
        galaxy_name=galaxy_info["name"],
        pixel_scale=params["pixel_scale"],
        zeropoint=params["zeropoint"],
        center=params["center"],
        eps=params["eps"],
        pa_rad_math=params["pa"],
        background=params["background"],
        background_noise=params["background_noise"],
    )


# ---------------------------------------------------------------------------
# Profile comparison
# ---------------------------------------------------------------------------

def interpolate_to_common_grid(
    prof_a: dict, prof_b: dict, n_points: int = 200
) -> tuple[np.ndarray, dict, dict]:
    """Interpolate two profiles onto a common log-spaced SMA grid.

    Parameters
    ----------
    prof_a, prof_b : dict
        Profile dicts with 'sma', 'intens', 'eps', 'pa'.
    n_points : int
        Number of points in the common grid.

    Returns
    -------
    sma_common : ndarray
        Common SMA grid.
    a_interp : dict
        Interpolated profile A (intens, eps, pa).
    b_interp : dict
        Interpolated profile B (intens, eps, pa).
    """
    sma_min = max(prof_a["sma"].min(), prof_b["sma"].min())
    sma_max = min(prof_a["sma"].max(), prof_b["sma"].max())

    if sma_min >= sma_max:
        return np.array([]), {}, {}

    sma_common = np.geomspace(sma_min, sma_max, n_points)

    def _interp(prof, sma_grid):
        return {
            "intens": np.interp(sma_grid, prof["sma"], prof["intens"]),
            "eps": np.interp(sma_grid, prof["sma"], prof["eps"]),
            "pa": np.interp(sma_grid, prof["sma"], prof["pa"]),
        }

    return sma_common, _interp(prof_a, sma_common), _interp(prof_b, sma_common)


def compute_deviations(isoster_prof: dict, autoprof_prof: dict) -> dict:
    """Compute deviations between isoster and AutoProf profiles.

    Returns
    -------
    dict
        Deviation statistics: med/max relative intensity, absolute eps,
        absolute PA (degrees), sma_range, n_points.
    """
    sma, iso_interp, ap_interp = interpolate_to_common_grid(
        isoster_prof, autoprof_prof
    )

    if len(sma) == 0:
        return {
            "med_rel_intens": np.nan,
            "max_rel_intens": np.nan,
            "med_abs_eps": np.nan,
            "max_abs_eps": np.nan,
            "med_abs_pa_deg": np.nan,
            "max_abs_pa_deg": np.nan,
            "sma_range": [np.nan, np.nan],
            "n_points": 0,
        }

    # Relative intensity deviation
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_intens = np.abs(iso_interp["intens"] - ap_interp["intens"]) / np.abs(ap_interp["intens"])
        rel_intens = np.nan_to_num(rel_intens, nan=0.0, posinf=0.0)

    # Absolute ellipticity deviation
    abs_eps = np.abs(iso_interp["eps"] - ap_interp["eps"])

    # Absolute PA deviation (handle wrapping at pi)
    pa_diff = np.abs(iso_interp["pa"] - ap_interp["pa"])
    pa_diff = np.minimum(pa_diff, np.pi - pa_diff)
    abs_pa_deg = np.degrees(pa_diff)

    return {
        "med_rel_intens": float(np.nanmedian(rel_intens)),
        "max_rel_intens": float(np.nanmax(rel_intens)),
        "med_abs_eps": float(np.nanmedian(abs_eps)),
        "max_abs_eps": float(np.nanmax(abs_eps)),
        "med_abs_pa_deg": float(np.nanmedian(abs_pa_deg)),
        "max_abs_pa_deg": float(np.nanmax(abs_pa_deg)),
        "sma_range": [float(sma.min()), float(sma.max())],
        "n_points": len(sma),
    }


# ---------------------------------------------------------------------------
# 2D model building
# ---------------------------------------------------------------------------

def compute_fractional_residual(image: np.ndarray, model: np.ndarray | None) -> np.ndarray:
    """Compute (model - data) / data * 100 (%), NaN where invalid or model=0."""
    if model is None:
        return None
    residual = np.full(image.shape, np.nan, dtype=float)
    valid = np.isfinite(image) & (np.abs(image) > 0.0) & np.isfinite(model) & (model > 0.0)
    residual[valid] = (model[valid] - image[valid]) / image[valid] * 100.0
    return residual


def autoprof_profile_to_isophote_list(
    autoprof_profile: dict, center: tuple[float, float]
) -> list[dict]:
    """Convert parsed AutoProf profile to an isoster-style isophote list.

    AutoProf fits a single global center; here we assign a fixed center to all
    isophotes so that build_isoster_model can apply identical inverse-mapping
    logic to both methods for a fair "common reconstruction" comparison.

    Parameters
    ----------
    autoprof_profile : dict
        Parsed AutoProf profile with sma, intens, eps, pa arrays.
    center : tuple of float
        (x0, y0) center coordinates in pixels.

    Returns
    -------
    list of dict
        Isophote dicts compatible with build_isoster_model.
    """
    x0, y0 = center
    return [
        {
            "sma": float(autoprof_profile["sma"][i]),
            "intens": float(autoprof_profile["intens"][i]),
            "eps": float(autoprof_profile["eps"][i]),
            "pa": float(autoprof_profile["pa"][i]),
            "x0": x0,
            "y0": y0,
        }
        for i in range(len(autoprof_profile["sma"]))
    ]


def load_autoprof_native_model(
    model_fits_path: str, image_shape: tuple[int, int]
) -> np.ndarray | None:
    """Load native AutoProf 2D model FITS, resized to image_shape if needed.

    AutoProf's EllipseModel saves a FITS with the same pixel dimensions as
    the input image, so resizing should rarely be necessary.

    Parameters
    ----------
    model_fits_path : str
        Path to ``*_genmodel.fits`` produced by AutoProf's EllipseModel step.
    image_shape : tuple of int
        Target (height, width) of the returned array.

    Returns
    -------
    np.ndarray or None
        Float64 model image, or None if the file cannot be loaded.
    """
    try:
        with afits.open(str(model_fits_path)) as hdul:
            for hdu in hdul:
                if hdu.data is not None and hdu.data.ndim == 2:
                    model = hdu.data.astype(float)
                    if model.shape == image_shape:
                        return model
                    # Pad zeros to match expected shape (should be rare)
                    padded = np.zeros(image_shape, dtype=float)
                    h0, w0 = min(model.shape[0], image_shape[0]), min(model.shape[1], image_shape[1])
                    padded[:h0, :w0] = model[:h0, :w0]
                    return padded
    except Exception as exc:
        print(f"  Warning: could not load AutoProf model FITS: {exc}")
    return None


def _build_all_models(
    image: np.ndarray,
    params: dict,
    isoster_result: dict,
    autoprof_result: dict | None,
) -> tuple[dict, dict]:
    """Build all 4 2D models and compute their fractional residuals.

    Models built:
    - ``iso_native``   : isoster model with A3/A4 harmonics
    - ``iso_no_harm``  : isoster model without harmonics (common-recon baseline)
    - ``ap_native``    : AutoProf EllipseModel FITS (proximity-shell, mag→flux)
    - ``ap_common``    : AutoProf profile reconstructed via isoster inverse-mapping

    The ``ap_native`` vs ``iso_native`` comparison reflects end-to-end method
    differences (fitting algorithm + reconstruction algorithm).  The
    ``ap_common`` vs ``iso_no_harm`` comparison isolates 1D profile quality by
    using identical reconstruction logic for both methods.

    Parameters
    ----------
    image : np.ndarray
        Original galaxy image in counts/pixel.
    params : dict
        Galaxy parameter dict with 'center'.
    isoster_result : dict
        Output of run_isoster_benchmark with 'isophotes' key.
    autoprof_result : dict or None
        Output of run_autoprof_benchmark with optional 'model_fits_path'.

    Returns
    -------
    models : dict
        Keys: iso_native, iso_no_harm, ap_native, ap_common (each ndarray or None).
    residuals : dict
        Keys same as models, values are fractional residual arrays (% or None).
    """
    isophotes = isoster_result.get("isophotes", [])
    iso_model_native = build_isoster_model(image.shape, isophotes, use_harmonics=True)
    iso_model_no_harm = build_isoster_model(image.shape, isophotes, use_harmonics=False)

    ap_model_native = None
    ap_model_common = None
    if autoprof_result is not None:
        # Native AutoProf model from EllipseModel step
        model_fits_path = autoprof_result.get("model_fits_path")
        if model_fits_path:
            ap_model_native = load_autoprof_native_model(model_fits_path, image.shape)
            if ap_model_native is None:
                print("  Warning: AutoProf native model could not be loaded.")

        # Common reconstruction: same inverse-mapping as isoster, no harmonics
        ap_isos = autoprof_profile_to_isophote_list(autoprof_result, params["center"])
        ap_model_common = build_isoster_model(image.shape, ap_isos, use_harmonics=False)

    models = {
        "iso_native": iso_model_native,
        "iso_no_harm": iso_model_no_harm,
        "ap_native": ap_model_native,
        "ap_common": ap_model_common,
    }
    residuals = {key: compute_fractional_residual(image, m) for key, m in models.items()}
    return models, residuals


# ---------------------------------------------------------------------------
# Per-galaxy runner
# ---------------------------------------------------------------------------

def run_single_galaxy(
    galaxy_info: dict, output_dir: Path, skip_autoprof: bool = False
) -> dict | None:
    """Run benchmark for a single galaxy.

    Parameters
    ----------
    galaxy_info : dict
        Galaxy registry entry.
    output_dir : Path
        Output directory.
    skip_autoprof : bool
        If True, skip AutoProf fit (only run isoster).

    Returns
    -------
    dict or None
        Results dict with isoster/autoprof profiles and deviations.
    """
    name = galaxy_info["name"]
    fits_path = galaxy_info["fits_path"]

    if not fits_path.exists():
        print(f"  FITS file not found: {fits_path}")
        return None

    print(f"\n{'='*60}")
    print(f"Galaxy: {name}")
    print(f"{'='*60}")

    # Load image — find the first 2D data extension
    from astropy.io import fits as afits
    autoprof_fits_path = fits_path  # default: use original file for AutoProf
    with afits.open(str(fits_path)) as hdu_list:
        image = None
        for hdu in hdu_list:
            if hdu.data is not None and hdu.data.ndim == 2:
                image = hdu.data.astype(float)
                break
        if image is None:
            # Multi-band cube: extract first band and write temp 2D FITS
            for hdu in hdu_list:
                if hdu.data is not None and hdu.data.ndim == 3:
                    print(f"  Multi-band FITS ({hdu.data.shape}), using band 0")
                    image = hdu.data[0].astype(float)
                    # Write a temporary 2D FITS for AutoProf
                    temp_fits = output_dir / f"{name}_band0.fits"
                    afits.PrimaryHDU(image, header=hdu.header).writeto(
                        str(temp_fits), overwrite=True
                    )
                    autoprof_fits_path = temp_fits
                    break
        if image is None:
            print(f"  No usable 2D image data found in {fits_path}")
            return None
    print(f"  Image shape: {image.shape}")

    # Estimate missing parameters
    params = estimate_initial_params(image, galaxy_info)

    # JIT warmup before timing
    print("  Warming up JIT...")
    warmup_jit(image)

    # Run isoster
    print("  Running isoster...")
    isoster_result = run_isoster_benchmark(image, params)
    print(f"  isoster: {isoster_result['n_isophotes']} isophotes in "
          f"{isoster_result['runtime_s']:.3f}s")
    print(f"  Stop codes: {isoster_result['stop_codes']}")

    # Run AutoProf
    autoprof_result = None
    deviations = None
    if not skip_autoprof:
        if check_autoprof_available():
            print("  Running AutoProf...")
            autoprof_result = run_autoprof_benchmark(
                galaxy_info, params, output_dir,
                fits_path_override=autoprof_fits_path if autoprof_fits_path != fits_path else None,
            )
            if autoprof_result is not None:
                print(f"  AutoProf: {autoprof_result['n_isophotes']} isophotes in "
                      f"{autoprof_result['runtime_s']:.3f}s")
                deviations = compute_deviations(isoster_result, autoprof_result)
                print(f"  Deviations: med_rel_I={deviations['med_rel_intens']:.4f}, "
                      f"med_eps={deviations['med_abs_eps']:.4f}, "
                      f"med_PA={deviations['med_abs_pa_deg']:.2f} deg")
            else:
                print("  AutoProf fit failed.")
        else:
            print("  AutoProf not installed — skipping.")

    result = {
        "galaxy": name,
        "params": {
            "center": list(params["center"]),
            "eps": params["eps"],
            "pa": params["pa"],
            "background": params["background"],
            "background_noise": params["background_noise"],
            "pixel_scale": params["pixel_scale"],
            "zeropoint": params["zeropoint"],
            "sma0": params["sma0"],
            "maxsma": params["maxsma"],
        },
        "isoster": {
            "runtime_s": isoster_result["runtime_s"],
            "n_isophotes": isoster_result["n_isophotes"],
            "stop_codes": {str(k): v for k, v in isoster_result["stop_codes"].items()},
        },
    }

    if autoprof_result is not None:
        result["autoprof"] = {
            "runtime_s": autoprof_result["runtime_s"],
            "n_isophotes": autoprof_result["n_isophotes"],
        }
        result["deviations"] = deviations

    # Build 2D models for residual comparison
    print("  Building 2D models for residual comparison...")
    models, residuals = _build_all_models(image, params, isoster_result, autoprof_result)

    # Store profiles and models for plotting (not serialized to JSON)
    result["_profiles"] = {
        "isoster": isoster_result,
        "autoprof": autoprof_result,
        "image": image,
        "models": models,
        "residuals": residuals,
    }

    return result


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_all_benchmarks(
    output_dir: Path | None = None,
    quick: bool = False,
    skip_autoprof: bool = False,
    galaxies: list[str] | None = None,
) -> dict:
    """Run benchmarks on all (or selected) galaxies.

    Parameters
    ----------
    output_dir : Path, optional
        Explicit output directory.
    quick : bool
        If True, run only QUICK_GALAXIES.
    skip_autoprof : bool
        If True, skip AutoProf fits.
    galaxies : list of str, optional
        Galaxy names to run (overrides quick).

    Returns
    -------
    dict
        Full results payload with environment, summary, and per-galaxy results.
    """
    output_dir = resolve_benchmark_output_directory(output_dir)

    # Filter galaxy list
    registry = GALAXY_REGISTRY
    if galaxies:
        names_set = set(galaxies)
        registry = [g for g in registry if g["name"] in names_set]
    elif quick:
        names_set = set(QUICK_GALAXIES)
        registry = [g for g in registry if g["name"] in names_set]

    print(f"\nBenchmark: isoster vs AutoProf")
    print(f"Galaxies: {[g['name'] for g in registry]}")
    print(f"AutoProf available: {check_autoprof_available()}")
    print(f"Output: {output_dir}")

    results = []
    for galaxy_info in registry:
        result = run_single_galaxy(galaxy_info, output_dir, skip_autoprof=skip_autoprof)
        if result is not None:
            results.append(result)

    # Build summary
    isoster_times = [r["isoster"]["runtime_s"] for r in results]
    autoprof_times = [r["autoprof"]["runtime_s"] for r in results if "autoprof" in r]

    summary = {
        "n_galaxies": len(results),
        "galaxies_run": [r["galaxy"] for r in results],
        "isoster_mean_runtime_s": float(np.mean(isoster_times)) if isoster_times else None,
        "autoprof_mean_runtime_s": float(np.mean(autoprof_times)) if autoprof_times else None,
        "autoprof_available": check_autoprof_available(),
        "skip_autoprof": skip_autoprof,
    }

    # Serialize (strip _profiles which contain numpy arrays)
    results_json = []
    for r in results:
        r_clean = {k: v for k, v in r.items() if not k.startswith("_")}
        results_json.append(r_clean)

    output_payload = {
        "environment": collect_environment_metadata(project_root=PROJECT_ROOT),
        "summary": summary,
        "galaxies": results_json,
    }

    # Save JSON
    json_path = output_dir / "summary.json"
    write_json(json_path, output_payload)

    # Save CSV
    csv_path = output_dir / "summary.csv"
    _write_summary_csv(results_json, csv_path)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Galaxies:           {summary['n_galaxies']}")
    print(f"isoster mean time:  {summary['isoster_mean_runtime_s']:.3f}s" if summary["isoster_mean_runtime_s"] else "")
    if summary["autoprof_mean_runtime_s"] is not None:
        print(f"AutoProf mean time: {summary['autoprof_mean_runtime_s']:.3f}s")
    print(f"\nJSON: {json_path}")
    print(f"CSV:  {csv_path}")

    # Attach profiles for plotting
    output_payload["_results_with_profiles"] = results

    return output_payload


def _write_summary_csv(results: list[dict], csv_path: Path):
    """Write a summary CSV with one row per galaxy."""
    field_names = [
        "galaxy",
        "isoster_runtime_s",
        "isoster_n_isophotes",
        "autoprof_runtime_s",
        "autoprof_n_isophotes",
        "med_rel_intens",
        "max_rel_intens",
        "med_abs_eps",
        "max_abs_eps",
        "med_abs_pa_deg",
        "max_abs_pa_deg",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=field_names)
        writer.writeheader()
        for r in results:
            row = {
                "galaxy": r["galaxy"],
                "isoster_runtime_s": r["isoster"]["runtime_s"],
                "isoster_n_isophotes": r["isoster"]["n_isophotes"],
            }
            if "autoprof" in r:
                row["autoprof_runtime_s"] = r["autoprof"]["runtime_s"]
                row["autoprof_n_isophotes"] = r["autoprof"]["n_isophotes"]
            if "deviations" in r and r["deviations"] is not None:
                dev = r["deviations"]
                row["med_rel_intens"] = dev["med_rel_intens"]
                row["max_rel_intens"] = dev["max_rel_intens"]
                row["med_abs_eps"] = dev["med_abs_eps"]
                row["max_abs_eps"] = dev["max_abs_eps"]
                row["med_abs_pa_deg"] = dev["med_abs_pa_deg"]
                row["max_abs_pa_deg"] = dev["max_abs_pa_deg"]
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def generate_comparison_plots(results_payload: dict, output_dir: Path | None = None):
    """Generate per-galaxy comparison plots and runtime bar chart.

    Parameters
    ----------
    results_payload : dict
        Output from run_all_benchmarks (must contain _results_with_profiles).
    output_dir : Path, optional
        Output directory.
    """
    output_dir = resolve_benchmark_output_directory(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    results = results_payload.get("_results_with_profiles", [])

    for result in results:
        generate_galaxy_comparison_figure(result, plots_dir)

    # Runtime bar chart
    _plot_runtime_comparison(results, plots_dir)


def _make_arcsinh_display(image: np.ndarray):
    """Stretch image for display using arcsinh normalization."""
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return image, 0, 1
    p5 = float(np.nanpercentile(finite, 0.5))
    p995 = float(np.nanpercentile(finite, 99.5))
    scale = max(p995 - p5, 1e-10)
    stretched = np.arcsinh((image - p5) / scale * 3.0)
    vmin = float(np.arcsinh(0.0))
    vmax = float(np.arcsinh(3.0))
    return stretched, vmin, vmax


def _common_residual_colorlimit(residuals: list[np.ndarray | None], pct: float = 99.0) -> float:
    """Return a symmetric colormap limit from the union of all residual maps."""
    all_vals = []
    for r in residuals:
        if r is not None:
            finite = r[np.isfinite(r)]
            if finite.size > 0:
                all_vals.append(np.abs(finite))
    if not all_vals:
        return 20.0
    combined = np.concatenate(all_vals)
    clim = float(np.nanpercentile(combined, pct))
    return float(np.clip(clim, 2.0, 50.0))


def _draw_isophote_ellipses(ax, sma_arr, x0_arr, y0_arr, eps_arr, pa_arr, step, color, lw=0.9, alpha=0.8):
    """Overlay every `step`-th isophote as a matplotlib Ellipse patch."""
    for i in range(0, len(sma_arr), step):
        sma = float(sma_arr[i])
        if not np.isfinite(sma) or sma < 2.0:
            continue
        x0, y0 = float(x0_arr[i]), float(y0_arr[i])
        eps = float(np.clip(eps_arr[i], 0.0, 0.99))
        pa_deg = float(np.degrees(pa_arr[i]))
        patch = MplEllipse(
            (x0, y0), 2.0 * sma, 2.0 * sma * (1.0 - eps),
            angle=pa_deg, fill=False, linewidth=lw, edgecolor=color, alpha=alpha,
        )
        ax.add_patch(patch)


def _zone_stats(residual: np.ndarray | None, ell_radius: np.ndarray, maxsma: float) -> dict:
    """Compute median |residual %| in inner/mid/outer radial zones."""
    if residual is None or maxsma <= 0:
        return {"inner": np.nan, "mid": np.nan, "outer": np.nan}
    zones = {
        "inner": ell_radius < 0.25 * maxsma,
        "mid": (ell_radius >= 0.25 * maxsma) & (ell_radius < 0.75 * maxsma),
        "outer": ell_radius >= 0.75 * maxsma,
    }
    stats = {}
    for zone_name, mask in zones.items():
        valid = mask & np.isfinite(residual)
        stats[zone_name] = float(np.nanmedian(np.abs(residual[valid]))) if valid.any() else np.nan
    return stats


def _rough_ell_radius_map(image_shape, isoster_result):
    """Compute an approximate elliptical radius map using the outermost isophote geometry."""
    isophotes = isoster_result.get("isophotes", [])
    if not isophotes:
        h, w = image_shape
        return np.sqrt((np.mgrid[:h, :w][0] - h / 2) ** 2 + (np.mgrid[:h, :w][1] - w / 2) ** 2)
    outer = max(isophotes, key=lambda iso: iso["sma"])
    x0, y0 = outer["x0"], outer["y0"]
    eps = float(np.clip(outer["eps"], 0, 0.99))
    pa = float(outer["pa"])
    h, w = image_shape
    y_grid, x_grid = np.mgrid[:h, :w].astype(float)
    dx = x_grid - x0
    dy = y_grid - y0
    cos_pa, sin_pa = np.cos(pa), np.sin(pa)
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    return np.sqrt(x_rot ** 2 + (y_rot / (1.0 - eps)) ** 2)


def _fold_pa_to_reference(pa_rad: np.ndarray, n_inner_frac: float = 0.33) -> np.ndarray:
    """Fold PA values to within ±90° of the inner-profile median.

    PA has 180° periodicity (an ellipse at angle θ is identical to θ+180°).
    This function folds each value independently to be within [-π/2, +π/2)
    of the inner-profile reference, making the plot robust to noisy outer
    isophotes with random or scattered PA values.

    Parameters
    ----------
    pa_rad : np.ndarray
        PA values in radians (math convention, CCW from +x).
    n_inner_frac : float
        Fraction of isophotes to use as the reference (innermost).
    """
    if len(pa_rad) == 0:
        return pa_rad
    n_inner = max(1, int(len(pa_rad) * n_inner_frac))
    ref = float(np.nanmedian(pa_rad[:n_inner]))
    delta = ((pa_rad - ref) + np.pi / 2) % np.pi - np.pi / 2
    return ref + delta


def generate_galaxy_comparison_figure(result: dict, plots_dir: Path, dpi: int = 150):
    """Generate a rich multi-panel comparison figure for one galaxy.

    Layout
    ------
    Left column (2D panels, 5 rows):
      Row 0  Data image with isophote overlays from both methods
      Row 1  isoster residual — native model with A3/A4 harmonics
      Row 2  AutoProf residual — native EllipseModel (proximity-shell)
      Row 3  isoster residual — no harmonics (common-recon baseline)
      Row 4  AutoProf residual — common inverse-mapping reconstruction

    Right column (1D profiles, 5 rows):
      Row 0 (tall)  SB profile on SMA^0.25 x-axis
      Row 1         Relative intensity deviation (%)
      Row 2         Ellipticity
      Row 3         PA (degrees, unwrapped)
      Row 4         Per-zone residual statistics

    The two reconstruction strategies isolate different aspects:
    - Native (rows 1-2): end-to-end method comparison (fitting + reconstruction)
    - Common (rows 3-4): isolates 1D fitting quality by using identical
      inverse-mapping reconstruction for both methods
    """
    name = result["galaxy"]
    profiles = result.get("_profiles", {})
    iso_prof = profiles.get("isoster")
    ap_prof = profiles.get("autoprof")
    image = profiles.get("image")
    models = profiles.get("models", {})
    residuals = profiles.get("residuals", {})

    if iso_prof is None or image is None:
        return

    has_autoprof = ap_prof is not None
    has_native_model = models.get("ap_native") is not None

    # -----------------------------------------------------------------------
    # Shared display data
    # -----------------------------------------------------------------------
    img_stretched, img_vmin, img_vmax = _make_arcsinh_display(image)

    resid_list = [residuals.get(k) for k in ("iso_native", "ap_native", "iso_no_harm", "ap_common")]
    clim = _common_residual_colorlimit(resid_list)

    isoster_sma = iso_prof["sma"]
    isoster_x0 = np.array([iso["x0"] for iso in iso_prof["isophotes"]])
    isoster_y0 = np.array([iso["y0"] for iso in iso_prof["isophotes"]])
    isoster_eps = iso_prof["eps"]
    isoster_pa = iso_prof["pa"]
    isoster_stop = iso_prof.get("stop_code", np.zeros(len(isoster_sma), dtype=int))

    overlay_step = max(1, len(isoster_sma) // 15)

    ell_radius = _rough_ell_radius_map(image.shape, iso_prof)
    maxsma = float(isoster_sma[-1]) if len(isoster_sma) > 0 else 1.0

    # Per-zone statistics for all 4 residual maps
    zone_data = {
        key: _zone_stats(residuals.get(key), ell_radius, maxsma)
        for key in ("iso_native", "ap_native", "iso_no_harm", "ap_common")
    }

    # -----------------------------------------------------------------------
    # Figure and gridspec
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 16), dpi=dpi)
    outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.05, 1.95], wspace=0.25)
    left = gridspec.GridSpecFromSubplotSpec(
        5, 2, subplot_spec=outer[0],
        width_ratios=[1.0, 0.04],
        height_ratios=[1.5, 1.0, 1.0, 1.0, 1.0],
        hspace=0.07, wspace=-0.15,
    )
    right = gridspec.GridSpecFromSubplotSpec(
        5, 1, subplot_spec=outer[1],
        height_ratios=[2.0, 1.0, 1.0, 1.0, 0.8],
        hspace=0.0,
    )

    iso_t = result.get("isoster", {}).get("runtime_s", float("nan"))
    ap_t = result.get("autoprof", {}).get("runtime_s", float("nan"))
    speedup_str = f"  |  speedup {ap_t / iso_t:.0f}x" if (np.isfinite(iso_t) and np.isfinite(ap_t) and iso_t > 0) else ""
    fig.suptitle(
        f"{name}  |  isoster vs AutoProf{speedup_str}  |  "
        f"isoster {iso_t:.2f}s, AutoProf {ap_t:.2f}s",
        fontsize=13, y=0.993,
    )

    # -----------------------------------------------------------------------
    # Left panel 0: Data image with isophote overlays
    # -----------------------------------------------------------------------
    ax_data = fig.add_subplot(left[0, 0])
    ax_data.imshow(img_stretched, origin="lower", cmap="viridis",
                   vmin=img_vmin, vmax=img_vmax, interpolation="none")
    _draw_isophote_ellipses(
        ax_data, isoster_sma, isoster_x0, isoster_y0, isoster_eps, isoster_pa,
        step=overlay_step, color="cyan", lw=0.9, alpha=0.75,
    )
    if has_autoprof:
        ap_sma = ap_prof["sma"]
        ap_eps = ap_prof["eps"]
        ap_pa = ap_prof["pa"]
        cx, cy = result.get("params", {}).get("center", (image.shape[1] / 2, image.shape[0] / 2))
        x0_arr = np.full_like(ap_sma, cx)
        y0_arr = np.full_like(ap_sma, cy)
        _draw_isophote_ellipses(
            ax_data, ap_sma, x0_arr, y0_arr, ap_eps, ap_pa,
            step=overlay_step, color="orangered", lw=0.8, alpha=0.65,
        )
    ax_data.set_title("Data  (cyan=isoster, orange=AutoProf)", fontsize=9)
    ax_data.axis("off")

    cbar_ax0 = fig.add_subplot(left[0, 1])
    cbar_ax0.axis("off")

    # -----------------------------------------------------------------------
    # Left panels 1-4: Residual maps
    # -----------------------------------------------------------------------
    residual_panel_specs = [
        ("iso_native",  "isoster native (with A3/A4)",   "left"),
        ("ap_native",   "AutoProf native (EllipseModel)", "left"),
        ("iso_no_harm", "isoster no-harm (common recon)", "right"),
        ("ap_common",   "AutoProf common recon",          "right"),
    ]
    _STOP_COLORS = {0: "green", 1: "orange", 2: "red", 3: "purple", -1: "black"}

    for row_idx, (key, label, recon_side) in enumerate(residual_panel_specs):
        ax_r = fig.add_subplot(left[row_idx + 1, 0])
        cbar_r = fig.add_subplot(left[row_idx + 1, 1])
        resid = residuals.get(key)
        if resid is not None:
            im = ax_r.imshow(
                resid, origin="lower", cmap="coolwarm",
                vmin=-clim, vmax=clim, interpolation="nearest",
            )
            fig.colorbar(im, cax=cbar_r, label="%")
        else:
            ax_r.text(0.5, 0.5, "N/A", ha="center", va="center",
                      transform=ax_r.transAxes, fontsize=11, color="gray")
            cbar_r.axis("off")
        # Zone stats annotation
        zs = zone_data.get(key, {})
        stats_text = (
            f"inner {zs.get('inner', float('nan')):.1f}%  "
            f"mid {zs.get('mid', float('nan')):.1f}%  "
            f"outer {zs.get('outer', float('nan')):.1f}%"
        )
        border_color = "steelblue" if "iso" in key else "coral"
        for spine in ax_r.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2.0)
        recon_label = "[native]" if recon_side == "left" else "[common recon]"
        ax_r.set_title(f"{label}  {recon_label}  |  {stats_text}", fontsize=8.0)
        ax_r.axis("off")

    # -----------------------------------------------------------------------
    # Right panel 0: SB profile
    # -----------------------------------------------------------------------
    ax_sb = fig.add_subplot(right[0])
    ax_sb.set_yscale("log")

    x_iso = np.power(isoster_sma, 0.25)
    for sc in np.unique(isoster_stop):
        mask = isoster_stop == sc
        color = _STOP_COLORS.get(int(sc), "gray")
        yerr = iso_prof.get("intens_err")
        err = yerr[mask] if yerr is not None else None
        ax_sb.errorbar(
            x_iso[mask], iso_prof["intens"][mask],
            yerr=err, fmt="o", ms=3, color=color, alpha=0.8, linewidth=0,
            elinewidth=0.7, label=f"isoster sc={sc}" if mask.any() else None,
        )
    if has_autoprof:
        x_ap = np.power(ap_prof["sma"], 0.25)
        ax_sb.plot(x_ap, ap_prof["intens"], "--", color="coral", linewidth=1.2,
                   label="AutoProf", zorder=5)

    ax_sb.set_ylabel("Intensity (counts/px)", fontsize=9)
    ax_sb.legend(loc="upper right", fontsize=7, ncol=2)
    ax_sb.grid(alpha=0.25)
    ax_sb.tick_params(labelbottom=False)

    # -----------------------------------------------------------------------
    # Right panel 1: Relative intensity deviation (%)
    # -----------------------------------------------------------------------
    ax_dev = fig.add_subplot(right[1], sharex=ax_sb)
    if has_autoprof:
        sma_com, iso_interp, ap_interp = interpolate_to_common_grid(iso_prof, ap_prof)
        if len(sma_com) > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_dev = (iso_interp["intens"] - ap_interp["intens"]) / np.abs(ap_interp["intens"]) * 100.0
            x_com = np.power(sma_com, 0.25)
            ax_dev.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax_dev.plot(x_com, rel_dev, color="steelblue", linewidth=1.2)
            rms = float(np.sqrt(np.nanmean(rel_dev ** 2)))
            ax_dev.set_title(f"RMS {rms:.2f}%", fontsize=8, loc="right")
    ax_dev.set_ylabel("ΔI/I (%)", fontsize=9)
    ax_dev.grid(alpha=0.25)
    ax_dev.tick_params(labelbottom=False)

    # -----------------------------------------------------------------------
    # Right panel 2: Ellipticity
    # -----------------------------------------------------------------------
    ax_eps = fig.add_subplot(right[2], sharex=ax_sb)
    ax_eps.scatter(x_iso, isoster_eps, s=5, color="steelblue", alpha=0.8, label="isoster")
    if has_autoprof:
        ax_eps.plot(x_ap, ap_prof["eps"], "--", color="coral", linewidth=1.2, label="AutoProf")
    ax_eps.set_ylabel("Ellipticity", fontsize=9)
    ax_eps.set_ylim(-0.05, 1.0)
    ax_eps.grid(alpha=0.25)
    ax_eps.tick_params(labelbottom=False)

    # -----------------------------------------------------------------------
    # Right panel 3: PA
    # -----------------------------------------------------------------------
    ax_pa = fig.add_subplot(right[3], sharex=ax_sb)
    pa_iso_deg = np.degrees(_fold_pa_to_reference(isoster_pa))
    ax_pa.scatter(x_iso, pa_iso_deg, s=5, color="steelblue", alpha=0.8, label="isoster")
    if has_autoprof:
        pa_ap_deg = np.degrees(_fold_pa_to_reference(ap_prof["pa"]))
        ax_pa.plot(x_ap, pa_ap_deg, "--", color="coral", linewidth=1.2, label="AutoProf")
    # Clamp y-range to ±60° around reference PA to keep plot readable
    pa_ref_deg = np.degrees(float(np.nanmedian(isoster_pa[:max(1, len(isoster_pa) // 3)])))
    ax_pa.set_ylim(pa_ref_deg - 60.0, pa_ref_deg + 60.0)
    ax_pa.set_ylabel("PA (deg)", fontsize=9)
    ax_pa.grid(alpha=0.25)
    ax_pa.set_xlabel("SMA$^{0.25}$ (pixels$^{0.25}$)", fontsize=9)

    # -----------------------------------------------------------------------
    # Right panel 4: Per-zone statistics text summary
    # -----------------------------------------------------------------------
    ax_txt = fig.add_subplot(right[4])
    ax_txt.axis("off")
    lines = ["Per-zone median |residual %|:  inner (<25% SMA) / mid / outer (>75% SMA)"]
    panel_labels = {
        "iso_native": "isoster native  ",
        "ap_native": "AutoProf native ",
        "iso_no_harm": "isoster no-harm ",
        "ap_common": "AutoProf common ",
    }
    for key, lbl in panel_labels.items():
        zs = zone_data.get(key, {})
        lines.append(
            f"  {lbl}: {zs.get('inner', float('nan')):.2f}% / "
            f"{zs.get('mid', float('nan')):.2f}% / "
            f"{zs.get('outer', float('nan')):.2f}%"
        )
    ax_txt.text(0.01, 0.98, "\n".join(lines), va="top", ha="left",
                transform=ax_txt.transAxes, fontsize=8, family="monospace")

    fig.savefig(plots_dir / f"comparison_{name}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {plots_dir / f'comparison_{name}.png'}")


def _plot_runtime_comparison(results: list[dict], plots_dir: Path):
    """Generate runtime bar chart comparing both tools across galaxies."""
    names = [r["galaxy"] for r in results]
    iso_times = [r["isoster"]["runtime_s"] for r in results]
    ap_times = [r.get("autoprof", {}).get("runtime_s", 0) for r in results]
    has_any_autoprof = any(t > 0 for t in ap_times)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    width = 0.35

    ax.bar(x - width / 2, iso_times, width, label="isoster", color="steelblue")
    if has_any_autoprof:
        ax.bar(x + width / 2, ap_times, width, label="AutoProf", color="coral")
        for i, (it, at) in enumerate(zip(iso_times, ap_times)):
            if at > 0 and it > 0:
                ax.text(i, max(it, at) * 1.03, f"{at/it:.0f}x", ha="center", fontsize=9)

    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Runtime: isoster vs AutoProf")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = plots_dir / "runtime_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Runtime plot saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark isoster vs AutoProf on real galaxy images"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--quick", "-q", action="store_true",
        help="Run only IC3370_mock2 (smoke test)",
    )
    parser.add_argument(
        "--plots", "-p", action="store_true",
        help="Generate comparison plots",
    )
    parser.add_argument(
        "--skip-autoprof", action="store_true",
        help="Skip AutoProf fitting (isoster only)",
    )
    parser.add_argument(
        "--galaxies", "-g", nargs="+", default=None,
        help="Galaxy names to benchmark (overrides --quick)",
    )

    args = parser.parse_args()

    results = run_all_benchmarks(
        output_dir=args.output,
        quick=args.quick,
        skip_autoprof=args.skip_autoprof,
        galaxies=args.galaxies,
    )

    if args.plots:
        generate_comparison_plots(results, output_dir=args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
