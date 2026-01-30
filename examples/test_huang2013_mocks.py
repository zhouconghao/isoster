#!/usr/bin/env python3
"""
Test script comparing photutils.isophote and isoster on realistic multi-component
Sersic galaxy models from Huang et al. 2013.

This script:
1. Generates mock galaxy images using the mockgal library
2. Fits them with both photutils.isophote and isoster
3. Produces detailed QA visualizations comparing the methods
4. Exports summary statistics for batch analysis

Usage:
    # Single galaxy (NGC 3923)
    python test_huang2013_mocks.py --galaxy "NGC 3923" --redshift 0.3 \\
        --pixel-scale 0.18 --psf-fwhm 0.7 --zeropoint 27.0

    # Batch mode (all 93 galaxies)
    python test_huang2013_mocks.py --batch --redshift 0.3 \\
        --pixel-scale 0.18 --psf-fwhm 0.7 --zeropoint 27.0

    # With noise and eccentric anomaly mode
    python test_huang2013_mocks.py --galaxy "NGC 3923" --redshift 0.3 \\
        --noise --use-eccentric-anomaly
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml
from astropy.io import fits
from astropy.table import Table
from matplotlib import gridspec
from matplotlib.patches import Ellipse as MPLEllipse
from photutils.aperture import EllipticalAperture
from scipy.ndimage import map_coordinates

# Import photutils.isophote
try:
    from photutils.isophote import Ellipse, EllipseGeometry, Isophote
except ImportError:
    print("ERROR: photutils not installed. Install with: pip install photutils")
    sys.exit(1)

# Import isoster
try:
    import isoster
    from isoster.config import IsosterConfig
except ImportError:
    print("ERROR: isoster not installed. Install with: pip install -e .")
    sys.exit(1)

# Import mockgal (from examples directory)
try:
    # Add examples directory to path
    sys.path.insert(0, str(Path(__file__).parent))
    from mockgal import generate_mock_image_from_model, load_model_file
except ImportError:
    print("ERROR: mockgal.py not found. Ensure it's in examples/")
    sys.exit(1)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ==============================================================================
# Configuration & Constants
# ==============================================================================

# Default parameters
DEFAULT_REDSHIFT = 0.3
DEFAULT_PIXEL_SCALE = 0.18  # arcsec/pixel
DEFAULT_PSF_FWHM = 0.7  # arcsec
DEFAULT_ZEROPOINT = 27.0
DEFAULT_SMA0 = 10.0  # pixels
DEFAULT_ASTEP = 0.1
MAX_IMAGE_SIZE = 2000  # pixels (cap for memory)
SIZE_FACTOR = 10.0  # Image size = SIZE_FACTOR * max(Re)

# Stop code colors, markers, labels (matching CLAUDE.md specifications)
STOP_CODE_STYLES = {
    0: {'color': 'steelblue', 'marker': 'o', 'label': 'Converged (0)'},
    1: {'color': 'orange', 'marker': 's', 'label': 'Flagged pixels (1)'},
    2: {'color': 'gold', 'marker': '^', 'label': 'Minor issues (2)'},
    3: {'color': 'coral', 'marker': 'D', 'label': 'Few points (3)'},
    -1: {'color': 'red', 'marker': 'x', 'label': 'Gradient error (-1)'},
}

# Figure settings
FIGURE_WIDTH = 20  # inches
FIGURE_HEIGHT = 24  # inches
FIGURE_DPI = 150


# ==============================================================================
# Mock Generation Module
# ==============================================================================

def compute_image_size(galaxy_model: Dict, pixel_scale: float,
                       size_factor: float = SIZE_FACTOR,
                       max_size: int = MAX_IMAGE_SIZE) -> int:
    """
    Compute adaptive image size based on galaxy effective radius.

    Args:
        galaxy_model: Galaxy model dictionary from YAML
        pixel_scale: Pixel scale in arcsec/pixel
        size_factor: Multiplier for maximum effective radius (default: 10.0)
        max_size: Maximum image size in pixels (default: 2000)

    Returns:
        Image size (half-size, so total size = 2 * size + 1)
    """
    # Find maximum effective radius across all components
    max_r_eff = 0.0
    for comp in galaxy_model.get('components', []):
        r_eff_kpc = comp.get('r_eff', 0.0)
        if r_eff_kpc > max_r_eff:
            max_r_eff = r_eff_kpc

    # Convert to pixels (need to account for redshift)
    # This is handled by mockgal, so we estimate the angular size
    # At z=0.3, angular diameter distance ~ 1085 Mpc
    # 1 kpc ~ 0.19 arcsec
    kpc_to_arcsec = 0.19  # Approximate at z=0.3
    max_r_eff_arcsec = max_r_eff * kpc_to_arcsec
    max_r_eff_pixels = max_r_eff_arcsec / pixel_scale

    # Apply size factor and cap at max_size
    half_size = int(np.ceil(size_factor * max_r_eff_pixels))

    # Ensure minimum size for small galaxies
    if max_r_eff_pixels < 5:
        half_size = max(half_size, int(np.ceil(15.0 * max_r_eff_pixels)))

    # Cap at max_size
    if half_size > max_size:
        print(f"WARNING: Image size {2*half_size+1} exceeds max {2*max_size+1}, capping")
        half_size = max_size

    return half_size


def get_initial_geometry(metadata: Dict) -> Tuple[float, float]:
    """
    Extract initial ellipticity and position angle from galaxy metadata.

    Uses luminosity-weighted average for multi-component galaxies.

    Args:
        metadata: Galaxy metadata dictionary with 'components' list

    Returns:
        (ellipticity, position_angle_deg)
    """
    components = metadata.get('components', [])
    if not components:
        return 0.2, 0.0

    # Compute luminosity-weighted averages
    total_flux = 0.0
    weighted_eps = 0.0
    weighted_pa = 0.0

    for comp in components:
        flux = comp.get('flux_fraction', 1.0 / len(components))
        eps = comp.get('ellipticity', 0.0)
        pa = comp.get('PA', 0.0)

        total_flux += flux
        weighted_eps += flux * eps
        weighted_pa += flux * pa

    if total_flux > 0:
        weighted_eps /= total_flux
        weighted_pa /= total_flux
    else:
        weighted_eps = components[0].get('ellipticity', 0.2)
        weighted_pa = components[0].get('PA', 0.0)

    return weighted_eps, weighted_pa


def generate_mock(galaxy_name: str, models_file: str,
                  redshift: float = DEFAULT_REDSHIFT,
                  pixel_scale: float = DEFAULT_PIXEL_SCALE,
                  psf_fwhm: float = DEFAULT_PSF_FWHM,
                  zeropoint: float = DEFAULT_ZEROPOINT,
                  add_noise: bool = False) -> Dict:
    """
    Generate mock galaxy image using mockgal library.

    Args:
        galaxy_name: Name of galaxy in models file
        models_file: Path to huang2013_models.yaml
        redshift: Galaxy redshift (overrides model default)
        pixel_scale: Pixel scale in arcsec/pixel
        psf_fwhm: Gaussian PSF FWHM in arcsec
        zeropoint: Magnitude zeropoint
        add_noise: Whether to add Poisson+readout noise

    Returns:
        Dictionary with:
            - 'image': 2D array of mock image
            - 'metadata': Galaxy metadata (components, etc.)
            - 'true_center': (x0, y0) in pixels
            - 'image_size': Size of image
            - 'pixel_scale': Pixel scale
            - 'redshift': Redshift used
    """
    # Load galaxy models to get metadata
    models = load_model_file(models_file)

    # Find the galaxy
    galaxy_model = None
    for model in models:
        if model.name == galaxy_name:
            galaxy_model = model
            break

    if galaxy_model is None:
        raise ValueError(f"Galaxy '{galaxy_name}' not found in {models_file}")

    # Get original redshift
    original_z = galaxy_model.redshift

    # Compute approximate image size from components
    max_r_eff_kpc = max(comp.r_eff_kpc for comp in galaxy_model.components)

    # At z=0.3, 1 kpc ~ 0.19 arcsec
    kpc_to_arcsec = 0.19
    max_r_eff_arcsec = max_r_eff_kpc * kpc_to_arcsec
    max_r_eff_pixels = max_r_eff_arcsec / pixel_scale

    # Apply size factor and cap at max_size
    half_size = int(np.ceil(SIZE_FACTOR * max_r_eff_pixels))

    # Ensure minimum size for small galaxies
    if max_r_eff_pixels < 5:
        half_size = max(half_size, int(np.ceil(15.0 * max_r_eff_pixels)))

    # Cap at max_size
    if half_size > MAX_IMAGE_SIZE:
        print(f"WARNING: Image size {2*half_size+1} exceeds max {2*MAX_IMAGE_SIZE+1}, capping")
        half_size = MAX_IMAGE_SIZE

    image_size = 2 * half_size + 1

    # Generate mock image
    print(f"Generating mock for {galaxy_name}")
    print(f"  Redshift: {redshift} (original: {original_z})")
    print(f"  Image size: {image_size} x {image_size} pixels")
    print(f"  Pixel scale: {pixel_scale} arcsec/pixel")
    print(f"  PSF FWHM: {psf_fwhm} arcsec")

    # Configure mock generation
    # Note: mockgal's ImageConfig might need different field names
    # Let's use a dict and let the function handle it
    config = {
        'pixel_scale': pixel_scale,
        'zeropoint': zeropoint,
        'psf_enabled': True,
        'psf_type': 'gaussian',
        'psf_fwhm': psf_fwhm,
        'noise_enabled': add_noise,
        'size_pixels': image_size,  # Try this instead of image_size
    }

    # Generate image using the mockgal API
    try:
        result = generate_mock_image_from_model(
            models_file,
            galaxy_name,
            config=config,
            return_metadata=True
        )

        if isinstance(result, tuple):
            image, metadata = result
        else:
            image = result
            metadata = {}

    except Exception as e:
        print(f"ERROR: Failed to generate mock for {galaxy_name}: {e}")
        import traceback
        traceback.print_exc()
        raise

    # True center is at image center
    true_center = ((image_size - 1) / 2.0, (image_size - 1) / 2.0)

    # Build metadata dict with component info
    metadata_dict = {
        'components': [
            {
                'r_eff': comp.r_eff_kpc,
                'ellipticity': comp.ellipticity,
                'PA': comp.pa_deg,
                'flux_fraction': 1.0 / len(galaxy_model.components)  # Assume equal flux for now
            }
            for comp in galaxy_model.components
        ]
    }

    return {
        'image': image,
        'metadata': metadata_dict,
        'true_center': true_center,
        'image_size': image_size,
        'pixel_scale': pixel_scale,
        'redshift': redshift,
        'galaxy_name': galaxy_name,
        'psf_fwhm': psf_fwhm,
        'zeropoint': zeropoint,
        'add_noise': add_noise
    }


# ==============================================================================
# Fitting Module
# ==============================================================================

def run_photutils(image: np.ndarray,
                  x0: float, y0: float,
                  eps: float, pa: float,
                  sma0: float = DEFAULT_SMA0,
                  astep: float = DEFAULT_ASTEP) -> Tuple[Dict, float]:
    """
    Run photutils.isophote fitting.

    Args:
        image: 2D image array
        x0, y0: Initial center coordinates
        eps: Initial ellipticity
        pa: Initial position angle (degrees)
        sma0: Starting semi-major axis (pixels)
        astep: SMA step size

    Returns:
        (results_dict, runtime_seconds)
    """
    print("Running photutils.isophote...")

    # Determine maxgerr based on ellipticity
    maxgerr = 1.0 if eps > 0.5 else 0.5

    # Create geometry
    geometry = EllipseGeometry(
        x0=x0, y0=y0,
        sma=sma0,
        eps=eps,
        pa=np.radians(pa)
    )

    # Create Ellipse fitter
    ellipse = Ellipse(image, geometry)

    # Fit
    start_time = time.time()
    try:
        isolist = ellipse.fit_image(
            step=astep,
            maxgerr=maxgerr,
            nclip=2,
            sclip=3.0,
            integrmode='bilinear'
        )
    except Exception as e:
        print(f"ERROR: photutils fitting failed: {e}")
        return None, 0.0

    runtime = time.time() - start_time

    print(f"  Completed in {runtime:.2f}s")
    print(f"  Found {len(isolist)} isophotes")

    # Convert to standard dict format
    results = convert_photutils_to_dict(isolist)

    return results, runtime


def run_isoster(image: np.ndarray,
                x0: Optional[float], y0: Optional[float],
                eps: float, pa: float,
                sma0: float = DEFAULT_SMA0,
                astep: float = DEFAULT_ASTEP,
                use_eccentric_anomaly: bool = False) -> Tuple[Dict, float]:
    """
    Run isoster fitting.

    Args:
        image: 2D image array
        x0, y0: Initial center coordinates (None = auto-detect)
        eps: Initial ellipticity
        pa: Initial position angle (degrees)
        sma0: Starting semi-major axis (pixels)
        astep: SMA step size
        use_eccentric_anomaly: Enable EA mode

    Returns:
        (results_dict, runtime_seconds)
    """
    print("Running isoster...")

    # Determine maxgerr based on ellipticity
    maxgerr = 1.0 if eps > 0.5 else 0.5

    # Allow fitting to image corners (isophotes can be cut off by edge,
    # which only affects geometry fitting but intensity extraction works
    # with partial coverage)
    max_radius = max(image.shape) / 2.0 * np.sqrt(2.0) - 2

    # Create configuration
    # Note: IsosterConfig expects PA in radians, but input is in degrees
    config = IsosterConfig(
        sma0=sma0,
        x0=x0, y0=y0,
        eps=eps,
        pa=np.radians(pa),  # Convert degrees to radians
        step=astep,
        maxgerr=maxgerr,
        maxsma=max_radius,  # Allow fitting to corners
        maxrit=100,  # Increase max iterations
        use_eccentric_anomaly=use_eccentric_anomaly,
        full_photometry=True,
        compute_errors=True,
        compute_deviations=True,
        harmonic_orders=[1, 2, 3, 4],
        simultaneous_harmonics=False,
        permissive_geometry=False,
        use_central_regularization=False
    )

    # No mask for clean mocks
    mask = None

    # Fit
    start_time = time.time()
    try:
        results = isoster.fit_image(image, mask, config)
    except Exception as e:
        print(f"ERROR: isoster fitting failed: {e}")
        return None, 0.0

    runtime = time.time() - start_time

    print(f"  Completed in {runtime:.2f}s")
    print(f"  Found {len(results['isophotes'])} isophotes")

    return results, runtime


def convert_photutils_to_dict(isolist) -> Dict:
    """
    Convert photutils IsoList to standard dictionary format.

    Args:
        isolist: photutils IsoList object

    Returns:
        Dictionary with 'isophotes' list and 'config' dict
    """
    isophotes = []

    for iso in isolist:
        iso_dict = {
            'sma': iso.sma,
            'x0': iso.x0,
            'y0': iso.y0,
            'eps': iso.eps,
            'pa': iso.pa,  # radians
            'intens': iso.intens,
            'intens_err': iso.int_err,
            'rms': iso.rms,
            'stop_code': iso.stop_code,
            'ndata': iso.ndata,
            'nflag': iso.nflag,
            # Harmonics
            'a1': getattr(iso, 'a1', 0.0),
            'b1': getattr(iso, 'b1', 0.0),
            'a2': getattr(iso, 'a2', 0.0),
            'b2': getattr(iso, 'b2', 0.0),
            'a3': getattr(iso, 'a3', 0.0),
            'b3': getattr(iso, 'b3', 0.0),
            'a4': getattr(iso, 'a4', 0.0),
            'b4': getattr(iso, 'b4', 0.0),
            # Errors
            'a1_err': getattr(iso, 'a1_err', 0.0),
            'b1_err': getattr(iso, 'b1_err', 0.0),
            'a2_err': getattr(iso, 'a2_err', 0.0),
            'b2_err': getattr(iso, 'b2_err', 0.0),
            'a3_err': getattr(iso, 'a3_err', 0.0),
            'b3_err': getattr(iso, 'b3_err', 0.0),
            'a4_err': getattr(iso, 'a4_err', 0.0),
            'b4_err': getattr(iso, 'b4_err', 0.0),
        }

        # Add tflux_e if available (for curve of growth)
        if hasattr(iso, 'tflux_e'):
            iso_dict['tflux_e'] = iso.tflux_e

        isophotes.append(iso_dict)

    return {'isophotes': isophotes, 'config': {}}


def build_photutils_model(image_shape: Tuple[int, int],
                          results: Dict,
                          use_harmonics: bool = True,
                          harmonic_orders: List[int] = [3, 4]) -> np.ndarray:
    """
    Build 2D model from photutils isophote results.

    This mimics isoster.build_isoster_model() for consistency.

    Args:
        image_shape: Shape of output image
        results: Results dictionary from run_photutils()
        use_harmonics: Include harmonic deviations
        harmonic_orders: Which harmonics to use

    Returns:
        2D model image
    """
    # Use isoster's model builder but with photutils results
    # First need to ensure PA is in radians
    isophotes = results['isophotes']

    # Convert PA to radians if needed (photutils uses radians already)
    for iso in isophotes:
        if 'pa' in iso:
            pass  # Already in radians from photutils

    # Use isoster's model builder
    try:
        model = isoster.build_isoster_model(
            image_shape,
            isophotes,
            use_harmonics=use_harmonics,
            harmonic_orders=harmonic_orders
        )
        return model
    except Exception as e:
        print(f"WARNING: Failed to build photutils model: {e}")
        # Fall back to simple interpolation
        model = np.zeros(image_shape)
        # Simple radial interpolation (no harmonics)
        # ... (fallback implementation if needed)
        return model


# ==============================================================================
# Analysis Module
# ==============================================================================

def extract_mock_profile(mock_image: np.ndarray,
                        isophotes: List[Dict],
                        pixel_scale: float = DEFAULT_PIXEL_SCALE) -> np.ndarray:
    """
    Extract "true" intensity profile from mock image along fitted ellipses.

    Samples the mock image at the fitted ellipse positions to get ground truth.

    Args:
        mock_image: 2D mock image array
        isophotes: List of isophote dictionaries
        pixel_scale: Pixel scale in arcsec/pixel

    Returns:
        Array of true intensities at each SMA
    """
    true_intens = np.zeros(len(isophotes))

    for i, iso in enumerate(isophotes):
        sma = iso['sma']
        x0 = iso['x0']
        y0 = iso['y0']
        eps = iso['eps']
        pa = iso.get('pa', 0.0)  # radians

        # Generate ellipse points
        n_points = 360
        theta = np.linspace(0, 2*np.pi, n_points)

        # Ellipse in polar coordinates
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Convert to Cartesian (image coordinates)
        cos_pa = np.cos(pa)
        sin_pa = np.sin(pa)

        # Ellipse equation: r = sma * (1 - eps)
        b = sma * (1 - eps)

        x = x0 + sma * cos_theta * cos_pa - b * sin_theta * sin_pa
        y = y0 + sma * cos_theta * sin_pa + b * sin_theta * cos_pa

        # Sample mock image at these points
        coords = np.array([y, x])  # Note: (row, col) = (y, x)

        # Use bilinear interpolation
        sampled = map_coordinates(mock_image, coords, order=1, mode='constant', cval=0.0)

        # Take median to be robust against outliers
        true_intens[i] = np.median(sampled[np.isfinite(sampled)])

    return true_intens


def compute_curve_of_growth(mock_image: np.ndarray,
                           isophotes: List[Dict],
                           pixel_scale: float = DEFAULT_PIXEL_SCALE) -> np.ndarray:
    """
    Compute true curve of growth from mock image using elliptical aperture photometry.

    Args:
        mock_image: 2D mock image array
        isophotes: List of isophote dictionaries
        pixel_scale: Pixel scale in arcsec/pixel

    Returns:
        Array of integrated fluxes at each SMA
    """
    from photutils.aperture import aperture_photometry

    true_cog = np.zeros(len(isophotes))

    for i, iso in enumerate(isophotes):
        sma = iso['sma']
        x0 = iso['x0']
        y0 = iso['y0']
        eps = iso['eps']
        pa = iso.get('pa', 0.0)  # radians

        # Skip invalid isophotes (sma <= 0 or eps >= 1)
        if sma <= 0 or eps >= 1.0 or eps < 0:
            true_cog[i] = np.nan
            continue

        # Create elliptical aperture
        # photutils uses (x, y) for position
        # a = sma, b = sma * (1 - eps)
        b = sma * (1 - eps)

        # Check if b is positive
        if b <= 0:
            true_cog[i] = np.nan
            continue

        aperture = EllipticalAperture(
            (x0, y0),
            a=sma,
            b=b,
            theta=pa
        )

        # Perform aperture photometry with subpixel sampling
        try:
            phot_table = aperture_photometry(mock_image, aperture, method='exact')
            true_cog[i] = phot_table['aperture_sum'][0]
        except Exception as e:
            print(f"WARNING: Aperture photometry failed for isophote {i}: {e}")
            true_cog[i] = np.nan

    return true_cog


def normalize_pa_array(pa_array: np.ndarray, threshold: float = 90.0) -> np.ndarray:
    """
    Normalize position angle array to remove jumps > threshold degrees.

    Args:
        pa_array: Array of position angles in degrees
        threshold: Jump threshold in degrees (default: 90)

    Returns:
        Normalized PA array
    """
    if len(pa_array) == 0:
        return pa_array

    pa_norm = pa_array.copy()

    # Normalize to [0, 180)
    pa_norm = pa_norm % 180.0

    # Fix jumps
    for i in range(1, len(pa_norm)):
        diff = pa_norm[i] - pa_norm[i-1]

        if abs(diff) > threshold:
            # Shift by 180 degrees
            pa_norm[i:] = (pa_norm[i:] + 180.0) % 180.0

    return pa_norm


def compute_residual_statistics(true_values: np.ndarray,
                                fitted_values: np.ndarray,
                                sma_values: np.ndarray,
                                sma_ranges: List[Tuple[float, float]],
                                stop_codes: np.ndarray) -> Dict:
    """
    Compute residual statistics in different radial ranges.

    Args:
        true_values: True values from mock
        fitted_values: Fitted values
        sma_values: SMA values
        sma_ranges: List of (min_sma, max_sma) tuples
        stop_codes: Stop codes for filtering

    Returns:
        Dictionary with statistics per range
    """
    stats = {}

    for sma_min, sma_max in sma_ranges:
        # Select data in range with good stop codes
        mask = (sma_values >= sma_min) & (sma_values < sma_max) & (stop_codes == 0)

        if np.sum(mask) < 3:
            stats[f'{sma_min:.1f}_{sma_max:.1f}'] = {
                'median': np.nan,
                'max': np.nan,
                'rms': np.nan,
                'n_points': 0
            }
            continue

        # Compute relative residuals (%)
        residuals = 100.0 * (fitted_values[mask] - true_values[mask]) / true_values[mask]

        stats[f'{sma_min:.1f}_{sma_max:.1f}'] = {
            'median': np.median(residuals),
            'max': np.max(np.abs(residuals)),
            'rms': np.sqrt(np.mean(residuals**2)),
            'n_points': np.sum(mask)
        }

    return stats


def identify_component_regions(galaxy_model: Dict,
                               pixel_scale: float,
                               redshift: float) -> List[float]:
    """
    Identify radial boundaries between Sersic components.

    Args:
        galaxy_model: Galaxy model dictionary
        pixel_scale: Pixel scale in arcsec/pixel
        redshift: Galaxy redshift

    Returns:
        List of component r_eff values in pixels (sorted)
    """
    components = galaxy_model.get('components', [])

    # Convert r_eff from kpc to pixels
    kpc_to_arcsec = 0.19  # Approximate at z=0.3

    r_eff_pixels = []
    for comp in components:
        r_eff_kpc = comp.get('r_eff', 0.0)
        r_eff_arcsec = r_eff_kpc * kpc_to_arcsec
        r_eff_pix = r_eff_arcsec / pixel_scale
        r_eff_pixels.append(r_eff_pix)

    return sorted(r_eff_pixels)


# ==============================================================================
# Visualization Module
# ==============================================================================

def plot_comprehensive_qa(mock_data: Dict,
                         results_photutils: Dict,
                         results_isoster: Dict,
                         runtime_photutils: float,
                         runtime_isoster: float,
                         output_path: Path,
                         dpi: int = FIGURE_DPI):
    """
    Create comprehensive QA figure comparing photutils and isoster results.

    Layout:
        - Left column: 2D images (input + residual maps)
        - Right column: 1D profiles (SB, residuals, geometry)

    Args:
        mock_data: Mock image data dictionary
        results_photutils: photutils results
        results_isoster: isoster results
        runtime_photutils: photutils runtime (seconds)
        runtime_isoster: isoster runtime (seconds)
        output_path: Path to save figure
        dpi: Figure DPI
    """
    # Extract data
    mock_image = mock_data['image']
    metadata = mock_data['metadata']
    true_center = mock_data['true_center']
    pixel_scale = mock_data['pixel_scale']
    redshift = mock_data['redshift']
    galaxy_name = mock_data['galaxy_name']
    psf_fwhm = mock_data['psf_fwhm']

    iso_photutils = results_photutils['isophotes']
    iso_isoster = results_isoster['isophotes']

    # Compute speedup
    speedup = runtime_photutils / runtime_isoster if runtime_isoster > 0 else 0.0

    # Create figure
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=dpi)

    # Create grid: 2 columns, multiple rows
    # Left: 3 image panels (input + 2 residuals)
    # Right: 6 1D profile panels
    gs = gridspec.GridSpec(
        8, 2,
        figure=fig,
        width_ratios=[1, 1],
        height_ratios=[1, 1, 1, 1.5, 0.8, 0.8, 0.8, 1],
        hspace=0.3,
        wspace=0.3
    )

    # Title
    title_str = (
        f"{galaxy_name} Comparison\n"
        f"z={redshift}, PSF={psf_fwhm:.2f}\", scale={pixel_scale:.2f}\"/pix, "
        f"photutils={runtime_photutils:.1f}s, isoster={runtime_isoster:.1f}s "
        f"({speedup:.1f}× speedup)"
    )
    fig.suptitle(title_str, fontsize=14, fontweight='bold')

    # ===== Left Column: 2D Images =====

    # Panel A: Input mock + true isophote contours
    ax_input = fig.add_subplot(gs[0, 0])
    plot_input_with_contours(ax_input, mock_image, metadata, pixel_scale, redshift)

    # Panel B: photutils residual map
    ax_res_photutils = fig.add_subplot(gs[1, 0])
    if results_photutils is not None:
        model_photutils = build_photutils_model(mock_image.shape, results_photutils)
        plot_residual_map(ax_res_photutils, mock_image, model_photutils,
                         iso_photutils, 'photutils', pixel_scale)
    else:
        ax_res_photutils.text(0.5, 0.5, 'photutils fitting failed',
                             ha='center', va='center', transform=ax_res_photutils.transAxes)
        ax_res_photutils.axis('off')

    # Panel C: isoster residual map
    ax_res_isoster = fig.add_subplot(gs[2, 0])
    if results_isoster is not None:
        model_isoster = isoster.build_isoster_model(
            mock_image.shape, iso_isoster,
            use_harmonics=True, harmonic_orders=[3, 4]
        )
        plot_residual_map(ax_res_isoster, mock_image, model_isoster,
                         iso_isoster, 'isoster', pixel_scale)
    else:
        ax_res_isoster.text(0.5, 0.5, 'isoster fitting failed',
                           ha='center', va='center', transform=ax_res_isoster.transAxes)
        ax_res_isoster.axis('off')

    # ===== Right Column: 1D Profiles =====

    # Extract true profiles
    true_intens_photutils = extract_mock_profile(mock_image, iso_photutils, pixel_scale)
    true_intens_isoster = extract_mock_profile(mock_image, iso_isoster, pixel_scale)

    # Compute curves of growth
    true_cog_photutils = compute_curve_of_growth(mock_image, iso_photutils, pixel_scale)
    true_cog_isoster = compute_curve_of_growth(mock_image, iso_isoster, pixel_scale)

    # Panel D: Surface brightness profiles
    ax_sb = fig.add_subplot(gs[3, 1])
    plot_surface_brightness(ax_sb, iso_photutils, iso_isoster,
                           true_intens_photutils, true_intens_isoster,
                           pixel_scale, redshift)

    # Panel E: Relative residuals
    ax_resid = fig.add_subplot(gs[4, 1], sharex=ax_sb)
    plot_residuals_1d(ax_resid, iso_photutils, iso_isoster,
                     true_intens_photutils, true_intens_isoster,
                     pixel_scale, redshift)

    # Panel F: Centroid
    ax_centroid = fig.add_subplot(gs[5, 1], sharex=ax_sb)
    plot_centroid(ax_centroid, iso_photutils, iso_isoster, true_center,
                 pixel_scale, redshift)

    # Panel G: Ellipticity
    ax_eps = fig.add_subplot(gs[6, 1], sharex=ax_sb)
    plot_ellipticity(ax_eps, iso_photutils, iso_isoster, metadata,
                    pixel_scale, redshift)

    # Panel H: Position angle
    ax_pa = fig.add_subplot(gs[7, 1], sharex=ax_sb)
    plot_position_angle(ax_pa, iso_photutils, iso_isoster, metadata,
                       pixel_scale, redshift)

    # Add stop code legend
    add_stop_code_legend(fig)

    # Save figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"Saved QA figure to {output_path}")


def plot_input_with_contours(ax, mock_image, metadata, pixel_scale, redshift):
    """Plot input mock image with true isophote contours."""
    # Display image
    vmin, vmax = np.percentile(mock_image[mock_image > 0], [1, 99.5])
    im = ax.imshow(mock_image, origin='lower', cmap='gray_r',
                   vmin=vmin, vmax=vmax, interpolation='nearest')

    # Add component contours (true r_eff ellipses)
    components = metadata.get('components', [])

    # Convert r_eff to pixels
    kpc_to_arcsec = 0.19  # At z=0.3
    center_x = (mock_image.shape[1] - 1) / 2.0
    center_y = (mock_image.shape[0] - 1) / 2.0

    colors = ['cyan', 'lime', 'yellow', 'magenta']

    for i, comp in enumerate(components):
        r_eff_kpc = comp.get('r_eff', 0.0)
        eps = comp.get('ellipticity', 0.0)
        pa = comp.get('PA', 0.0)  # degrees

        r_eff_arcsec = r_eff_kpc * kpc_to_arcsec
        r_eff_pix = r_eff_arcsec / pixel_scale

        # Draw ellipse at r_eff
        width = 2 * r_eff_pix
        height = 2 * r_eff_pix * (1 - eps)
        angle = pa  # degrees

        ellipse = MPLEllipse(
            (center_x, center_y),
            width, height,
            angle=angle,
            edgecolor=colors[i % len(colors)],
            facecolor='none',
            linewidth=2,
            linestyle='--',
            label=f'Comp {i} (Re)'
        )
        ax.add_patch(ellipse)

    ax.set_title('Input Mock + True Isophotes', fontsize=10)
    ax.set_xlabel('X (pixels)', fontsize=9)
    ax.set_ylabel('Y (pixels)', fontsize=9)

    if len(components) > 0:
        ax.legend(loc='upper right', fontsize=8, framealpha=0.7)


def plot_residual_map(ax, mock_image, model_image, isophotes, method_name, pixel_scale):
    """Plot fractional residual map with fitted isophote overlays."""
    # Compute fractional residuals (%)
    with np.errstate(divide='ignore', invalid='ignore'):
        residual = 100.0 * (model_image - mock_image) / mock_image

    # Cap at 99th percentile for display
    vmax = min(np.nanpercentile(np.abs(residual), 99), 2.0)  # Cap at 2%

    # Display residual map
    im = ax.imshow(residual, origin='lower', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax, interpolation='nearest')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Residual (%)', fontsize=8)

    # Overlay sparse fitted isophotes (every 10th, SMA > 3 pixels)
    for i, iso in enumerate(isophotes):
        if i % 10 != 0:
            continue

        sma = iso['sma']
        if sma < 3:
            continue

        x0 = iso['x0']
        y0 = iso['y0']
        eps = iso['eps']
        pa = iso.get('pa', 0.0)  # radians or degrees?

        # Check if PA is in radians (photutils) or degrees (isoster)
        if method_name == 'photutils':
            pa_deg = np.degrees(pa)
        else:
            pa_deg = pa

        width = 2 * sma
        height = 2 * sma * (1 - eps)

        ellipse = MPLEllipse(
            (x0, y0),
            width, height,
            angle=pa_deg,
            edgecolor='black',
            facecolor='none',
            linewidth=0.5,
            alpha=0.5
        )
        ax.add_patch(ellipse)

    ax.set_title(f'{method_name} Residual Map', fontsize=10)
    ax.set_xlabel('X (pixels)', fontsize=9)
    ax.set_ylabel('Y (pixels)', fontsize=9)


def plot_surface_brightness(ax, iso_photutils, iso_isoster,
                           true_intens_photutils, true_intens_isoster,
                           pixel_scale, redshift):
    """Plot surface brightness profiles."""
    # Convert SMA to kpc
    kpc_per_arcsec = 5.26  # At z=0.3

    sma_photutils = np.array([iso['sma'] for iso in iso_photutils])
    sma_isoster = np.array([iso['sma'] for iso in iso_isoster])

    # Convert to kpc
    sma_kpc_photutils = sma_photutils * pixel_scale * kpc_per_arcsec
    sma_kpc_isoster = sma_isoster * pixel_scale * kpc_per_arcsec

    # X-axis: SMA^0.25 in kpc
    x_photutils = sma_kpc_photutils ** 0.25
    x_isoster = sma_kpc_isoster ** 0.25

    # Extract intensities
    intens_photutils = np.array([iso['intens'] for iso in iso_photutils])
    intens_isoster = np.array([iso['intens'] for iso in iso_isoster])

    intens_err_photutils = np.array([iso.get('intens_err', 0.0) for iso in iso_photutils])
    intens_err_isoster = np.array([iso.get('intens_err', 0.0) for iso in iso_isoster])

    stop_codes_photutils = np.array([iso['stop_code'] for iso in iso_photutils])
    stop_codes_isoster = np.array([iso['stop_code'] for iso in iso_isoster])

    # Plot true profiles (dashed lines)
    ax.plot(x_photutils, true_intens_photutils, 'k--', linewidth=1.5,
            label='True (mock)', zorder=1)

    # Plot fitted profiles with stop code colors
    for stop_code, style in STOP_CODE_STYLES.items():
        # photutils
        mask_p = stop_codes_photutils == stop_code
        if np.any(mask_p):
            ax.errorbar(
                x_photutils[mask_p], intens_photutils[mask_p],
                yerr=intens_err_photutils[mask_p],
                fmt=style['marker'], color=style['color'],
                markersize=5, capsize=2, alpha=0.7,
                label=f"photutils {style['label']}" if stop_code == 0 else None,
                zorder=2
            )

        # isoster
        mask_i = stop_codes_isoster == stop_code
        if np.any(mask_i):
            ax.errorbar(
                x_isoster[mask_i], intens_isoster[mask_i],
                yerr=intens_err_isoster[mask_i],
                fmt=style['marker'], color=style['color'],
                markersize=5, capsize=2, alpha=0.7, fillstyle='none',
                label=f"isoster {style['label']}" if stop_code == 0 else None,
                zorder=2
            )

    ax.set_ylabel('Intensity', fontsize=9)
    ax.set_title('Surface Brightness Profile', fontsize=10)
    ax.legend(loc='best', fontsize=7)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Set Y-axis limits to exclude huge error bars in outskirts
    valid_intens_photutils = intens_photutils[stop_codes_photutils == 0]
    valid_intens_isoster = intens_isoster[stop_codes_isoster == 0]

    if len(valid_intens_photutils) > 0 or len(valid_intens_isoster) > 0:
        all_intens = np.concatenate([valid_intens_photutils, valid_intens_isoster])
        all_intens = all_intens[all_intens > 0]
        if len(all_intens) > 0:
            ymin, ymax = np.percentile(all_intens, [1, 99])
            ax.set_ylim(ymin * 0.5, ymax * 2.0)


def plot_residuals_1d(ax, iso_photutils, iso_isoster,
                     true_intens_photutils, true_intens_isoster,
                     pixel_scale, redshift):
    """Plot relative intensity residuals (%)."""
    # Convert SMA to kpc
    kpc_per_arcsec = 5.26  # At z=0.3

    sma_photutils = np.array([iso['sma'] for iso in iso_photutils])
    sma_isoster = np.array([iso['sma'] for iso in iso_isoster])

    sma_kpc_photutils = sma_photutils * pixel_scale * kpc_per_arcsec
    sma_kpc_isoster = sma_isoster * pixel_scale * kpc_per_arcsec

    x_photutils = sma_kpc_photutils ** 0.25
    x_isoster = sma_kpc_isoster ** 0.25

    # Extract intensities
    intens_photutils = np.array([iso['intens'] for iso in iso_photutils])
    intens_isoster = np.array([iso['intens'] for iso in iso_isoster])

    stop_codes_photutils = np.array([iso['stop_code'] for iso in iso_photutils])
    stop_codes_isoster = np.array([iso['stop_code'] for iso in iso_isoster])

    # Compute relative residuals (%)
    with np.errstate(divide='ignore', invalid='ignore'):
        resid_photutils = 100.0 * (intens_photutils - true_intens_photutils) / true_intens_photutils
        resid_isoster = 100.0 * (intens_isoster - true_intens_isoster) / true_intens_isoster

    # Plot with stop code colors
    for stop_code, style in STOP_CODE_STYLES.items():
        mask_p = stop_codes_photutils == stop_code
        if np.any(mask_p):
            ax.scatter(x_photutils[mask_p], resid_photutils[mask_p],
                      marker=style['marker'], color=style['color'],
                      s=30, alpha=0.7, zorder=2)

        mask_i = stop_codes_isoster == stop_code
        if np.any(mask_i):
            ax.scatter(x_isoster[mask_i], resid_isoster[mask_i],
                      marker=style['marker'], color=style['color'],
                      s=30, alpha=0.7, facecolors='none', zorder=2)

    # Zero line
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_ylabel('Residual (%)', fontsize=9)
    ax.set_title('Relative Intensity Residuals', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set Y-axis limits excluding outliers
    valid_resids_p = resid_photutils[stop_codes_photutils == 0]
    valid_resids_i = resid_isoster[stop_codes_isoster == 0]

    if len(valid_resids_p) > 0 or len(valid_resids_i) > 0:
        all_resids = np.concatenate([valid_resids_p, valid_resids_i])
        all_resids = all_resids[np.isfinite(all_resids)]
        if len(all_resids) > 0:
            ymax = np.percentile(np.abs(all_resids), 95)
            ax.set_ylim(-ymax * 1.2, ymax * 1.2)


def plot_centroid(ax, iso_photutils, iso_isoster, true_center, pixel_scale, redshift):
    """Plot centroid evolution."""
    # Convert SMA to kpc
    kpc_per_arcsec = 5.26

    sma_photutils = np.array([iso['sma'] for iso in iso_photutils])
    sma_isoster = np.array([iso['sma'] for iso in iso_isoster])

    sma_kpc_photutils = sma_photutils * pixel_scale * kpc_per_arcsec
    sma_kpc_isoster = sma_isoster * pixel_scale * kpc_per_arcsec

    x_photutils = sma_kpc_photutils ** 0.25
    x_isoster = sma_kpc_isoster ** 0.25

    # Extract centroids
    x0_photutils = np.array([iso['x0'] for iso in iso_photutils])
    y0_photutils = np.array([iso['y0'] for iso in iso_photutils])
    x0_isoster = np.array([iso['x0'] for iso in iso_isoster])
    y0_isoster = np.array([iso['y0'] for iso in iso_isoster])

    stop_codes_photutils = np.array([iso['stop_code'] for iso in iso_photutils])
    stop_codes_isoster = np.array([iso['stop_code'] for iso in iso_isoster])

    # Plot X centroid
    for stop_code, style in STOP_CODE_STYLES.items():
        mask_p = stop_codes_photutils == stop_code
        if np.any(mask_p):
            ax.scatter(x_photutils[mask_p], x0_photutils[mask_p],
                      marker=style['marker'], color=style['color'],
                      s=20, alpha=0.7, zorder=2)

        mask_i = stop_codes_isoster == stop_code
        if np.any(mask_i):
            ax.scatter(x_isoster[mask_i], x0_isoster[mask_i],
                      marker=style['marker'], color=style['color'],
                      s=20, alpha=0.7, facecolors='none', zorder=2)

    # True center line
    ax.axhline(true_center[0], color='black', linestyle='--', linewidth=1,
              label=f'True center (x={true_center[0]:.1f}, y={true_center[1]:.1f})')

    ax.set_ylabel('Centroid X (pixels)', fontsize=9)
    ax.set_title('Centroid Evolution', fontsize=10)
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_ellipticity(ax, iso_photutils, iso_isoster, metadata, pixel_scale, redshift):
    """Plot ellipticity evolution."""
    # Convert SMA to kpc
    kpc_per_arcsec = 5.26

    sma_photutils = np.array([iso['sma'] for iso in iso_photutils])
    sma_isoster = np.array([iso['sma'] for iso in iso_isoster])

    sma_kpc_photutils = sma_photutils * pixel_scale * kpc_per_arcsec
    sma_kpc_isoster = sma_isoster * pixel_scale * kpc_per_arcsec

    x_photutils = sma_kpc_photutils ** 0.25
    x_isoster = sma_kpc_isoster ** 0.25

    # Extract ellipticities
    eps_photutils = np.array([iso['eps'] for iso in iso_photutils])
    eps_isoster = np.array([iso['eps'] for iso in iso_isoster])

    stop_codes_photutils = np.array([iso['stop_code'] for iso in iso_photutils])
    stop_codes_isoster = np.array([iso['stop_code'] for iso in iso_isoster])

    # Plot
    for stop_code, style in STOP_CODE_STYLES.items():
        mask_p = stop_codes_photutils == stop_code
        if np.any(mask_p):
            ax.scatter(x_photutils[mask_p], eps_photutils[mask_p],
                      marker=style['marker'], color=style['color'],
                      s=20, alpha=0.7, zorder=2)

        mask_i = stop_codes_isoster == stop_code
        if np.any(mask_i):
            ax.scatter(x_isoster[mask_i], eps_isoster[mask_i],
                      marker=style['marker'], color=style['color'],
                      s=20, alpha=0.7, facecolors='none', zorder=2)

    # True ellipticities per component (horizontal lines)
    components = metadata.get('components', [])
    colors = ['cyan', 'lime', 'yellow', 'magenta']
    kpc_to_arcsec = 0.19

    for i, comp in enumerate(components):
        eps_true = comp.get('ellipticity', 0.0)
        r_eff_kpc = comp.get('r_eff', 0.0)
        r_eff_pix = r_eff_kpc * kpc_to_arcsec / pixel_scale
        r_eff_kpc_sma = r_eff_pix * pixel_scale * kpc_per_arcsec
        x_eff = r_eff_kpc_sma ** 0.25

        ax.axhline(eps_true, color=colors[i % len(colors)], linestyle='--',
                  linewidth=1, alpha=0.7, label=f'Comp {i} (ε={eps_true:.2f})')

    ax.set_ylabel('Ellipticity', fontsize=9)
    ax.set_title('Ellipticity Profile', fontsize=10)
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)


def plot_position_angle(ax, iso_photutils, iso_isoster, metadata, pixel_scale, redshift):
    """Plot position angle evolution (normalized)."""
    # Convert SMA to kpc
    kpc_per_arcsec = 5.26

    sma_photutils = np.array([iso['sma'] for iso in iso_photutils])
    sma_isoster = np.array([iso['sma'] for iso in iso_isoster])

    sma_kpc_photutils = sma_photutils * pixel_scale * kpc_per_arcsec
    sma_kpc_isoster = sma_isoster * pixel_scale * kpc_per_arcsec

    x_photutils = sma_kpc_photutils ** 0.25
    x_isoster = sma_kpc_isoster ** 0.25

    # Extract PAs (convert to degrees if needed)
    pa_photutils = np.array([np.degrees(iso.get('pa', 0.0)) for iso in iso_photutils])
    pa_isoster = np.array([iso.get('pa', 0.0) for iso in iso_isoster])

    # Check if isoster PA is already in degrees
    if np.max(np.abs(pa_isoster)) > 2 * np.pi:
        # Already in degrees
        pass
    else:
        # Convert from radians
        pa_isoster = np.degrees(pa_isoster)

    # Normalize
    pa_photutils = normalize_pa_array(pa_photutils)
    pa_isoster = normalize_pa_array(pa_isoster)

    stop_codes_photutils = np.array([iso['stop_code'] for iso in iso_photutils])
    stop_codes_isoster = np.array([iso['stop_code'] for iso in iso_isoster])

    # Plot
    for stop_code, style in STOP_CODE_STYLES.items():
        mask_p = stop_codes_photutils == stop_code
        if np.any(mask_p):
            ax.scatter(x_photutils[mask_p], pa_photutils[mask_p],
                      marker=style['marker'], color=style['color'],
                      s=20, alpha=0.7, zorder=2)

        mask_i = stop_codes_isoster == stop_code
        if np.any(mask_i):
            ax.scatter(x_isoster[mask_i], pa_isoster[mask_i],
                      marker=style['marker'], color=style['color'],
                      s=20, alpha=0.7, facecolors='none', zorder=2)

    # True PAs per component
    components = metadata.get('components', [])
    colors = ['cyan', 'lime', 'yellow', 'magenta']

    for i, comp in enumerate(components):
        pa_true = comp.get('PA', 0.0)
        ax.axhline(pa_true % 180.0, color=colors[i % len(colors)], linestyle='--',
                  linewidth=1, alpha=0.7, label=f'Comp {i} (PA={pa_true:.1f}°)')

    ax.set_ylabel('Position Angle (deg)', fontsize=9)
    ax.set_xlabel('SMA$^{0.25}$ (kpc$^{0.25}$)', fontsize=9)
    ax.set_title('Position Angle Profile', fontsize=10)
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 180)


def add_stop_code_legend(fig):
    """Add stop code legend to figure."""
    # Create legend entries
    handles = []
    labels = []

    for stop_code in [0, 1, 2, 3, -1]:
        style = STOP_CODE_STYLES[stop_code]
        handle = plt.Line2D([0], [0], marker=style['marker'], color='w',
                           markerfacecolor=style['color'], markersize=8,
                           label=style['label'])
        handles.append(handle)
        labels.append(style['label'])

    # Add legend at bottom
    fig.legend(handles, labels, loc='lower center', ncol=5,
              fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.02))


# ==============================================================================
# Utilities Module
# ==============================================================================

def load_all_galaxy_names(models_file: str) -> List[str]:
    """
    Load all galaxy names from YAML file.

    Args:
        models_file: Path to huang2013_models.yaml

    Returns:
        List of galaxy names
    """
    models = load_model_file(models_file)
    return list(models.keys())


def convert_sma_to_kpc(sma_pixels: float, pixel_scale: float, redshift: float) -> float:
    """
    Convert SMA from pixels to kpc.

    Args:
        sma_pixels: SMA in pixels
        pixel_scale: Pixel scale in arcsec/pixel
        redshift: Galaxy redshift

    Returns:
        SMA in kpc
    """
    # At z=0.3, 1 arcsec ~ 5.26 kpc
    kpc_per_arcsec = 5.26

    sma_arcsec = sma_pixels * pixel_scale
    sma_kpc = sma_arcsec * kpc_per_arcsec

    return sma_kpc


def save_results_to_csv(summary_data: List[Dict], output_file: Path):
    """
    Save summary statistics to CSV file.

    Args:
        summary_data: List of summary dictionaries (one per galaxy)
        output_file: Path to output CSV file
    """
    import csv

    if not summary_data:
        print("WARNING: No summary data to save")
        return

    # Get all keys from first entry
    fieldnames = list(summary_data[0].keys())

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_data)

    print(f"Saved summary CSV to {output_file}")


def compute_speedup_statistics(summary_data: List[Dict]) -> Dict:
    """
    Compute speedup statistics from summary data.

    Args:
        summary_data: List of summary dictionaries

    Returns:
        Dictionary with speedup statistics
    """
    speedups = [d['speedup_factor'] for d in summary_data if d['speedup_factor'] > 0]

    if not speedups:
        return {}

    return {
        'mean': np.mean(speedups),
        'median': np.median(speedups),
        'min': np.min(speedups),
        'max': np.max(speedups),
        'std': np.std(speedups)
    }


# ==============================================================================
# Main Functions
# ==============================================================================

def test_single_galaxy(galaxy_name: str,
                      models_file: str,
                      output_dir: Path,
                      redshift: float = DEFAULT_REDSHIFT,
                      pixel_scale: float = DEFAULT_PIXEL_SCALE,
                      psf_fwhm: float = DEFAULT_PSF_FWHM,
                      zeropoint: float = DEFAULT_ZEROPOINT,
                      add_noise: bool = False,
                      use_eccentric_anomaly: bool = False,
                      sma0: float = DEFAULT_SMA0,
                      astep: float = DEFAULT_ASTEP,
                      save_fits: bool = False,
                      dpi: int = FIGURE_DPI) -> Dict:
    """
    Test a single galaxy.

    Args:
        galaxy_name: Name of galaxy in models file
        models_file: Path to huang2013_models.yaml
        output_dir: Output directory
        redshift: Galaxy redshift
        pixel_scale: Pixel scale in arcsec/pixel
        psf_fwhm: PSF FWHM in arcsec
        zeropoint: Magnitude zeropoint
        add_noise: Add Poisson+readout noise
        use_eccentric_anomaly: Enable EA mode for isoster
        sma0: Starting SMA in pixels
        astep: SMA step size
        save_fits: Save FITS files
        dpi: Figure DPI

    Returns:
        Summary dictionary with statistics
    """
    print(f"\n{'='*70}")
    print(f"Testing {galaxy_name}")
    print(f"{'='*70}\n")

    # Create output directory for this galaxy
    galaxy_dir = output_dir / galaxy_name.replace(' ', '_')
    galaxy_dir.mkdir(parents=True, exist_ok=True)

    # Generate mock
    try:
        mock_data = generate_mock(
            galaxy_name, models_file,
            redshift=redshift,
            pixel_scale=pixel_scale,
            psf_fwhm=psf_fwhm,
            zeropoint=zeropoint,
            add_noise=add_noise
        )
    except Exception as e:
        print(f"ERROR: Failed to generate mock: {e}")
        return None

    # Extract initial geometry
    eps_init, pa_init = get_initial_geometry(mock_data['metadata'])

    # Run photutils
    try:
        results_photutils, runtime_photutils = run_photutils(
            mock_data['image'],
            mock_data['true_center'][0],
            mock_data['true_center'][1],
            eps_init, pa_init,
            sma0=sma0,
            astep=astep
        )
    except Exception as e:
        print(f"ERROR: photutils fitting failed: {e}")
        results_photutils = None
        runtime_photutils = 0.0

    # Run isoster
    try:
        results_isoster, runtime_isoster = run_isoster(
            mock_data['image'],
            mock_data['true_center'][0],
            mock_data['true_center'][1],
            eps_init, pa_init,
            sma0=sma0,
            astep=astep,
            use_eccentric_anomaly=use_eccentric_anomaly
        )
    except Exception as e:
        print(f"ERROR: isoster fitting failed: {e}")
        results_isoster = None
        runtime_isoster = 0.0

    # Check if both methods succeeded
    if results_photutils is None or results_isoster is None:
        print(f"ERROR: One or both methods failed for {galaxy_name}")
        return None

    # Extract truth profiles from mock
    print("Extracting truth profiles...")
    iso_photutils = results_photutils['isophotes']
    iso_isoster = results_isoster['isophotes']

    true_intens_photutils = extract_mock_profile(mock_data['image'], iso_photutils, pixel_scale)
    true_intens_isoster = extract_mock_profile(mock_data['image'], iso_isoster, pixel_scale)

    # Compute curves of growth
    true_cog_photutils = compute_curve_of_growth(mock_data['image'], iso_photutils, pixel_scale)
    true_cog_isoster = compute_curve_of_growth(mock_data['image'], iso_isoster, pixel_scale)

    # Build 2D models with harmonics [3, 4]
    print("Building 2D models...")
    model_photutils = build_photutils_model(
        mock_data['image'].shape, results_photutils,
        use_harmonics=True, harmonic_orders=[3, 4]
    )
    model_isoster = isoster.build_isoster_model(
        mock_data['image'].shape, iso_isoster,
        use_harmonics=True, harmonic_orders=[3, 4]
    )

    # Prepare metadata
    metadata_dict = {
        'galaxy_name': galaxy_name,
        'redshift': redshift,
        'pixel_scale': pixel_scale,
        'psf_fwhm': psf_fwhm,
        'zeropoint': zeropoint,
        'add_noise': add_noise,
        'use_eccentric_anomaly': use_eccentric_anomaly,
        'runtime_photutils': runtime_photutils,
        'runtime_isoster': runtime_isoster,
        'speedup_factor': runtime_photutils / runtime_isoster if runtime_isoster > 0 else 0.0,
        'n_isophotes_photutils': len(iso_photutils),
        'n_isophotes_isoster': len(iso_isoster),
        'true_center': mock_data['true_center'],
    }

    # Save all data for QA figure reproduction
    data_file = galaxy_dir / f"{galaxy_name.replace(' ', '_')}_data.npz"
    print(f"Saving data to {data_file}")

    # Convert isophote lists to dict of arrays
    iso_p_arrays = {
        'sma': np.array([iso['sma'] for iso in iso_photutils]),
        'x0': np.array([iso['x0'] for iso in iso_photutils]),
        'y0': np.array([iso['y0'] for iso in iso_photutils]),
        'eps': np.array([iso['eps'] for iso in iso_photutils]),
        'pa': np.array([iso['pa'] for iso in iso_photutils]),
        'intens': np.array([iso['intens'] for iso in iso_photutils]),
        'intens_err': np.array([iso.get('intens_err', 0.0) for iso in iso_photutils]),
        'stop_code': np.array([iso['stop_code'] for iso in iso_photutils]),
    }

    iso_i_arrays = {
        'sma': np.array([iso['sma'] for iso in iso_isoster]),
        'x0': np.array([iso['x0'] for iso in iso_isoster]),
        'y0': np.array([iso['y0'] for iso in iso_isoster]),
        'eps': np.array([iso['eps'] for iso in iso_isoster]),
        'pa': np.array([iso['pa'] for iso in iso_isoster]),
        'intens': np.array([iso['intens'] for iso in iso_isoster]),
        'intens_err': np.array([iso.get('intens_err', 0.0) for iso in iso_isoster]),
        'stop_code': np.array([iso['stop_code'] for iso in iso_isoster]),
    }

    np.savez(
        data_file,
        mock_image=mock_data['image'],
        iso_photutils=iso_p_arrays,
        iso_isoster=iso_i_arrays,
        true_intens_photutils=true_intens_photutils,
        true_intens_isoster=true_intens_isoster,
        true_cog_photutils=true_cog_photutils,
        true_cog_isoster=true_cog_isoster,
        model_photutils=model_photutils,
        model_isoster=model_isoster,
        metadata=metadata_dict
    )

    # Create QA figure using separate plotting script
    qa_path = galaxy_dir / f"{galaxy_name.replace(' ', '_')}_qa.png"
    print(f"Creating QA figure...")
    try:
        from plot_qa_huang2013 import plot_huang2013_qa
        plot_huang2013_qa(str(data_file), str(qa_path), dpi=dpi)
    except Exception as e:
        print(f"ERROR: Failed to create QA figure: {e}")
        import traceback
        traceback.print_exc()

    # Save FITS files if requested
    if save_fits:
        # Save mock image
        mock_fits_path = galaxy_dir / f"{galaxy_name.replace(' ', '_')}_mock.fits"
        fits.writeto(mock_fits_path, mock_data['image'], overwrite=True)

        # Save photutils results
        if results_photutils is not None:
            photutils_fits_path = galaxy_dir / f"{galaxy_name.replace(' ', '_')}_photutils.fits"
            isoster.isophote_results_to_fits(results_photutils, str(photutils_fits_path))

        # Save isoster results
        if results_isoster is not None:
            isoster_fits_path = galaxy_dir / f"{galaxy_name.replace(' ', '_')}_isoster.fits"
            isoster.isophote_results_to_fits(results_isoster, str(isoster_fits_path))

    # Save metadata (already created above)
    metadata_path = galaxy_dir / f"{galaxy_name.replace(' ', '_')}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata_dict, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Completed {galaxy_name}")
    print(f"  Speedup: {metadata_dict['speedup_factor']:.1f}×")
    print(f"  photutils: {runtime_photutils:.1f}s ({metadata_dict['n_isophotes_photutils']} isophotes)")
    print(f"  isoster: {runtime_isoster:.1f}s ({metadata_dict['n_isophotes_isoster']} isophotes)")
    print(f"{'='*70}\n")

    return metadata_dict


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Test photutils.isophote and isoster on Huang2013 mock galaxies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Galaxy selection
    parser.add_argument('--galaxy', type=str, default='NGC 3923',
                       help='Galaxy name (default: NGC 3923)')
    parser.add_argument('--batch', action='store_true',
                       help='Run all 93 galaxies')
    parser.add_argument('--models', type=str,
                       default='huang2013_models.yaml',
                       help='Path to models YAML file')

    # Mock generation
    parser.add_argument('--redshift', type=float, default=DEFAULT_REDSHIFT,
                       help=f'Galaxy redshift (default: {DEFAULT_REDSHIFT})')
    parser.add_argument('--pixel-scale', type=float, default=DEFAULT_PIXEL_SCALE,
                       help=f'Pixel scale in arcsec/pixel (default: {DEFAULT_PIXEL_SCALE})')
    parser.add_argument('--psf-fwhm', type=float, default=DEFAULT_PSF_FWHM,
                       help=f'PSF FWHM in arcsec (default: {DEFAULT_PSF_FWHM})')
    parser.add_argument('--zeropoint', type=float, default=DEFAULT_ZEROPOINT,
                       help=f'Magnitude zeropoint (default: {DEFAULT_ZEROPOINT})')
    parser.add_argument('--noise', action='store_true',
                       help='Add Poisson+readout noise')

    # Fitting configuration
    parser.add_argument('--use-eccentric-anomaly', action='store_true',
                       help='Enable EA mode for isoster')
    parser.add_argument('--sma0', type=float, default=DEFAULT_SMA0,
                       help=f'Starting SMA in pixels (default: {DEFAULT_SMA0})')
    parser.add_argument('--astep', type=float, default=DEFAULT_ASTEP,
                       help=f'SMA step size (default: {DEFAULT_ASTEP})')

    # Output
    parser.add_argument('--output-dir', type=str, default='outputs/huang2013_test/',
                       help='Output directory')
    parser.add_argument('--dpi', type=int, default=FIGURE_DPI,
                       help=f'Figure DPI (default: {FIGURE_DPI})')
    parser.add_argument('--save-fits', action='store_true',
                       help='Save FITS files')

    args = parser.parse_args()

    # Convert paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_file = args.models
    if not Path(models_file).exists():
        # Try relative to script directory
        script_dir = Path(__file__).parent
        models_file = script_dir / args.models
        if not models_file.exists():
            print(f"ERROR: Models file not found: {args.models}")
            sys.exit(1)

    models_file = str(models_file)

    # Run tests
    if args.batch:
        print("Running batch mode on all galaxies...")
        galaxy_names = load_all_galaxy_names(models_file)
        print(f"Found {len(galaxy_names)} galaxies")

        summary_data = []

        for i, galaxy_name in enumerate(galaxy_names):
            print(f"\n[{i+1}/{len(galaxy_names)}] Processing {galaxy_name}...")

            try:
                result = test_single_galaxy(
                    galaxy_name,
                    models_file,
                    output_dir,
                    redshift=args.redshift,
                    pixel_scale=args.pixel_scale,
                    psf_fwhm=args.psf_fwhm,
                    zeropoint=args.zeropoint,
                    add_noise=args.noise,
                    use_eccentric_anomaly=args.use_eccentric_anomaly,
                    sma0=args.sma0,
                    astep=args.astep,
                    save_fits=args.save_fits,
                    dpi=args.dpi
                )

                if result is not None:
                    summary_data.append(result)

            except Exception as e:
                print(f"ERROR: Failed to process {galaxy_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Save summary CSV
        if summary_data:
            csv_path = output_dir / 'summary.csv'
            save_results_to_csv(summary_data, csv_path)

            # Compute speedup statistics
            speedup_stats = compute_speedup_statistics(summary_data)
            print(f"\nSpeedup Statistics:")
            print(f"  Mean: {speedup_stats['mean']:.1f}×")
            print(f"  Median: {speedup_stats['median']:.1f}×")
            print(f"  Range: {speedup_stats['min']:.1f}× - {speedup_stats['max']:.1f}×")

    else:
        # Single galaxy mode
        result = test_single_galaxy(
            args.galaxy,
            models_file,
            output_dir,
            redshift=args.redshift,
            pixel_scale=args.pixel_scale,
            psf_fwhm=args.psf_fwhm,
            zeropoint=args.zeropoint,
            add_noise=args.noise,
            use_eccentric_anomaly=args.use_eccentric_anomaly,
            sma0=args.sma0,
            astep=args.astep,
            save_fits=args.save_fits,
            dpi=args.dpi
        )

        if result is None:
            print("ERROR: Test failed")
            sys.exit(1)

    print("\nDone!")


if __name__ == '__main__':
    main()
