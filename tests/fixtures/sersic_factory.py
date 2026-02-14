"""
Sersic model factory for test fixtures.

Provides a unified implementation for creating synthetic Sersic profile images
used across unit tests, integration tests, and benchmarks.

Per CLAUDE.md guidelines:
- Image half-size should be >= 10 * R_e (15x is better)
- High-Sersic index and high ellipticity require higher oversampling
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.special import gammaincinv


def compute_bn(n: float) -> float:
    """
    Compute the Sersic b_n parameter.

    Args:
        n: Sersic index

    Returns:
        b_n parameter value
    """
    return float(gammaincinv(2.0 * n, 0.5))


def sersic_1d(r: np.ndarray, R_e: float, n: float, I_e: float) -> np.ndarray:
    """
    Compute 1D Sersic profile intensity.

    Args:
        r: Array of radii
        R_e: Effective radius
        n: Sersic index
        I_e: Intensity at effective radius

    Returns:
        Array of intensities at each radius
    """
    b_n = compute_bn(n)
    return I_e * np.exp(-b_n * ((r / R_e) ** (1.0 / n) - 1.0))


@dataclass
class SersicModelResult:
    """Result container for Sersic model generation."""
    image: np.ndarray
    true_profile: Callable[[np.ndarray], np.ndarray]
    params: dict


def create_sersic_model(
    R_e: float,
    n: float,
    I_e: float,
    eps: float,
    pa: float,
    oversample: int = 1,
    noise_snr: Optional[float] = None,
    size_factor: float = 15.0,
    min_half_size: int = 150,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Callable, dict]:
    """
    Create a centered 2D Sersic profile image with optional noise.

    Per CLAUDE.md guidelines:
    - Image half-size should be >= 10 * R_e (15x is recommended)
    - High-Sersic index and high ellipticity require higher oversampling

    Args:
        R_e: Effective radius (pixels)
        n: Sersic index (1=exponential, 4=de Vaucouleurs)
        I_e: Intensity at effective radius
        eps: Ellipticity (1 - b/a), in range [0, 1)
        pa: Position angle (radians, counter-clockwise from x-axis)
        oversample: Oversampling factor for subpixel accuracy (default: 1)
        noise_snr: Signal-to-noise ratio at R_e. If None, no noise is added.
        size_factor: Image half-size as multiple of R_e (default: 15)
        min_half_size: Minimum half-size in pixels (default: 150)
        seed: Random seed for noise generation (default: None)

    Returns:
        Tuple of (image, true_profile, params) where:
        - image: 2D numpy array with the Sersic profile
        - true_profile: Function that computes true 1D intensity at any SMA
        - params: Dict with all model parameters including center coordinates
    """
    # Compute image size per CLAUDE.md guidelines
    half_size = max(int(size_factor * R_e), min_half_size)
    shape = (2 * half_size, 2 * half_size)
    x0, y0 = float(half_size), float(half_size)

    # Compute b_n parameter
    b_n = compute_bn(n)

    # Generate the image
    if oversample > 1:
        # Create oversampled grid with proper pixel centering
        y_hr = np.linspace(0, shape[0], shape[0] * oversample, endpoint=False) + 0.5 / oversample
        x_hr = np.linspace(0, shape[1], shape[1] * oversample, endpoint=False) + 0.5 / oversample
        yy_hr, xx_hr = np.meshgrid(y_hr, x_hr, indexing='ij')

        # Compute elliptical radius in rotated frame
        dx_hr = xx_hr - x0
        dy_hr = yy_hr - y0
        x_rot_hr = dx_hr * np.cos(pa) + dy_hr * np.sin(pa)
        y_rot_hr = -dx_hr * np.sin(pa) + dy_hr * np.cos(pa)
        r_ell_hr = np.sqrt(x_rot_hr**2 + (y_rot_hr / (1 - eps))**2)

        # Compute Sersic profile on oversampled grid
        image_hr = I_e * np.exp(-b_n * ((r_ell_hr / R_e) ** (1.0 / n) - 1.0))

        # Downsample by averaging (reshape and mean is faster than loop)
        image = image_hr.reshape(shape[0], oversample, shape[1], oversample).mean(axis=(1, 3))
    else:
        # Standard resolution
        y, x = np.mgrid[:shape[0], :shape[1]].astype(np.float64)
        dx = x - x0
        dy = y - y0
        x_rot = dx * np.cos(pa) + dy * np.sin(pa)
        y_rot = -dx * np.sin(pa) + dy * np.cos(pa)
        r_ell = np.sqrt(x_rot**2 + (y_rot / (1 - eps))**2)

        image = I_e * np.exp(-b_n * ((r_ell / R_e) ** (1.0 / n) - 1.0))

    # Add noise if requested
    if noise_snr is not None:
        rng = np.random.default_rng(seed)
        noise_level = I_e / noise_snr
        image = image + rng.normal(0, noise_level, image.shape)

    # Create true profile function
    def true_profile(sma: np.ndarray) -> np.ndarray:
        """Compute true 1D Sersic intensity at given semi-major axis values."""
        return I_e * np.exp(-b_n * ((sma / R_e) ** (1.0 / n) - 1.0))

    # Collect all parameters
    params = {
        'R_e': R_e,
        'n': n,
        'I_e': I_e,
        'eps': eps,
        'pa': pa,
        'b_n': b_n,
        'x0': x0,
        'y0': y0,
        'shape': shape,
        'oversample': oversample,
        'noise_snr': noise_snr,
    }

    return image, true_profile, params


class SersicFactory:
    """
    Factory class for creating Sersic test images with configurable defaults.

    This provides a convenient way to generate multiple test images with
    consistent base parameters while varying specific properties.

    Example:
        factory = SersicFactory(R_e=20, n=4, I_e=1000)

        # Create noiseless image
        img1, profile1, params1 = factory.create()

        # Create noisy image
        img2, profile2, params2 = factory.create(noise_snr=100)

        # Create high ellipticity version
        img3, profile3, params3 = factory.create(eps=0.7, oversample=10)
    """

    def __init__(
        self,
        R_e: float = 20.0,
        n: float = 4.0,
        I_e: float = 1000.0,
        eps: float = 0.0,
        pa: float = 0.0,
        oversample: int = 1,
        size_factor: float = 15.0,
        min_half_size: int = 150,
    ):
        """
        Initialize factory with default parameters.

        Args:
            R_e: Default effective radius
            n: Default Sersic index
            I_e: Default intensity at R_e
            eps: Default ellipticity
            pa: Default position angle
            oversample: Default oversampling factor
            size_factor: Default size factor (half-size = size_factor * R_e)
            min_half_size: Minimum half-size in pixels
        """
        self.defaults = {
            'R_e': R_e,
            'n': n,
            'I_e': I_e,
            'eps': eps,
            'pa': pa,
            'oversample': oversample,
            'size_factor': size_factor,
            'min_half_size': min_half_size,
        }

    def create(self, **kwargs) -> Tuple[np.ndarray, Callable, dict]:
        """
        Create a Sersic model image with optional parameter overrides.

        Args:
            **kwargs: Override any default parameters

        Returns:
            Tuple of (image, true_profile, params)
        """
        params = {**self.defaults, **kwargs}
        return create_sersic_model(**params)

    def create_circular(self, **kwargs) -> Tuple[np.ndarray, Callable, dict]:
        """Create a circular Sersic model (eps=0)."""
        return self.create(eps=0.0, pa=0.0, **kwargs)

    def create_elliptical(
        self,
        eps: float = 0.4,
        pa: float = 0.785,  # π/4
        **kwargs
    ) -> Tuple[np.ndarray, Callable, dict]:
        """Create an elliptical Sersic model with given geometry."""
        return self.create(eps=eps, pa=pa, **kwargs)

    def create_noisy(
        self,
        noise_snr: float = 100.0,
        seed: int = 42,
        **kwargs
    ) -> Tuple[np.ndarray, Callable, dict]:
        """Create a Sersic model with Gaussian noise."""
        return self.create(noise_snr=noise_snr, seed=seed, **kwargs)
