"""
Sersic model generation utilities for benchmarking isophote fitting.

This module provides functions to create 2D Sersic profile images with
proper central region oversampling and optional noise.
"""

import numpy as np
from scipy.special import gammaincinv


def compute_bn(n):
    """
    Compute the Sersic b_n parameter for a given Sersic index.

    The b_n parameter ensures that R_e encloses half of the total luminosity.
    Uses the exact solution via incomplete gamma function inversion.

    Parameters
    ----------
    n : float
        Sersic index.

    Returns
    -------
    float
        The b_n parameter.
    """
    # Exact solution: b_n = gammaincinv(2n, 0.5)
    # For n >= 0.36, this is accurate
    return gammaincinv(2 * n, 0.5)


def sersic_1d(r, I_e, R_e, n):
    """
    Compute the 1D Sersic intensity profile.

    I(r) = I_e * exp(-b_n * ((r/R_e)^(1/n) - 1))

    Parameters
    ----------
    r : float or array
        Radial distance (semi-major axis for elliptical profiles).
    I_e : float
        Intensity at the effective radius.
    R_e : float
        Effective radius.
    n : float
        Sersic index (n=1 exponential, n=4 de Vaucouleurs).

    Returns
    -------
    float or array
        Intensity at radius r.
    """
    b_n = compute_bn(n)
    return I_e * np.exp(-b_n * ((r / R_e) ** (1.0 / n) - 1))


def create_sersic_image(n, R_e, I_e, eps, pa, shape, center=None,
                        oversample=10, oversample_radius=5.0):
    """
    Create a 2D Sersic profile image with central oversampling.

    Parameters
    ----------
    n : float
        Sersic index.
    R_e : float
        Effective radius in pixels.
    I_e : float
        Intensity at the effective radius.
    eps : float
        Ellipticity (1 - b/a).
    pa : float
        Position angle in radians (counter-clockwise from x-axis).
    shape : tuple
        Image shape (height, width).
    center : tuple, optional
        Center coordinates (x0, y0). Default is image center.
    oversample : int
        Oversampling factor for central region (e.g., 10 means 10x10 subpixels).
    oversample_radius : float
        Radius in pixels within which to apply oversampling.

    Returns
    -------
    image : ndarray
        2D Sersic image.
    params : dict
        Dictionary with true parameters for reference.
    """
    h, w = shape

    if center is None:
        x0, y0 = w / 2.0, h / 2.0
    else:
        x0, y0 = center

    image = np.zeros((h, w), dtype=np.float64)

    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)
    q = 1.0 - eps  # axis ratio b/a

    # Process each pixel
    for iy in range(h):
        for ix in range(w):
            # Distance from center
            dist = np.sqrt((ix - x0)**2 + (iy - y0)**2)

            if dist < oversample_radius:
                # Oversample this pixel
                subpixel_sum = 0.0
                for sy in range(oversample):
                    for sx in range(oversample):
                        # Subpixel coordinates (center of subpixel)
                        sub_x = ix - 0.5 + (sx + 0.5) / oversample
                        sub_y = iy - 0.5 + (sy + 0.5) / oversample

                        # Transform to ellipse frame
                        dx = sub_x - x0
                        dy = sub_y - y0

                        # Rotate to align with major axis
                        x_rot = dx * cos_pa + dy * sin_pa
                        y_rot = -dx * sin_pa + dy * cos_pa

                        # Elliptical radius (semi-major axis equivalent)
                        r = np.sqrt(x_rot**2 + (y_rot / q)**2)

                        # Compute intensity (avoid r=0 singularity for high n)
                        if r < 1e-6:
                            r = 1e-6
                        subpixel_sum += sersic_1d(r, I_e, R_e, n)

                image[iy, ix] = subpixel_sum / (oversample * oversample)
            else:
                # No oversampling needed
                dx = ix - x0
                dy = iy - y0

                x_rot = dx * cos_pa + dy * sin_pa
                y_rot = -dx * sin_pa + dy * cos_pa

                r = np.sqrt(x_rot**2 + (y_rot / q)**2)
                if r < 1e-6:
                    r = 1e-6

                image[iy, ix] = sersic_1d(r, I_e, R_e, n)

    params = {
        'n': n,
        'R_e': R_e,
        'I_e': I_e,
        'eps': eps,
        'pa': pa,
        'x0': x0,
        'y0': y0,
        'shape': shape
    }

    return image, params


def create_sersic_image_vectorized(n, R_e, I_e, eps, pa, shape, center=None,
                                   oversample=10, oversample_radius=5.0):
    """
    Create a 2D Sersic profile image with central oversampling (vectorized version).

    This is a faster version using numpy vectorization.

    Parameters
    ----------
    n : float
        Sersic index.
    R_e : float
        Effective radius in pixels.
    I_e : float
        Intensity at the effective radius.
    eps : float
        Ellipticity (1 - b/a).
    pa : float
        Position angle in radians (counter-clockwise from x-axis).
    shape : tuple
        Image shape (height, width).
    center : tuple, optional
        Center coordinates (x0, y0). Default is image center.
    oversample : int
        Oversampling factor for central region.
    oversample_radius : float
        Radius in pixels within which to apply oversampling.

    Returns
    -------
    image : ndarray
        2D Sersic image.
    params : dict
        Dictionary with true parameters for reference.
    """
    h, w = shape

    if center is None:
        x0, y0 = w / 2.0, h / 2.0
    else:
        x0, y0 = center

    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)
    q = 1.0 - eps  # axis ratio b/a

    # Create coordinate grid
    y, x = np.mgrid[:h, :w].astype(np.float64)

    # Distance from center for each pixel
    dist = np.sqrt((x - x0)**2 + (y - y0)**2)

    # Identify pixels needing oversampling
    needs_oversample = dist < oversample_radius

    # Process non-oversampled pixels (vectorized)
    dx = x - x0
    dy = y - y0
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    r = np.sqrt(x_rot**2 + (y_rot / q)**2)
    r = np.maximum(r, 1e-6)  # Avoid singularity

    image = sersic_1d(r, I_e, R_e, n)

    # Process oversampled pixels
    oversample_indices = np.where(needs_oversample)
    for iy, ix in zip(*oversample_indices):
        subpixel_sum = 0.0
        for sy in range(oversample):
            for sx in range(oversample):
                sub_x = ix - 0.5 + (sx + 0.5) / oversample
                sub_y = iy - 0.5 + (sy + 0.5) / oversample

                dx_sub = sub_x - x0
                dy_sub = sub_y - y0

                x_rot_sub = dx_sub * cos_pa + dy_sub * sin_pa
                y_rot_sub = -dx_sub * sin_pa + dy_sub * cos_pa

                r_sub = np.sqrt(x_rot_sub**2 + (y_rot_sub / q)**2)
                if r_sub < 1e-6:
                    r_sub = 1e-6

                subpixel_sum += sersic_1d(r_sub, I_e, R_e, n)

        image[iy, ix] = subpixel_sum / (oversample * oversample)

    params = {
        'n': n,
        'R_e': R_e,
        'I_e': I_e,
        'eps': eps,
        'pa': pa,
        'x0': x0,
        'y0': y0,
        'shape': shape
    }

    return image, params


def add_noise(image, snr_at_Re, R_e, I_e, seed=None):
    """
    Add Gaussian noise to an image with specified S/N at the effective radius.

    Parameters
    ----------
    image : ndarray
        Input image.
    snr_at_Re : float
        Signal-to-noise ratio at the effective radius.
    R_e : float
        Effective radius (for reference).
    I_e : float
        Intensity at effective radius (used to compute noise level).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    noisy_image : ndarray
        Image with added Gaussian noise.
    noise_sigma : float
        Standard deviation of the added noise.
    """
    if seed is not None:
        np.random.seed(seed)

    # Noise sigma such that S/N = I_e / sigma at R_e
    noise_sigma = I_e / snr_at_Re

    noise = np.random.normal(0, noise_sigma, image.shape)
    noisy_image = image + noise

    return noisy_image, noise_sigma


def get_true_profile_at_sma(sma_values, params):
    """
    Compute true Sersic profile at given semi-major axis values.

    Parameters
    ----------
    sma_values : array
        Semi-major axis values.
    params : dict
        Parameters from create_sersic_image.

    Returns
    -------
    dict
        Dictionary with 'sma', 'intens', 'eps', 'pa' arrays.
    """
    intens = sersic_1d(sma_values, params['I_e'], params['R_e'], params['n'])

    return {
        'sma': np.array(sma_values),
        'intens': intens,
        'eps': np.full_like(sma_values, params['eps']),
        'pa': np.full_like(sma_values, params['pa'])
    }


if __name__ == '__main__':
    # Quick test
    import matplotlib.pyplot as plt

    # Create a de Vaucouleurs profile (n=4)
    image, params = create_sersic_image_vectorized(
        n=4.0, R_e=50, I_e=1000, eps=0.3, pa=np.pi/4,
        shape=(256, 256), oversample=10, oversample_radius=5.0
    )

    print(f"Image shape: {image.shape}")
    print(f"Max intensity: {image.max():.2f}")
    print(f"Intensity at center: {image[128, 128]:.2f}")

    # Check radial profile
    r_test = np.array([10, 25, 50, 100])
    true_I = sersic_1d(r_test, params['I_e'], params['R_e'], params['n'])
    print(f"\nTrue intensity at r={r_test}: {true_I}")

    plt.figure(figsize=(10, 4))

    plt.subplot(121)
    plt.imshow(np.log10(image + 1), origin='lower', cmap='viridis')
    plt.colorbar(label='log10(I+1)')
    plt.title(f"Sersic n={params['n']}, eps={params['eps']:.1f}")

    plt.subplot(122)
    # Extract radial profile along major axis
    cx, cy = int(params['x0']), int(params['y0'])
    profile = image[cy, cx:]
    r_pix = np.arange(len(profile))
    plt.semilogy(r_pix, profile, 'b-', label='Image')
    plt.semilogy(r_pix, sersic_1d(r_pix + 0.5, params['I_e'], params['R_e'], params['n']),
                 'r--', label='True 1D')
    plt.axvline(params['R_e'], color='g', linestyle=':', label=f"R_e={params['R_e']}")
    plt.xlabel('Radius (pixels)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.title('Radial Profile')

    plt.tight_layout()
    plt.savefig('sersic_test.png', dpi=100)
    print("\nSaved test image to sersic_test.png")
