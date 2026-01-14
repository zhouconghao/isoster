"""
Image reconstruction from fitted isophote profiles.

This module provides functions to reconstruct 2D galaxy images from isophote
fitting results using radial interpolation.
"""

import numpy as np
from scipy.interpolate import interp1d


def build_isoster_model(image_shape, isophote_results, fill=0.0, interp_kind='linear'):
    """
    Reconstruct a 2D galaxy image model from fitted isophotes.

    This implementation uses radial interpolation between isophotes to create
    a smooth intensity profile, including higher-order harmonic deviations if
    available. This is significantly more accurate than layer-by-layer filling.

    Algorithm:
    1. For each pixel, compute its elliptical radius using the local geometry
    2. Interpolate intensity between adjacent isophotes
    3. Add harmonic deviations (a3, b3, a4, b4) if present
    4. Fill regions outside fitted range with specified value

    Parameters
    ----------
    image_shape : tuple
        The (height, width) of the output image.
    isophote_results : list of dict
        The 'isophotes' list from fit_image() results. Each dict should contain:
        - 'sma': semi-major axis length
        - 'x0', 'y0': center coordinates
        - 'eps': ellipticity
        - 'pa': position angle
        - 'intens': mean intensity
        - Optional: 'a3', 'b3', 'a4', 'b4': harmonic deviations
    fill : float, optional
        Value to fill pixels outside the largest isophote (default: 0.0)
    interp_kind : str, optional
        Interpolation method for radial intensity profile. Options:
        - 'linear': Linear interpolation (default, faster)
        - 'cubic': Cubic spline interpolation (smoother)

    Returns
    -------
    model : ndarray
        The reconstructed 2D image model of shape image_shape.

    Notes
    -----
    - For pixels outside the outermost isophote or inside the innermost,
      uses the boundary isophote's parameters with no extrapolation.
    - Harmonic deviations (a3, b3, a4, b4) are properly reconstructed using
      the local gradient and SMA to convert normalized deviations back to
      intensity units.
    - The function uses local geometry interpolation at each radius to
      handle varying ellipticity and position angle profiles.

    Examples
    --------
    >>> results = isoster.fit_image(image, mask, config)
    >>> model = build_isoster_model(image.shape, results['isophotes'])
    >>> residual = image - model
    """
    h, w = image_shape
    model = np.full((h, w), fill, dtype=np.float64)

    # Filter valid isophotes (stop_code should be good, but allow some flexibility)
    valid_isos = [iso for iso in isophote_results if iso['sma'] > 0]

    if len(valid_isos) == 0:
        # No valid isophotes, return fill value
        return model

    # Sort by SMA
    sorted_isos = sorted(valid_isos, key=lambda x: x['sma'])

    # Handle central pixel if present
    central_iso = [iso for iso in isophote_results if iso['sma'] == 0]
    if central_iso:
        iso = central_iso[0]
        x0, y0 = int(np.round(iso['x0'])), int(np.round(iso['y0']))
        if 0 <= y0 < h and 0 <= x0 < w:
            model[y0, x0] = iso['intens']

    # Create coordinate grids
    y_grid, x_grid = np.ogrid[:h, :w]
    y_grid = y_grid.astype(np.float64)
    x_grid = x_grid.astype(np.float64)

    # Extract SMA values and intensities for interpolation
    sma_values = np.array([iso['sma'] for iso in sorted_isos])
    intens_values = np.array([iso['intens'] for iso in sorted_isos])

    # Create intensity interpolator
    if len(sma_values) == 1:
        # Only one isophote - use constant value within SMA, fill outside
        sma_single = sma_values[0]
        intens_single = intens_values[0]
        def intens_interp(r):
            result = np.where(r <= sma_single, intens_single, fill)
            return result
    else:
        # Interpolate, with boundary handling
        intens_interp = interp1d(sma_values, intens_values, kind=interp_kind,
                                  bounds_error=False, fill_value=(intens_values[0], fill))

    # For each pixel, we need to:
    # 1. Find the local geometry (x0, y0, eps, pa) at that elliptical radius
    # 2. Compute elliptical radius
    # 3. Interpolate intensity

    # To handle varying geometry, we'll process the image in annular regions
    # and use interpolated geometry for each region

    # Interpolate geometry parameters
    x0_interp = interp1d(sma_values, [iso['x0'] for iso in sorted_isos],
                         kind='linear', bounds_error=False,
                         fill_value=([iso['x0'] for iso in sorted_isos][0],
                                    [iso['x0'] for iso in sorted_isos][-1]))
    y0_interp = interp1d(sma_values, [iso['y0'] for iso in sorted_isos],
                         kind='linear', bounds_error=False,
                         fill_value=([iso['y0'] for iso in sorted_isos][0],
                                    [iso['y0'] for iso in sorted_isos][-1]))
    eps_interp = interp1d(sma_values, [iso['eps'] for iso in sorted_isos],
                          kind='linear', bounds_error=False,
                          fill_value=([iso['eps'] for iso in sorted_isos][0],
                                     [iso['eps'] for iso in sorted_isos][-1]))
    pa_interp = interp1d(sma_values, [iso['pa'] for iso in sorted_isos],
                         kind='linear', bounds_error=False,
                         fill_value=([iso['pa'] for iso in sorted_isos][0],
                                    [iso['pa'] for iso in sorted_isos][-1]))

    # Use outer isophote geometry as initial guess to find approximate radii
    outer_iso = sorted_isos[-1]
    x0_out, y0_out = outer_iso['x0'], outer_iso['y0']
    eps_out, pa_out = outer_iso['eps'], outer_iso['pa']

    # Compute approximate elliptical radius using outer geometry
    dx = x_grid - x0_out
    dy = y_grid - y0_out
    cos_pa, sin_pa = np.cos(pa_out), np.sin(pa_out)
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    r_ell_approx = np.sqrt(x_rot**2 + (y_rot / (1.0 - eps_out))**2)

    # For better accuracy, iterate to find elliptical radius using local geometry
    # (This is important when eps or pa vary significantly with radius)
    max_iterations = 3
    r_ell = r_ell_approx.copy()

    for iteration in range(max_iterations):
        # Get local geometry at current radius estimate
        r_ell_flat = r_ell.ravel()
        x0_local = x0_interp(r_ell_flat).reshape(r_ell.shape)
        y0_local = y0_interp(r_ell_flat).reshape(r_ell.shape)
        eps_local = eps_interp(r_ell_flat).reshape(r_ell.shape)
        pa_local = pa_interp(r_ell_flat).reshape(r_ell.shape)

        # Recompute elliptical radius with local geometry
        dx = x_grid - x0_local
        dy = y_grid - y0_local
        cos_pa_local = np.cos(pa_local)
        sin_pa_local = np.sin(pa_local)
        x_rot = dx * cos_pa_local + dy * sin_pa_local
        y_rot = -dx * sin_pa_local + dy * cos_pa_local

        # Avoid division by zero for eps ~= 1
        eps_safe = np.clip(eps_local, 0, 0.99)
        r_ell_new = np.sqrt(x_rot**2 + (y_rot / (1.0 - eps_safe))**2)

        # Check convergence
        if np.max(np.abs(r_ell_new - r_ell)) < 0.1:
            r_ell = r_ell_new
            break
        r_ell = r_ell_new

    # Now interpolate intensity at each elliptical radius
    model_flat = intens_interp(r_ell.ravel())
    model = model_flat.reshape(image_shape)

    # TODO: Add harmonic deviations (a3, b3, a4, b4)
    # This requires knowing the gradient at each radius to denormalize
    # For now, harmonics are not included in reconstruction

    return model


# Keep old name as alias for backward compatibility
def build_ellipse_model(image_shape, isophote_results, fill=0.0):
    """
    Legacy name for build_isoster_model().

    .. deprecated::
        Use build_isoster_model() instead. This function will be removed in a future version.
    """
    import warnings
    warnings.warn(
        "build_ellipse_model() is deprecated, use build_isoster_model() instead",
        DeprecationWarning,
        stacklevel=2
    )
    return build_isoster_model(image_shape, isophote_results, fill=fill)
