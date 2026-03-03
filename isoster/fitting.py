import warnings

import numpy as np
from scipy.optimize import leastsq
from .sampling import extract_isophote_data
from .config import IsosterConfig

# Import numba-accelerated kernels (with numpy fallback)
from .numba_kernels import (
    harmonic_model,
    build_harmonic_matrix,
    NUMBA_AVAILABLE
)

def compute_central_regularization_penalty(current_geom, previous_geom, sma, config):
    """
    Compute regularization penalty for geometry changes in central region.
    
    Adds penalty to discourage large geometry changes at low SMA, helping stabilize
    fits in low S/N central regions.
    
    Args:
        current_geom (dict): Current geometry {'x0', 'y0', 'eps', 'pa'}
        previous_geom (dict): Previous isophote geometry (or None)
        sma (float): Current semi-major axis
        config (IsosterConfig): Configuration with regularization parameters
        
    Returns:
        float: Regularization penalty value
    """
    if not config.use_central_regularization:
        return 0.0
    
    if previous_geom is None:
        return 0.0
    
    # Regularization strength decays exponentially from center
    # λ(sma) = max_strength * exp(-(sma/threshold)²)
    lambda_sma = config.central_reg_strength * np.exp(
        -(sma / config.central_reg_sma_threshold)**2
    )
    
    # No regularization beyond 3× threshold
    if lambda_sma < 1e-6:
        return 0.0
    
    weights = config.central_reg_weights
    
    # Compute changes from previous isophote
    delta_eps = current_geom['eps'] - previous_geom['eps']
    delta_pa = current_geom['pa'] - previous_geom['pa']
    
    # Handle PA wrap-around (force to [-π, π])
    delta_pa = ((delta_pa + np.pi) % (2 * np.pi)) - np.pi
    
    delta_x0 = current_geom['x0'] - previous_geom['x0']
    delta_y0 = current_geom['y0'] - previous_geom['y0']
    
    # Penalty: λ(sma) * weighted sum of squared changes
    penalty = lambda_sma * (
        weights.get('eps', 1.0) * delta_eps**2 +
        weights.get('pa', 1.0) * delta_pa**2 +
        weights.get('center', 1.0) * (delta_x0**2 + delta_y0**2)
    )
    
    return penalty

def extract_forced_photometry(image, mask, x0, y0, sma, eps, pa, integrator='mean', sclip=3.0, nclip=0, use_eccentric_anomaly=False, config=None):
    """
    Extract forced photometry at a single SMA without fitting.

    This is used for template-based forced mode where geometry is predetermined.

    Args:
        image (np.ndarray): Input image.
        mask (np.ndarray): Bad pixel mask.
        x0, y0 (float): Center coordinates.
        sma (float): Semi-major axis length.
        eps (float): Ellipticity.
        pa (float): Position angle in radians.
        integrator (str): 'mean' or 'median'.
        sclip (float): Sigma clipping threshold.
        nclip (int): Number of sigma clipping iterations.
        use_eccentric_anomaly (bool): Whether to use eccentric anomaly sampling mode.
        config (IsosterConfig, optional): Configuration object. When provided,
            harmonic keys are only included if compute_deviations or
            simultaneous_harmonics is enabled, using config.harmonic_orders
            for key names. When None, harmonics default to [3, 4] for
            backward compatibility.

    Returns:
        dict: Isophote structure with intensity from the target image.
    """
    # Determine whether to include harmonic keys and which orders
    if config is not None:
        include_harmonics = config.compute_deviations or config.simultaneous_harmonics
        harmonic_orders = config.harmonic_orders if include_harmonics else []
    else:
        # Backward compatibility: include default [3, 4] when called without config
        include_harmonics = True
        harmonic_orders = [3, 4]

    def _build_harmonic_fields(orders):
        """Build zero-valued harmonic key-value pairs for given orders."""
        fields = {}
        for order in orders:
            fields[f'a{order}'] = 0.0
            fields[f'b{order}'] = 0.0
            fields[f'a{order}_err'] = 0.0
            fields[f'b{order}_err'] = 0.0
        return fields

    # Sample along the ellipse
    data = extract_isophote_data(image, mask, x0, y0, sma, eps, pa, use_eccentric_anomaly=use_eccentric_anomaly)
    phi = data.angles  # position angle or eccentric anomaly depending on mode
    intens = data.intens

    if len(intens) == 0:
        result = {
            'x0': x0, 'y0': y0, 'eps': eps, 'pa': pa, 'sma': sma,
            'intens': np.nan, 'rms': np.nan, 'intens_err': np.nan,
            'x0_err': 0.0, 'y0_err': 0.0, 'eps_err': 0.0, 'pa_err': 0.0,
            'tflux_e': np.nan, 'tflux_c': np.nan, 'npix_e': 0, 'npix_c': 0,
            'stop_code': 3, 'niter': 0, 'valid': False
        }
        if include_harmonics:
            result.update(_build_harmonic_fields(harmonic_orders))
        return result

    # Sigma clipping
    if nclip > 0:
        phi, intens, _ = sigma_clip(phi, intens, sclip, nclip)

    # Compute intensity
    if integrator == 'median':
        intensity = np.median(intens)
    else:
        intensity = np.mean(intens)

    rms = np.std(intens)
    intens_err = rms / np.sqrt(len(intens))

    result = {
        'x0': x0, 'y0': y0, 'eps': eps, 'pa': pa, 'sma': sma,
        'intens': intensity, 'rms': rms, 'intens_err': intens_err,
        'x0_err': 0.0, 'y0_err': 0.0, 'eps_err': 0.0, 'pa_err': 0.0,
        'tflux_e': np.nan, 'tflux_c': np.nan, 'npix_e': 0, 'npix_c': 0,
        'stop_code': 0, 'niter': 0, 'valid': True
    }
    if include_harmonics:
        result.update(_build_harmonic_fields(harmonic_orders))
    return result

def fit_first_and_second_harmonics(phi, intensity):
    """
    Fit the 1st and 2nd harmonics to the intensity profile.

    The model is:
    y = y0 + A1*sin(E) + B1*cos(E) + A2*sin(2E) + B2*cos(2E)

    Args:
        phi (np.ndarray): eccentric anomaly angles.
        intensity (np.ndarray): sampled intensity values.

    Returns:
        tuple: (coeffs, ata_inv)
            coeffs: array of [y0, A1, B1, A2, B2]
            ata_inv: inverse covariance matrix (A^T A)^-1
    """
    # Use numba-accelerated design matrix construction
    A = build_harmonic_matrix(phi)

    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(A, intensity, rcond=None)

        # Compute covariance matrix (A^T * A)^-1
        # This is used for parameter error estimation later
        ata_inv = np.linalg.inv(np.dot(A.T, A))
        return coeffs, ata_inv
    except np.linalg.LinAlgError:
        return np.array([np.mean(intensity), 0.0, 0.0, 0.0, 0.0]), None

def harmonic_function(phi, coeffs):
    """
    Evaluate harmonic model at given angles.

    Uses numba-accelerated implementation when available for better performance.
    """
    return harmonic_model(phi, coeffs)

def sigma_clip(phi, intens, sclip=3.0, nclip=0, sclip_low=None, sclip_high=None, extra_arrays=None):
    """Perform iterative sigma clipping on intensity data.

    Args:
        phi: Angle array to clip in parallel with intens.
        intens: Intensity array used for computing clip thresholds.
        sclip: Symmetric sigma threshold (used if sclip_low/sclip_high not set).
        nclip: Number of clipping iterations (0 = no clipping).
        sclip_low: Lower sigma threshold (overrides sclip).
        sclip_high: Upper sigma threshold (overrides sclip).
        extra_arrays: Optional list of additional arrays to clip with the same mask.
            Each must have the same length as phi/intens.

    Returns:
        tuple: (phi_clipped, intens_clipped, total_clipped) when extra_arrays is None.
            (phi_clipped, intens_clipped, total_clipped, *extra_clipped) when extra_arrays
            is provided.
    """
    if nclip <= 0:
        if extra_arrays:
            return phi, intens, 0, *extra_arrays
        return phi, intens, 0

    s_low = sclip_low if sclip_low is not None else sclip
    s_high = sclip_high if sclip_high is not None else sclip

    phi_c = phi.copy()
    intens_c = intens.copy()
    extras_c = [a.copy() for a in extra_arrays] if extra_arrays else []
    total_clipped = 0

    for _ in range(nclip):
        if len(intens_c) < 3:
            break
        mean = np.mean(intens_c)
        std = np.std(intens_c)

        lower = mean - s_low * std
        upper = mean + s_high * std

        mask = (intens_c >= lower) & (intens_c <= upper)
        n_clipped = len(intens_c) - np.sum(mask)

        if n_clipped == 0:
            break

        total_clipped += n_clipped
        phi_c = phi_c[mask]
        intens_c = intens_c[mask]
        extras_c = [a[mask] for a in extras_c]

    if extras_c:
        return phi_c, intens_c, total_clipped, *extras_c
    return phi_c, intens_c, total_clipped

def compute_parameter_errors(phi, intens, x0, y0, sma, eps, pa, gradient, gradient_error=None, cov_matrix=None, coeffs=None, var_residual_floor=None):
    """Compute parameter errors based on the covariance matrix of harmonic coefficients.

    Args:
        phi: Angles array
        intens: Intensity array
        x0, y0, sma, eps, pa: Geometry parameters
        gradient: Intensity gradient
        gradient_error: Uncertainty in the intensity gradient (optional)
        cov_matrix: Covariance matrix from harmonic fit (optional)
        coeffs: Harmonic coefficients [y0, A1, B1, A2, B2] (optional, avoids re-fitting)
        var_residual_floor: Minimum variance for residuals (optional, e.g. sigma_bg^2)

    Returns:
        Tuple of (x0_err, y0_err, eps_err, pa_err)
    """
    # Guard against zero/None gradient to prevent division errors
    if gradient is None or abs(gradient) < 1e-10:
        return 0.0, 0.0, 0.0, 0.0
    # Guard against too few data points for the harmonic model (5 params)
    if len(intens) <= 5:
        return 0.0, 0.0, 0.0, 0.0
    
    g_err_sq = gradient_error**2 if gradient_error is not None else 0.0
    g_sq = gradient**2

    try:
        # 1. Determine coefficients if not provided
        if coeffs is None:
            if cov_matrix is not None:
                # We have covariance but no coeffs? This shouldn't happen in EFF-2 path
                # but we'll re-fit just in case.
                coeffs, _ = fit_first_and_second_harmonics(phi, intens)
            else:
                # No covariance, no coeffs: use fit_first_and_second_harmonics
                coeffs, cov_matrix = fit_first_and_second_harmonics(phi, intens)

        # 2. Determine residual variance
        n_params = 5
        model = harmonic_function(phi, coeffs)
        var_residual = np.var(intens - model, ddof=n_params)
        if var_residual_floor is not None:
            var_residual = max(var_residual, var_residual_floor)

        # 3. Scale covariance matrix by residual variance to get parameter covariance
        if cov_matrix is None:
            return 0.0, 0.0, 0.0, 0.0
            
        covariance = cov_matrix * var_residual
        errors = np.sqrt(np.diagonal(covariance))
        
        # Parameter error formulas (Corrected to include gradient uncertainty)
        # 1. Harmonic coefficient variances
        sig_a1_sq, sig_b1_sq = errors[1]**2, errors[2]**2
        sig_a2_sq, sig_b2_sq = errors[3]**2, errors[4]**2
        
        # 2. Geometric harmonic amplitudes (A1, B1, A2, B2)
        # indexing matches fit_first_and_second_harmonics: [y0, A1, B1, A2, B2]
        a1, b1 = coeffs[1], coeffs[2]
        a2, b2 = coeffs[3], coeffs[4]

        # 3. Propagated variances for major/minor axis shifts
        # Major axis: Var(B1/g) = (1/g^2) * [Var(B1) + (B1/g)^2 * Var(g)]
        var_major = (sig_b1_sq + (b1**2 / g_sq) * g_err_sq) / g_sq
        # Minor axis: Var(A1*(1-eps)/g) = ((1-eps)^2/g^2) * [Var(A1) + (A1/g)^2 * Var(g)]
        var_minor = ((1.0 - eps)**2 / g_sq) * (sig_a1_sq + (a1**2 / g_sq) * g_err_sq)
        
        x0_err = np.sqrt((var_minor * np.sin(pa)**2) + (var_major * np.cos(pa)**2))
        y0_err = np.sqrt((var_minor * np.cos(pa)**2) + (var_major * np.sin(pa)**2))
        
        # 4. Ellipticity variance
        # Var(eps) = Var(2*(1-eps)*B2 / (a*g))
        #          = (2*(1-eps)/(a*g))^2 * [Var(B2) + (B2/g)^2 * Var(g)]
        var_eps = (2.0 * (1.0 - eps) / (sma * gradient))**2 * (sig_b2_sq + (b2**2 / g_sq) * g_err_sq)
        eps_err = np.sqrt(var_eps)
        
        # 5. Position angle variance
        if abs(eps) > np.finfo(float).resolution:
            # denom = (1-eps)^2 - 1
            denom = (1.0 - eps)**2 - 1.0
            if abs(denom) < 1e-10: denom = -1e-10
            
            # Var(PA) = Var(2*(1-eps)*A2 / (a*g*denom))
            #         = (2*(1-eps)/(a*g*denom))^2 * [Var(A2) + (A2/g)^2 * Var(g)]
            var_pa = (2.0 * (1.0 - eps) / (sma * gradient * denom))**2 * (sig_a2_sq + (a2**2 / g_sq) * g_err_sq)
            pa_err = np.sqrt(var_pa)
        else:
            pa_err = 0.0
            
        return x0_err, y0_err, eps_err, pa_err
    except (np.linalg.LinAlgError, ValueError) as e:
        # Singular matrix or numerical instability - return zero errors
        warnings.warn(
            f"compute_parameter_errors failed (singular matrix or numerical issue): {e}. "
            f"Returning zero errors.",
            RuntimeWarning
        )
        return 0.0, 0.0, 0.0, 0.0
    except (np.linalg.LinAlgError, ValueError) as e:
        # Singular matrix or numerical instability - return zero errors
        warnings.warn(
            f"compute_parameter_errors failed (singular matrix or numerical issue): {e}. "
            f"Returning zero errors.",
            RuntimeWarning
        )
        return 0.0, 0.0, 0.0, 0.0
    # Note: Removed broad 'except Exception' - unexpected errors should be raised for debugging

def compute_deviations(phi, intens, sma, gradient, order):
    """Compute deviations from perfect ellipticity (higher order harmonics)."""
    try:
        s_n = np.sin(order * phi)
        c_n = np.cos(order * phi)
        y0_init = np.mean(intens)
        params_init = [y0_init, 0.0, 0.0]
        
        def residual(params):
            model = params[0] + params[1]*s_n + params[2]*c_n
            return intens - model
            
        solution = leastsq(residual, params_init, full_output=True)
        coeffs = solution[0]
        cov_matrix = solution[1]
        
        if cov_matrix is None:
            return 0.0, 0.0, 0.0, 0.0
            
        model = coeffs[0] + coeffs[1]*s_n + coeffs[2]*c_n
        if len(intens) <= len(coeffs):
            return 0.0, 0.0, 0.0, 0.0
        var_residual = np.std(intens - model, ddof=len(coeffs))**2
        covariance = cov_matrix * var_residual
        errors = np.sqrt(np.diagonal(covariance))
        
        factor = sma * abs(gradient)
        if factor == 0:
            return 0.0, 0.0, 0.0, 0.0
            
        a = coeffs[1] / factor
        b = coeffs[2] / factor
        a_err = errors[1] / factor
        b_err = errors[2] / factor
        
        return a, b, a_err, b_err
    except (np.linalg.LinAlgError, ValueError, TypeError) as e:
        # Singular matrix, numerical instability, or degenerate input - return zeros
        # TypeError can occur from scipy.optimize.leastsq when N_params > N_data
        warnings.warn(
            f"compute_deviations (order={order}) failed (singular matrix, numerical issue, or degenerate input): {e}. "
            f"Returning zeros.",
            RuntimeWarning
        )
        return 0.0, 0.0, 0.0, 0.0
    # Note: Removed broad 'except Exception' - unexpected errors should be raised for debugging

def build_isofit_design_matrix(angles, orders):
    """
    Build extended design matrix for ISOFIT simultaneous harmonic fitting.

    Columns: [1, sin(θ), cos(θ), sin(2θ), cos(2θ), sin(n₁θ), cos(n₁θ), ...]
    First 5 columns are identical to build_harmonic_matrix() output.

    Args:
        angles: Array of angles in radians (ψ in EA mode, φ in regular mode).
        orders: List of higher-order harmonic orders (e.g. [3, 4]).

    Returns:
        Design matrix of shape (n_samples, 5 + 2*len(orders)).
    """
    n = len(angles)
    n_cols = 5 + 2 * len(orders)
    A = np.zeros((n, n_cols))

    # First 5 columns: constant, sin(θ), cos(θ), sin(2θ), cos(2θ)
    A[:, 0] = 1.0
    A[:, 1] = np.sin(angles)
    A[:, 2] = np.cos(angles)
    A[:, 3] = np.sin(2.0 * angles)
    A[:, 4] = np.cos(2.0 * angles)

    # Higher-order columns
    for k, n_order in enumerate(orders):
        A[:, 5 + 2 * k] = np.sin(n_order * angles)
        A[:, 5 + 2 * k + 1] = np.cos(n_order * angles)

    return A


def fit_all_harmonics(angles, intens, orders):
    """
    Fit all harmonics simultaneously via least squares (ISOFIT approach).

    Fits the model: I(θ) = I₀ + A₁sin(θ) + B₁cos(θ) + A₂sin(2θ) + B₂cos(2θ)
                          + Σₙ [Aₙsin(nθ) + Bₙcos(nθ)]

    Coefficients layout: [I₀, A₁, B₁, A₂, B₂, A_{n1}, B_{n1}, A_{n2}, B_{n2}, ...]

    Args:
        angles: Angle array in radians.
        intens: Intensity values at each angle.
        orders: List of higher-order harmonic orders (e.g. [3, 4]).

    Returns:
        (coeffs, ata_inv): coeffs array and inverse of (A^T A) for error estimation.
        On failure, returns (zeros, None).
    """
    n_params = 5 + 2 * len(orders)
    A = build_isofit_design_matrix(angles, orders)

    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(A, intens, rcond=None)
        ata_inv = np.linalg.inv(np.dot(A.T, A))
        return coeffs, ata_inv
    except np.linalg.LinAlgError:
        return np.zeros(n_params), None


def evaluate_harmonic_model(angles, coeffs, orders):
    """
    Evaluate the full ISOFIT harmonic model at given angles.

    Model: I(θ) = I₀ + A₁sin(θ) + B₁cos(θ) + A₂sin(2θ) + B₂cos(2θ)
                + Σₙ [Aₙsin(nθ) + Bₙcos(nθ)]

    When orders is empty, this is equivalent to harmonic_function() (5-param model).

    Args:
        angles: Array of angles in radians.
        coeffs: Full coefficient array from fit_all_harmonics().
        orders: List of higher-order harmonic orders matching the coeffs layout.

    Returns:
        Array of model intensities.
    """
    model = (coeffs[0]
             + coeffs[1] * np.sin(angles)
             + coeffs[2] * np.cos(angles)
             + coeffs[3] * np.sin(2.0 * angles)
             + coeffs[4] * np.cos(2.0 * angles))

    for k, n_order in enumerate(orders):
        model += coeffs[5 + 2 * k] * np.sin(n_order * angles)
        model += coeffs[5 + 2 * k + 1] * np.cos(n_order * angles)

    return model


def fit_higher_harmonics_simultaneous(angles, intens, sma, gradient, orders=None):
    """
    Fit multiple higher-order harmonics simultaneously.

    This implements the ISOFIT approach from Ciambur 2015, fitting all higher-order
    harmonics (n >= 3) in a single least squares solve. This accounts for cross-correlations
    between harmonics and provides better error estimates compared to sequential fitting.

    Model: I(θ) = I₀ + Σₙ [Aₙ*sin(nθ) + Bₙ*cos(nθ)]

    The coefficients are normalized by (sma * |gradient|) to give dimensionless
    deviations that can be directly compared across different radii.

    Args:
        angles: Angle array in radians (ψ in EA mode, φ in regular mode)
        intens: Intensity values at each angle
        sma: Semi-major axis length
        gradient: Radial intensity gradient (should be negative for typical galaxies)
        orders: List of harmonic orders to fit (default [3, 4])

    Returns:
        dict: {n: (a_n, b_n, a_n_err, b_n_err) for n in orders}
              a_n = sin coefficient / (sma * |gradient|)
              b_n = cos coefficient / (sma * |gradient|)
              Errors are similarly normalized.

    Raises:
        ValueError: If orders is empty or contains invalid values
    """
    if orders is None:
        orders = [3, 4]

    if len(orders) == 0:
        return {}

    if len(intens) < 2 * len(orders) + 1:
        # Not enough data points for the number of parameters
        return {n: (0.0, 0.0, 0.0, 0.0) for n in orders}

    # Build design matrix: [1, sin(n1*θ), cos(n1*θ), sin(n2*θ), cos(n2*θ), ...]
    n_params = 1 + 2 * len(orders)  # intercept + 2 params per harmonic
    A = np.zeros((len(angles), n_params))
    A[:, 0] = 1.0  # constant term

    col_idx = 1
    for n in orders:
        A[:, col_idx] = np.sin(n * angles)
        A[:, col_idx + 1] = np.cos(n * angles)
        col_idx += 2

    try:
        # Solve least squares
        coeffs, residuals, rank, s = np.linalg.lstsq(A, intens, rcond=None)

        # Compute covariance matrix (A^T * A)^-1
        ata_inv = np.linalg.inv(np.dot(A.T, A))

        # Compute model and residual variance
        model = np.dot(A, coeffs)
        if len(intens) <= n_params:
            return {n: (0.0, 0.0, 0.0, 0.0) for n in orders}
        var_residual = np.var(intens - model, ddof=n_params)

        # Coefficient errors from covariance
        covariance = ata_inv * var_residual
        errors = np.sqrt(np.diagonal(covariance))

    except (np.linalg.LinAlgError, ValueError) as e:
        warnings.warn(
            f"fit_higher_harmonics_simultaneous failed: {e}. Returning zeros.",
            RuntimeWarning
        )
        return {n: (0.0, 0.0, 0.0, 0.0) for n in orders}

    # Normalization factor
    factor = sma * abs(gradient) if gradient != 0 else 1.0
    if factor == 0:
        factor = 1.0

    # Extract results for each harmonic order
    result = {}
    col_idx = 1
    err_idx = 1
    for n in orders:
        sin_coeff = coeffs[col_idx]
        cos_coeff = coeffs[col_idx + 1]
        sin_err = errors[err_idx]
        cos_err = errors[err_idx + 1]

        # Normalize by (sma * gradient) to get dimensionless deviations
        a_n = sin_coeff / factor
        b_n = cos_coeff / factor
        a_n_err = sin_err / factor
        b_n_err = cos_err / factor

        result[n] = (a_n, b_n, a_n_err, b_n_err)

        col_idx += 2
        err_idx += 2

    return result


def compute_gradient(image, mask, geometry, config, previous_gradient=None, current_data=None):
    """Compute the radial intensity gradient.

    Args:
        image: 2D image array
        mask: 2D boolean mask array (True = masked)
        geometry: dict with keys {x0, y0, sma, eps, pa}
        config: dict or IsosterConfig with keys/attrs {astep, linear_growth, integrator, use_eccentric_anomaly}
        previous_gradient: Previous gradient value for comparison (optional)
        current_data: Cached (phi, intens) tuple to avoid re-extraction (optional)

    Returns:
        (gradient, gradient_error): Tuple of gradient value and its error estimate
    """
    # Unpack geometry
    x0, y0, sma, eps, pa = geometry['x0'], geometry['y0'], geometry['sma'], geometry['eps'], geometry['pa']

    # Unpack config (dict from fit_isophote, or IsosterConfig in tests)
    _get = config.get if isinstance(config, dict) else lambda k: getattr(config, k)
    step = _get('astep')
    linear_growth = _get('linear_growth')
    integrator = _get('integrator')
    use_eccentric_anomaly = _get('use_eccentric_anomaly')

    if current_data is not None:
        phi_c, intens_c = current_data
    else:
        # Extract current SMA data
        data_c = extract_isophote_data(image, mask, x0, y0, sma, eps, pa, use_eccentric_anomaly=use_eccentric_anomaly)
        phi_c = data_c.angles  # Use angles for fitting (φ in this case)
        intens_c = data_c.intens
    
    if len(intens_c) == 0:
        return previous_gradient * 0.8 if previous_gradient else -1.0, None
        
    if integrator == 'median':
        mean_c = np.median(intens_c)
    else:
        mean_c = np.mean(intens_c)
    
    if linear_growth:
        gradient_sma = sma + step
    else:
        gradient_sma = sma * (1.0 + step)
        
    # Extract gradient SMA data
    data_g = extract_isophote_data(image, mask, x0, y0, gradient_sma, eps, pa, use_eccentric_anomaly=use_eccentric_anomaly)
    phi_g = data_g.angles
    intens_g = data_g.intens
    
    if len(intens_g) == 0:
        return previous_gradient * 0.8 if previous_gradient else -1.0, None
        
    if integrator == 'median':
        mean_g = np.median(intens_g)
    else:
        mean_g = np.mean(intens_g)
    delta_r = step if linear_growth else sma * step
    gradient = (mean_g - mean_c) / delta_r

    sigma_c = np.std(intens_c)
    sigma_g = np.std(intens_g)
    gradient_error = (np.sqrt(sigma_c**2 / len(intens_c) + sigma_g**2 / len(intens_g))
                     / delta_r)
    
    if previous_gradient is None:
        previous_gradient = gradient + gradient_error

    # EFF-1: Early termination if first gradient is reliable
    # Compute relative error to decide if second gradient SMA is needed
    relative_error = abs(gradient_error / gradient) if (gradient is not None and gradient != 0) else np.inf

    # Skip second gradient if:
    # 1. First gradient looks good (< previous_gradient / 3)
    # 2. OR first gradient is reliable (relative_error < 0.3)
    need_second_gradient = (gradient >= (previous_gradient / 3.0)) and (relative_error >= 0.3)

    if need_second_gradient:
        if linear_growth:
            gradient_sma_2 = sma + 2 * step
        else:
            gradient_sma_2 = sma * (1.0 + 2 * step)

        # Extract second gradient SMA
        data_g2 = extract_isophote_data(image, mask, x0, y0, gradient_sma_2, eps, pa, use_eccentric_anomaly=use_eccentric_anomaly)
        phi_g2 = data_g2.angles
        intens_g2 = data_g2.intens

        if len(intens_g2) > 0:
            if integrator == 'median':
                mean_g2 = np.median(intens_g2)
            else:
                mean_g2 = np.mean(intens_g2)
            delta_r_2 = 2 * step if linear_growth else sma * 2 * step
            gradient = (mean_g2 - mean_c) / delta_r_2
            sigma_g2 = np.std(intens_g2)
            gradient_error = (np.sqrt(sigma_c**2 / len(intens_c) + sigma_g2**2 / len(intens_g2))
                            / delta_r_2)

    if gradient >= (previous_gradient / 3.0):
        gradient = previous_gradient * 0.8
        gradient_error = None
        
    return gradient, gradient_error

def compute_aperture_photometry(image, mask, x0, y0, sma, eps, pa):
    """
    Compute total flux and pixel counts within elliptical and circular apertures.
    
    This uses a vectorized numpy approach for speed.
    """
    h, w = image.shape
    
    # Bounding box
    x_min = max(0, int(x0 - sma - 1))
    x_max = min(w, int(x0 + sma + 1))
    y_min = max(0, int(y0 - sma - 1))
    y_max = min(h, int(y0 + sma + 1))
    
    if x_max <= x_min or y_max <= y_min:
        return 0.0, 0.0, 0, 0
        
    y, x = np.mgrid[y_min:y_max, x_min:x_max]
    
    # Circular aperture
    r2 = (x - x0)**2 + (y - y0)**2
    mask_c = r2 <= sma**2
    
    # Elliptical aperture
    dx = x - x0
    dy = y - y0
    cos_pa, sin_pa = np.cos(pa), np.sin(pa)
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    sma_pix2 = x_rot**2 + (y_rot / (1.0 - eps))**2
    mask_e = sma_pix2 <= sma**2
    
    # Data extraction
    data = image[y_min:y_max, x_min:x_max]
    if mask is not None:
        mdata = mask[y_min:y_max, x_min:x_max]
        valid = ~mdata
    else:
        valid = np.ones_like(data, dtype=bool)
        
    valid &= ~np.isnan(data)
    
    # Calculate metrics
    tflux_c = np.sum(data[mask_c & valid])
    npix_c = np.sum(mask_c & valid)
    
    tflux_e = np.sum(data[mask_e & valid])
    npix_e = np.sum(mask_e & valid)
    
    return tflux_e, tflux_c, npix_e, npix_c


def _attach_full_photometry(isophote_result, image, mask):
    """Populate aperture photometry fields in-place for a fitted isophote result."""
    tflux_e, tflux_c, npix_e, npix_c = compute_aperture_photometry(
        image,
        mask,
        isophote_result['x0'],
        isophote_result['y0'],
        isophote_result['sma'],
        isophote_result['eps'],
        isophote_result['pa'],
    )
    isophote_result.update({
        'tflux_e': tflux_e,
        'tflux_c': tflux_c,
        'npix_e': npix_e,
        'npix_c': npix_c,
    })

def _compute_posthoc_harmonics(best_geometry, angles, intens, gradient,
                               best_angles, best_intens, best_gradient,
                               sma, harmonic_orders, isofit_mode,
                               simultaneous_harmonics):
    """Compute post-hoc harmonic deviations and store them in best_geometry.

    Called after convergence (stop_code=0 via harmonic or geometry criteria)
    or after max-iteration fallback (stop_code=2). For 'original' ISOFIT mode,
    uses saved best-iteration data; otherwise uses current-iteration data.

    Args:
        best_geometry: Dict to update with harmonic coefficients (mutated in place).
        angles: Current-iteration angle array (psi in EA mode, phi otherwise).
        intens: Current-iteration intensity array.
        gradient: Current-iteration gradient value.
        best_angles: Saved best-iteration angles (may be None).
        best_intens: Saved best-iteration intensities (may be None).
        best_gradient: Saved best-iteration gradient (may be None).
        sma: Semi-major axis length.
        harmonic_orders: List of harmonic orders to compute (e.g. [3, 4]).
        isofit_mode: 'in_loop' or 'original'.
        simultaneous_harmonics: Whether to use simultaneous fitting.
    """
    use_best_data = (isofit_mode == 'original' and simultaneous_harmonics
                     and best_angles is not None)
    posthoc_angles = best_angles if use_best_data else angles
    posthoc_intens = best_intens if use_best_data else intens
    posthoc_gradient = best_gradient if use_best_data else gradient

    if simultaneous_harmonics:
        sim_harmonics = fit_higher_harmonics_simultaneous(
            posthoc_angles, posthoc_intens, sma, posthoc_gradient, harmonic_orders
        )
        for n in harmonic_orders:
            an, bn, an_err, bn_err = sim_harmonics.get(n, (0.0, 0.0, 0.0, 0.0))
            best_geometry[f'a{n}'] = an
            best_geometry[f'b{n}'] = bn
            best_geometry[f'a{n}_err'] = an_err
            best_geometry[f'b{n}_err'] = bn_err
    else:
        for n in harmonic_orders:
            an, bn, an_err, bn_err = compute_deviations(
                posthoc_angles, posthoc_intens, sma, posthoc_gradient, n
            )
            best_geometry[f'a{n}'] = an
            best_geometry[f'b{n}'] = bn
            best_geometry[f'a{n}_err'] = an_err
            best_geometry[f'b{n}_err'] = bn_err


def fit_isophote(image, mask, sma, start_geometry, config, going_inwards=False, previous_geometry=None):
    """
    Fit a single isophote with quality control.

    This function performs the iterative harmonic fitting for a single semi-major axis (SMA).

    Args:
        image (np.ndarray): Input image.
        mask (np.ndarray): Bad pixel mask.
        sma (float): Semi-major axis length to fit.
        start_geometry (dict): Initial guess for {'x0', 'y0', 'eps', 'pa'}.
        config (dict or IsosterConfig): Configuration object.
        going_inwards (bool): Flag indicating if the fitting is progressing inwards (affecting gradient checks).
        previous_geometry (dict): Previous isophote geometry for regularization (optional).

    Returns:
        dict: The best fitted geometry and metadata for this isophote.
    """
    # Normalize configuration
    if isinstance(config, IsosterConfig):
        cfg = config
    else:
        cfg = IsosterConfig(**(config or {}))

    maxit = cfg.maxit
    conver = cfg.conver
    minit = cfg.minit
    astep = cfg.astep
    linear_growth = cfg.linear_growth
    fix_center = cfg.fix_center
    fix_pa = cfg.fix_pa
    fix_eps = cfg.fix_eps
    sclip = cfg.sclip
    nclip = cfg.nclip
    sclip_low = cfg.sclip_low
    sclip_high = cfg.sclip_high
    fflag = cfg.fflag
    maxgerr = cfg.maxgerr
    debug = cfg.debug
    full_photometry = cfg.full_photometry or debug
    use_eccentric_anomaly = cfg.use_eccentric_anomaly
    compute_errors = cfg.compute_errors
    compute_deviations_flag = cfg.compute_deviations
    integrator = cfg.integrator
    lsb_sma_threshold = cfg.lsb_sma_threshold
    simultaneous_harmonics = cfg.simultaneous_harmonics
    harmonic_orders = cfg.harmonic_orders
    permissive_geometry = cfg.permissive_geometry
    isofit_mode = cfg.isofit_mode
    geometry_update_mode = cfg.geometry_update_mode


    x0, y0, eps, pa = start_geometry['x0'], start_geometry['y0'], start_geometry['eps'], start_geometry['pa']
    stop_code, niter, best_geometry = 0, 0, None
    converged = False
    min_amplitude, previous_gradient, lexceed = np.inf, None, False
    
    # Compute convergence scale factor once (sma is constant within the loop)
    if cfg.convergence_scaling == 'sector_area':
        n_samples = max(64, int(2 * np.pi * sma))
        angular_width = 2 * np.pi / n_samples
        delta_sma = sma * astep if not linear_growth else astep
        convergence_scale = max(1.0, sma * delta_sma * angular_width)
    elif cfg.convergence_scaling == 'sqrt_sma':
        convergence_scale = max(1.0, np.sqrt(sma))
    else:  # 'none'
        convergence_scale = 1.0

    # ISOFIT: minimum data points needed for full extended design matrix
    if simultaneous_harmonics:
        isofit_min_points = 1 + 2 * (2 + len(harmonic_orders))
    else:
        isofit_min_points = 6
    best_isofit_harmonics_stored = False
    # Saved data from best-geometry iteration for post-hoc harmonic fitting
    best_angles = None
    best_intens = None
    best_gradient = None

    # Geometry convergence tracking
    prev_geom = (x0, y0, eps, pa)
    stable_count = 0
    
    # Lazy gradient tracking
    cached_gradient = None
    cached_gradient_error = None
    no_improvement_count = 0

    for i in range(maxit):
        niter = i + 1
        
        # Extract isophote data - returns named tuple
        # For EA mode: angles=ψ (for harmonics), phi=φ (for geometry)
        # For regular: angles=φ (for harmonics), phi=φ (same)
        data = extract_isophote_data(image, mask, x0, y0, sma, eps, pa, use_eccentric_anomaly=use_eccentric_anomaly)
        
        angles = data.angles  # ψ for EA mode, φ for regular mode
        phi = data.phi        # φ (always available for geometry updates)
        intens = data.intens
        
        total_points = len(angles)
        
        # Sigma clipping operates on (angle, intensity) pairs.
        # In EA mode, phi differs from angles (psi) and must be clipped with the same mask.
        if use_eccentric_anomaly:
            angles, intens, n_clipped, phi = sigma_clip(
                angles, intens, sclip, nclip, sclip_low, sclip_high,
                extra_arrays=[phi]
            )
        else:
            angles, intens, n_clipped = sigma_clip(angles, intens, sclip, nclip, sclip_low, sclip_high)
            phi = angles  # In regular mode, angles and phi are identical
        actual_points = len(angles)
        
        if actual_points < (total_points * (1.0 - fflag)):
            if best_geometry is not None:
                best_geometry['stop_code'], best_geometry['niter'] = 1, niter  # STOP_CODE 1: Too many flagged pixels
                return best_geometry
            else:
                res = {'x0': x0, 'y0': y0, 'eps': eps, 'pa': pa, 'sma': sma,
                       'intens': np.nan, 'rms': np.nan, 'intens_err': np.nan,
                       'stop_code': 1, 'niter': niter,  # STOP_CODE 1: Too many flagged pixels
                       'tflux_e': np.nan, 'tflux_c': np.nan, 'npix_e': 0, 'npix_c': 0}
                if debug:
                    res.update({'ndata': actual_points, 'nflag': total_points - actual_points,
                                'grad': np.nan, 'grad_error': np.nan, 'grad_r_error': np.nan})
                return res
        
        if len(intens) < 6:
            stop_code = 3  # STOP_CODE 3: Too few points (need ≥6 for 5-parameter harmonic fit)
            break

        # ISOFIT per-iteration fallback: use full ISOFIT only when enough points
        use_isofit_this_iter = simultaneous_harmonics and actual_points >= isofit_min_points
        if simultaneous_harmonics and not use_isofit_this_iter and actual_points >= 6:
            if i == 0:
                warnings.warn(
                    f"ISOFIT: SMA={sma:.1f} has {actual_points} points, need "
                    f"{isofit_min_points} for orders {harmonic_orders}. "
                    f"Falling back to 5-param fit.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Determine effective integrator for this isophote
        eff_integrator = integrator
        if integrator == 'adaptive':
            if lsb_sma_threshold is not None and sma > lsb_sma_threshold:
                eff_integrator = 'median'
            else:
                eff_integrator = 'mean'

        # FIT HARMONICS in appropriate angle space
        # ISOFIT in_loop: fit all orders simultaneously in a single design matrix
        # ISOFIT original: always 5-param inside loop (post-hoc simultaneous after convergence)
        # Default path: fit only 1st and 2nd harmonics (5-param, numba-accelerated)
        use_isofit_in_loop = use_isofit_this_iter and isofit_mode == 'in_loop'
        if use_isofit_in_loop:
            coeffs, cov_matrix = fit_all_harmonics(angles, intens, harmonic_orders)
        else:
            coeffs, cov_matrix = fit_first_and_second_harmonics(angles, intens)
        y0_fit = coeffs[0]
        A1, B1, A2, B2 = coeffs[1], coeffs[2], coeffs[3], coeffs[4]
        
        # GRADIENT computed using φ and current geometry
        # Lazy Evaluation: reuse gradient unless convergence stalls
        if i == 0 or not cfg.use_lazy_gradient or no_improvement_count >= 3:
            geometry = {'x0': x0, 'y0': y0, 'sma': sma, 'eps': eps, 'pa': pa}
            gradient_config = {
                'astep': astep,
                'linear_growth': linear_growth,
                'integrator': eff_integrator,
                'use_eccentric_anomaly': use_eccentric_anomaly
            }
            gradient, gradient_error = compute_gradient(
                image, mask, geometry, gradient_config,
                previous_gradient=previous_gradient,
                current_data=(phi, intens)
            )
            cached_gradient = gradient
            cached_gradient_error = gradient_error
            if no_improvement_count >= 3:
                no_improvement_count = 0  # Reset after forced re-evaluation
        else:
            gradient = cached_gradient
            gradient_error = cached_gradient_error

        if gradient_error is not None:
            previous_gradient = gradient
        
        gradient_relative_error = abs(gradient_error / gradient) if (gradient_error is not None and gradient is not None and gradient < 0) else None
        if not going_inwards:
            # In permissive mode, skip the check if gradient_relative_error is None
            # This matches photutils's behavior which continues when gradient error can't be computed
            if permissive_geometry and gradient_relative_error is None:
                pass  # Accept fit, continue
            elif gradient_relative_error is None or gradient_relative_error > maxgerr or gradient >= 0:
                if lexceed:
                    stop_code = -1  # STOP_CODE -1: Gradient error (high relative error exceeded twice)
                    break
                else:
                    lexceed = True

        if gradient == 0:
            stop_code = -1  # STOP_CODE -1: Zero gradient (cannot compute corrections)
            break
            
        # Evaluate model in angle space used for fitting
        # ISOFIT in_loop: full model includes higher-order harmonics → cleaner RMS
        # ISOFIT original / default: 5-param model only
        if use_isofit_in_loop:
            model = evaluate_harmonic_model(angles, coeffs, harmonic_orders)
        else:
            model = harmonic_function(angles, coeffs)
        rms = np.std(intens - model)
        harmonics = [A1, B1, A2, B2]
        if fix_center: harmonics[0] = harmonics[1] = 0
        if fix_pa: harmonics[2] = 0
        if fix_eps: harmonics[3] = 0
            
        max_idx = np.argmax(np.abs(harmonics))
        max_amp = harmonics[max_idx]
        
        # Apply central region regularization penalty if enabled
        current_geom = {'x0': x0, 'y0': y0, 'eps': eps, 'pa': pa}
        reg_penalty = compute_central_regularization_penalty(current_geom, previous_geometry, sma, cfg)
        
        # Effective amplitude includes regularization penalty
        # This discourages large geometry changes in the central region
        effective_amp = abs(max_amp) + reg_penalty
        
        if effective_amp < min_amplitude:
            min_amplitude = effective_amp
            no_improvement_count = 0
            intens_err = rms / np.sqrt(len(intens))
            # For ISOFIT in_loop, pass 5x5 sub-matrix and first 5 coefficients
            # so compute_parameter_errors uses correct dimensions
            if compute_errors:
                cov_5x5 = cov_matrix[:5, :5] if (use_isofit_in_loop and cov_matrix is not None) else cov_matrix
                coeffs_5 = coeffs[:5]
                x0_err, y0_err, eps_err, pa_err = compute_parameter_errors(
                    phi, intens, x0, y0, sma, eps, pa, gradient,
                    gradient_error=gradient_error if cfg.use_corrected_errors else None,
                    cov_matrix=cov_5x5, coeffs=coeffs_5,
                    var_residual_floor=cfg.sigma_bg**2 if cfg.sigma_bg is not None else None
                )
            else:
                x0_err, y0_err, eps_err, pa_err = 0.0, 0.0, 0.0, 0.0
            if eff_integrator == 'median':
                reported_intens = np.median(intens)
            else:
                reported_intens = y0_fit
                
            best_geometry = {'x0': x0, 'y0': y0, 'eps': eps, 'pa': pa, 'sma': sma, 'intens': reported_intens, 'rms': rms, 'intens_err': intens_err,
                             'x0_err': x0_err, 'y0_err': y0_err, 'eps_err': eps_err, 'pa_err': pa_err,
                             'tflux_e': np.nan, 'tflux_c': np.nan, 'npix_e': 0, 'npix_c': 0,
                             'use_eccentric_anomaly': use_eccentric_anomaly}
            # Initialize harmonics for all requested orders when deviations or ISOFIT active
            if compute_deviations_flag or simultaneous_harmonics:
                for n in harmonic_orders:
                    best_geometry[f'a{n}'] = 0.0
                    best_geometry[f'b{n}'] = 0.0
                    best_geometry[f'a{n}_err'] = 0.0
                    best_geometry[f'b{n}_err'] = 0.0
            # Save best-iteration data for post-hoc ISOFIT original mode
            best_angles = angles.copy()
            best_intens = intens.copy()
            best_gradient = gradient
            # ISOFIT in_loop: extract and store higher-order harmonics from the joint fit
            if use_isofit_in_loop and gradient != 0:
                factor = sma * abs(gradient)
                if factor > 0:
                    # Compute residual variance once outside the per-order loop (R26-08)
                    if cov_matrix is not None:
                        n_params = len(coeffs)
                        var_residual = np.var(intens - model, ddof=n_params) if len(intens) > n_params else 0.0
                    for k, n_order in enumerate(harmonic_orders):
                        sin_coeff = coeffs[5 + 2 * k]
                        cos_coeff = coeffs[5 + 2 * k + 1]
                        best_geometry[f'a{n_order}'] = sin_coeff / factor
                        best_geometry[f'b{n_order}'] = cos_coeff / factor
                        if cov_matrix is not None:
                            sin_err = np.sqrt(cov_matrix[5 + 2 * k, 5 + 2 * k] * var_residual) if var_residual > 0 else 0.0
                            cos_err = np.sqrt(cov_matrix[5 + 2 * k + 1, 5 + 2 * k + 1] * var_residual) if var_residual > 0 else 0.0
                            best_geometry[f'a{n_order}_err'] = sin_err / factor
                            best_geometry[f'b{n_order}_err'] = cos_err / factor
                    best_isofit_harmonics_stored = True
            if debug:
                best_geometry.update({'ndata': actual_points, 'nflag': total_points - actual_points, 'grad': gradient,
                                      'grad_error': gradient_error if gradient_error is not None else np.nan,
                                      'grad_r_error': gradient_relative_error if gradient_relative_error is not None else np.nan})
        else:
            no_improvement_count += 1
            
        effective_rms = rms
        if cfg.sigma_bg is not None and len(intens) > 0:
            noise_floor = cfg.sigma_bg / np.sqrt(len(intens))
            effective_rms = max(rms, noise_floor)

        if abs(max_amp) < conver * convergence_scale * effective_rms and i >= minit:
            stop_code = 0  # STOP_CODE 0: Converged successfully
            converged = True
            if compute_deviations_flag and not best_isofit_harmonics_stored:
                _compute_posthoc_harmonics(
                    best_geometry, angles, intens, gradient,
                    best_angles, best_intens, best_gradient,
                    sma, harmonic_orders, isofit_mode, simultaneous_harmonics,
                )

            # 6. FULL PHOTOMETRY (If requested)
            if full_photometry:
                _attach_full_photometry(best_geometry, image, mask)
            break
            
        # Update geometry (apply damping to reduce oscillations at large SMA)
        damping = cfg.geometry_damping
        if geometry_update_mode == 'simultaneous':
            # Simultaneous: update ALL geometry parameters each iteration
            # Center corrections (minor/major axis)
            if not fix_center:
                aux_minor = -harmonics[0] * (1.0 - eps) / gradient * damping
                aux_major = -harmonics[1] / gradient * damping
                x0 += -aux_minor * np.sin(pa) + aux_major * np.cos(pa)
                y0 += aux_minor * np.cos(pa) + aux_major * np.sin(pa)
            # PA correction
            if not fix_pa:
                denom = (1.0 - eps)**2 - 1.0
                if abs(denom) < 1e-10: denom = -1e-10
                pa_corr = harmonics[2] * 2.0 * (1.0 - eps) / sma / gradient / denom * damping
                pa = (pa + pa_corr) % np.pi
            # Ellipticity correction
            if not fix_eps:
                eps_corr = harmonics[3] * 2.0 * (1.0 - eps) / sma / gradient * damping
                eps = min(eps - eps_corr, 0.95)
                if eps < 0.0:
                    eps = min(-eps, 0.95)
                    pa = (pa + np.pi/2) % np.pi
                if eps == 0.0: eps = 0.05
        else:
            # Largest: update only the parameter with the largest harmonic amplitude
            if max_idx == 0:
                aux = -max_amp * (1.0 - eps) / gradient * damping
                x0 -= aux * np.sin(pa)
                y0 += aux * np.cos(pa)
            elif max_idx == 1:
                aux = -max_amp / gradient * damping
                x0 += aux * np.cos(pa)
                y0 += aux * np.sin(pa)
            elif max_idx == 2:
                # denom = (1-eps)^2 - 1 = -eps*(2-eps), always <= 0 (matches photutils)
                denom = (1.0 - eps)**2 - 1.0
                if abs(denom) < 1e-10: denom = -1e-10  # Avoid division by zero for eps~0
                pa_corr = max_amp * 2.0 * (1.0 - eps) / sma / gradient / denom * damping
                pa = (pa + pa_corr) % np.pi
            elif max_idx == 3:
                eps_corr = max_amp * 2.0 * (1.0 - eps) / sma / gradient * damping
                eps = min(eps - eps_corr, 0.95)
                if eps < 0.0:
                    eps = min(-eps, 0.95)
                    pa = (pa + np.pi/2) % np.pi
                if eps == 0.0: eps = 0.05

        # Geometry-stability convergence check
        if cfg.geometry_convergence and i >= minit:
            gx0, gy0, geps, gpa = prev_geom
            delta_x0 = abs(x0 - gx0) / max(sma, 1.0)
            delta_y0 = abs(y0 - gy0) / max(sma, 1.0)
            delta_eps = abs(eps - geps)
            delta_pa_raw = abs(pa - gpa)
            delta_pa = min(delta_pa_raw, np.pi - delta_pa_raw) / np.pi
            max_delta = max(delta_x0, delta_y0, delta_eps, delta_pa)

            if max_delta < cfg.geometry_tolerance:
                stable_count += 1
            else:
                stable_count = 0

            if stable_count >= cfg.geometry_stable_iters:
                stop_code = 0
                converged = True
                if compute_deviations_flag and not best_isofit_harmonics_stored:
                    _compute_posthoc_harmonics(
                        best_geometry, angles, intens, gradient,
                        best_angles, best_intens, best_gradient,
                        sma, harmonic_orders, isofit_mode, simultaneous_harmonics,
                    )
                if full_photometry:
                    _attach_full_photometry(best_geometry, image, mask)
                break

        prev_geom = (x0, y0, eps, pa)

    if best_geometry is None:
        best_geometry = {'x0': x0, 'y0': y0, 'eps': eps, 'pa': pa, 'sma': sma, 'intens': np.nan, 'rms': np.nan, 'intens_err': np.nan,
                         'x0_err': 0.0, 'y0_err': 0.0, 'eps_err': 0.0, 'pa_err': 0.0,
                         'tflux_e': np.nan, 'tflux_c': np.nan, 'npix_e': 0, 'npix_c': 0,
                         'use_eccentric_anomaly': use_eccentric_anomaly}
        # Initialize harmonics when deviations or ISOFIT active
        if compute_deviations_flag or simultaneous_harmonics:
            for n in harmonic_orders:
                best_geometry[f'a{n}'] = 0.0
                best_geometry[f'b{n}'] = 0.0
                best_geometry[f'a{n}_err'] = 0.0
                best_geometry[f'b{n}_err'] = 0.0
        if debug: best_geometry.update({'ndata': 0, 'nflag': 0, 'grad': np.nan, 'grad_error': np.nan, 'grad_r_error': np.nan})

    if niter >= maxit and stop_code == 0 and not converged:
        stop_code = 2  # STOP_CODE 2: Reached max iterations without convergence
        # Best-effort harmonic deviations from best-geometry iteration
        if compute_deviations_flag and best_geometry is not None and not best_isofit_harmonics_stored:
            _compute_posthoc_harmonics(
                best_geometry, angles, intens, gradient,
                best_angles, best_intens, best_gradient,
                sma, harmonic_orders, isofit_mode, simultaneous_harmonics,
            )
        if full_photometry:
            _attach_full_photometry(best_geometry, image, mask)

    best_geometry['stop_code'], best_geometry['niter'] = stop_code, niter
    return best_geometry
