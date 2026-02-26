import warnings

import numpy as np
from .fitting import fit_isophote
from .config import IsosterConfig

ACCEPTABLE_STOP_CODES = {0, 1, 2}


def _is_acceptable_stop_code(stop_code):
    """Return True when a stop code is acceptable for geometry propagation."""
    return stop_code in ACCEPTABLE_STOP_CODES


def _is_error_field(field_name):
    """Return True when the field represents an uncertainty/error quantity."""
    return field_name.endswith('_err') or field_name.endswith('_error')


def _validate_non_negative_error_fields(isophotes):
    """Ensure all finite error values in fitted isophotes are non-negative."""
    if not isophotes:
        return

    error_fields = {
        key
        for iso in isophotes
        if isinstance(iso, dict)
        for key in iso.keys()
        if _is_error_field(key)
    }
    if not error_fields:
        return

    for iso_index, iso in enumerate(isophotes):
        if not isinstance(iso, dict):
            continue
        for field_name in error_fields:
            if field_name not in iso:
                continue
            try:
                numeric_value = float(iso[field_name])
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric_value) and numeric_value < 0.0:
                raise ValueError(
                    "isoster produced negative error value "
                    f"(field={field_name}, index={iso_index}, value={numeric_value})"
                )


def _build_fit_result(isophotes, config):
    """Package fit results with mandatory post-run sanity validation."""
    _validate_non_negative_error_fields(isophotes)
    return {'isophotes': isophotes, 'config': config}


def fit_central_pixel(image, mask, x0, y0, debug=False):
    """
    Fit the central pixel (SMA=0).

    Args:
        image (np.ndarray): Input image.
        mask (np.ndarray, optional): Boolean mask (True=bad).
        x0 (float): Center x coordinate.
        y0 (float): Center y coordinate.
        debug (bool): If True, include extra debug fields.

    Returns:
        dict: A dictionary containing the fitting result for the central pixel.
    """
    # Simple estimation for center
    # Use np.round() instead of int() to avoid truncation bias
    val = image[int(np.round(y0)), int(np.round(x0))]
    valid = True
    if mask is not None:
        if mask[int(np.round(y0)), int(np.round(x0))]:
            valid = False
            
    result = {
        'x0': x0, 'y0': y0, 'eps': 0.0, 'pa': 0.0, 'sma': 0.0,
        'intens': val if valid else np.nan,
        'rms': 0.0, 'intens_err': 0.0,
        'x0_err': 0.0, 'y0_err': 0.0, 'eps_err': 0.0, 'pa_err': 0.0,
        'a3': 0.0, 'b3': 0.0, 'a3_err': 0.0, 'b3_err': 0.0,
        'a4': 0.0, 'b4': 0.0, 'a4_err': 0.0, 'b4_err': 0.0,
        'tflux_e': np.nan, 'tflux_c': np.nan, 'npix_e': 0, 'npix_c': 0,
        'stop_code': 0 if valid else -1,
        'niter': 0, 'valid': valid
    }
    if debug:
        result.update({'ndata': 1 if valid else 0, 'nflag': 0,
                       'grad': np.nan, 'grad_error': np.nan, 'grad_r_error': np.nan})
    return result

def fit_image(image, mask=None, config=None, template_isophotes=None):
    """
    Main driver to fit isophotes to an entire image.

    This function orchestrates the fitting process, starting from a central guess,
    fitting outward to the edge, and optionally inward to the center.

    Args:
        image (np.ndarray): 2D Input image.
        mask (np.ndarray, optional): 2D Bad pixel mask (True=bad).
        config (dict or IsosterConfig, optional): Configuration parameters.
            If None, default configuration is used.
        template_isophotes (list of dict, optional): List of isophote dicts to use
            as geometry template. When provided, photometry is extracted at each
            template SMA using the template's geometry (x0, y0, eps, pa) at that SMA.
            This enables multiband analysis where one band (e.g., g-band) defines
            geometry, and the same variable geometry is applied to other bands for
            consistent color profile measurement.

    Returns:
        dict: A dictionary containing:
            - 'isophotes': List of dictionaries, each representing a fitted isophote.
            - 'config': The IsosterConfig object used for the fit.

    Examples
    --------
    >>> # Normal fitting
    >>> results_g = fit_image(image_g, mask_g, config)
    >>>
    >>> # Template-based forced photometry (multiband)
    >>> results_r = fit_image(image_r, mask_r, config,
    ...                       template_isophotes=results_g['isophotes'])
    """
    if config is None:
        cfg = IsosterConfig()
    elif isinstance(config, dict):
        cfg = IsosterConfig(**config)
    else:
        cfg = config

    # V6: warn when both template_isophotes and forced=True are active
    if template_isophotes is not None and cfg.forced:
        warnings.warn(
            "template_isophotes takes priority over forced=True; "
            "forced mode will be skipped",
            UserWarning, stacklevel=2
        )

    # Handle template-based forced mode (takes priority over regular forced mode)
    if template_isophotes is not None:
        return _fit_image_template_forced(image, mask, cfg, template_isophotes)

    # Handle forced mode
    if cfg.forced:
        from .fitting import extract_forced_photometry
        
        isophotes = []
        for sma in cfg.forced_sma:
            iso = extract_forced_photometry(
                image, mask,
                cfg.x0 if cfg.x0 is not None else image.shape[1]/2,
                cfg.y0 if cfg.y0 is not None else image.shape[0]/2,
                sma,
                cfg.eps, cfg.pa,
                integrator=cfg.integrator,
                sclip=cfg.sclip,
                nclip=cfg.nclip,
                use_eccentric_anomaly=cfg.use_eccentric_anomaly
            )
            isophotes.append(iso)
        
        return _build_fit_result(isophotes, cfg)
    
    # Regular fitting mode
    h, w = image.shape
    
    # Initial Parameters
    x0 = cfg.x0 if cfg.x0 is not None else w / 2.0
    y0 = cfg.y0 if cfg.y0 is not None else h / 2.0
    sma0 = cfg.sma0
    minsma = cfg.minsma
    maxsma = cfg.maxsma if cfg.maxsma is not None else max(h, w) / 2.0
    astep = cfg.astep
    linear_growth = cfg.linear_growth
    
    # 1. Fit Central Pixel (Approximation)
    central_result = fit_central_pixel(image, mask, x0, y0, debug=cfg.debug)
    
    # 2. Fit First Isophote at SMA0
    start_geometry = {
        'x0': x0, 'y0': y0, 
        'eps': cfg.eps, 'pa': cfg.pa
    }
    
    # Pass cfg object to fit_isophote
    first_iso = fit_isophote(image, mask, sma0, start_geometry, cfg)
    
    # 3. Grow Outwards
    outwards_results = []
    if _is_acceptable_stop_code(first_iso['stop_code']):
        outwards_results.append(first_iso)
        current_iso = first_iso
        current_sma = first_iso['sma']
        
        while True:
            if linear_growth:
                next_sma = current_sma + astep
            else:
                next_sma = current_sma * (1.0 + astep)
                
            if next_sma > maxsma:
                break
            
            # Update sma tracking
            current_sma = next_sma
                
            next_iso = fit_isophote(
                image, mask, next_sma, current_iso, cfg,
                previous_geometry=current_iso
            )
            outwards_results.append(next_iso)

            # If good fit, update geometry for next step
            # In permissive mode, always update to prevent cascading failures
            if _is_acceptable_stop_code(next_iso['stop_code']) or cfg.permissive_geometry:
                current_iso = next_iso
                
    # 4. Grow Inwards
    inwards_results = []
    if minsma < sma0 and _is_acceptable_stop_code(first_iso['stop_code']):
        current_iso = first_iso
        current_sma = first_iso['sma']
        
        while True:
            if linear_growth:
                next_sma = current_sma - astep
            else:
                next_sma = current_sma / (1.0 + astep)
            
            # Stop if smaller than minsma or effectively too small (e.g. 0.5 pixel)
            limit_sma = max(minsma, 0.5) 
            if next_sma < limit_sma:
                break
            
            current_sma = next_sma
            
            # Use going_inwards=True flag
            next_iso = fit_isophote(
                image, mask, next_sma, current_iso, cfg,
                going_inwards=True,
                previous_geometry=current_iso
            )
            inwards_results.append(next_iso)

            # In permissive mode, always update to prevent cascading failures
            if _is_acceptable_stop_code(next_iso['stop_code']) or cfg.permissive_geometry:
                current_iso = next_iso

    # Combine results
    # Inwards list needs to be reversed so SMAs are ascending
    if minsma <= 0.0:
        # Prepend central pixel
        final_list = [central_result] + inwards_results[::-1] + outwards_results
    else:
        final_list = inwards_results[::-1] + outwards_results
    
    # Compute Curve-of-Growth if requested
    if cfg.compute_cog:
        from .cog import compute_cog, add_cog_to_isophotes
        
        # Determine if geometry was fixed
        fix_geometry = cfg.fix_center and cfg.fix_pa and cfg.fix_eps
        
        cog_results = compute_cog(final_list, 
                                  fix_center=cfg.fix_center, 
                                  fix_geometry=fix_geometry)
        
        # Add CoG data to isophotes
        add_cog_to_isophotes(final_list, cog_results)
            
    # Return as dict matching legacy structure + config object
    return _build_fit_result(final_list, cfg)


def _fit_image_template_forced(image, mask, config, template_isophotes):
    """
    Extract forced photometry using geometry from template isophotes.

    This function extracts photometry at each SMA from the template, using the
    template's geometry (x0, y0, eps, pa) at that specific SMA. Unlike regular
    forced mode which uses a single fixed geometry for all SMA values, this
    allows variable geometry along the radial profile.

    Args:
        image (np.ndarray): 2D input image.
        mask (np.ndarray, optional): 2D bad pixel mask (True=bad).
        config (IsosterConfig): Configuration object. Uses integrator, sclip,
            nclip, and use_eccentric_anomaly settings.
        template_isophotes (list of dict): List of isophote dicts from a
            previous isoster run. Each dict must contain 'sma', 'x0', 'y0',
            'eps', and 'pa' keys.

    Returns:
        dict: Results dictionary with 'isophotes' (list of dicts) and 'config'
            (IsosterConfig) keys. The isophotes have intensity from the target
            image but geometry from the template.

    Raises:
        ValueError: If template_isophotes is empty or None.

    Notes
    -----
    The output isophotes preserve the template's geometry exactly. Only the
    intensity-related fields (intens, rms, intens_err) and derived quantities
    come from the target image.

    This is designed for multiband analysis where one band (e.g., g-band)
    defines the geometry, and the same geometry is applied to other bands
    (r, i, z) for consistent color profile measurement.
    """
    from .fitting import extract_forced_photometry

    if not template_isophotes or len(template_isophotes) == 0:
        raise ValueError("template_isophotes cannot be empty")

    # Sort template by SMA for consistent ordering
    sorted_template = sorted(template_isophotes, key=lambda x: x['sma'])

    # Extract forced photometry at each SMA with its own geometry
    isophotes = []
    for template_iso in sorted_template:
        sma = template_iso['sma']

        # Handle central pixel (sma=0) specially
        if sma == 0:
            iso = fit_central_pixel(
                image, mask, template_iso['x0'], template_iso['y0'],
                debug=config.debug
            )
        else:
            iso = extract_forced_photometry(
                image, mask,
                template_iso['x0'], template_iso['y0'],
                sma,
                template_iso['eps'], template_iso['pa'],
                integrator=config.integrator,
                sclip=config.sclip,
                nclip=config.nclip,
                use_eccentric_anomaly=config.use_eccentric_anomaly
            )
        isophotes.append(iso)

    return _build_fit_result(isophotes, config)
