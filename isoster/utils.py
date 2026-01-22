import numpy as np
from astropy.table import Table
from astropy.io import fits

def isophote_results_to_astropy_tables(results):
    """
    Convert isophote results to an Astropy Table.
    
    Parameters
    ----------
    results : dict
        The dictionary returned by fit_image(), containing 'isophotes' and 'config'.
        
    Returns
    -------
    table : astropy.table.Table
        Astropy Table containing the isophote data with standard columns:
        - sma: Semi-major axis length
        - intens: Mean intensity
        - intens_err: Intensity error
        - eps: Ellipticity
        - pa: Position angle (radians)
        - x0, y0: Center coordinates
        - stop_code: Fitting status code
        - niter: Number of iterations
    """
    # Handle case where results is just the list (backward compatibility or direct list usage)
    if isinstance(results, list):
        isophotes = results
    else:
        isophotes = results.get('isophotes', [])
    
    if not isophotes:
        return Table()
        
    # Create table from list of dicts
    tbl = Table(rows=isophotes)
    
    # Reorder columns for better readability
    # Common columns first
    common_cols = ['sma', 'intens', 'intens_err', 'eps', 'pa', 'x0', 'y0', 'rms', 'stop_code', 'niter']
    
    # Get all columns present
    all_cols = tbl.colnames
    
    # Create new order
    new_order = [c for c in common_cols if c in all_cols] + [c for c in all_cols if c not in common_cols]
    
    tbl = tbl[new_order]
    
    return tbl

def isophote_results_to_fits(results, filename, overwrite=True):
    """
    Save isophote results to a FITS table, including configuration parameters as header keywords.
    
    Parameters
    ----------
    results : dict
        The dictionary returned by fit_image().
    filename : str
        Output filename.
    overwrite : bool
        Whether to overwrite existing file.
    """
    tbl = isophote_results_to_astropy_tables(results)
    
    # Add config to header
    # We can add them to the table's meta, which writes to the primary header or table header
    if isinstance(results, dict):
        config = results.get('config', {})
        
        # Convert config object to dict if needed
        if hasattr(config, 'model_dump'):
            config_dict = config.model_dump()
        elif hasattr(config, 'dict'):
            config_dict = config.dict()
        elif isinstance(config, dict):
            config_dict = config
        else:
            config_dict = getattr(config, '__dict__', {})

        for key, value in config_dict.items():
            # FITS keywords should be short. We'll use the key as is, astropy handles it.
            # We filter out None values as they can't be written to FITS header easily
            if value is None:
                continue
                
            # Check value type
            if isinstance(value, (str, int, float, bool, np.number)):
                tbl.meta[key] = value
            else:
                # Convert to string if complex (e.g. list)
                tbl.meta[key] = str(value)
            
    tbl.write(filename, format='fits', overwrite=overwrite)


def isophote_results_from_fits(filename):
    """
    Load isophote results from a FITS table file.

    This is the inverse of isophote_results_to_fits(). It reconstructs
    the results dict from a saved FITS file, enabling template-based
    forced photometry workflows where geometry from one band is applied
    to other bands.

    Parameters
    ----------
    filename : str
        Path to FITS file containing isophote table.

    Returns
    -------
    results : dict
        Dictionary with:
        - 'isophotes': list of dicts, each containing isophote data
        - 'config': None (config reconstruction not fully supported)

    Notes
    -----
    The returned isophotes contain all columns saved in the FITS table.
    Config parameters stored in the header are not reconstructed into
    an IsosterConfig object to avoid potential validation issues with
    partial config data.

    Examples
    --------
    >>> # Load previously saved isophotes
    >>> template = isophote_results_from_fits('galaxy_gband.fits')
    >>> # Use as template for forced photometry
    >>> results_r = fit_image(image_r, mask_r, config,
    ...                       template_isophotes=template['isophotes'])
    """
    tbl = Table.read(filename, format='fits')

    # Convert table rows to list of dicts
    isophotes = []
    for row in tbl:
        iso_dict = {}
        for colname in tbl.colnames:
            value = row[colname]
            # Convert numpy types to Python types for consistency
            if isinstance(value, np.integer):
                value = int(value)
            elif isinstance(value, np.floating):
                value = float(value)
            elif isinstance(value, np.bool_):
                value = bool(value)
            iso_dict[colname] = value
        isophotes.append(iso_dict)

    # Note: We don't attempt to reconstruct config from header metadata
    # as this could cause validation issues with partial data
    return {'isophotes': isophotes, 'config': None}
