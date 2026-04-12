# HSC Edge Cases

Challenging real-data test cases from HSC coadd images for evaluating isoster behavior under difficult conditions.

## Cases

| ID | Description | Challenge |
|---|---|---|
| 10140088 | Clear case | Baseline — clean galaxy with minimal contamination |
| 10140002 | Nearby bright star | Bright star halo overlapping the galaxy outskirts |
| 10140006 | Nearby large galaxy | Extended neighbor affecting isophote geometry |
| 10140009 | Blending bright star | Star blended with the galaxy core region |
| 10140056 | Artifact | Image artifact contaminating the field |
| 10140093 | Small blending source | Small source blended with galaxy isophotes |

## Data Extraction

The source data is in HDF format. Run the extraction script to produce per-galaxy FITS files:

```bash
uv run python examples/example_hsc_edgecases/extract_hdf_to_fits.py \
    --hdf-dir /path/to/hdf/files
```

This produces per-galaxy subdirectories under `data/` with:
- `{id}_{band}_image.fits` — science image (HSC-G, HSC-R, HSC-I)
- `{id}_{band}_variance.fits` — variance map
- `{id}_{band}_psf.fits` — PSF image
- `{id}_mask_combined.fits` — combined object mask (True = bad pixel)
- `{id}_mask_layers.fits` — multi-extension FITS with individual mask layers

## Object Mask Construction

The combined object mask is built from the HDF bitplane mask (`MASK_BIT`):

```
object_mask = (ITER & ~UNSHARPED) | BRIGHT | BADPIX | EXTREME
```

- **ITER** (bit 32): all detected objects (full detection footprint)
- **UNSHARPED** (bit 64): the target galaxy footprint (excluded from mask)
- **BRIGHT** (bit 8): Gaia bright star halos
- **BADPIX** (bit 4): bad/dead pixels
- **EXTREME** (bit 16): extreme-value pixels

This masks neighboring sources while keeping the target galaxy unmasked.
