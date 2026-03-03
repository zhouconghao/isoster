---
name: mockgal
description: Generate mock galaxy images with multi-component Sersic profiles using the mockgal.py engine from isophote_test. Use when you need realistic synthetic data for isophote fitting tests or benchmarking stability in LSB regimes.
---

# MockGal Skill

This skill provides a standardized way to generate mock galaxy images using the `mockgal.py` engine and `profit-cli` backend.

## Environment

- **Engine**: `../isophote_test/mockgal.py`
- **Backend**: `libprofit` (via `profit-cli` at `../isophote_test/libprofit/mbp`)
- **Models**: Huang+2013 multi-Sersic models are available at `../isophote_test/inputs/huang2013/models/huang2013_models.yaml`.

## Usage

Use the provided wrapper script to run `mockgal.py` with the correct environment variables set.

### Single Galaxy Generation

```bash
./mockgal/scripts/mockgal_wrapper.sh --single \
    --name my_galaxy \
    -z 0.2 \
    --r-eff 5.0 \
    --abs-mag -21.0 \
    --sersic-n 4.0 \
    --psf --psf-type gaussian --psf-fwhm 0.7 \
    --sky-sb-limit 24.5 \
    -o output/
```

### Batch Generation from Huang2013

```bash
./mockgal/scripts/mockgal_wrapper.sh \
    --models ../isophote_test/inputs/huang2013/models/huang2013_models.yaml \
    --config ../isophote_test/inputs/huang2013/configs/huang2013_test_config.yaml \
    --galaxy "NGC 1453" "NGC 3585" \
    -o outputs/mock_galaxies
```

## Recommended HSC Configuration

For realistic HSC-like mock images (z=0.2):
- `pixel_scale`: 0.168
- `psf_fwhm`: 0.7
- `sky_sb_limit`: 24.5 (Wide survey) or 27.0 (Reference/Deep)
- `size_factor`: 15 (to cover outer disks)
