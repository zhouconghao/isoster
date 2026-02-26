# Testing and Benchmark Directives

Guidelines for writing tests and benchmarks in the isoster project.

## General Directives

- The canonical basic real-data dataset is `examples/data/m51/M51.fits`; the basic real-data test should be named `m51_test`.
- For Huang2013 workflows, use a fixed default initial SMA of `6.0` pixels (`sma0`) instead of deriving it from `RE_PX1`.
- For high-fidelity mock generation, use `examples/huang2013/mockgal.py` (libprofit-based) when PSF convolution and realistic background-noise controls are required.
- For `mockgal.py` benchmark/test workflows, force `--engine libprofit` and do not rely on astropy fallback rendering.
- For noiseless single-Sersic validation without PSF convolution, compare against an analytic 1-D Sersic truth using accurate `b_n` evaluation (e.g., `scipy.special.gammaincinv`) rather than low-accuracy approximations.
- Tests and benchmarks must be quantitative, with explicit statistics from 1-D profile deviations and 2-D residual diagnostics.
- Treat 2-D residual metrics as system-level diagnostics (profile extraction + model reconstruction combined), not as an isolated extraction-only metric.

## Mock Single-Sersic Model Tests

- The mock galaxy model shall be centered with the image array.
- The half-size of the mock image shall be at least 10 times of the effective radius of the mock model; if the mock model's effective radius is not very large, 15x would be even better.
- Pay attention to the oversampling ratio, high-Sersic index and high ellipticity often require higher oversampling in the central region.
- When comparing results between the truth or the reference profile:
  - Ignore the region smaller than 3 pixels when there is no PSF convolution because sampling the central region often has numerical issues; ignore the region `<= 2 * psf_fwhm` due to PSF convolution.
  - Ignore the region in the outskirt where problematic data points (using stop code) begin to appear or the intensity error bars become huge.
- Metrics to evaluate the results:
  - Median or maximum difference of a property between `0.5 * r_effective` (or 3 pixels, whichever is larger) to `8 * r_effective` is good for noiseless mocks; and to `5 * r_effective` is good for mocks with noise.
  - Using median or maximum absolute difference is a more strict standard.
  - `isoster` can provide the curve of growth measurement, the relative difference of the curve of growth values at a few typical radius could be useful metrics.

## Build Ellipse Model Tests

### Key Residual Statistics

- Fractional residual level: `100.0 * (model - data) / data`
- Fractional absolute residual level: `100.0 * |model - data| / data`
- Chi-Square statistics: `(model - data) ** 2.0 / (sigma ** 2)`

### Key Metrics

1. Statistics of the fractional residual level (e.g., median or maximum values) within different radial ranges. This works best for noiseless mocks.
2. Statistics of the integrated values of the fractional absolute residual level within different radial ranges. This works best for noiseless mocks.
3. Integrated Chi-square statistics within different radial ranges. This works best for noise-added mocks or real images.

### Radial Ranges

- < 0.5 Re (effective radius)
- 0.5-4 Re
- 4-8 Re
