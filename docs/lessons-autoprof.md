# AutoProf Lessons

Lessons learned from integrating AutoProf into the benchmark suite.

## PA Convention

- **isoster**: radians, math convention (CCW from +x axis)
- **AutoProf init** (`ap_isoinit_pa_set`): degrees, astro convention (CCW from +y)
- Conversion: `pa_autoprof_deg = degrees(pa_isoster_rad) - 90`
- AutoProf internally applies `PA_shift_convention` during processing

## Bypassing Automatic Steps

AutoProf's pipeline includes automatic background subtraction, PSF estimation,
and center finding.  For fair benchmarking, these must be bypassed:

- `ap_set_center`: dict `{'x': x0, 'y': y0}` — skips center finding
- `ap_set_background`: float — skips background estimation
- `ap_set_background_noise`: float — skips noise estimation
- These parameters replace the pipeline steps entirely (not just initial guesses)

## Profile Output Format

- AutoProf `.prof` files use `ascii.commented_header` format (astropy-readable)
- Columns: `R` (arcsec), `SB` (mag/arcsec^2 or flux/arcsec^2), `ellip`, `pa` (deg)
- SMA in arcsec must be converted to pixels: `sma_px = R / pixel_scale`
- Intensity in flux/arcsec^2 must be converted: `intens_px = I * pixel_scale**2`

## Logging

- AutoProf writes a `AutoProf.log` file in the current directory by default
- Always pass `loggername=str(output_dir / "AutoProf.log")` to redirect
- The logger name must be a string, not a Path object

## Environment Incompatibility

AutoProf 1.3.4 depends on photutils 1.5.0 which requires numpy <2.  isoster's
environment uses numpy 2.x.  **These cannot coexist in the same virtualenv.**

Solution: run AutoProf via subprocess using the system Python (miniforge3) where
AutoProf is installed with numpy 1.26.  The adapter (`autoprof_adapter.py`)
spawns a subprocess, runs the fit, and parses the `.prof` output file back in
the isoster environment.

Set `AUTOPROF_PYTHON=/path/to/python` to override the default system Python.

## Pipeline API

The correct import path is `autoprof.Pipeline`, **not** `autoprof.pipeline_steps`:

```python
from autoprof.Pipeline import Isophote_Pipeline
pipeline = Isophote_Pipeline(loggername=str(log_path))
pipeline.Process_Image(options_dict)
```

## Profile File Format

AutoProf `.prof` files have a two-line header:
- Line 1: `# units` (comma-separated, e.g. `# arcsec,flux*arcsec^-2,...`)
- Line 2: column names (e.g. `R,I,I_e,totflux,totflux_e,ellip,ellip_e,pa,pa_e,...`)

Parse with `astropy.table.Table.read(path, format="ascii.csv", comment="#")`.
The `ascii.commented_header` format does **not** work because line 2 is not
preceded by `#`.

## Known Limitations

- AutoProf may struggle with highly elliptical / edge-on galaxies
- No per-isophote stop codes — convergence is global
- Output profiles may have different SMA sampling than isoster
- `ap_fluxunits='intensity'` ensures raw flux output (not magnitudes)
- At large radii, AutoProf's FFT approach tends to produce flatter (more
  regularized) ellipticity and PA profiles compared to isoster's per-isophote
  fitting — this is expected behavior, not a bug
