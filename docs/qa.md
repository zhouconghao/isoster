# QA Plotting Reference

This document covers the QA plotting functions exported by `isoster.plotting`.
For style conventions and layout rules, see [qa-figures.md](qa-figures.md).

## Quick Start

```python
import isoster
from isoster.config import IsosterConfig
from isoster.model import build_isoster_model

# Fit isophotes
config = IsosterConfig(x0=128, y0=128, sma0=10.0)
results = isoster.fit_image(image, mask, config)
isophotes = results["isophotes"]

# Build 2D model for residual display
model = build_isoster_model(image.shape, isophotes)

# Single-method QA figure
isoster.plot_qa_summary(
    title="NGC 1234",
    image=image,
    isoster_model=model,
    isoster_res=isophotes,
    mask=mask,
    filename="qa_summary.png",
)
```

## Functions

### `plot_qa_summary`

Standard QA figure for a single isoster fit.

**Layout**: left column shows the galaxy image with isophote overlays,
the reconstructed model, and the residual map; right column shows 1D
profiles (surface brightness, residual, ellipticity, PA, a3/b3, a4/b4)
sharing an SMA^0.25 x-axis.

```python
isoster.plot_qa_summary(
    title,              # Figure title
    image,              # 2D galaxy image
    isoster_model,      # 2D reconstructed model
    isoster_res,        # list[dict] from fit_image()["isophotes"]
    photutils_res=None, # optional photutils IsophoteList for comparison
    mask=None,          # 2D bool mask (True = masked)
    filename="qa_summary.png",
    relative_residual=False,  # True: show (model-data)/data [%]
)
```

### `plot_qa_summary_extended`

Extended QA figure with dedicated harmonic visualization panels.
Adds odd-order (3, 5, 7) and even-order (4, 6) harmonic panels below
the standard profile plots.

```python
isoster.plot_qa_summary_extended(
    title,
    image,
    isoster_model,
    isoster_res,
    harmonic_orders=None,       # auto-detect from isophote keys
    harmonic_mode="coefficients",  # "coefficients" or "amplitude"
    normalize_harmonics=False,  # show A_n/I when mode="amplitude"
    relative_residual=False,
    mask=None,
    filename="qa_summary_extended.png",
)
```

| `harmonic_mode` | Display |
|-----------------|---------|
| `"coefficients"` | Individual a_n (filled) and b_n (open) per order |
| `"amplitude"` | Combined A_n = sqrt(a_n^2 + b_n^2) per order |

### `plot_comparison_qa_figure`

Multi-method comparison figure with automatic layout selection.

```python
isoster.plot_comparison_qa_figure(
    image,                # 2D galaxy image
    profiles,             # dict[str, dict[str, ndarray]]
    title="",
    output_path="qa_comparison.png",
    models=None,          # dict[str, 2D ndarray]
    mask=None,
    method_styles=None,   # override METHOD_STYLES
    relative_residual=False,
    dpi=150,
)
```

**Automatic layout modes** (selected by number of methods in `profiles`):

| Methods | Mode | Left column |
|---------|------|-------------|
| 1 | Solo | Image, model, residual |
| 2 | One-on-one | Image, isoster residual, other residual |
| 3+ | Three-way | Image, one residual panel per method |

Right column (all modes): SB with errorbars, relative SB difference,
ellipticity, PA, centroid offset — sharing SMA^0.25 x-axis.

**Building profiles** for this function:

```python
from isoster.plotting import build_method_profile

# From isoster/photutils results (list of dicts)
profile = build_method_profile(isophotes)

# From autoprof results (dict of arrays)
profile = build_method_profile(autoprof_data)

# Inject optional metadata
profile["runtime_seconds"] = 0.15
profile["retries"] = 1

profiles = {"isoster": profile_iso, "photutils": profile_phot}
models = {"isoster": model_iso, "photutils": model_phot}

isoster.plot_comparison_qa_figure(image, profiles, models=models)
```

### `draw_isophote_overlays`

Draw isophote contours on a matplotlib axis.  When harmonic coefficients
(a3/b3, a4/b4, ...) are present, draws the actual non-elliptical shape.

```python
from isoster.plotting import draw_isophote_overlays

draw_isophote_overlays(
    axis,                 # matplotlib Axes
    isophotes,            # list[dict] with sma, x0, y0, eps, pa
    step=10,              # draw every N-th isophote
    line_width=1.0,
    alpha=0.7,
    edge_color=None,      # None = color by stop code
    draw_harmonics=True,  # False = force pure ellipses
)
```

When `draw_harmonics=True` (the default) and the isophote dicts contain
keys like `a3`, `b3`, `a4`, `b4`, the overlay traces the
harmonic-perturbed contour using 360 sample points.  This reveals
disky (b4 > 0) and boxy (b4 < 0) shapes directly on the image.
Set `draw_harmonics=False` to fall back to pure-ellipse patches.

## Helper Functions

| Function | Purpose |
|----------|---------|
| `build_method_profile(data)` | Convert list-of-dicts or array-dict into standardized profile arrays. Preserves harmonic keys. |
| `configure_qa_plot_style()` | Apply shared matplotlib rcParams for QA figures. |
| `derive_arcsinh_parameters(image)` | Compute arcsinh stretch parameters for display. |
| `make_arcsinh_display(image)` | Apply arcsinh stretch to an image for display. |
| `plot_profile_by_stop_code(ax, x, y, stop_codes, ...)` | Scatter plot with per-stop-code color and marker. |
| `normalize_pa_degrees(pa_deg, anchor=None)` | Unwrap PA jumps using the double-angle trick. |
| `style_for_stop_code(code)` | Return color/marker/label dict for a stop code. |

## Method Styles

The `METHOD_STYLES` dict defines default visual styles for each method
in comparison figures:

| Method | Color | Marker | Overlay color |
|--------|-------|--------|---------------|
| isoster | `#1f77b4` (blue) | `o` filled | white |
| photutils | `#d62728` (red) | `s` open | orangered |
| autoprof | `#2ca02c` (green) | `^` open | limegreen |

Override per-call via the `method_styles` parameter.

## Synthetic Comparison Script

`examples/example_qa_comparison/generate_comparison_figures.py` runs
isoster (and optionally photutils) on synthetic Sersic galaxies with
known truth, then generates QA figures using `plot_comparison_qa_figure`.

### CLI

```bash
# Run all preset cases
uv run python examples/example_qa_comparison/generate_comparison_figures.py

# Single case
uv run python examples/example_qa_comparison/generate_comparison_figures.py \
    --case n4_eps04_snr100_noisy

# Custom output directory, skip photutils
uv run python examples/example_qa_comparison/generate_comparison_figures.py \
    --output outputs/my_qa_run --no-photutils
```

### Preset cases

| Name | n | eps | SNR | Notes |
|------|---|-----|-----|-------|
| `n1_eps07_high_ellipticity` | 1.0 | 0.7 | inf | Stress test for high ellipticity |
| `n1_eps04_snr100_noisy` | 1.0 | 0.4 | 100 | Exponential disk with noise |
| `n4_eps04_snr100_noisy` | 4.0 | 0.4 | 100 | de Vaucouleurs profile with noise |

### Reuse from other benchmarks

Import `PRESET_CASES` and `run_single_case()` to run comparisons
from any benchmark script:

```python
from examples.example_qa_comparison.generate_comparison_figures import (
    PRESET_CASES, run_single_case,
)

# Run a preset case
stats = run_single_case(
    PRESET_CASES["n4_eps04_snr100_noisy"],
    output_dir="outputs/my_benchmark",
    case_name="devaucouleurs_noisy",
)

# Define a custom case
custom_case = {
    "n": 2.0, "R_e": 30.0, "I_e": 500.0,
    "eps": 0.5, "pa": 0.8, "snr": 50,
    "oversample": 5, "shape": (800, 800),
}
stats = run_single_case(
    custom_case,
    output_dir="outputs/my_benchmark",
    case_name="custom_n2",
    isoster_config_overrides={"maxit": 100},
)

# stats keys: case_name, n_isophotes, wall_time_seconds,
#   median_frac_resid, max_abs_frac_resid, convergence_rate
```

## Stop Code Colors

Isophote overlays and profile scatter plots are colored by stop code
when no explicit `edge_color` is given:

| Code | Meaning | Color |
|------|---------|-------|
| 0 | Converged | Blue (`#1f77b4`) |
| 1 | Too many flagged pixels | Orange (`#ff7f0e`) |
| 2 | Max iterations reached | Green (`#2ca02c`) |
| 3 | Too few valid points | Red (`#d62728`) |
| -1 | Gradient error / strike | Purple (`#9467bd`) |
