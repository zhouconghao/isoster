# AutoProf vs Isoster: Algorithmic Comparison and Feasibility Analysis

## Executive Summary

This document provides a detailed comparison between **AutoProf**'s isophote fitting pipeline (from `src/autoprof/pipeline_steps`) and **isoster**'s core algorithms. The analysis covers initialization, fitting, extraction/photometry, and 2D model reconstruction, followed by an assessment of feasibility for implementing AutoProf-style algorithms as optional modes in isoster.

**Key Finding**: AutoProf and isoster use fundamentally different algorithmic approaches:
- **AutoProf**: Global FFT-based objectives with explicit radial regularization, stochastic optimization across all isophotes simultaneously
- **isoster**: Local harmonic/gradient-based per-isophote fitting with deterministic Newton-style updates

Both approaches have strengths; porting selected AutoProf ideas into isoster as **opt-in modes** is feasible and would broaden capabilities without sacrificing isoster's current performance advantages.

---

## 1. Core Algorithmic Differences

### 1.1 Initialization: Global PA/Ellipticity Estimation

#### AutoProf (`Isophote_Initialize.py`)

**Strategy**: Derives **global** position angle and ellipticity from the image itself using **Fourier analysis of circular isophotes**.

**Algorithm**:
1. **Circular isophote sampling**:
   - Builds sequence of circular radii `circ_ellipse_radii` starting from 1 pixel, growing geometrically (factor 1.2) until reaching background noise level
   - Stops when 80th percentile flux drops below `(ap_fit_limit+1) * background_noise`
   - For each radius, extracts intensity values along circle using `_iso_extract` with sigma clipping

2. **Global PA estimation**:
   - Computes FFT of intensity array along each circular isophote
   - Stores 2nd Fourier coefficient `coefs[2]` in `allphase` list
   - Uses **phase of 2nd harmonic** from outer 5 radii: `phase = (-Angle_Median(np.angle(allphase[-5:])) / 2) % π`
   - This phase directly indicates the major axis direction of the elliptical galaxy

3. **Global ellipticity optimization**:
   - Defines loss function `_fitEllip_loss(e, ...)`:
     - Extracts elliptical isophote values at test ellipticity `e` and fixed PA `p`
     - Robustly clips intensities at 85th percentile
     - Computes FFT and combines IQR and |F₂| normalized by median and noise:
       ```
       loss = (IQR/2 + |F₂|/N) / (max(0, median) + noise)
       ```
   - Two-pass optimization:
     - Coarse grid search: test `e ∈ [0.05, 0.95]` at multiple radii (0.8× to 1.2× reference radius)
     - Fine Nelder-Mead minimization using transformed parameter space (`_x_to_eps` / `_inv_x_to_eps`)

4. **Error estimation**:
   - **PA error**: Standard deviation of phase solutions when varying radius by ±PSF FWHM
   - **Ellipticity error**: IQR of best-fit ellipticities from multiple nearby radii

**Variants**:
- `Isophote_Initialize_mean`: Uses mean/std instead of robust IQR for low-count data
- `Isophote_Init_Forced`: Reads global ellipse from `.aux` file (forced photometry mode)

#### isoster (`driver.fit_image` + config)

**Strategy**: **Configuration-driven** initialization; no dedicated global FFT-based search.

**Algorithm**:
- Initial geometry from `IsosterConfig`:
  - `x0, y0`: Center coordinates (default: image center)
  - `eps, pa`: Initial ellipticity and position angle (from config, not derived)
  - `sma0`: Starting semi-major axis (config parameter)
- Pipeline:
  1. Fit central pixel at `(x0, y0)` for SMA=0
  2. Fit first isophote at `sma0` using `fit_isophote(image, mask, sma0, start_geometry, cfg)`
  3. Grow outward/inward from there, using previous isophote geometry as initial guess

**Key Conceptual Difference**:
- **AutoProf**: Global, image-driven initialization using FFT of circular isophotes; explicitly solves for PA and ellipticity before any radial fitting
- **isoster**: Local, config-driven initialization; relies on good priors and per-isophote corrections to refine geometry

---

### 1.2 Core Isophote Fitting Algorithms

#### AutoProf (`Isophote_Fit.py`) - FFT/Regularization Based

**Main Algorithm: `Isophote_Fit_FFT_Robust`**

1. **Radius selection**:
   - Start near PSF scale; grow geometrically with factor `(1 + scale/(1+shrink))`
   - Stop when median intensity drops below `ap_fit_limit * background_noise`
   - Enforce minimum ~15 radii; reduce scale if too few found

2. **Parameterization**:
   - For each radius `R[i]`, store dict with:
     - `ellip`, `pa` (ellipticity, position angle)
     - Optional `C` (superellipse exponent; `C=2` is pure ellipse)
     - Optional `m`, `Am`, `Phim` (Fourier shape mode indices, amplitudes, phases)
   - All radii stored in `parameters` list, **coupled together**

3. **Loss function `_FFT_Robust_loss`**:
   - Extract isophote values with `_iso_extract` (bicubic interpolation, mask support)
   - Robust clipping at `(1 - robust_clip)` quantile (default 0.85)
   - Compute FFT of clipped intensities
   - **Data term**: Minimize relative 2nd FFT mode (or average of specified `fit_coefs`):
     ```
     f2_loss = |F₂| / (N * (max(0, median) + noise/√N))
     ```
   - **Regularization term**: L1 penalties on differences between neighboring radii:
     - Ellipticity: `|Δeps| / (1 - eps_neighbor)`
     - PA: `|sin(Δpa)| / 0.2`
     - Fourier modes: Penalties on `Am` differences and phase differences
     - Superellipse C: `|log₁₀(C₁/C₂)| / 0.1`
   - **Total loss**: `f2_loss * (1 + reg_loss * reg_scale)`
   - This **couples all radii** and promotes smooth profiles

4. **Optimization**:
   - **Stochastic local search** across all radii:
     - For each iteration:
       - Shuffle radius indices
       - For each radius `i`:
         - Evaluate current parameters (baseline)
         - Generate `N_perturb` random perturbations of **one parameter family**:
           - Cycle through: ellip, pa, C, then each `Am[m]`, `Phim[m]`
         - Compute loss for each candidate
         - Accept best if better than baseline
     - Convergence: Stop when no updates for `iterstopnochange * (len(R)-1)` iterations
   - **Progressive parameter activation**:
     - First: fit only (ellip, pa)
     - Then: activate superellipse C if requested
     - Then: activate Fourier modes if requested
     - Optionally initialize Fourier amplitudes from FFT of current isophotes

5. **Error estimation**:
   - For each radius, re-fit with perturbed radii (±5%)
   - Compute IQR of resulting ellipticities and PA scatter

**Other modes**:
- `Isophote_Fit_FFT_mean`: Simpler variant for low-S/N data (uses mean/std instead of robust stats)
- `Photutils_Fit`: Thin wrapper around `photutils.isophote.Ellipse.fit_image`
- `Isophote_Fit_FixedPhase`: Uses fixed global PA/ellip; only selects radii

#### isoster (`fitting.py`) - Harmonic/Gradient Based, Per-Isophote

**Core Algorithm: `fit_isophote`**

1. **Sampling** (`sampling.extract_isophote_data`):
   - Vectorized sampling along ellipse using `compute_ellipse_coords` + `map_coordinates`
   - Returns `IsophoteData(angles, phi, intens, radii)`:
     - `angles`: ψ (EA mode) or φ (regular) for harmonic fitting
     - `phi`: Position angle for geometry updates
   - Optional **eccentric anomaly (EA) mode** (Ciambur 2015) for high ellipticity

2. **Per-isophote fitting**:
   - Extract `(angles, phi, intens)` via `extract_isophote_data`
   - Apply **sigma clipping** in angle-intensity space
   - Fit **1st and 2nd harmonics** via linear least squares:
     ```
     I(θ) = y₀ + A₁sin(θ) + B₁cos(θ) + A₂sin(2θ) + B₂cos(2θ)
     ```
   - Compute **radial gradient** `dI/dsma` via `compute_gradient`:
     - Sample at `sma` and neighbor(s) (1-2 steps)
     - Use mean/median intensities to estimate gradient and error
     - Includes heuristics to avoid noisy gradients (relative error > `maxgerr`)

3. **Convergence check**:
   - Build `harmonics = [A₁, B₁, A₂, B₂]` (zero out if `fix_center/pa/eps` flags set)
   - `max_amp = max(|A₁|, |B₁|, |A₂|, |B₂|)`
   - Apply **central regularization penalty** (decaying near center) to discourage large geometry changes
   - Keep geometry with smallest "effective amplitude" as `best_geometry`
   - Converge when `|max_amp| < conver * rms` after at least `minit` iterations

4. **Geometry updates** (Jedrzejewski-style):
   - Depending on which harmonic dominates, update `(x0, y0, pa, eps)` using analytic formulas involving `gradient`, `eps`, `sma`
   - Includes safeguards for extreme ellipticity and PA flips

5. **Higher-order harmonics** (optional):
   - On convergence, compute `a₃, b₃, a₄, b₄, ...` either:
     - Sequentially: `compute_deviations(angles, intens, sma, gradient, n)`
     - Simultaneously: `fit_higher_harmonics_simultaneous(angles, intens, sma, gradient, orders)`
   - These are **diagnostics** for modeling, not used in geometry updates

**Coupling across isophotes**:
- Each SMA uses previous gradient as prior and optional central regularization vs previous geometry
- **No joint optimization** over all SMAs; smoothness arises from per-radius convergence and local inheritance

#### Summary of Fitting Differences

| Aspect | AutoProf | isoster |
|--------|----------|---------|
| **Optimization variable** | Geometry across **all isophotes simultaneously** | Geometry **per isophote** |
| **Objective function** | FFT amplitude of selected modes + **explicit smoothness regularization** | Harmonic coefficients relative to RMS + gradient constraints |
| **Solver** | **Stochastic, coupled, non-linear** (simulated-annealing-like) | **Deterministic, per-radius, Newton-like** |
| **Geometry models** | Superellipses + arbitrary Fourier modes built-in | Higher harmonics as diagnostics only |
| **Coupling** | Explicit L1 regularization between neighbors | Implicit via gradient inheritance and central regularization |

---

### 1.3 Profile Extraction and Curve of Growth

#### AutoProf (`Isophote_Extract.py`)

**Input**: Geometry from fit (`fit R`, `fit ellip`, `fit pa`, errors, optional Fmodes, C) or forced profile file

**Sampling radii**:
- Flexible sampling styles: `'linear'`, `'geometric'`, `'geometric-linear'`
- Start: `ap_sampleinitR` or `min(1 px, 0.5*PSF)`
- End: `ap_sampleendR` or `3 * fitR_max` or full image if `ap_extractfull`

**Geometry interpolation**:
- Use `UnivariateSpline` on `sin(2*pa)`, `cos(2*pa)` to handle wrap-around
- Ellipticity via spline on `_inv_x_to_eps` (numerically stable)
- Interpolate Fourier modes `A_m(R)`, `Phi_m(R)` and superellipse `C(R)` if present

**Flux extraction** (`_Generate_Profile`):
- For each radius:
  - **Line vs band sampling**:
    - If median flux > `ap_isoband_start * background_noise`: sample line via `_iso_extract`
    - Otherwise: switch to band `_iso_between(R±ΔR)` with `ΔR` from `ap_isoband_width`
  - Compute:
    - `medflux` via configurable method: `'median'|'mean'|'mode'`
    - `scatflux` via corresponding scatter estimator
    - `isotot` via `_iso_between(0, R[i])` (direct CoG sum)
  - Optionally measure **Fourier coefficients** along isophote (`ap_iso_measurecoefs`)
  - Track `pixels`, `maskedpixels`

**Units and CoG**:
- If `fluxunits == "intensity"`: SB = `medflux / pixscale²`; CoG via `Fmode_fluxdens_to_fluxsum_errorprop`
- If `fluxunits == "mag"`: SB in mag/arcsec²; CoG via `SBprof_to_COG_errorprop`
- Output: `prof header`, `prof units`, `prof data` (dictionary of column lists)

#### isoster

**Strategy**: Fitting and photometry are **fused**; no separate "extract" stage

- Each isophote dict contains:
  - `sma`, `x0, y0, eps, pa`, `intens`, `rms`, `intens_err`
  - Optional `a₃, b₃, a₄, b₄, ...` and errors
  - If `full_photometry`: `tflux_e`, `tflux_c`, `npix_e`, `npix_c`

**CoG**:
- `cog.compute_cog` takes isophote list and computes CoG
- `add_cog_to_isophotes` attaches CoG fields back to isophotes
- Simpler integration strategy than AutoProf's full SB-profile-based approach

#### Extraction Differences

| Aspect | AutoProf | isoster |
|--------|----------|---------|
| **Flexibility** | Very configurable: sampling mode, band vs line, averaging method, Fourier measurements | Streamlined: tied to fitting loop, limited but robust options |
| **Output format** | Explicit SB profile table with units metadata | Isophote list with embedded photometry |
| **CoG computation** | Two methods: direct sum + analytic integration from SB profile | Single method from isophote list |
| **Use case** | Rich control for experimentation and external pipelines | High-throughput, pipeline-embedded extraction |

---

### 1.4 2D Model Reconstruction

#### AutoProf (`Ellipse_Model.py`)

**Inputs**: SB profile + geometry (PA, ellip, optional C, Fmodes) from `results["prof data"]`

**Algorithm**:
1. **Filter and spline construction**:
   - Filter to radii where `SB_e < 0.5` mag (avoid noisy outer regions)
   - Build splines: `sb(R)`, `pa(R)`, `q(R) = 1-ellip(R)`, optional `C(R)`
   - If Fmodes: splines for `A_m(R)`, `Phi_m(R)`; compute `Rlimscale ≈ exp(sum |A_m(R_end)|)`

2. **Pixel-wise radius computation**:
   - Define ROI around galaxy center
   - For each pixel: compute `(XX, YY)`, `Radius`, `theta = atan2(YY, XX)`

3. **Multi-radius "nearest ellipse" loop**:
   - Iterate radii `r` in logspace from `R[0]/2` to `R[-1]` (resolution controlled by `ap_ellipsemodel_resolution`)
   - For each `r`:
     - Compute local geometry: `pa(r)`, `q(r)`, `C(r)`
     - Evaluate **superellipse radius scaling**: `Rscale_SuperEllipse(theta - pa(r), 1-q(r), C(r))`
     - If Fmodes: multiply by `exp(sum_m A_m(r) * cos(m*(theta + Phi_m(r) - pa(r))))`
     - Compute **effective elliptical radius**: `RR = Radius / Rscale`
     - For each pixel: if `|RR - r|` is smaller than current best, update best radius `MM` and proximity `Prox`

4. **Intensity assignment**:
   - Convert radius map to SB: `MM_sb = sb(MM)`
   - Convert SB to flux: `F = 10^(-(SB - zeropoint - 5*log₁₀(pixscale))/2.5)`
   - Truncate: set model to zero for `RR > R[-1]` (outside outer isophote)
   - Embed ROI into full-frame array

**Outputs**: Model FITS file, optional mask-replaced image, diagnostic plots

#### isoster (`model.build_isoster_model`)

**Inputs**: List of isophote dicts with `sma, x0, y0, eps, pa, intens` and optional harmonics

**Algorithm**:
1. **Filter and interpolation**:
   - Filter isophotes with `sma > 0`; handle central pixel separately
   - Build 1D interpolators: `intens(sma)`, `x0(sma)`, `y0(sma)`, `eps(sma)`, `pa(sma)`

2. **Approximate elliptical radius**:
   - Start with approximate radius using outermost geometry
   - Iterate 2-3 times:
     - Evaluate local geometry at current `r_ell`
     - Compute `r_ell_new = sqrt(x_rot² + (y_rot/(1-eps))²)`
     - Stop when max change < 0.1 px

3. **Harmonics** (if requested):
   - For each harmonic order `n`:
     - Build interpolators for `a_n(sma)`, `b_n(sma)`
   - Compute position angle: `theta = atan2(y_rot, x_rot)`
   - For each pixel:
     - Evaluate `Δr/r = Σ_n [a_n(sma) sin(nθ) + b_n(sma) cos(nθ)]`
     - Correct radius: `r_corrected = r_ell * (1 - Δr/r)`

4. **Intensity assignment**:
   - Interpolate: `I = intens_interp(r_effective)`
   - Fill inside smallest isophote with inner boundary intensity
   - Fill outside largest with `fill` value

#### Model Differences

| Aspect | AutoProf | isoster |
|--------|----------|---------|
| **Geometry models** | Superellipse + full Fourier distortions | Elliptical + harmonic radial corrections |
| **Radius assignment** | Nearest-ellipse search over log-spaced radii | Direct computation with iterative refinement |
| **Computation** | Heavier: loop over radius grid with windowed regions | Efficient: few global passes |
| **Truncation** | Hard cutoff at outer isophote | Smooth fill with configurable value |

---

## 2. Pros and Cons Analysis

### 2.1 Initialization: FFT Global vs Config-Based

#### AutoProf FFT Global Init

**Pros**:
- ✅ Robust to poor user guesses: derives PA/ellip from **signal-dominated outer isophotes**
- ✅ Uses **phase of F₂** in circular isophotes (physically motivated, relatively stable)
- ✅ Built-in **error estimates** using PSF-scale variations
- ✅ Grid + Nelder-Mead is simple and robust for 1D ellipticity search

**Cons**:
- ❌ Extra **computational cost** upfront (FFT for each radius, multiple loss evaluations)
- ❌ Assumes well-behaved elliptical galaxy; may be biased by strong asymmetries, bars, bright structures
- ❌ More complexity and hyperparameters (fit limit, PSF, etc.)

#### isoster Config-Based Init

**Pros**:
- ✅ Very **simple and fast**
- ✅ Appropriate when approximate center/geometry already known (e.g., from detection pipeline)
- ✅ Less machinery, fewer failure modes

**Cons**:
- ❌ If `eps, pa` config is poor, first isophote may fail or converge to bad local minimum
- ❌ No automatic global check that initial geometry matches image's large-scale structure

---

### 2.2 Fitting: FFT Regularized Global vs Local Harmonic-Gradient

#### AutoProf FFT-Robust Fitting

**Pros**:
- ✅ **Global regularization**: Explicit smoothness on `ellip`, `pa`, `C`, F-modes → smooth, physically plausible profiles
- ✅ Robust to **outliers/masks**: Robust clipping + regularization; loss normalized by median+noise
- ✅ Very **flexible geometry**: Superellipses and arbitrary Fourier modes built into core solver
- ✅ Less sensitive to gradient issues: Doesn't rely directly on `dI/dsma`; can handle noisy/flat gradients

**Cons**:
- ❌ **High algorithmic complexity**: Many hyperparameters (scales, regularization, perturb scales, iteration limits)
- ❌ **Non-deterministic**: Random perturbations → harder to reproduce exactly
- ❌ Harder to reason about **convergence guarantees**: Heuristics (no-change iterations) rather than clear Newton steps
- ❌ Stochastic search may require tuning or long runtimes for large/complex galaxies

#### isoster Harmonic/Gradient Fitting

**Pros**:
- ✅ **Principled, classical algorithm** (Jedrzejewski + Ciambur EA) with well-understood behavior
- ✅ **Analytic relationships**: Clear mapping from A₁/B₁/A₂/B₂ to geometry corrections
- ✅ Direct use of **radial gradient** gives physical scaling of corrections
- ✅ **Deterministic and fast**: Linear least squares per isophote; gradient evaluations reuse sampling
- ✅ **EA-mode support** for high ellipticity (Ciambur 2015 best practice)
- ✅ Higher harmonics readily available for modeling/QA

**Cons**:
- ❌ More sensitive to **gradient quality**: When gradient is small/noisy, corrections blow up or stop (handled with heuristics, but delicate)
- ❌ **Weaker coupling** across radii: No explicit joint loss penalizing jagged profiles; smoothness from per-radius convergence and local inheritance
- ❌ No built-in **superelliptic shapes** or arbitrary Fourier distortions; harmonics measured, not fitted jointly with geometry

---

### 2.3 Extraction / CoG

#### AutoProf

**Pros**:
- ✅ Very **rich control surface**: Radius sampling, band vs line, sigma clipping, averaging method, Fourier measurements
- ✅ Detailed error models and unit metadata
- ✅ SB profiles and CoG as **explicit tables** → convenient for external pipelines, plotting, post-processing
- ✅ Band sampling in low-S/N regime helpful to avoid numerical issues

**Cons**:
- ❌ Complexity and many options → larger configuration space; misconfiguration can bias results
- ❌ Some features may be overkill for users only needing basic isophote geometry + photometry

#### isoster

**Pros**:
- ✅ Simpler, **pipeline-embedded** extraction: Intensities and CoG as extension of fitting
- ✅ Harder to misconfigure; easier to test and maintain
- ✅ Well-aligned with high-throughput use (batch runs)

**Cons**:
- ❌ Less flexible for **experimentation**: No explicit band vs line, limited averaging statistics, no direct "measure Fourier modes along isophote" separate from fit
- ❌ CoG less "exposed" as first-class data product than AutoProf's SB table

---

### 2.4 Model Reconstruction

#### AutoProf EllipseModel

**Pros**:
- ✅ Strongly **geometry-driven**: Captures full radial dependence of `q(R)`, `pa(R)`, `C(R)`, F-modes
- ✅ **Nearest-ellipse** assignment across log-spaced radii → resolves structure on multiple scales, smooth transitions
- ✅ Naturally incorporates superelliptic and F-mode distortions (boxiness, lopsidedness, etc.)
- ✅ Ability to replace masked zones with model values → practical for image cleaning

**Cons**:
- ❌ Heavier computation: Repeated loop over radius grid with windowed regions, per-pixel operations
- ❌ Additional complexity from superelliptic and F-mode logic → more chances for subtle bugs
- ❌ Thresholding and `Rlimscale` heuristics may need tuning for very structured galaxies

#### isoster build_isoster_model

**Pros**:
- ✅ Conceptually clean: Define radius in varying-geometry elliptical coordinates, **interpolate intensity** radially
- ✅ Efficient: Small number of global passes; few iterations of local geometry refinement
- ✅ Easy to reason about and test
- ✅ Already close in spirit to AutoProf but with fewer knobs

**Cons**:
- ❌ Currently limited to harmonics a₃/b₃, a₄/b₄, ... as radial corrections; no direct support for superelliptic shapes (C parameter)
- ❌ Lacks explicit use of SB errors to gate radii (could ignore noisy outer isophotes, but that's an extra policy decision)

---

## 3. Feasibility of Implementing AutoProf-Style Algorithms in isoster

### Overall Assessment: **Yes, feasible** ✅

AutoProf-style algorithms can be implemented as **optional modes** or **alternative backends** in isoster without requiring core rewrites. They map cleanly to configuration switches and new solver functions.

---

### 3.1 Global FFT-Based Initialization

**Feasibility**: ✅ **High - Low Risk**

**What to Port**:
- Add optional initialization backend that:
  - Builds list of circular radii up to S/N criterion (using `IsosterConfig` fields for background, noise, PSF FWHM)
  - Uses isoster's own ellipse sampling (`extract_isophote_data` with `eps=0`) to extract isovals
  - Computes FFT of isovals; derives global PA from phase of F₂ (median of outer radii)
  - Performs 1D ellipticity search at representative radius using AutoProf-like loss (IQR/median + |F₂|) and Nelder-Mead
  - Produces `init_eps`, `init_pa`, `init_R` and errors

**Integration Points**:
- In `driver.fit_image`, before `fit_central_pixel` and `fit_isophote`:
  - If `config.init_mode == "fft_global"`: Run FFT-init routine and override `cfg.eps`, `cfg.pa`, potentially `cfg.sma0`
  - Otherwise: Keep current config-based behavior

**Complexity / Risk**:
- **Contained**: Doesn't modify main isophote solver, only starting geometry
- **Runtime**: FFTs along 10-20 circles are cheap compared to full fitting
- **Recommendation**: Keep clearly optional and off by default

---

### 3.2 AutoProf-Style Coupled/Robust Fitting Backend

**Feasibility**: ✅ **Medium-High - Medium Risk**

**What to Port**:
- Implement alternative solver that:
  - Defines shared `sample_radii` list (similar to AutoProf) using isoster's config for `astep` / PSF / thresholds
  - Constructs `parameters` list per radius with `x0, y0, eps, pa` (optionally C and harmonics)
  - Uses **isoster's fast sampling** to produce intensity arrays along each ellipse, then FFT them
  - Implements `_FFT_Robust_loss` (data + regularization) using isoster's parameter structure
  - Runs stochastic or modern optimizer (coordinate-wise random search, CMA-ES) over `(eps, pa, ...)` across all radii

**Integration Pattern**:
- Add solver switch in config:
  - `config.solver = "jedrzejewski"` (default) or `"autoprof_fft"`
- Implement:
  - `fit_isophote_jedrzejewski` (current, renamed)
  - `fit_isophotes_autoprof_style(image, mask, cfg)` → returns isophote list in isoster's dict format
- `driver.fit_image` branches on `solver` and builds `isophotes` list accordingly

**Trade-offs**:
- ✅ **Gains**: Option for AutoProf-like behavior (coupled solutions, superellipse/F-modes, robust FFT)
- ⚠️ **Costs**: Complexity, runtime, reproducibility; would need strong tests vs AutoProf before enabling by default

---

### 3.3 Richer Extraction / Photometry Options

**Feasibility**: ✅ **High - Low Risk**

**What to Port**:
- Extend or parallel existing CoG/photometry stack:
  - Allow band-based extraction around each SMA (similar to `_iso_between` vs `_iso_extract`)
  - Add config flags for averaging method (`mean/median/mode`), sigma-clipping iterations, Fourier-mode measurement along isophotes
  - Return profile object/table similar to AutoProf's `prof header/prof units/prof data`, while still producing current isophote list

**Integration**:
- Add helper: `build_sb_profile(isophotes, image, mask, config)` that:
  - Uses isoster's sampling machinery to re-sample along fitted geometry (for bands)
  - Matches AutoProf's column semantics
- CoG routines could either:
  - Keep using existing `compute_cog`, or
  - Gain alternative "profile-based CoG" more like AutoProf's

**Recommendation**: **High value, low risk** - Many users (and QA/benchmarking) would benefit from AutoProf-like SB tables and band sampling

---

### 3.4 Superellipse and Fourier-Shape Support in `build_isoster_model`

**Feasibility**: ✅ **Medium - Medium Risk**

**What to Port**:
- Extend isophote dicts to carry optional fields:
  - `C` (superellipse exponent vs radius)
  - Possibly `Fmodes` (`m`, `Am`, `Phim`) in AutoProf-like representation, or re-interpret existing `a_n, b_n` harmonics into equivalent radius-scaling factors
- Implement analogues of:
  - `Rscale_SuperEllipse(theta, eps, C)`
  - `Rscale_Fmodes(theta, A_m, Phi_m, m)`
- Replace or augment current harmonic Δr/r correction with these radius scalings when flags set

**Integration**:
- `build_isoster_model(..., use_superellipse=False, use_fmodes=False, ...)`:
  - If disabled: Current behavior unchanged
  - If enabled: Follow AutoProf-like Rscale construction to derive `r_effective` for each pixel

**Risks / Constraints**:
- Need consistent definitions of `C` and F-modes between fit and model (if adopting AutoProf-like fitter)
- **Important**: Don't have to tie this to Jedrzejewski solver; can treat C and F-modes purely as modeling extras at first

---

### 3.5 Overall Recommendations

#### High Value, Lower Risk (Implement Soon) ✅

1. **FFT-based global initialization** as optional `init_mode`:
   - Clear, encapsulated, improves robustness when config-based guesses are poor
   - Low risk: Doesn't touch main solver

2. **Richer extraction options & SB profile table**:
   - Many users benefit from AutoProf-like SB tables and band sampling
   - Low risk: Additive feature, doesn't change existing behavior

#### Higher Complexity, But Feasible as Optional Backends ⚠️

3. **Full AutoProf-style coupled FFT fitting**:
   - Feasible but substantial effort
   - Best implemented as **separate solver backend** with well-defined inputs/outputs and comprehensive tests vs AutoProf
   - Keep as opt-in; don't replace default Jedrzejewski solver

4. **Superellipse / F-mode reconstruction in `build_isoster_model`**:
   - Mostly modeling-layer enhancement
   - Consider adding once you either:
     - Fit such parameters directly, or
     - Decide on mapping from existing harmonics to these shape descriptors

---

## 4. Summary and Conclusions

### Algorithmic Complementarity

**AutoProf** and **isoster** are algorithmically complementary:

- **AutoProf**: Emphasizes **global FFT-based objectives** with **strong radial regularization**; stochastic optimization across all isophotes simultaneously; very flexible geometry models (superellipses, Fourier modes)
- **isoster**: Focuses on **fast, deterministic harmonic/gradient-based per-isophote fitting** with EA support; efficient modeling; streamlined pipeline

### Key Differences Summary

| Component | AutoProf Approach | isoster Approach |
|-----------|-------------------|------------------|
| **Initialization** | FFT of circular isophotes → global PA/ellip | Config-based starting geometry |
| **Fitting** | FFT amplitude + regularization; stochastic global optimization | Harmonic least squares + gradient; deterministic per-radius |
| **Extraction** | Very configurable (band/line, averaging, Fourier measurements) | Streamlined, pipeline-embedded |
| **Modeling** | Superellipse + F-modes; nearest-ellipse assignment | Elliptical + harmonics; direct interpolation |

### Feasibility Conclusion

Porting selected AutoProf ideas into isoster as **opt-in modes** is:
- ✅ **Technically feasible**: Algorithms map cleanly to optional backends
- ✅ **Low risk for core functionality**: Can be implemented as separate solver/init modes without touching existing code
- ✅ **High value**: Would broaden isoster's capabilities (robust initialization, richer extraction, advanced geometry models) without sacrificing current strengths (speed, determinism, EA support)

### Recommended Implementation Order

1. **Phase 1 (Low Risk, High Value)**:
   - FFT-based global initialization (`init_mode="fft_global"`)
   - Richer extraction options and SB profile table output

2. **Phase 2 (Medium Risk, Medium-High Value)**:
   - Superellipse / F-mode support in model reconstruction (if needed for science cases)

3. **Phase 3 (Higher Risk, High Value if Needed)**:
   - Full AutoProf-style coupled FFT fitting backend (`solver="autoprof_fft"`)
   - Comprehensive testing and validation vs AutoProf on shared test cases

---

## References

- **AutoProf**: Stone et al. (2021), MNRAS 508, 1870; https://autoprof.readthedocs.io/
- **isoster**: ISOSTER (ISOphote on STERoid) - Accelerated Python library for elliptical isophote fitting
- **Jedrzejewski (1987)**: Classical isophote fitting algorithm using harmonic analysis
- **Ciambur (2015)**: ApJ 810, 120 - Eccentric anomaly sampling for high-ellipticity isophotes

---

*Document generated: 2026-01-23*
*Reviewer: Senior Python Programmer / Software Engineer*
*Scope: AutoProf `src/autoprof/pipeline_steps` vs isoster core pipeline*
