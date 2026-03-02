# Future Work

This document outlines the strategic roadmap for long-term improvements to `isoster`, driven by theoretical analysis from first principles and benchmark comparisons.

## 1. Lazy Gradient Evaluation (Efficiency)
**Goal:** Eliminate redundant image sampling during iterative fitting.
**Details:** Currently, computing the intensity gradient $\partial I/\partial a$ requires extracting a forward offset isophote at $a+\Delta a$ in every micro-iteration. This effectively doubles interpolation costs. We plan to implement a "Modified Newton Method" where the radial gradient (the Jacobian) is evaluated once at iteration 0 and then frozen (reused) for subsequent iterations.
**Expected Impact:** Cuts `map_coordinates` sampling calls per isophote by ~40-50%, the primary bottleneck in the fitting loop, with mathematically zero impact on final convergence accuracy.

## 2. Explicit Noise Floor & Gradient Error (Accuracy & Robustness)
**Goal:** Prevent overfitting in the outer disk and ensure rigorous error characterization.
**Details:**
- **Explicit Noise Floor:** Introduce an explicit `sigma_bg` parameter to establish a hard lower bound on the convergence threshold: $max(rms, \sigma_{bg}/\sqrt{N})$. This stops the solver from chasing vanishing asymmetries in the noise floor.
- **Missing Gradient Error:** Correct the geometry error formulas (for $\Delta\epsilon$, $\Delta PA$) to include the uncertainty of the gradient measurement itself ($\sigma^2_g$), which currently dominates at low surface brightness but is omitted.

## 3. Vectorized Template Photometry (Efficiency)
**Goal:** Instantaneous multiband forced extraction.
**Details:** Instead of drawing 1D elliptical paths and interpolating them sequentially, map the entire $(x, y)$ coordinate grid of the image to elliptical coordinates $(a, \theta)$ given a fixed geometry map. Then use `scipy.stats.binned_statistic` to calculate the mean/median intensity for all $a$ bins simultaneously.
**Expected Impact:** Transforms template photometry from $O(N_{sma} \times N_{samples})$ sequential interpolations to $O(N_{pixels})$ fast array operations. Massive 10x-100x speedup for multiband survey pipelines.

## 4. Gradient-Free Fallback for LSB (Stability)
**Goal:** Prevent premature failures (`stop_code=-1`) in the noise-dominated outer disk.
**Details:** The analytic Newton-Raphson geometry updates fail when the radial gradient approaches zero at the noise floor. We plan to implement a gradient-free fallback optimization (e.g., Brent's method minimizing sample variance) that activates only when the gradient signal-to-noise drops below a reliable threshold.

## 5. Variance Maps & Exact Covariance (Accuracy)
**Goal:** Exact parameter covariance and automatic outlier handling.
**Details:** Support a per-pixel variance map to perform Weighted Least Squares (WLS) harmonic fitting. This replaces residual-based noise estimates with exact photon noise and automatically down-weights outlier pixels (e.g., unmasked cosmic rays).

*(For detailed theoretical derivations of these plans, see the synthesis documents in `docs/archive/review/`)*
