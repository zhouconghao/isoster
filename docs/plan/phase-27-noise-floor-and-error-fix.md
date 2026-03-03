# Phase 27: Explicit Noise Floor & Gradient Error Fix

## Context

Theoretical analysis highlights two critical improvements for low surface brightness (LSB) stability and statistical accuracy in the outer disk:
1.  **Explicit Noise Floor (`sigma_bg`)**: Prevents the optimizer from over-fitting vanishing asymmetries in the noise floor by setting a hard lower bound on the convergence threshold.
2.  **Missing Gradient Error**: Corrects the geometric error propagation for ellipticity ($\epsilon$) and position angle ($PA$) by including the uncertainty of the radial gradient ($\sigma_g$). Currently, `isoster` only accounts for harmonic amplitude variance, leading to underestimated error bars in LSB regions.

## Deliverables

| ID | File | Purpose |
|----|------|---------|
| P27-001 | `docs/plan/phase-27-noise-floor-and-error-fix.md` | This plan |
| P27-002 | `isoster/config.py` | Add `sigma_bg` to `IsosterConfig`. |
| P27-003 | `isoster/fitting.py` | Implement noise-floor-aware convergence in `fit_isophote`. |
| P27-004 | `isoster/fitting.py` | Update `compute_parameter_errors` with gradient uncertainty terms. |
| P27-005 | `benchmarks/noise_floor/run_benchmark.py` | Stability and accuracy benchmark using mock galaxies. |
| P27-006 | `outputs/benchmark_noise_floor/` | Benchmark reports and high-quality QA figures. |

## Implementation Plan

### Step 1: Configuration and Noise Floor Logic
1.  **Config**: Add `sigma_bg: Optional[float] = None` to `IsosterConfig`.
2.  **Convergence Threshold**: Update the threshold in `fit_isophote`:
    $$ 	ext{threshold} = 	ext{conver} \cdot 	ext{scale} \cdot \max(	ext{rms}, \sigma_{bg} / \sqrt{N}) $$
3.  **Termination**: Optionally add logic to terminate the profile if `mean_intensity` falls below a certain fraction of `sigma_bg`.

### Step 2: Corrected Error Propagation
1.  **Gradient Uncertainty**: Ensure `compute_gradient` returns a reliable $\sigma_g$.
2.  **Error Formulas**: Update `compute_parameter_errors` to include the $\sigma_g$ term in the variance of $\Delta\epsilon$ and $\Delta PA$:
    $$ \sigma^2_{\Delta\epsilon} \propto \frac{\sigma^2_{B_2}}{g^2} + B_2^2 \frac{\sigma^2_g}{g^4} $$
    $$ \sigma^2_{\Delta PA} \propto \frac{\sigma^2_{A_2}}{g^2} + A_2^2 \frac{\sigma^2_g}{g^4} $$

### Step 3: Mock Galaxy Generation (Validation Case)
1.  **Selection**: Use NGC 1453 and NGC 3585 from Huang+2013 (high-n outer components).
2.  **Generation**: Use the `mockgal` skill to generate large mock images at $z=0.2$ with HSC-like noise ($	ext{sky\_sb\_limit}=24.5$).
3.  **Estimation**: Estimate `sigma_bg` from the "empty" corners of the mock images.

### Step 4: Verification and Benchmarking
1.  **Stability**: Compare profiles fitted with and without `sigma_bg`. Assert that `sigma_bg` prevents erratic geometry swings in the LSB outskirts.
2.  **Accuracy**: Assert that the new error bars correctly encompass the true geometry in the noisy regime, unlike the current (underestimated) errors.
3.  **Efficiency**: Compare runtime with `photutils` on these large mock images.

## Verification

1.  Unit tests for the new error propagation math.
2.  Benchmark script produces comparison plots showing improved stability and more realistic error bars in the outer disk.
3.  Final profiles compared against the "true" input models of the mock galaxies.
