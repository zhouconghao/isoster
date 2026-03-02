# Phase 26: Core Loop Efficiency and Accuracy Pass

## Context

Theoretical analysis of the Jedrzejewski algorithm inside `isoster` reveals three critical, highly interconnected areas for immediate improvement within the core `fit_isophote` loop:
1. **Redundant Sampling:** The radial gradient (Jacobian) is evaluated every micro-iteration by extracting a second isophote, doubling interpolation overhead. Because geometry shifts are minute during a single isophote's convergence, the gradient can be frozen after the first evaluation (Modified Newton Method) to save ~40-50% sampling time.
2. **Missing Gradient Error:** The formal geometric error propagation ($\Delta \epsilon$, $\Delta PA$) currently only accounts for harmonic amplitude variance, omitting the crucial term for gradient uncertainty ($\sigma^2_g / g^4$), which leads to grossly underestimated errors in the low surface brightness (LSB) outer disk.
3. **Missing Noise Floor:** The lack of an explicit background noise (`sigma_bg`) floor allows the optimizer to overfit the noise floor, chasing random pixel noise deep into the outer disk.

## Branch

`feat/phase-26-efficiency-accuracy` (to be created off `main`)

## Deliverables

| ID | File | Purpose |
|----|------|---------|
| P26-001 | `docs/plan/phase-26-efficiency-accuracy.md` | This plan |
| P26-002 | `isoster/config.py` | Add `sigma_bg` to `IsosterConfig`. |
| P26-003 | `isoster/fitting.py` | Implement Lazy Gradient Evaluation in `fit_isophote`. |
| P26-004 | `isoster/fitting.py` | Add $\sigma^2_g$ calculation and missing geometric error terms to `compute_parameter_errors`. |
| P26-005 | `tests/unit/test_fitting.py` | Unit tests for error propagation math and lazy gradient logic. |
| P26-006 | `tests/integration/test_benchmarks.py` | Validate that total `extract_isophote_data` call counts drop by ~40-50%. |

## Step-by-Step Implementation Plan

### Step 1: Explicit Noise Floor (`sigma_bg`)
1. **Config Update:** Add `sigma_bg: Optional[float] = None` to `IsosterConfig`.
2. **Convergence Logic:** In `fit_isophote`, modify the convergence condition. 
   - Currently: `abs(max_amp) < conver * convergence_scale * rms`
   - New: Calculate `noise_floor = sigma_bg / sqrt(N)` if `sigma_bg` is set.
   - Update threshold: `abs(max_amp) < conver * convergence_scale * max(rms, noise_floor)`.
3. **Termination Logic:** If `mean_intensity < sigma_bg` (e.g., `< 0.5 * sigma_bg`), optionally trigger a specific stop code to halt the outward sweep cleanly before gradient breakdown.

### Step 2: Lazy Gradient Evaluation (Modified Newton Method)
1. **State Tracking:** In `fit_isophote`, introduce variables to cache the `gradient` and `gradient_error` from iteration 0.
2. **Evaluation Logic:** 
   - On `i == 0`, compute the gradient normally via `compute_gradient`.
   - On `i > 0`, skip calling `compute_gradient` and reuse the cached values.
   - *Safety Valve:* If the effective maximum harmonic amplitude (`effective_amp`) fails to decrease for 3 consecutive iterations, invalidate the cache and force a re-evaluation of the gradient to break stalls.
3. **Verification:** Add a debug counter for `map_coordinates` calls or patch `extract_isophote_data` in a test to assert that total extraction calls per isophote drop from $2 	imes N_{iter}$ to $1 	imes N_{iter} + (	ext{small constant})$.

### Step 3: Exact Error Propagation (Missing Gradient Term)
1. **Update `compute_gradient`:** Ensure it formally returns the proper standard error of the gradient. 
   - Currently, it calculates `sigma_c = np.std(intens_c)` and `gradient_error = sqrt(sigma_c^2/N_c + sigma_g^2/N_g) / \Delta R`. Leave this as the baseline estimate.
2. **Update `compute_parameter_errors`:**
   - Take `gradient_error` as a new required argument.
   - Inject the missing error terms into the existing variance equations:
     - $\sigma^2_{\Delta\epsilon} = \dots_{old} + [B_2 \cdot 2(1-\epsilon) / (a \cdot g^2)]^2 \cdot \sigma^2_g$
     - $\sigma^2_{\Delta PA} = \dots_{old} + [A_2 \cdot 2(1-\epsilon) / (a \cdot g^2 \cdot ((1-\epsilon)^2 - 1))]^2 \cdot \sigma^2_g$
3. **Validation:** In LSB regions (e.g., $g \approx \sigma_g$), the output `eps_err` and `pa_err` should inflate correctly to reflect true uncertainty, stopping the falsely confident error bars currently reported.

## Testing Strategy

1. **Unit Testing (Math Correctness):**
   - Mock a flat noise field and assert that `compute_parameter_errors` produces larger errors than the current main branch, matching theoretical hand-calculated derivations.
2. **Integration Testing (Performance):**
   - Run the `test_bench_performance.py` or a dedicated test script on a sample FITS image.
   - Assert that the total execution time drops significantly.
   - Assert that the fitted `eps` and `PA` profiles are statistically identical (within tiny numerical noise tolerances) to the current `main` branch profiles, proving the Lazy Gradient does not compromise geometric accuracy.
3. **Robustness Testing:**
   - Inject a known `sigma_bg` into a mock data run and ensure that `stop_code` triggers accurately without endless `-1` gradient failures.
