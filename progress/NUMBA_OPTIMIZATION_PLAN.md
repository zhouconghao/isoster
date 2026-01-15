# Numba Optimization Plan for ISOSTER

## Executive Summary

This document outlines the plan for implementing Numba JIT compilation to improve the performance of ISOSTER's isophote fitting routines. Based on code analysis, we identify the hot paths that would benefit most from Numba optimization while ensuring zero degradation in numerical precision.

**Target:** 2-5x speedup on hot paths (based on EFFICIENCY_RESULTS.md recommendations)
**Branch:** `perf/numba-optimization`
**Date:** 2026-01-15

---

## Part 1: Code Analysis and Hot Path Identification

### 1.1 Call Frequency Analysis

Based on the fitting algorithm, for a typical galaxy image with ~25 isophotes:
- `fit_isophote()` called ~25 times per image
- Each `fit_isophote()` iterates ~10-50 times (default `maxit=50`)
- Per iteration, these functions are called:
  - `extract_isophote_data()`: 1-3 times (current SMA + gradient SMAs)
  - `fit_first_and_second_harmonics()`: 1 time
  - `sigma_clip()`: 1 time
  - `harmonic_function()`: 1-2 times
  - `compute_gradient()`: 1 time (calls extract_isophote_data internally)

**Estimated total calls per image:**
| Function | Calls per Image |
|----------|-----------------|
| `extract_isophote_data()` | 500-2000 |
| `fit_first_and_second_harmonics()` | 250-1250 |
| `sigma_clip()` | 250-1250 |
| `harmonic_function()` | 250-2500 |

### 1.2 Profiling Targets

Based on call frequency and computational complexity, the priority targets are:

#### Priority 1: `extract_isophote_data()` (sampling.py:83-184)
- **Frequency:** Highest
- **Operations:**
  - Trig computations: `np.sin()`, `np.cos()` (4 calls)
  - Ellipse coordinate math: sqrt, division
  - scipy.ndimage.map_coordinates (already optimized C code)
  - Masking operations
- **Numba potential:** High for coordinate calculations
- **Constraint:** `map_coordinates` is already C-optimized; can't be replaced

#### Priority 2: `fit_first_and_second_harmonics()` (fitting.py:121-152)
- **Frequency:** High
- **Operations:**
  - Trig computations: `np.sin()`, `np.cos()` (4 calls)
  - Matrix construction: `np.column_stack()`
  - Linear algebra: `np.linalg.lstsq()`, `np.linalg.inv()`
- **Numba potential:** Medium (trig + matrix construction)
- **Constraint:** `np.linalg.lstsq` is LAPACK-optimized

#### Priority 3: `sigma_clip()` (fitting.py:159-190)
- **Frequency:** High
- **Operations:**
  - Iterative loop with array slicing
  - Statistics: `np.mean()`, `np.std()`
- **Numba potential:** Medium (loop optimization)

#### Priority 4: `harmonic_function()` (fitting.py:154-157)
- **Frequency:** Very high
- **Operations:** Simple trig + multiplication
- **Numba potential:** High (simple, hot function)

#### Priority 5: `eccentric_anomaly_to_position_angle()` (sampling.py:12-41)
- **Frequency:** Moderate (only when `use_eccentric_anomaly=True`)
- **Operations:** Trig computations
- **Numba potential:** High for trig optimization

### 1.3 Functions NOT Suitable for Numba

- `compute_aperture_photometry()`: Uses `np.mgrid` and complex slicing, Numba support is limited
- `compute_parameter_errors()`: Uses `scipy.optimize.leastsq` which cannot be JIT-compiled
- `compute_deviations()`: Same issue with `scipy.optimize.leastsq`
- `compute_gradient()`: Orchestration function, calls other functions

---

## Part 2: Implementation Strategy

### 2.1 Approach: Numba-Accelerated Helper Functions

Rather than trying to JIT-compile entire functions (which would fail due to scipy/complex numpy calls), we will:

1. **Extract pure numerical kernels** that are Numba-compatible
2. **Create `@njit` decorated helper functions** for these kernels
3. **Keep the original functions as wrappers** that call the JIT-compiled helpers

### 2.2 Proposed Numba Functions

#### 2.2.1 `_compute_ellipse_coordinates_numba()` (NEW)
```python
@njit(cache=True)
def _compute_ellipse_coordinates_numba(n_samples, sma, eps, pa, x0, y0, use_ea):
    """
    Compute ellipse sampling coordinates.

    Returns: (x_coords, y_coords, angles, phi)
    """
    # Angle sampling
    angles = np.linspace(0, 2 * np.pi, n_samples)

    if use_ea:
        # Eccentric anomaly to position angle conversion
        phi = np.arctan2((1 - eps) * np.sin(angles), np.cos(angles))
        phi = phi % (2 * np.pi)
    else:
        phi = angles.copy()

    # Ellipse equation
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    denom = np.sqrt(((1.0 - eps) * cos_phi)**2 + sin_phi**2)
    r = sma * (1.0 - eps) / denom

    x_rot = r * cos_phi
    y_rot = r * sin_phi

    # Rotation to image frame
    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)

    x = x0 + x_rot * cos_pa - y_rot * sin_pa
    y = y0 + x_rot * sin_pa + y_rot * cos_pa

    return x, y, angles, phi
```

#### 2.2.2 `_build_harmonic_matrix_numba()` (NEW)
```python
@njit(cache=True)
def _build_harmonic_matrix_numba(phi):
    """
    Build the design matrix for harmonic fitting.

    Returns: A matrix (n_samples x 5)
    """
    n = len(phi)
    A = np.empty((n, 5))

    for i in range(n):
        A[i, 0] = 1.0
        A[i, 1] = np.sin(phi[i])
        A[i, 2] = np.cos(phi[i])
        A[i, 3] = np.sin(2 * phi[i])
        A[i, 4] = np.cos(2 * phi[i])

    return A
```

#### 2.2.3 `_harmonic_model_numba()` (NEW)
```python
@njit(cache=True)
def _harmonic_model_numba(phi, coeffs):
    """
    Evaluate harmonic model at given angles.
    """
    n = len(phi)
    result = np.empty(n)

    for i in range(n):
        result[i] = (coeffs[0] +
                     coeffs[1] * np.sin(phi[i]) +
                     coeffs[2] * np.cos(phi[i]) +
                     coeffs[3] * np.sin(2 * phi[i]) +
                     coeffs[4] * np.cos(2 * phi[i]))

    return result
```

#### 2.2.4 `_sigma_clip_numba()` (NEW)
```python
@njit(cache=True)
def _sigma_clip_numba(phi, intens, sclip_low, sclip_high, nclip):
    """
    Perform iterative sigma clipping.

    Returns: (phi_clipped, intens_clipped, n_clipped)
    """
    phi_c = phi.copy()
    intens_c = intens.copy()
    total_clipped = 0

    for _ in range(nclip):
        if len(intens_c) < 3:
            break

        # Compute stats
        mean = 0.0
        for i in range(len(intens_c)):
            mean += intens_c[i]
        mean /= len(intens_c)

        var = 0.0
        for i in range(len(intens_c)):
            var += (intens_c[i] - mean)**2
        std = np.sqrt(var / len(intens_c))

        lower = mean - sclip_low * std
        upper = mean + sclip_high * std

        # Count valid
        n_valid = 0
        for i in range(len(intens_c)):
            if intens_c[i] >= lower and intens_c[i] <= upper:
                n_valid += 1

        if n_valid == len(intens_c):
            break

        # Create new arrays
        new_phi = np.empty(n_valid)
        new_intens = np.empty(n_valid)
        j = 0
        for i in range(len(intens_c)):
            if intens_c[i] >= lower and intens_c[i] <= upper:
                new_phi[j] = phi_c[i]
                new_intens[j] = intens_c[i]
                j += 1

        total_clipped += len(intens_c) - n_valid
        phi_c = new_phi
        intens_c = new_intens

    return phi_c, intens_c, total_clipped
```

### 2.3 File Structure

```
isoster/
├── sampling.py          # Modified to use numba helpers
├── fitting.py           # Modified to use numba helpers
├── numba_kernels.py     # NEW: All @njit decorated functions
└── ...

benchmarks/
├── numba_benchmark.py   # NEW: Performance comparison benchmark
└── results/
    └── numba_benchmark_results.json

progress/
├── NUMBA_OPTIMIZATION_PLAN.md    # This file
└── NUMBA_OPTIMIZATION_RESULTS.md # Results summary (to be created)

tests/
├── test_numba_accuracy.py   # NEW: Numerical accuracy validation
└── qa_outputs/
    └── numba_comparison_*.png  # QA plots
```

---

## Part 3: Implementation Plan

### Phase 1: Create Numba Kernels Module
1. Create `isoster/numba_kernels.py` with all `@njit` decorated functions
2. Include fallback functions for systems without Numba
3. Add comprehensive docstrings

### Phase 2: Integrate Numba Kernels
1. Modify `sampling.py` to use `_compute_ellipse_coordinates_numba()`
2. Modify `fitting.py` to use:
   - `_build_harmonic_matrix_numba()`
   - `_harmonic_model_numba()`
   - `_sigma_clip_numba()`
3. Ensure identical behavior with fallback option

### Phase 3: Benchmarking
1. Create `benchmarks/numba_benchmark.py`:
   - Measure baseline (without numba)
   - Measure with numba (first run = compilation)
   - Measure with numba (subsequent runs = cached)
2. Test on multiple Sersic profiles
3. Compare timing and numerical results

### Phase 4: Validation
1. Run all existing tests (should pass without modification)
2. Create numerical accuracy tests comparing numba vs non-numba results
3. Generate QA plots showing identical profiles

### Phase 5: Documentation
1. Write `progress/NUMBA_OPTIMIZATION_RESULTS.md`
2. Update CODE_REVIEW.md with results

---

## Part 4: Acceptance Criteria

### Performance
- [ ] **>20% speedup** on overall fit_image() runtime
- [ ] **>2x speedup** on individual hot path functions
- [ ] Consistent speedup across test cases (noiseless, noisy, high ellipticity)

### Correctness (CRITICAL)
- [ ] **Bit-for-bit identical** intensity profiles (within floating point tolerance)
- [ ] **Identical** ellipticity profiles
- [ ] **Identical** PA profiles
- [ ] **Identical** stop codes and convergence behavior
- [ ] All existing 58 tests pass

### Quality
- [ ] QA plots show no visible difference
- [ ] Residuals vs truth unchanged
- [ ] No new warnings or errors

### Code Quality
- [ ] Graceful fallback when Numba not available
- [ ] Clear docstrings and inline comments
- [ ] No changes to public API

---

## Part 5: Risk Assessment

### Low Risk
- `_harmonic_model_numba()`: Simple function, easy to validate
- `_build_harmonic_matrix_numba()`: Pure math, no edge cases

### Medium Risk
- `_compute_ellipse_coordinates_numba()`: More complex, needs careful testing
- `_sigma_clip_numba()`: Iterative algorithm, needs boundary testing

### Mitigation
1. Create comprehensive unit tests for each numba function
2. Compare outputs against non-numba versions
3. Test with edge cases (eps=0, eps=0.95, small SMA, large SMA)
4. Keep non-numba fallback always available

---

## Part 6: Expected Results

Based on similar projects and the nature of the optimizations:

| Function | Expected Speedup | Confidence |
|----------|-----------------|------------|
| `_compute_ellipse_coordinates_numba()` | 3-5x | High |
| `_build_harmonic_matrix_numba()` | 2-4x | High |
| `_harmonic_model_numba()` | 5-10x | High |
| `_sigma_clip_numba()` | 2-3x | Medium |
| **Overall `fit_image()`** | **1.5-2.5x** | Medium |

Note: Overall speedup is limited because:
1. `map_coordinates` (C code) and `np.linalg.lstsq` (LAPACK) are already optimized
2. Numba compilation adds first-run overhead (mitigated by caching)
3. Memory bandwidth may be the actual bottleneck for some operations

---

## Appendix: Numba Best Practices

1. **Use `cache=True`** to persist compiled code between sessions
2. **Use `parallel=True` cautiously** - only for embarrassingly parallel loops
3. **Avoid Python objects** - use only numpy arrays and scalars
4. **Pre-allocate arrays** - don't use `np.append()` or list growth
5. **Use explicit loops** - sometimes faster than numpy broadcasting in Numba
6. **Test with `NUMBA_DISABLE_JIT=1`** environment variable for debugging
