# Numba Optimization Code Review

**Date:** 2026-01-15  
**Reviewer:** Code Review  
**Branch:** `perf/numba-optimization`

## Executive Summary

The Numba optimization implementation is **generally well-executed** with good performance gains (26x speedup) and maintains numerical accuracy. However, several **critical issues** and **improvements** are identified that should be addressed before merging.

---

## Critical Issues

### 1. **Missing `NUMBA_DISABLE_JIT` Environment Variable Check** ⚠️ HIGH PRIORITY

**Location:** `isoster/numba_kernels.py` lines 23-35

**Issue:** The code only checks if numba is importable (`NUMBA_AVAILABLE`), but does not check the `NUMBA_DISABLE_JIT` environment variable. According to the documentation, setting `NUMBA_DISABLE_JIT=1` should force numpy fallback, but currently it does not.

**Current behavior:**
- If numba is installed, `NUMBA_AVAILABLE=True` → numba functions are selected
- Even with `NUMBA_DISABLE_JIT=1`, numba functions still run (just in interpreted mode, which is slow)

**Expected behavior:**
- If `NUMBA_DISABLE_JIT=1` is set, should use numpy fallback regardless of numba availability

**Fix:**
```python
# Try to import numba, set flag for availability
try:
    from numba import njit, prange
    import os
    # Check if JIT is disabled via environment variable
    NUMBA_AVAILABLE = os.environ.get('NUMBA_DISABLE_JIT', '0') != '1'
except ImportError:
    NUMBA_AVAILABLE = False
    # ... rest of code
```

**Impact:** Documentation claims this feature exists but it doesn't work. Users expecting to disable numba for debugging will be confused.

---

### 2. **No Input Validation in Numba Functions** ⚠️ MEDIUM PRIORITY

**Location:** All `@njit` decorated functions

**Issues:**

#### 2a. `_compute_ellipse_coords_numba()` - Division by Zero Risk
- **Line 171:** `delta = two_pi / n_samples` - No check if `n_samples == 0`
- **Line 197-198:** `denom` could theoretically be zero (though mathematically unlikely)
- **Impact:** Will crash with `ZeroDivisionError` or produce `inf`/`nan` values

#### 2b. `_harmonic_model_numba()` - Array Bounds
- **Line 59:** Accesses `coeffs[0]` through `coeffs[4]` without checking array length
- **Impact:** Will crash with `IndexError` if `len(coeffs) < 5`

#### 2c. `_build_harmonic_matrix_numba()` - Empty Array
- **Line 268:** `n = len(phi)` - No check if `phi` is empty
- **Impact:** Will create empty array, may cause downstream issues

**Recommendation:** Add input validation at the wrapper level (before calling numba functions) or add guards inside numba functions. Numba functions should be fast, so validation should be minimal but sufficient.

**Example fix:**
```python
def compute_ellipse_coords(n_samples, sma, eps, pa, x0, y0, use_ea):
    """Wrapper with validation."""
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")
    if not (0 <= eps < 1):
        raise ValueError(f"eps must be in [0, 1), got {eps}")
    # ... then call numba function
    return _compute_ellipse_coords_numba(n_samples, sma, eps, pa, x0, y0, use_ea)
```

---

### 3. **Inconsistent Angle Generation** ⚠️ LOW PRIORITY

**Location:** `isoster/numba_kernels.py` lines 170-173 vs 216

**Issue:** The numba version generates angles using a loop:
```python
delta = two_pi / n_samples
for i in range(n_samples):
    angles[i] = i * delta
```

The numpy version uses:
```python
angles = np.linspace(0, 2.0 * np.pi, n_samples, endpoint=False)
```

**Difference:** 
- Numba: `angles = [0, delta, 2*delta, ..., (n-1)*delta]` (excludes 2π)
- Numpy: `np.linspace(..., endpoint=False)` (also excludes 2π)

**Analysis:** Both should be equivalent, but the numba version may have slightly different floating-point rounding. The test suite passes, so this is likely fine, but worth noting.

**Recommendation:** Add a comment explaining why the loop-based approach is used (performance in numba).

---

## Potential Bugs

### 4. **Edge Case: eps = 1.0**

**Location:** `isoster/numba_kernels.py` line 166, 197-198

**Issue:** If `eps = 1.0` (degenerate ellipse):
- `one_minus_eps = 0.0`
- Line 197: `denom = np.sqrt(0 + sin_phi**2) = |sin_phi|`
- Line 198: `r = sma * 0 / |sin_phi|` → `r = 0` when `sin_phi != 0`, but `inf` when `sin_phi = 0`

**Current behavior:** The code doesn't explicitly handle `eps >= 1.0`. The calling code in `sampling.py` line 135 ensures `n_samples >= 64`, but doesn't validate `eps`.

**Recommendation:** Add validation that `0 <= eps < 1.0` at the wrapper level.

---

### 5. **Missing Type Hints and Documentation**

**Location:** All functions in `numba_kernels.py`

**Issue:** Functions lack type hints, making it harder to understand expected input types and catch type errors early.

**Recommendation:** Add type hints to wrapper functions (numba functions themselves can't use standard type hints, but wrappers can):

```python
from typing import Tuple
import numpy as np

def compute_ellipse_coords(
    n_samples: int,
    sma: float,
    eps: float,
    pa: float,
    x0: float,
    y0: float,
    use_ea: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute ellipse sampling coordinates."""
    # ...
```

---

## Code Quality Issues

### 6. **Unused Import: `prange`**

**Location:** `isoster/numba_kernels.py` line 24

**Issue:** `prange` is imported but never used. The code uses `range` in all loops.

**Fix:** Remove `prange` import, or document why it's kept for future use.

---

### 7. **Inconsistent Error Handling**

**Location:** Comparison between numba and numpy fallbacks

**Issue:** Numba functions will raise different exceptions than numpy functions for the same invalid input (e.g., `IndexError` vs `ValueError`). This could cause inconsistent behavior.

**Recommendation:** Add consistent input validation in wrapper functions to ensure both paths raise the same exceptions.

---

### 8. **Missing Edge Case Tests**

**Location:** `tests/test_numba_validation.py`

**Missing test cases:**
- `n_samples = 0` or negative
- `n_samples = 1` (minimum case)
- `eps = 0.99` (very high ellipticity)
- `eps = 1.0` (should be rejected)
- `coeffs` array with wrong length
- Empty `phi` array

**Recommendation:** Add edge case tests to ensure robustness.

---

## Improvements

### 9. **Performance: Pre-compute Trigonometric Values**

**Location:** `isoster/numba_kernels.py` lines 191-194

**Current:** Computes `cos(phi_i)` and `sin(phi_i)` inside the loop for each `phi_i`.

**Potential optimization:** If `phi` values are uniformly spaced (which they are in this case), could use recurrence relations or pre-compute all values. However, this may not be worth it given numba's efficiency.

**Status:** Low priority - current implementation is already fast.

---

### 10. **Documentation: Clarify NUMBA_DISABLE_JIT Behavior**

**Location:** `progress/NUMBA_OPTIMIZATION_RESULTS.md` line 76

**Issue:** Documentation claims `NUMBA_DISABLE_JIT=1` works, but code doesn't implement it.

**Fix:** Either implement the feature or update documentation to reflect current behavior.

---

### 11. **Add Runtime Performance Monitoring**

**Location:** New feature suggestion

**Recommendation:** Add optional performance logging to track:
- Whether numba is actually being used
- JIT compilation time (first call)
- Function call counts

This would help users understand performance characteristics.

---

## Positive Aspects

✅ **Good separation of concerns:** Numba kernels are cleanly separated from business logic  
✅ **Comprehensive fallback:** Graceful degradation when numba is unavailable  
✅ **Good test coverage:** Validation tests ensure numerical accuracy  
✅ **Clear documentation:** Plan and results documents are well-written  
✅ **Performance gains:** 26x speedup is impressive  
✅ **No API changes:** Drop-in replacement maintains backward compatibility  

---

## Recommendations Summary

### Must Fix Before Merge:
1. ✅ Implement `NUMBA_DISABLE_JIT` environment variable check - **FIXED 2026-01-16**
2. ✅ Add input validation for `n_samples > 0` and `0 <= eps < 1` - **FIXED 2026-01-16**
3. ✅ Add array bounds checking for `coeffs` length - **FIXED 2026-01-16**

### Should Fix:
4. ✅ Add edge case tests - **FIXED 2026-01-16** (16 new tests)
5. ✅ Add type hints to wrapper functions - **FIXED 2026-01-16**
6. ✅ Remove unused `prange` import - **FIXED 2026-01-16**
7. ✅ Update documentation to match implementation - **FIXED 2026-01-16**

### Nice to Have:
8. Add performance monitoring
9. Consider pre-computing trig values (if profiling shows benefit)

---

## Testing Recommendations

1. **Test `NUMBA_DISABLE_JIT=1` behavior:**
   ```python
   import os
   os.environ['NUMBA_DISABLE_JIT'] = '1'
   from isoster.numba_kernels import compute_ellipse_coords
   # Should use numpy fallback
   ```

2. **Test edge cases:**
   - `n_samples = 0, 1, 2`
   - `eps = 0.0, 0.99, 1.0`
   - `coeffs` with length < 5
   - Empty arrays

3. **Test error messages:**
   - Ensure helpful error messages for invalid inputs
   - Ensure consistent exceptions between numba and numpy paths

---

## Conclusion

The Numba optimization is **functionally correct** and achieves ~1.4x performance gains over numpy vectorized code.

**Update 2026-01-16:** All critical issues have been addressed:
- `NUMBA_DISABLE_JIT` environment variable now properly triggers numpy fallback
- Input validation added for all wrapper functions
- 16 new edge case tests ensure robustness
- Type hints added for better IDE support
- Documentation updated to reflect actual performance (1.4x vs numpy, not 26x vs interpreted Python)

**Recommendation:** **All issues fixed. Ready for merge.**
