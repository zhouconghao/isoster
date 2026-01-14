# High-Priority Issues - RESOLVED

All high-priority issues from the previous review have been addressed on branch `fix/remaining-high-priority-issues`.

## Completed Work

### ✅ ISSUE-1: Bare Exception Handlers
**Status:** FIXED
**Changes:** Removed broad `except Exception` blocks from `compute_parameter_errors()` and `compute_deviations()`.
- Kept specific handlers for `LinAlgError`, `ValueError`, `TypeError`
- Added `TypeError` to handle scipy.optimize.leastsq edge cases (N_params > N_data)
- Unexpected errors now raise instead of silently returning zeros
- Improves debugging by making numerical failures visible

**Location:** `isoster/fitting.py:247-256, 292-302`

---

### ✅ ISSUE-5: Stop Code Inconsistency
**Status:** FIXED
**Changes:** Standardized stop codes across forced photometry and main fitter.
- Forced photometry now returns `stop_code=3` (TOO_FEW_POINTS) when `len(intens)==0`
- Previously returned `-1` which conflicted with gradient error semantics
- Code `-1` now exclusively reserved for gradient errors
- Downstream tooling can now reliably distinguish masked data from gradient failures

**Location:** `isoster/fitting.py:97`
**Documentation:** `docs/STOP_CODES.md`

---

### ✅ ISSUE-6: Test Coverage Gaps
**Status:** FIXED
**Changes:** Added comprehensive test suite with 17 new tests.

**New test file:** `tests/test_edge_cases.py` (397 lines)
- **Forced mode (4 tests):** Basic forced photometry, comparison with fitted mode, masked regions, direct function test
- **CoG mode (2 tests):** Basic curve-of-growth, monotonic increase validation, with/without comparison
- **Masked images (4 tests):** All-masked, center masked, partially masked, empty/zero images
- **Config validation (7 tests):** Invalid ellipticity, SMA range, integrator, forced mode requirements, defaults

**Coverage improvement:**
- Overall: 53% → 63%
- config.py: 94% → 100%
- driver.py: 80% → 96%
- fitting.py: 77% → 85%
- sampling.py: 96% → 100%
- cog.py: 0% → 94%

**Total tests:** 48 passing (31 → 48)

---

### ✅ ISSUE-7: Long Function Signature
**Status:** FIXED
**Changes:** Refactored `compute_gradient()` to use structured arguments.

**Before (12 parameters):**
```python
def compute_gradient(image, mask, x0, y0, sma, eps, pa,
                    step=0.1, linear_growth=False, previous_gradient=None,
                    current_data=None, integrator='mean', use_eccentric_anomaly=False)
```

**After (6 parameters):**
```python
def compute_gradient(image, mask, geometry, config,
                    previous_gradient=None, current_data=None)
```

**Benefits:**
- Reduced positional arguments from 12 to 6 (4 required + 2 optional)
- Grouped related parameters into structured dicts:
  - `geometry`: {x0, y0, sma, eps, pa}
  - `config`: {astep, linear_growth, integrator, use_eccentric_anomaly}
- Improved readability and maintainability
- Reduced error-prone positional argument passing
- **No public API breakage** (internal-only function)

**Location:** `isoster/fitting.py:303-397` (definition), `isoster/fitting.py:549-560` (call site)

---

## Summary

All 4 high-priority issues identified in TODO.md have been successfully resolved:

1. ✅ **Exception handling:** Removed broad handlers, improved error visibility
2. ✅ **Stop codes:** Standardized meanings, eliminated inconsistencies
3. ✅ **Test coverage:** Added 17 tests, improved coverage by 10 percentage points
4. ✅ **Code quality:** Refactored long function signatures into structured arguments

**Verification:**
- All 48 tests passing
- No public API breakages
- Coverage improved from 53% to 63%
- Code quality and maintainability enhanced

**Branch:** `fix/remaining-high-priority-issues` (ready for review and merge)

---

## Next Steps (Future Work)

These items are out of scope for the current high-priority fixes but may be addressed in future work:

- [ ] Increase coverage of `isoster/utils.py` (currently 15%)
- [ ] Add tests for `isoster/plotting.py` (currently 3%)
- [ ] Consider further refactoring of other long-signature functions if needed
- [ ] Add benchmarks comparing performance before/after refactoring
