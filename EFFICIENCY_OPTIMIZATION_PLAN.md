# Efficiency Optimization Plan

## Executive Summary

This document outlines the plan to optimize computational efficiency of isoster without degrading 1-D profile extraction quality. All optimizations will be validated against photutils.isophote on identical mock data.

---

## Efficiency Issues to Address

### EFF-1: Redundant Gradient Extractions (HIGH PRIORITY)
**Location:** `fitting.py:303-397` (compute_gradient function)
**Problem:** Extracts isophote data at up to 3 SMAs per gradient computation
**Impact:** ~3x overhead in gradient computation (called for every isophote)
**Fix:** Early termination - skip 2nd gradient SMA if 1st gradient is good enough

### EFF-2: Repeated Harmonic Fits (MEDIUM PRIORITY)
**Location:** `fitting.py:195-256` (compute_parameter_errors function)
**Problem:** Re-fits harmonics even when coefficients already available
**Impact:** ~50% overhead in error computation
**Fix:** Pass coefficients as parameter to avoid re-fitting

### EFF-3: Unused Coordinate Grid (ALREADY FIXED)
**Location:** `model.py:29`
**Status:** ✅ Fixed in previous model.py rewrite
**Impact:** Eliminated 32 MB waste

### EFF-4: Repeated Trig Computations (LOW PRIORITY)
**Location:** `sampling.py:152-153`
**Problem:** `np.cos(pa)`, `np.sin(pa)` recomputed for each SMA (~100 times)
**Impact:** ~0.1% of total time (negligible)
**Fix:** Cache at driver level or accept minor cost

### EFF-5: PA Wrap-Around Using While Loops (LOW PRIORITY)
**Location:** `fitting.py:44-48`
**Problem:** Uses while loop instead of modulo arithmetic
**Impact:** <0.01% of total time (negligible)
**Fix:** Replace with `((delta_pa + π) % (2π)) - π`

---

## Optimization Priority

Based on profiling estimates:

| Issue | Impact | Risk | Priority |
|-------|--------|------|----------|
| EFF-1 | ~3x gradient overhead (~30% total) | Medium | **HIGH** |
| EFF-2 | ~50% error overhead (~10% total) | Low | **MEDIUM** |
| EFF-5 | Trivial | None | **LOW** |
| EFF-4 | <1% | Low | **SKIP** (negligible gain, adds complexity) |

---

## Testing Strategy

### Principle: Zero Degradation in Profile Quality

**All optimizations MUST satisfy:**
1. Identical 1-D profiles (intensity, geometry) to within numerical precision
2. Identical stop codes and convergence behavior
3. Identical results compared to photutils.isophote (where applicable)
4. Measurable performance improvement

### Three-Tier Testing

#### Tier 1: Unit Tests
**Purpose:** Verify correctness of individual optimizations
**Tests:**
- `test_gradient_early_termination()`: Verify identical gradient when early-terminated
- `test_harmonic_coeff_reuse()`: Verify identical errors with passed coefficients
- `test_pa_wraparound_vectorized()`: Verify identical PA normalization

#### Tier 2: Integration Tests
**Purpose:** Verify full pipeline produces identical results
**Tests:**
- Same Sersic models as `test_integration_qa.py`
- Bit-for-bit identical isophote results (except timing)
- All 48 existing tests must pass

#### Tier 3: Benchmark & Comparison Tests
**Purpose:** Validate quality and measure performance
**Files to create:**
1. `benchmarks/efficiency_benchmark.py`: Performance measurement
2. `tests/test_photutils_comparison.py`: isoster vs photutils validation

**Mock Models:**
- Sersic n=1, 4 (exponential, de Vaucouleurs)
- eps=0.0, 0.4, 0.7 (circular, moderate, high ellipticity)
- S/N = ∞, 100, 50 (noiseless, high, medium noise)
- Total: 2×3×3 = 18 test cases

**Comparison Metrics:**
1. **Profile accuracy** (vs truth):
   - Intensity: max/median relative difference in 0.5-4 Re
   - Geometry (eps, PA): max absolute difference
2. **Cross-validation** (isoster vs photutils):
   - Intensity agreement: <1% median difference
   - Geometry agreement: eps <0.01, PA <5°
3. **Performance**:
   - Speedup vs photutils (must maintain >4x)
   - Speedup vs pre-optimization isoster

---

## Implementation Plan

### Phase 1: Setup & Baseline (Commit 1)
**Tasks:**
1. Create benchmark infrastructure
2. Run baseline benchmarks (pre-optimization)
3. Create photutils comparison framework
4. Document baseline performance

**Deliverables:**
- `benchmarks/efficiency_benchmark.py`
- `tests/test_photutils_comparison.py`
- `EFFICIENCY_BASELINE.md` with results

### Phase 2: EFF-5 PA Wrap-Around (Commit 2)
**Why first:** Trivial, zero risk, validates testing framework

**Changes:**
- Replace while loop with modulo in `fitting.py:44-48`

**Tests:**
- Unit test: `test_pa_wraparound_vectorized()`
- Integration: All existing tests pass
- Benchmark: No significant change expected

### Phase 3: EFF-2 Harmonic Coefficient Reuse (Commit 3)
**Risk:** Low (parameter passing only)

**Changes:**
- Modify `compute_parameter_errors()` signature to accept `coeffs` parameter
- Update call site in `fit_isophote()` to pass coefficients
- Fall back to re-fitting if coeffs=None (backward compat)

**Tests:**
- Unit test: `test_harmonic_coeff_reuse()`
- Integration: All existing tests pass
- Benchmark: ~10% speedup expected

### Phase 4: EFF-1 Gradient Early Termination (Commit 4)
**Risk:** Medium (affects gradient computation logic)

**Changes:**
- Add early termination in `compute_gradient()` when first gradient is reliable
- Use heuristic: if `gradient_error / abs(gradient) < 0.3`, skip 2nd SMA
- Keep safety: always compute 2nd SMA if first gradient suspicious

**Tests:**
- Unit test: `test_gradient_early_termination()`
- Integration: All existing tests pass
- Photutils comparison: <1% median difference
- Benchmark: ~20-30% speedup expected

### Phase 5: Validation & Documentation (Commit 5)
**Tasks:**
1. Run comprehensive photutils comparison on all 18 test cases
2. Generate QA figures comparing isoster vs photutils vs truth
3. Document final speedup and quality metrics
4. Update CODE_REVIEW.md

**Deliverables:**
- `EFFICIENCY_RESULTS.md` with all benchmarks and QA figures
- `efficiency_qa/` directory with comparison plots
- Updated `CODE_REVIEW.md`

---

## Acceptance Criteria

### Performance
- ✅ Overall speedup: >20% vs baseline isoster
- ✅ Speedup vs photutils: maintained >4x (ideally >6x with optimizations)

### Quality (vs baseline isoster)
- ✅ Intensity profiles: bit-for-bit identical (or <1e-10 relative difference)
- ✅ Geometry profiles: bit-for-bit identical
- ✅ Stop codes: 100% identical
- ✅ All 48 existing tests pass

### Quality (vs photutils.isophote)
- ✅ Intensity agreement: <1% median difference in 0.5-4 Re
- ✅ Ellipticity agreement: <0.01 median difference
- ✅ PA agreement: <5° median difference
- ✅ Convergence: similar or better (fewer failed isophotes)

### Quality (vs truth for noiseless)
- ✅ Intensity residuals: <2% max in 0.5-4 Re (as before)
- ✅ Geometry recovery: <1% error in eps, <3° in PA

---

## Risk Mitigation

### EFF-1 (Gradient Early Termination) - Highest Risk

**Potential issues:**
- Early termination might skip cases where 2nd gradient needed
- Could affect convergence in noisy regions

**Mitigation:**
1. Conservative threshold: only terminate if gradient_error < 0.3 × gradient
2. Never terminate on first isophote (need stable reference)
3. Keep diagnostics: log termination rate in debug mode
4. Extensive testing against photutils

**Rollback criteria:**
- If >5% difference vs photutils in any test case
- If convergence degrades (more failed isophotes)
- If any existing test fails

### EFF-2 (Coefficient Reuse) - Low Risk

**Potential issues:**
- None expected (pure parameter passing)

**Mitigation:**
- Maintain backward compatibility (coeffs=None falls back)
- Unit test with various coefficient inputs

---

## Timeline & Commits

1. **Commit 1**: Benchmark infrastructure + baseline
2. **Commit 2**: EFF-5 PA wrap-around (trivial)
3. **Commit 3**: EFF-2 harmonic coefficient reuse (low risk)
4. **Commit 4**: EFF-1 gradient early termination (medium risk, high reward)
5. **Commit 5**: Comprehensive validation + documentation

Each commit will include:
- Code changes
- Unit tests
- Benchmark results
- Integration test validation

---

## Post-Optimization

### If Time Permits (Future Work)
1. Numba JIT for hot loops (5-10x potential)
2. Better gradient estimation (3-point stencil)
3. Adaptive convergence thresholds

### Not Pursuing (Too Risky or Low Reward)
- ❌ EFF-4 Trig caching: <1% gain, adds state management complexity
- ❌ Changing core algorithms: risk degrading quality
- ❌ Removing safety checks: not worth the risk

---

## Success Metrics

**Target:**
- 25-35% faster than baseline isoster
- Maintain >6x faster than photutils
- Zero degradation in 1-D profile quality
- All tests green

**Stretch:**
- 40% faster than baseline
- >8x faster than photutils
- Identify additional safe optimizations for future work
