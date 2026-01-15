# Efficiency Optimization Baseline

This document records the performance baseline for isoster **before** implementing efficiency optimizations. All measurements are used to validate that optimizations improve speed without degrading profile quality.

**Date:** 2026-01-14
**Branch:** `perf/efficiency-optimizations`
**Commit:** Pre-optimization baseline

---

## Benchmark Configuration

### Test Suite

9 test cases covering different galaxy morphologies and noise levels:

| Test Case | n | Re | eps | PA | SNR | Oversample |
|-----------|---|-----|-----|-----|-----|------------|
| n1_small_circular | 1.0 | 10.0 | 0.0 | 0.0 | ∞ | 5 |
| n4_small_circular | 4.0 | 10.0 | 0.0 | 0.0 | ∞ | 10 |
| n1_medium_circular | 1.0 | 20.0 | 0.0 | 0.0 | ∞ | 5 |
| n1_medium_eps04 | 1.0 | 20.0 | 0.4 | π/4 | ∞ | 5 |
| n4_medium_eps04 | 4.0 | 20.0 | 0.4 | π/4 | ∞ | 10 |
| n1_medium_eps07 | 1.0 | 20.0 | 0.7 | π/3 | ∞ | 5 |
| n4_medium_eps06 | 4.0 | 20.0 | 0.6 | π/4 | ∞ | 15 |
| n1_medium_snr100 | 1.0 | 20.0 | 0.4 | π/4 | 100 | 5 |
| n4_medium_snr100 | 4.0 | 20.0 | 0.4 | π/4 | 100 | 10 |

### Methodology

- Each test case run 3 times, mean and std reported
- Noiseless models use high oversampling (5-15x) to simulate true profiles
- Noisy models use SNR=100 with oversample=5-10
- Config parameters:
  - `sma0=10.0`, `minsma=3.0`, `maxsma=8*Re`
  - `astep=0.15`, `minit=10`, `maxit=50`, `conver=0.05`
  - `maxgerr=1.0` for eps>0.6, else `maxgerr=0.5`
  - `use_eccentric_anomaly=True` for eps>0.3

---

## Baseline Performance Results

### Summary Table

| Test Case | Mean Time (s) | Std (s) | Isophotes | Converged | Conv Rate |
|-----------|--------------|---------|-----------|-----------|-----------|
| n1_small_circular | 0.102 | 0.001 | 23 | 23 | 100.0% |
| n4_small_circular | 0.084 | 0.001 | 23 | 23 | 100.0% |
| n1_medium_circular | 0.131 | 0.001 | 28 | 28 | 100.0% |
| n1_medium_eps04 | 0.113 | 0.001 | 28 | 28 | 100.0% |
| n4_medium_eps04 | 0.086 | 0.001 | 28 | 28 | 100.0% |
| n1_medium_eps07 | 0.227 | 0.001 | 28 | 23 | 82.1% |
| n4_medium_eps06 | 0.086 | 0.001 | 28 | 28 | 100.0% |
| n1_medium_snr100 | 0.211 | 0.001 | 28 | 23 | 82.1% |
| n4_medium_snr100 | 0.255 | 0.000 | 28 | 27 | 96.4% |
| **TOTAL** | **1.294** | - | **242** | **231** | **95.5%** |

### Key Observations

1. **Fast cases** (circular, low ellipticity):
   - n=4 cases are faster than n=1 (steeper profiles, fewer iterations?)
   - Typical time: 0.08-0.13 seconds

2. **Slow cases** (high ellipticity or noisy):
   - n1_medium_eps07: 0.227s (2x slower than moderate ellipticity)
   - Noisy cases: 0.21-0.26s (2-3x slower than noiseless)

3. **Convergence**:
   - Noiseless, low-to-moderate ellipticity: 100% convergence
   - High ellipticity (eps=0.7): 82% convergence (5/28 failed)
   - Noisy cases: 82-96% convergence (expected due to noise)

4. **Overall metrics**:
   - Total time: 1.294 seconds (9 test cases)
   - Average time per case: 0.144 seconds
   - Overall convergence rate: 95.5%

---

## Performance Breakdown (Estimated)

Based on profiling estimates from CODE_REVIEW.md:

| Component | Est. % of Time | Notes |
|-----------|----------------|-------|
| Gradient computation | ~30-40% | EFF-1: 3x overhead from redundant extractions |
| Harmonic fitting | ~25-30% | Includes both fitting and error computation |
| Error computation | ~10-15% | EFF-2: 50% overhead from re-fitting harmonics |
| Sampling | ~15-20% | Vectorized, generally efficient |
| Other | ~10-15% | Convergence checks, geometry updates, etc. |

### Targeted Optimizations

1. **EFF-1: Gradient early termination** (HIGH PRIORITY)
   - Expected speedup: 20-30% overall (reduce gradient overhead)
   - Risk: Medium (affects core algorithm)

2. **EFF-2: Harmonic coefficient reuse** (MEDIUM PRIORITY)
   - Expected speedup: 5-10% overall (reduce error computation overhead)
   - Risk: Low (parameter passing only)

3. **EFF-5: PA wrap-around vectorization** (LOW PRIORITY)
   - Expected speedup: <0.1% (trivial)
   - Risk: None (straightforward refactor)

---

## Acceptance Criteria for Optimizations

### Performance Targets

- ✅ **Overall speedup**: >20% vs this baseline (target: <1.03 seconds total)
- ✅ **Individual cases**: At least 15% faster on noiseless cases
- ✅ **Maintain vs photutils**: >4x faster (ideally >6x with optimizations)

### Quality Requirements (Zero Degradation)

Must maintain identical results to this baseline:

1. **Intensity profiles**:
   - Bit-for-bit identical (or <1e-10 relative difference)
   - Median difference in 0.5-4 Re: <0.01%

2. **Geometry profiles**:
   - Ellipticity: <1e-10 absolute difference
   - Position angle: <0.001° difference

3. **Convergence**:
   - Stop codes: 100% identical
   - Convergence rate: ≥95.5% (no degradation)
   - Number of isophotes: identical (±1 acceptable for edge cases)

4. **Integration tests**:
   - All 48 existing tests must pass
   - No new warnings or errors

### Quality vs Photutils (Cross-Validation)

Per EFFICIENCY_OPTIMIZATION_PLAN.md:

- ✅ Intensity agreement: <1% median difference in 0.5-4 Re
- ✅ Ellipticity agreement: <0.01 median difference
- ✅ PA agreement: <5° median difference
- ✅ Convergence: similar or better than photutils

---

## Baseline Data Files

- **Benchmark results**: `benchmarks/efficiency_baseline.json`
- **Test script**: `benchmarks/efficiency_benchmark.py`
- **Comparison framework**: `tests/test_photutils_comparison.py`

---

## Next Steps

### Phase 2: EFF-5 PA Wrap-Around (Trivial)
- Replace while loop in `fitting.py:44-48` with modulo
- Validate: identical results, minimal/no speedup expected
- Purpose: Validate testing framework with zero-risk change

### Phase 3: EFF-2 Harmonic Coefficient Reuse (Low Risk)
- Modify `compute_parameter_errors()` to accept `coeffs` parameter
- Expected: 5-10% speedup, identical results
- Validate: Unit test + full benchmark + photutils comparison

### Phase 4: EFF-1 Gradient Early Termination (Medium Risk)
- Add early termination when first gradient is reliable
- Expected: 20-30% speedup
- Validate: Extensive testing vs baseline + photutils comparison
- Rollback if >1% difference vs photutils or any test fails

### Phase 5: Comprehensive Validation
- Run full 18-case test suite (add 9 more cases)
- Generate QA figures comparing isoster vs photutils vs truth
- Document final speedup and quality metrics
- Update CODE_REVIEW.md

---

## Historical Notes

This baseline is measured after:
- Model building rewrite (20x accuracy improvement)
- High-priority issue fixes (ISSUE-1 through ISSUE-8)
- Test coverage improvements (48 tests passing, 63% coverage)

The baseline represents the current state-of-the-art isoster performance before efficiency optimizations.
