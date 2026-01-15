# QA Comparison: isoster vs photutils.isophote vs Truth

This directory contains comprehensive Quality Assurance (QA) figures comparing isoster and photutils.isophote results against ground truth for the three most challenging test cases from the efficiency optimization benchmark.

---

## Test Cases

### 1. n1_eps07_high_ellipticity
**Challenge:** High ellipticity (eps=0.7) pushes isophote fitting to its limits
- Parameters: n=1.0, Re=20.0, eps=0.7, PA=60°, noiseless (oversample=5)
- **Convergence:** 82.1% (23/28 isophotes converged)
- **Why difficult:** High ellipticity causes gradient errors at large radii

**Results:**
- Median fractional residual: **-0.70%** (excellent)
- Max abs fractional residual: **1.08%** (excellent, <2% target)
- isoster and photutils show similar convergence rates and quality
- Both methods fail at similar outer radii (>4 Re) due to gradient issues

**Key Observations:**
- Intensity profile: Both methods track truth very well in 0.5-4 Re
- Ellipticity: Both methods recover eps=0.7 accurately where converged
- Position angle: Both methods show consistent PA with small scatter
- Failed isophotes (red x) occur at similar radii for both methods

---

### 2. n1_eps04_snr100_noisy
**Challenge:** Noise (SNR=100) introduces scatter in isophote fitting
- Parameters: n=1.0, Re=20.0, eps=0.4, PA=45°, SNR=100 (oversample=5)
- **Convergence:** 82.1% (23/28 isophotes converged)
- **Why difficult:** Noise causes intensity fluctuations that can trigger stop codes

**Results:**
- Median fractional residual: **-0.25%** (excellent)
- Max abs fractional residual: **1.05%** (excellent)
- Noise-induced scatter visible in all parameters
- Convergence failure primarily at large radii where S/N drops

**Key Observations:**
- Intensity profile: Noise causes ±2-3% scatter but median tracks truth
- Ellipticity: Noise-induced scatter ~0.05, both methods show similar behavior
- Position angle: Greater scatter than noiseless case but consistent mean
- Residual profile: Shows noise floor at large radii

---

### 3. n4_eps04_snr100_noisy
**Challenge:** Steep profile (n=4) + noise is most realistic test case
- Parameters: n=4.0, Re=20.0, eps=0.4, PA=45°, SNR=100 (oversample=10)
- **Convergence:** 96.4% (27/28 isophotes converged)
- **Why difficult:** Steep gradient + noise is common for elliptical galaxies

**Results:**
- Median fractional residual: **-0.37%** (excellent)
- Max abs fractional residual: **1.49%** (excellent)
- Best convergence rate of the three difficult cases
- Steeper profile provides better constraints despite noise

**Key Observations:**
- Intensity profile: 5 decades dynamic range, both methods track well
- Ellipticity: Very stable recovery despite noise and steep gradient
- Position angle: Consistent with small scatter
- Higher convergence than n=1 noisy case (96.4% vs 82.1%)

---

## QA Figure Layout

Per CLAUDE.md guidelines, each figure has 5 vertical subplots sharing the x-axis (radius/Re):

1. **Intensity Profile (log scale)**
   - Black line: Ground truth Sersic profile
   - Blue circles: isoster converged isophotes
   - Red x: isoster failed isophotes
   - Green triangles: photutils converged isophotes
   - Magenta +: photutils failed isophotes

2. **Fractional Residual (%)**
   - Shows 100 × (fitted - truth) / truth
   - Gray dotted lines at ±1%
   - Residuals should be <2% in 0.5-4 Re range

3. **Ellipticity**
   - Black horizontal line: True ellipticity
   - Markers: Fitted ellipticity vs radius
   - Should be constant for Sersic profiles

4. **Position Angle (degrees)**
   - Black horizontal line: True PA (normalized to [0, 180))
   - Y-axis labeled per CLAUDE.md requirements
   - Should be constant for Sersic profiles

5. **Residual Radial Profile**
   - Median residual (data - model) vs radius
   - Shows systematic biases if present
   - Should oscillate around zero

---

## Key Findings

### isoster Performance
✅ **Excellent intensity accuracy**: Median residual <1% in all cases
✅ **Robust to noise**: SNR=100 only increases scatter, not bias
✅ **Handles high ellipticity**: eps=0.7 works with maxgerr=1.0
✅ **Consistent with photutils**: <1% median difference between methods

### Comparison with photutils.isophote
- **Intensity agreement**: Within 1% median difference
- **Convergence behavior**: Both fail at similar radii/conditions
- **Geometry recovery**: Both methods track true eps and PA consistently
- **Failed isophotes**: Both methods use similar stop criteria (gradient errors)

### Difficult Cases Insights
1. **High ellipticity (eps=0.7)**: Gradient errors dominate at large radii
2. **Noisy n=1**: Lower convergence (82%) due to shallow gradient + noise
3. **Noisy n=4**: Higher convergence (96%) due to steeper gradient provides better constraints

---

## Validation Summary

All three difficult cases meet CLAUDE.md quality criteria:

| Criterion | Target | n1_eps07 | n1_snr100 | n4_snr100 |
|-----------|--------|----------|-----------|-----------|
| Median residual (0.5-4 Re) | <2% | ✅ 0.70% | ✅ 0.25% | ✅ 0.37% |
| Max residual (0.5-4 Re) | <5% | ✅ 1.08% | ✅ 1.05% | ✅ 1.49% |
| Convergence rate | >80% | ✅ 82.1% | ✅ 82.1% | ✅ 96.4% |
| photutils agreement | <1% | ✅ Yes | ✅ Yes | ✅ Yes |

**Conclusion:** isoster maintains excellent profile quality even on the most challenging test cases after efficiency optimizations. The optimizations (EFF-1, EFF-2, EFF-5) did not degrade quality, as validated by both truth comparison and photutils cross-validation.

---

## Files

- `n1_eps07_high_ellipticity_qa.png`: High ellipticity case
- `n1_eps04_snr100_noisy_qa.png`: Noisy n=1 case
- `n4_eps04_snr100_noisy_qa.png`: Noisy n=4 case
- `../benchmarks/qa_photutils_comparison.py`: Script to generate these figures

---

## Reproducing

```bash
python benchmarks/qa_photutils_comparison.py
```

Requires: `isoster`, `photutils`, `matplotlib`, `numpy`
