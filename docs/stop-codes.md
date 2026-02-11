# Isophote Fitting Stop Codes

Stop codes indicate the termination condition for each isophote fit. Understanding these codes is essential for interpreting fitting results and diagnosing problems.

## Stop Code Reference

| Code | Name | Meaning | Action |
|------|------|---------|--------|
| **0** | `SUCCESS` | **Converged successfully.** Max harmonic amplitude < threshold. | Use this isophote. |
| **1** | `TOO_MANY_FLAGGED` | **Too many pixels flagged.** Fraction of valid pixels < `fflag` threshold (default 0.5). | Geometry may be unreliable. Consider masking or adjusting `fflag`. |
| **2** | `MINOR_ISSUES` | **Minor convergence issues.** Used by driver when propagating geometry. | Usually safe to use. Check adjacent isophotes. |
| **3** | `TOO_FEW_POINTS` | **Insufficient data.** Less than 6 valid pixels after clipping. | Cannot fit - discard this isophote. |
| **-1** | `GRADIENT_ERROR` | **Invalid radial gradient.** Gradient is non-negative, zero, or has high relative error. | Fit failed - discard this isophote. Often occurs at image edges or in flat regions. |

## Stop Code Details

### Code 0: SUCCESS

**Trigger:** `abs(max_harmonic_amplitude) < conver * rms` after at least `minit` iterations

**Parameters affected:**
- `conver` (default 0.05): Convergence threshold
- `minit` (default 10): Minimum iterations before convergence

**Interpretation:** Harmonic fitting converged successfully. The ellipse geometry is stable, and the harmonic amplitudes (deviations from perfect ellipticity) are negligible compared to the noise level.

**Example:**
```python
# Well-behaved isophote in smooth galaxy region
# rms = 2.0, max_harmonic_amplitude = 0.08
# 0.08 < 0.05 * 2.0 = 0.1 ✓ Converged
```

---

### Code 1: TOO_MANY_FLAGGED

**Trigger:** `actual_pixels < total_pixels * fflag`

**Parameters affected:**
- `fflag` (default 0.5): Minimum fraction of valid pixels required
- `sclip` (sigma clipping threshold): Aggressive clipping increases flagged pixels

**Interpretation:** More than 50% of sampled pixels were masked or sigma-clipped. The isophote may be:
- Passing through a masked region (stars, defects, gaps)
- In a noisy area requiring aggressive sigma clipping
- At an SMA where the ellipse extends beyond image bounds
- Affected by contamination not in the mask

**Recommendations:**
- Inspect the mask image at this SMA
- Check if `fflag` threshold is too strict (try lowering to 0.3)
- Review `sclip` setting (3.0 is standard; higher values = more clipping)
- Verify ellipse geometry hasn't diverged (check adjacent isophotes)

**Example:**
```python
# 100 pixels sampled, 60 masked or clipped
# 40 < 100 * 0.5 ❌ Too many flagged
```

---

### Code 2: MINOR_ISSUES

**Trigger:** Set by `driver.fit_image()` when propagating geometry from previous isophote

**Interpretation:** The previous isophote had minor issues (code 1 or 2) but geometry was propagated anyway. This allows fitting to continue through mildly problematic regions while marking the uncertainty.

**Recommendations:**
- Usually safe to use
- Check consistency with adjacent isophotes (plot geometry vs SMA)
- If many consecutive isophotes have code 2, consider:
  - Improving the mask
  - Adjusting `fflag` or `sclip`
  - Checking for systematic issues (e.g., galaxy merger, tidal features)

---

### Code 3: TOO_FEW_POINTS

**Trigger:** `len(valid_pixels) < 6` after sigma clipping

**Interpretation:** Cannot fit 5-parameter harmonic model (I₀, A₁, B₁, A₂, B₂) with fewer than 6 points. This typically occurs when:
- SMA is extremely small (< 3 pixels) with insufficient sampling
- Mask is too aggressive in small regions
- Sigma clipping removed almost all points (outlier-dominated region)

**Recommendations:**
- Discard this isophote
- Check if `minsma` is too small (should be ≥ 3-5 pixels)
- Review mask near this SMA
- If outer isophotes: may indicate you've reached the detection limit

**Example:**
```python
# SMA = 2.0, only 4 pixels sampled after mask + clipping
# 4 < 6 ❌ Too few points for 5-parameter fit
```

---

### Code -1: GRADIENT_ERROR

**Trigger (multiple conditions):**

1. **`gradient >= 0`** (intensity increasing outward - unphysical for galaxies)
2. **`gradient == 0`** (flat profile - cannot compute geometry corrections)
3. **`gradient_relative_error > maxgerr`** (default 0.5) for 2 consecutive iterations
4. **Central pixel is masked**
5. **Forced photometry extraction failed** (no valid pixels)

**Parameters affected:**
- `maxgerr` (default 0.5): Maximum relative gradient error tolerated

**Interpretation:**
The radial intensity gradient is unreliable or unphysical. Geometry corrections scale as `1/gradient`, so gradient errors cause divergence. Common scenarios:

- **Image edges:** Gradient flattens or reverses due to truncation
- **Sky background:** Approaching noise floor (gradient → 0)
- **Very noisy regions:** High relative uncertainty in gradient
- **Masked regions:** Central pixel or too many masked pixels
- **Photometric artifacts:** Local spikes, detector issues

**Recommendations:**
- Discard this isophote
- Stop codes -1 typically mark the **valid SMA range boundary**
- For outward growth: maxsma may be too large (reduce it)
- For inward growth: minsma may be too small (increase it)

**Example:**
```python
# At SMA=80, gradient = -0.01 ± 0.008
# Relative error: 0.008/0.01 = 0.8 > 0.5 ❌ Gradient too uncertain

# Or: At SMA=100, gradient = +0.02 (increasing!)
# gradient >= 0 ❌ Unphysical
```

---

## Usage in Code

### Filtering Isophotes by Stop Code

```python
import isoster

# Fit galaxy
results = isoster.fit_image(image, mask, config)
isophotes = results['isophotes']

# Filter for good isophotes only
good_isos = [iso for iso in isophotes if iso['stop_code'] == 0]
print(f"Converged: {len(good_isos)} / {len(isophotes)}")

# Filter for usable isophotes (including minor issues)
usable_isos = [iso for iso in isophotes if iso['stop_code'] in [0, 2]]

# Find where fitting failed
failed_isos = [iso for iso in isophotes if iso['stop_code'] < 0]
if failed_isos:
    max_reliable_sma = min(iso['sma'] for iso in failed_isos)
    print(f"Fitting became unreliable beyond SMA = {max_reliable_sma}")
```

### Plotting with Stop Code Indicators

```python
import matplotlib.pyplot as plt
import numpy as np

# Extract data
sma = np.array([iso['sma'] for iso in isophotes])
intens = np.array([iso['intens'] for iso in isophotes])
stop_codes = np.array([iso['stop_code'] for iso in isophotes])

# Plot surface brightness profile
fig, ax = plt.subplots()

# Good isophotes (green)
good = stop_codes == 0
ax.scatter(sma[good], intens[good], c='green', label='Converged (0)', zorder=3)

# Minor issues (yellow)
minor = stop_codes == 2
if minor.any():
    ax.scatter(sma[minor], intens[minor], c='yellow', label='Minor issues (2)', zorder=2)

# Flagged (orange)
flagged = stop_codes == 1
if flagged.any():
    ax.scatter(sma[flagged], intens[flagged], c='orange', label='Too many flagged (1)', zorder=1)

# Failed (red)
failed = stop_codes < 0
if failed.any():
    ax.scatter(sma[failed], intens[failed], c='red', marker='x', label='Failed (-1, 3)', zorder=0)

ax.set_xlabel('SMA (pixels)')
ax.set_ylabel('Intensity')
ax.set_yscale('log')
ax.legend()
plt.show()
```

---

## Configuration Parameters

Control stop code behavior via `IsosterConfig`:

```python
from isoster.config import IsosterConfig

cfg = IsosterConfig(
    # Stop code 0 (SUCCESS)
    conver=0.05,       # Convergence threshold (lower = stricter)
    minit=10,          # Minimum iterations before convergence
    maxit=50,          # Maximum iterations (prevents infinite loops)

    # Stop code 1 (TOO_MANY_FLAGGED)
    fflag=0.5,         # Minimum fraction of valid pixels (lower = more permissive)
    sclip=3.0,         # Sigma clipping threshold (higher = less aggressive)
    nclip=0,           # Number of clipping iterations (0 = single pass)

    # Stop code -1 (GRADIENT_ERROR)
    maxgerr=0.5,       # Max relative gradient error (higher = more permissive)

    # SMA range (affects where code -1 occurs)
    minsma=1.0,        # Minimum SMA (increase if central region problematic)
    maxsma=None,       # Maximum SMA (decrease if outer region noisy)
)
```

### Tuning for Noisy Data

```python
# More permissive for low S/N images
cfg_noisy = IsosterConfig(
    conver=0.10,       # Relax convergence (2x default)
    fflag=0.3,         # Allow more flagged pixels
    sclip=2.5,         # Less aggressive clipping
    maxgerr=0.7,       # Tolerate higher gradient errors
)
```

### Tuning for Clean Data

```python
# Stricter for high S/N images
cfg_clean = IsosterConfig(
    conver=0.02,       # Tighter convergence
    fflag=0.7,         # Require more valid pixels
    sclip=3.5,         # More aggressive outlier removal
    maxgerr=0.3,       # Lower gradient error tolerance
)
```

---

## Diagnostic Workflow

### Step 1: Assess Overall Success Rate

```python
results = isoster.fit_image(image, mask, config)
isos = results['isophotes']

stop_code_counts = {}
for iso in isos:
    code = iso['stop_code']
    stop_code_counts[code] = stop_code_counts.get(code, 0) + 1

print("Stop Code Distribution:")
for code in sorted(stop_code_counts.keys(), reverse=True):
    count = stop_code_counts[code]
    pct = 100 * count / len(isos)
    print(f"  Code {code:2d}: {count:3d} isophotes ({pct:5.1f}%)")

# Target: >70% code 0 for good fit
success_rate = stop_code_counts.get(0, 0) / len(isos)
if success_rate < 0.5:
    print("⚠️ Low success rate - consider adjusting parameters")
```

### Step 2: Identify Problematic Regions

```python
# Find where each stop code occurs
for code in [-1, 1, 3]:
    bad_isos = [iso for iso in isos if iso['stop_code'] == code]
    if bad_isos:
        smas = [iso['sma'] for iso in bad_isos]
        print(f"Code {code}: SMA range {min(smas):.1f} - {max(smas):.1f}")
```

### Step 3: Adjust Parameters

| Issue | Symptom | Solution |
|-------|---------|----------|
| Many code 1 | fflag threshold too strict | Increase `fflag` or decrease `sclip` |
| Many code -1 (inner) | Central gradient errors | Increase `minsma` |
| Many code -1 (outer) | Edge gradient errors | Decrease `maxsma` |
| Many code 3 | Too few points | Increase `minsma`, review mask |
| No code 0 | Never converges | Increase `conver` or `maxit` |

---

## Historical Note

Stop codes follow the `photutils.isophote` convention:

- **Non-negative codes (0, 1, 2, 3):** Fit completed with varying quality
- **Negative codes (-1):** Fit failed and should be discarded

This convention allows simple filtering:
```python
# Include all fits that completed
completed = [iso for iso in isophotes if iso['stop_code'] >= 0]

# Include only reliable fits
reliable = [iso for iso in isophotes if iso['stop_code'] in [0, 2]]
```

---

## See Also

- `IsosterConfig` documentation - Parameter tuning guide
- `fit_isophote()` source code - Implementation details
- `examples/` directory - Typical usage patterns
- `docs/archive/review/code-review.md` - Historical code analysis including stop code consistency
