# Eccentric Anomaly (EA) Implementation in ISOSTER

This document describes the eccentric anomaly-based isophote fitting implementation in ISOSTER, based on Ciambur 2015 (ApJ 810:120).

## Background

Traditional isophote fitting (Jedrzejewski 1987) samples ellipses uniformly in position angle (φ). This works well for nearly circular isophotes but introduces biases for high-ellipticity galaxies:

1. **Non-uniform arc-length sampling**: Uniform φ sampling concentrates points near the minor axis
2. **Harmonic fitting bias**: The resulting non-uniform sampling affects harmonic decomposition accuracy
3. **Higher-order harmonic contamination**: Sequential fitting of harmonics doesn't account for cross-correlations

Ciambur 2015 proposes using **eccentric anomaly (ψ)** instead of position angle (φ) to achieve uniform arc-length sampling.

## Mathematical Foundation

### Eccentric Anomaly Definition

For an ellipse with semi-major axis `a` and semi-minor axis `b = a(1-ε)`:

- **Standard definition**: `x = a·cos(ψ)`, `y = b·sin(ψ)`
- **Relationship to position angle**: `tan(φ) = (1-ε)·tan(ψ)`

This means sampling uniformly in ψ produces uniform arc-length coverage.

### Conversion Formulas

From ψ to φ (used in ISOSTER):
```python
φ = arctan2((1-ε)·sin(ψ), cos(ψ))
```

From φ to ψ (used in ISOFIT):
```
ψ = -arctan(tan(φ)/(1-ε))  # Note: ISOFIT uses negative sign
```

**Important**: ISOSTER uses the standard convention where ψ and φ rotate in the same direction. ISOFIT uses the Ciambur 2015 convention where they rotate oppositely.

## ISOSTER Implementation

### Configuration

Enable EA sampling with:
```python
from isoster import fit_image, IsosterConfig

config = IsosterConfig(
    use_eccentric_anomaly=True,        # Enable EA sampling
    simultaneous_harmonics=False,      # Optional: enable simultaneous fitting
    harmonic_orders=[3, 4],            # Harmonics to fit
)
```

### Key Components

#### 1. EA Sampling (`isoster/sampling.py`)

When `use_eccentric_anomaly=True`:
- Sample uniformly in ψ (eccentric anomaly)
- Convert ψ → φ for coordinate calculation
- Return both `angles` (ψ for harmonics) and `phi` (φ for geometry)

```python
IsophoteData = namedtuple('IsophoteData', ['angles', 'phi', 'intens', 'radii'])
```

#### 2. Harmonic Fitting (`isoster/fitting.py`)

1st and 2nd harmonics are fitted in the appropriate angle space:
- **EA mode**: Fit `I(ψ) = I₀ + A₁sin(ψ) + B₁cos(ψ) + A₂sin(2ψ) + B₂cos(2ψ)`
- **Regular mode**: Fit `I(φ) = I₀ + A₁sin(φ) + B₁cos(φ) + A₂sin(2φ) + B₂cos(2φ)`

#### 3. Geometry Updates

Geometry updates (x0, y0, eps, pa) are computed using the Jedrzejewski formulas which operate in φ-space. The harmonic coefficients from ψ-space are used directly as they represent the same geometric deviations.

#### 4. Higher-Order Harmonics

Two modes available:

**Sequential (default)**:
```python
a3, b3 = compute_deviations(angles, intens, sma, gradient, 3)
a4, b4 = compute_deviations(angles, intens, sma, gradient, 4)
```

**Simultaneous (ISOFIT-style)**:
```python
config = IsosterConfig(
    use_eccentric_anomaly=True,
    simultaneous_harmonics=True,
    harmonic_orders=[3, 4, 5, 6],  # Can extend beyond [3, 4]
)
```

This fits all higher harmonics together, accounting for cross-correlations.

## Comparison with ISOFIT (IRAF)

| Feature | ISOSTER | ISOFIT |
|---------|---------|--------|
| ψ convention | Same direction as φ | Opposite direction (negative) |
| 1st/2nd harmonics | Simultaneous (5 params) | Separate call (4 params) |
| Higher harmonics | Sequential or simultaneous | Simultaneous only |
| Quadrant handling | `arctan2` | Manual quadrant correction |

### ISOFIT's el_harmonics2

ISOFIT converts φ → ψ at fitting time:
```fortran
if ((bufx[i] > PI/2) && (bufx[i] < 3*PI/2))
    psi = -arctan(tan(bufx[i])/(1-eps)) + PI
else
    psi = -arctan(tan(bufx[i])/(1-eps))
```

ISOSTER instead samples directly in ψ and stores both ψ and φ.

## When to Use EA Mode

Recommended for:
- **High ellipticity**: ε > 0.3 (b/a < 0.7)
- **Precision measurements**: When a3, b3, a4, b4 accuracy matters
- **Large galaxies**: More samples benefit from uniform arc-length

Not necessary for:
- Nearly circular isophotes (ε < 0.2)
- Quick/rough measurements
- Low S/N data where sampling uniformity is dominated by noise

## Implementation Details

### Files Modified

- `isoster/numba_kernels.py`: EA-to-PA conversion (`ea_to_pa`, `compute_ellipse_coords`)
- `isoster/sampling.py`: EA sampling mode
- `isoster/fitting.py`: EA harmonic fitting, `fit_higher_harmonics_simultaneous`
- `isoster/config.py`: New options (`simultaneous_harmonics`, `harmonic_orders`)

### Bug Fix (2024)

**Issue**: Higher-order harmonics (a3, b3, a4, b4) were computed using φ instead of ψ in EA mode.

**Location**: `isoster/fitting.py:646-647`

**Fix**: Changed `compute_deviations(phi, ...)` to `compute_deviations(angles, ...)` to use the correct angle array.

## References

1. Ciambur, B. C. 2015, ApJ, 810, 120 - "ISOFIT: Improved modeling of isophotal shapes"
2. Jedrzejewski, R. I. 1987, MNRAS, 226, 747 - "CCD surface photometry of elliptical galaxies"
