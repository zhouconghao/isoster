# High-Order Harmonics Implementation Review — 2026-02-24

## Discovery: Harmonics Are Post-Hoc Only (Not True ISOFIT)

The current `simultaneous_harmonics=True` option does **not** implement true ISOFIT (Ciambur 2015) behavior. Higher-order harmonics (n≥3) are computed as post-hoc diagnostics after convergence, never entering the iterative fitting loop.

## Current Two-Phase Architecture

### Phase 1 — Iterative geometry fitting (`fit_isophote`, lines 701-931)

- Fits ONLY 1st and 2nd order harmonics (5 parameters):
  `I(θ) = I₀ + A₁sin(θ) + B₁cos(θ) + A₂sin(2θ) + B₂cos(2θ)`
- Uses A₁, B₁ to update center (x0, y0)
- Uses A₂ to update position angle (pa)
- Uses B₂ to update ellipticity (eps)
- Convergence: `max(|A₁|, |B₁|, |A₂|, |B₂|) < conver × scale × rms`
- RMS = `std(data - 5-param model)`

### Phase 2 — Post-hoc harmonic extraction (runs ONLY after Phase 1 converges)

- `simultaneous_harmonics=False` (default): `compute_deviations()` fits each order independently — `I(θ) = c₀ + aₙsin(nθ) + bₙcos(nθ)` per order n
- `simultaneous_harmonics=True`: `fit_higher_harmonics_simultaneous()` fits all orders together: `I(θ) = c₀ + Σₙ[aₙsin(nθ) + bₙcos(nθ)]`
- Both normalize coefficients by `sma × |gradient|` for dimensionless deviations

## Three Problems with This Approach

### 1. Higher-order signal contaminates geometry

If a galaxy has strong a₄ (boxy/disky) structure, that signal is present during Phase 1. The 5-parameter model doesn't account for it, so higher-order power leaks into the RMS and can bias A₂/B₂ estimates (which control pa/eps). Cross-correlations between harmonic orders are ignored.

### 2. Inflated RMS affects convergence behavior

Convergence tests harmonic amplitudes against `conver × rms`, but the RMS includes unmodeled higher-order structure. For galaxies with strong deviations from pure ellipses, the RMS is artificially high, making the threshold higher. The geometry converges but may be less accurate.

### 3. Post-hoc coefficients use biased geometry

Since geometry (x0, y0, eps, pa) was fitted without accounting for higher-order terms, the angles/radii used for Phase 2 harmonic extraction may be slightly wrong.

## What True ISOFIT (Ciambur 2015) Does

ISOFIT fits ALL harmonics simultaneously **within the iterative loop**:

1. Design matrix includes all orders: `I(θ) = I₀ + A₁sin + B₁cos + A₂sin2 + B₂cos2 + a₃sin3 + b₃cos3 + a₄sin4 + b₄cos4 + ...`
2. Geometry updates still use only A₁, B₁, A₂, B₂ (same formulas)
3. RMS is computed from residuals **after removing all harmonic components** → reflects truly random scatter
4. A₁, B₁, A₂, B₂ are fitted jointly with higher orders → cross-correlations properly handled
5. Convergence tested against this cleaner RMS

## Current Options Summary

| Option | What it controls | Enters iterative loop? |
|--------|-----------------|----------------------|
| `simultaneous_harmonics=False` | Each order fitted independently (post-hoc) | No |
| `simultaneous_harmonics=True` | All orders fitted jointly (post-hoc) | No |
| `harmonic_orders=[3,4]` | Which orders to compute | No |
| `harmonic_orders=[3,4,5,6,7]` | More orders to compute | No |

**None of these change the iterative fitting.** The geometry loop always uses the same 5-parameter model.

## Why ESO243-49 and NGC3610 Aren't Benefiting

These galaxies have strong non-elliptical structure (edge-on S0, boxy elliptical). The current approach correctly *measures* deviations (a₃, a₄, etc.) but doesn't *account for* them during fitting:
- Geometry may be biased by harmonic contamination
- Model reconstruction uses coefficients from a potentially biased geometry
- Residuals still show structured patterns because coefficients aren't optimal

## Required Change: Integrate Harmonics Into the Iterative Loop

To implement true ISOFIT:

1. **Replace** the 5-parameter `fit_first_and_second_harmonics()` inside the iteration loop with a joint fit of all requested orders (1, 2, 3, 4, ...) when `simultaneous_harmonics=True`
2. **Compute RMS** from residuals after removing ALL fitted harmonics
3. **Still use only A₁, B₁, A₂, B₂** for geometry updates (no change to update formulas)
4. **Test convergence** against the cleaned RMS
5. **Store all harmonic coefficients** at each iteration (not just at convergence)

### Key Functions to Modify

- `fit_isophote()` in `fitting.py` — main iteration loop
- `fit_first_and_second_harmonics()` → generalize to fit N orders simultaneously
- Convergence test (line 834) — use cleaned RMS
- `best_geometry` tracking — store harmonics during iteration, not only at convergence

### Design Considerations

- When `simultaneous_harmonics=False`, behavior should remain identical (backward compatible)
- When `simultaneous_harmonics=True`, the design matrix grows: 5 params → `1 + 2*len(harmonic_orders) + 4` (intercept + 2 per higher order + 4 for orders 1,2)
- Need enough data points: `n_samples > n_params` (already checked in `fit_higher_harmonics_simultaneous`)
- The `compute_deviations()` sequential function becomes unnecessary when using true simultaneous mode, but should be kept for backward compatibility

## Related Files

- `isoster/fitting.py` — core fitting loop and harmonic functions
- `isoster/sampling.py` — data extraction (no changes needed)
- `isoster/model.py` — model reconstruction (no changes needed, uses same coefficients)
- `isoster/driver.py` — orchestration (no changes needed)
- `isoster/config.py` — may need new config option to distinguish legacy vs true ISOFIT mode
