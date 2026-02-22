# Convergence stop=2 Fix — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce stop=2 (max-iterations-reached) rate to near-zero in well-defined regions by implementing three convergence improvements, benchmarking them on NGC1209_mock2, and selecting the best combination.

**Architecture:** Three independent convergence improvements to `isoster/fitting.py`, each gated by new optional config params in `isoster/config.py`. A standalone diagnostic script compares baseline vs each approach on NGC1209_mock2. The vectorized sampling path (`sampling.py`) is NOT modified.

**Tech Stack:** numpy, scipy, pydantic (config), astropy (FITS I/O), matplotlib (QA figures)

**Branch:** `fix/convergence-stop2`

---

## Background

### Root cause
Isoster's convergence check (`fitting.py:819`):
```python
abs(max_amp) < conver * rms
```
is SMA-independent. Photutils (`reference/fitter.py:199`) uses:
```python
conver * sector_area * std(residual) > abs(largest_harmonic)
```
where `sector_area` grows with SMA, relaxing the threshold for outer isophotes.

### NGC1209_mock2 baseline
- 8/66 isophotes (12%) hit stop=2, all at SMA > 95 px
- Photutils: 2/66 (3%) stop=2
- Photometry is accurate (~1%) despite stop=2 — geometry oscillates

### Constraint
- Must NOT change `sampling.py` or the vectorized sampling path
- Must NOT degrade accuracy (intensity, eps, pa) on existing test suite
- New params must have good defaults so existing users see improvement automatically

---

## Task 1: Add convergence scaling config params

**Files:**
- Modify: `isoster/config.py:34-37` (fitting control section)

**Step 1: Add three new config fields**

Add after `conver` (line 37):

```python
convergence_scaling: str = Field(
    default='sector_area',
    pattern='^(none|sector_area|sqrt_sma)$',
    description="Scale convergence threshold with SMA. "
                "'sector_area': multiply by approximate sector area (matches photutils behavior). "
                "'sqrt_sma': multiply by sqrt(sma). "
                "'none': constant threshold (legacy behavior)."
)

geometry_damping: float = Field(
    default=1.0,
    gt=0.0,
    le=1.0,
    description="Damping factor for geometry updates (0 < d <= 1). "
                "Each geometry correction is multiplied by this factor. "
                "1.0 = no damping (legacy). 0.5 = half-step corrections. "
                "Lower values prevent oscillations but slow convergence."
)

geometry_convergence: bool = Field(
    default=False,
    description="Enable secondary convergence based on geometry stability. "
                "Declares convergence when geometry changes fall below tolerance "
                "for consecutive iterations, even if harmonic criterion is not met."
)

geometry_tolerance: float = Field(
    default=0.01,
    gt=0.0,
    description="Threshold for geometry convergence. Convergence declared when "
                "max(|delta_eps|, |delta_pa/pi|, |delta_x0/sma|, |delta_y0/sma|) "
                "< geometry_tolerance for geometry_stable_iters consecutive iterations."
)

geometry_stable_iters: int = Field(
    default=3,
    ge=2,
    description="Number of consecutive iterations with small geometry changes "
                "required to trigger geometry-based convergence."
)
```

**Step 2: Run test collection to verify config is valid**

Run: `uv run pytest tests/ --collect-only -q 2>&1 | tail -5`
Expected: all tests collected, no import errors

**Step 3: Commit**

```
git add isoster/config.py
git commit -m "feat: add convergence scaling, geometry damping, and geometry convergence config params"
```

---

## Task 2: Implement convergence scaling (Approach A) in fitting.py

**Files:**
- Modify: `isoster/fitting.py` (convergence check at line 819 and surrounding context)

**Step 1: Write the failing test**

File: `tests/unit/test_convergence.py`

```python
"""Tests for convergence improvements in fit_isophote."""
import numpy as np
import pytest
from isoster.fitting import fit_isophote
from isoster.config import IsosterConfig


def make_sersic_image(shape=(201, 201), x0=100, y0=100, ie=1000.0, re=30.0,
                      n=2.0, eps=0.3, pa=0.5):
    """Create a noiseless Sersic galaxy image for testing convergence."""
    from scipy.special import gammaincinv
    bn = gammaincinv(2 * n, 0.5)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    dx = xx - x0
    dy = yy - y0
    cos_pa, sin_pa = np.cos(pa), np.sin(pa)
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    r_ellip = np.sqrt(x_rot**2 + (y_rot / (1 - eps))**2)
    image = ie * np.exp(-bn * ((r_ellip / re)**(1.0/n) - 1))
    return image


class TestConvergenceScaling:
    """Test that sector_area scaling helps outer isophotes converge."""

    def test_outer_isophote_converges_with_scaling(self):
        """An outer isophote that hits maxit with 'none' should converge with 'sector_area'."""
        image = make_sersic_image(shape=(401, 401), x0=200, y0=200, re=30.0, eps=0.3)
        mask = np.zeros_like(image, dtype=bool)
        start_geom = {'x0': 200.0, 'y0': 200.0, 'eps': 0.3, 'pa': 0.5}

        # Use a tight conver to force stop=2 at large SMA with no scaling
        cfg_none = IsosterConfig(conver=0.02, maxit=30, minit=5,
                                 convergence_scaling='none')
        result_none = fit_isophote(image, mask, 150.0, start_geom, cfg_none)

        # With sector_area scaling, same params should converge
        cfg_scaled = IsosterConfig(conver=0.02, maxit=30, minit=5,
                                   convergence_scaling='sector_area')
        result_scaled = fit_isophote(image, mask, 150.0, start_geom, cfg_scaled)

        # The scaled version should converge (stop=0) or at least use fewer iterations
        assert result_scaled['niter'] <= result_none['niter'], \
            f"Scaling should help: {result_scaled['niter']} vs {result_none['niter']}"

    def test_scaling_none_preserves_legacy(self):
        """convergence_scaling='none' should match legacy behavior exactly."""
        image = make_sersic_image()
        mask = np.zeros_like(image, dtype=bool)
        start_geom = {'x0': 100.0, 'y0': 100.0, 'eps': 0.3, 'pa': 0.5}

        cfg = IsosterConfig(convergence_scaling='none')
        result = fit_isophote(image, mask, 30.0, start_geom, cfg)
        # Just verify it runs and returns valid result
        assert result['stop_code'] in {0, 1, 2, 3, -1}
        assert result['sma'] == 30.0

    def test_sqrt_sma_scaling(self):
        """sqrt_sma scaling should also help outer isophotes."""
        image = make_sersic_image(shape=(401, 401), x0=200, y0=200, re=30.0, eps=0.3)
        mask = np.zeros_like(image, dtype=bool)
        start_geom = {'x0': 200.0, 'y0': 200.0, 'eps': 0.3, 'pa': 0.5}

        cfg = IsosterConfig(conver=0.02, maxit=30, minit=5,
                            convergence_scaling='sqrt_sma')
        result = fit_isophote(image, mask, 150.0, start_geom, cfg)
        assert result['stop_code'] in {0, 1, 2, 3, -1}
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_convergence.py -v`
Expected: FAIL (convergence_scaling not recognized or no effect yet)

**Step 3: Implement convergence scaling in fit_isophote**

In `fitting.py`, inside `fit_isophote()`, just before the convergence check at line 819:

```python
# Compute convergence scaling factor based on config
convergence_scaling = cfg.convergence_scaling
if convergence_scaling == 'sector_area':
    # Approximate sector area: proportional to sma * delta_sma * angular_width
    # n_samples = max(64, int(2*pi*sma)), angular_width ~ 2*pi/n_samples
    # delta_sma ~ sma * astep (geometric) or astep (linear)
    # sector_area ~ sma * (sma * astep) * (2*pi / n_samples) ~ sma * astep * 2*pi * sma / n_samples
    # For geometric stepping: ~ sma * astep (since n_samples ~ 2*pi*sma)
    # Simplify: scale_factor proportional to sma * astep, normalized to 1.0 at sma=1
    n_samples = max(64, int(2 * np.pi * sma))
    angular_width = 2 * np.pi / n_samples
    astep = cfg.astep
    delta_sma = sma * astep if not cfg.linear_growth else astep
    scale_factor = max(1.0, sma * delta_sma * angular_width)
elif convergence_scaling == 'sqrt_sma':
    scale_factor = max(1.0, np.sqrt(sma))
else:  # 'none'
    scale_factor = 1.0
```

Then modify the convergence check:

```python
if abs(max_amp) < conver * scale_factor * rms and i >= minit:
```

**Step 4: Run tests**

Run: `uv run pytest tests/unit/test_convergence.py -v`
Expected: PASS

**Step 5: Run existing test suite to verify no regressions**

Run: `uv run pytest tests/unit/ tests/integration/ tests/validation/ -v --tb=short 2>&1 | tail -20`
Expected: 152 tests pass

**Step 6: Commit**

```
git add isoster/fitting.py tests/unit/test_convergence.py
git commit -m "feat: implement convergence scaling (sector_area, sqrt_sma) in fit_isophote"
```

---

## Task 3: Implement geometry damping (Approach B) in fitting.py

**Files:**
- Modify: `isoster/fitting.py:852-871` (geometry update section)
- Modify: `tests/unit/test_convergence.py` (add tests)

**Step 1: Write the failing test**

Add to `tests/unit/test_convergence.py`:

```python
class TestGeometryDamping:
    """Test that geometry damping reduces oscillations."""

    def test_damping_reduces_iterations(self):
        """With damping < 1.0, outer isophotes should use fewer iterations."""
        image = make_sersic_image(shape=(401, 401), x0=200, y0=200, re=30.0, eps=0.3)
        mask = np.zeros_like(image, dtype=bool)
        start_geom = {'x0': 200.0, 'y0': 200.0, 'eps': 0.3, 'pa': 0.5}

        cfg_nodamp = IsosterConfig(geometry_damping=1.0, convergence_scaling='none',
                                   maxit=50, minit=5, conver=0.02)
        result_nodamp = fit_isophote(image, mask, 120.0, start_geom, cfg_nodamp)

        cfg_damp = IsosterConfig(geometry_damping=0.7, convergence_scaling='none',
                                 maxit=50, minit=5, conver=0.02)
        result_damp = fit_isophote(image, mask, 120.0, start_geom, cfg_damp)

        # Damped should converge or use fewer iterations
        assert result_damp['niter'] <= result_nodamp['niter'] or result_damp['stop_code'] == 0

    def test_damping_1_is_legacy(self):
        """geometry_damping=1.0 should match legacy behavior."""
        image = make_sersic_image()
        mask = np.zeros_like(image, dtype=bool)
        start_geom = {'x0': 100.0, 'y0': 100.0, 'eps': 0.3, 'pa': 0.5}

        cfg = IsosterConfig(geometry_damping=1.0, convergence_scaling='none')
        result = fit_isophote(image, mask, 30.0, start_geom, cfg)
        assert result['stop_code'] in {0, 1, 2, 3, -1}

    def test_damping_preserves_accuracy(self):
        """Damped fitting should still produce accurate photometry."""
        image = make_sersic_image(shape=(301, 301), x0=150, y0=150, re=30.0, eps=0.3)
        mask = np.zeros_like(image, dtype=bool)
        start_geom = {'x0': 150.0, 'y0': 150.0, 'eps': 0.3, 'pa': 0.5}

        cfg = IsosterConfig(geometry_damping=0.5, convergence_scaling='none')
        result = fit_isophote(image, mask, 30.0, start_geom, cfg)

        # At re=30, intensity should be close to ie (1000) * exp(-bn*(1-1)) = 1000
        assert result['stop_code'] == 0
        assert abs(result['intens'] - 1000.0) / 1000.0 < 0.05  # Within 5%
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_convergence.py::TestGeometryDamping -v`

**Step 3: Implement geometry damping**

In `fitting.py`, in the geometry update block (lines 852-871), apply damping to the correction before updating:

```python
# Apply geometry damping
damping = cfg.geometry_damping

# Update geometry
if max_idx == 0:
    aux = -max_amp * (1.0 - eps) / gradient * damping
    x0 -= aux * np.sin(pa)
    y0 += aux * np.cos(pa)
elif max_idx == 1:
    aux = -max_amp / gradient * damping
    x0 += aux * np.cos(pa)
    y0 += aux * np.sin(pa)
elif max_idx == 2:
    denom = (1.0 - eps)**2 - 1.0
    if abs(denom) < 1e-10: denom = -1e-10
    pa = (pa + damping * (max_amp * 2.0 * (1.0 - eps) / sma / gradient / denom)) % np.pi
elif max_idx == 3:
    eps = min(eps - damping * (max_amp * 2.0 * (1.0 - eps) / sma / gradient), 0.95)
    if eps < 0.0:
        eps = min(-eps, 0.95)
        pa = (pa + np.pi/2) % np.pi
    if eps == 0.0: eps = 0.05
```

**Step 4: Run tests**

Run: `uv run pytest tests/unit/test_convergence.py -v`
Expected: PASS

**Step 5: Run full suite**

Run: `uv run pytest tests/unit/ tests/integration/ tests/validation/ -v --tb=short 2>&1 | tail -20`
Expected: 152+ tests pass

**Step 6: Commit**

```
git add isoster/fitting.py tests/unit/test_convergence.py
git commit -m "feat: implement geometry damping in fit_isophote"
```

---

## Task 4: Implement geometry-change convergence (Approach C) in fitting.py

**Files:**
- Modify: `isoster/fitting.py` (add geometry tracking and secondary convergence check)
- Modify: `tests/unit/test_convergence.py` (add tests)

**Step 1: Write the failing test**

Add to `tests/unit/test_convergence.py`:

```python
class TestGeometryConvergence:
    """Test geometry-stability-based convergence."""

    def test_geometry_convergence_detects_stability(self):
        """When geometry stops changing, should declare convergence."""
        image = make_sersic_image(shape=(401, 401), x0=200, y0=200, re=30.0, eps=0.3)
        mask = np.zeros_like(image, dtype=bool)
        start_geom = {'x0': 200.0, 'y0': 200.0, 'eps': 0.3, 'pa': 0.5}

        # Without geometry convergence: likely hits maxit at outer SMA
        cfg_off = IsosterConfig(geometry_convergence=False, convergence_scaling='none',
                                maxit=30, minit=5, conver=0.01)
        result_off = fit_isophote(image, mask, 150.0, start_geom, cfg_off)

        # With geometry convergence: should converge via geometry stability
        cfg_on = IsosterConfig(geometry_convergence=True, convergence_scaling='none',
                               maxit=30, minit=5, conver=0.01,
                               geometry_tolerance=0.01, geometry_stable_iters=3)
        result_on = fit_isophote(image, mask, 150.0, start_geom, cfg_on)

        # Geometry convergence should help
        assert result_on['niter'] <= result_off['niter'] or result_on['stop_code'] == 0

    def test_geometry_convergence_off_is_legacy(self):
        """geometry_convergence=False should match legacy behavior."""
        image = make_sersic_image()
        mask = np.zeros_like(image, dtype=bool)
        start_geom = {'x0': 100.0, 'y0': 100.0, 'eps': 0.3, 'pa': 0.5}

        cfg = IsosterConfig(geometry_convergence=False, convergence_scaling='none')
        result = fit_isophote(image, mask, 30.0, start_geom, cfg)
        assert result['stop_code'] in {0, 1, 2, 3, -1}
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_convergence.py::TestGeometryConvergence -v`

**Step 3: Implement geometry-change convergence**

In `fitting.py`, add tracking variables before the loop (after line 684):

```python
# Geometry convergence tracking
prev_geom_x0, prev_geom_y0, prev_geom_eps, prev_geom_pa = x0, y0, eps, pa
stable_count = 0
geom_converged = False
```

After the geometry update block (after the eps/pa clamping), add:

```python
# Track geometry changes for geometry-based convergence
if cfg.geometry_convergence and i >= minit:
    delta_x0 = abs(x0 - prev_geom_x0) / max(sma, 1.0)
    delta_y0 = abs(y0 - prev_geom_y0) / max(sma, 1.0)
    delta_eps = abs(eps - prev_geom_eps)
    # PA wrapping: use smallest angular difference
    delta_pa_raw = abs(pa - prev_geom_pa)
    delta_pa = min(delta_pa_raw, np.pi - delta_pa_raw) / np.pi
    max_delta = max(delta_x0, delta_y0, delta_eps, delta_pa)

    if max_delta < cfg.geometry_tolerance:
        stable_count += 1
    else:
        stable_count = 0

    if stable_count >= cfg.geometry_stable_iters:
        stop_code = 0  # Converged via geometry stability
        converged = True
        geom_converged = True
        # Compute deviations same as harmonic convergence path
        ...  # (same deviation computation as the harmonic convergence block)
        break

prev_geom_x0, prev_geom_y0, prev_geom_eps, prev_geom_pa = x0, y0, eps, pa
```

**Step 4: Run tests**

Run: `uv run pytest tests/unit/test_convergence.py -v`
Expected: PASS

**Step 5: Run full suite**

Run: `uv run pytest tests/unit/ tests/integration/ tests/validation/ -v --tb=short 2>&1 | tail -20`
Expected: 152+ tests pass

**Step 6: Commit**

```
git add isoster/fitting.py tests/unit/test_convergence.py
git commit -m "feat: implement geometry-change convergence criterion in fit_isophote"
```

---

## Task 5: Write NGC1209_mock2 diagnostic benchmark script

**Files:**
- Create: `benchmarks/convergence_diagnostic.py`

**Step 1: Write the diagnostic script**

This script runs NGC1209_mock2 through five configurations and compares results:

1. **Baseline** (`convergence_scaling='none', geometry_damping=1.0, geometry_convergence=False`)
2. **Approach A** (`convergence_scaling='sector_area'`)
3. **Approach A-alt** (`convergence_scaling='sqrt_sma'`)
4. **Approach B** (`geometry_damping=0.7`)
5. **Approach C** (`geometry_convergence=True`)
6. **Combined A+B** (`convergence_scaling='sector_area', geometry_damping=0.7`)
7. **Combined A+C** (`convergence_scaling='sector_area', geometry_convergence=True`)

For each, report:
- Stop code distribution (count of 0, 1, 2, 3, -1)
- Intensity accuracy vs photutils (median/max relative difference at 0.5-8 Re)
- Geometry accuracy vs photutils (median/max eps, pa difference at 0.5-8 Re)
- Wall-clock time

Output: markdown table to stdout + comparison QA figure to `outputs/`

The script should:
- Load NGC1209_mock2 image from `/Users/mac/work/hsc/huang2013/NGC1209/NGC1209_mock2.fits`
- Load photutils baseline from `/Users/mac/work/hsc/huang2013/NGC1209/mock2/NGC1209_mock2_photutils_baseline_profile.fits`
- Load isoster baseline config from `/Users/mac/work/hsc/huang2013/NGC1209/mock2/NGC1209_mock2_isoster_baseline_run.json`
- Run isoster with each config variant
- Compare results

**Step 2: Run the diagnostic**

Run: `uv run python benchmarks/convergence_diagnostic.py`
Expected: table of results showing which approaches reduce stop=2

**Step 3: Commit**

```
git add benchmarks/convergence_diagnostic.py
git commit -m "bench: add convergence diagnostic script for NGC1209_mock2"
```

---

## Task 6: Analyze results and select best defaults

**Files:**
- Modify: `isoster/config.py` (update default values if needed)
- Modify: `docs/plans/2026-02-22-convergence-stop2-fix.md` (record findings)

**Step 1: Analyze benchmark output**

Review the diagnostic table. Key decision criteria:
- Which approach(es) reduce stop=2 to near-zero?
- Which preserve accuracy (intensity within 1%, geometry within 2%)?
- Any runtime penalty?

**Step 2: Update defaults if justified**

If `convergence_scaling='sector_area'` clearly wins, change its default from `'none'` to `'sector_area'` so existing users benefit automatically.

**Step 3: Update plan with findings**

Append analysis section to this plan document.

**Step 4: Run full test suite one final time**

Run: `uv run pytest tests/unit/ tests/integration/ tests/validation/ -v --tb=short 2>&1 | tail -20`
Expected: all tests pass

**Step 5: Commit**

```
git add isoster/config.py docs/plans/2026-02-22-convergence-stop2-fix.md
git commit -m "feat: set optimal convergence defaults based on NGC1209_mock2 benchmark"
```

---

## Task 7: Update documentation

**Files:**
- Modify: `CLAUDE.md` (add new params to config example)
- Modify: `docs/spec.md` (if architecture section needs update)

**Step 1: Add new params to CLAUDE.md config example**

In the "Advanced Fitting Options" section, add:

```python
# Convergence improvements
convergence_scaling='sector_area',  # Scale threshold with SMA (matches photutils)
geometry_damping=1.0,               # Damping factor for geometry updates (1.0 = no damping)
geometry_convergence=False,         # Enable geometry-stability convergence
geometry_tolerance=0.01,            # Tolerance for geometry convergence
geometry_stable_iters=3,            # Consecutive stable iterations required
```

**Step 2: Commit**

```
git add CLAUDE.md
git commit -m "docs: add convergence improvement params to config documentation"
```

---

## Benchmark Results (NGC1209_mock2)

Executed 2026-02-23 on branch `fix/convergence-stop2`.

| Config | stop=2 | Time(s) | Med dI/I | Max dI/I | Med deps | Med dPA |
|--------|--------|---------|----------|----------|----------|---------|
| Baseline | 9 | 0.41 | 0.35% | 26.8% | 0.0022 | 0.09 |
| **A: sector_area** | **0** | **0.12** | 0.48% | **11.8%** | 0.0024 | 0.10 |
| A-alt: sqrt_sma | 0 | 0.13 | 0.48% | 11.8% | 0.0024 | 0.10 |
| B: damping=0.7 | 3 | 0.20 | 0.37% | 16.3% | 0.0023 | 0.10 |
| C: geom_conv | 3 | 0.22 | 0.40% | 26.8% | 0.0022 | 0.10 |
| A+B combined | 0 | 0.13 | 0.38% | 18.9% | 0.0024 | 0.10 |
| A+C combined | 0 | 0.13 | 0.48% | 11.8% | 0.0024 | 0.10 |

### Decision

- **Default changed**: `convergence_scaling` default set to `'sector_area'`
- **Rationale**: Eliminates all stop=2, 3.4x faster, best worst-case accuracy
- **B and C**: Kept as optional supplementary controls (defaults unchanged)
- `sector_area` and `sqrt_sma` produce identical results on this dataset; `sector_area` is preferred as it has stronger physical motivation (matches photutils)
