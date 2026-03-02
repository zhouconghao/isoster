# IC3370 Exhaustive Configuration Registry

Exhaustive benchmark of all meaningful isoster configuration parameters on IC3370_mock2
(Huang2013 sample). IC3370 is a challenging galaxy: baseline shows only 30/59 isophotes
converging (stop=0), with 25 stop=2 and 4 stop=-1 failures.

## Galaxy-Specific Base Parameters

From IC3370 baseline JSON:

| Parameter | Value |
|-----------|-------|
| x0 | 566.0 |
| y0 | 566.0 |
| eps | 0.239 |
| pa | -0.489 rad |
| sma0 | 6.0 |
| maxsma | 283.0 |
| nclip | 2 |
| maxit | 100 |
| astep | 0.1 |
| maxgerr | 0.5 |

## Baselines

| ID | Description |
|----|-------------|
| P00 | Photutils baseline (`photutils.isophote.Ellipse`) |
| S00 | Isoster baseline (current defaults: sector_area, damping=0.7, largest, maxit=100) |

## Single-Parameter Sweeps (S01-S23)

Each changes ONE parameter from S00 (except where noted).

| ID | Changed Parameter | Value | Default | Notes |
|----|------------------|-------|---------|-------|
| S01 | `convergence_scaling` | `'none'` | `'sector_area'` | Legacy constant threshold |
| S02 | `convergence_scaling` | `'sqrt_sma'` | `'sector_area'` | Alternative scaling |
| S03 | `geometry_damping` | `0.5` | `0.7` | Stronger damping |
| S04 | `geometry_damping` | `1.0` | `0.7` | No damping (legacy) |
| S05 | `geometry_update_mode` | `'simultaneous'` (+damping=0.5) | `'largest'` | All 4 params per iter |
| S06 | `geometry_convergence` | `True` | `False` | Geometry stability criterion |
| S07 | `permissive_geometry` | `True` | `False` | Photutils-style best effort |
| S08 | `use_eccentric_anomaly` | `True` | `False` | Uniform arc-length sampling |
| S09 | `simultaneous_harmonics` | `True` (in_loop) | `False` | True ISOFIT simultaneous |
| S10 | `simultaneous_harmonics` + `isofit_mode` | `True`, `'original'` | `False` | Ciambur 2015 post-hoc |
| S11 | `harmonic_orders` | `[3,4,5,6,7]` | `[3,4]` | Extended harmonics |
| S12 | `use_central_regularization` | `True` | `False` | Central geometry stabilization |
| S13 | `integrator` | `'median'` | `'mean'` | Median integration |
| S14 | `integrator` + `lsb_sma_threshold` | `'adaptive'`, `100.0` | `'mean'` | Adaptive integration |
| S15 | `conver` | `0.02` | `0.05` | Stricter convergence |
| S16 | `conver` | `0.10` | `0.05` | Looser convergence |
| S17 | `maxit` | `50` | `100` | Halved iteration budget |
| S18 | `maxit` | `300` | `100` | Extreme iteration budget |
| S19 | `sclip` | `2.0` | `3.0` | Stricter sigma clipping |
| S20 | `sclip_low` + `sclip_high` | `3.0`, `2.0` | symmetric `3.0` | Asymmetric clipping |
| S21 | `full_photometry` + `compute_cog` | `True`, `True` | `False`, `False` | Full photometry + CoG |
| S22 | `fix_center` | `True` (at photutils median x0/y0) | `False` | Fixed center |
| S23 | `fix_center` + `fix_pa` + `fix_eps` | all `True` (photutils medians) | `False` | Fully fixed geometry |

**Notes:**
- S05 uses damping=0.5 (documented recommendation for simultaneous mode).
- S10 must set `simultaneous_harmonics=True` (config validator warns otherwise).
- S14 must set `lsb_sma_threshold` (hard validation requirement for adaptive integrator).
- S22/S23 derive fixed geometry values from photutils baseline median within 3 Re.

## Combination Configs (C01-C12)

| ID | Parameters | Values | Rationale |
|----|-----------|--------|-----------|
| C01 | scaling + damping | `'none'`, `1.0` | Pure legacy (no modern improvements) |
| C02 | permissive + geom_conv | `True`, `True` | Maximum tolerance |
| C03 | EA + ISOFIT in_loop | `True`, `True` | EA sampling + simultaneous harmonics |
| C04 | EA + extended harmonics | `True`, `[3,4,5,6,7]` | EA + more harmonic orders |
| C05 | ISOFIT + extended harmonics | `True`, `[3,4,5,6,7]` | Joint fitting with extended orders |
| C06 | simul_geom + damping=0.5 + geom_conv | `'simultaneous'`, `0.5`, `True` | Full simultaneous approach |
| C07 | permissive + scaling=none | `True`, `'none'` | Permissive without area scaling |
| C08 | central_reg + damping=0.5 | `True`, `0.5` | Inner stability + stronger damping |
| C09 | EA + ISOFIT_orig + extended | `True`, `True`, `[3,4,5,6,7]`, `'original'` | Full Ciambur 2015 |
| C10 | permissive + EA + ISOFIT | `True`, `True`, `True` | Kitchen sink |
| C11 | conver=0.02 + maxit=50 | `0.02`, `50` | Stress test: strict + low budget |
| C12 | sector_area + damping=0.5 + permissive | `'sector_area'`, `0.5`, `True` | Recommended robust combo |

## Metrics

### Global

| Metric | Description |
|--------|-------------|
| `wall_time` | Wall-clock seconds |
| `n_isophotes` | Total isophote count |
| `stop_0/1/2/3/-1` | Stop-code distribution |
| `n_matched` | Matched to photutils by SMA |
| `model_frac_med`, `model_rms` | 2D model residual stats |

### Per Radial Zone

Zones defined by photutils effective radius (Re):

| Zone | Range | Purpose |
|------|-------|---------|
| inner | SMA < 1 Re | Central stability |
| mid | 1 Re < SMA < 4 Re | Core accuracy |
| outer | SMA > 4 Re | Low-S/N outskirt behavior |

Per zone: `med_rel_intens`, `max_rel_intens`, `med_abs_eps`, `max_abs_eps`,
`med_abs_pa_deg`, `max_abs_pa_deg`.
