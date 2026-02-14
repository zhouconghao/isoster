# Future Work

This document tracks realistic long-term upgrades that are not yet implemented.

## Long-Term Engineering Upgrades

### 1. Parallel forced/template execution path

Why:

- Forced and template-based forced photometry process each SMA independently.
- Current driver loops are serial (`isoster/driver.py`).

Target:

- Optional parallel backend for forced/template modes with deterministic ordering.
- Keep regular mode serial unless dependency-safe decomposition is proven.

### 2. Typed isophote result schema

Why:

- Core outputs are dict-based and shape can vary by config/options.
- This increases downstream fragility in modeling/serialization code (`isoster/model.py`, `isoster/utils.py`).

Target:

- Introduce a typed result model (dataclass or pydantic model) with explicit optional blocks.
- Preserve backward-compatible dict export methods.

### 3. Model builder robustness against low-quality rows

Why:

- `build_isoster_model` currently accepts all `sma > 0` rows, regardless of stop code and NaN content (`isoster/model.py`).

Target:

- Add explicit stop-code and finite-value filtering options.
- Provide strict/default filter presets for reproducible QA workflows.

### 4. CLI option parity with advanced config features

Why:

- CLI currently exposes only a small subset of `IsosterConfig` fields (`isoster/cli.py`).

Target:

- Add structured CLI support (or schema-driven config validation command) for advanced options:
  - eccentric anomaly mode
  - harmonic orders/simultaneous fitting
  - CoG options
  - permissive geometry

## Long-Term Algorithm and Performance Research

### 1. Gradient estimator redesign

Why:

- Current gradient logic is tightly coupled to fixed normalization and two-radius sampling in `compute_gradient` (`isoster/fitting.py`).

Research direction:

- Compare alternative finite-difference schemes and robust local profile fits.
- Validate against reference profiles with quantitative error statistics before adoption.

### 2. Adaptive sampling density with error controls

Why:

- Sampling density currently follows `max(64, int(2*pi*sma))` (`isoster/sampling.py`).

Research direction:

- Evaluate adaptive density by ellipticity/SNR/local curvature.
- Require measured impact on profile accuracy and runtime variance.

### 3. Geometry-history-aware stabilization policy

Why:

- Central-regularization intent exists, but long-term stability policy should be made explicit across inward/outward passes.

Research direction:

- Design a history-aware stabilization mechanism with traceable diagnostics.
- Benchmark against real-data and mock workflows before promoting to default behavior.

## Deferred Until Evidence Exists

The following are intentionally excluded from active plans until supported by benchmark evidence and maintenance cost analysis:

- full GPU rewrite paths
- JAX rewrite of the core fitting stack
- Rust/C++ core replacement

These may be revisited after current architecture reaches stability and test depth targets.
