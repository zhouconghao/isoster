"""
Multi-band isoster (Stage-1, experimental).

Joint elliptical isophote fitting on multiple aligned same-pixel-grid
images. Produces a single shared geometry per SMA with per-band
intensities and per-band harmonic deviations, replacing the traditional
forced-photometry workflow.

This is a parallel codebase: it does not modify any module under the
top-level ``isoster`` package. See ``docs/10-multiband.md`` for the
user-facing reference and
``docs/agent/plan-2026-04-29-multiband-feasibility.md`` for the locked
design record.
"""

from .config_mb import IsosterConfigMB

__all__ = ["IsosterConfigMB"]
