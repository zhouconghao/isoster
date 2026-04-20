"""Exhausted benchmark campaign framework.

Multi-tool, multi-arm, multi-galaxy, multi-dataset benchmark for
isoster's ellipse-fitting algorithm. Driven by a single campaign YAML;
outputs land outside the repository at the root declared in the YAML.

Top-level entry point: ``python -m benchmarks.exhausted.orchestrator.cli``
(future: ``isoster-campaign`` console script).

Structure:
- ``configs/``       arm rosters and campaign YAML templates
- ``adapters/``      per-dataset loaders returning (image, variance, mask, geometry, Re)
- ``fitters/``       per-tool single-galaxy fit drivers (isoster, photutils, autoprof)
- ``analysis/``      metrics, residual zones, quality flags, inventory writer
- ``plotting/``      per-galaxy QA and multi-galaxy summary grids
- ``orchestrator/``  CLI, runner, stats, reports
- ``legacy/``        archived 39-config single-galaxy sweep (see legacy/README.md)
"""
