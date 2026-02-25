# Active Plan and Review

## Completed Summary

Phases 1-22 are complete. Detailed history has been archived to keep this file operational.

## History Archive

- Phase 1-6 progress summary: `docs/journal/2026-02-13-phase1-6-progress-summary.md`
- Phase 7 kickoff plan: `docs/journal/2026-02-13-phase7-doc-audit-and-code-review-plan.md`
- Phase 7-22 detailed checklist and review notes: `docs/todo-archive-phase7-22.md`

## Open Items

| # | Source | Summary | Priority |
|---|--------|---------|----------|
| Phase 22.3 | todo | Decide worktree hygiene policy for tracked `*.pyc` and session prompt files | Low |

## Phase 23: Config Validation Gaps

Audit of all 42 `IsosterConfig` parameters revealed 10+ validation gaps. Tracked below.

| ID | Category | Summary | Status | File(s) |
|----|----------|---------|--------|---------|
| V1 | Warning | `isofit_mode` is no-op when `simultaneous_harmonics=False` | Done | `config.py` |
| V2 | Warning | `maxsma < sma0` produces only one isophote + inward sweep | Done | `config.py` |
| V3 | Warning | `minsma >= sma0` means inward loop never runs | Done | `config.py` |
| V4 | Warning | `geometry_update_mode='simultaneous'` + `geometry_damping > 0.7` | Done | `config.py` |
| V5 | Warning | `forced=True` silently drops multiple config params | Done | `config.py` |
| V6 | Warning | `template_isophotes` + `forced=True` both active | Done | `driver.py` |
| V7 | Validate | `central_reg_weights` keys must be subset of `{'eps', 'pa', 'center'}` | Done | `config.py` |
| V8 | Cleanup | Remove harmonic keys when `compute_deviations=False` and `simultaneous_harmonics=False` | Done | `fitting.py` |
| V9 | Bug | Template forced mode hardcodes `debug=False` instead of `cfg.debug` | Done | `driver.py` |
| V10 | Warning | `maxit < minit + geometry_stable_iters` when `geometry_convergence=True` | Done | `config.py` |
| V11 | Warning | `lsb_sma_threshold` provided with non-adaptive integrator | Done | `config.py` |

Deliverables: `docs/configuration-reference.md`, `tests/unit/test_config_validation.py`

## Short-Term Findings (Actionable, from Phase 7)

### P1. Central regularization is effectively inactive in regular `fit_image` flow

- Regularization returns `0.0` if `previous_geom is None` in `isoster/fitting.py:32`.
- Driver calls to `fit_isophote` do not pass `previous_geometry`.
- **Resolved in Phase 8**: driver now passes `previous_geometry=current_iso` for both growth directions.

### P1. Stop code `2` policy

- **Resolved in Phase 18**: `fit_isophote` now emits `stop_code=2` when `maxit` is reached. Driver acceptable set is `{0, 1, 2}`.

### P1. Inward pass can start from a failed first isophote

- **Resolved in Phase 8**: inward startup now gates on acceptable stop codes.

### P2. Linear growth gradient normalization

- **Resolved in I3**: gradient formula now uses `delta_r = step` for linear mode.

### P2. Model reconstruction trusts all `sma > 0` rows

- **Resolved in I4**: NaN intensities and geometry now filtered before interpolation.

### Model reconstruction fixes (Session 6, 2026-02-24)

- **Resolved**: `harmonic_orders=None` auto-detects from isophote `a{n}` keys — no more silent dropping of higher orders.
- **Resolved**: `use_eccentric_anomaly` param on `build_isoster_model()` — auto-detects from isophote dicts for correct EA-mode harmonic evaluation.
- 5 new unit tests added in `tests/unit/test_model.py`.

## Review Notes

- Long-term upgrades and deferred research items: `docs/future.md`
- Stop-code canonical user docs: `docs/user-guide.md`
- Milestone code review (2026-02-22): `docs/review/claude_2026-02-22.md`
