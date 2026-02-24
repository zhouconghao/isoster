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
