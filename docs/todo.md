# Active Plan and Review

## Completed Summary

Phases 1-6 are complete and archived, including docs reorganization, uv workflow adoption, test/benchmark hardening, numba diagnostics, and Huang2013 workflow stabilization. The active focus is now Phase 7 documentation integrity and evidence-based code review follow-through. Detailed completed history has been moved out of this file to keep this checklist operational. Future updates should keep this file short, actionable, and tied to verifiable artifacts.

## History Archive

- Detailed Phase 1-6 progress summary: `docs/journal/2026-02-13-phase1-6-progress-summary.md`
- Phase 7 kickoff plan note: `docs/journal/2026-02-13-phase7-doc-audit-and-code-review-plan.md`

## Phase 7 Checklist

| Item | Status | Notes |
|---|---|---|
| 1. Build claim-vs-code matrix for algorithm/spec/stop-code docs | [x] | verified against `driver/fitting/sampling/config/model/cog` |
| 2. Update `docs/spec.md` and `docs/algorithm.md` for correctness | [x] | removed stale claims and aligned implementation behavior |
| 3. Consolidate stop-code docs into canonical location | [x] | canonical location set to `docs/user-guide.md` |
| 4. Refactor `docs/todo.md` to active-focused and archive completed history | [x] | summary header + archive link added |
| 5. Maintain `docs/future.md` as long-term roadmap | [x] | trimmed to realistic long-term items and removed implemented work |
| 6. Run careful code review and record findings | [x] | short-term findings below; long-term in `docs/future.md` |
| 7. Run final link/reference sweep | [x] | active docs/navigation updated; no stale markdown links remain |

## Short-Term Findings (Actionable)

### P1. Central regularization is effectively inactive in regular `fit_image` flow

Evidence:

- Regularization returns `0.0` if `previous_geom is None` in `isoster/fitting.py:32`.
- Driver calls to `fit_isophote` do not pass `previous_geometry` in `isoster/driver.py:126`, `isoster/driver.py:147`, `isoster/driver.py:175`.

Action:

- Pass prior accepted geometry into `fit_isophote(..., previous_geometry=...)` in outward/inward loops and validate with a targeted regression test.

### P1. Stop code `2` is documented but not emitted by current core fitter

Evidence:

- `fit_isophote` emits only `0`, `1`, `3`, `-1` in `isoster/fitting.py:651`, `isoster/fitting.py:664`, `isoster/fitting.py:705`, `isoster/fitting.py:711`, `isoster/fitting.py:757`.
- `extract_forced_photometry` emits `0` or `3` in `isoster/fitting.py:101`, `isoster/fitting.py:125`.
- `fit_central_pixel` emits `0` or `-1` in `isoster/driver.py:35`.

Action:

- Decide whether to reintroduce explicit code `2` emission in `isoster` or remove it from compatibility descriptions in code/docs.

### P1. Inward pass can start from a failed first isophote

Evidence:

- Inward loop starts when `minsma < sma0` without checking first-isophote quality in `isoster/driver.py:157`.
- It seeds with `current_iso = first_iso` in `isoster/driver.py:158`.

Action:

- Gate inward growth on acceptable initial stop codes, mirroring outward pass quality gating.

### P2. `linear_growth=True` gradient normalization likely uses inconsistent radius delta

Evidence:

- Linear mode sets `gradient_sma = sma + step` in `isoster/fitting.py:464`.
- Gradient still divides by `sma * step` in `isoster/fitting.py:480` and `isoster/fitting.py:485`.

Action:

- Benchmark and verify intended finite-difference definition for linear mode, then align denominator and tests.

### P2. Model reconstruction currently trusts all `sma > 0` rows

Evidence:

- `build_isoster_model` keeps rows by `iso['sma'] > 0` only in `isoster/model.py:80`.
- Interpolation then consumes raw `intens` values in `isoster/model.py:104` and `isoster/model.py:116`.

Action:

- Add optional quality filter (`stop_code` and finite-intensity checks) before interpolation.

## Review Notes

- Long-term upgrades and deferred research items are tracked in `docs/future.md`.
- Stop-code canonical user docs now live in `docs/user-guide.md`.

## Phase 8 Kickoff (Code Quality Improvement)

- Closeout summary and clean-window starter prompt: `docs/journal/2026-02-14-phase7-closeout-and-phase8-kickoff.md`
