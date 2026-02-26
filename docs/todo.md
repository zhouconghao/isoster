# Active Plan and Review

## Completed Summary

Phases 1-23 are complete. Detailed history has been archived to keep this file operational.

## History Archive

- Phase 1-6 progress summary: `docs/journal/2026-02-13-phase1-6-progress-summary.md`
- Phase 7 kickoff plan: `docs/journal/2026-02-13-phase7-doc-audit-and-code-review-plan.md`
- Phase 7-22 detailed checklist and review notes: `docs/todo-archive-phase7-22.md`

## Open Items (Legacy)

| # | Source | Summary | Priority |
|---|--------|---------|----------|
| Phase 22.3 | todo | Decide worktree hygiene policy for tracked `*.pyc` and session prompt files | Low |

---

## Code Review 2026-02-26 (R26)

Full review: `docs/review/claude_2026-02-26.md`

Baseline: `main` at `a23e5ce`, 203 tests passing. Branch: `fix/r26-review-fixes`, 208 tests passing.

### Tracing Table

| Task ID | Severity | Summary | Priority | Status | Blocked by |
|---------|----------|---------|----------|--------|------------|
| R26-01 | Important | Second-gradient formula wrong for linear growth (`fitting.py:642-645`) | P1 | **Done** | — |
| R26-02 | Important | `extract_forced_photometry` hardcodes harmonic keys | P2 | Open | R26-05 |
| R26-03 | Important | Post-hoc harmonic code duplicated 3x in `fit_isophote` | P2 | **Done** | — |
| R26-04 | Important | V5 warning fires for every forced-mode config | P3 | Open | R26-05 |
| R26-05 | Design | Unify forced photometry: replace `forced`/`forced_sma` with template-based API | P2 | Open | — |
| R26-06 | Minor | V9 test assertion always true + `fit_central_pixel` missing debug fields | P3 | **Done** | — |
| R26-07 | Minor | Wrong comment in `extract_forced_photometry` (`fitting.py:92`) | P4 | Open | — |
| R26-08 | Minor | `var_residual` recomputed per coefficient in ISOFIT loop (`fitting.py:984-988`) | P4 | Open | — |
| R26-09 | Minor | `fit_isophote` ~450 lines, hard to maintain | P3 | Partially addressed | R26-03 |
| R26-10 | Minor | `model.py` `has_harmonics` checks only first isophote | P4 | Open | — |
| R26-11 | Minor | Stop codes 4,5 defined in plotting but never produced | P4 | Open | — |
| R26-12 | Minor | Vestigial duck typing in `compute_gradient` | P4 | Open | — |
| R26-13 | Minor | No key validation in template forced mode (`driver.py:318`) | P3 | Open | R26-05 |
| R26-14 | Minor | Default residual type change undocumented | P4 | Open | — |
| R26-T1 | Test | Integration test: `geometry_update_mode='simultaneous'` with `fit_image` | P3 | **Done** | — |
| R26-T2 | Test | Integration test: `isofit_mode='original'` post-hoc harmonics | P3 | **Done** | — |
| R26-T3 | Test | Add test for invalid `convergence_scaling` values | P4 | **Done** | — |
| R26-T4 | Test | Strengthen V9 test assertion (= R26-06) | P3 | **Done** | — |

### Dependency Graph

```
R26-05 (unified forced photometry API)
├── R26-02 (harmonic key inconsistency — subsumed)
├── R26-04 (V5 warning noise — subsumed)
└── R26-13 (template key validation — redesign scope)

R26-03 (extract posthoc harmonic helper)
└── R26-09 (fit_isophote size — partially addressed)
```

### Recommended Execution Order

**Phase A — Bug fix:** COMPLETE
1. ~~R26-01: Fix second-gradient linear growth formula + add test~~

**Phase B — Refactoring:** COMPLETE
2. ~~R26-03: Extract `_compute_posthoc_harmonics` helper~~
3. ~~R26-T1: Integration test for simultaneous geometry with `fit_image`~~
4. ~~R26-T2: Integration test for `isofit_mode='original'`~~
5. ~~R26-06/R26-T4: Fix V9 test assertion + `fit_central_pixel` debug fields~~
6. ~~R26-T3: Convergence scaling validation test~~

**Phase C — API redesign (feature branch, needs design discussion):**
7. R26-05: Design unified template-based forced photometry API
   - Deprecate `forced`/`forced_sma` config fields
   - Rename `template_isophotes` to `template`, accept str/dict/list
   - Resolve R26-02, R26-04, R26-13 as part of the redesign

**Phase D — Polish (can interleave):**
8. R26-07 through R26-14, R26-08, R26-10, R26-11, R26-12

## Review Notes

- Long-term upgrades and deferred research items: `docs/future.md`
- Stop-code canonical user docs: `docs/user-guide.md`
- Previous code review (2026-02-22): `docs/review/claude_2026-02-22.md`
- Current code review (2026-02-26): `docs/review/claude_2026-02-26.md`
