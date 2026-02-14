# 2026-02-13 Phase 7 Plan: Documentation Integrity + Code Review

## Goal

Execute a full documentation-and-code review pass focused on:

1. Algorithm documentation fidelity (`docs/spec.md`, `docs/algorithm.md`)
2. Stop-code correctness and doc consolidation decision (`docs/stop-codes.md`)
3. `docs/todo.md` restructuring into active plan + short summary with archived progress references
4. legacy optimization-notes migration to `docs/future.md`
5. A careful bug/risk review with near-term and long-term outputs split across `docs/todo.md` and `docs/future.md`

## Constraints and Ground Rules

- Keep markdown names in lowercase kebab-case.
- Keep edits minimal but complete; avoid speculative claims.
- For every technical statement in docs, verify against current code before editing.
- Do not guess metrics or performance claims; keep only measured/defensible statements.
- Preserve finished historical details by moving them to journal/history references, not deleting project memory.

## Scope of Review

### Code modules to verify against docs

- `isoster/driver.py`
- `isoster/fitting.py`
- `isoster/sampling.py`
- `isoster/config.py`
- `isoster/model.py`
- `isoster/cog.py`
- `isoster/utils.py`
- `isoster/cli.py`

### Documentation files to update

- `docs/spec.md`
- `docs/algorithm.md`
- `docs/stop-codes.md`
- `docs/user-guide.md` (if stop-code merge target)
- `docs/todo.md`
- legacy optimization-notes page -> `docs/future.md`
- `docs/index.md`
- `docs/index.md`
- `mkdocs.yml`
- `README.md` and any other references to renamed docs

## Execution Plan

## Step 1: Establish truth baseline

- Extract current algorithmic behavior and stop-code generation paths from code.
- Create a compact “docs claim vs code reality” matrix for:
  - fitting loop and convergence
  - sampling modes (PA vs eccentric anomaly)
  - geometry update policy
  - CoG/photometry semantics
  - template/forced modes
  - stop-code triggers and propagation

Deliverable:

- Internal notes in working scratch (no final file required), used as source-of-truth for doc edits.

## Step 2: Audit and fix `docs/spec.md` + `docs/algorithm.md`

- Remove stale claims and replace with verified behavior.
- Ensure parameter names and defaults match `isoster/config.py`.
- Ensure algorithm equations/mapping to implementation are accurate.
- Add explicit caveats where implementation differs from idealized math.

Deliverable:

- Updated `docs/spec.md`
- Updated `docs/algorithm.md`

Acceptance checks:

- No claim remains that cannot be traced to current code.
- Terms and symbols are consistent between spec and algorithm docs.

## Step 3: Audit `docs/stop-codes.md` and merge decision

- Verify each stop code against actual emitting logic in code paths.
- Fix incorrect triggers, names, or recommendations.
- Decide consolidation target based on maintainability:
  - merge into `docs/algorithm.md` if tightly implementation-specific
  - merge into `docs/user-guide.md` if primarily user-facing operations
- Execute merge and de-duplicate text.

Deliverable:

- Corrected stop-code content in final destination file.
- If merged, leave `docs/stop-codes.md` as a short redirect note or remove and update links.

Acceptance checks:

- One canonical stop-code reference location.
- All repo links point to canonical location.

## Step 4: Restructure `docs/todo.md` and archive completed progress

- Create a compact progress summary file for completed historical phases in `docs/journal/`.
- Add a 3-5 sentence summary at top of `docs/todo.md` describing what is completed.
- Reference the new progress summary file at top of `docs/todo.md`.
- Keep only active/pending tasks and near-term findings in `docs/todo.md`.

Deliverable:

- New progress summary file under `docs/journal/` (Phase 1-6 archive summary).
- Updated top section of `docs/todo.md` with concise status + link.

Acceptance checks:

- `docs/todo.md` is clearly actionable, not a full historical ledger.
- Historical detail remains discoverable via referenced file.

## Step 5: Migrate legacy optimization notes to `docs/future.md`

- Rename file to `docs/future.md`.
- Remove already-implemented items (for example existing JIT integration details).
- Remove not-relevant or disproven ideas.
- Keep only realistic, high-value future opportunities.
- Split “future” content into:
  - long-term engineering upgrades
  - long-term algorithm/performance research directions

Deliverable:

- `docs/future.md` (cleaned, scoped, current)
- legacy optimization-notes content migrated

Acceptance checks:

- No implemented work remains in `future.md` as future work.
- No dead links to legacy optimization-notes page remain.

## Step 6: Conduct careful code review and capture findings

Review strategy:

- Focus on bugs, correctness risks, robustness gaps, and missing validations.
- Prioritize by severity and user impact.
- Include concrete file references and, where possible, minimal fix direction.

Output split:

- Short-term actionable issues -> add checklist entries to `docs/todo.md`
- Long-term upgrades/optimizations -> add to `docs/future.md`

Deliverable:

- Updated `docs/todo.md` with prioritized bug/risk/easy-win list.
- Updated `docs/future.md` with long-term roadmap entries.

Acceptance checks:

- Findings are specific, reproducible, and non-overlapping.
- Short-term vs long-term boundary is clear.

## Step 7: Link integrity and minimal verification

- Update all references in docs and navigation files:
  - `mkdocs.yml`
  - `docs/index.md`
  - `docs/index.md`
  - `README.md`
  - any remaining internal links
- Run quick consistency checks:
  - search for stale path references
  - optional markdown/link sanity scan if available

Deliverable:

- clean link graph for renamed/merged docs

Acceptance checks:

- no references to removed/renamed docs remain
- docs navigation reflects canonical files

## Expected Final Artifacts

- Updated: `docs/spec.md`
- Updated: `docs/algorithm.md`
- Updated or merged: stop-code documentation
- New: `docs/future.md`
- Updated: `docs/todo.md` (active tasks + concise summary header)
- New: `docs/journal/<date>-phase1-6-progress-summary.md`
- Updated links: `mkdocs.yml`, `docs/index.md`, `docs/index.md`, `README.md` as needed

## Risks and Mitigations

- Risk: accidental loss of historical context while cleaning `docs/todo.md`.
  - Mitigation: archive first in journal summary before simplifying `docs/todo.md`.
- Risk: doc claims drift from code during edits.
  - Mitigation: enforce claim-by-claim verification against source modules.
- Risk: stop-code behavior may be subtle across driver/fitting interaction.
  - Mitigation: verify both origin and propagation paths before rewriting docs.
