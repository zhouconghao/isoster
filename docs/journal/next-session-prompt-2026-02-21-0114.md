Read docs/journal/2026-02-21_handover_2.md for full context.

Immediate next step:
Run one full Huang2013 campaign with the updated scripts to regenerate `outputs/huang2013_campaign_full/huang2013_campaign_summary.json` and `.md`, then confirm failed/timeout case lists now include both timeout and manifest-level failed extraction cases.

Warnings/constraints:
- Keep current timeout semantics unchanged (subprocess timeout does not continue retries in-process).
- Keep QA artifact-missing warning behavior unchanged for intentionally skipped methods (#3 deferred).
- Core `isoster/model.py` sanitization (#4) is deferred; only Huang2013 helper path is hardened now.
- Worktree still contains pycache and untracked journal artifacts; do not delete/revert user-owned files without explicit instruction.
