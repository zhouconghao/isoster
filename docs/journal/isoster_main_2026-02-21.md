---
date: 2026-02-21
repo: isoster
branch: main
tags:
  - journal
  - development
  - huang2013
---

## Progress

- Committed `d9e7a67`: added post-run negative-error sanity checks in `isoster.fit_image` and updated driver unit tests.
- Committed `d51bd8b`: introduced unified Huang2013 retry policy for photutils/isoster and updated related docs/tests.
- Committed `dbcefbd` and `7abe175`: added journal handover and next-session prompt notes.
- Committed `65cd268`: improved Huang2013 QA readability and stop-code visual contrast.
- Committed `1bf66f7`: harmonized Huang2013 CoG sources and added completeness regression coverage.
- Committed `7c29d4b`: emitted `stop_code=2` on max-iteration path and aligned parity docs/tests.
- Committed `8628412`: fixed integration stop-code assumptions and stabilized test imports.
- Committed `0180eb3`: unified Huang2013 CoG sourcing and added 5% `maxsma` retry decay.
- Committed `143897e`: reorganized Huang2013 workflow with shared contract helpers, per-test output layout (`<GALAXY>/mock<ID>/`), and cleanup tooling.

## Lessons Learned

- Keeping extraction/QA/campaign filename contracts stable while changing parent output layout reduces migration risk.
- Retry and CoG rules should be shared across methods and validated with dedicated regression tests.
- Long-run workflows benefit from explicit stage telemetry and checkpoint journals to preserve context between sessions.
- Cleanup tooling must support both legacy flat outputs and new nested output layouts during transitions.

## Key Issues

- Decide whether to push `main` (`143897e`) to `origin/main`; local branch is ahead by one commit.
- Consider adding campaign dry-run visibility for planned output directories (current dry-run skips mkdir stage logs).
- Review and optionally clean local untracked handover prompt files if they are not needed for the next session.
