Goal: reorganize the Huang2013 campaign workflow while preserving current extraction/QA behavior and manifest compatibility.

Current status:
- Latest merged commit on `main`: `0180eb3`.
- Retry policy now decays `maxsma` by 5% each failed attempt.
- CoG harmonization now prefers `compute_cog`-derived `cog` for both photutils and isoster.
- Real-case validation exists at `outputs/huang2013_mock1_photutils_retry_decay/`.

Key files:
- `examples/huang2013/run_huang2013_campaign.py`
- `examples/huang2013/run_huang2013_profile_extraction.py`
- `examples/huang2013/run_huang2013_qa_afterburner.py`
- `docs/todo.md`, `docs/spec.md`

First actions:
1. Propose the new campaign module boundaries and manifest contract updates.
2. Update `docs/todo.md` with phased reorg tasks and checkpoints.
3. Implement the first refactor slice (small, testable) and run targeted verification.

Verification commands:
- `uv run pytest tests/unit/test_huang2013_campaign_fault_tolerance.py -q`
- `MPLCONFIGDIR=/tmp/mplconfig uv run python examples/huang2013/run_huang2013_campaign.py --huang-root /Users/mac/work/hsc/huang2013 --output-root /Users/mac/work/hsc/huang2013 --galaxies ESO185-G054 --method both --config-tag baseline --limit 1 --verbose --save-log --update`
