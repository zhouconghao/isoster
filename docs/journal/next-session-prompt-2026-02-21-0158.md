Goal: continue a few more small QA-figure improvements for Huang2013 ESO185-G054.

Current status:
- QA plotting edits are already in `examples/huang2013/run_huang2013_real_mock_demo.py`.
- QA afterburner runtime parsing null-guard is in `examples/huang2013/run_huang2013_qa_afterburner.py`.
- QA figures for mock1-4 were regenerated via afterburner only (no extraction rerun).

Key files:
- `examples/huang2013/run_huang2013_real_mock_demo.py`
- `examples/huang2013/run_huang2013_qa_afterburner.py`
- `docs/todo.md` (Phase 14)

First actions:
1. Visually inspect regenerated QA PNGs in `/Users/mac/work/hsc/huang2013/ESO185-G054/` and list exact remaining tweaks.
2. Implement minimal plotting edits.
3. Re-run `run_huang2013_qa_afterburner.py` for affected mock IDs only.

Verification command:
- `uv run ruff check examples/huang2013/run_huang2013_real_mock_demo.py examples/huang2013/run_huang2013_qa_afterburner.py`
