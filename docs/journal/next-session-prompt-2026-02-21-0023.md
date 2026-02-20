Goal: add an iterative recovery loop for Huang2013 campaign extraction when the initial isophote fit fails, for both photutils and isoster.
Current status: core sanity checks and Huang2013 defaults are already merged on `main` (commits `ae1c6ae`, `d9e7a67`), and ESO185-G054 update run is green.
Key files: `examples/huang2013/run_huang2013_profile_extraction.py`, `examples/huang2013/run_huang2013_real_mock_demo.py`, `tests/unit/test_huang2013_campaign_fault_tolerance.py`.
First actions:
1. Read current retry behavior for photutils and current single-pass isoster flow.
2. Draft a concrete retry policy: how to increase `sma0`, how to relax `astep`, max attempts, and final failure criteria.
3. Present the proposed plan and test strategy.
Important: DO NOT implement any code yet; wait for explicit user instruction after presenting the plan.
Suggested verification commands after approval:
- `uv run pytest tests/unit/test_huang2013_campaign_fault_tolerance.py -q`
- `uv run python examples/huang2013/run_huang2013_campaign.py --huang-root /Users/mac/work/hsc/huang2013 --galaxies ESO185-G054 --mock-ids 1 2 3 4 --method both --config-tag baseline --update --verbose --save-log --max-runtime-seconds 900 --summary-dir outputs/huang2013_campaign_eso185_g054`
