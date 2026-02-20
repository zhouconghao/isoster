Goal: validate updated Huang2013 initialization/fault-tolerance behavior on real outputs for `ESO185-G054`.

Current status:
- Branch: `feat/huang2013-empty-profile-failure-detection`.
- Implemented: `PA_init = PA_header - 90 deg` (Huang2013 helper), `RE_PX1` fallback `6 px`, extraction success gate `isophote_count >= 3`, QA mismatch/artifact warnings.
- Targeted tests are passing.

Key files:
- `examples/huang2013/run_huang2013_real_mock_demo.py`
- `examples/huang2013/run_huang2013_profile_extraction.py`
- `examples/huang2013/run_huang2013_qa_afterburner.py`
- `tests/unit/test_huang2013_campaign_fault_tolerance.py`

First actions:
1. Run one-galaxy refresh:
   `uv run python examples/huang2013/run_huang2013_campaign.py --huang-root /Users/mac/work/hsc/huang2013 --galaxies ESO185-G054 --mock-ids 1 2 3 4 --method both --config-tag baseline --update --verbose --save-log --max-runtime-seconds 900 --summary-dir outputs/huang2013_campaign_eso185_g054`
2. Review `*_profiles_manifest.json`, `*_qa_manifest.json`, and `*_report.md` under `/Users/mac/work/hsc/huang2013/ESO185-G054`.
3. Confirm PA overlay alignment and failure labeling for mock3/mock4.

Verification command:
- `uv run pytest tests/unit/test_huang2013_campaign_fault_tolerance.py -q`
