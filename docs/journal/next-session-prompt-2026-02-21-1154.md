Goal: finalize and validate stop-code parity work for `isoster` max-iteration fallback on `ESO185-G054_mock3`.

Current status:
- Branch: `stopcode2-maxit-label`.
- `fit_isophote` now emits `stop_code=2` when `maxit` is reached without convergence and fills `tflux_e` on that path.
- Matching mock3 outputs exist for both methods:
  - `outputs/huang2013_mock3_isoster_stopcode2/`
  - `outputs/huang2013_mock3_photutils_stopcode2/`
- Side-by-side report exists: `outputs/huang2013_mock3_stopcode2_compare/ESO185-G054_mock3_stopcode2_side_by_side_report.md`.

First actions:
1. Run broader regression: `uv run pytest tests/ -q`.
2. Review stop-code assumptions in integration tests/docs for any remaining drift.
3. If clean, prepare commit with stop-code policy + test/doc updates.

Verification commands:
- `uv run pytest tests/unit/test_driver.py tests/unit/test_public_api.py tests/unit/test_fitting.py tests/unit/test_huang2013_campaign_fault_tolerance.py -q`
- `cat outputs/huang2013_mock3_stopcode2_compare/ESO185-G054_mock3_stopcode2_side_by_side_report.md`
