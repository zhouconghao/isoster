Goal: investigate the isoster CoG calculation path for `ESO185-G054_mock3` and identify why `tflux_e` is NaN on rows where SB/intensity remain finite.

Current status:
- Branch: `photutils-2d-model-pass1`.
- Method-specific 2-D model generation is already fixed (photutils uses `build_ellipse_model`; isoster uses `build_isoster_model`).
- QA CoG completeness workaround is in place (`harmonize_method_cog_columns`), and mock3 isoster QA now shows full CoG points.

Key files:
- `examples/huang2013/run_huang2013_real_mock_demo.py`
- `examples/huang2013/run_huang2013_qa_afterburner.py`
- `isoster/fitting.py`, `isoster/driver.py`, `isoster/cog.py`

First actions:
1. Trace `tflux_e` NaN generation in `compute_aperture_photometry` path for mock3 indices.
2. Decide canonical CoG source policy (`cog` vs `tflux_e`) for isoster QA tables.
3. Add a targeted regression test for CoG completeness contract.

Verification command:
- `uv run ruff check examples/huang2013/run_huang2013_real_mock_demo.py examples/huang2013/run_huang2013_qa_afterburner.py`
