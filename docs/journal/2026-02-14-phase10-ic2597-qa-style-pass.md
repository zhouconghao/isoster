# Phase 10 IC2597 QA Figure Style Pass (2026-02-14)

## Scope

Refined the Huang2013 IC2597 QA plotting style for the baseline run, focusing on typography, panel alignment, figure geometry, image/model rendering, and monochrome 1-D profile styling.

## What Changed

- Edited `examples/huang2013/run_huang2013_real_mock_demo.py`:
  - Added shared plotting helpers:
    - `configure_qa_plot_style()`
    - `latex_safe_text()`
    - `make_arcsinh_display()`
  - Updated `build_method_qa_figure(...)`:
    - left panel converted to 3x2 layout (`panel + colorbar`) for strict alignment
    - image/model switched to viridis + arcsinh scaling for low-SB visibility
    - right-side 1-D profile plotting switched to monochrome stop-code markers
    - figure width/spacing adjusted to avoid title/panel overlap
  - Updated `build_comparison_qa_figure(...)` with matching layout/scaling and black-style 1-D markers.

## Verification

- Lint:
  - `uv run ruff check examples/huang2013/run_huang2013_real_mock_demo.py`
  - Result: passed.
- QA generation:
  - `uv run python examples/huang2013/run_huang2013_qa_afterburner.py --galaxy IC2597 --mock-id 1 --method both --config-tag baseline --output-dir outputs/huang2013_ic2597_qa_style --photutils-profile-fits /Users/mac/work/hsc/huang2013/IC2597/IC2597_mock1_photutils_baseline_profile.fits --isoster-profile-fits /Users/mac/work/hsc/huang2013/IC2597/IC2597_mock1_isoster_baseline_profile.fits --photutils-run-json /Users/mac/work/hsc/huang2013/IC2597/IC2597_mock1_photutils_baseline_run.json --isoster-run-json /Users/mac/work/hsc/huang2013/IC2597/IC2597_mock1_isoster_baseline_run.json`
  - Result: completed and regenerated QA outputs.

## Key Artifacts

- `outputs/huang2013_ic2597_qa_style/IC2597_mock1_photutils_baseline_qa.png`
- `outputs/huang2013_ic2597_qa_style/IC2597_mock1_isoster_baseline_qa.png`
- `outputs/huang2013_ic2597_qa_style/IC2597_mock1_compare_baseline_qa.png`
- `outputs/huang2013_ic2597_qa_style/IC2597_mock1_qa_manifest.json`

## Resume

- Re-run same QA build command above after any further plotting edits.
- Main code touchpoint remains `examples/huang2013/run_huang2013_real_mock_demo.py`.
