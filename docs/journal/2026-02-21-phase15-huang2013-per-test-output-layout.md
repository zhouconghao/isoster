# Phase 15: Huang2013 Per-Test Output Layout

## Objective

Store generated artifacts under per-test folders:

- input FITS stays in `<GALAXY>/`
- generated outputs move to `<GALAXY>/<TEST>/` (for example `<GALAXY>/mock1/`)

## Code Changes

- `examples/huang2013/huang2013_campaign_contract.py`
  - added `build_test_name()`
  - added `build_case_output_dir()`
  - `build_case_prefix()` now uses shared test-name helper
- `examples/huang2013/run_huang2013_campaign.py`
  - default output directory now resolves to `<output-root>/<GALAXY>/mock<ID>/`
  - added per-case output-directory prepare step with create/skip telemetry
- `examples/huang2013/run_huang2013_profile_extraction.py`
  - default output directory now `<huang-root>/<GALAXY>/mock<ID>/`
  - added output-directory create/skip telemetry
- `examples/huang2013/run_huang2013_qa_afterburner.py`
  - default output directory now `<huang-root>/<GALAXY>/mock<ID>/`
  - added output-directory create/skip telemetry
- `examples/huang2013/run_huang2013_real_mock_demo.py`
  - aligned default output directory to per-test folder layout
- docs:
  - `docs/spec.md`
  - `examples/huang2013/README.md`
  - `docs/todo.md`

## Compatibility

- Artifact filenames and manifest field names remain unchanged.
- Only the parent output directory moved from `<GALAXY>/` to `<GALAXY>/mock<ID>/`.

## Verification

- `uv run pytest tests/unit/test_huang2013_campaign_fault_tolerance.py -q`
  - `15 passed in 2.01s`
- Real run:
  - `MPLCONFIGDIR=/tmp/mplconfig uv run python examples/huang2013/run_huang2013_profile_extraction.py --huang-root /Users/mac/work/hsc/huang2013 --galaxy ESO185-G054 --mock-id 1 --method isoster --config-tag baseline --verbose --save-log --update`
  - Output directory created: `/Users/mac/work/hsc/huang2013/ESO185-G054/mock1`
  - Manifest written: `/Users/mac/work/hsc/huang2013/ESO185-G054/mock1/ESO185-G054_mock1_profiles_manifest.json`
- Real QA run:
  - `MPLCONFIGDIR=/tmp/mplconfig uv run python examples/huang2013/run_huang2013_qa_afterburner.py --huang-root /Users/mac/work/hsc/huang2013 --galaxy ESO185-G054 --mock-id 1 --method isoster --config-tag baseline --verbose --skip-comparison`
  - Output directory reuse logged as skip-existing
  - QA manifest written in the same per-test folder
- Real campaign smoke run:
  - `MPLCONFIGDIR=/tmp/mplconfig uv run python examples/huang2013/run_huang2013_campaign.py --huang-root /Users/mac/work/hsc/huang2013 --output-root /Users/mac/work/hsc/huang2013 --galaxies ESO185-G054 --mock-ids 1 --method isoster --config-tag baseline --limit 1 --verbose --save-log --update`
  - `mkdir` stage logged as skip-existing for `/Users/mac/work/hsc/huang2013/ESO185-G054/mock1`
  - Extraction + QA succeeded
