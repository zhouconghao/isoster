# E6: Extract shared functions from huang2013 demo script — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract shared functions from the 2,682-line `run_huang2013_real_mock_demo.py` into a dedicated `huang2013_shared.py` module.

**Architecture:** Pure mechanical extraction — move lines 1-83 (imports/constants) and 211-2361 (all shared functions) into a new shared module. The demo script keeps only `parse_arguments()` (lines 85-210) and `main()` (lines 2363-2682), importing from the shared module. Two downstream scripts update their import source.

**Tech Stack:** Python, no new dependencies.

---

### Task 1: Create branch

**Step 1: Create and switch to feature branch**

Run: `git checkout -b refactor/e6-huang2013-shared`

---

### Task 2: Create `huang2013_shared.py`

**Files:**
- Create: `examples/huang2013/huang2013_shared.py`

**Step 1: Extract shared code**

Create `examples/huang2013/huang2013_shared.py` containing:
- The module docstring (new, describing purpose)
- All imports from `run_huang2013_real_mock_demo.py` lines 12-43
- All constants and style dicts from lines 46-82
- All functions from lines 211-2361 (everything except `parse_arguments` and `main`), preserving their exact order, signatures, and docstrings

The functions to extract (in order by line number):
1. `sanitize_label` (L211)
2. `build_input_path` (L224)
3. `read_mock_image` (L231)
4. `get_header_value` (L241)
5. `compute_kpc_per_arcsec` (L248)
6. `normalize_pa_degrees` (L254)
7. `style_for_stop_code` (L270)
8. `compute_sha256` (L281)
9. `infer_initial_geometry` (L290)
10. `run_with_runtime_profile` (L355)
11. `convert_photutils_isolist` (L383)
12. `run_photutils_fit` (L441)
13. `run_isoster_fit` (L469)
14. `ensure_numeric_column` (L478)
15. `is_error_column_name` (L486)
16. `summarize_negative_error_values` (L496)
17. `compute_true_curve_of_growth` (L526)
18. `ensure_isoster_style_cog_columns` (L560)
19. `harmonize_method_cog_columns` (L600)
20. `prepare_profile_table` (L638)
21. `table_to_isophote_dicts` (L721)
22. `extract_photutils_model_columns` (L767)
23. `build_photutils_model_image` (L819)
24. `extract_isoster_model_rows` (L835)
25. `build_model_image` (L926)
26. `compute_fractional_residual_percent` (L979)
27. `draw_isophote_overlays` (L989)
28. `robust_limits` (L1029)
29. `set_axis_limits_from_finite_values` (L1044)
30. `set_x_limits_with_right_margin` (L1091)
31. `configure_qa_plot_style` (L1110)
32. `latex_safe_text` (L1131)
33. `derive_arcsinh_parameters` (L1138)
34. `make_arcsinh_display_from_parameters` (L1170)
35. `make_arcsinh_display` (L1184)
36. `plot_profile_by_stop_code` (L1207)
37. `build_method_qa_figure` (L1266)
38. `interpolate_column` (L1632)
39. `build_comparison_qa_figure` (L1657)
40. `summarize_table` (L2228)
41. `dump_json` (L2260)
42. `write_markdown_report` (L2265)
43. `collect_software_versions` (L2352)

**Step 2: Verify the new module imports cleanly**

Run: `cd /Users/mac/Dropbox/work/project/otters/isoster/examples/huang2013 && uv run python -c "import huang2013_shared; print(f'Loaded: {len(dir(huang2013_shared))} names')"`

Expected: No import errors, prints count of exported names.

---

### Task 3: Rewrite `run_huang2013_real_mock_demo.py`

**Files:**
- Modify: `examples/huang2013/run_huang2013_real_mock_demo.py`

**Step 1: Replace the demo script body**

Replace the entire file with:
- Minimal imports: `__future__`, `argparse`, `json`, `numpy`, `Path`, `Any`, `Table`
- One wildcard-free import block from `huang2013_shared` covering all needed symbols
- Import from `huang2013_campaign_contract` (already there)
- `parse_arguments()` (lines 85-210, unchanged)
- `main()` (lines 2363-2682, unchanged)
- `if __name__ == "__main__": main()`

The script should drop from ~2,682 lines to ~530 lines.

**Step 2: Verify the demo script imports cleanly**

Run: `cd /Users/mac/Dropbox/work/project/otters/isoster/examples/huang2013 && uv run python -c "from run_huang2013_real_mock_demo import parse_arguments, main; print('OK')"`

Expected: `OK`

---

### Task 4: Update imports in `run_huang2013_profile_extraction.py`

**Files:**
- Modify: `examples/huang2013/run_huang2013_profile_extraction.py` (lines 25-43)

**Step 1: Change import source**

Replace `from run_huang2013_real_mock_demo import (` with `from huang2013_shared import (`

The imported symbol list (lines 26-42) stays exactly the same.

**Step 2: Verify**

Run: `cd /Users/mac/Dropbox/work/project/otters/isoster/examples/huang2013 && uv run python -c "from run_huang2013_profile_extraction import main; print('OK')"`

Expected: `OK`

---

### Task 5: Update imports in `run_huang2013_qa_afterburner.py`

**Files:**
- Modify: `examples/huang2013/run_huang2013_qa_afterburner.py` (lines 30-47)

**Step 1: Change import source**

Replace `from run_huang2013_real_mock_demo import (` with `from huang2013_shared import (`

The imported symbol list (lines 31-46) stays exactly the same.

**Step 2: Verify**

Run: `cd /Users/mac/Dropbox/work/project/otters/isoster/examples/huang2013 && uv run python -c "from run_huang2013_qa_afterburner import main; print('OK')"`

Expected: `OK`

---

### Task 6: Run full test suite

**Step 1: Run tests**

Run: `uv run pytest tests/unit/ tests/integration/ tests/validation/ -v`

Expected: 152 tests pass.

**Step 2: Verify line counts**

Run: `wc -l examples/huang2013/huang2013_shared.py examples/huang2013/run_huang2013_real_mock_demo.py`

Expected: shared ~2,200 lines, demo ~530 lines.

---

### Task 7: Commit

**Step 1: Commit the refactoring**

```bash
git add examples/huang2013/huang2013_shared.py examples/huang2013/run_huang2013_real_mock_demo.py examples/huang2013/run_huang2013_profile_extraction.py examples/huang2013/run_huang2013_qa_afterburner.py
git commit -m "refactor(E6): extract shared functions into huang2013_shared.py

Move 43 shared functions and constants from run_huang2013_real_mock_demo.py
into huang2013_shared.py. Update imports in profile_extraction and
qa_afterburner scripts. Demo script reduced from 2682 to ~530 lines.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
