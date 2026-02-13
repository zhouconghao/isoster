# Phase 1 Plan: Documentation and Structure Reorganization

## Program Status

Architecture review program (Phases 1-3) is completed as of 2026-02-11.

## Scope
This plan covers the first implementation phase:
- documentation architecture cleanup
- naming consistency for markdown files
- folder-level README tables of contents for `tests`, `benchmarks`, and `examples`
- alignment updates in `README.md` and `CLAUDE.md`

## Checklist
| Item | Status | Notes |
|---|---|---|
| 1. Preserve existing planning memory and legacy docs in archive | [x] | `docs/archive/legacy_todo_photutils_parity.md` created |
| 2. Define canonical docs structure and naming convention | [x] | Convention set to lowercase kebab-case markdown names |
| 3. Reorganize `docs/` with a new `docs/README.md` and stable sections | [x] | Stable docs promoted; legacy material moved to `docs/archive/` |
| 4. Add `README.md` files to `tests/`, `benchmarks/`, and `examples/` | [x] | Added folder-level scope, reproducible commands, and output policy |
| 5. Update top-level `README.md` links and testing guidance | [x] | Updated links and added run entry points |
| 6. Update `CLAUDE.md` with architecture/file naming rules and memory-preservation workflow | [x] | Added explicit context and memory-preservation section |
| 7. Fix `mkdocs.yml` navigation to existing files | [x] | Removed missing API pages and linked current docs |
| 8. Verify by checking links/paths and running a fast sanity check | [x] | `pytest --collect-only -q` passed (102 collected, 4 deselected) |
| 9. Write phase review summary in this file | [x] | Added below |

## Review

### Completed

- Preserved legacy plan memory in `docs/archive/legacy_todo_photutils_parity.md`.
- Reorganized docs into stable root docs and historical archive:
  - moved `docs/development/` -> `docs/archive/development/`
  - moved `docs/review/` -> `docs/archive/review/`
- Normalized markdown naming to lowercase kebab-case for active docs and archive files.
- Added `docs/README.md` as docs table of contents and maintenance guide.
- Added `docs/spec.md` to serve as the architecture specification baseline.
- Added `docs/lessons.md` for persistent development lessons.
- Added folder-level READMEs for `tests/`, `benchmarks/`, and `examples/`.
- Updated `README.md` links and test/benchmark entry-point commands.
- Updated `CLAUDE.md` with:
  - docs naming convention
  - output policy
  - context and memory-preservation workflow
  - corrected test command examples
- Updated `mkdocs.yml` to remove missing API pages and point to existing docs.

### Remaining for Phase 2

- Validate MkDocs rendering in an environment with `mkdocs` installed (CLI unavailable in current shell).

## Phase 2 Plan: Output Path Standardization

### Scope
- consolidate generated artifacts under `outputs/`
- remove hardcoded mixed output paths from tests/benchmarks/examples
- keep reproducibility via explicit overrides

### Checklist
| Item | Status | Notes |
|---|---|---|
| 1. Add shared output-path utility | [x] | Added `isoster/output_paths.py` with `resolve_output_directory()` |
| 2. Migrate integration/validation/real-data test artifacts to standardized output folders | [x] | Updated `tests/integration/*`, `tests/validation/*`, `tests/real_data/*` |
| 3. Migrate benchmark outputs to standardized folders | [x] | Updated `bench_vs_photutils.py` and `bench_numba_speedup.py` |
| 4. Migrate example outputs to standardized folders | [x] | Updated `examples/basic_usage.py` and Huang2013 default output path |
| 5. Update folder README guidance for reproducible output roots | [x] | Added `ISOSTER_OUTPUT_ROOT` note in tests/benchmarks/examples READMEs |
| 6. Verify with collection and targeted execution | [x] | `pytest --collect-only -q` + targeted tests and script checks passed |
| 7. Record phase review and residual risks | [x] | Added below |

### Review

- Added a reusable path helper in `isoster/output_paths.py`.
- Replaced `tests/qa_outputs` and `outputs/figures` writers in active code paths with standardized `outputs/<category>/<run>` layout.
- Removed legacy `tests/qa_outputs/` directory to prevent further artifact drift.
- Updated benchmark scripts to support explicit `--output` override and standardized defaults.
- Fixed a benchmark import-path issue in `bench_vs_photutils.py` so CLI invocation now works.

Residual risks:
- `qa/` reference materials remain as historical content and still describe legacy generation workflows.
- `mkdocs` CLI is unavailable in current shell, so docs rendering was not validated here.

## Phase 3 Plan: Adopt uv Environment Management

### Scope
- adopt `uv` for dependency locking and environment sync
- align dependency metadata with actual runtime/test/docs usage
- install project dependencies with docs tooling (`mkdocs`) locally

### Checklist
| Item | Status | Notes |
|---|---|---|
| 1. Audit imports vs declared dependencies | [x] | Audited runtime/test/benchmark/docs imports and dependency gaps |
| 2. Update `pyproject.toml` dependency metadata for uv workflows | [x] | Added `pydantic` runtime dependency and `dev`/`docs` extras with Python markers |
| 3. Update docs (`README.md`, `CLAUDE.md`) with uv-first commands | [x] | Updated install/test/benchmark/docs command examples to uv-first |
| 4. Generate/update lockfile with `uv lock` | [x] | Generated `uv.lock` |
| 5. Install dependencies via `uv sync` including docs tooling | [x] | Installed core + dev + docs extras (including mkdocs) into `.venv` |
| 6. Verify with uv-based commands | [x] | `uv run pytest --collect-only -q` passed; `mkdocs --version` confirmed from `.venv` |
| 7. Record review and lessons | [x] | Updated this file, `docs/lessons.md`, and journal entry |

### Review

- `uv` is practical for this repo and now configured end-to-end:
  - dependency metadata in `pyproject.toml`
  - lockfile in `uv.lock`
  - synced environment in `.venv`
- Runtime dependency gap fixed: `pydantic` is now declared in core dependencies.
- Added optional extras:
  - `dev`: pytest, photutils, numba, ruff, pre-commit
  - `docs`: mkdocs, mkdocs-material, pymdown-extensions
- Added Python markers for some dev tools to keep resolution compatible with `requires-python >=3.8`.
- Updated project docs to uv-first command paths.

Notes:
- In this environment, `uv` commands required elevated execution due sandbox/system-configuration constraints.

## Phase 4 Plan: Test and Benchmark Improvement Program

### Scope
- harden tests so regressions cannot pass silently
- expand API and CLI coverage
- standardize quantitative validation criteria
- improve benchmark/profiling diagnostics and artifact outputs
- prepare future libprofit-based mock generation integration

### Checklist
| Item | Status | Notes |
|---|---|---|
| 1. Draft detailed test/benchmark improvement roadmap in docs | [x] | Added `docs/test-benchmark-improvement-plan.md` |
| 2. Persist new user-defined constraints in project memory | [x] | Updated `CLAUDE.md` testing directives section |
| 3. Baseline-measure current tests/benchmarks and lock quantitative thresholds from measurements | [x] | Locked in `benchmarks/baselines/phase4_profile_thresholds_2026-02-11.json` |
| 4. Remove false-pass patterns (`if valid.sum() > 0`) and enforce minimum valid-point assertions | [x] | Hardened in `tests/integration/test_sersic_accuracy.py` and `tests/integration/test_numba_validation.py` |
| 5. Add missing direct tests for public API and CLI surfaces | [x] | Added `tests/unit/test_public_api.py` and `tests/integration/test_cli.py` |
| 6. Normalize basic real-data test name to `m51_test` and ensure canonical M51 path usage | [x] | Renamed to `TestM51::test_m51_test` in `tests/real_data/test_m51.py` |
| 7. Standardize all benchmark/profiling outputs under `outputs/` with JSON + figure + metadata artifacts | [x] | Updated performance + profiling scripts; `.prof` persistence added |
| 8. Add optional future adapter workflow for libprofit `mockgal.py` mock generation | [x] | Added `benchmarks/baselines/mockgal_adapter.py` with dry-run + metadata output |
| 9. Record phase review and closeout summary | [x] | Added below and journal closeout |
| 10. Save compact context-handoff snapshot for next conversation window | [x] | Added `docs/journal/2026-02-11-phase4-step2-closeout.md` |

### Review (Implementation Progress: Step 2 Closeout)

- Program requirements were captured as an executable roadmap in `docs/test-benchmark-improvement-plan.md`.
- The roadmap defines workstreams, execution phases, artifact contracts, and quantitative metric formulas.
- Threshold policy remains measurement-first and is now locked from measured baseline data.
- Added baseline collection script:
  - `benchmarks/baselines/collect_phase4_profile_baseline.py`
  - output artifact: `outputs/tests_integration/baseline_metrics/phase4_profile_baseline_metrics.json`
- Added threshold locking script and locked threshold file:
  - `benchmarks/baselines/lock_phase4_thresholds.py`
  - `benchmarks/baselines/phase4_profile_thresholds_2026-02-11.json`
- Baseline script run summary:
  - `sersic_n4_noiseless`: valid=19, median|ΔI|=0.000096, max|ΔI|=0.002098
  - `sersic_n1_high_eps_noise`: valid=15, median|ΔI|=0.003759, max|ΔI|=0.114202
  - `sersic_n4_extreme_eps_noise`: valid=15, median|ΔI|=0.002971, max|ΔI|=0.009876
- Replaced false-pass behavior by enforcing explicit valid-point assertions before metric computation in:
  - `tests/integration/test_sersic_accuracy.py`
  - `tests/integration/test_numba_validation.py`
- Added missing API/CLI coverage:
  - `fit_isophote`, `isophote_results_to_astropy_tables`, `plot_qa_summary`, `build_ellipse_model` regression in `tests/unit/test_public_api.py`
  - CLI smoke run in `tests/integration/test_cli.py`
- Normalized M51 canonical basic test to `test_m51_test` in `tests/real_data/test_m51.py`.
- Standardized benchmark/profiling artifacts and metadata:
  - `benchmarks/performance/bench_efficiency.py` now writes JSON+CSV+summary plot under `outputs/benchmarks_performance/bench_efficiency`
  - `benchmarks/performance/bench_vs_photutils.py` now writes metadata in JSON + CSV and default failing/borderline diagnostics
  - `benchmarks/performance/bench_numba_speedup.py` now writes metadata JSON + CSV + speedup plot
  - `benchmarks/profiling/profile_hotpaths.py` and `benchmarks/profiling/profile_isophote.py` now persist `.prof` and parsed summaries under `outputs/benchmarks_profiling`
- Added optional mockgal adapter workflow:
  - `benchmarks/baselines/mockgal_adapter.py`
  - dry-run validation output: `outputs/benchmarks_performance/mockgal_adapter/mockgal_adapter_run.json`
- Verification run results:
  - `uv run pytest tests/unit/test_public_api.py tests/integration/test_cli.py -q` passed (`5 passed`)
  - `uv run pytest tests/integration/test_sersic_accuracy.py tests/integration/test_numba_validation.py -q` passed (`27 passed`)
  - `uv run pytest tests/real_data/test_m51.py --collect-only -q -m real_data` collected `test_m51_test`
  - `uv run python benchmarks/performance/bench_efficiency.py --quick --n-runs 1` generated JSON/CSV/plots
  - `uv run python benchmarks/performance/bench_vs_photutils.py --quick` generated JSON/CSV with all quick cases passing
  - `uv run python benchmarks/performance/bench_numba_speedup.py` generated JSON/CSV/plot
  - `uv run python benchmarks/profiling/profile_hotpaths.py --multi 1 --top-n 10` generated `.prof` + summaries
  - `uv run python benchmarks/profiling/profile_isophote.py --repetitions 1 --top-n 10` generated `.prof` + summaries
  - `uv run python benchmarks/baselines/mockgal_adapter.py --dry-run` generated adapter metadata output

### Residual Risks

- `bench_numba_speedup.py` currently shows modest speedup (~1.24x in this environment), so expectations should remain environment-dependent and tracked over time.
- `mockgal_adapter.py` is intentionally generic; concrete argument presets for science-ready scenarios still need to be codified once `mockgal.py` interface choices are finalized.

## Phase 5 Plan: Numba Diagnostics and mockgal Presets

### Scope
- review generated benchmark/profiling artifacts for actionable gaps
- improve numba speedup diagnostics beyond single mean-runtime reporting
- add science-ready `mockgal` adapter presets with structured overrides
- save compact continuation notes for next context window

### Checklist
| Item | Status | Notes |
|---|---|---|
| 1. Review existing outputs under `outputs/benchmarks_performance` and `outputs/benchmarks_profiling` | [x] | Reviewed speedup/profiling summaries and identified diagnostic gaps |
| 2. Propose follow-up diagnostics plan for numba benchmarking | [x] | Added steady-state/variability/overhead diagnostic plan and implemented directly |
| 3. Implement improved numba benchmark diagnostics (`n_runs`, `scale_factor`, steady-state metrics, slowdown flags) | [x] | Updated `benchmarks/performance/bench_numba_speedup.py` |
| 4. Validate diagnostics on baseline scale and scaled workload | [x] | Ran scale1 and scale2 outputs with persisted artifacts |
| 5. Add science-ready argument presets for `mockgal_adapter.py` | [x] | Added `single_noiseless_truth`, `single_psf_noise_sblimit`, `models_config_batch` |
| 6. Validate preset listing and dry-run execution with overrides | [x] | Verified preset list and two dry-run outputs |
| 7. Document commands and outcomes in docs and journal | [x] | Updated `benchmarks/README.md` and added journal closeout below |

### Review (Implementation Progress: Step 3)

- Updated `benchmarks/performance/bench_numba_speedup.py` to include:
  - configurable `--n-runs`
  - configurable `--scale-factor`
  - first-run vs steady-state diagnostics
  - timing variability diagnostics (coefficient of variation)
  - slowdown/high-variability case flags in JSON summary
  - enhanced CSV and three-panel diagnostic plot
- Baseline diagnostic run (`--n-runs 4 --scale-factor 1.0`) summary:
  - mean speedup: `1.32x`, steady-state mean speedup: `1.32x`
  - slowdown cases: `0`
  - high variability cases: `0`
- Scaled workload diagnostic run (`--n-runs 3 --scale-factor 2.0`) summary:
  - mean speedup: `1.22x`, steady-state mean speedup: `1.25x`
  - slowdown cases (steady-state): `1`
  - high variability cases: `3`
- Added structured `mockgal` presets in `benchmarks/baselines/mockgal_adapter.py`:
  - `single_noiseless_truth`
  - `single_psf_noise_sblimit`
  - `models_config_batch`
- Verified:
  - `--list-presets`
  - dry-run for `single_psf_noise_sblimit`
  - dry-run override flow for `models_config_batch`

### Residual Risks (Updated)

- Numba speedup remains sensitive to workload shape and run-to-run noise; use steady-state plus variability diagnostics instead of a single aggregate speedup number.
- `mockgal` presets are operational but should be expanded with project-specific model/config templates once science campaign parameter files are finalized.

## Phase 5 Follow-up Plan: Flagged Case Drills and Preset Templates

### Scope
- implement targeted numba profiling drills for slowdown/high-variability cases from measured JSON artifacts
- prioritize `n1_medium_eps07__scale2` in drill execution order
- add reusable `models_config_batch` YAML templates for adapter presets
- verify both profiling drill and template-backed preset dry-run outputs

### Checklist
| Item | Status | Notes |
|---|---|---|
| 1. Parse measured case flags from scale1 and scale2 numba benchmark JSON files | [x] | Implemented in `benchmarks/profiling/profile_numba_flagged_cases.py` (`summary` + `case_diagnostics`) |
| 2. Implement case-specific numba drill runner with per-case profiler artifacts | [x] | New script writes per-case `.prof`, text summaries, and `case_drill_summary.json` |
| 3. Ensure `n1_medium_eps07__scale2` is prioritized in default drill selection | [x] | `--focus-case` default set to `n1_medium_eps07__scale2` |
| 4. Add aggregate machine-readable drill summary | [x] | `numba_case_drill_summary.json` includes ranking by drill CV and source diagnostics |
| 5. Add reusable `models_config_batch` templates under `examples/` | [x] | Added `examples/mockgal/models_config_batch/galaxies.yaml` and `image_config.yaml` |
| 6. Point `models_config_batch` preset defaults to tracked template files | [x] | Updated default `models_file`/`config_file` in `mockgal_adapter.py` |
| 7. Verify script execution and preset dry-run outputs | [x] | Verified with uv commands listed below; artifacts written under `outputs/` |
| 8. Update docs/journal/spec references for continuation | [x] | Updated `benchmarks/README.md`, `examples/README.md`, `docs/spec.md`, and journal entry |

### Review (Implementation Progress: Step 4)

- Added new script:
  - `benchmarks/profiling/profile_numba_flagged_cases.py`
  - default source inputs:
    - `outputs/benchmarks_performance/bench_numba_speedup/numba_benchmark_results.json`
    - `outputs/benchmarks_performance/bench_numba_speedup_scale2/numba_benchmark_results.json`
  - default focus case:
    - `n1_medium_eps07__scale2`
- Drill outputs now include:
  - per-case profiler artifacts:
    - `case_profile.prof`
    - `top_cumulative.txt`
    - `top_tottime.txt`
    - `case_drill_summary.json`
  - aggregate summary:
    - `numba_case_drill_summary.json`
- Smoke verification run selected and executed flagged scale2 cases:
  - `n1_medium_eps07__scale2`
  - `n1_large_eps06__scale2`
  - `n1_medium_eps04__scale2`
- Added reusable templates:
  - `examples/mockgal/models_config_batch/galaxies.yaml`
  - `examples/mockgal/models_config_batch/image_config.yaml`
- Updated `models_config_batch` preset defaults to these tracked template files.

Verification commands:
- `/bin/zsh -lc 'UV_CACHE_DIR=.uv-cache uv run python benchmarks/profiling/profile_numba_flagged_cases.py --timing-runs 2 --profile-runs 1 --top-n 8 --output outputs/benchmarks_profiling/profile_numba_flagged_cases_smoke'`
- `/bin/zsh -lc 'UV_CACHE_DIR=.uv-cache uv run python benchmarks/baselines/mockgal_adapter.py --preset models_config_batch --dry-run --output outputs/benchmarks_performance/mockgal_adapter_template_example'`

### Residual Risks (Follow-up)

- Case-level variability is sensitive to local runtime noise; use repeated drill runs when comparing machines or commits.
- The template YAMLs are intentionally minimal examples and should be replaced with campaign-specific model/config files for production benchmark batches.

## Phase 5 Follow-up Extension: Long-Run Drill + Template Scaffold Helper

### Checklist
| Item | Status | Notes |
|---|---|---|
| 1. Run longer flagged-case drill execution | [x] | `profile_numba_flagged_cases.py --timing-runs 8 --profile-runs 2` |
| 2. Summarize variability deltas against source benchmark diagnostics | [x] | Wrote `variability_delta_summary.json` under long-run output directory |
| 3. Add helper script to scaffold template YAML files into new outputs run directory | [x] | Added `benchmarks/baselines/scaffold_models_config_batch_templates.py` |
| 4. Verify helper copies template files and persists manifest | [x] | Verified in `outputs/benchmarks_performance/mockgal_models_config_batch_templates_run1` |

### Review

- Long-run drill artifacts:
  - `outputs/benchmarks_profiling/profile_numba_flagged_cases_longrun/`
  - `outputs/benchmarks_profiling/profile_numba_flagged_cases_longrun/numba_case_drill_summary.json`
  - `outputs/benchmarks_profiling/profile_numba_flagged_cases_longrun/variability_delta_summary.json`
- Key focused case (`n1_medium_eps07__scale2`) improved strongly in long-run drill:
  - source numba CV: `0.16849`
  - drill steady-state CV: `0.00167`
- Added helper script:
  - `benchmarks/baselines/scaffold_models_config_batch_templates.py`
  - command:
    - `uv run python benchmarks/baselines/scaffold_models_config_batch_templates.py --output outputs/benchmarks_performance/mockgal_models_config_batch_templates_run1`
  - copied files:
    - `galaxies.yaml`
    - `image_config.yaml`
  - manifest:
    - `outputs/benchmarks_performance/mockgal_models_config_batch_templates_run1/template_scaffold_manifest.json`

## Pre-Merge Final Check (2026-02-12)

### Checklist
| Item | Status | Notes |
|---|---|---|
| 1. Check benchmark/tests/docs README consistency | [x] | Updated `README.md` and `tests/README.md`; `benchmarks/README.md` already aligned |
| 2. Lint all changed Python files | [x] | `ruff check` passed after non-behavioral cleanup |
| 3. Re-run focused Phase 4/5 regression tests | [x] | `5 + 27` tests passed; real-data collection validated |
| 4. Re-run smoke checks for new scripts | [x] | `profile_numba_flagged_cases.py` and `scaffold_models_config_batch_templates.py` passed |

### Verification Commands

- `/bin/zsh -lc 'UV_CACHE_DIR=.uv-cache uv run ruff check ...'`
- `/bin/zsh -lc 'UV_CACHE_DIR=.uv-cache uv run pytest tests/unit/test_public_api.py tests/integration/test_cli.py -q'`
- `/bin/zsh -lc 'UV_CACHE_DIR=.uv-cache uv run pytest tests/integration/test_sersic_accuracy.py tests/integration/test_numba_validation.py -q'`
- `/bin/zsh -lc 'UV_CACHE_DIR=.uv-cache uv run pytest tests/real_data/test_m51.py --collect-only -q -m real_data'`
- `/bin/zsh -lc 'UV_CACHE_DIR=.uv-cache uv run python benchmarks/profiling/profile_numba_flagged_cases.py --case n1_medium_eps07__scale2 --timing-runs 1 --profile-runs 1 --top-n 5 --output outputs/benchmarks_profiling/profile_numba_flagged_cases_premerge_smoke'`
- `/bin/zsh -lc 'UV_CACHE_DIR=.uv-cache uv run python benchmarks/baselines/scaffold_models_config_batch_templates.py --dry-run --output outputs/benchmarks_performance/mockgal_models_config_batch_templates_premerge_smoke'`

## Phase 6 Plan: Huang2013 Real Mock Demo (IC2597 mock1)

### Scope
- formalize durable run requirements for real Huang2013 mock datasets
- implement a reproducible demo runner for `IC2597_mock1.fits`
- keep photutils and isoster executions independently repeatable with shared artifact conventions
- generate per-method QA, cross-method comparison QA, and a concise markdown run report

### Checklist
| Item | Status | Notes |
|---|---|---|
| 1. Persist user requirements in `examples/huang2013` memory file | [x] | Create a requirement-spec markdown for this campaign |
| 2. Define artifact naming and manifest schema using `GALAXY_MOCKID` prefix | [x] | Include profile FITS/ECSV, JSON metadata, QA PNG, report MD |
| 3. Implement a script that reads external mock FITS and runs photutils independently | [x] | Save profile + runtime + true CoG columns |
| 4. Implement a script that reads external mock FITS and runs isoster independently | [x] | Save profile + runtime + true CoG columns |
| 5. Include run profiling and persist runtime diagnostics | [x] | Save wall/cpu timings and cProfile top-call summary |
| 6. Generate per-method QA figure with image/isophotes, model, residual, and 1-D panels | [x] | Use shared x-axis in `kpc^0.25`, stop-code styling, PA normalization |
| 7. Generate method-comparison QA figure from saved artifacts | [x] | Include relative surface-brightness difference panel |
| 8. Generate concise markdown report and JSON manifest linking all artifacts | [x] | Summary includes runtime and fit-quality counts |
| 9. Verify script entrypoint and help output | [x] | `--help` + smoke runs completed with reduced `maxsma` |
| 10. Split extraction and QA into separate scripts (afterburner workflow) | [x] | Added dedicated extraction + QA afterburner entrypoints and verified both |
| 11. Execute full IC2597 mock1 baseline in external dataset folder | [x] | Extraction + afterburner completed under `/Users/mac/work/hsc/huang2013/IC2597` |

### Review

- Added requirements-memory document:
  - `examples/huang2013/real-huang2013-requirements.md`
- Added split workflow entrypoints:
  - extraction: `examples/huang2013/run_huang2013_profile_extraction.py`
  - QA afterburner: `examples/huang2013/run_huang2013_qa_afterburner.py`
- Kept shared reusable implementation in:
  - `examples/huang2013/run_huang2013_real_mock_demo.py`
- Implemented behavior and conventions:
  - independent method execution (`photutils`, `isoster`, or both)
  - output naming based on `GALAXY_MOCKID` and method/config suffix
  - profile FITS/ECSV outputs with derived columns needed for 1-D and 2-D reproduction
  - runtime profiling (`wall`, `cpu`, and cProfile top-call dump)
  - true CoG append using high-subpixel elliptical apertures (`subpixels=9` default for extraction)
  - per-method QA and comparison QA generated in afterburner stage from saved profile artifacts
  - markdown report and manifest JSON written in afterburner stage
- User-confirmed defaults applied:
  - pixel scale source: FITS header (`PIXSCALE`)
  - `use_eccentric_anomaly=False`
  - true CoG subpixel sampling factor: `9`
- Lightweight validation completed on `IC2597_mock1.fits` with reduced `maxsma`:
  - extraction smoke run (`--method both`)
  - QA afterburner smoke run (`--method both`)
  - split artifact generation verified in `/tmp/huang2013_ic2597_split_smoke`
- Production baseline execution completed in external target folder:
  - profile extraction:
    - `/Users/mac/work/hsc/huang2013/IC2597/IC2597_mock1_profiles_manifest.json`
  - QA/report afterburner:
    - `/Users/mac/work/hsc/huang2013/IC2597/IC2597_mock1_qa_manifest.json`
    - `/Users/mac/work/hsc/huang2013/IC2597/IC2597_mock1_report.md`
- Production run highlights (`IC2597_mock1`):
  - runtime:
    - photutils wall `7.261 s`
    - isoster wall `0.513 s`
  - isophotes:
    - photutils `64` (`62` converged, stop codes `0:62, 2:2`)
    - isoster `64` (`64` converged, stop code `0:64`)
  - median absolute relative surface-brightness difference:
    - `0.01884%`

## Phase 6 Follow-up: Huang2013 Folder Cleanup and Guide

### Checklist
| Item | Status | Notes |
|---|---|---|
| 1. Remove legacy Huang2013 relic files with `git rm` | [x] | Removed old model/mock/test/QA scripts from `examples/huang2013` |
| 2. Add `examples/huang2013/README.md` with test setup and TOC | [x] | Added workflow overview, commands, and artifact naming |
| 3. Document exact QA reproduction command for IC2597 | [x] | Added under README section “Reproduce IC2597 QA Figures” |

### Review

- Removed tracked legacy files:
  - `examples/huang2013/huang2013_models.yaml`
  - `examples/huang2013/mockgal.py`
  - `examples/huang2013/plot_qa_huang2013.py`
  - `examples/huang2013/test_huang2013_mocks.py`
- Added workflow README:
  - `examples/huang2013/README.md`
- README now provides:
  - concise setup assumptions
  - table of contents
  - extraction and QA afterburner usage
  - direct IC2597 QA reproduction commands
  - plot customization entry points
