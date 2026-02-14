# Phase 10 Clean-Context Handoff (2026-02-14, v2)

## Current Branch and Workspace

- Branch: `phase10-qa-prep-p1`
- Workspace is intentionally dirty (do not reset):
  - benchmark/gate files and docs from ongoing Phase 10
  - Huang2013 QA plotting updates in `examples/huang2013/run_huang2013_real_mock_demo.py`

## What Is Completed So Far

1. Basic Huang2013 QA style is finalized for IC2597 and accepted by user.
2. `build_method_qa_figure()` style has been propagated to `build_comparison_qa_figure()`.
3. Data/model display scaling consistency fix is applied:
   - model panel now uses scaling parameters derived from the original image.
4. Error-bar defaults added:
   - centroid `dx/dy`: uses `x0_err` and `y0_err` when available
   - axis ratio: uses `ellip_err`, fallback to `eps_err`
5. QA style constraint persisted to `CLAUDE.md` under `## QA Figure Rules`.

## Key Files Touched for QA Work

- `examples/huang2013/run_huang2013_real_mock_demo.py`
- `docs/todo.md`
- `docs/lessons.md`
- `CLAUDE.md`
- `docs/journal/2026-02-14-phase10-ic2597-qa-style-pass.md`

## Latest Verification Artifacts

- `outputs/huang2013_ic2597_qa_style/IC2597_mock1_photutils_baseline_qa.png`
- `outputs/huang2013_ic2597_qa_style/IC2597_mock1_isoster_baseline_qa.png`
- `outputs/huang2013_ic2597_qa_style/IC2597_mock1_compare_baseline_qa.png`
- `outputs/huang2013_ic2597_qa_style/IC2597_mock1_qa_manifest.json`

## Remaining Phase-10 Work (Core)

From `docs/todo.md` Phase 10 plan:

1. Integrate QA figure generation into `benchmarks/baselines/run_benchmark_gate.py`.
2. Persist per-case QA artifact paths in `benchmark_gate_report.json`.
3. Keep gate policy unchanged: locked efficiency + locked 1-D profile decide pass/fail; 2-D remains system-level diagnostics with caveat.
4. Run quick smoke first, then full lock-refresh + full gate.
5. Update docs with final evidence and output paths.

## Resume Commands (Suggested)

```bash
# 1) Validate current edited QA script
uv run ruff check examples/huang2013/run_huang2013_real_mock_demo.py

# 2) Quick QA regeneration sanity (optional)
uv run python examples/huang2013/run_huang2013_qa_afterburner.py \
  --galaxy IC2597 --mock-id 1 --method both --config-tag baseline \
  --output-dir outputs/huang2013_ic2597_qa_style \
  --photutils-profile-fits /Users/mac/work/hsc/huang2013/IC2597/IC2597_mock1_photutils_baseline_profile.fits \
  --isoster-profile-fits /Users/mac/work/hsc/huang2013/IC2597/IC2597_mock1_isoster_baseline_profile.fits \
  --photutils-run-json /Users/mac/work/hsc/huang2013/IC2597/IC2597_mock1_photutils_baseline_run.json \
  --isoster-run-json /Users/mac/work/hsc/huang2013/IC2597/IC2597_mock1_isoster_baseline_run.json

# 3) Continue Phase-10 gate integration work
uv run ruff check benchmarks/baselines/run_benchmark_gate.py
```

## First Prompt for New Clean Context Window

Use this prompt verbatim:

"Continue on branch `phase10-qa-prep-p1` for ISOSTER Phase 10. Start by reading `CLAUDE.md`, `docs/todo.md`, and `docs/journal/2026-02-14-phase10-clean-context-handoff-v2.md`. Respect the persisted QA-style rule in `CLAUDE.md`: use the finalized IC2597 Huang2013 basic-QA style as the default for future QA figures. Then proceed with remaining Phase-10 tasks: integrate QA generation into `benchmarks/baselines/run_benchmark_gate.py`, store per-case QA artifact paths in gate JSON, run quick smoke then full lock-refresh/full gate, and update docs with verification evidence and output paths. Do not reset unrelated dirty files." 
