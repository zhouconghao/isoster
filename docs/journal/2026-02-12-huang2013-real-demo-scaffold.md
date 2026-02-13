# 2026-02-12 Huang2013 Real Demo Scaffold

## Context

User requested a scalable workflow starting from one demo target:

- galaxy: `IC2597`
- image: `IC2597_mock1.fits` from external root `/Users/mac/work/hsc/huang2013`
- objective: independent `photutils.isophote` and `isoster` runs with reproducible artifacts, QA figures, runtime profiling, and concise report outputs.

## Implemented

1. Requirements memory document:
   - `examples/huang2013/real-huang2013-requirements.md`
2. New runnable script:
   - `examples/huang2013/run_huang2013_real_mock_demo.py`
3. Script outputs (method-specific):
   - profile FITS + ECSV
   - run JSON metadata
   - runtime cProfile text
   - per-method QA figure
4. Script outputs (joint):
   - cross-method comparison QA figure
   - run report markdown
   - manifest JSON
5. Added true CoG append to profile tables using high-subpixel elliptical aperture photometry on the noiseless image.
6. Added PA normalization for plotting and stop-code-aware point styling.

## Validation Performed

Lightweight smoke validations only (reduced `maxsma`), no full production run yet:

- photutils-only smoke:
  - `MPLCONFIGDIR=/tmp/matplotlib-cache XDG_CACHE_HOME=/tmp/xdg-cache python examples/huang2013/run_huang2013_real_mock_demo.py --galaxy IC2597 --mock-id 1 --method photutils --maxsma 18 --skip-comparison --output-dir /tmp/huang2013_ic2597_demo_smoke2 --qa-dpi 80`
- photutils + isoster smoke:
  - `MPLCONFIGDIR=/tmp/matplotlib-cache XDG_CACHE_HOME=/tmp/xdg-cache python examples/huang2013/run_huang2013_real_mock_demo.py --galaxy IC2597 --mock-id 1 --method both --maxsma 15 --output-dir /tmp/huang2013_ic2597_demo_smoke_both --qa-dpi 70`

Both commands generated expected artifacts and manifests.

## Notes

- FITS header in `IC2597_mock1.fits` reports `PIXSCALE=0.168`, which differs from the user-provided context value `0.176`. Current script defaults to FITS header values unless explicitly overridden by CLI.
- Full baseline execution for IC2597 with production-radius settings is deferred pending user confirmation of final baseline parameters.

## Addendum: Workflow Split

After user feedback, the workflow was split into two explicit stages:

1. Profile extraction only:
   - `examples/huang2013/run_huang2013_profile_extraction.py`
2. QA/report afterburner:
   - `examples/huang2013/run_huang2013_qa_afterburner.py`

Current defaults aligned with user decisions:

- pixel scale source: FITS header
- `use_eccentric_anomaly=False`
- true CoG aperture subpixel factor: `9`

Smoke validation of the split pipeline:

- extraction command generated profile artifacts + profile manifest
- afterburner command consumed those artifacts and generated per-method QA, comparison QA, report, and QA manifest

## Production Baseline Execution (IC2597 mock1)

Executed in external folder:

- `/Users/mac/work/hsc/huang2013/IC2597`

Generated:

- profile manifest: `IC2597_mock1_profiles_manifest.json`
- QA manifest: `IC2597_mock1_qa_manifest.json`
- report: `IC2597_mock1_report.md`
- per-method profile products and QA figures
- comparison QA: `IC2597_mock1_compare_baseline_qa.png`

Key numbers:

- runtime:
  - photutils wall: `7.261 s`
  - isoster wall: `0.513 s`
- isophotes:
  - photutils: `64` total, `62` converged
  - isoster: `64` total, `64` converged
- median absolute relative surface-brightness difference:
  - `0.01884%`

Robustness fixes applied during production run:

1. Added photutils `maxsma` retry ladder with decreasing radii.
2. Switched automatic `maxsma` default to header-driven scale (`RE_PX*` based).
3. Fixed runtime profiler cleanup to always disable in `finally`.
