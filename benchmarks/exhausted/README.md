# Exhausted Benchmark Campaign

Multi-tool, multi-arm, multi-galaxy, multi-dataset benchmark for
`isoster`, `photutils.isophote`, and `autoprof`.

## Status

Under active development on branch `feat/benchmark-exhausted-v2`.

- **Phase A (scaffolding)**: in progress — directory skeleton, YAML
  schema, Huang2013 adapter stub, CLI dry-run.
- **Phase B (isoster fits)**: not started.
- **Phase C (statistics)**: not started.
- **Phase D (photutils + autoprof)**: not started.
- **Phase E (hardening + parallelism)**: not started.

Full plan: `docs/agent/plans/2026-04-20-benchmark-exhausted-v2.md`
(to be written).

## Quick start (planned — not yet runnable)

```bash
# Copy the example campaign YAML and edit it
cp benchmarks/exhausted/configs/campaign.example.yaml /tmp/my_campaign.yaml

# Preview the planned work without running any fits
uv run python -m benchmarks.exhausted.orchestrator.cli dry-run /tmp/my_campaign.yaml

# Execute the campaign (Phase B+ only)
uv run python -m benchmarks.exhausted.orchestrator.cli run /tmp/my_campaign.yaml
```

## Layout

```
benchmarks/exhausted/
├── configs/       YAML arm rosters and the campaign YAML template
├── adapters/      dataset loaders (Huang2013, HSC edge-real, etc.)
├── fitters/       per-tool fit drivers (isoster / photutils / autoprof)
├── analysis/      metrics, residual zones, quality flags, inventory writer
├── plotting/      per-galaxy QA and multi-galaxy summary grids
├── orchestrator/  CLI, runner, stats, reports
└── legacy/        archived single-galaxy 39-config sweep (pre-v2)
```

## Output location

All outputs land **outside the repo** at the `output_root` declared in
the campaign YAML. Default: `$HOME/isoster_campaign_runs/<campaign_name>/`.
