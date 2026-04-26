"""Phase F.c.2+ — cross-tool extended metrics (efficiency, residuals, combined).

Walks every per-tool ``inventory.fits`` under a campaign root and
aggregates three complementary views that the prior-violation audit
by itself does not show:

1. **Efficiency** — median and p95 of ``wall_time_fit_s`` per (tool,
   arm). Answers "how fast does each winner actually fit?"
2. **Residuals** — median of ``resid_rms_zone / image_sigma_adu`` for
   inner / mid / outer zones. Noise-normalized so numbers are
   comparable across tools (same 0-1-ish scale regardless of depth).
3. **Combined** — a single score per (tool, arm):

       combined = w_inner * R_in/sigma + w_mid * R_mid/sigma
                + w_outer * R_out/sigma
                + w_time  * (T_fit / T_ref)
                + w_viol  * V

   with defaults ``(w_inner, w_mid, w_outer) = (1, 1, 2)`` and
   ``w_time = 0.1`` (matching ``cross_tool_score_weights`` in every
   campaign YAML) and ``w_viol = 1.0`` (new — prior-violation rate).
   ``T_ref`` is the median ``wall_time_fit_s`` of
   ``isoster:ref_default`` pooled across all 837 galaxies. Lower is
   better, same convention as the existing composite.

The new combined score extends the existing ``cross_tool_score``
(residual + runtime only, present in per-campaign cross_tool_table)
by mixing in the prior-violation rate from the audit pipeline, so
regularization arms that visibly fix Prior 3 at a minor residual /
runtime cost can legitimately rank above arms that look clean on
residuals but leave the geometry wobbling.

The driver also joins in the Phase F.c.2 ``clean_frac`` per arm from
the per-tool audit CSVs, so every table has both the prior-based
shortlist fraction *and* the new cross-tool quality score side by
side.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
from astropy.io import fits

mpl.use("Agg")

from .cross_tool_composite import (  # noqa: E402
    PRIOR_COLS,
    _is_clean,
    _parse_bool_or_none,
)


TOOLS = ("isoster", "photutils", "autoprof")


# ---------------------------------------------------------------------------
# Inventory walker
# ---------------------------------------------------------------------------


@dataclass
class ArmRecord:
    """One arm's record as read from a per-tool inventory.fits row."""

    galaxy_id: str
    scenario: str
    tool: str
    arm_id: str
    status: str
    wall_time_fit_s: float
    resid_rms_inner: float
    resid_rms_mid: float
    resid_rms_outer: float
    image_sigma_adu: float
    composite_score: float
    combined_drift_pix: float
    frac_stop_nonzero: float
    flag_severity_max: float


def _fval(row: Any, name: str) -> float:
    try:
        v = float(row[name])
    except (TypeError, ValueError, KeyError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _iter_inventory_rows(
    campaign_root: Path, dataset: str
) -> list[ArmRecord]:
    """Walk every per-tool inventory.fits under the campaign tree."""
    out: list[ArmRecord] = []
    for campaign_dir in sorted(campaign_root.iterdir()):
        if not campaign_dir.is_dir() or campaign_dir.name.startswith("_"):
            continue
        scenario = campaign_dir.name.removeprefix(f"{dataset}_")
        ds_dir = campaign_dir / dataset
        if not ds_dir.is_dir():
            continue
        for gdir in sorted(ds_dir.iterdir()):
            if not gdir.is_dir() or "__" not in gdir.name:
                continue
            for tool in TOOLS:
                inv = gdir / tool / "inventory.fits"
                if not inv.is_file():
                    continue
                try:
                    with fits.open(inv) as hdul:
                        data = hdul[1].data
                except (OSError, ValueError, IndexError):
                    continue
                if data is None:
                    continue
                for row in data:
                    out.append(
                        ArmRecord(
                            galaxy_id=str(row["galaxy_id"]),
                            scenario=scenario,
                            tool=tool,
                            arm_id=str(row["arm_id"]),
                            status=str(row["status"]),
                            wall_time_fit_s=_fval(row, "wall_time_fit_s"),
                            resid_rms_inner=_fval(row, "resid_rms_inner"),
                            resid_rms_mid=_fval(row, "resid_rms_mid"),
                            resid_rms_outer=_fval(row, "resid_rms_outer"),
                            image_sigma_adu=_fval(row, "image_sigma_adu"),
                            composite_score=_fval(row, "composite_score"),
                            combined_drift_pix=_fval(row, "combined_drift_pix"),
                            frac_stop_nonzero=_fval(row, "frac_stop_nonzero"),
                            flag_severity_max=_fval(row, "flag_severity_max"),
                        )
                    )
    return out


# ---------------------------------------------------------------------------
# Audit CSV -> clean_frac per (tool, arm)
# ---------------------------------------------------------------------------


def _safe_arm_id(arm_id: str) -> str:
    """Normalize arm id so the audit CSV's safe form matches the
    inventory's raw form. The benchmark runner stores harmonic-sweep
    arms as ``harm_higher_orders::<joined>`` on the FITS side but
    writes ``harm_higher_orders__<joined>`` in per-arm paths and the
    audit CSV. Collapsing ``::`` -> ``__`` is sufficient for the
    known cases; other arm ids pass through unchanged.
    """
    return arm_id.replace("::", "__")


def _load_audit_clean_fracs(analysis_root: Path) -> dict[tuple[str, str], float]:
    """Map (tool, arm_id) -> pooled clean_frac from the per-tool audit CSV.

    Keys use the normalized (safe) arm id so the join against inventory
    rows (which carry the raw ``::``-separated form for harmonic-sweep
    arms) works uniformly.
    """
    out: dict[tuple[str, str], float] = {}
    for tool in TOOLS:
        csv_path = (
            analysis_root
            / f"cross_scenario_audit_{tool}"
            / "cross_scenario_ideal_case_table.csv"
        )
        if not csv_path.is_file():
            continue
        with csv_path.open() as handle:
            reader = csv.DictReader(handle)
            per_arm_counts: dict[str, list[bool]] = defaultdict(list)
            for raw in reader:
                if raw.get("status") != "ok":
                    continue
                # Reconstruct _is_clean on the fly.
                rec: dict[str, Any] = dict(raw)
                for col, _ in PRIOR_COLS:
                    rec[col] = _parse_bool_or_none(raw.get(col, ""))
                per_arm_counts[_safe_arm_id(str(raw["arm_id"]))].append(_is_clean(rec))
        for arm, bits in per_arm_counts.items():
            if bits:
                out[(tool, arm)] = float(np.mean(bits))
    return out


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


@dataclass
class ArmAggregate:
    tool: str
    arm_id: str
    n_ok: int = 0
    wall_times: list[float] = field(default_factory=list)
    resid_in_norm: list[float] = field(default_factory=list)
    resid_mid_norm: list[float] = field(default_factory=list)
    resid_out_norm: list[float] = field(default_factory=list)
    composite_scores: list[float] = field(default_factory=list)
    drifts: list[float] = field(default_factory=list)
    stop_nonzero_fracs: list[float] = field(default_factory=list)


def _aggregate(records: list[ArmRecord]) -> dict[tuple[str, str], ArmAggregate]:
    """Per-(tool, arm) pooled lists across every galaxy/scenario."""
    agg: dict[tuple[str, str], ArmAggregate] = {}
    for r in records:
        if r.status != "ok":
            continue
        key = (r.tool, r.arm_id)
        e = agg.setdefault(key, ArmAggregate(tool=r.tool, arm_id=r.arm_id))
        e.n_ok += 1
        if np.isfinite(r.wall_time_fit_s):
            e.wall_times.append(r.wall_time_fit_s)
        sigma = r.image_sigma_adu if r.image_sigma_adu > 0 else float("nan")
        if np.isfinite(sigma):
            if np.isfinite(r.resid_rms_inner):
                e.resid_in_norm.append(r.resid_rms_inner / sigma)
            if np.isfinite(r.resid_rms_mid):
                e.resid_mid_norm.append(r.resid_rms_mid / sigma)
            if np.isfinite(r.resid_rms_outer):
                e.resid_out_norm.append(r.resid_rms_outer / sigma)
        if np.isfinite(r.composite_score):
            e.composite_scores.append(r.composite_score)
        if np.isfinite(r.combined_drift_pix):
            e.drifts.append(r.combined_drift_pix)
        if np.isfinite(r.frac_stop_nonzero):
            e.stop_nonzero_fracs.append(r.frac_stop_nonzero)
    return agg


def _med(xs: list[float]) -> float:
    return float(np.median(xs)) if xs else float("nan")


def _p95(xs: list[float]) -> float:
    return float(np.percentile(xs, 95)) if xs else float("nan")


# ---------------------------------------------------------------------------
# Combined score
# ---------------------------------------------------------------------------


@dataclass
class Weights:
    w_inner: float = 1.0
    w_mid: float = 1.0
    w_outer: float = 2.0
    w_time: float = 0.1
    w_viol: float = 1.0


def _combined_score_rows(
    aggregates: dict[tuple[str, str], ArmAggregate],
    clean_fracs: dict[tuple[str, str], float],
    t_ref_s: float,
    weights: Weights,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (tool, arm), agg in aggregates.items():
        if agg.n_ok == 0:
            continue
        r_in = _med(agg.resid_in_norm)
        r_mid = _med(agg.resid_mid_norm)
        r_out = _med(agg.resid_out_norm)
        t_med = _med(agg.wall_times)
        t_ratio = t_med / t_ref_s if (t_ref_s > 0 and np.isfinite(t_med)) else float("nan")
        clean = clean_fracs.get((tool, _safe_arm_id(arm)), float("nan"))
        v = (1.0 - clean) if np.isfinite(clean) else float("nan")
        parts = {
            "inner": weights.w_inner * r_in if np.isfinite(r_in) else 0.0,
            "mid": weights.w_mid * r_mid if np.isfinite(r_mid) else 0.0,
            "outer": weights.w_outer * r_out if np.isfinite(r_out) else 0.0,
            "time": weights.w_time * t_ratio if np.isfinite(t_ratio) else 0.0,
            "viol": weights.w_viol * v if np.isfinite(v) else 0.0,
        }
        total = sum(parts.values())
        rows.append({
            "tool": tool,
            "arm_id": arm,
            "n_ok": agg.n_ok,
            "clean_frac": clean,
            "viol_rate": v,
            "R_in_norm": r_in,
            "R_mid_norm": r_mid,
            "R_out_norm": r_out,
            "T_med_s": t_med,
            "T_p95_s": _p95(agg.wall_times),
            "T_ratio": t_ratio,
            "composite_score_med": _med(agg.composite_scores),
            "drift_med_pix": _med(agg.drifts),
            "stop_nonzero_med": _med(agg.stop_nonzero_fracs),
            "score_part_inner": parts["inner"],
            "score_part_mid": parts["mid"],
            "score_part_outer": parts["outer"],
            "score_part_time": parts["time"],
            "score_part_viol": parts["viol"],
            "combined": total,
        })
    rows.sort(key=lambda r: r["combined"] if np.isfinite(r["combined"]) else float("inf"))
    return rows


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _f(v: float, fmt: str = ".3f") -> str:
    return format(v, fmt) if np.isfinite(v) else "N/A"


def write_efficiency(
    rows: list[dict[str, Any]],
    outfile: Path,
    top_k: int = 5,
    t_ref_s: float = float("nan"),
    dataset: str = "huang2013",
) -> None:
    with outfile.open("w") as h:
        h.write(f"# {dataset} cross-tool efficiency (runtime)\n\n")
        h.write(
            "`wall_time_fit_s` is the per-arm fit wall time as recorded in "
            "each arm's `inventory.fits` row (campaign-level field). "
            f"`T_ref = isoster/ref_default` median = **{_f(t_ref_s, '.3f')} s**; "
            "`T_ratio` expresses each arm's median fit time as a multiple of "
            "that reference.\n\n"
            "Per tool: top-k arms by Phase F.c.2 `clean_frac` (from the "
            "audit pipeline), with their cross-scenario runtime summary.\n\n"
        )
        for tool in TOOLS:
            per_tool = [r for r in rows if r["tool"] == tool]
            if not per_tool:
                continue
            per_tool.sort(
                key=lambda r: -r["clean_frac"] if np.isfinite(r["clean_frac"]) else 0.0
            )
            h.write(f"## {tool}\n\n")
            h.write(
                "| arm | clean_frac | T_median [s] | T_p95 [s] | T_ratio |\n"
                "|---|---|---|---|---|\n"
            )
            for r in per_tool[:top_k]:
                h.write(
                    f"| `{r['arm_id']}` | {_f(r['clean_frac'], '.3f')} "
                    f"| {_f(r['T_med_s'], '.3f')} | {_f(r['T_p95_s'], '.3f')} "
                    f"| {_f(r['T_ratio'], '.2f')} |\n"
                )
            h.write("\n")


def write_residuals(
    rows: list[dict[str, Any]],
    outfile: Path,
    top_k: int = 5,
    dataset: str = "huang2013",
) -> None:
    with outfile.open("w") as h:
        h.write(f"# {dataset} cross-tool residual quality (noise-normalized)\n\n")
        h.write(
            "Each entry is the **median** of `resid_rms_zone / image_sigma_adu` "
            f"pooled across the {dataset} scenario grid. Value of 1.0 means the "
            "residual RMS in that zone matches the per-image noise; values < 1.0 "
            "indicate sub-noise residuals, > 1.0 indicates systematic mismatch.\n\n"
            "Per tool: top-k arms by Phase F.c.2 `clean_frac`, with their "
            "residual-zone fingerprint.\n\n"
        )
        for tool in TOOLS:
            per_tool = [r for r in rows if r["tool"] == tool]
            if not per_tool:
                continue
            per_tool.sort(
                key=lambda r: -r["clean_frac"] if np.isfinite(r["clean_frac"]) else 0.0
            )
            h.write(f"## {tool}\n\n")
            h.write(
                "| arm | clean_frac | R_inner / σ | R_mid / σ | R_outer / σ |\n"
                "|---|---|---|---|---|\n"
            )
            for r in per_tool[:top_k]:
                h.write(
                    f"| `{r['arm_id']}` | {_f(r['clean_frac'], '.3f')} "
                    f"| {_f(r['R_in_norm'], '.3f')} | {_f(r['R_mid_norm'], '.3f')} "
                    f"| {_f(r['R_out_norm'], '.3f')} |\n"
                )
            h.write("\n")


def write_combined(
    rows: list[dict[str, Any]],
    outfile: Path,
    weights: Weights,
    top_k_per_tool: int = 5,
    global_top: int = 15,
    t_ref_s: float = float("nan"),
    dataset: str = "huang2013",
) -> None:
    with outfile.open("w") as h:
        h.write(f"# {dataset} cross-tool combined score\n\n")
        h.write(
            "Combined score (lower = better):\n\n"
            "    combined = w_in · R_in/σ + w_mid · R_mid/σ + w_out · R_out/σ\n"
            "             + w_time · (T_fit / T_ref) + w_viol · V\n\n"
            f"Weights: `w_in={weights.w_inner}`, `w_mid={weights.w_mid}`, "
            f"`w_out={weights.w_outer}`, `w_time={weights.w_time}`, "
            f"`w_viol={weights.w_viol}`.  "
            f"`T_ref = isoster/ref_default` median = **{_f(t_ref_s, '.3f')} s**. "
            "`V = 1 − clean_frac` (from the Phase F.c.2 audit); N/A priors "
            "(autoprof Prior 2) are excluded from V's denominator.\n\n"
            "The existing `cross_tool_score` column in each per-campaign "
            "`cross_tool_table.csv` uses only the residual + runtime terms; "
            "the combined score adds the prior-violation term so arms that "
            "visibly fix Prior 3 at a small residual / runtime cost can rank "
            "above arms that look fine on residuals but leave the geometry "
            "wobbling.\n\n"
        )
        h.write("## Global top-" + str(global_top) + " across every (tool, arm)\n\n")
        h.write(
            "| rank | tool | arm | clean_frac | R_in/σ | R_mid/σ | R_out/σ "
            "| T_med [s] | combined |\n"
            "|---|---|---|---|---|---|---|---|---|\n"
        )
        for i, r in enumerate(rows[:global_top], start=1):
            h.write(
                f"| {i} | {r['tool']} | `{r['arm_id']}` "
                f"| {_f(r['clean_frac'], '.3f')} | {_f(r['R_in_norm'], '.3f')} "
                f"| {_f(r['R_mid_norm'], '.3f')} | {_f(r['R_out_norm'], '.3f')} "
                f"| {_f(r['T_med_s'], '.3f')} | {_f(r['combined'], '.3f')} |\n"
            )
        h.write("\n## Per-tool top-" + str(top_k_per_tool) + "\n\n")
        for tool in TOOLS:
            per_tool = [r for r in rows if r["tool"] == tool]
            if not per_tool:
                continue
            per_tool.sort(
                key=lambda r: r["combined"]
                if np.isfinite(r["combined"]) else float("inf")
            )
            h.write(f"### {tool}\n\n")
            h.write(
                "| arm | clean_frac | R_out/σ | T_med [s] | combined "
                "| (inner+mid+out) | w_t·T̂ | w_v·V |\n"
                "|---|---|---|---|---|---|---|---|\n"
            )
            for r in per_tool[:top_k_per_tool]:
                res_sum = (
                    r["score_part_inner"]
                    + r["score_part_mid"]
                    + r["score_part_outer"]
                )
                h.write(
                    f"| `{r['arm_id']}` | {_f(r['clean_frac'], '.3f')} "
                    f"| {_f(r['R_out_norm'], '.3f')} "
                    f"| {_f(r['T_med_s'], '.3f')} | {_f(r['combined'], '.3f')} "
                    f"| {_f(res_sum, '.3f')} "
                    f"| {_f(r['score_part_time'], '.3f')} "
                    f"| {_f(r['score_part_viol'], '.3f')} |\n"
                )
            h.write("\n")


def write_full_csv(rows: list[dict[str, Any]], outfile: Path) -> None:
    if not rows:
        outfile.write_text("")
        return
    fields = list(rows[0].keys())
    with outfile.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cross_tool_extended_metrics",
        description="Cross-tool efficiency / residual / combined-score tables.",
    )
    parser.add_argument(
        "--campaigns-root",
        type=Path,
        default=Path("/Volumes/galaxy/isophote/huang2013/_campaigns"),
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=Path("/Volumes/galaxy/isophote/huang2013/_campaigns/_analysis"),
    )
    parser.add_argument("--dataset", default="huang2013")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output dir (default: <analysis-root>/cross_tool_extended_metrics).",
    )
    parser.add_argument("--top-k-per-tool", type=int, default=5)
    parser.add_argument("--global-top", type=int, default=15)
    parser.add_argument("--w-inner", type=float, default=1.0)
    parser.add_argument("--w-mid", type=float, default=1.0)
    parser.add_argument("--w-outer", type=float, default=2.0)
    parser.add_argument("--w-time", type=float, default=0.1)
    parser.add_argument("--w-violation", type=float, default=1.0)
    parser.add_argument(
        "--ref-arm-for-time",
        default="isoster:ref_default",
        help="(tool:arm) whose median wall_time_fit_s is T_ref.",
    )
    args = parser.parse_args(argv)

    out_dir = args.out or (args.analysis_root / "cross_tool_extended_metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Walking inventory.fits under {args.campaigns_root} ...")
    records = _iter_inventory_rows(args.campaigns_root, args.dataset)
    print(f"  loaded {len(records)} arm rows")
    aggregates = _aggregate(records)
    print(f"  aggregated into {len(aggregates)} (tool, arm) groups")

    # T_ref from the reference arm.
    ref_tool, ref_arm = args.ref_arm_for_time.split(":", 1)
    ref_agg = aggregates.get((ref_tool, ref_arm))
    t_ref_s = _med(ref_agg.wall_times) if ref_agg is not None else float("nan")
    print(f"  T_ref ({args.ref_arm_for_time}) median = {t_ref_s:.3f} s")

    clean_fracs = _load_audit_clean_fracs(args.analysis_root)
    print(f"  loaded clean_frac for {len(clean_fracs)} (tool, arm) pairs")

    weights = Weights(
        w_inner=args.w_inner,
        w_mid=args.w_mid,
        w_outer=args.w_outer,
        w_time=args.w_time,
        w_viol=args.w_violation,
    )
    rows = _combined_score_rows(aggregates, clean_fracs, t_ref_s, weights)

    write_efficiency(
        rows, out_dir / "cross_tool_efficiency.md",
        top_k=args.top_k_per_tool, t_ref_s=t_ref_s,
        dataset=args.dataset,
    )
    print(f"  wrote {out_dir / 'cross_tool_efficiency.md'}")
    write_residuals(
        rows, out_dir / "cross_tool_residuals.md",
        top_k=args.top_k_per_tool, dataset=args.dataset,
    )
    print(f"  wrote {out_dir / 'cross_tool_residuals.md'}")
    write_combined(
        rows, out_dir / "cross_tool_combined.md",
        weights=weights,
        top_k_per_tool=args.top_k_per_tool,
        global_top=args.global_top,
        t_ref_s=t_ref_s,
        dataset=args.dataset,
    )
    print(f"  wrote {out_dir / 'cross_tool_combined.md'}")
    write_full_csv(rows, out_dir / "cross_tool_extended_table.csv")
    print(f"  wrote {out_dir / 'cross_tool_extended_table.csv'}")

    print(f"\nDone. Artifacts under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
