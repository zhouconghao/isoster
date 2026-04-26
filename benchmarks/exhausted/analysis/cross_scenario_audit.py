"""Phase F.a — cross-scenario ranking across the 9 huang2013 campaigns.

Applies the overhauled three-prior metrics from ``scenario_summary`` to
every ``huang2013_{clean,wide,deep}_z{005,020,035,050}`` campaign and
emits:

1. ``cross_scenario_ideal_case_table.csv`` — long-form (galaxy, arm,
   scenario, metrics, violations).
2. ``cross_scenario_violations_heatmap.pdf`` — one page per prior;
   cell = fraction of galaxies (per arm, per scenario) violating the
   prior.
3. ``cross_scenario_arm_ranking.md`` — arm ranking aggregated over all
   9 scenarios, plus per-scenario tables.
4. ``cross_scenario_baseline_calibration.md`` — per-metric distribution
   on ``harm_higher_orders__5_6`` (best clean_z005 arm) across all 9
   scenarios; used to calibrate thresholds.

Scenarios are inferred from the campaign directory names of the form
``huang2013_<depth>_z<redshift>``.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from .scenario_summary import (
    classify_violations,
    compute_prior_metrics,
    list_arms,
    list_campaign_galaxies,
    load_arm_profile,
    load_arm_record,
    load_galaxy_manifest,
)


PRIOR_COLS = [
    ("violates_drift_any", "Prior 1: IW drift (any zone)"),
    ("violates_harmonics_outer", "Prior 2: normalized A3/A4 outer"),
    ("violates_geometry_outer", "Prior 3: eps/PA local residual"),
]

SCENARIO_RE = re.compile(
    r"(?:huang2013|s4g)_(?P<depth>clean|wide|deep)_z(?P<z>\d{3})$"
)
DEPTH_ORDER = ("clean", "wide", "deep")
REDSHIFT_ORDER = ("005", "010", "020", "035", "050")


def scenario_key(campaign_dir: Path) -> str:
    """Return the short scenario tag (``clean_z005``, ``wide_z020``, …)."""
    m = SCENARIO_RE.match(campaign_dir.name)
    if not m:
        raise ValueError(f"unrecognized campaign dir name: {campaign_dir.name}")
    return f"{m.group('depth')}_z{m.group('z')}"


def collect_campaign_rows(
    campaign_dir: Path,
    scenario: str,
    dataset: str,
    tool: str,
) -> list[dict[str, Any]]:
    """Walk one campaign dir and return one row per (galaxy, arm).

    For each galaxy, the tool's default arm
    (:data:`DEFAULT_ARM_PER_TOOL`) is scored first; its metrics become
    the per-galaxy reference passed into every subsequent arm's
    :func:`classify_violations` call so the Prior 2 diagnostic
    (``prior2_regularization_induced``) can compare each arm's
    ``|A_n_norm|`` against the data-driven floor set by the default.
    """
    # Imported lazily to avoid a circular import at module load.
    from ..plotting.cross_tool_comparison import DEFAULT_ARM_PER_TOOL

    default_arm = DEFAULT_ARM_PER_TOOL.get(tool)
    rows: list[dict[str, Any]] = []
    for gdir in list_campaign_galaxies(campaign_dir, dataset):
        try:
            manifest = load_galaxy_manifest(gdir)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"[warn] {gdir.name}: manifest skipped ({exc})", file=sys.stderr)
            continue
        arm_ids = list_arms(gdir, tool)
        # Build the per-galaxy reference metrics first (if the default
        # arm ran successfully on this galaxy).
        reference_metrics: dict[str, Any] | None = None
        if default_arm is not None and default_arm in arm_ids:
            ref_profile = load_arm_profile(gdir, tool, default_arm)
            if ref_profile is not None:
                reference_metrics = compute_prior_metrics(ref_profile, manifest)
        for arm_id in arm_ids:
            profile = load_arm_profile(gdir, tool, arm_id)
            record = load_arm_record(gdir, tool, arm_id)
            status = (record or {}).get("status", "unknown")
            metrics = (
                compute_prior_metrics(profile, manifest)
                if profile is not None
                else {}
            )
            viol = classify_violations(
                metrics, reference_metrics=reference_metrics,
            )
            row: dict[str, Any] = {
                "scenario": scenario,
                "galaxy_id": manifest.galaxy_id,
                "galaxy_name": manifest.galaxy_name,
                "arm_id": arm_id,
                "tool": tool,
                "status": status,
                "image_size_pix": int(min(manifest.image_shape)),
                "psf_fwhm_pix": manifest.psf_fwhm_pix,
                "re_outer_pix": manifest.re_outer_pix,
                "effective_Re_pix": manifest.effective_Re_pix,
            }
            row.update(metrics)
            row.update(viol)
            rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], outfile: Path) -> None:
    if not rows:
        outfile.write_text("")
        return
    key_order: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                key_order.append(k)
    with outfile.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=key_order, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in key_order})


def _arm_scenario_matrix(
    rows: list[dict[str, Any]], column: str, scenarios: list[str]
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return ``(arm_order, rates[n_arms, n_scenarios], n[n_arms, n_scenarios])``.

    ``None`` flags (Prior 2 N/A when the arm did not fit harmonics)
    are excluded from both numerator and denominator of the cell rate.
    """
    per_key: dict[tuple[str, str], list[bool]] = defaultdict(list)
    arm_set: set[str] = set()
    for r in rows:
        if r.get("status") != "ok":
            continue
        arm = r["arm_id"]
        sc = r["scenario"]
        arm_set.add(arm)
        val = r.get(column)
        if val is None:
            continue
        per_key[(arm, sc)].append(bool(val))
    arm_order = sorted(arm_set)
    rates = np.full((len(arm_order), len(scenarios)), np.nan, dtype=float)
    counts = np.zeros((len(arm_order), len(scenarios)), dtype=int)
    for i, arm in enumerate(arm_order):
        for j, sc in enumerate(scenarios):
            vs = per_key.get((arm, sc), [])
            counts[i, j] = len(vs)
            if vs:
                rates[i, j] = float(np.mean(vs))
    return arm_order, rates, counts


def write_scenario_heatmap_pdf(
    rows: list[dict[str, Any]], outfile: Path,
) -> None:
    """One page per prior: arms x scenarios heatmap."""
    scenarios = sorted({r["scenario"] for r in rows},
                       key=lambda s: (DEPTH_ORDER.index(s.split("_z")[0]),
                                      REDSHIFT_ORDER.index(s.split("_z")[1])))
    with PdfPages(outfile) as pdf:
        for col, title in PRIOR_COLS:
            arm_order, rates, _ = _arm_scenario_matrix(rows, col, scenarios)
            if not arm_order:
                continue
            fig, ax = plt.subplots(figsize=(1.1 * len(scenarios) + 3.0,
                                            0.28 * len(arm_order) + 1.6))
            im = ax.imshow(rates, aspect="auto", cmap="magma_r", vmin=0, vmax=1)
            ax.set_xticks(range(len(scenarios)))
            ax.set_xticklabels(scenarios, rotation=30, ha="right", fontsize=8)
            ax.set_yticks(range(len(arm_order)))
            ax.set_yticklabels(arm_order, fontsize=8)
            ax.set_title(title)
            for i in range(len(arm_order)):
                for j in range(len(scenarios)):
                    val = rates[i, j]
                    if not np.isfinite(val):
                        continue
                    color = "white" if val > 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            color=color, fontsize=6)
            cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label("violation fraction")
            fig.tight_layout()
            pdf.savefig(fig, dpi=130)
            plt.close(fig)


def write_arm_ranking(
    rows: list[dict[str, Any]], outfile: Path,
) -> None:
    """Aggregate clean-fraction per arm (overall + per scenario)."""
    ok = [r for r in rows if r.get("status") == "ok"]
    prior_cols = [c for c, _ in PRIOR_COLS]

    # Overall (all scenarios pooled)
    per_arm_all: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in ok:
        per_arm_all[r["arm_id"]].append(r)

    def _rate(arm_rows: list[dict[str, Any]], col: str) -> float:
        """Mean over rows whose flag is not None (N/A excluded)."""
        pool = [bool(r[col]) for r in arm_rows
                if col in r and r[col] is not None]
        return float(np.mean(pool)) if pool else float("nan")

    def _is_clean(r: dict[str, Any]) -> bool:
        """Galaxy is clean when no applicable prior fires."""
        return not any(bool(r.get(c)) for c in prior_cols
                       if r.get(c) is not None)

    def _entries(arm_rows_map: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        out = []
        for arm, arm_rows in arm_rows_map.items():
            n = len(arm_rows)
            clean = sum(1 for r in arm_rows if _is_clean(r))
            rates = {c: _rate(arm_rows, c) for c in prior_cols}
            out.append({
                "arm_id": arm,
                "n": n,
                "n_clean": clean,
                "clean_frac": clean / n if n else 0.0,
                **{f"rate_{k}": v for k, v in rates.items()},
            })
        out.sort(key=lambda e: (-e["n_clean"], e["arm_id"]))
        return out

    overall = _entries(per_arm_all)

    # Per-scenario clean-fraction (for the arm-vs-scenario table)
    scenarios = sorted({r["scenario"] for r in ok},
                       key=lambda s: (DEPTH_ORDER.index(s.split("_z")[0]),
                                      REDSHIFT_ORDER.index(s.split("_z")[1])))
    per_arm_per_sc: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"n": 0, "n_clean": 0})
    )
    for r in ok:
        arm = r["arm_id"]
        sc = r["scenario"]
        per_arm_per_sc[arm][sc]["n"] += 1
        if _is_clean(r):
            per_arm_per_sc[arm][sc]["n_clean"] += 1

    with outfile.open("w") as h:
        h.write("# huang2013 cross-scenario arm ranking\n\n")
        h.write(
            "All nine `clean/wide/deep × z{005,020,035,050}` campaigns, "
            "isoster, overhauled three-prior metrics.\n\n"
        )

        h.write("## 1. Pooled ranking (9 scenarios × 93 galaxies = 837 galaxies/arm)\n\n")
        h.write(
            "| arm | clean_frac | n_clean/N | rate(drift) | rate(harm) | rate(geom) |\n"
            "|---|---|---|---|---|---|\n"
        )
        def _fmt(v: float) -> str:
            return f"{v:.2f}" if np.isfinite(v) else "N/A"
        for e in overall:
            h.write(
                f"| {e['arm_id']} | {e['clean_frac']:.3f} "
                f"| {e['n_clean']}/{e['n']} "
                f"| {_fmt(e['rate_violates_drift_any'])} "
                f"| {_fmt(e['rate_violates_harmonics_outer'])} "
                f"| {_fmt(e['rate_violates_geometry_outer'])} |\n"
            )

        h.write("\n## 2. Clean-fraction matrix (arm × scenario)\n\n")
        h.write("| arm | " + " | ".join(scenarios) + " | mean |\n")
        h.write("|" + "---|" * (len(scenarios) + 2) + "\n")
        for e in overall:
            arm = e["arm_id"]
            row = f"| {arm} "
            vals: list[float] = []
            for sc in scenarios:
                d = per_arm_per_sc[arm][sc]
                if d["n"] == 0:
                    row += "| - "
                else:
                    f = d["n_clean"] / d["n"]
                    vals.append(f)
                    row += f"| {f:.2f} "
            mean = float(np.mean(vals)) if vals else float("nan")
            row += f"| {mean:.2f} |\n"
            h.write(row)


def write_baseline_calibration(
    rows: list[dict[str, Any]], outfile: Path,
    baseline_arms: list[str] | None = None,
) -> None:
    """Per-scenario metric distributions on one or more baseline arms.

    Each baseline is emitted as its own section. Recommended pair:
    a regularization arm (top of the pooled ranking) together with the
    canonical ``ref_default`` to separate regularization-induced signal
    from the intrinsic noise floor.
    """
    if not baseline_arms:
        baseline_arms = ["reg_outer_damp", "ref_default"]

    metrics = [
        ("drift_iw_inner_pix", "drift_iw_inner [px]"),
        ("drift_iw_mid_pix", "drift_iw_mid [px]"),
        ("drift_iw_outer_pix", "drift_iw_outer [px]"),
        ("a4n_err_ratio_outer", "a4n err-ratio"),
        ("abs_a4n_median_outer", "|A4n| median"),
        ("max_local_resid_eps_outer", "max local resid eps"),
        ("max_local_resid_pa_outer_deg", "max local resid PA [deg]"),
    ]

    all_scenarios = sorted({r["scenario"] for r in rows
                            if r.get("status") == "ok"},
                           key=lambda s: (DEPTH_ORDER.index(s.split("_z")[0]),
                                          REDSHIFT_ORDER.index(s.split("_z")[1])))

    with outfile.open("w") as h:
        h.write("# Threshold-calibration baselines\n\n")
        h.write(
            "Per-scenario distributions of the Phase F.0 metrics on each "
            "baseline arm. Use these percentiles to calibrate the "
            "`DEFAULT_THRESHOLDS` entries in `scenario_summary.py`.\n\n"
            "Note: `harm_higher_orders__5_6` is not a valid Prior 2 "
            "baseline because isoster emits A3/A4 NaN beyond the seed "
            "isophote when 5-6 orders are enabled.\n\n"
        )
        for arm in baseline_arms:
            arm_rows = [r for r in rows
                        if r.get("status") == "ok" and r.get("arm_id") == arm]
            if not arm_rows:
                h.write(f"## `{arm}` — no rows\n\n")
                continue
            h.write(f"## `{arm}`\n\n")
            h.write("Per-scenario summary (p50 / p95 / max):\n\n")
            head = "| scenario | N |"
            sep = "|---|---|"
            for _, label in metrics:
                head += f" {label} |"
                sep += "---|"
            h.write(head + "\n" + sep + "\n")
            for sc in all_scenarios:
                scen_rows = [r for r in arm_rows if r["scenario"] == sc]
                line = f"| {sc} | {len(scen_rows)} |"
                for key, _ in metrics:
                    vals = np.array(
                        [float(r[key]) for r in scen_rows
                         if r.get(key) is not None and np.isfinite(
                             float(r.get(key, np.nan)))
                         ],
                        dtype=float,
                    )
                    if vals.size == 0:
                        line += " - |"
                    else:
                        line += (
                            f" {np.median(vals):.3g} / "
                            f"{np.percentile(vals, 95):.3g} / "
                            f"{np.max(vals):.3g} |"
                        )
                h.write(line + "\n")
            h.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cross_scenario_audit",
        description="Phase F.a cross-scenario audit of huang2013 campaigns.",
    )
    parser.add_argument(
        "--campaigns-root",
        type=Path,
        default=Path("/Volumes/galaxy/isophote/huang2013/_campaigns"),
        help="Directory that contains the huang2013_<depth>_z<z> campaigns.",
    )
    parser.add_argument("--dataset", default="huang2013")
    parser.add_argument("--tool", default="isoster")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output dir (default: <campaigns-root>/_analysis/cross_scenario_audit).",
    )
    parser.add_argument(
        "--baseline-arm",
        action="append",
        default=None,
        help=(
            "Arm(s) used for the calibration-baseline summary. Pass "
            "multiple times to emit side-by-side sections. Default pair: "
            "`reg_outer_damp` + `ref_default`. `harm_higher_orders__5_6` "
            "is not usable for Prior 2 (A3/A4 NaN beyond the seed "
            "isophote)."
        ),
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="Subset of scenario tags (clean_z005, wide_z020, ...). Default = all 9.",
    )
    args = parser.parse_args(argv)

    out_dir = args.out or (args.campaigns_root / "_analysis" / "cross_scenario_audit")
    out_dir.mkdir(parents=True, exist_ok=True)

    campaign_dirs = sorted(
        p for p in args.campaigns_root.iterdir()
        if p.is_dir() and SCENARIO_RE.match(p.name)
    )
    if args.scenarios:
        want = set(args.scenarios)
        campaign_dirs = [p for p in campaign_dirs if scenario_key(p) in want]
    if not campaign_dirs:
        print(f"no campaign dirs under {args.campaigns_root}", file=sys.stderr)
        return 2

    rows: list[dict[str, Any]] = []
    for cdir in campaign_dirs:
        sc = scenario_key(cdir)
        print(f"  collecting {sc} ...")
        rows.extend(
            collect_campaign_rows(cdir, sc, args.dataset, args.tool)
        )

    csv_path = out_dir / "cross_scenario_ideal_case_table.csv"
    write_csv(rows, csv_path)
    print(f"  wrote {csv_path} ({len(rows)} rows)")

    heat_path = out_dir / "cross_scenario_violations_heatmap.pdf"
    write_scenario_heatmap_pdf(rows, heat_path)
    print(f"  wrote {heat_path}")

    rank_path = out_dir / "cross_scenario_arm_ranking.md"
    write_arm_ranking(rows, rank_path)
    print(f"  wrote {rank_path}")

    base_path = out_dir / "cross_scenario_baseline_calibration.md"
    write_baseline_calibration(rows, base_path, baseline_arms=args.baseline_arm)
    print(f"  wrote {base_path}")

    print(f"\nDone. Artifacts under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
