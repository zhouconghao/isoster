"""Phase F.c.2 — cross-tool composite over the three per-tool audits.

Reads the CSVs emitted by :mod:`cross_scenario_audit` for isoster,
photutils, and autoprof (each produced by its own ``--tool`` run) and
writes four summary artifacts:

1. ``cross_tool_pooled_ranking.md`` — side-by-side top-k arms per tool
   with their pooled clean-fraction and per-prior rates.
2. ``cross_tool_best_arm_matrix.md`` — for every (scenario, tool), the
   best-scoring arm and its clean-fraction. Quick glance at which arm
   wins in which corner of the grid.
3. ``cross_tool_best_available.md`` — per scenario, the single
   best (arm, tool) across every tool; shows whether one tool
   dominates or whether different scenarios want different tools.
4. ``cross_tool_heatmap.pdf`` — 3 tools x 9 scenarios: cell value =
   clean-fraction of each tool's top arm. Companion to #2.

Prior 2 N/A handling: ``violates_harmonics_outer`` is serialized as an
empty string in the CSV when the defensive rule marked it not
applicable (e.g. every autoprof row). The loader reads that as ``None``
and this driver treats it the same way as the per-tool ranker — N/A
excluded from the clean-all tally and from the harm rate.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PRIOR_COLS = [
    ("violates_drift_any", "drift"),
    ("violates_harmonics_outer", "harm"),
    ("violates_geometry_outer", "geom"),
]

DEPTH_ORDER = ("clean", "wide", "deep")
REDSHIFT_ORDER = ("005", "010", "020", "035", "050")


def _scenario_sort_key(scenario: str) -> tuple[int, int]:
    depth, z = scenario.split("_z")
    return (DEPTH_ORDER.index(depth), REDSHIFT_ORDER.index(z))


def _parse_bool_or_none(s: str) -> bool | None:
    """Convert CSV cell to {True, False, None}. Empty string = N/A (None)."""
    if s is None:
        return None
    text = str(s).strip()
    if text == "" or text.lower() == "none":
        return None
    return text.lower() in ("true", "1", "yes")


def load_audit_rows(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rec: dict[str, Any] = dict(raw)
            for col, _ in PRIOR_COLS:
                rec[col] = _parse_bool_or_none(raw.get(col, ""))
            # Phase F.c.4 diagnostic, same tri-state (True / False / None)
            # as violates_harmonics_outer. Absent from pre-F.c.4 CSVs
            # where the loader will record None.
            rec["prior2_regularization_induced"] = _parse_bool_or_none(
                raw.get("prior2_regularization_induced", "")
            )
            rows.append(rec)
    return rows


def _rate_and_count(rows: list[dict[str, Any]], col: str) -> tuple[float, int]:
    pool = [bool(r[col]) for r in rows if r.get(col) is not None]
    if not pool:
        return float("nan"), 0
    return float(np.mean(pool)), len(pool)


def _is_clean(r: dict[str, Any]) -> bool:
    return not any(bool(r.get(c)) for c, _ in PRIOR_COLS if r.get(c) is not None)


def _harm_rate_excl_reg(rows: list[dict[str, Any]]) -> float:
    """Prior 2 violation rate after dropping rows flagged
    ``prior2_regularization_induced``. Used to surface the "defect"
    component of a regularization arm's harmonic signal separately
    from the expected component.
    """
    pool: list[bool] = []
    for r in rows:
        vh = r.get("violates_harmonics_outer")
        if vh is None:
            continue
        induced = r.get("prior2_regularization_induced")
        if induced is True:
            continue  # excluded by design
        pool.append(bool(vh))
    return float(np.mean(pool)) if pool else float("nan")


def _per_arm_entries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    per_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("status") != "ok":
            continue
        per_arm[r["arm_id"]].append(r)
    out: list[dict[str, Any]] = []
    for arm, arm_rows in per_arm.items():
        n = len(arm_rows)
        clean = sum(1 for r in arm_rows if _is_clean(r))
        rates = {c: _rate_and_count(arm_rows, c)[0] for c, _ in PRIOR_COLS}
        out.append({
            "arm_id": arm,
            "n": n,
            "n_clean": clean,
            "clean_frac": clean / n if n else 0.0,
            "harm_rate_excl_reg": _harm_rate_excl_reg(arm_rows),
            **{f"rate_{k}": v for k, v in rates.items()},
        })
    out.sort(key=lambda e: (-e["n_clean"], e["arm_id"]))
    return out


def _per_scenario_per_arm(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, int]]]:
    """scenario -> arm -> {'n', 'n_clean'}."""
    out: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"n": 0, "n_clean": 0})
    )
    for r in rows:
        if r.get("status") != "ok":
            continue
        sc = r["scenario"]
        arm = r["arm_id"]
        out[sc][arm]["n"] += 1
        if _is_clean(r):
            out[sc][arm]["n_clean"] += 1
    return out


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_pooled_ranking(
    tools: dict[str, list[dict[str, Any]]], outfile: Path, top_k: int = 8,
) -> None:
    entries = {t: _per_arm_entries(rows)[:top_k] for t, rows in tools.items()}
    with outfile.open("w") as h:
        h.write("# huang2013 cross-tool pooled ranking\n\n")
        h.write(
            "Top arms per tool, pooled across 9 scenarios x 93 galaxies = "
            "837 galaxies per arm. `rate(harm) = N/A` means the tool does "
            "not produce scorable A3/A4 errors (autoprof). "
            "`harm_excl_reg` excludes rows flagged "
            "`prior2_regularization_induced` (same galaxy's reference arm "
            "has |A_n_norm| below 0.02 AND this arm is >= 2x larger — "
            "the violation is expected from the arm's regularization, "
            "not a defect). Gap between `harm` and `harm_excl_reg` = "
            "fraction of the arm's Prior 2 hits that are by design.\n\n"
        )

        def _fmt(v: float) -> str:
            return f"{v:.2f}" if np.isfinite(v) else "N/A"

        for tool in ("isoster", "photutils", "autoprof"):
            if tool not in entries:
                continue
            h.write(f"## {tool} — top {top_k}\n\n")
            h.write(
                "| rank | arm | clean_frac | n_clean/N | drift | harm "
                "| harm_excl_reg | geom |\n"
                "|---|---|---|---|---|---|---|---|\n"
            )
            for i, e in enumerate(entries[tool], start=1):
                h.write(
                    f"| {i} | {e['arm_id']} | {e['clean_frac']:.3f} "
                    f"| {e['n_clean']}/{e['n']} "
                    f"| {_fmt(e['rate_violates_drift_any'])} "
                    f"| {_fmt(e['rate_violates_harmonics_outer'])} "
                    f"| {_fmt(e['harm_rate_excl_reg'])} "
                    f"| {_fmt(e['rate_violates_geometry_outer'])} |\n"
                )
            h.write("\n")


def write_best_arm_matrix(
    tools: dict[str, list[dict[str, Any]]], outfile: Path,
) -> None:
    """For each (tool, scenario), best arm by clean-fraction."""
    per_tool_sc = {
        t: _per_scenario_per_arm(rows) for t, rows in tools.items()
    }
    scenarios = sorted(
        {sc for per_sc in per_tool_sc.values() for sc in per_sc},
        key=_scenario_sort_key,
    )
    with outfile.open("w") as h:
        h.write("# huang2013 best arm per (tool, scenario)\n\n")
        h.write(
            "Cell format: `arm (clean_frac)` where clean_frac is the "
            "fraction of galaxies that pass all applicable priors for "
            "that arm on that scenario.\n\n"
        )
        h.write("| tool | " + " | ".join(scenarios) + " |\n")
        h.write("|---" * (len(scenarios) + 1) + "|\n")
        for tool in ("isoster", "photutils", "autoprof"):
            if tool not in per_tool_sc:
                continue
            line = f"| {tool} "
            for sc in scenarios:
                per_arm = per_tool_sc[tool].get(sc, {})
                if not per_arm:
                    line += "| - "
                    continue
                best_arm, best = max(
                    per_arm.items(),
                    key=lambda kv: (
                        kv[1]["n_clean"] / kv[1]["n"] if kv[1]["n"] else 0.0
                    ),
                )
                frac = best["n_clean"] / best["n"] if best["n"] else 0.0
                line += f"| `{best_arm}` ({frac:.2f}) "
            line += "|\n"
            h.write(line)


def write_best_available(
    tools: dict[str, list[dict[str, Any]]], outfile: Path,
) -> None:
    """Per scenario, the single best (tool, arm) across every tool."""
    per_tool_sc = {
        t: _per_scenario_per_arm(rows) for t, rows in tools.items()
    }
    scenarios = sorted(
        {sc for per_sc in per_tool_sc.values() for sc in per_sc},
        key=_scenario_sort_key,
    )
    with outfile.open("w") as h:
        h.write("# huang2013 best-available per scenario\n\n")
        h.write(
            "For each scenario, the single `(tool, arm)` with the highest "
            "clean-fraction across all three tools. The margin column is the "
            "gap between this winner and the second-best (any tool, any arm).\n\n"
        )
        h.write(
            "| scenario | winner_tool | winner_arm | clean_frac | runner-up | "
            "margin |\n|---|---|---|---|---|---|\n"
        )
        for sc in scenarios:
            # Collect all (tool, arm, clean_frac) triples.
            triples: list[tuple[str, str, float]] = []
            for tool, per_sc in per_tool_sc.items():
                for arm, stats in per_sc.get(sc, {}).items():
                    if stats["n"] == 0:
                        continue
                    triples.append((tool, arm, stats["n_clean"] / stats["n"]))
            if not triples:
                h.write(f"| {sc} | - | - | - | - | - |\n")
                continue
            triples.sort(key=lambda t: -t[2])
            w_tool, w_arm, w_frac = triples[0]
            if len(triples) >= 2:
                r_tool, r_arm, r_frac = triples[1]
                runner = f"{r_tool}/`{r_arm}` ({r_frac:.2f})"
                margin = f"{w_frac - r_frac:.2f}"
            else:
                runner = "-"
                margin = "-"
            h.write(
                f"| {sc} | {w_tool} | `{w_arm}` | {w_frac:.2f} "
                f"| {runner} | {margin} |\n"
            )


def write_heatmap_pdf(
    tools: dict[str, list[dict[str, Any]]], outfile: Path,
) -> None:
    """3 rows (tool) x 9 cols (scenario). Cell = each tool's best arm
    clean-fraction at that scenario.
    """
    per_tool_sc = {
        t: _per_scenario_per_arm(rows) for t, rows in tools.items()
    }
    scenarios = sorted(
        {sc for per_sc in per_tool_sc.values() for sc in per_sc},
        key=_scenario_sort_key,
    )
    tool_order = [t for t in ("isoster", "photutils", "autoprof") if t in tools]
    grid = np.full((len(tool_order), len(scenarios)), np.nan, dtype=float)
    best_arms = np.full_like(grid, fill_value=0, dtype=object)
    for i, tool in enumerate(tool_order):
        for j, sc in enumerate(scenarios):
            per_arm = per_tool_sc.get(tool, {}).get(sc, {})
            if not per_arm:
                continue
            best_arm, best = max(
                per_arm.items(),
                key=lambda kv: (
                    kv[1]["n_clean"] / kv[1]["n"] if kv[1]["n"] else 0.0
                ),
            )
            grid[i, j] = (
                best["n_clean"] / best["n"] if best["n"] else 0.0
            )
            best_arms[i, j] = best_arm

    fig, ax = plt.subplots(
        figsize=(1.2 * len(scenarios) + 3.0, 0.8 * len(tool_order) + 2.4)
    )
    im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(tool_order)))
    ax.set_yticklabels(tool_order, fontsize=11)
    ax.set_title(
        "huang2013 cross-tool: best arm's clean-fraction by (tool, scenario)"
    )
    for i in range(len(tool_order)):
        for j in range(len(scenarios)):
            val = grid[i, j]
            if not np.isfinite(val):
                continue
            color = "white" if val < 0.55 else "black"
            arm = str(best_arms[i, j])
            # Compact the arm label if long.
            arm_short = arm if len(arm) <= 18 else arm[:16] + ".."
            ax.text(
                j, i, f"{val:.2f}\n{arm_short}",
                ha="center", va="center", color=color, fontsize=7,
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("best-arm clean-fraction")
    fig.tight_layout()
    fig.savefig(outfile, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cross_tool_composite",
        description="Build a cross-tool composite over three per-tool "
        "cross_scenario_audit outputs.",
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=Path("/Volumes/galaxy/isophote/huang2013/_campaigns/_analysis"),
        help="Directory containing cross_scenario_audit_<tool>/ subdirs.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output dir (default: <analysis-root>/cross_tool_composite).",
    )
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args(argv)

    tools: dict[str, list[dict[str, Any]]] = {}
    for tool in ("isoster", "photutils", "autoprof"):
        csv_path = (
            args.analysis_root
            / f"cross_scenario_audit_{tool}"
            / "cross_scenario_ideal_case_table.csv"
        )
        if csv_path.is_file():
            tools[tool] = load_audit_rows(csv_path)
            print(f"  loaded {tool}: {len(tools[tool])} rows")
        else:
            print(f"  [skip] {tool}: {csv_path} missing")

    if not tools:
        print("no per-tool audits found; aborting")
        return 2

    out_dir = args.out or (args.analysis_root / "cross_tool_composite")
    out_dir.mkdir(parents=True, exist_ok=True)

    write_pooled_ranking(tools, out_dir / "cross_tool_pooled_ranking.md",
                         top_k=args.top_k)
    print(f"  wrote {out_dir / 'cross_tool_pooled_ranking.md'}")

    write_best_arm_matrix(tools, out_dir / "cross_tool_best_arm_matrix.md")
    print(f"  wrote {out_dir / 'cross_tool_best_arm_matrix.md'}")

    write_best_available(tools, out_dir / "cross_tool_best_available.md")
    print(f"  wrote {out_dir / 'cross_tool_best_available.md'}")

    write_heatmap_pdf(tools, out_dir / "cross_tool_heatmap.pdf")
    print(f"  wrote {out_dir / 'cross_tool_heatmap.pdf'}")

    print(f"\nDone. Artifacts under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
