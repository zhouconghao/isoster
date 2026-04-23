"""Phase F.0 — clean_z005 ideal-case audit driver.

Aggregates per-(galaxy, arm) prior-violation metrics across the 93
huang2013 ``clean_z005`` scenarios and emits:

1. ``clean_z005_ideal_case_table.csv`` — long-form metric table.
2. ``clean_z005_violations_heatmap.pdf`` — arms x priors,
   cell = fraction of galaxies violating the prior.
3. ``clean_z005_worst_cases.pdf`` — top-N worst violators per prior,
   stacked QA panels (SB + geometry + harmonics + center trace).
4. ``clean_z005_arm_shortlist.md`` — arms clearing all three priors
   on >= min_clean_fraction of galaxies.
5. ``clean_z005_worst_cases_summary.md`` — readable dossier with QA
   PNG paths for direct inspection.

Runs against a campaign directory like
``/Volumes/galaxy/isophote/huang2013/_campaigns/huang2013_clean_z005``.
Output lands under ``<campaign_root>/_analysis/clean_z005_audit/``
unless overridden via ``--out``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
from astropy.table import Table

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .scenario_summary import (
    GalaxyManifest,
    classify_violations,
    compute_prior_metrics,
    list_arms,
    list_campaign_galaxies,
    load_arm_profile,
    load_arm_record,
    load_galaxy_manifest,
    sma_zones,
)


# Prior labels shown in the heatmap / shortlist tables.
PRIOR_LABELS = [
    ("violates_drift_any", "1. IW drift (any zone)"),
    ("violates_harmonics_outer", "2. norm. A3/A4 outer"),
    ("violates_geometry_outer", "3. eps/PA local resid"),
]

PRIOR_SHORT = {
    "violates_drift_any": "drift",
    "violates_harmonics_outer": "harm",
    "violates_geometry_outer": "geom",
}


def collect_audit_table(
    campaign_dir: Path,
    dataset: str = "huang2013",
    tool: str = "isoster",
) -> list[dict[str, Any]]:
    """Walk every galaxy x arm under the campaign dir and build a long table."""
    rows: list[dict[str, Any]] = []
    for gdir in list_campaign_galaxies(campaign_dir, dataset):
        try:
            manifest = load_galaxy_manifest(gdir)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"[warn] {gdir.name}: manifest skipped ({exc})", file=sys.stderr)
            continue
        for arm_id in list_arms(gdir, tool):
            profile = load_arm_profile(gdir, tool, arm_id)
            record = load_arm_record(gdir, tool, arm_id)
            status = (record or {}).get("status", "unknown")
            metrics = (
                compute_prior_metrics(profile, manifest)
                if profile is not None
                else {}
            )
            viol = classify_violations(metrics)
            row: dict[str, Any] = {
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


def write_audit_csv(rows: list[dict[str, Any]], outfile: Path) -> None:
    """Dump long-form rows as CSV. Columns = union of keys, stable-ordered."""
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


def _aggregate_violation_rates(
    rows: list[dict[str, Any]],
) -> tuple[list[str], np.ndarray, dict[str, int]]:
    """Return (arm_order, rates[n_arms, n_priors], n_galaxies_per_arm)."""
    prior_cols = [c for c, _ in PRIOR_LABELS]
    per_arm_viol: dict[str, list[list[bool]]] = defaultdict(
        lambda: [[] for _ in PRIOR_LABELS]
    )
    for r in rows:
        if r.get("status") != "ok":
            continue
        arm = r["arm_id"]
        for j, col in enumerate(prior_cols):
            per_arm_viol[arm][j].append(bool(r.get(col, False)))
    arm_order = sorted(per_arm_viol.keys())
    rates = np.zeros((len(arm_order), len(PRIOR_LABELS)), dtype=float)
    n_per_arm: dict[str, int] = {}
    for i, arm in enumerate(arm_order):
        for j in range(len(PRIOR_LABELS)):
            vs = per_arm_viol[arm][j]
            rates[i, j] = float(np.mean(vs)) if vs else float("nan")
        n_per_arm[arm] = len(per_arm_viol[arm][0])
    return arm_order, rates, n_per_arm


def write_violations_heatmap(
    rows: list[dict[str, Any]], outfile: Path,
) -> None:
    """Cell = fraction of galaxies (per arm) that violate each prior."""
    arm_order, rates, n_per_arm = _aggregate_violation_rates(rows)
    if not arm_order:
        return
    n_gal_ref = max(n_per_arm.values()) if n_per_arm else 0

    fig, ax = plt.subplots(figsize=(7.2, 0.28 * len(arm_order) + 1.4))
    im = ax.imshow(rates, aspect="auto", cmap="magma_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(PRIOR_LABELS)))
    ax.set_xticklabels([label for _, label in PRIOR_LABELS], rotation=30, ha="right")
    ax.set_yticks(range(len(arm_order)))
    ax.set_yticklabels(arm_order, fontsize=8)
    ax.set_xlabel("Prior")
    ax.set_title(
        f"clean_z005: fraction of galaxies violating each prior\n"
        f"({n_gal_ref} galaxies per arm, status=ok only)"
    )
    for i, _ in enumerate(arm_order):
        for j in range(len(PRIOR_LABELS)):
            val = rates[i, j]
            if not np.isfinite(val):
                continue
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("violation fraction")
    fig.tight_layout()
    fig.savefig(outfile, dpi=130)
    plt.close(fig)


def write_worst_cases_pdf(
    rows: list[dict[str, Any]],
    campaign_dir: Path,
    dataset: str,
    tool: str,
    outfile: Path,
    top_n: int = 3,
) -> None:
    """For each prior, pick top_n worst (arm, galaxy) and render QA strips."""
    ok_rows = [r for r in rows if r.get("status") == "ok"]

    def _top_by(column: str, label: str) -> list[tuple[str, str, str, float]]:
        scored = [
            (r, float(r[column])) for r in ok_rows
            if r.get(column) is not None and np.isfinite(float(r.get(column, np.nan)))
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            (label, r["galaxy_id"], r["arm_id"], v)
            for r, v in scored[:top_n]
        ]

    picks: list[tuple[str, str, str, float]] = []
    picks += _top_by("drift_iw_outer_pix", "drift outer (IW)")
    picks += _top_by("drift_iw_mid_pix", "drift mid (IW)")
    picks += _top_by("a4n_err_ratio_outer", "|A4n|/err outer")
    picks += _top_by("abs_a4n_median_outer", "|A4n| median outer")
    picks += _top_by("max_local_resid_pa_outer_deg", "PA local resid")
    picks += _top_by("max_local_resid_eps_outer", "eps local resid")

    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(outfile) as pdf:
        for prior, gid, arm_id, value in picks:
            gdir = Path(campaign_dir) / dataset / gid.replace("/", "__")
            try:
                manifest = load_galaxy_manifest(gdir)
            except (FileNotFoundError, KeyError, ValueError):
                continue
            profile = load_arm_profile(gdir, tool, arm_id)
            if profile is None:
                continue
            _render_worst_case_page(
                pdf, prior, gid, arm_id, value, manifest, profile
            )


def _render_worst_case_page(
    pdf,
    prior: str,
    galaxy_id: str,
    arm_id: str,
    value: float,
    manifest: GalaxyManifest,
    profile: Table,
) -> None:
    ok = np.asarray(profile["stop_code"]) == 0
    sma = np.asarray(profile["sma"], dtype=float)
    x0 = np.asarray(profile["x0"], dtype=float)
    y0 = np.asarray(profile["y0"], dtype=float)
    eps = np.asarray(profile["eps"], dtype=float)
    pa = np.degrees(np.asarray(profile["pa"], dtype=float))
    intens = np.asarray(profile["intens"], dtype=float)
    a3 = np.asarray(profile["a3"], dtype=float)
    a4 = np.asarray(profile["a4"], dtype=float)
    drift = np.sqrt((x0 - manifest.true_center[0]) ** 2
                    + (y0 - manifest.true_center[1]) ** 2)

    zones = sma_zones(profile, manifest, ok_mask=ok)
    s_in_end = zones.inner_end
    s_mid_end = zones.mid_end

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    fig.suptitle(
        f"worst {prior}: {galaxy_id} / {arm_id} "
        f"(value={value:.3g})",
        fontsize=11,
    )
    # intens profile
    ax = axes[0, 0]
    valid_i = ok & (intens > 0)
    ax.loglog(sma[valid_i], intens[valid_i], "k-", lw=0.8)
    ax.loglog(sma[valid_i], intens[valid_i], "k.", ms=2)
    ax.axvspan(0, s_in_end, alpha=0.08, color="C0", label="inner")
    ax.axvspan(s_in_end, s_mid_end, alpha=0.08, color="green", label="mid")
    ax.axvspan(s_mid_end, sma.max(), alpha=0.08, color="orange", label="outer")
    ax.set_xlabel("sma [pix]"); ax.set_ylabel("intens"); ax.set_title("SB profile")
    ax.legend(fontsize=7, loc="lower left")
    # eps
    ax = axes[0, 1]
    ax.plot(sma[ok], eps[ok], "b.-", ms=3, lw=0.7)
    ax.axvspan(s_mid_end, sma.max(), alpha=0.08, color="orange")
    ax.set_xlabel("sma [pix]"); ax.set_ylabel("eps"); ax.set_title("ellipticity")
    # pa
    ax = axes[0, 2]
    ax.plot(sma[ok], pa[ok], "b.-", ms=3, lw=0.7)
    ax.axvspan(s_mid_end, sma.max(), alpha=0.08, color="orange")
    ax.set_xlabel("sma [pix]"); ax.set_ylabel("pa [deg]"); ax.set_title("position angle")
    # drift
    ax = axes[1, 0]
    ax.plot(sma[ok], drift[ok], "r.-", ms=3, lw=0.7)
    ax.axhline(0.5, color="k", ls="--", lw=0.6, label="inner/mid thresh 0.5 px")
    ax.axhline(2.0, color="gray", ls="--", lw=0.6, label="outer thresh 2 px")
    ax.set_xlabel("sma [pix]"); ax.set_ylabel("drift [pix]"); ax.set_title("center drift (per-iso)")
    ax.legend(fontsize=7)
    # a3, a4
    ax = axes[1, 1]
    ax.plot(sma[ok], a3[ok], "g.-", ms=3, lw=0.7, label="a3")
    ax.plot(sma[ok], a4[ok], "m.-", ms=3, lw=0.7, label="a4")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvspan(s_mid_end, sma.max(), alpha=0.08, color="orange")
    ax.set_xlabel("sma [pix]"); ax.set_ylabel("harmonic"); ax.set_title("A3 / A4 (raw)")
    ax.legend(fontsize=7)
    # center trace in xy
    ax = axes[1, 2]
    ax.plot(x0[ok] - manifest.true_center[0], y0[ok] - manifest.true_center[1],
            "k.-", ms=3, lw=0.7)
    ax.axhline(0, color="gray", lw=0.3); ax.axvline(0, color="gray", lw=0.3)
    ax.set_aspect("equal")
    ax.set_xlabel("dx [pix]"); ax.set_ylabel("dy [pix]"); ax.set_title("center trace")
    fig.tight_layout()
    pdf.savefig(fig, dpi=110)
    plt.close(fig)


def write_worst_cases_summary(
    rows: list[dict[str, Any]],
    outfile: Path,
    campaign_dir: Path,
    dataset: str,
    tool: str,
    top_n: int = 10,
    default_arm: str = "ref_default",
) -> None:
    """Write a readable markdown dossier of worst-offending galaxies."""
    ok_rows = [r for r in rows if r.get("status") == "ok"]

    def _f(r: dict[str, Any], key: str) -> float:
        v = r.get(key)
        try:
            out = float(v) if v not in (None, "") else float("nan")
        except (TypeError, ValueError):
            return float("nan")
        return out

    def _top_n(col: str, n: int, arms: set[str] | None = None) -> list[dict[str, Any]]:
        pool = [r for r in ok_rows if np.isfinite(_f(r, col))]
        if arms is not None:
            pool = [r for r in pool if r["arm_id"] in arms]
        pool.sort(key=lambda r: -_f(r, col))
        return pool[:n]

    def _qa_path(galaxy_id: str, arm_id: str) -> str:
        safe = galaxy_id.replace("/", "__")
        return (
            f"{campaign_dir}/{dataset}/{safe}/{tool}/arms/{arm_id}/qa.png"
        )

    gal_agg: dict[str, dict[str, int]] = defaultdict(
        lambda: {"drift": 0, "harm": 0, "geom": 0, "n": 0}
    )
    for r in ok_rows:
        d = gal_agg[r["galaxy_id"]]
        d["n"] += 1
        if bool(r.get("violates_drift_any", False)):
            d["drift"] += 1
        if bool(r.get("violates_harmonics_outer", False)):
            d["harm"] += 1
        if bool(r.get("violates_geometry_outer", False)):
            d["geom"] += 1
    gal_rank = sorted(
        gal_agg.items(),
        key=lambda x: -(x[1]["drift"] + x[1]["harm"] + x[1]["geom"]),
    )[:top_n]

    with outfile.open("w") as h:
        h.write("# clean_z005 worst-case dossier (overhauled metrics)\n\n")
        h.write(
            "Companion to `clean_z005_ideal_case_table.csv`, "
            "`clean_z005_violations_heatmap.pdf`, and "
            "`clean_z005_worst_cases.pdf`. This file lists the galaxies + "
            "arms whose QA PNGs are worth opening first.\n\n"
        )
        h.write(
            f"- Dataset: `{dataset}`, tool: `{tool}`\n"
            f"- Campaign: `{campaign_dir}`\n"
            f"- Default arm used for per-arm baselines: `{default_arm}`\n\n"
        )
        h.write("## Metric definitions\n\n")
        h.write(
            "- **Prior 1 drift_iw_{inner,mid,outer}_pix**: intensity-weighted "
            "|(xbar, ybar) - true_center| over `stop_code==0` isophotes in the "
            "residual_zones inner (< 0.5 R_ref), mid ([0.5, 2) R_ref), outer "
            "(>= 2 R_ref) bins. Thresholds: 0.5 px (inner/mid), 2 px (outer).\n"
            "- **Prior 2 a{3,4}n_err_ratio_outer + abs_{a,b}{3,4}n_median_outer**: "
            "median |A_n_norm|/err(A_n_norm) AND median |A_n_norm| over the outer "
            "zone, where A_n_norm = -A_n / (a * dI/da). Gradient uses the shipped "
            "`grad` column with an on-the-fly finite-difference fallback; points "
            "near turnovers (|a*dI/da| below 0.1% of its peak) are masked. "
            "Violation requires err-ratio > 3 sigma AND |A_n_norm| median > 0.01 "
            "(tiny harmonics with optimistic formal errors do not trip the prior).\n"
            "- **Prior 3 max_local_resid_{eps,pa}_outer**: peak 3-point local "
            "residual of eps / pa over consecutive outer-zone isophote triplets. "
            "PA differences wrapped to (-pi/2, pi/2]. Thresholds: 0.05 for eps, "
            "8 deg for PA.\n\n"
        )
        h.write("QA figure path template:\n")
        h.write(f"    {campaign_dir}/{dataset}/<galaxy>__clean_z005/{tool}/arms/<arm>/qa.png\n\n")

        h.write("## 1. Galaxy-level top offenders\n\n")
        h.write(
            "Total prior violations across all arms (max = 3 priors x N arms).\n\n"
        )
        h.write(
            "| rank | galaxy | total | drift | harm | geom |\n"
            "|---|---|---|---|---|---|\n"
        )
        for i, (gid, d) in enumerate(gal_rank, start=1):
            total = d["drift"] + d["harm"] + d["geom"]
            h.write(
                f"| {i} | {gid} | {total}/{3*d['n']} "
                f"| {d['drift']}/{d['n']} | {d['harm']}/{d['n']} "
                f"| {d['geom']}/{d['n']} |\n"
            )

        h.write("\n## 2. Prior 1 — intensity-weighted centroid drift\n\n")
        h.write(f"### 2a. Worst outer-zone drift on default arm (`{default_arm}`)\n\n")
        h.write("| galaxy | drift_outer_iw [px] | drift_mid_iw [px] | QA figure |\n|---|---|---|---|\n")
        for r in _top_n("drift_iw_outer_pix", top_n, arms={default_arm}):
            h.write(
                f"| {r['galaxy_id']} | {_f(r,'drift_iw_outer_pix'):.2f} "
                f"| {_f(r,'drift_iw_mid_pix'):.2f} "
                f"| `{_qa_path(r['galaxy_id'], r['arm_id'])}` |\n"
            )
        h.write("\n### 2b. Worst outer-zone drift across every arm\n\n")
        h.write("| galaxy | arm | drift_outer_iw [px] | drift_mid_iw [px] | QA figure |\n|---|---|---|---|---|\n")
        for r in _top_n("drift_iw_outer_pix", top_n):
            h.write(
                f"| {r['galaxy_id']} | {r['arm_id']} "
                f"| {_f(r,'drift_iw_outer_pix'):.2f} "
                f"| {_f(r,'drift_iw_mid_pix'):.2f} "
                f"| `{_qa_path(r['galaxy_id'], r['arm_id'])}` |\n"
            )

        h.write("\n## 3. Prior 2 — outer normalized harmonics beyond formal errors\n\n")
        reg_arms = {"reg_outer_damp", "reg_outer_strong", "stack_all"}
        h.write("### 3a. Worst |A4n|/err (regularization arms)\n\n")
        h.write(
            "| galaxy | arm | A4n/err | |A4n| median | QA figure |\n"
            "|---|---|---|---|---|\n"
        )
        for r in _top_n("a4n_err_ratio_outer", top_n, arms=reg_arms):
            h.write(
                f"| {r['galaxy_id']} | {r['arm_id']} "
                f"| {_f(r,'a4n_err_ratio_outer'):.2f} "
                f"| {_f(r,'abs_a4n_median_outer'):.4f} "
                f"| `{_qa_path(r['galaxy_id'], r['arm_id'])}` |\n"
            )
        h.write("\n### 3b. Worst |A4n|/err (any arm)\n\n")
        h.write(
            "| galaxy | arm | A4n/err | |A4n| median | QA figure |\n"
            "|---|---|---|---|---|\n"
        )
        for r in _top_n("a4n_err_ratio_outer", top_n):
            h.write(
                f"| {r['galaxy_id']} | {r['arm_id']} "
                f"| {_f(r,'a4n_err_ratio_outer'):.2f} "
                f"| {_f(r,'abs_a4n_median_outer'):.4f} "
                f"| `{_qa_path(r['galaxy_id'], r['arm_id'])}` |\n"
            )
        h.write("\n### 3c. Worst |A3n|/err (any arm)\n\n")
        h.write(
            "| galaxy | arm | A3n/err | QA figure |\n"
            "|---|---|---|---|\n"
        )
        for r in _top_n("a3n_err_ratio_outer", top_n):
            h.write(
                f"| {r['galaxy_id']} | {r['arm_id']} "
                f"| {_f(r,'a3n_err_ratio_outer'):.2f} "
                f"| `{_qa_path(r['galaxy_id'], r['arm_id'])}` |\n"
            )

        h.write("\n## 4. Prior 3 — outer 3-point local residual\n\n")
        h.write("### 4a. Worst PA local residual (outer, deg)\n\n")
        h.write(
            "| galaxy | arm | max PA local resid [deg] | QA figure |\n"
            "|---|---|---|---|\n"
        )
        for r in _top_n("max_local_resid_pa_outer_deg", top_n):
            h.write(
                f"| {r['galaxy_id']} | {r['arm_id']} "
                f"| {_f(r,'max_local_resid_pa_outer_deg'):.2f} "
                f"| `{_qa_path(r['galaxy_id'], r['arm_id'])}` |\n"
            )
        h.write("\n### 4b. Worst eps local residual (outer)\n\n")
        h.write(
            "| galaxy | arm | max eps local resid | QA figure |\n"
            "|---|---|---|---|\n"
        )
        for r in _top_n("max_local_resid_eps_outer", top_n):
            h.write(
                f"| {r['galaxy_id']} | {r['arm_id']} "
                f"| {_f(r,'max_local_resid_eps_outer'):.3f} "
                f"| `{_qa_path(r['galaxy_id'], r['arm_id'])}` |\n"
            )


def write_arm_shortlist(rows: list[dict[str, Any]], outfile: Path,
                        min_clean_fraction: float = 90 / 93) -> None:
    """Emit ``clean_z005_arm_shortlist.md`` listing arms that clear all priors."""
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    prior_cols = [c for c, _ in PRIOR_LABELS]

    per_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in ok_rows:
        per_arm[r["arm_id"]].append(r)

    entries: list[dict[str, Any]] = []
    for arm_id, arm_rows in per_arm.items():
        n = len(arm_rows)
        viol_rates = {
            col: float(np.mean([bool(r.get(col, False)) for r in arm_rows]))
            if n else float("nan")
            for col in prior_cols
        }
        n_clean_all = sum(
            1 for r in arm_rows
            if not any(bool(r.get(col, False)) for col in prior_cols)
        )
        entries.append({
            "arm_id": arm_id,
            "n_galaxies": n,
            "n_clean_all": n_clean_all,
            "clean_fraction": n_clean_all / n if n else 0.0,
            "drift_rate": viol_rates["violates_drift_any"],
            "harmonics_rate": viol_rates["violates_harmonics_outer"],
            "geometry_rate": viol_rates["violates_geometry_outer"],
        })
    entries.sort(key=lambda e: (-e["n_clean_all"], e["arm_id"]))
    n_max = max((e["n_galaxies"] for e in entries), default=0)
    threshold_n = int(round(min_clean_fraction * n_max))

    with outfile.open("w") as handle:
        handle.write("# clean_z005 arm shortlist (overhauled metrics)\n\n")
        handle.write(
            f"Threshold: arm must pass all three priors on "
            f">= {min_clean_fraction * 100:.1f}% of galaxies "
            f"({threshold_n} of {n_max}).\n\n"
        )
        handle.write("## Full ranking\n\n")
        handle.write(
            "| arm | clean_fraction | n_clean/N | drift | harm | geom |\n"
            "|---|---|---|---|---|---|\n"
        )
        for e in entries:
            handle.write(
                f"| {e['arm_id']} | {e['clean_fraction']:.3f} "
                f"| {int(e['n_clean_all'])}/{int(e['n_galaxies'])} "
                f"| {e['drift_rate']:.2f} | {e['harmonics_rate']:.2f} "
                f"| {e['geometry_rate']:.2f} |\n"
            )
        handle.write("\n## Arms at or above shortlist threshold\n\n")
        short = [e for e in entries if e["clean_fraction"] >= min_clean_fraction]
        if not short:
            handle.write("(none)\n")
        else:
            for e in short:
                handle.write(
                    f"- **{e['arm_id']}** — clean on "
                    f"{int(e['n_clean_all'])}/{int(e['n_galaxies'])} galaxies "
                    f"({e['clean_fraction'] * 100:.1f}%)\n"
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="clean_z005_audit",
        description="Phase F.0 ideal-case audit of huang2013/clean_z005.",
    )
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=Path("/Volumes/galaxy/isophote/huang2013/_campaigns/huang2013_clean_z005"),
        help="Campaign directory (contains <dataset>/<galaxy_id>/...).",
    )
    parser.add_argument("--dataset", default="huang2013")
    parser.add_argument("--tool", default="isoster")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <campaign-dir>/../_analysis/clean_z005_audit).",
    )
    parser.add_argument(
        "--top-n", type=int, default=3,
        help="Number of worst cases per prior in the QA PDF.",
    )
    parser.add_argument(
        "--top-n-summary", type=int, default=10,
        help="Number of rows per table in the worst-cases markdown summary.",
    )
    parser.add_argument(
        "--min-clean-fraction", type=float, default=90.0 / 93.0,
        help="Shortlist threshold (default 90/93 ~= 0.968).",
    )
    args = parser.parse_args(argv)

    out_dir = args.out or (
        args.campaign_dir.parent / "_analysis" / "clean_z005_audit"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting audit rows for {args.campaign_dir} / {args.dataset} / {args.tool} ...")
    rows = collect_audit_table(args.campaign_dir, dataset=args.dataset, tool=args.tool)
    csv_path = out_dir / "clean_z005_ideal_case_table.csv"
    write_audit_csv(rows, csv_path)
    print(f"  wrote {csv_path} ({len(rows)} rows)")

    heatmap_path = out_dir / "clean_z005_violations_heatmap.pdf"
    write_violations_heatmap(rows, heatmap_path)
    print(f"  wrote {heatmap_path}")

    worst_path = out_dir / "clean_z005_worst_cases.pdf"
    write_worst_cases_pdf(
        rows, args.campaign_dir, args.dataset, args.tool, worst_path, top_n=args.top_n,
    )
    print(f"  wrote {worst_path}")

    shortlist_path = out_dir / "clean_z005_arm_shortlist.md"
    write_arm_shortlist(rows, shortlist_path, min_clean_fraction=args.min_clean_fraction)
    print(f"  wrote {shortlist_path}")

    summary_path = out_dir / "clean_z005_worst_cases_summary.md"
    write_worst_cases_summary(
        rows, summary_path,
        campaign_dir=args.campaign_dir,
        dataset=args.dataset,
        tool=args.tool,
        top_n=args.top_n_summary,
    )
    print(f"  wrote {summary_path}")

    print(f"\nDone. Artifacts under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
