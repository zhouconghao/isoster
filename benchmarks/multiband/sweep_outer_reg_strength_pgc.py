"""Outer-region damping strength × weights sweep on PGC006669.

Stage-3 Stage-B follow-up. The Stage-B real-data QA (journal entry
``2026-05-04_stage_b_outer_reg_damping.md``) showed that the all-
axes ``{1, 1, 1}`` damping default mirrors single-band but is
**too aggressive on multi-band targets with genuine outer-disc
geometry evolution**: on PGC006669 (barred galaxy), the outer
disc inherits the bar's PA / ellipticity. Three open questions
remain:

1. Is ``strength=2.0`` simply too high?
2. Should the default weights be ``{1, 0.5, 0.5}`` /
   ``{1, 0.25, 0.25}`` (partial damping on eps/pa) instead of
   ``{1, 1, 1}``?
3. Could ``{1, 0, 0}`` (center-only) be the right multi-band
   default since the joint gradient already constrains eps/pa?

This sweep runs 7 configurations on PGC006669 LegacySurvey grz to
empirically tune the multi-band default. Per the user's request,
the sweep down-weights PA and ellipticity rather than zeroing
them outright (between the all-axes failure mode and the center-
only escape hatch). Outputs:

- ``<out_dir>/PGC006669_outer_reg_strength_sweep.png`` — 4-row
  geometry-vs-SMA panel with one line per config (overlaid).
- ``<out_dir>/PGC006669_outer_reg_strength_sweep_stats.json`` —
  per-config outer-tail metrics + bias vs free baseline.
- ``<out_dir>/PGC006669_outer_reg_strength_sweep_<config>.fits``
  per config for downstream QA.
- Console summary table with bias and scatter metrics.

A *bias* metric (mean signed difference between damped outer
geometry and the free baseline's outer geometry) is the diagnostic
that the Stage-B journal said was missing — MAD alone reads
"pinned" and "tracked" identically. Bias is the right test of
"the damper is doing its job without over-pinning."

Stage-3 Stage-B follow-up (deferred from 2026-05-04 Stage-B
session; opened by the real-data QA findings on PGC006669).
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from isoster.multiband import (
    IsosterConfigMB,
    fit_image_multiband,
    isophote_results_mb_to_fits,
)

from legacysurvey_loader import (
    LEGACYSURVEY_ZP,
    asinh_softening_from_log10_match,
    load_legacysurvey_grz,
)


# (label, use_outer_reg, strength, weights). Order drives the legend
# and the comparison-panel line ordering.
SWEEP_CONFIGS: Tuple[Tuple[str, bool, float, Dict[str, float]], ...] = (
    ("baseline (off)", False, 0.0, {"center": 1.0, "eps": 1.0, "pa": 1.0}),
    ("center-only / s=2", True, 2.0, {"center": 1.0, "eps": 0.0, "pa": 0.0}),
    ("all-axes / s=2", True, 2.0, {"center": 1.0, "eps": 1.0, "pa": 1.0}),
    ("down-weight w=0.25 / s=2", True, 2.0, {"center": 1.0, "eps": 0.25, "pa": 0.25}),
    ("down-weight w=0.5  / s=2", True, 2.0, {"center": 1.0, "eps": 0.5, "pa": 0.5}),
    ("down-weight w=0.25 / s=1", True, 1.0, {"center": 1.0, "eps": 0.25, "pa": 0.25}),
    ("down-weight w=0.25 / s=4", True, 4.0, {"center": 1.0, "eps": 0.25, "pa": 0.25}),
)

# Color palette: baseline gets black; the rest cycle through a
# perceptually distinct palette so we can read all 7 lines at once.
SWEEP_COLORS = (
    "k", "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b",
)


def _build_cfg(
    bands: List[str], maxsma: float, *,
    use_outer_reg: bool, strength: float, weights: Dict[str, float],
) -> IsosterConfigMB:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return IsosterConfigMB(
            bands=bands, reference_band="r",
            harmonic_combination="joint",
            band_weights={b: 1.0 for b in bands},
            sma0=10.0, minsma=1.0, maxsma=maxsma,
            astep=0.10, linear_growth=False, debug=True,
            use_outer_center_regularization=use_outer_reg,
            outer_reg_sma_onset=40.0,  # match the existing PGC demo
            outer_reg_strength=strength,
            outer_reg_weights=weights,
        )


def _collect_geometry(result: dict) -> Dict[str, np.ndarray]:
    rows = result["isophotes"]
    return {
        "sma": np.array([float(r["sma"]) for r in rows]),
        "x0": np.array([float(r["x0"]) for r in rows]),
        "y0": np.array([float(r["y0"]) for r in rows]),
        "eps": np.array([float(r["eps"]) for r in rows]),
        "pa": np.array([float(r["pa"]) for r in rows]),
        "stop_code": np.array([int(r["stop_code"]) for r in rows]),
    }


def _outer_metrics(
    geom: Dict[str, np.ndarray],
    geom_baseline: Dict[str, np.ndarray],
    n_outer: int = 20,
    sma_onset: float = 40.0,
) -> dict:
    """Outer-region diagnostics:
       - eps / pa MAD (scatter — Stage-B's headline metric)
       - eps / pa BIAS vs free baseline (mean signed difference,
         the metric the Stage-B journal flagged as missing)
       - n_converged in the outer region (sma >= onset)
    """
    sma = geom["sma"]
    outer_mask = sma >= sma_onset
    n_outer_iso = int(outer_mask.sum())
    if n_outer_iso < 4:
        # Fallback: use last n_outer valid isophotes.
        valid = np.isfinite(geom["sma"])
        outer_mask = np.zeros_like(sma, dtype=bool)
        idx = np.where(valid)[0][-n_outer:]
        outer_mask[idx] = True

    def _mad(x: np.ndarray) -> float:
        finite = x[np.isfinite(x)]
        if finite.size < 4:
            return float("nan")
        return float(np.median(np.abs(finite - np.median(finite))))

    def _bias(x: np.ndarray, x0: np.ndarray) -> float:
        diff = x - x0
        finite = diff[np.isfinite(diff)]
        if finite.size < 4:
            return float("nan")
        return float(np.mean(finite))

    # Restrict baseline arrays to the same index set.
    sma_base = geom_baseline["sma"]
    n_match = min(sma.size, sma_base.size)
    common_outer = outer_mask[:n_match] & (sma_base[:n_match] >= sma_onset)

    eps_o = geom["eps"][:n_match][common_outer]
    pa_o = geom["pa"][:n_match][common_outer]
    eps_b = geom_baseline["eps"][:n_match][common_outer]
    pa_b = geom_baseline["pa"][:n_match][common_outer]

    return {
        "n_outer_isophotes": int(common_outer.sum()),
        "n_converged_outer": int(np.sum(geom["stop_code"][:n_match][common_outer] == 0)),
        "eps_mad": _mad(eps_o),
        "pa_mad": _mad(pa_o),
        # Bias relative to the unregularized baseline at the same SMA.
        # Bias near 0 = damper not pinning. Large negative or positive
        # bias = damper systematically biased the outer geometry.
        "eps_bias_vs_baseline": _bias(eps_o, eps_b),
        "pa_bias_vs_baseline": _bias(pa_o, pa_b),
    }


def _comparison_figure(
    geometries: Dict[str, Dict[str, np.ndarray]],
    out_path: Path,
    title_prefix: str,
    sma_onset: float,
) -> None:
    """4-row geometry-vs-SMA panel; all 7 configs overlaid per row.
    Vertical dashed line at the outer-reg sma_onset."""
    rows = ("x0", "y0", "eps", "pa")
    fig, axes = plt.subplots(
        len(rows), 1, figsize=(8.0, 9.5), sharex=True,
    )
    for row_idx, key in enumerate(rows):
        ax = axes[row_idx]
        for color, (label, geom) in zip(SWEEP_COLORS, geometries.items()):
            sma = geom["sma"]
            y = geom[key]
            lw = 1.6 if "baseline" in label else 1.0
            ax.plot(sma, y, lw=lw, color=color, alpha=0.9, label=label)
        ax.axvline(sma_onset, color="0.6", lw=0.8, ls=":", alpha=0.7,
                   zorder=-1)
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("SMA (px)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper right", frameon=False, fontsize=9,
        bbox_to_anchor=(0.99, 0.985),
    )
    fig.suptitle(title_prefix, fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 0.78, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(galaxy_dir: Path, galaxy_prefix: str, out_dir: Path) -> None:
    cutout = load_legacysurvey_grz(galaxy_dir, galaxy_prefix)
    bands = list(cutout.bands)
    masks_per_band = [cutout.combined_mask.copy() for _ in bands]
    maxsma = float(min(cutout.shape) * 0.45)

    out_dir.mkdir(parents=True, exist_ok=True)

    geometries: Dict[str, Dict[str, np.ndarray]] = {}
    metrics_by_label: Dict[str, dict] = {}
    fits_paths: Dict[str, str] = {}

    # Run baseline first; metrics for damper configs use it as the
    # reference for the bias calculation.
    print("=== Strength × weights sweep on PGC006669 ===")
    for label, on, strength, weights in SWEEP_CONFIGS:
        print(f"  fitting: {label}")
        cfg = _build_cfg(
            bands, maxsma,
            use_outer_reg=on, strength=strength, weights=weights,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = fit_image_multiband(
                cutout.images, masks_per_band, cfg,
                variance_maps=cutout.variances,
            )
        geometries[label] = _collect_geometry(res)
        slug = (
            label.replace(" ", "_").replace("(", "")
            .replace(")", "").replace("/", "_")
            .replace("=", "")
        )
        fits_path = out_dir / f"{galaxy_prefix}_outer_reg_strength_sweep_{slug}.fits"
        isophote_results_mb_to_fits(res, str(fits_path))
        fits_paths[label] = str(fits_path)

    baseline_geom = geometries["baseline (off)"]
    for label, geom in geometries.items():
        metrics_by_label[label] = _outer_metrics(
            geom, baseline_geom, sma_onset=40.0,
        )

    cmp_path = out_dir / f"{galaxy_prefix}_outer_reg_strength_sweep.png"
    _comparison_figure(
        geometries, cmp_path,
        title_prefix=(
            f"PGC006669 — outer_reg strength × weights sweep "
            f"(dotted line: outer_reg_sma_onset = 40 px)"
        ),
        sma_onset=40.0,
    )
    print(f"Wrote {cmp_path}")

    stats_path = out_dir / f"{galaxy_prefix}_outer_reg_strength_sweep_stats.json"
    stats_path.write_text(json.dumps({
        "galaxy_prefix": galaxy_prefix,
        "sweep_configs": [
            {"label": l, "use_outer_reg": on, "strength": s, "weights": w}
            for (l, on, s, w) in SWEEP_CONFIGS
        ],
        "outer_metrics": metrics_by_label,
        "fits_paths": fits_paths,
        "sma_onset_px": 40.0,
    }, indent=2))
    print(f"Wrote {stats_path}")

    print()
    print(
        f"{'config':<28} {'n_outer':>7} {'conv':>5} "
        f"{'eps MAD':>9} {'eps bias':>10} "
        f"{'pa MAD':>9} {'pa bias':>10}"
    )
    print("-" * 86)
    for label, _, _, _ in SWEEP_CONFIGS:
        m = metrics_by_label[label]
        print(
            f"{label:<28} {m['n_outer_isophotes']:>7} "
            f"{m['n_converged_outer']:>5} "
            f"{m['eps_mad']:>9.3e} {m['eps_bias_vs_baseline']:>10.3e} "
            f"{m['pa_mad']:>9.3e} {m['pa_bias_vs_baseline']:>10.3e}"
        )
    print()
    print(
        "bias = mean(damped_outer - baseline_outer) at the same SMAs above "
        "the onset.\n"
        "      bias ≈ 0 → damper not pinning; |bias| large → damper has "
        "biased the geometry."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--galaxy-dir",
        default=Path("/Volumes/galaxy/isophote/sga2020/data/demo/PGC006669"),
        type=Path,
    )
    parser.add_argument("--galaxy-prefix", default="PGC006669-largegalaxy")
    parser.add_argument(
        "--out-dir",
        default=Path("outputs/benchmark_multiband/outer_reg_strength_sweep_pgc"),
        type=Path,
    )
    args = parser.parse_args()
    run(args.galaxy_dir, args.galaxy_prefix, args.out_dir)


if __name__ == "__main__":
    main()
