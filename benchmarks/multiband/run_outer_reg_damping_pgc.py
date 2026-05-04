"""Outer-region damping demo on PGC006669 (LegacySurvey grz).

Sweeps three damping configurations on the same PGC006669 cutout and
writes:

- ``<out_dir>/PGC006669_outer_reg_<config>.fits`` per config.
- ``<out_dir>/PGC006669_outer_reg_<config>.png`` per-config QA panel.
- ``<out_dir>/PGC006669_outer_reg_compare.png`` cross-config geometry
  (x0, y0, eps, pa) vs SMA panel — outer-reg's primary effect surface.
- ``<out_dir>/PGC006669_outer_reg_stats.json`` per-config summary.

Configurations match the asteris demo:
1. ``baseline`` — feature off.
2. ``center-only`` — ``weights={center: 1, eps: 0, pa: 0}``.
3. ``all-axes`` — ``weights={center: 1, eps: 1, pa: 1}`` (Stage-B default).

Uses the LegacySurvey real invvar maps (no MAD-bias risk in
sky-std). The PGC006669 cutout's outer-LSB regime is the ideal real-
data stress test for outer-region damping.

Stage-3 Stage-B (plan section 7).
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from isoster.multiband import (
    IsosterConfigMB,
    fit_image_multiband,
    isophote_results_mb_to_fits,
    plot_qa_summary_mb,
)

from legacysurvey_loader import (
    LEGACYSURVEY_ZP,
    asinh_softening_from_log10_match,
    load_legacysurvey_grz,
)


DAMPING_CONFIGS = (
    ("baseline", False, {"center": 1.0, "eps": 1.0, "pa": 1.0}),
    ("center-only", True, {"center": 1.0, "eps": 0.0, "pa": 0.0}),
    ("all-axes", True, {"center": 1.0, "eps": 1.0, "pa": 1.0}),
)


def _build_cfg(
    bands: List[str], maxsma: float, *,
    use_outer_reg: bool, weights: Dict[str, float],
) -> IsosterConfigMB:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return IsosterConfigMB(
            bands=bands,
            reference_band="r",
            harmonic_combination="joint",
            band_weights={b: 1.0 for b in bands},
            sma0=10.0, minsma=1.0, maxsma=maxsma,
            astep=0.10, linear_growth=False, debug=True,
            use_outer_center_regularization=use_outer_reg,
            outer_reg_sma_onset=40.0,  # PGC's outer LSB transition
            outer_reg_strength=2.0,
            outer_reg_weights=weights,
        )


def _outer_tail_metrics(result: dict, bands: List[str], n_outer: int = 20) -> dict:
    rows = result["isophotes"]
    smas = np.array([float(r.get("sma", float("nan"))) for r in rows])
    valid = np.isfinite(smas)
    valid_rows = [r for r, v in zip(rows, valid) if v]
    tail = valid_rows[-n_outer:] if len(valid_rows) >= n_outer else valid_rows

    intens_mad: Dict[str, float] = {}
    for b in bands:
        col = np.array(
            [float(r.get(f"intens_{b}", float("nan"))) for r in tail],
            dtype=np.float64,
        )
        finite = col[np.isfinite(col)]
        if finite.size < 4:
            intens_mad[b] = float("nan")
        else:
            intens_mad[b] = float(np.median(np.abs(finite - np.median(finite))))

    eps_arr = np.array([float(r.get("eps", float("nan"))) for r in tail])
    pa_arr = np.array([float(r.get("pa", float("nan"))) for r in tail])
    eps_mad = (
        float(np.median(np.abs(eps_arr - np.median(eps_arr))))
        if eps_arr.size else float("nan")
    )
    pa_mad = (
        float(np.median(np.abs(pa_arr - np.median(pa_arr))))
        if pa_arr.size else float("nan")
    )
    return {
        "n_outer_isophotes": len(tail),
        "intens_mad_per_band": intens_mad,
        "eps_mad": eps_mad,
        "pa_mad": pa_mad,
    }


def _summarize(result: dict, bands: List[str]) -> dict:
    iso = result["isophotes"]
    if not iso:
        return {"n_isophotes": 0}
    sma_arr = np.asarray([r["sma"] for r in iso], dtype=np.float64)
    stop_codes = [int(r["stop_code"]) for r in iso]
    rms = np.asarray([r["rms"] for r in iso], dtype=np.float64)
    rms_finite = rms[np.isfinite(rms)]
    return {
        "n_isophotes": len(iso),
        "n_converged": sum(1 for c in stop_codes if c == 0),
        "stop_code_counts": {
            str(c): stop_codes.count(c) for c in sorted(set(stop_codes))
        },
        "sma_range": [float(sma_arr.min()), float(sma_arr.max())],
        "mean_rms": float(np.nanmean(rms_finite)) if rms_finite.size else float("nan"),
        "median_rms": float(np.nanmedian(rms_finite)) if rms_finite.size else float("nan"),
        "outer_tail": _outer_tail_metrics(result, bands),
    }


def _comparison_figure(
    results_by_label: Dict[str, dict],
    out_path: Path,
    title_prefix: str,
) -> None:
    rows = ("x0", "y0", "eps", "pa")
    n_rows = len(rows)
    cols = list(results_by_label.keys())
    n_cols = len(cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.6 * n_cols, 2.4 * n_rows),
        sharex=True, sharey="row",
    )
    if n_rows == 1:
        axes = axes[None, :]
    for col_idx, label in enumerate(cols):
        result = results_by_label.get(label)
        if result is None:
            continue
        iso = result["isophotes"]
        sma = np.asarray([r["sma"] for r in iso], dtype=np.float64)
        for row_idx, key in enumerate(rows):
            ax = axes[row_idx, col_idx]
            y = np.asarray([float(r.get(key, np.nan)) for r in iso])
            ax.plot(sma, y, lw=1.0, color="#1f77b4", alpha=0.9)
            if row_idx == 0:
                ax.set_title(label, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(key)
            if row_idx == n_rows - 1:
                ax.set_xlabel("SMA (px)")
    fig.suptitle(title_prefix, fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(galaxy_dir: Path, galaxy_prefix: str, out_dir: Path) -> None:
    cutout = load_legacysurvey_grz(galaxy_dir, galaxy_prefix)
    bands = list(cutout.bands)
    masks_per_band = [cutout.combined_mask.copy() for _ in bands]
    maxsma = float(min(cutout.shape) * 0.45)

    out_dir.mkdir(parents=True, exist_ok=True)
    softening = {
        b: asinh_softening_from_log10_match(
            cutout.pixel_scale_arcsec, LEGACYSURVEY_ZP, bright_mu=22.0,
        )
        for b in bands
    }

    results_by_label: Dict[str, dict] = {}
    stats_by_label: Dict[str, dict] = {}
    for label, on, weights in DAMPING_CONFIGS:
        print(f"=== Running PGC006669 with config={label!r} ===")
        cfg = _build_cfg(bands, maxsma, use_outer_reg=on, weights=weights)
        res = fit_image_multiband(
            cutout.images, masks_per_band, cfg, variance_maps=cutout.variances,
        )
        results_by_label[label] = res
        stats_by_label[label] = _summarize(res, bands)

        slug = label.replace(" ", "_").replace("-", "_")
        fits_path = out_dir / f"{galaxy_prefix}_outer_reg_{slug}.fits"
        isophote_results_mb_to_fits(res, str(fits_path))
        print(f"  wrote {fits_path}")

        qa_path = out_dir / f"{galaxy_prefix}_outer_reg_{slug}.png"
        plot_qa_summary_mb(
            result=res,
            images=cutout.images,
            bands=bands,
            sb_zeropoint=LEGACYSURVEY_ZP,
            pixel_scale_arcsec=cutout.pixel_scale_arcsec,
            softening_per_band=softening,
            object_mask=cutout.combined_mask,
            output_path=str(qa_path),
            title=f"PGC006669 — outer_reg config: {label}",
        )
        print(f"  wrote {qa_path}")

    cmp_path = out_dir / f"{galaxy_prefix}_outer_reg_compare.png"
    _comparison_figure(
        results_by_label, cmp_path,
        title_prefix=f"PGC006669 — geometry (x0/y0/eps/pa) vs SMA across damping configs",
    )
    print(f"Wrote {cmp_path}")

    stats_path = out_dir / f"{galaxy_prefix}_outer_reg_stats.json"
    stats_path.write_text(json.dumps(stats_by_label, indent=2))
    print(f"Wrote {stats_path}")

    print()
    print(
        f"{'config':<15} {'n_iso':>5} {'conv':>5} {'mean_rms':>10}  "
        f"{'eps MAD':>10} {'pa MAD':>10}"
    )
    print("-" * 70)
    for label, _, _ in DAMPING_CONFIGS:
        s = stats_by_label[label]
        ot = s["outer_tail"]
        print(
            f"{label:<15} {s['n_isophotes']:>5} {s['n_converged']:>5} "
            f"{s['mean_rms']:>10.3e}  "
            f"{ot['eps_mad']:>10.3e} {ot['pa_mad']:>10.3e}"
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
        default=Path("outputs/benchmark_multiband/outer_reg_damping_pgc"),
        type=Path,
    )
    args = parser.parse_args()
    run(args.galaxy_dir, args.galaxy_prefix, args.out_dir)


if __name__ == "__main__":
    main()
