"""Higher-harmonics demo on PGC006669 (LegacySurvey grz).

Sweeps all four ``multiband_higher_harmonics`` enum values on the same
PGC006669 cutout, generates a QA figure for each, and writes:

- ``<out_dir>/PGC006669_higher_harmonics_<mode>.fits`` for each mode
  (full Schema-1 round-trippable result).
- ``<out_dir>/PGC006669_higher_harmonics_<mode>.png`` per-mode QA panel.
- ``<out_dir>/PGC006669_higher_harmonics_compare.png`` 4-panel side-by-side
  comparison of per-band a3/b3/a4/b4 vs SMA across all four modes.
- ``<out_dir>/PGC006669_higher_harmonics_stats.json`` per-mode summary
  (n_isophotes, mean RMS, mean stop_code, harmonic-amplitude band-spread).

The reference baseline is ``independent`` mode (Stage-1 default). Other
modes are compared against it for sanity (geometry should agree to
1e-4 since only the higher-order block changes; per-band a_n / b_n
necessarily differ because they're shared values vs per-band values).

Section 6 of plan-2026-04-29-multiband-feasibility.md.
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


HARMONICS_MODES = (
    "independent",
    "shared",
    "simultaneous_in_loop",
    "simultaneous_original",
)


def _build_cfg(
    bands: List[str],
    maxsma: float,
    *,
    mode: str,
) -> IsosterConfigMB:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return IsosterConfigMB(
            bands=bands,
            reference_band="r",
            harmonic_combination="joint",
            band_weights={b: 1.0 for b in bands},
            integrator="median",
            sma0=10.0,
            minsma=1.0,
            maxsma=maxsma,
            astep=0.10,
            linear_growth=False,
            debug=True,
            multiband_higher_harmonics=mode,
        )


def _summarize(result: dict, bands: List[str]) -> dict:
    iso = result["isophotes"]
    if not iso:
        return {"n_isophotes": 0}
    sma_arr = np.asarray([r["sma"] for r in iso], dtype=np.float64)
    stop_codes = [int(r["stop_code"]) for r in iso]
    n_converged = sum(1 for c in stop_codes if c == 0)
    rms = np.asarray([r["rms"] for r in iso], dtype=np.float64)
    rms_finite = rms[np.isfinite(rms)]

    # Per-band a3, b3, a4, b4 spread across bands at every isophote.
    # Under shared / simultaneous_* modes spread should be exactly 0 by
    # construction. Independent mode has finite spread.
    spread: Dict[str, float] = {}
    for n_order in (3, 4):
        for prefix in ("a", "b"):
            cols = [f"{prefix}{n_order}_{b}" for b in bands]
            try:
                vals = np.asarray(
                    [[r[c] for c in cols] for r in iso], dtype=np.float64,
                )
            except KeyError:
                continue
            row_range = np.nanmax(vals, axis=1) - np.nanmin(vals, axis=1)
            spread[f"max_band_spread_{prefix}{n_order}"] = float(np.nanmax(row_range))

    return {
        "n_isophotes": len(iso),
        "n_converged": n_converged,
        "stop_code_counts": {
            str(c): stop_codes.count(c) for c in sorted(set(stop_codes))
        },
        "sma_range": [float(sma_arr.min()), float(sma_arr.max())],
        "mean_rms": float(np.nanmean(rms_finite)) if rms_finite.size else float("nan"),
        "median_rms": float(np.nanmedian(rms_finite)) if rms_finite.size else float("nan"),
        "harmonics_shared": bool(result.get("harmonics_shared", False)),
        "multiband_higher_harmonics": result.get("multiband_higher_harmonics"),
        "harmonic_orders": result.get("harmonic_orders"),
        **spread,
    }


def _comparison_figure(
    results_by_mode: Dict[str, dict],
    bands: List[str],
    out_path: Path,
    title_prefix: str,
) -> None:
    """Side-by-side 4-panel comparison of harmonic profiles vs SMA.

    Rows: a3, b3, a4, b4 (one per harmonic component). Columns: one per
    enum mode. Each panel overlays per-band curves so the user can read
    off (1) whether bands collapse onto a single curve under shared /
    simultaneous_*, and (2) whether the recovered amplitude is comparable
    across modes at the same SMA.
    """
    rows = ("a3", "b3", "a4", "b4")
    n_rows = len(rows)
    n_cols = len(HARMONICS_MODES)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.6 * n_cols, 2.4 * n_rows),
        sharex=True, sharey="row",
    )
    if n_rows == 1:
        axes = axes[None, :]
    band_colors = {b: c for b, c in zip(bands, ("#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"))}
    for col_idx, mode in enumerate(HARMONICS_MODES):
        result = results_by_mode.get(mode)
        if result is None:
            continue
        iso = result["isophotes"]
        sma = np.asarray([r["sma"] for r in iso], dtype=np.float64)
        for row_idx, harm in enumerate(rows):
            ax = axes[row_idx, col_idx]
            for b in bands:
                col = f"{harm}_{b}"
                try:
                    y = np.asarray([r[col] for r in iso], dtype=np.float64)
                except KeyError:
                    continue
                ax.plot(
                    sma, y, lw=1.0, alpha=0.85,
                    color=band_colors[b], label=b if (row_idx == 0 and col_idx == 0) else None,
                )
            ax.axhline(0.0, color="0.6", lw=0.5, zorder=-1)
            if row_idx == 0:
                ax.set_title(mode, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(harm)
            if row_idx == n_rows - 1:
                ax.set_xlabel("SMA (px)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", ncol=len(bands), frameon=False, fontsize=9)
    fig.suptitle(title_prefix, fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(
    galaxy_dir: Path,
    galaxy_prefix: str,
    out_dir: Path,
) -> None:
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

    results_by_mode: Dict[str, dict] = {}
    stats_by_mode: Dict[str, dict] = {}
    for mode in HARMONICS_MODES:
        print(f"=== Running PGC006669 with multiband_higher_harmonics={mode!r} ===")
        cfg = _build_cfg(bands, maxsma, mode=mode)
        res = fit_image_multiband(
            cutout.images, masks_per_band, cfg, variance_maps=cutout.variances,
        )
        results_by_mode[mode] = res
        stats_by_mode[mode] = _summarize(res, bands)

        fits_path = out_dir / f"{galaxy_prefix}_higher_harmonics_{mode}.fits"
        isophote_results_mb_to_fits(res, str(fits_path))
        print(f"  wrote {fits_path}")

        qa_path = out_dir / f"{galaxy_prefix}_higher_harmonics_{mode}.png"
        plot_qa_summary_mb(
            result=res,
            images=cutout.images,
            bands=bands,
            sb_zeropoint=LEGACYSURVEY_ZP,
            pixel_scale_arcsec=cutout.pixel_scale_arcsec,
            softening_per_band=softening,
            object_mask=cutout.combined_mask,
            output_path=str(qa_path),
            title=f"PGC006669 — multiband_higher_harmonics={mode!r}",
        )
        print(f"  wrote {qa_path}")

    # Cross-mode comparison figure.
    cmp_path = out_dir / f"{galaxy_prefix}_higher_harmonics_compare.png"
    _comparison_figure(
        results_by_mode, bands, cmp_path,
        title_prefix=f"PGC006669 — per-band a_n/b_n vs SMA across modes",
    )
    print(f"Wrote {cmp_path}")

    # Stats JSON.
    stats_path = out_dir / f"{galaxy_prefix}_higher_harmonics_stats.json"
    stats_path.write_text(json.dumps(stats_by_mode, indent=2))
    print(f"Wrote {stats_path}")

    # Console summary.
    print()
    print(f"{'mode':<25} {'n_iso':>5} {'conv':>5} {'mean_rms':>10}  band-spread (a3, b3, a4, b4)")
    print("-" * 90)
    for mode in HARMONICS_MODES:
        s = stats_by_mode[mode]
        spread = " | ".join(
            f"{s.get(f'max_band_spread_{k}', float('nan')):.2e}"
            for k in ("a3", "b3", "a4", "b4")
        )
        print(
            f"{mode:<25} {s['n_isophotes']:>5} {s['n_converged']:>5} "
            f"{s['mean_rms']:>10.3e}  {spread}"
        )
    print()
    print(
        "Note: max-band-spread = max_isophote(max_band(a_n) - min_band(a_n)). "
        "Under shared / simultaneous_* this is exactly 0 by construction."
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
        default=Path("outputs/benchmark_multiband/higher_harmonics_pgc"),
        type=Path,
    )
    args = parser.parse_args()
    run(args.galaxy_dir, args.galaxy_prefix, args.out_dir)


if __name__ == "__main__":
    main()
