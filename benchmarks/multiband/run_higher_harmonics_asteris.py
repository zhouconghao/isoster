"""Higher-harmonics demo on an asteris HSC g/r/i/z/y cutout.

Sweeps all four ``multiband_higher_harmonics`` enum values on the same
asteris denoised cutout (object 37484563299062823 by default), generates
a QA figure for each, and writes:

- ``<out_dir>/asteris_<obj_id>_higher_harmonics_<mode>.fits`` per mode.
- ``<out_dir>/asteris_<obj_id>_higher_harmonics_<mode>.png`` per-mode QA panel.
- ``<out_dir>/asteris_<obj_id>_higher_harmonics_compare.png`` 4-panel
  side-by-side comparison of per-band a3/b3/a4/b4 vs SMA across modes.
- ``<out_dir>/asteris_<obj_id>_higher_harmonics_stats.json`` per-mode summary.

Asteris cutouts are 768x768 with B=5 bands, larger than the LegacySurvey
PGC006669 demo. This is the harder workload — used by the perf
benchmark (``bench_higher_harmonics_asteris.py``) for the ≤2.5×
single-band quality bar.

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
    subtract_outermost_sky_offset,
)

DEMO_BAND_FOLDERS = ["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"]
DEMO_BANDS = ["g", "r", "i", "z", "y"]
HSC_ZP = 27.0
HSC_PIXEL_SCALE_ARCSEC = 0.168
ASTERIS_OBJ_ID = "37484563299062823"

HARMONICS_MODES = (
    "independent",
    "shared",
    "simultaneous_in_loop",
    "simultaneous_original",
)


def _galaxy_dir(asteris_root: Path, obj_id: str, band_folder: str) -> Path:
    cutout_name = (
        f"obj_{obj_id}_pdr3_dud_rev_8523_0_3_cutout768_uniform8_t8_v1"
    )
    return asteris_root / cutout_name / band_folder


def _load_denoised(asteris_root: Path, obj_id: str, band_folder: str) -> np.ndarray:
    from astropy.io import fits

    path = _galaxy_dir(asteris_root, obj_id, band_folder) / "denoised.fits"
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None:
                return np.asarray(hdu.data, dtype=np.float64)
    raise IOError(f"No data HDU found in {path}")


def _load_object_mask(asteris_root: Path, obj_id: str) -> np.ndarray:
    from astropy.io import fits

    mpath = _galaxy_dir(asteris_root, obj_id, "HSC-I") / "object_mask.fits"
    with fits.open(mpath) as hdul:
        data = hdul[0].data
    return np.asarray(data, dtype=np.float64) > 0.5


def _sky_std(image: np.ndarray) -> float:
    """Sigma-clipped sky std matching the asteris example script.

    Mirrors :func:`examples.example_asteris_denoised.common.sky_std` so
    variance maps are bit-identical to the reference example used by the
    asteris demo (avoids a 12% sky-std bias that shifts borderline
    outer-SMA convergence behavior).
    """
    from astropy.stats import sigma_clipped_stats

    _, _, std = sigma_clipped_stats(image, sigma=3.0, maxiters=5)
    return float(std)


def _uniform_variance_map(image: np.ndarray) -> np.ndarray:
    return np.full_like(image, _sky_std(image) ** 2, dtype=np.float64)


def _build_cfg(
    bands: List[str],
    h: int, w: int,
    *,
    mode: str,
) -> IsosterConfigMB:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return IsosterConfigMB(
            bands=bands, reference_band="i",
            sma0=20.0, eps=0.2, pa=0.0, astep=0.1, linear_growth=False,
            maxsma=float(np.hypot(h, w)) / 2.0, debug=True,
            compute_deviations=True, nclip=2, max_retry_first_isophote=5,
            band_weights=None,
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
    rows = ("a3", "b3", "a4", "b4")
    n_rows = len(rows)
    n_cols = len(HARMONICS_MODES)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.6 * n_cols, 2.4 * n_rows),
        sharex=True, sharey="row",
    )
    if n_rows == 1:
        axes = axes[None, :]
    band_colors = {
        b: c for b, c in zip(
            bands, ("#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"),
        )
    }
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
                    color=band_colors[b],
                    label=b if (row_idx == 0 and col_idx == 0) else None,
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
        fig.legend(
            handles, labels, loc="upper right",
            ncol=len(bands), frameon=False, fontsize=9,
        )
    fig.suptitle(title_prefix, fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(
    asteris_root: Path,
    obj_id: str,
    out_dir: Path,
    n_outer_for_sky: int = 8,
) -> None:
    print(f"=== Loading asteris cutout {obj_id} ===")
    images: List[np.ndarray] = []
    sky_rms_per_band: Dict[str, float] = {}
    for folder, b in zip(DEMO_BAND_FOLDERS, DEMO_BANDS):
        img = _load_denoised(asteris_root, obj_id, folder)
        images.append(img)
        sky_rms_per_band[b] = _sky_std(img)
    mask = _load_object_mask(asteris_root, obj_id)
    variance_maps = [_uniform_variance_map(im) for im in images]
    h, w = images[0].shape

    out_dir.mkdir(parents=True, exist_ok=True)
    # Mirror the asteris example's QA pipeline:
    # - softening matched to per-band sky RMS so the asinh-mu transform
    #   handles outer-region noise without log10 blowups at zero crossings.
    # - subtract per-band outermost-ring sky offset before plotting so the
    #   surface brightness profile reflects galaxy signal rather than
    #   asymptoting to the per-band I0_b sky-residual plateau.
    softening = {b: max(s, 1e-6) for b, s in sky_rms_per_band.items()}
    # ``n_outer_for_sky`` is a function parameter; the post-process picks
    # the median I0_b over this many outermost valid isophotes per band.

    results_by_mode: Dict[str, dict] = {}
    stats_by_mode: Dict[str, dict] = {}
    for mode in HARMONICS_MODES:
        print(f"=== Running asteris {obj_id} with multiband_higher_harmonics={mode!r} ===")
        cfg = _build_cfg(DEMO_BANDS, h, w, mode=mode)
        res = fit_image_multiband(
            images=images, masks=mask, config=cfg, variance_maps=variance_maps,
        )
        results_by_mode[mode] = res
        stats_by_mode[mode] = _summarize(res, DEMO_BANDS)

        fits_path = out_dir / f"asteris_{obj_id}_higher_harmonics_{mode}.fits"
        isophote_results_mb_to_fits(res, str(fits_path))
        print(f"  wrote {fits_path}")

        # Sky-offset correction for the QA figure (mirrors the asteris
        # example's pipeline). The raw FITS already saved above keeps the
        # uncorrected I0_b values so downstream tools see the joint
        # solver's output; the QA plot uses the corrected copy + the
        # correspondingly shifted images so the residual mosaic stays in
        # matched units.
        res_corr, sky_offsets = subtract_outermost_sky_offset(
            res, n_outer=n_outer_for_sky,
        )
        images_corr = [im - sky_offsets[b] for im, b in zip(images, DEMO_BANDS)]

        qa_path = out_dir / f"asteris_{obj_id}_higher_harmonics_{mode}.png"
        plot_qa_summary_mb(
            result=res_corr,
            images=images_corr,
            bands=DEMO_BANDS,
            sb_zeropoint=HSC_ZP,
            pixel_scale_arcsec=HSC_PIXEL_SCALE_ARCSEC,
            softening_per_band=softening,
            object_mask=mask,
            output_path=str(qa_path),
            title=f"asteris {obj_id} — multiband_higher_harmonics={mode!r}",
        )
        print(f"  wrote {qa_path}  (sky offsets: " + ", ".join(f"{b}={sky_offsets[b]:+.3g}" for b in DEMO_BANDS) + ")")

    cmp_path = out_dir / f"asteris_{obj_id}_higher_harmonics_compare.png"
    _comparison_figure(
        results_by_mode, DEMO_BANDS, cmp_path,
        title_prefix=f"asteris {obj_id} — per-band a_n/b_n vs SMA across modes",
    )
    print(f"Wrote {cmp_path}")

    stats_path = out_dir / f"asteris_{obj_id}_higher_harmonics_stats.json"
    stats_path.write_text(json.dumps(stats_by_mode, indent=2))
    print(f"Wrote {stats_path}")

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asteris-root",
        default=Path.home() / "Downloads" / "asteris" / "objs",
        type=Path,
    )
    parser.add_argument("--obj-id", default=ASTERIS_OBJ_ID)
    parser.add_argument(
        "--out-dir",
        default=Path("outputs/benchmark_multiband/higher_harmonics_asteris"),
        type=Path,
    )
    parser.add_argument(
        "--n-outer-for-sky",
        type=int,
        default=8,
        help="Number of outermost valid isophotes per band whose median "
        "I0_b is used as the inferred sky offset (subtracted from "
        "intens_<b> and from the images before plotting). Default 8.",
    )
    args = parser.parse_args()
    run(
        args.asteris_root, args.obj_id, args.out_dir,
        n_outer_for_sky=args.n_outer_for_sky,
    )


if __name__ == "__main__":
    main()
