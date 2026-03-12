#!/usr/bin/env python
"""
QA comparison: isoster vs photutils vs analytic truth.

Generate QA figures comparing isoster (and optionally photutils) results
against ground-truth Sersic profiles on synthetic images.

Designed for reuse by other benchmarks: import ``PRESET_CASES`` and
``run_single_case()`` to plug into any benchmark pipeline.

Usage
-----
# Run all preset cases
uv run python examples/example_qa_comparison/generate_comparison_figures.py

# Single case, custom output directory
uv run python examples/example_qa_comparison/generate_comparison_figures.py \
    --case n4_eps04_snr100_noisy --output outputs/my_qa_run

# Skip photutils
uv run python examples/example_qa_comparison/generate_comparison_figures.py --no-photutils

Reuse from another script
-------------------------
>>> from examples.example_qa_comparison.generate_comparison_figures import (
...     PRESET_CASES, run_single_case,
... )
>>> stats = run_single_case(PRESET_CASES["n4_eps04_snr100_noisy"], output_dir="outputs/qa")
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.rcParams["text.usetex"] = False

from isoster import fit_image
from isoster.config import IsosterConfig
from isoster.plotting import build_method_profile, plot_comparison_qa_figure

from benchmarks.utils.sersic_model import (
    add_noise,
    create_sersic_image_vectorized,
    get_true_profile_at_sma,
    sersic_1d,
)

try:
    from photutils.isophote import Ellipse, EllipseGeometry
    PHOTUTILS_AVAILABLE = True
except ImportError:
    PHOTUTILS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Preset test cases
# ---------------------------------------------------------------------------

PRESET_CASES: dict[str, dict[str, Any]] = {
    "n1_eps07_high_ellipticity": {
        "n": 1.0, "R_e": 20.0, "I_e": 1000.0,
        "eps": 0.7, "pa": np.pi / 3, "snr": None,
        "oversample": 5, "shape": (600, 600),
    },
    "n1_eps04_snr100_noisy": {
        "n": 1.0, "R_e": 20.0, "I_e": 1000.0,
        "eps": 0.4, "pa": np.pi / 4, "snr": 100,
        "oversample": 5, "shape": (600, 600),
    },
    "n4_eps04_snr100_noisy": {
        "n": 4.0, "R_e": 20.0, "I_e": 1000.0,
        "eps": 0.4, "pa": np.pi / 4, "snr": 100,
        "oversample": 10, "shape": (600, 600),
    },
}


# ---------------------------------------------------------------------------
# Fitting runners
# ---------------------------------------------------------------------------

def _run_isoster(
    image: np.ndarray,
    params: dict,
    config_overrides: dict | None = None,
) -> tuple[list[dict], float]:
    """Run isoster and return (isophotes, wall_time_seconds)."""
    eps = params["eps"]
    kwargs = dict(
        x0=params["x0"], y0=params["y0"],
        eps=eps, pa=params["pa"],
        sma0=10.0, minsma=3.0, maxsma=8 * params["R_e"],
        astep=0.15, minit=10, maxit=50, conver=0.05,
        maxgerr=1.0 if eps > 0.6 else 0.5,
        use_eccentric_anomaly=(eps > 0.3),
    )
    if config_overrides:
        kwargs.update(config_overrides)
    config = IsosterConfig(**kwargs)

    t0 = time.perf_counter()
    results = fit_image(image, None, config)
    wall = time.perf_counter() - t0
    return results["isophotes"], wall


def _run_photutils(
    image: np.ndarray,
    params: dict,
) -> tuple[list[dict], float]:
    """Run photutils.isophote and return (isophotes, wall_time_seconds)."""
    if not PHOTUTILS_AVAILABLE:
        return [], 0.0

    geometry = EllipseGeometry(
        params["x0"], params["y0"], 10.0, params["eps"], params["pa"],
    )
    ellipse = Ellipse(image, geometry)

    t0 = time.perf_counter()
    try:
        isolist = ellipse.fit_image()
    except Exception as exc:
        print(f"    photutils failed: {exc}")
        return [], time.perf_counter() - t0
    wall = time.perf_counter() - t0

    minsma, maxsma = 3.0, 8 * params["R_e"]
    isophotes = []
    for iso in isolist:
        if iso.sma < minsma or iso.sma > maxsma:
            continue
        isophotes.append({
            "sma": iso.sma, "intens": iso.intens,
            "eps": iso.eps, "pa": iso.pa,
            "x0": iso.x0, "y0": iso.y0,
            "stop_code": 0 if iso.stop_code == 0 else -1,
        })
    return isophotes, wall


# ---------------------------------------------------------------------------
# Main entry point — importable by other benchmarks
# ---------------------------------------------------------------------------

def run_single_case(
    case: dict[str, Any],
    *,
    output_dir: str | Path = "outputs/example_qa_comparison",
    case_name: str = "case",
    run_photutils: bool = True,
    isoster_config_overrides: dict | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Run a single QA comparison case and return fit statistics.

    Parameters
    ----------
    case : dict
        Test-case specification with keys: n, R_e, I_e, eps, pa, snr,
        oversample, shape.
    output_dir : str or Path
        Directory for output figures.
    case_name : str
        Label used in filenames and figure titles.
    run_photutils : bool
        If True and photutils is installed, include photutils results.
    isoster_config_overrides : dict or None
        Extra keyword arguments forwarded to ``IsosterConfig``.
    seed : int
        Random seed for noise generation.

    Returns
    -------
    dict
        Statistics: median_frac_resid, max_abs_frac_resid,
        convergence_rate, n_isophotes, wall_time_seconds.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Generate synthetic image ---
    image, params = create_sersic_image_vectorized(
        n=case["n"], R_e=case["R_e"], I_e=case["I_e"],
        eps=case["eps"], pa=case["pa"], shape=case["shape"],
        oversample=case.get("oversample", 5),
    )
    if case.get("snr") is not None:
        image, _ = add_noise(image, case["snr"], case["R_e"], case["I_e"], seed=seed)

    snr_label = case["snr"] if case["snr"] is not None else "inf"
    title = (
        f"{case_name}: n={case['n']}, Re={case['R_e']}, "
        f"eps={case['eps']:.2f}, PA={np.degrees(case['pa']):.1f} deg, "
        f"SNR={snr_label}"
    )

    # --- Fit with isoster ---
    print(f"  [isoster] fitting ...")
    isophotes_iso, wall_iso = _run_isoster(
        image, params, config_overrides=isoster_config_overrides,
    )
    print(f"    {len(isophotes_iso)} isophotes, {wall_iso:.2f}s")

    # --- Fit with photutils (optional) ---
    isophotes_phot = []
    wall_phot = 0.0
    if run_photutils and PHOTUTILS_AVAILABLE:
        print(f"  [photutils] fitting ...")
        isophotes_phot, wall_phot = _run_photutils(image, params)
        print(f"    {len(isophotes_phot)} isophotes, {wall_phot:.2f}s")
    elif run_photutils and not PHOTUTILS_AVAILABLE:
        print("  [photutils] skipped (not installed)")

    # --- Build truth profile at isoster SMA values ---
    if len(isophotes_iso) > 0:
        sma_iso = np.array([iso["sma"] for iso in isophotes_iso])
        sma_dense = np.linspace(sma_iso.min(), sma_iso.max(), max(200, len(sma_iso)))
    else:
        sma_dense = np.linspace(3.0, 8 * case["R_e"], 200)

    truth_raw = get_true_profile_at_sma(sma_dense, params)
    truth_raw["x0"] = np.full_like(sma_dense, params["x0"])
    truth_raw["y0"] = np.full_like(sma_dense, params["y0"])
    truth_raw["stop_code"] = np.zeros_like(sma_dense, dtype=int)

    # --- Assemble profiles for plot_comparison_qa_figure ---
    profiles: dict[str, dict[str, np.ndarray]] = {}
    models: dict[str, np.ndarray] = {}

    # Truth (plotted as a method)
    truth_profile = build_method_profile(truth_raw)
    if truth_profile is not None:
        truth_profile["runtime_seconds"] = 0.0
        profiles["truth"] = truth_profile

    # Isoster
    iso_profile = build_method_profile(isophotes_iso)
    if iso_profile is not None:
        iso_profile["runtime_seconds"] = wall_iso
        profiles["isoster"] = iso_profile

        from isoster.model import build_isoster_model
        models["isoster"] = build_isoster_model(image.shape, isophotes_iso)

    # Photutils
    if isophotes_phot:
        phot_profile = build_method_profile(isophotes_phot)
        if phot_profile is not None:
            phot_profile["runtime_seconds"] = wall_phot
            profiles["photutils"] = phot_profile

    # --- Generate QA figure ---
    truth_style = {
        "color": "black", "marker": ".", "marker_face": "filled",
        "overlay_color": "black", "overlay_width": 0.5,
        "label": "truth",
    }
    plot_comparison_qa_figure(
        image, profiles,
        title=title,
        output_path=output_dir / f"{case_name}_qa.png",
        models=models,
        method_styles={"truth": truth_style},
        dpi=150,
    )
    print(f"  Saved: {output_dir / f'{case_name}_qa.png'}")

    # --- Compute statistics (isoster vs truth, 0.5–4 Re) ---
    stats: dict[str, Any] = {
        "case_name": case_name,
        "n_isophotes": len(isophotes_iso),
        "wall_time_seconds": wall_iso,
        "median_frac_resid": np.nan,
        "max_abs_frac_resid": np.nan,
        "convergence_rate": 0.0,
    }
    if len(isophotes_iso) > 0:
        sma_arr = np.array([iso["sma"] for iso in isophotes_iso])
        intens_arr = np.array([iso["intens"] for iso in isophotes_iso])
        stop_arr = np.array([iso.get("stop_code", 0) for iso in isophotes_iso])
        converged = stop_arr == 0

        intens_true = sersic_1d(sma_arr, params["I_e"], params["R_e"], params["n"])
        frac_resid = np.where(
            intens_true > 0,
            100.0 * (intens_arr - intens_true) / intens_true,
            0.0,
        )

        in_range = (sma_arr >= 0.5 * case["R_e"]) & (sma_arr <= 4.0 * case["R_e"]) & converged
        if np.any(in_range):
            stats["median_frac_resid"] = float(np.median(frac_resid[in_range]))
            stats["max_abs_frac_resid"] = float(np.max(np.abs(frac_resid[in_range])))
        stats["convergence_rate"] = float(np.sum(converged) / len(sma_arr) * 100)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="QA comparison: isoster vs photutils vs analytic truth",
    )
    parser.add_argument(
        "--case", type=str, default=None,
        help=f"Run a single preset case. Choices: {', '.join(PRESET_CASES)}",
    )
    parser.add_argument(
        "--output", type=str, default="outputs/example_qa_comparison",
        help="Output directory (default: outputs/example_qa_comparison)",
    )
    parser.add_argument(
        "--no-photutils", action="store_true",
        help="Skip photutils comparison",
    )
    args = parser.parse_args()

    cases = (
        {args.case: PRESET_CASES[args.case]}
        if args.case else PRESET_CASES
    )

    print("=" * 70)
    print("QA COMPARISON: isoster vs truth")
    if not args.no_photutils and PHOTUTILS_AVAILABLE:
        print("  photutils: YES")
    else:
        print("  photutils: SKIP")
    print("=" * 70)
    print()

    all_stats = []
    for name, case in cases.items():
        print(f"--- {name} ---")
        print(f"  n={case['n']}, Re={case['R_e']}, eps={case['eps']:.2f}, "
              f"PA={np.degrees(case['pa']):.1f} deg, SNR={case.get('snr', 'inf')}")
        stats = run_single_case(
            case,
            output_dir=args.output,
            case_name=name,
            run_photutils=not args.no_photutils,
        )
        all_stats.append(stats)
        print()

    # Summary table
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Case':<35} {'Med %':>8} {'Max %':>8} {'Conv':>7} {'N':>5} {'Time':>6}")
    print("-" * 70)
    for s in all_stats:
        print(
            f"{s['case_name']:<35} "
            f"{s['median_frac_resid']:>8.2f} "
            f"{s['max_abs_frac_resid']:>8.2f} "
            f"{s['convergence_rate']:>6.1f}% "
            f"{s['n_isophotes']:>5d} "
            f"{s['wall_time_seconds']:>5.2f}s"
        )
    print(f"\nOutput: {args.output}/")


if __name__ == "__main__":
    main()
