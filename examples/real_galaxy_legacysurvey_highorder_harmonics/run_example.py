"""
LegacySurvey High-Order Harmonics Example
==========================================

Runs isoster on real LegacySurvey galaxy images (ESO243-49 and NGC3610) with
six fitting conditions spanning PA/EA sampling and three harmonic-order sets.

Usage
-----
::

    # ESO243-49 r-band (index 1)
    uv run python examples/real_galaxy_legacysurvey_highorder_harmonics/run_example.py \\
        --galaxy eso243-49 --band-index 1 \\
        --output-dir outputs/legacysurvey_highorder_harmonics/

    # NGC3610 r-band
    uv run python examples/real_galaxy_legacysurvey_highorder_harmonics/run_example.py \\
        --galaxy ngc3610 --band-index 1 \\
        --output-dir outputs/legacysurvey_highorder_harmonics/

Output layout
-------------
``<output-dir>/<galaxy>/band_<N>/``

- ``mask.fits``, ``mask_qa.png``
- ``<condition>/isophotes.fits``, ``<condition>/isophotes.ecsv``,
  ``<condition>/qa.png``
- ``comparison_qa.png``
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import isoster
from isoster.config import IsosterConfig
from isoster.plotting import (
    configure_qa_plot_style,
    derive_arcsinh_parameters,
    draw_isophote_overlays,
    make_arcsinh_display_from_parameters,
    plot_qa_summary_extended,
)

# Example-local modules (same directory)
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from masking import load_mask_fits, make_object_mask, save_mask_fits  # noqa: E402
from shared import (  # noqa: E402
    BAND_NAMES,
    CONDITION_ORDER,
    CONDITION_DISPLAY,
    FITS_FILENAME,
    MASK_PARAMS,
    PIXEL_SCALE,
    SUPPORTED_GALAXIES,
    load_legacysurvey_fits,
    make_isoster_configs,
    plot_harmonic_comparison_qa,
)

# ---------------------------------------------------------------------------
# Data root (relative to project root)
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LegacySurvey high-order harmonics example campaign.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--galaxy",
        choices=list(SUPPORTED_GALAXIES),
        required=True,
        help="Galaxy to process.",
    )
    parser.add_argument(
        "--band-index",
        type=int,
        default=1,
        help="Band plane index (0-based) to extract from the 3D FITS cube.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/legacysurvey_highorder_harmonics"),
        help="Root output directory.",
    )
    parser.add_argument(
        "--sma0",
        type=float,
        default=None,
        help="Override initial SMA (pixels).",
    )
    parser.add_argument(
        "--skip-mask",
        action="store_true",
        help="Skip mask generation and use an empty mask instead.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITION_ORDER,
        default=None,
        help="Subset of conditions to run (default: all 6).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Mask QA figure
# ---------------------------------------------------------------------------

def save_mask_qa(
    image: np.ndarray,
    mask: np.ndarray,
    output_path: str | Path,
) -> None:
    """Save a QA figure showing the image with the mask overlay."""
    configure_qa_plot_style()

    fig, ax = plt.subplots(figsize=(6, 6))
    low, high, scale, vmax = derive_arcsinh_parameters(image)
    display, vmin, _ = make_arcsinh_display_from_parameters(
        image, low=low, high=high, scale=scale, vmax=vmax,
    )
    ax.imshow(display, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    if mask is not None and np.any(mask):
        rgba = np.zeros((*image.shape, 4))
        rgba[mask] = [1, 0, 0, 0.45]
        ax.imshow(rgba, origin="lower")
    n_masked = int(np.sum(mask)) if mask is not None else 0
    frac = 100.0 * n_masked / image.size if image.size > 0 else 0.0
    ax.set_title(f"Mask QA  ({n_masked} px masked, {frac:.1f}%)", fontsize=12)
    ax.set_xlabel("x [pixel]")
    ax.set_ylabel("y [pixel]")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  Saved mask QA → {output_path}")


# ---------------------------------------------------------------------------
# Per-condition QA figure
# ---------------------------------------------------------------------------

def save_condition_qa(
    image: np.ndarray,
    mask: np.ndarray | None,
    results: dict,
    condition_label: str,
    output_path: str | Path,
) -> None:
    """Save per-condition extended QA figure."""
    isophotes = results["isophotes"]
    config: IsosterConfig = results["config"]

    # Reconstruct 2D model (auto-detects harmonic orders and EA mode from isophotes)
    try:
        model = isoster.build_isoster_model(
            image.shape,
            isophotes,
            use_harmonics=True,
        )
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Model reconstruction failed for {condition_label}: {exc}")
        model = np.zeros_like(image)

    harmonic_orders = config.harmonic_orders if config.simultaneous_harmonics else None

    plot_qa_summary_extended(
        title=f"{condition_label} — {CONDITION_DISPLAY[condition_label]}",
        image=image,
        isoster_model=model,
        isoster_res=isophotes,
        harmonic_orders=harmonic_orders,
        mask=mask,
        filename=str(output_path),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_arguments()
    galaxy = args.galaxy
    band_index = args.band_index
    output_dir = args.output_dir

    # Output sub-directory for this galaxy / band
    band_dir = output_dir / galaxy / f"band_{band_index}"
    band_dir.mkdir(parents=True, exist_ok=True)

    # Flat QA directory — all PNG figures copied here with descriptive names
    qa_flat_dir = band_dir / "qa_figures"
    qa_flat_dir.mkdir(parents=True, exist_ok=True)

    band_name = BAND_NAMES[galaxy][band_index]
    print(f"\n{'='*60}")
    print(f"  Galaxy : {galaxy}")
    print(f"  Band   : {band_name}  (index {band_index})")
    print(f"  Output : {band_dir}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1. Load image
    # ------------------------------------------------------------------
    fits_path = _DATA_DIR / FITS_FILENAME[galaxy]
    if not fits_path.exists():
        sys.exit(
            f"ERROR: FITS file not found: {fits_path}\n"
            f"Expected under examples/data/{FITS_FILENAME[galaxy]}"
        )

    print("Loading image …")
    image, pixel_scale_from_header = load_legacysurvey_fits(fits_path, band_index)

    # Use known pixel scale if header value is unreliable (both are stored in metadata)
    known_pixel_scale = PIXEL_SCALE[galaxy]
    print(f"  Image shape : {image.shape}")
    print(f"  Pixel scale : {known_pixel_scale} arcsec/px (known)  "
          f"| {pixel_scale_from_header:.4f} arcsec/px (header)")

    # ------------------------------------------------------------------
    # 2. Generate / load mask
    # ------------------------------------------------------------------
    mask_fits_path = band_dir / "mask.fits"
    mask_qa_path = band_dir / "mask_qa.png"

    if args.skip_mask:
        print("Skipping mask generation (--skip-mask).")
        mask = np.zeros(image.shape, dtype=bool)
    else:
        print("Generating object mask …")
        mask_kwargs = MASK_PARAMS.get(galaxy, {})
        center_xy = ((image.shape[1] - 1) / 2.0, (image.shape[0] - 1) / 2.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask = make_object_mask(image, center_xy=center_xy, **mask_kwargs)
        print(f"  Masked pixels: {int(np.sum(mask))} / {mask.size} "
              f"({100.0*np.sum(mask)/mask.size:.1f}%)")
        save_mask_fits(mask, str(mask_fits_path))
        print(f"  Saved mask → {mask_fits_path}")

    save_mask_qa(image, mask, mask_qa_path)
    shutil.copy2(
        mask_qa_path,
        qa_flat_dir / f"{galaxy}_band{band_index}_mask_qa.png",
    )

    # ------------------------------------------------------------------
    # 3. Build IsosterConfig objects for all conditions
    # ------------------------------------------------------------------
    conditions_to_run = args.conditions or CONDITION_ORDER
    all_configs = make_isoster_configs(galaxy, sma0=args.sma0)
    configs = {k: v for k, v in all_configs.items() if k in conditions_to_run}

    print(f"\nRunning {len(configs)} fitting condition(s): {list(configs)}\n")

    # ------------------------------------------------------------------
    # 4. Run each condition
    # ------------------------------------------------------------------
    all_results: dict[str, list[dict]] = {}

    for condition_label, config in configs.items():
        cond_dir = band_dir / condition_label
        cond_dir.mkdir(parents=True, exist_ok=True)

        fits_out = cond_dir / "isophotes.fits"
        ecsv_out = cond_dir / "isophotes.ecsv"
        qa_out = cond_dir / "qa.png"

        print(f"  [{condition_label}]  EA={config.use_eccentric_anomaly}  "
              f"sim_harm={config.simultaneous_harmonics}  "
              f"orders={config.harmonic_orders}")

        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = isoster.fit_image(image, mask, config)
        elapsed = time.perf_counter() - t0

        isophotes = results["isophotes"]
        n_iso = len(isophotes)
        n_conv = sum(1 for r in isophotes if r.get("stop_code", -99) == 0)
        print(f"    → {n_iso} isophotes, {n_conv} converged (stop=0), "
              f"{elapsed:.1f} s")

        # Save FITS
        isoster.isophote_results_to_fits(results, str(fits_out))
        print(f"    → saved FITS: {fits_out}")

        # Save ECSV (Astropy table)
        try:
            table = isoster.isophote_results_to_astropy_tables(results)
            if table is not None and len(table) > 0:
                table.write(str(ecsv_out), format="ascii.ecsv", overwrite=True)
                print(f"    → saved ECSV: {ecsv_out}")
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"ECSV save failed: {exc}")

        # Per-condition QA
        save_condition_qa(image, mask, results, condition_label, qa_out)
        print(f"    → saved QA : {qa_out}")
        shutil.copy2(
            qa_out,
            qa_flat_dir / f"{galaxy}_band{band_index}_{condition_label}_qa.png",
        )

        all_results[condition_label] = isophotes

    # ------------------------------------------------------------------
    # 5. Cross-condition comparison figure
    # ------------------------------------------------------------------
    if len(all_results) > 1:
        comp_qa_path = band_dir / "comparison_qa.png"
        print(f"\nBuilding comparison QA figure → {comp_qa_path}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot_harmonic_comparison_qa(
                galaxy=galaxy,
                image=image,
                mask=mask,
                condition_results=all_results,
                filename=str(comp_qa_path),
            )
        shutil.copy2(
            comp_qa_path,
            qa_flat_dir / f"{galaxy}_band{band_index}_comparison_qa.png",
        )
    else:
        print("\nSkipping comparison QA (fewer than 2 conditions).")

    print(f"\nDone.  Outputs in: {band_dir}")
    print(f"  Flat QA figures: {qa_flat_dir}\n")


if __name__ == "__main__":
    main()
