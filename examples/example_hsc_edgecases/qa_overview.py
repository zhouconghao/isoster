#!/usr/bin/env python3
"""QA overview figure: 3-color image and i-band object mask for each galaxy.

Produces a 2-row × 6-column panel figure. Top row shows gri color composites
(arcsinh stretch); bottom row shows the HSC-I image with the combined object
mask overlaid in semi-transparent red.

Usage
-----
    uv run python examples/example_hsc_edgecases/qa_overview.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent

GALAXIES = [
    ("10140088", "clear case"),
    ("10140002", "nearby bright star"),
    ("10140006", "nearby large galaxy"),
    ("10140009", "blending bright star"),
    ("10140056", "artifact"),
    ("10140093", "small blending source"),
]


def arcsinh_stretch(image, scale=0.02, clip_percentile=99.8):
    """Apply arcsinh stretch to an image for visualization.

    Args:
        image: 2D array of pixel values.
        scale: Softening parameter controlling the transition from
            linear to logarithmic behavior.
        clip_percentile: Upper percentile for clipping bright pixels.

    Returns:
        Stretched image normalized to [0, 1].
    """
    vmax = np.nanpercentile(image, clip_percentile)
    normed = np.clip(image, 0, vmax) / max(vmax, 1e-10)
    stretched = np.arcsinh(normed / scale) / np.arcsinh(1.0 / scale)
    return np.clip(stretched, 0, 1)


def make_color_image(g_data, r_data, i_data, scale=0.02, clip_percentile=99.8):
    """Create an RGB color composite from gri bands.

    Maps HSC-G -> blue, HSC-R -> green, HSC-I -> red.
    """
    r_ch = arcsinh_stretch(i_data, scale=scale, clip_percentile=clip_percentile)
    g_ch = arcsinh_stretch(r_data, scale=scale, clip_percentile=clip_percentile)
    b_ch = arcsinh_stretch(g_data, scale=scale, clip_percentile=clip_percentile)
    return np.stack([r_ch, g_ch, b_ch], axis=-1)


def make_mask_overlay(i_data, mask, scale=0.02, clip_percentile=99.8, mask_alpha=0.45):
    """Create an i-band grayscale image with red mask overlay."""
    gray = arcsinh_stretch(i_data, scale=scale, clip_percentile=clip_percentile)
    rgb = np.stack([gray, gray, gray], axis=-1)

    mask_bool = mask > 0
    rgb[mask_bool, 0] = rgb[mask_bool, 0] * (1 - mask_alpha) + mask_alpha
    rgb[mask_bool, 1] = rgb[mask_bool, 1] * (1 - mask_alpha)
    rgb[mask_bool, 2] = rgb[mask_bool, 2] * (1 - mask_alpha)

    return np.clip(rgb, 0, 1)


def main():
    n_gal = len(GALAXIES)
    fig, axes = plt.subplots(
        2, n_gal,
        figsize=(3.2 * n_gal, 7.0),
    )

    for col, (obj_id, desc) in enumerate(GALAXIES):
        galaxy_dir = DATA_DIR / obj_id

        # Load data
        g_data = fits.getdata(galaxy_dir / f"{obj_id}_HSC_G_image.fits")
        r_data = fits.getdata(galaxy_dir / f"{obj_id}_HSC_R_image.fits")
        i_data = fits.getdata(galaxy_dir / f"{obj_id}_HSC_I_image.fits")
        mask = fits.getdata(galaxy_dir / f"{obj_id}_HSC_I_mask.fits")

        mask_frac = np.sum(mask > 0) / mask.size * 100

        # Top row: 3-color composite
        ax_color = axes[0, col]
        color_img = make_color_image(g_data, r_data, i_data)
        ax_color.imshow(color_img, origin="lower", aspect="equal")
        ax_color.set_title(f"{obj_id}\n{desc}", fontsize=9, fontweight="bold")
        ax_color.set_xticks([])
        ax_color.set_yticks([])

        # Bottom row: i-band + mask overlay
        ax_mask = axes[1, col]
        overlay = make_mask_overlay(i_data, mask)
        ax_mask.imshow(overlay, origin="lower", aspect="equal")
        ax_mask.set_title(f"mask: {mask_frac:.1f}%", fontsize=8)
        ax_mask.set_xticks([])
        ax_mask.set_yticks([])

    # Row labels on the left edge
    axes[0, 0].set_ylabel("gri color", fontsize=10)
    axes[1, 0].set_ylabel("I + mask", fontsize=10)

    fig.suptitle(
        "HSC Edge Cases: Overview",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    output_path = OUTPUT_DIR / "qa_overview.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved QA overview to {output_path}")


if __name__ == "__main__":
    main()
