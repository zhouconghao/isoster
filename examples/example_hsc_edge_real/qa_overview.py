#!/usr/bin/env python3
"""QA overview figure for the three real HSC edge-case galaxies.

Produces a 3-row x 3-column figure (row per galaxy, column per panel):

    Col 0: gri color composite with arcsinh stretch, frame-center crosshair,
           detected target-center marker, and candidate contaminant centers.
    Col 1: HSC-I grayscale image with the default combined mask overlaid in
           semi-transparent red. Marks frame center and target center.
    Col 2: HSC-I asinh-stretched image with the same markers and a 200 px
           scale bar — used for spotting off-center contaminants and the
           extent of the target.

Usage
-----
    uv run python examples/example_hsc_edge_real/qa_overview.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy import ndimage

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent

GALAXIES = [
    ("37498869835124888", "very extended target, large ITER footprint"),
    ("42177291811318246", "bright star + extended neighbor"),
    ("42310032070569600", "bright star halo overlapping outskirts"),
]


def arcsinh_stretch(image, scale=0.02, clip_percentile=99.7):
    """arcsinh stretch an image to the [0, 1] interval for display."""
    vmax = np.nanpercentile(image, clip_percentile)
    normed = np.clip(image, 0, vmax) / max(vmax, 1e-10)
    stretched = np.arcsinh(normed / scale) / np.arcsinh(1.0 / scale)
    return np.clip(stretched, 0, 1)


def make_color_image(g_data, r_data, i_data, scale=0.02, clip_percentile=99.7):
    """gri -> RGB color composite. g->B, r->G, i->R."""
    r_ch = arcsinh_stretch(i_data, scale=scale, clip_percentile=clip_percentile)
    g_ch = arcsinh_stretch(r_data, scale=scale, clip_percentile=clip_percentile)
    b_ch = arcsinh_stretch(g_data, scale=scale, clip_percentile=clip_percentile)
    return np.stack([r_ch, g_ch, b_ch], axis=-1)


def mask_overlay(gray_bg, mask, alpha=0.45):
    """Overlay a semi-transparent red mask on a grayscale background."""
    rgb = np.stack([gray_bg, gray_bg, gray_bg], axis=-1)
    m = mask > 0
    rgb[m, 0] = rgb[m, 0] * (1 - alpha) + alpha
    rgb[m, 1] = rgb[m, 1] * (1 - alpha)
    rgb[m, 2] = rgb[m, 2] * (1 - alpha)
    return np.clip(rgb, 0, 1)


def find_bright_peaks(image, n=3, sigma=8, min_distance=80):
    """Find up to ``n`` local peaks in a smoothed image, separated by at
    least ``min_distance`` pixels. Returns a list of ``(x, y, value)``.
    """
    sm = ndimage.gaussian_filter(image, sigma=sigma)
    peaks = []
    working = sm.copy()
    h, w = working.shape
    for _ in range(n):
        py, px = np.unravel_index(np.argmax(working), working.shape)
        val = float(working[py, px])
        if val <= 0:
            break
        peaks.append((int(px), int(py), val))
        yy, xx = np.ogrid[:h, :w]
        zero = (xx - px) ** 2 + (yy - py) ** 2 <= min_distance ** 2
        working[zero] = 0
    return peaks


def draw_markers(ax, cx_frame, cy_frame, target_x, target_y, peaks):
    """Add center, target, and contaminant peak markers to an axis."""
    ax.plot(cx_frame, cy_frame, marker="+", color="cyan", ms=12, mew=1.5,
            label="frame center")
    ax.plot(target_x, target_y, marker="x", color="yellow", ms=10, mew=2,
            label="target center")
    for i, (px, py, _) in enumerate(peaks):
        ax.plot(px, py, marker="o", mfc="none", mec="magenta", ms=14, mew=1.5,
                label="bright peak" if i == 0 else None)


def main():
    fig, axes = plt.subplots(
        len(GALAXIES), 3, figsize=(13.5, 4.5 * len(GALAXIES)),
    )

    for row, (obj_id, desc) in enumerate(GALAXIES):
        galaxy_dir = DATA_DIR / obj_id
        g = fits.getdata(galaxy_dir / f"{obj_id}_HSC_G_image.fits")
        r = fits.getdata(galaxy_dir / f"{obj_id}_HSC_R_image.fits")
        with fits.open(galaxy_dir / f"{obj_id}_HSC_I_image.fits") as hdul:
            i = hdul[0].data
            hdr = hdul[0].header
        target_x = float(hdr.get("X_OBJ", i.shape[1] / 2.0))
        target_y = float(hdr.get("Y_OBJ", i.shape[0] / 2.0))
        mask = fits.getdata(galaxy_dir / f"{obj_id}_HSC_I_mask_default.fits")

        h, w = i.shape
        cx_frame, cy_frame = w / 2.0, h / 2.0
        peaks = find_bright_peaks(i, n=3, sigma=8, min_distance=120)

        # Col 0: color composite
        ax = axes[row, 0]
        ax.imshow(make_color_image(g, r, i), origin="lower", aspect="equal")
        ax.set_title(f"{obj_id}\n{desc}", fontsize=9, fontweight="bold")
        draw_markers(ax, cx_frame, cy_frame, target_x, target_y, peaks)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.legend(loc="upper left", fontsize=7, framealpha=0.7)

        # Col 1: I-band + default mask overlay
        ax = axes[row, 1]
        gray = arcsinh_stretch(i)
        ax.imshow(mask_overlay(gray, mask), origin="lower", aspect="equal")
        mask_pct = float(np.sum(mask > 0)) / mask.size * 100
        ax.set_title(f"I + default mask ({mask_pct:.1f}%)", fontsize=9)
        draw_markers(ax, cx_frame, cy_frame, target_x, target_y, peaks)
        ax.set_xticks([]); ax.set_yticks([])

        # Col 2: I-band asinh (no mask) — for spotting extent + contaminants
        ax = axes[row, 2]
        ax.imshow(gray, origin="lower", aspect="equal", cmap="gray")
        ax.set_title("I-band asinh (no mask)", fontsize=9)
        draw_markers(ax, cx_frame, cy_frame, target_x, target_y, peaks)
        # 200 px scale bar
        ax.plot([40, 240], [40, 40], color="white", lw=2.5)
        ax.text(140, 55, "200 px", color="white", ha="center", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        "Real HSC edge cases: overview",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    out_path = OUTPUT_DIR / "qa_overview.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved QA overview to {out_path}")


if __name__ == "__main__":
    main()
