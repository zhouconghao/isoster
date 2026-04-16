#!/usr/bin/env python3
"""High-resolution zoomed inspection figures for each real edge-case galaxy.

Writes one PNG per galaxy showing a central region and a full-frame version,
both with contaminant peak markers. This is a one-shot diagnostic used to
plan per-galaxy custom masks — not part of the regular run sequence.

Usage
-----
    uv run python examples/example_hsc_edge_real/qa_zoom.py
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

ZOOM_HALF = 300  # half-width of the central zoom in pixels


def arcsinh_stretch(image, scale=0.02, clip_percentile=99.7):
    vmax = np.nanpercentile(image, clip_percentile)
    normed = np.clip(image, 0, vmax) / max(vmax, 1e-10)
    return np.clip(np.arcsinh(normed / scale) / np.arcsinh(1.0 / scale), 0, 1)


def color_img(g, r, i):
    return np.stack(
        [arcsinh_stretch(i), arcsinh_stretch(r), arcsinh_stretch(g)],
        axis=-1,
    )


def top_peaks(image, n=8, sigma=5, min_distance=60):
    sm = ndimage.gaussian_filter(image, sigma=sigma)
    peaks = []
    work = sm.copy()
    h, w = work.shape
    yy, xx = np.ogrid[:h, :w]
    for _ in range(n):
        py, px = np.unravel_index(np.argmax(work), work.shape)
        val = float(work[py, px])
        if val <= 0:
            break
        peaks.append((int(px), int(py), val))
        zero = (xx - px) ** 2 + (yy - py) ** 2 <= min_distance ** 2
        work[zero] = 0
    return peaks


def main():
    for obj_id, desc in GALAXIES:
        galaxy_dir = DATA_DIR / obj_id
        g = fits.getdata(galaxy_dir / f"{obj_id}_HSC_G_image.fits")
        r = fits.getdata(galaxy_dir / f"{obj_id}_HSC_R_image.fits")
        with fits.open(galaxy_dir / f"{obj_id}_HSC_I_image.fits") as hdul:
            i = hdul[0].data
            hdr = hdul[0].header
        tx = float(hdr.get("X_OBJ", i.shape[1] / 2.0))
        ty = float(hdr.get("Y_OBJ", i.shape[0] / 2.0))
        h, w = i.shape
        fx, fy = w / 2.0, h / 2.0

        peaks = top_peaks(i, n=8, sigma=5, min_distance=60)

        # Print peak info for the plan
        print(f"\n=== {obj_id} ({desc}) ===")
        print(f"  image shape: {h}x{w}, frame center = ({fx:.0f}, {fy:.0f})")
        print(f"  auto target_center = ({tx:.1f}, {ty:.1f})")
        print(f"  top peaks (smoothed sigma=5):")
        for idx, (px, py, val) in enumerate(peaks):
            raw = float(i[py, px])
            dist = np.hypot(px - fx, py - fy)
            print(f"    #{idx+1}: ({px:4d},{py:4d})  smooth={val:7.2f}  "
                  f"raw={raw:7.2f}  dist_from_frame_center={dist:5.0f}")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: full-frame color + markers
        ax = axes[0]
        ax.imshow(color_img(g, r, i), origin="lower", aspect="equal")
        ax.set_title(f"{obj_id}\nfull frame", fontsize=9, fontweight="bold")
        ax.plot(fx, fy, "+", color="cyan", ms=14, mew=1.8)
        ax.plot(tx, ty, "x", color="yellow", ms=12, mew=2)
        for idx, (px, py, _) in enumerate(peaks):
            ax.plot(px, py, "o", mfc="none", mec="magenta", ms=16, mew=1.5)
            ax.text(px + 15, py + 15, f"{idx+1}", color="magenta", fontsize=9,
                    fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

        # Panel 2: central zoom
        y0, y1 = int(fy - ZOOM_HALF), int(fy + ZOOM_HALF)
        x0, x1 = int(fx - ZOOM_HALF), int(fx + ZOOM_HALF)
        ax = axes[1]
        ax.imshow(
            color_img(g[y0:y1, x0:x1], r[y0:y1, x0:x1], i[y0:y1, x0:x1]),
            origin="lower", aspect="equal",
            extent=(x0, x1, y0, y1),
        )
        ax.set_title(f"central {2*ZOOM_HALF}x{2*ZOOM_HALF} px", fontsize=9)
        ax.plot(fx, fy, "+", color="cyan", ms=14, mew=1.8)
        ax.plot(tx, ty, "x", color="yellow", ms=12, mew=2)
        ax.set_xticks([]); ax.set_yticks([])

        # Panel 3: I-band log with peak labels
        ax = axes[2]
        ax.imshow(arcsinh_stretch(i), origin="lower", cmap="gray")
        ax.set_title("I-band asinh", fontsize=9)
        ax.plot(fx, fy, "+", color="cyan", ms=14, mew=1.8)
        ax.plot(tx, ty, "x", color="yellow", ms=12, mew=2)
        for idx, (px, py, _) in enumerate(peaks):
            ax.plot(px, py, "o", mfc="none", mec="magenta", ms=16, mew=1.5)
            ax.text(px + 15, py + 15, f"{idx+1}", color="magenta", fontsize=9,
                    fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

        fig.suptitle(desc, fontsize=11)
        fig.tight_layout()
        out = OUTPUT_DIR / f"qa_zoom_{obj_id}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
