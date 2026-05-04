"""Mask QA figure for the LegacySurvey grz multi-band benchmark.

Renders a single PNG that lets the user decide whether the pre-built
combined mask is good enough or whether it needs editing before the
multi-band fit runs.

Layout (one PNG per galaxy):

  Row 1: image stretch (asinh) per band, with COMBINED mask overlay.
  Row 2: COMBINED, BAD_PIXEL, BRIGHT_STAR, TRACTOR, SEP_HOT, SEP_COLD
         binary masks side by side.
  Row 3: invvar map (log) per band so dead pixels show.
  Row 4: sky-pixel histogram (unmasked pixels only) per band.

The script is self-contained — it does not call any isoster fit code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval

from legacysurvey_loader import load_legacysurvey_grz


_BAND_COLORS = {"g": "#1f77b4", "r": "#2ca02c", "z": "#d62728"}


def _asinh_norm(image: np.ndarray, mask: np.ndarray | None = None) -> ImageNormalize:
    finite = np.isfinite(image)
    if mask is not None:
        finite &= ~mask
    sample = image[finite]
    if sample.size == 0:
        return ImageNormalize(image, stretch=AsinhStretch())
    return ImageNormalize(
        sample,
        interval=PercentileInterval(99.5),
        stretch=AsinhStretch(a=0.05),
    )


def render_mask_qa(galaxy_dir: Path, galaxy_prefix: str, out_path: Path) -> None:
    cutout = load_legacysurvey_grz(galaxy_dir, galaxy_prefix)
    bands = cutout.bands
    n_bands = len(bands)

    layer_names = ["COMBINED", "BAD_PIXEL", "BRIGHT_STAR", "TRACTOR", "SEP_HOT", "SEP_COLD"]
    layers: dict[str, np.ndarray] = {}
    if cutout.combined_mask is not None:
        layers["COMBINED"] = cutout.combined_mask
    layers.update(cutout.extra_mask_layers)

    fig = plt.figure(figsize=(4.0 * max(n_bands, len(layer_names)), 14.5))
    gs = fig.add_gridspec(4, max(n_bands, len(layer_names)), hspace=0.35, wspace=0.10)

    for col, band in enumerate(bands):
        image = cutout.images[col]
        ax = fig.add_subplot(gs[0, col])
        norm = _asinh_norm(image, mask=cutout.combined_mask)
        ax.imshow(image, origin="lower", cmap="Greys_r", norm=norm)
        if cutout.combined_mask.any():
            overlay = np.zeros((*cutout.shape, 4), dtype=np.float32)
            overlay[cutout.combined_mask] = (1.0, 0.2, 0.2, 0.45)
            ax.imshow(overlay, origin="lower", interpolation="nearest")
        masked_frac = cutout.combined_mask.mean() * 100.0
        ax.set_title(
            f"{band}-band image  (combined mask {masked_frac:.2f}%)",
            color=_BAND_COLORS.get(band, "k"),
        )
        ax.set_xticks([])
        ax.set_yticks([])

    for col, layer in enumerate(layer_names):
        ax = fig.add_subplot(gs[1, col])
        if layer in layers:
            mask = layers[layer]
            ax.imshow(mask.astype(np.uint8), origin="lower", cmap="Reds", vmin=0, vmax=1)
            ax.set_title(f"{layer}  ({mask.mean()*100:.2f}%)", fontsize=10)
        else:
            ax.text(
                0.5,
                0.5,
                f"{layer}\n(missing)",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="0.4",
            )
            ax.set_title(layer, fontsize=10, color="0.4")
        ax.set_xticks([])
        ax.set_yticks([])

    for col, band in enumerate(bands):
        variance = cutout.variances[col]
        with np.errstate(divide="ignore", invalid="ignore"):
            log_var = np.log10(np.where(variance > 0, variance, np.nan))
        ax = fig.add_subplot(gs[2, col])
        finite = np.isfinite(log_var)
        if finite.any():
            vmin, vmax = np.percentile(log_var[finite], [1, 99])
        else:
            vmin, vmax = -3.0, 3.0
        ax.imshow(log_var, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        bad_count = int(cutout.invvar_zero_masks[col].sum())
        ax.set_title(
            f"log10(variance) {band}  (invvar≤0 pixels: {bad_count})",
            color=_BAND_COLORS.get(band, "k"),
        )
        ax.set_xticks([])
        ax.set_yticks([])

    for col, band in enumerate(bands):
        image = cutout.images[col]
        good = ~cutout.combined_mask & np.isfinite(image)
        sky = image[good]
        ax = fig.add_subplot(gs[3, col])
        if sky.size:
            lo, hi = np.percentile(sky, [0.5, 99.5])
            ax.hist(
                sky[(sky >= lo) & (sky <= hi)],
                bins=80,
                histtype="stepfilled",
                color=_BAND_COLORS.get(band, "0.5"),
                alpha=0.65,
            )
            ax.axvline(np.median(sky), color="k", lw=0.8, ls="--",
                       label=f"median = {np.median(sky):.3e}")
            ax.axvline(0.0, color="0.3", lw=0.6, ls=":")
            ax.legend(fontsize=8, loc="upper right")
        ax.set_xlabel(f"{band} sky pixel value (nanomaggy)")
        ax.set_ylabel("count")
        ax.set_title(f"{band} unmasked pixel histogram",
                     color=_BAND_COLORS.get(band, "k"))

    fig.suptitle(
        f"{galaxy_prefix}  ({cutout.shape[0]}×{cutout.shape[1]} px,  "
        f"pixel_scale = {cutout.pixel_scale_arcsec:.4f}\" ,  ZP = {cutout.zp})",
        fontsize=14,
        y=0.995,
    )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--galaxy-dir",
        default="/Volumes/galaxy/isophote/sga2020/data/demo/PGC006669",
        type=Path,
    )
    parser.add_argument("--galaxy-prefix", default="PGC006669-largegalaxy")
    parser.add_argument(
        "--out",
        default=Path(
            "outputs/benchmark_multiband/PGC006669/PGC006669_mask_qa.png"
        ),
        type=Path,
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    render_mask_qa(args.galaxy_dir, args.galaxy_prefix, args.out)
    print(f"Saved mask QA figure to {args.out.resolve()}")


if __name__ == "__main__":
    main()
