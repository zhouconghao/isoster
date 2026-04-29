#!/usr/bin/env python3
"""Detection-based object masks for the asteris denoised demo set.

Adapted from ``examples/example_hsc_edge_real/build_custom_masks.py``. The
asteris cutouts ship without an HSC ``VAR`` map, so the detection threshold
falls back to ``Background2D`` RMS — there is no ``--no-variance`` toggle.

Detection is run on the **noisy** i-band image (where contaminants are most
visible) and the same blended mask is reused later for the denoised image,
the noisy fit, and the forced-photometry pass on the denoised image.

Two passes:

1. **aggressive outer pass** — low threshold + generous dilation, picks up
   faint field sources in the wings.
2. **careful inner pass** — moderate threshold + small dilation + per-segment
   peak-flux filter, keeps BCG / disk substructure intact.

The two masks are radially blended around the smoothed-peak anchor
(``r < r_inner`` careful only, ``r > r_outer`` aggressive only, in-between
union). The anchor is found by a smoothed-peak search inside a small box
around the frame center (the asteris cutouts are centered on the target).

Outputs are written next to the source FITS in
``~/Downloads/asteris/objs/<obj_full>/HSC-I/`` so they live alongside the
data, plus a copy of the QA PNG goes under
``outputs/example_asteris_denoised/<obj_id>/``.

Usage
-----
    uv run python examples/example_asteris_denoised/build_object_mask.py
    uv run python examples/example_asteris_denoised/build_object_mask.py --only 37484563299062823
"""

from __future__ import annotations

import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.io import fits
from matplotlib.patches import Circle as MplCircle
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import deblend_sources, detect_sources
from scipy import ndimage

from common import GALAXIES, galaxy_dir, output_dir

DETECT_BAND = "HSC-I"
DETECT_FILE = "noisy.fits"
MASK_FILENAME = "object_mask.fits"

# Single shared default config — galaxies are similar in size and geometry,
# so we don't ship per-galaxy overrides yet. Edit GALAXY_OVERRIDES below if
# a specific cutout needs tuning.
DETECTION_DEFAULTS = {
    "aggressive": {
        "box_size": 128,
        "filter_size": 3,
        "detect_fwhm": 3.5,
        "nsigma": 1.5,
        "npixels": 8,
        "deblend_nlevels": 32,
        "deblend_contrast": 0.001,
        "dilate_fwhm": 10.0,
        "dilate_threshold": 0.02,
        "min_peak": None,
    },
    "careful": {
        "box_size": 32,
        "filter_size": 3,
        "detect_fwhm": 2.0,
        "nsigma": 1.5,
        "npixels": 6,
        "deblend_nlevels": 64,
        "deblend_contrast": 0.0001,
        "dilate_fwhm": 4.0,
        "dilate_threshold": 0.02,
        "min_peak": 0.5,
    },
}

# Per-galaxy overrides keyed by short id. Anchor falls back to frame center
# when ``target_center`` is omitted (the asteris cutouts are centered).
GALAXY_OVERRIDES: dict = {
    # "37484563299062823": {"r_inner": 120.0, "r_outer": 240.0},
}

DEFAULT_RADII = {"r_inner": 120.0, "r_outer": 240.0}


# ---------------------------------------------------------------------------
# Helpers (same logic as build_custom_masks.py, trimmed for this dataset)
# ---------------------------------------------------------------------------
def smoothed_peak(image, center_xy, box=80, smooth_sigma=2.0):
    h, w = image.shape
    cx, cy = center_xy
    half = box // 2
    x0 = max(int(cx - half), 0)
    x1 = min(int(cx + half), w)
    y0 = max(int(cy - half), 0)
    y1 = min(int(cy + half), h)
    cut = image[y0:y1, x0:x1]
    kernel = Gaussian2DKernel(smooth_sigma)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        smoothed = convolve(cut, kernel, normalize_kernel=True)
    iy, ix = np.unravel_index(int(np.argmax(smoothed)), smoothed.shape)
    return float(ix + x0), float(iy + y0)


def gaussian_dilation_kernel(fwhm):
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return Gaussian2DKernel(sigma)


def dilate_bool(mask_bool, fwhm, threshold=0.01):
    if not mask_bool.any() or fwhm <= 0:
        return mask_bool.astype(bool)
    kernel = gaussian_dilation_kernel(fwhm)
    conv = convolve(
        mask_bool.astype(float),
        np.asarray(kernel),
        normalize_kernel=True,
    )
    return conv > threshold


def radial_distance(shape, cx, cy):
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    return np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)


def merge_params(preset_name, overrides):
    params = dict(DETECTION_DEFAULTS[preset_name])
    params.update(overrides or {})
    return params


def run_detection_pass(image, params, target_anchor):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bkg = Background2D(
            image,
            box_size=params["box_size"],
            filter_size=params["filter_size"],
            bkg_estimator=MedianBackground(),
        )
    subtracted = image - bkg.background
    rms = np.asarray(bkg.background_rms)

    detect_kernel = gaussian_dilation_kernel(params["detect_fwhm"])
    convolved = convolve(subtracted, detect_kernel, normalize_kernel=True)

    threshold = params["nsigma"] * rms
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        segmap = detect_sources(convolved, threshold, npixels=params["npixels"])

    if segmap is None or segmap.nlabels == 0:
        return (
            np.zeros(image.shape, dtype=bool),
            {"n_labels": 0, "target_label": -1, "n_contaminants": 0, "coverage": 0.0},
        )

    if segmap.nlabels > 1:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                segmap = deblend_sources(
                    convolved,
                    segmap,
                    npixels=params["npixels"],
                    nlevels=params["deblend_nlevels"],
                    contrast=params["deblend_contrast"],
                    progress_bar=False,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"    deblend_sources failed ({exc}); keeping un-deblended map")

    seg_data = np.asarray(segmap.data)
    labels = np.array(segmap.labels, dtype=int)

    ax = int(round(target_anchor[0]))
    ay = int(round(target_anchor[1]))
    target_label = int(seg_data[ay, ax]) if seg_data[ay, ax] > 0 else -1
    if target_label <= 0:
        best_dist = np.inf
        for lab in labels:
            ys, xs = np.where(seg_data == lab)
            if len(xs) == 0:
                continue
            dist = np.hypot(xs.mean() - target_anchor[0], ys.mean() - target_anchor[1])
            if dist < best_dist:
                best_dist = dist
                target_label = int(lab)

    contam_labels = [lab for lab in labels if lab != target_label]

    min_peak = params.get("min_peak")
    if min_peak is not None and len(contam_labels) > 0:
        peaks = ndimage.maximum(subtracted, labels=seg_data, index=contam_labels)
        peaks = np.atleast_1d(peaks)
        contam_labels = [
            lab for lab, p in zip(contam_labels, peaks)
            if np.isfinite(p) and p >= min_peak
        ]

    contam_mask = np.isin(seg_data, contam_labels)
    dilated = dilate_bool(
        contam_mask,
        fwhm=params["dilate_fwhm"],
        threshold=params["dilate_threshold"],
    )
    dilated[seg_data == target_label] = False

    info = {
        "n_labels": int(segmap.nlabels),
        "target_label": int(target_label),
        "n_contaminants": int(len(contam_labels)),
        "coverage": float(dilated.sum()) / dilated.size,
    }
    return dilated.astype(bool), info


def radial_blend(mask_careful, mask_aggressive, anchor, r_inner, r_outer):
    rdist = radial_distance(mask_careful.shape, anchor[0], anchor[1])
    inner = rdist < r_inner
    outer = rdist > r_outer
    transition = ~inner & ~outer
    out = np.zeros(mask_careful.shape, dtype=bool)
    out[inner] = mask_careful[inner]
    out[outer] = mask_aggressive[outer]
    out[transition] = mask_careful[transition] | mask_aggressive[transition]
    return out


# ---------------------------------------------------------------------------
# QA figure
# ---------------------------------------------------------------------------
def arcsinh_stretch(image, scale=0.02, clip_percentile=99.7):
    vmax = np.nanpercentile(image, clip_percentile)
    normed = np.clip(image, 0, vmax) / max(vmax, 1e-10)
    return np.clip(np.arcsinh(normed / scale) / np.arcsinh(1.0 / scale), 0, 1)


def overlay(gray, mask, alpha=0.45, color=(1.0, 0.0, 0.0)):
    rgb = np.stack([gray, gray, gray], axis=-1)
    m = mask > 0
    for c in range(3):
        rgb[m, c] = rgb[m, c] * (1 - alpha) + color[c] * alpha
    return np.clip(rgb, 0, 1)


def plot_comparison(obj_id, desc, image, anchor, mask_aggr, mask_care, mask_blend,
                    info_aggr, info_care, r_inner, r_outer, qa_path):
    gray = arcsinh_stretch(image)
    h, w = image.shape

    fig, axes = plt.subplots(2, 3, figsize=(14.4, 9.6))

    def _decorate(ax, title):
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(anchor[0], anchor[1], "x", color="yellow", ms=10, mew=1.8)

    def _draw_rings(ax):
        for r, color in [(r_inner, "cyan"), (r_outer, "lime")]:
            ax.add_patch(MplCircle(anchor, r, fill=False, color=color, lw=1.2, linestyle="--"))

    ax = axes[0, 0]
    ax.imshow(gray, origin="lower", cmap="gray")
    _draw_rings(ax)
    _decorate(ax, f"{DETECT_BAND} noisy (asinh) — inner={r_inner:.0f}px outer={r_outer:.0f}px")

    ax = axes[0, 1]
    pct = float(mask_aggr.sum()) / mask_aggr.size * 100
    ax.imshow(overlay(gray, mask_aggr, color=(1.0, 0.2, 0.2)), origin="lower")
    _draw_rings(ax)
    _decorate(ax, f"aggressive ({pct:.1f}%) — {info_aggr['n_contaminants']}/{info_aggr['n_labels']} segs")

    ax = axes[0, 2]
    pct = float(mask_care.sum()) / mask_care.size * 100
    ax.imshow(overlay(gray, mask_care, color=(0.2, 1.0, 0.4)), origin="lower")
    _draw_rings(ax)
    _decorate(ax, f"careful ({pct:.1f}%) — {info_care['n_contaminants']}/{info_care['n_labels']} segs")

    ax = axes[1, 0]
    pct = float(mask_blend.sum()) / mask_blend.size * 100
    ax.imshow(overlay(gray, mask_blend, color=(1.0, 0.6, 0.1)), origin="lower")
    _draw_rings(ax)
    _decorate(ax, f"blended ({pct:.1f}%)")

    zoom_half = 200
    x0 = max(int(anchor[0] - zoom_half), 0)
    x1 = min(int(anchor[0] + zoom_half), w)
    y0 = max(int(anchor[1] - zoom_half), 0)
    y1 = min(int(anchor[1] + zoom_half), h)
    ax = axes[1, 1]
    gray_zoom = arcsinh_stretch(image[y0:y1, x0:x1])
    ax.imshow(
        overlay(gray_zoom, mask_blend[y0:y1, x0:x1], color=(1.0, 0.6, 0.1)),
        origin="lower",
    )
    ax.plot(anchor[0] - x0, anchor[1] - y0, "x", color="yellow", ms=10, mew=1.8)
    ax.add_patch(
        MplCircle(
            (anchor[0] - x0, anchor[1] - y0),
            r_inner,
            fill=False, color="cyan", lw=1.2, linestyle="--",
        )
    )
    ax.set_title("central 400 px zoom — blended", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1, 2]
    diff_rgb = np.stack([gray, gray, gray], axis=-1)
    only_aggr = mask_aggr & ~mask_care
    only_care = mask_care & ~mask_aggr
    for c, val in enumerate((1.0, 0.25, 0.25)):
        diff_rgb[only_aggr, c] = diff_rgb[only_aggr, c] * 0.4 + val * 0.6
    for c, val in enumerate((0.2, 1.0, 0.4)):
        diff_rgb[only_care, c] = diff_rgb[only_care, c] * 0.4 + val * 0.6
    ax.imshow(np.clip(diff_rgb, 0, 1), origin="lower")
    _draw_rings(ax)
    _decorate(ax, "red=aggressive only  green=careful only")

    fig.suptitle(
        f"{obj_id} — {desc}  [anchor=({anchor[0]:.1f}, {anchor[1]:.1f})]",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(qa_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FITS I/O
# ---------------------------------------------------------------------------
def write_mask_fits(path, mask, header_extras):
    hdu = fits.PrimaryHDU(data=mask.astype(np.uint8))
    for key, value in header_extras.items():
        hdu.header[key] = value
    hdu.header["COMMENT"] = "True (1) = masked pixel."
    hdu.writeto(path, overwrite=True)


def process_galaxy(obj_id: str, desc: str):
    overrides = GALAXY_OVERRIDES.get(obj_id, {})
    r_inner = float(overrides.get("r_inner", DEFAULT_RADII["r_inner"]))
    r_outer = float(overrides.get("r_outer", DEFAULT_RADII["r_outer"]))

    src_dir = galaxy_dir(obj_id, DETECT_BAND)
    image_path = src_dir / DETECT_FILE
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    image = fits.getdata(image_path).astype(np.float64)

    h, w = image.shape
    frame_center = ((w - 1) / 2.0, (h - 1) / 2.0)
    target_xy = overrides.get("target_center", frame_center)
    anchor = smoothed_peak(image, target_xy, box=80, smooth_sigma=2.0)
    print(f"\n=== {obj_id}: {desc} ===")
    print(
        f"  shape=({h}, {w}); frame center {frame_center}; "
        f"smoothed-peak anchor ({anchor[0]:.1f}, {anchor[1]:.1f})"
    )

    aggr_params = merge_params("aggressive", overrides.get("aggressive"))
    care_params = merge_params("careful", overrides.get("careful"))

    print("  running aggressive outer pass ...")
    mask_aggr, info_aggr = run_detection_pass(image, aggr_params, anchor)
    print(
        f"    {info_aggr['n_labels']} segments, "
        f"{info_aggr['n_contaminants']} contaminants, "
        f"coverage {info_aggr['coverage'] * 100:.1f}%"
    )

    print("  running careful inner pass ...")
    mask_care, info_care = run_detection_pass(image, care_params, anchor)
    print(
        f"    {info_care['n_labels']} segments, "
        f"{info_care['n_contaminants']} contaminants, "
        f"coverage {info_care['coverage'] * 100:.1f}%"
    )

    mask_blend = radial_blend(
        mask_care, mask_aggr, anchor=anchor, r_inner=r_inner, r_outer=r_outer
    )
    blend_pct = float(mask_blend.sum()) / mask_blend.size * 100
    print(f"  blended mask coverage: {blend_pct:.2f}%")

    header_extras = {
        "OBJECT": obj_id,
        "MASKTYPE": "asteris denoised demo: detection-based blended mask",
        "DESCRIP": desc,
        "DETBAND": DETECT_BAND,
        "DETFILE": DETECT_FILE,
        "X_OBJ": (float(anchor[0]), "anchor x (0-indexed, smoothed peak)"),
        "Y_OBJ": (float(anchor[1]), "anchor y (0-indexed, smoothed peak)"),
        "R_INNER": (float(r_inner), "inner blend radius (px)"),
        "R_OUTER": (float(r_outer), "outer blend radius (px)"),
        "AG_NSIG": (float(aggr_params["nsigma"]), "aggressive nsigma"),
        "CA_NSIG": (float(care_params["nsigma"]), "careful nsigma"),
        "CA_MPEAK": (
            float(care_params["min_peak"]) if care_params.get("min_peak") is not None else 0.0,
            "careful min-peak",
        ),
    }
    mask_path = src_dir / MASK_FILENAME
    write_mask_fits(mask_path, mask_blend, header_extras)
    print(f"  mask -> {mask_path}")

    out_dir = output_dir(obj_id)
    qa_path = out_dir / f"{obj_id}_mask_compare.png"
    plot_comparison(
        obj_id, desc, image, anchor, mask_aggr, mask_care, mask_blend,
        info_aggr, info_care, r_inner, r_outer, qa_path,
    )
    print(f"  QA figure -> {qa_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only",
        action="append",
        help="Process only this short galaxy id (repeatable).",
    )
    args = parser.parse_args()
    ids = args.only if args.only else [g[0] for g in GALAXIES]
    valid_ids = {g[0]: g for g in GALAXIES}
    for sid in ids:
        if sid not in valid_ids:
            print(f"ERROR: unknown id {sid}")
            continue
        _, _, desc = valid_ids[sid]
        process_galaxy(sid, desc)


if __name__ == "__main__":
    main()
