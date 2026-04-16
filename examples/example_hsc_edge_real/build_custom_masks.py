#!/usr/bin/env python3
"""Detection-based custom object masks for the real HSC edge cases.

The HDF bitplane mask is intentionally ignored. Instead, each galaxy
gets a two-pass photutils segmentation run on its i-band science image:

1. **Outer aggressive pass** — low threshold, generous deblend, large
   Gaussian dilation. Catches faint field sources and the broad wings
   of bright stars far from the BCG.
2. **Inner careful pass** — moderate threshold, gentle deblend, small
   dilation, plus a per-segment peak-flux filter. Only bright
   companions/stars near the core get masked, so BCG substructure is
   preserved.

The two masks are then radially blended around the target anchor:

    r < r_inner           → careful mask
    r_inner ≤ r ≤ r_outer → union(careful, aggressive)
    r > r_outer           → aggressive mask

All three galaxies in ``data/`` have their BCG peak within ~2 px of
the frame center (verified 2026-04-16), so the target anchor is
defined as the smoothed-peak location inside a small box around the
frame center. The target segment (the one containing that anchor) is
always excluded from both passes before dilation.

The same blended mask is written for all three bands, matching the
``{obj_id}_{band}_mask_custom.fits`` naming used by the downstream
``example_hsc_edgecases`` runners. A band-agnostic
``{obj_id}_mask_custom.fits`` is also written for quick inspection.

Detection threshold
-------------------
By default both passes use the HSC pipeline ``VAR`` map (variance in
image counts², extracted to ``{obj_id}_HSC_I_variance.fits``) to build
a per-pixel threshold ``nsigma * sqrt(var * var_scale)``. This is
better than a boxy ``Background2D`` RMS for two reasons:

* Near bright sources (e.g. the BCG envelope) the per-pixel variance
  rises with source photon noise, so faint substructure is less
  likely to get picked up by the careful pass.
* Over clean sky the per-pixel sigma is constant at the true sky
  noise, so small isolated bright sources are easier to detect than
  when the local ``Background2D`` box happens to include a neighbor.

Pass ``--no-variance`` to fall back to ``Background2D`` RMS for A/B
comparison.

Usage
-----
    uv run python examples/example_hsc_edge_real/build_custom_masks.py
    uv run python examples/example_hsc_edge_real/build_custom_masks.py --only 42310032070569600
    uv run python examples/example_hsc_edge_real/build_custom_masks.py --no-variance
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.io import fits
from matplotlib.patches import Circle as MplCircle
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import deblend_sources, detect_sources
from scipy import ndimage

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent
BANDS = ["HSC-G", "HSC-R", "HSC-I"]
DETECT_BAND = "HSC-I"


# ---------------------------------------------------------------------------
# Detection parameter presets
# ---------------------------------------------------------------------------
# "aggressive" = outer pass; "careful" = inner pass.
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
        # Use the HSC VAR map for per-pixel thresholds. The HDF ``VAR``
        # field is variance in image units squared (verified against
        # sigma-clipped sky std, ratio ~1.2 due to coadd correlated
        # noise — see ``var_scale`` to compensate).
        "use_variance": True,
        "var_scale": 1.0,
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
        "min_peak": 2.0,
        "use_variance": True,
        "var_scale": 1.0,
    },
}

# ---------------------------------------------------------------------------
# Per-galaxy configuration
# ---------------------------------------------------------------------------
GALAXY_CONFIGS = {
    "37498869835124888": {
        "description": "cluster BCG with multiple bright companions",
        "target_center": (595, 595),
        "r_inner": 140.0,
        "r_outer": 240.0,
        "aggressive": {},
        "careful": {"min_peak": 1.0},
    },
    "42177291811318246": {
        "description": "BCG with bright NW companion + bright star near edge",
        "target_center": (596, 595),
        "r_inner": 140.0,
        "r_outer": 240.0,
        "aggressive": {"dilate_fwhm": 14.0},
        "careful": {"min_peak": 1.0},
    },
    "42310032070569600": {
        "description": "BCG with bright halo star + blended bright source",
        "target_center": (596, 595),
        "r_inner": 150.0,
        "r_outer": 260.0,
        "aggressive": {"dilate_fwhm": 20.0},
        "careful": {"min_peak": 1.0},
    },
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def smoothed_peak(image, center_xy, box=60, smooth_sigma=2.0):
    """Return the smoothed brightest-pixel location within a box around center_xy.

    Used as the detection anchor so we don't rely on a single noisy pixel.
    """
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
    """Smooth dilation via Gaussian convolution + threshold.

    Produces a softer edge than a morphological dilation and matches the
    pattern already used in ``example_ls_highorder_harmonic/masking.py``.
    """
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


# ---------------------------------------------------------------------------
# Detection pass
# ---------------------------------------------------------------------------
def merge_params(preset_name, overrides):
    params = dict(DETECTION_DEFAULTS[preset_name])
    params.update(overrides or {})
    return params


def run_detection_pass(image, params, target_anchor, variance=None):
    """Run one background + detect + deblend pass.

    Parameters
    ----------
    image : 2D float ndarray
        Science image (background not pre-subtracted).
    params : dict
        Detection parameters (see ``DETECTION_DEFAULTS``).
    target_anchor : (x, y)
        Pixel coordinates of the target galaxy center.
    variance : 2D float ndarray or None
        Per-pixel variance (image units squared). When
        ``params['use_variance']`` is True and this is provided, the
        detection threshold becomes ``nsigma * sqrt(variance *
        var_scale)`` — a per-pixel threshold that rises near bright
        sources (so BCG substructure doesn't get over-detected) and
        stays low over clean sky (so small isolated bright sources
        are easier to pick up than with a boxy Background2D RMS).

    Returns
    -------
    mask : bool ndarray
        True on masked (contaminant) pixels, with the target segment
        already excluded. Dilation is applied here.
    info : dict
        Bookkeeping (n_labels, target_label, n_contaminants, coverage,
        threshold_source).
    """
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

    use_var = bool(params.get("use_variance", True)) and variance is not None
    if use_var:
        var_scale = float(params.get("var_scale", 1.0))
        var_eff = np.asarray(variance, dtype=np.float64) * var_scale
        # Guard against zero / negative / non-finite variance pixels —
        # fall back to the Background2D RMS there so the threshold is
        # never 0 (which would flag every pixel as a detection).
        sigma_map = np.zeros_like(var_eff)
        good = np.isfinite(var_eff) & (var_eff > 0)
        sigma_map[good] = np.sqrt(var_eff[good])
        sigma_map[~good] = rms[~good]
        threshold = params["nsigma"] * sigma_map
        threshold_source = "variance"
    else:
        threshold = params["nsigma"] * rms
        threshold_source = "background_rms"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        segmap = detect_sources(convolved, threshold, npixels=params["npixels"])

    if segmap is None or segmap.nlabels == 0:
        return (
            np.zeros(image.shape, dtype=bool),
            {"n_labels": 0, "target_label": -1, "kept_labels": 0},
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

    # Target segment = label at the anchor pixel (fall back to nearest label).
    ax = int(round(target_anchor[0]))
    ay = int(round(target_anchor[1]))
    target_label = int(seg_data[ay, ax]) if seg_data[ay, ax] > 0 else -1
    if target_label <= 0:
        # Pick the label whose centroid is closest to the anchor.
        best_dist = np.inf
        for lab in labels:
            ys, xs = np.where(seg_data == lab)
            if len(xs) == 0:
                continue
            dist = np.hypot(xs.mean() - target_anchor[0], ys.mean() - target_anchor[1])
            if dist < best_dist:
                best_dist = dist
                target_label = int(lab)

    # Start from all non-target segments.
    contam_labels = [lab for lab in labels if lab != target_label]

    # Peak-flux filter (careful pass only cares about bright contaminants).
    min_peak = params.get("min_peak")
    if min_peak is not None and len(contam_labels) > 0:
        peaks = ndimage.maximum(
            subtracted,
            labels=seg_data,
            index=contam_labels,
        )
        peaks = np.atleast_1d(peaks)
        kept = [lab for lab, p in zip(contam_labels, peaks) if np.isfinite(p) and p >= min_peak]
        contam_labels = kept

    contam_mask = np.isin(seg_data, contam_labels)
    dilated = dilate_bool(
        contam_mask,
        fwhm=params["dilate_fwhm"],
        threshold=params["dilate_threshold"],
    )

    # Make sure dilation doesn't accidentally eat the target segment.
    dilated[seg_data == target_label] = False

    info = {
        "n_labels": int(segmap.nlabels),
        "target_label": int(target_label),
        "n_contaminants": int(len(contam_labels)),
        "coverage": float(dilated.sum()) / dilated.size,
        "threshold_source": threshold_source,
    }
    return dilated.astype(bool), info


# ---------------------------------------------------------------------------
# Radial blending
# ---------------------------------------------------------------------------
def radial_blend(mask_careful, mask_aggressive, anchor, r_inner, r_outer):
    """Radially merge the inner and outer masks.

    r < r_inner  → careful; r > r_outer → aggressive; otherwise union.
    """
    shape = mask_careful.shape
    rdist = radial_distance(shape, anchor[0], anchor[1])

    inner_zone = rdist < r_inner
    outer_zone = rdist > r_outer
    transition = ~inner_zone & ~outer_zone

    out = np.zeros(shape, dtype=bool)
    out[inner_zone] = mask_careful[inner_zone]
    out[outer_zone] = mask_aggressive[outer_zone]
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


def plot_comparison(
    obj_id,
    config,
    image,
    anchor,
    mask_aggressive,
    mask_careful,
    mask_blend,
    info_aggr,
    info_care,
):
    """Six-panel QA figure: i-band + three masks + blend + central zoom."""
    gray = arcsinh_stretch(image)
    h, w = image.shape
    r_in = config["r_inner"]
    r_out = config["r_outer"]

    fig, axes = plt.subplots(2, 3, figsize=(14.4, 9.6))

    def _decorate(ax, title):
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(anchor[0], anchor[1], "x", color="yellow", ms=10, mew=1.8)

    def _draw_rings(ax):
        for r, color, ls in [(r_in, "cyan", "--"), (r_out, "lime", "--")]:
            ax.add_patch(MplCircle(anchor, r, fill=False, color=color, lw=1.2, linestyle=ls))

    # Panel 0: i-band + inner/outer rings
    ax = axes[0, 0]
    ax.imshow(gray, origin="lower", cmap="gray")
    _draw_rings(ax)
    _decorate(ax, f"{DETECT_BAND} (asinh) — inner={r_in:.0f}px outer={r_out:.0f}px")

    # Panel 1: aggressive mask overlay
    ax = axes[0, 1]
    pct = float(mask_aggressive.sum()) / mask_aggressive.size * 100
    ax.imshow(overlay(gray, mask_aggressive, color=(1.0, 0.2, 0.2)), origin="lower")
    _draw_rings(ax)
    _decorate(
        ax,
        f"aggressive outer ({pct:.1f}%) — {info_aggr['n_contaminants']}/{info_aggr['n_labels']} segs",
    )

    # Panel 2: careful mask overlay
    ax = axes[0, 2]
    pct = float(mask_careful.sum()) / mask_careful.size * 100
    ax.imshow(overlay(gray, mask_careful, color=(0.2, 1.0, 0.4)), origin="lower")
    _draw_rings(ax)
    _decorate(
        ax,
        f"careful inner ({pct:.1f}%) — {info_care['n_contaminants']}/{info_care['n_labels']} segs",
    )

    # Panel 3: blended mask overlay
    ax = axes[1, 0]
    pct = float(mask_blend.sum()) / mask_blend.size * 100
    ax.imshow(overlay(gray, mask_blend, color=(1.0, 0.6, 0.1)), origin="lower")
    _draw_rings(ax)
    _decorate(ax, f"blended ({pct:.1f}%)")

    # Panel 4: central zoom (400 px around anchor) with blended mask
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
            r_in,
            fill=False,
            color="cyan",
            lw=1.2,
            linestyle="--",
        )
    )
    ax.set_title("central 400 px zoom — blended", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    # Panel 5: difference map — aggressive only vs careful only
    ax = axes[1, 2]
    diff_rgb = np.stack([gray, gray, gray], axis=-1)
    only_aggr = mask_aggressive & ~mask_careful
    only_care = mask_careful & ~mask_aggressive
    for c, val in enumerate((1.0, 0.25, 0.25)):
        diff_rgb[only_aggr, c] = diff_rgb[only_aggr, c] * 0.4 + val * 0.6
    for c, val in enumerate((0.2, 1.0, 0.4)):
        diff_rgb[only_care, c] = diff_rgb[only_care, c] * 0.4 + val * 0.6
    ax.imshow(np.clip(diff_rgb, 0, 1), origin="lower")
    _draw_rings(ax)
    _decorate(
        ax,
        f"red=aggressive only  green=careful only",
    )

    fig.suptitle(
        f"{obj_id} — {config['description']}  [anchor=({anchor[0]:.1f}, {anchor[1]:.1f})]",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = OUTPUT_DIR / f"{obj_id}_mask_compare.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# FITS I/O
# ---------------------------------------------------------------------------
def write_mask_fits(path, mask, header_extras):
    hdu = fits.PrimaryHDU(data=mask.astype(np.uint8))
    for key, value in header_extras.items():
        hdu.header[key] = value
    hdu.header["COMMENT"] = "True (1) = masked pixel."
    hdu.writeto(path, overwrite=True)


def process_galaxy(obj_id, config, use_variance=True):
    galaxy_dir = DATA_DIR / obj_id
    print(f"\n=== {obj_id}: {config['description']} ===")

    band_safe = DETECT_BAND.replace("-", "_")
    i_path = galaxy_dir / f"{obj_id}_{band_safe}_image.fits"
    image = fits.getdata(i_path).astype(np.float64)

    # HSC HDF "VAR" is variance in image counts² — verified empirically
    # by sqrt(sigma-clipped VAR median) ≈ sigma-clipped sky std × ~1.2.
    # The ~1.2 factor is HSC coadd correlated-noise; compensate via
    # ``var_scale`` in the preset if you want to mimic the empirical RMS.
    variance = None
    if use_variance:
        var_path = galaxy_dir / f"{obj_id}_{band_safe}_variance.fits"
        if var_path.exists():
            variance = fits.getdata(var_path).astype(np.float64)
        else:
            print(f"  WARNING: {var_path.name} not found, falling back to Background2D RMS")

    frame_cx = (image.shape[1] - 1) / 2.0
    frame_cy = (image.shape[0] - 1) / 2.0
    cx0, cy0 = config["target_center"]
    anchor = smoothed_peak(image, (cx0, cy0), box=80, smooth_sigma=2.0)
    print(
        f"  frame center ({frame_cx:.1f}, {frame_cy:.1f}), "
        f"config target ({cx0}, {cy0}), "
        f"smoothed anchor ({anchor[0]:.1f}, {anchor[1]:.1f})"
    )
    print(f"  variance: {'loaded ' + str(variance.shape) if variance is not None else 'not used'}")

    aggr_params = merge_params("aggressive", config.get("aggressive"))
    care_params = merge_params("careful", config.get("careful"))
    if not use_variance:
        aggr_params["use_variance"] = False
        care_params["use_variance"] = False

    print("  running aggressive outer pass ...")
    mask_aggr, info_aggr = run_detection_pass(image, aggr_params, anchor, variance=variance)
    print(
        f"    {info_aggr['n_labels']} segments, "
        f"{info_aggr['n_contaminants']} kept as contaminants, "
        f"coverage {info_aggr['coverage'] * 100:.1f}% "
        f"[threshold={info_aggr['threshold_source']}]"
    )

    print("  running careful inner pass ...")
    mask_care, info_care = run_detection_pass(image, care_params, anchor, variance=variance)
    print(
        f"    {info_care['n_labels']} segments, "
        f"{info_care['n_contaminants']} kept as contaminants, "
        f"coverage {info_care['coverage'] * 100:.1f}% "
        f"[threshold={info_care['threshold_source']}]"
    )

    mask_blend = radial_blend(
        mask_care,
        mask_aggr,
        anchor=anchor,
        r_inner=config["r_inner"],
        r_outer=config["r_outer"],
    )
    blend_pct = float(mask_blend.sum()) / mask_blend.size * 100
    print(f"  blended mask coverage: {blend_pct:.1f}%")

    header_base = {
        "OBJECT": obj_id,
        "MASKTYPE": "detection-based object mask",
        "DESCRIP": config["description"],
        "DETBAND": DETECT_BAND,
        "X_OBJ": (float(anchor[0]), "anchor x (0-indexed, smoothed peak)"),
        "Y_OBJ": (float(anchor[1]), "anchor y (0-indexed, smoothed peak)"),
        "R_INNER": (float(config["r_inner"]), "inner blend radius (px)"),
        "R_OUTER": (float(config["r_outer"]), "outer blend radius (px)"),
        "AG_NSIG": (float(aggr_params["nsigma"]), "aggressive nsigma"),
        "AG_DFWHM": (float(aggr_params["dilate_fwhm"]), "aggressive dilate fwhm"),
        "CA_NSIG": (float(care_params["nsigma"]), "careful nsigma"),
        "CA_DFWHM": (float(care_params["dilate_fwhm"]), "careful dilate fwhm"),
        "CA_MPEAK": (
            float(care_params["min_peak"]) if care_params.get("min_peak") is not None else 0.0,
            "careful min-peak (image units)",
        ),
        "USE_VAR": (bool(use_variance), "detection used HSC VAR map"),
        "VAR_SCAG": (float(aggr_params.get("var_scale", 1.0)), "aggressive var_scale"),
        "VAR_SCCA": (float(care_params.get("var_scale", 1.0)), "careful var_scale"),
    }

    # Band-agnostic mask (used for quick inspection / color composites).
    write_mask_fits(
        galaxy_dir / f"{obj_id}_mask_custom.fits",
        mask_blend,
        header_base,
    )

    # Per-band mask files — detection is on i-band only, so all three bands
    # receive the same blended mask (downstream runners expect per-band paths).
    for band in BANDS:
        band_safe = band.replace("-", "_")
        extras = dict(header_base)
        extras["FILTER"] = band
        write_mask_fits(
            galaxy_dir / f"{obj_id}_{band_safe}_mask_custom.fits",
            mask_blend,
            extras,
        )
        print(f"  {band}: {blend_pct:.1f}% masked -> {obj_id}_{band_safe}_mask_custom.fits")

    qa_path = plot_comparison(
        obj_id,
        config,
        image,
        anchor,
        mask_aggr,
        mask_care,
        mask_blend,
        info_aggr,
        info_care,
    )
    print(f"  QA figure -> {qa_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Detection-based custom object masks for the three real HSC edge cases (two-pass photutils, radial blend)."
        ),
    )
    parser.add_argument(
        "--only",
        action="append",
        help="Process only this galaxy ID (can be repeated).",
    )
    parser.add_argument(
        "--no-variance",
        action="store_true",
        help=(
            "Disable the HSC VAR map threshold and fall back to the "
            "Background2D RMS for both passes (useful for A/B comparison)."
        ),
    )
    args = parser.parse_args()

    use_variance = not args.no_variance
    ids = args.only if args.only else list(GALAXY_CONFIGS.keys())
    for obj_id in ids:
        if obj_id not in GALAXY_CONFIGS:
            print(f"ERROR: no config for {obj_id}")
            continue
        process_galaxy(obj_id, GALAXY_CONFIGS[obj_id], use_variance=use_variance)


if __name__ == "__main__":
    main()
