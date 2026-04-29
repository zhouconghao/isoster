#!/usr/bin/env python3
"""Run isoster on the noisy / denoised i-band pair for one asteris galaxy.

Two-stage workflow:

1. **Free fit on ``noisy.fits``** — anchor at the smoothed peak from the mask
   header, free geometry, debug + harmonics + cog enabled, uniform variance
   from sigma-clipped sky std. Saves the results FITS, builds a 2-D model,
   and writes a standard ``plot_qa_summary`` PNG.

2. **Forced photometry on ``denoised.fits``** — reuse the noisy isophote list
   as a list-of-dicts ``template`` for ``fit_image``. Geometry is frozen,
   only intensities and harmonics are re-extracted. Uniform variance from
   the (lower) denoised sky std. Saves the forced results FITS.

Comparison QA figure has two columns:

* Left: noisy + isophote overlays, denoised + same overlays, and a 2-D
  ``(noisy - denoised) / denoised`` map with a diverging cmap.
* Right: μ(SMA¹ᐟ⁴) for noisy and denoised with errorbars, Δμ
  (noisy - denoised), and the intensity ratio ``I_noisy / I_denoised``.

Usage
-----
    uv run python examples/example_asteris_denoised/run_isoster_pair.py \\
        --only 37484563299062823
"""

from __future__ import annotations

import argparse
import time
import warnings

import matplotlib

matplotlib.rcParams["text.usetex"] = False

import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources
from scipy.ndimage import binary_dilation
from scipy.stats import gaussian_kde

from isoster import fit_image
from isoster.config import IsosterConfig
from isoster.model import build_isoster_model
from isoster.plotting import draw_isophote_overlays, plot_qa_summary
from isoster.utils import isophote_results_to_fits

from common import (
    GALAXIES,
    PIXEL_SCALE_ARCSEC,
    SB_ZEROPOINT,
    galaxy_dir,
    load_pair,
    load_target_anchor_from_mask,
    output_dir,
    sky_std,
    uniform_variance_map,
)

MASK_FILENAME = "object_mask.fits"

# Required fields a template entry needs for forced photometry.
TEMPLATE_KEYS = ("sma", "x0", "y0", "eps", "pa")


def make_template(isophotes):
    """Strip a noisy fit down to the geometry-only fields used as a template.

    Drops the central pixel (sma=0) — forced photometry treats sma=0 as a
    degenerate ring and ``fit_image`` re-injects its own central pixel for
    consistency.
    """
    template = []
    for iso in isophotes:
        sma = float(iso["sma"])
        if sma <= 0.0:
            continue
        template.append({k: float(iso[k]) for k in TEMPLATE_KEYS})
    return template


def intens_to_mu(intens, zeropoint, pixel_scale_arcsec):
    """μ = -2.5·log10(I / pixarea) + zp.  Returns NaN for non-positive I."""
    pixarea = pixel_scale_arcsec * pixel_scale_arcsec
    out = np.full_like(intens, np.nan, dtype=np.float64)
    good = np.isfinite(intens) & (intens > 0)
    out[good] = -2.5 * np.log10(intens[good] / pixarea) + zeropoint
    return out


def mu_error_from_intens(intens, intens_err):
    """1-sigma μ uncertainty from ``intens_err`` (small-error approximation)."""
    intens = np.asarray(intens, dtype=np.float64)
    intens_err = np.asarray(intens_err, dtype=np.float64)
    out = np.full_like(intens, np.nan, dtype=np.float64)
    good = np.isfinite(intens) & np.isfinite(intens_err) & (intens > 0)
    out[good] = (2.5 / np.log(10.0)) * intens_err[good] / intens[good]
    return out


def intens_to_mu_asinh(intens, zeropoint, pixel_scale_arcsec, scale):
    """Asinh surface brightness (Lupton+ 1999), well-defined for any I.

    For per-pixel intensity I in image units, with B = scale / pixarea
    the per-area softening parameter:

        mu_asinh = -2.5 / ln(10) * asinh(I_pp / (2*B))  +  zp - 2.5*log10(B)

    where ``I_pp = I / pixarea``. In the bright limit ``I_pp >> 2B`` this
    converges to the standard ``-2.5*log10(I_pp) + zp`` log10 form. At
    ``I = 0`` it is finite, with value ``zp - 2.5*log10(B)`` — the regime
    where the log10 form returns NaN.
    """
    pixarea = pixel_scale_arcsec * pixel_scale_arcsec
    intens = np.asarray(intens, dtype=np.float64)
    i_pp = intens / pixarea
    big_b = scale / pixarea
    return (
        -2.5 / np.log(10.0) * np.arcsinh(i_pp / (2.0 * big_b))
        + zeropoint
        - 2.5 * np.log10(big_b)
    )


def mu_asinh_error(intens, intens_err, pixel_scale_arcsec, scale):
    """1-sigma error on the asinh surface brightness, computed in the
    *log10 form* with a softening floor so the LSB tail behaves the way
    people are used to seeing.

    The exact derivative ``dmu_asinh/dI`` is mathematically smaller than
    the log10 ``2.5/ln10 * sigma_I/|I|`` whenever ``|I| <~ scale``
    (asinh magnitudes are designed to tame the LSB error). To preserve the
    log10 visual feel while keeping the errors finite when ``I -> 0``, we
    report

        sigma_mu = 2.5/ln(10) * sigma_I / max(|I|, scale).

    This matches the conventional log10 error in the high-S/N regime
    (``|I| > scale``) and saturates at a sensible LSB cap of
    ``2.5/ln10 * sigma_I/scale`` (~ 1 mag for ``sigma_I ~ scale``).
    """
    intens = np.asarray(intens, dtype=np.float64)
    intens_err = np.asarray(intens_err, dtype=np.float64)
    denom = np.maximum(np.abs(intens), scale)
    return (2.5 / np.log(10.0)) * intens_err / denom


def fit_asinh_scale_to_log(intens, intens_err, zeropoint, pixel_scale_arcsec,
                           snr_min=10.0):
    """Choose the asinh ``scale`` so the asinh and log10 SBs agree in the
    high-S/N central region.

    With our parameterization, ``mu_asinh - mu_log10 -> 0`` automatically
    for ``I >> 2*scale``, so any scale much smaller than the bright-end
    intensity works. We anchor it to the per-image sky noise: ``scale =
    median(intens_err) at high SNR`` is a natural choice — it places the
    asinh "linear region" at the noise floor and the log-like region above.

    Returns the chosen scale (in image flux units, same as ``intens``).
    """
    intens = np.asarray(intens, dtype=np.float64)
    intens_err = np.asarray(intens_err, dtype=np.float64)
    bright = np.isfinite(intens) & np.isfinite(intens_err) & (intens > 0)
    bright &= (intens / np.maximum(intens_err, 1e-30)) > snr_min
    if not bright.any():
        # Fallback: use median absolute deviation of all intens_err.
        return float(np.nanmedian(intens_err)) if intens_err.size else 1e-3
    return float(np.median(intens_err[bright]))


def asinh_mu_at_zero_intensity(zeropoint, pixel_scale_arcsec, scale):
    """The asinh surface brightness corresponding to I = 0.

    With the Lupton-1999 form ``mu = -2.5/ln10 * asinh(I_pp/(2B)) + zp -
    2.5*log10(B)`` and ``B = scale/pixarea``, evaluating at I=0 gives:

        mu(I=0) = zp - 2.5 * log10(B) = zp - 2.5*log10(scale/pixarea).
    """
    pixarea = pixel_scale_arcsec * pixel_scale_arcsec
    big_b = scale / pixarea
    return zeropoint - 2.5 * np.log10(big_b)


def collect_profile(isophotes):
    """Pack the per-isophote arrays we need for the comparison plots."""
    sma = np.array([iso["sma"] for iso in isophotes], dtype=np.float64)
    intens = np.array([iso["intens"] for iso in isophotes], dtype=np.float64)
    intens_err = np.array([iso["intens_err"] for iso in isophotes], dtype=np.float64)
    stop = np.array([iso.get("stop_code", 0) for iso in isophotes], dtype=int)
    # Drop the central pixel and any non-finite intensities.
    good = (sma > 0) & np.isfinite(intens)
    return {
        "sma": sma[good],
        "intens": intens[good],
        "intens_err": intens_err[good],
        "stop_code": stop[good],
    }


def align_profiles_by_sma(prof_a, prof_b, atol=1e-6):
    """Align forced-photometry output to its template by SMA.

    Forced photometry preserves the template SMA grid (minus the central
    pixel), so a strict equality match is fine — but ``fit_image`` may drop
    isophotes with empty samples. We intersect both grids on SMA.
    """
    sma_a = prof_a["sma"]
    sma_b = prof_b["sma"]
    keep_a = []
    keep_b = []
    j = 0
    for i, s in enumerate(sma_a):
        while j < len(sma_b) and sma_b[j] < s - atol:
            j += 1
        if j < len(sma_b) and abs(sma_b[j] - s) <= atol:
            keep_a.append(i)
            keep_b.append(j)
            j += 1
    keep_a = np.asarray(keep_a)
    keep_b = np.asarray(keep_b)

    def _slice(prof, idx):
        return {k: v[idx] for k, v in prof.items()}

    return _slice(prof_a, keep_a), _slice(prof_b, keep_b)


# ---------------------------------------------------------------------------
# Aggressive sky mask
# ---------------------------------------------------------------------------
def build_aggressive_sky_mask(
    image,
    nsigma=1.0,
    npixels=4,
    detect_fwhm=2.0,
    box_size=64,
    dilate_iter=4,
):
    """Detect every source above ``nsigma`` and return ``True`` everywhere a
    source (including the target galaxy) lives, so the complement is the
    cleanest possible sky-pixel set.

    Detection is run on the **denoised** image — the lower per-pixel sigma
    lets even the faintest BCG envelope and field sources be captured.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bkg = Background2D(
            image, box_size=box_size, filter_size=3,
            bkg_estimator=MedianBackground(),
        )
    sub = image - bkg.background
    rms = np.asarray(bkg.background_rms)
    sigma = detect_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    kernel = Gaussian2DKernel(sigma)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conv = convolve(sub, np.asarray(kernel), normalize_kernel=True)
        segmap = detect_sources(conv, nsigma * rms, npixels=npixels)
    if segmap is None or segmap.nlabels == 0:
        mask = np.zeros(image.shape, dtype=bool)
    else:
        mask = np.asarray(segmap.data) > 0
    if dilate_iter > 0:
        mask = binary_dilation(mask, iterations=dilate_iter)
    return mask


# ---------------------------------------------------------------------------
# Comparison QA figure
# ---------------------------------------------------------------------------
def arcsinh_stretch(image, scale=0.02, clip_percentile=99.7):
    vmax = np.nanpercentile(image, clip_percentile)
    normed = np.clip(image, 0, vmax) / max(vmax, 1e-10)
    return np.clip(np.arcsinh(normed / scale) / np.arcsinh(1.0 / scale), 0, 1)


def plot_sky_histograms(ax, sky_pixels_noisy, sky_pixels_denoised, sigma_clip=4.0):
    """Log-y histogram + KDE of unmasked sky pixels for both images.

    The X-axis range is set by sigma-clipped stats of the *combined* sample
    so the two distributions share a meaningful window.
    """
    combined = np.concatenate([sky_pixels_noisy, sky_pixels_denoised])
    mean_c, _, std_c = sigma_clipped_stats(combined, sigma=3.0, maxiters=5)
    lo = mean_c - sigma_clip * std_c
    hi = mean_c + sigma_clip * std_c
    bins = np.linspace(lo, hi, 80)

    n_keep = sky_pixels_noisy[(sky_pixels_noisy >= lo) & (sky_pixels_noisy <= hi)]
    d_keep = sky_pixels_denoised[(sky_pixels_denoised >= lo) & (sky_pixels_denoised <= hi)]

    counts_n, _, _ = ax.hist(
        n_keep, bins=bins, histtype="step",
        color="#d62728", lw=1.5, label="noisy",
    )
    counts_d, _, _ = ax.hist(
        d_keep, bins=bins, histtype="step",
        color="#1f77b4", lw=1.5, label="denoised",
    )

    # KDE overlay in counts-per-bin units so it sits on top of the histogram.
    bin_w = bins[1] - bins[0]
    xs = np.linspace(lo, hi, 400)
    if n_keep.size > 5:
        kde_n = gaussian_kde(n_keep)
        ax.plot(xs, kde_n(xs) * n_keep.size * bin_w,
                color="#d62728", lw=1.2, ls="--", alpha=0.8)
    if d_keep.size > 5:
        kde_d = gaussian_kde(d_keep)
        ax.plot(xs, kde_d(xs) * d_keep.size * bin_w,
                color="#1f77b4", lw=1.2, ls="--", alpha=0.8)

    ax.axvline(0.0, color="k", lw=0.7, ls="--", alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    # Explicit y-limits so empty-bin edges don't pull autoscale to log(0) = -inf.
    top = max(counts_n.max(), counts_d.max()) * 2.0
    ax.set_ylim(0.5, top)
    ax.set_xlabel("pixel value [image units]")
    ax.set_ylabel("count (log10 scale)")
    ax.set_title(
        f"Sky-pixel distribution (aggressive-mask complement, +/-{sigma_clip:.0f} sigma)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3, which="both")


def plot_pair_comparison(
    obj_id, desc,
    noisy, denoised,
    isophotes_noisy, isophotes_denoised_free,
    profile_noisy, profile_denoised, profile_denoised_free,
    sky_noisy, sky_denoised,
    sky_mask,
    output_path,
):
    """Two-column comparison figure.

    Left column (axes have identical aspect, no colorbars on individual
    panels — instead an inset colorbar inside the bottom panel — so the three
    image axes line up perfectly):

      * noisy + isophote overlays
      * denoised + same overlays
      * (noisy - denoised) / max(|denoised|, 3 sigma_sky)

    Right column:

      * surface brightness profile, noisy vs denoised (forced) with errorbars
      * μ residual Δμ = μ_noisy − μ_denoised with errorbars
      * sky-pixel histogram + KDE for both images, log-y, sigma-clipped X.
    """
    # ``isoster.plotting.configure_qa_plot_style`` (called by plot_qa_summary)
    # auto-enables LaTeX rendering when a TeX install is detected, which then
    # blows up on plain Unicode in our titles. Force it off for our own plot.
    plt.rcParams["text.usetex"] = False
    fig = plt.figure(figsize=(17.5, 14.0))
    gs = fig.add_gridspec(
        3, 2, width_ratios=[1.0, 1.1],
        hspace=0.28, wspace=0.12,
        left=0.04, right=0.97, top=0.965, bottom=0.05,
    )

    # --- Left column: images with isophote overlays + relative-difference map.
    img_kwargs = dict(origin="lower", cmap="gray")
    gray_n = arcsinh_stretch(noisy)
    gray_d = arcsinh_stretch(denoised)

    ax_n = fig.add_subplot(gs[0, 0])
    ax_n.imshow(gray_n, **img_kwargs)
    draw_isophote_overlays(
        ax_n, isophotes_noisy, step=8, line_width=0.8, alpha=0.7,
        edge_color=None, draw_harmonics=True,
    )
    ax_n.set_title("noisy.fits", fontsize=10)
    ax_n.set_xticks([]); ax_n.set_yticks([])
    ax_n.set_aspect("equal")

    ax_d = fig.add_subplot(gs[1, 0])
    ax_d.imshow(gray_d, **img_kwargs)
    draw_isophote_overlays(
        ax_d, isophotes_denoised_free, step=8, line_width=0.8, alpha=0.7,
        edge_color=None, draw_harmonics=True,
    )
    ax_d.set_title("denoised.fits", fontsize=10)
    ax_d.set_xticks([]); ax_d.set_yticks([])
    ax_d.set_aspect("equal")

    # Relative-difference image. We use ``max(|denoised|, 3*sky_denoised)`` as
    # the denominator so the bright BCG core shows the strict relative change
    # (noisy - denoised) / denoised, while the LSB tail / sky converts
    # naturally into a noise-normalized residual without 1/0 blow-up.
    denom = np.maximum(np.abs(denoised), 3.0 * sky_denoised)
    rel = (noisy - denoised) / denom

    ax_r = fig.add_subplot(gs[2, 0])
    vlim = np.nanpercentile(np.abs(rel), 99.0)
    if not np.isfinite(vlim) or vlim <= 0:
        vlim = 1.0
    vlim = float(np.clip(vlim, 0.1, 1.0))
    im = ax_r.imshow(rel, origin="lower", cmap="RdBu_r", vmin=-vlim, vmax=+vlim)
    ax_r.set_title("relative residual (noisy - denoised)", fontsize=10)
    ax_r.set_xticks([]); ax_r.set_yticks([])
    ax_r.set_aspect("equal")
    # Inset colorbar so the panel keeps the same outer footprint as the two
    # image panels above it (no axes-stealing colorbar that shrinks the plot).
    cax = ax_r.inset_axes([1.02, 0.0, 0.035, 1.0])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("rel. residual", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # --- Right column: asinh SB profile, intensity-residual, sky histogram.
    # The forced and noisy profiles share an SMA grid; the denoised-free
    # profile lives on its own grid. We align (noisy, forced) for the
    # residual panel and overlay denoised-free on the SB panel as a
    # third trace using its own asinh scale.
    pn, pd = align_profiles_by_sma(profile_noisy, profile_denoised)
    sma = pn["sma"]
    sma14 = sma ** 0.25

    scale_n = fit_asinh_scale_to_log(
        pn["intens"], pn["intens_err"], SB_ZEROPOINT, PIXEL_SCALE_ARCSEC,
    )
    scale_d = fit_asinh_scale_to_log(
        pd["intens"], pd["intens_err"], SB_ZEROPOINT, PIXEL_SCALE_ARCSEC,
    )
    scale_df = fit_asinh_scale_to_log(
        profile_denoised_free["intens"], profile_denoised_free["intens_err"],
        SB_ZEROPOINT, PIXEL_SCALE_ARCSEC,
    )
    mu_n = intens_to_mu_asinh(pn["intens"], SB_ZEROPOINT, PIXEL_SCALE_ARCSEC, scale_n)
    mu_d = intens_to_mu_asinh(pd["intens"], SB_ZEROPOINT, PIXEL_SCALE_ARCSEC, scale_d)
    mu_n_err = mu_asinh_error(pn["intens"], pn["intens_err"], PIXEL_SCALE_ARCSEC, scale_n)
    mu_d_err = mu_asinh_error(pd["intens"], pd["intens_err"], PIXEL_SCALE_ARCSEC, scale_d)
    mu0_n = asinh_mu_at_zero_intensity(SB_ZEROPOINT, PIXEL_SCALE_ARCSEC, scale_n)
    mu0_d = asinh_mu_at_zero_intensity(SB_ZEROPOINT, PIXEL_SCALE_ARCSEC, scale_d)
    sma14_df = profile_denoised_free["sma"] ** 0.25
    mu_df = intens_to_mu_asinh(
        profile_denoised_free["intens"], SB_ZEROPOINT, PIXEL_SCALE_ARCSEC, scale_df,
    )
    mu_df_err = mu_asinh_error(
        profile_denoised_free["intens"], profile_denoised_free["intens_err"],
        PIXEL_SCALE_ARCSEC, scale_df,
    )

    ax_sb = fig.add_subplot(gs[0, 1])
    ax_sb.errorbar(
        sma14, mu_n, yerr=mu_n_err, fmt="o", ms=3.5, lw=0.8,
        color="#d62728", label="noisy (free)", capsize=2,
    )
    ax_sb.errorbar(
        sma14, mu_d, yerr=mu_d_err, fmt="s", ms=3.5, lw=0.8,
        color="#1f77b4", label="denoised (forced)", capsize=2,
        markerfacecolor="none",
    )
    ax_sb.errorbar(
        sma14_df, mu_df, yerr=mu_df_err, fmt="^", ms=3.5, lw=0.8,
        color="#2ca02c", label="denoised (free)", capsize=2,
        markerfacecolor="none",
    )
    # Mark the asinh SB level corresponding to I=0. Any data points fainter
    # than these lines correspond to negative-intensity isophotes that the
    # log10 form cannot represent at all.
    ax_sb.axhline(
        mu0_n, color="#d62728", lw=0.8, ls=":", alpha=0.7,
        label=f"noisy I=0 ({mu0_n:.2f})",
    )
    ax_sb.axhline(
        mu0_d, color="#1f77b4", lw=0.8, ls=":", alpha=0.7,
        label=f"denoised I=0 ({mu0_d:.2f})",
    )
    ax_sb.invert_yaxis()
    ax_sb.set_xlabel("SMA$^{1/4}$ [pix$^{1/4}$]")
    ax_sb.set_ylabel(r"$\mu_\mathrm{asinh}$ [mag/arcsec$^2$]")
    ax_sb.set_title(
        f"Surface brightness, asinh form (HSC zp=27.0, 0.168\"; b_noisy={scale_n:.4g}, b_den={scale_d:.4g})",
        fontsize=9,
    )
    ax_sb.legend(loc="lower left", fontsize=8)
    ax_sb.grid(alpha=0.3)

    ax_dmu = fig.add_subplot(gs[1, 1])
    # Intensity-space relative residual (well-defined for any sign of I).
    # Forced rail (geometry-locked): noisy free vs denoised forced — same
    # SMA grid by construction.
    intens_n = pn["intens"]
    intens_d = pd["intens"]
    err_n = pn["intens_err"]
    err_d = pd["intens_err"]
    rel = (intens_n - intens_d) / intens_d
    rel_err = np.sqrt(
        (err_n / intens_d) ** 2
        + (intens_n / intens_d**2) ** 2 * err_d**2
    )
    ax_dmu.errorbar(
        sma14, rel, yerr=rel_err, fmt="o", ms=3.5, lw=0.8,
        color="#1f77b4", capsize=2, label="vs denoised (forced)",
    )
    # Free-rail (independent geometry): align noisy-free with denoised-free
    # by SMA where they coincide.
    pn_a, pdf_a = align_profiles_by_sma(profile_noisy, profile_denoised_free)
    if pn_a["sma"].size > 0:
        rel_f = (pn_a["intens"] - pdf_a["intens"]) / pdf_a["intens"]
        rel_f_err = np.sqrt(
            (pn_a["intens_err"] / pdf_a["intens"]) ** 2
            + (pn_a["intens"] / pdf_a["intens"] ** 2) ** 2 * pdf_a["intens_err"] ** 2
        )
        ax_dmu.errorbar(
            pn_a["sma"] ** 0.25, rel_f, yerr=rel_f_err,
            fmt="^", ms=3.5, lw=0.8,
            color="#2ca02c", capsize=2,
            markerfacecolor="none", label="vs denoised (free)",
        )
    ax_dmu.axhline(0.0, color="k", lw=0.7, ls="--", alpha=0.6)
    ax_dmu.set_xlabel("SMA$^{1/4}$ [pix$^{1/4}$]")
    ax_dmu.set_ylabel(r"$(I_\mathrm{noisy} - I_\mathrm{denoised}) / I_\mathrm{denoised}$")
    ax_dmu.set_title("intensity residual (noisy - denoised) / denoised", fontsize=10)
    # Clamp y-range: the LSB tail can blow up where denoised crosses zero.
    finite = np.isfinite(rel)
    if finite.any():
        cap = float(np.nanpercentile(np.abs(rel[finite]), 95.0))
        cap = max(cap, 0.5)
        ax_dmu.set_ylim(-cap, cap)
    ax_dmu.legend(loc="lower left", fontsize=8)
    ax_dmu.grid(alpha=0.3)

    ax_hist = fig.add_subplot(gs[2, 1])
    sky_pixels_noisy = noisy[~sky_mask]
    sky_pixels_denoised = denoised[~sky_mask]
    plot_sky_histograms(ax_hist, sky_pixels_noisy, sky_pixels_denoised)

    fig.suptitle(
        f"{obj_id} - {desc}: noisy vs denoised (forced)",
        fontsize=13, y=0.99,
    )
    fig.savefig(output_path, dpi=140, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_one(obj_id: str, desc: str):
    print(f"\n=== {obj_id}: {desc} ===")

    src = galaxy_dir(obj_id)
    mask_path = src / MASK_FILENAME
    if not mask_path.exists():
        raise FileNotFoundError(
            f"Mask not found at {mask_path}. Run build_object_mask.py first."
        )

    noisy, denoised = load_pair(obj_id)
    mask = fits.getdata(mask_path).astype(bool)
    anchor_x, anchor_y = load_target_anchor_from_mask(mask_path)

    h, w = noisy.shape
    # Force isophote sampling all the way past the cutout corner so the
    # outer-most rings cover the full diagonal — useful for the LSB tail and
    # for stitching with cross-survey comparisons. The half-diagonal is
    # ~ sqrt(h^2 + w^2) / 2; a small margin keeps the very last ring from
    # going entirely outside the image.
    max_sma = float(np.hypot(h, w) / 2.0)

    # Aggressive all-object mask: detected on denoised.fits where sources
    # show up at lower nsigma. The complement is the sky-pixel set used by
    # the histogram panel of the pair-comparison figure.
    aggressive_sky_mask = build_aggressive_sky_mask(
        denoised, nsigma=1.0, npixels=4, dilate_iter=4,
    )

    sky_noisy = sky_std(noisy)
    sky_denoised = sky_std(denoised)
    var_noisy = uniform_variance_map(noisy)
    var_denoised = uniform_variance_map(denoised)
    print(
        f"  shape=({h}, {w}); anchor=({anchor_x:.2f}, {anchor_y:.2f}); "
        f"object mask coverage {mask.sum() / mask.size * 100:.2f}%; "
        f"aggressive sky-mask coverage {aggressive_sky_mask.sum() / aggressive_sky_mask.size * 100:.2f}%"
    )
    print(f"  maxsma = {max_sma:.1f} px (half-diagonal of {h}x{w})")
    print(
        f"  sky std: noisy={sky_noisy:.4g}, denoised={sky_denoised:.4g}"
        f"  (ratio noisy/denoised = {sky_noisy / sky_denoised:.2f})"
    )

    out_dir = output_dir(obj_id)

    # ----- Stage A: free fit on noisy.fits -----
    config_free = IsosterConfig(
        x0=anchor_x, y0=anchor_y,
        eps=0.2, pa=0.0,
        sma0=10.0, minsma=0.0, astep=0.1, linear_growth=False, maxsma=max_sma,
        fix_center=False, fix_pa=False, fix_eps=False,
        debug=True,
        compute_deviations=True,   # default harmonic_orders=[3, 4]
        full_photometry=True,
        compute_cog=True,
        max_retry_first_isophote=5,
    )
    print("  [A] free fit on noisy.fits ...")
    t0 = time.perf_counter()
    res_noisy = fit_image(noisy, mask=mask, config=config_free, variance_map=var_noisy)
    dt_a = time.perf_counter() - t0
    iso_noisy = res_noisy["isophotes"]
    print(f"      {len(iso_noisy)} isophotes, {dt_a:.2f}s")

    fits_a = out_dir / f"{obj_id}_noisy_isophotes.fits"
    isophote_results_to_fits(res_noisy, str(fits_a))
    model_noisy = build_isoster_model(noisy.shape, iso_noisy, use_harmonics=True)
    qa_a = out_dir / f"{obj_id}_noisy_qa.png"
    plot_qa_summary(
        title=f"{obj_id} - {desc} (noisy free fit)",
        image=noisy,
        isoster_model=model_noisy,
        isoster_res=iso_noisy,
        mask=mask,
        filename=str(qa_a),
        relative_residual=False,
        sb_zeropoint=SB_ZEROPOINT,
        pixel_scale_arcsec=PIXEL_SCALE_ARCSEC,
    )
    print(f"      results -> {fits_a.name},  QA -> {qa_a.name}")

    # ----- Stage B: forced photo on denoised.fits using stage-A geometry -----
    template = make_template(iso_noisy)
    config_forced = IsosterConfig(
        # Anchor / sma schedule are still required by IsosterConfig validation,
        # but the template fully drives the SMA grid + geometry.
        x0=anchor_x, y0=anchor_y,
        sma0=10.0, minsma=0.0, astep=0.1, linear_growth=False, maxsma=max_sma,
        debug=True,
        compute_deviations=True,
        full_photometry=True,
        # compute_cog is regular-mode only; forced photometry will ignore it.
        max_retry_first_isophote=0,
    )
    print(f"  [B] forced photometry on denoised.fits ({len(template)} template rings) ...")
    t0 = time.perf_counter()
    res_denoised = fit_image(
        denoised, mask=mask, config=config_forced,
        template=template, variance_map=var_denoised,
    )
    dt_b = time.perf_counter() - t0
    iso_denoised = res_denoised["isophotes"]
    print(f"      {len(iso_denoised)} isophotes, {dt_b:.2f}s")

    fits_b = out_dir / f"{obj_id}_denoised_forced_isophotes.fits"
    isophote_results_to_fits(res_denoised, str(fits_b))
    print(f"      results -> {fits_b.name}")

    # ----- Stage A2: free fit on denoised.fits (independent geometry) -----
    print("  [A2] free fit on denoised.fits ...")
    t0 = time.perf_counter()
    res_denoised_free = fit_image(
        denoised, mask=mask, config=config_free, variance_map=var_denoised,
    )
    dt_a2 = time.perf_counter() - t0
    iso_denoised_free = res_denoised_free["isophotes"]
    print(f"      {len(iso_denoised_free)} isophotes, {dt_a2:.2f}s")

    fits_a2 = out_dir / f"{obj_id}_denoised_free_isophotes.fits"
    isophote_results_to_fits(res_denoised_free, str(fits_a2))
    model_denoised_free = build_isoster_model(
        denoised.shape, iso_denoised_free, use_harmonics=True,
    )
    qa_a2 = out_dir / f"{obj_id}_denoised_free_qa.png"
    plot_qa_summary(
        title=f"{obj_id} - {desc} (denoised free fit)",
        image=denoised,
        isoster_model=model_denoised_free,
        isoster_res=iso_denoised_free,
        mask=mask,
        filename=str(qa_a2),
        relative_residual=False,
        sb_zeropoint=SB_ZEROPOINT,
        pixel_scale_arcsec=PIXEL_SCALE_ARCSEC,
    )
    print(f"      results -> {fits_a2.name},  QA -> {qa_a2.name}")

    # ----- Stage C: comparison QA figure -----
    profile_noisy = collect_profile(iso_noisy)
    profile_denoised = collect_profile(iso_denoised)
    profile_denoised_free = collect_profile(iso_denoised_free)
    qa_pair = out_dir / f"{obj_id}_pair_comparison.png"
    plot_pair_comparison(
        obj_id, desc,
        noisy, denoised,
        iso_noisy, iso_denoised_free,
        profile_noisy, profile_denoised, profile_denoised_free,
        sky_noisy, sky_denoised,
        aggressive_sky_mask,
        qa_pair,
    )
    print(f"  [C] pair comparison -> {qa_pair.name}")

    # Save the aggressive sky mask for downstream re-use / inspection.
    sky_mask_path = out_dir / f"{obj_id}_aggressive_sky_mask.fits"
    fits.PrimaryHDU(aggressive_sky_mask.astype(np.uint8)).writeto(
        sky_mask_path, overwrite=True
    )
    return {
        "n_iso_noisy": len(iso_noisy),
        "n_iso_denoised": len(iso_denoised),
        "wall_a": dt_a,
        "wall_b": dt_b,
        "sky_noisy": sky_noisy,
        "sky_denoised": sky_denoised,
    }


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--only",
        action="append",
        help="Process only this short galaxy id (repeatable).",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    valid = {g[0]: g for g in GALAXIES}
    ids = args.only if args.only else list(valid.keys())
    for sid in ids:
        if sid not in valid:
            print(f"ERROR: unknown id {sid}")
            continue
        _, _, desc = valid[sid]
        run_one(sid, desc)


if __name__ == "__main__":
    main()
