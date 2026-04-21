"""Per-galaxy image-sigma estimator.

The composite score normalizes residual-zone RMS by a single image
noise scale so every arm on the same galaxy is judged against the
same denominator. This module computes that scale once, from the
galaxy bundle (image + optional variance + optional mask), before
any arm runs.

Precedence ladder (first applicable wins):

    1. ``sigma_bg``        — user-supplied value via IsosterConfig /
                              adapter metadata. Highest trust.
    2. ``variance`` map    — ``median(sqrt(variance))`` over the
                              far-outer annulus (``r_ell > 2 * R_ref``).
    3. Empirical MAD       — ``1.4826 * MAD(image_pixels)`` over the
                              same annulus, after three rounds of
                              iterative 3σ clipping. Fallback default.

The annulus is anchored on the adapter's initial geometry so the
result is arm-independent.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .residual_zones import compute_elliptical_radius_grid

MIN_SIGMA_VALUE = 1e-6


def compute_image_sigma(
    image: np.ndarray,
    *,
    x0: float,
    y0: float,
    eps: float,
    pa: float,
    R_ref: float | None,
    maxsma: float,
    mask: np.ndarray | None = None,
    variance: np.ndarray | None = None,
    sigma_bg: float | None = None,
    min_pixels: int = 200,
    annulus_factor: float = 2.0,
) -> dict[str, Any]:
    """Return ``{image_sigma_adu, sigma_method, n_pixels_used}``.

    When no method yields a positive finite value, the fallback is
    ``image_sigma_adu = max(MIN_SIGMA_VALUE, 0.01 * |median(image)|)``
    so the score denominator is always safe. ``sigma_method`` in that
    case is ``"fallback"``.
    """
    image = np.asarray(image, dtype=np.float64)

    # Precedence 1: explicit sigma_bg
    if sigma_bg is not None and np.isfinite(sigma_bg) and sigma_bg > 0:
        return {
            "image_sigma_adu": float(sigma_bg),
            "sigma_method": "sigma_bg",
            "n_pixels_used": 0,
        }

    # Annulus: r_ell > annulus_factor * R_ref. Fall back to half the
    # frame when the galaxy has no Re.
    r_ref = (
        float(R_ref)
        if R_ref is not None and np.isfinite(R_ref) and R_ref > 0
        else 0.25 * float(maxsma)
    )
    r_ref = max(r_ref, 1.0)
    r_ell = compute_elliptical_radius_grid(image.shape, x0, y0, eps, pa)
    annulus = r_ell > annulus_factor * r_ref
    valid = annulus & np.isfinite(image)
    if mask is not None:
        valid &= ~np.asarray(mask, dtype=bool)

    # Precedence 2: variance map
    if variance is not None:
        var = np.asarray(variance, dtype=np.float64)
        var_values = var[valid & np.isfinite(var) & (var > 0)]
        if var_values.size >= min_pixels:
            sigma = float(np.median(np.sqrt(var_values)))
            if np.isfinite(sigma) and sigma > 0:
                return {
                    "image_sigma_adu": sigma,
                    "sigma_method": "variance_map",
                    "n_pixels_used": int(var_values.size),
                }

    # Precedence 3: empirical MAD on the far-outer annulus
    annulus_pixels = image[valid]
    if annulus_pixels.size < min_pixels:
        # Not enough pixels in the annulus — widen to the full valid image.
        fallback_mask = np.isfinite(image)
        if mask is not None:
            fallback_mask &= ~np.asarray(mask, dtype=bool)
        annulus_pixels = image[fallback_mask]

    if annulus_pixels.size >= 10:
        clipped = _iterative_sigma_clip(annulus_pixels, n_sigma=3.0, n_iter=3)
        median = float(np.median(clipped))
        mad = float(np.median(np.abs(clipped - median)))
        sigma = 1.4826 * mad
        if np.isfinite(sigma) and sigma > 0:
            return {
                "image_sigma_adu": sigma,
                "sigma_method": "mad",
                "n_pixels_used": int(clipped.size),
            }

    # Last-resort fallback to avoid division-by-zero downstream.
    global_median = float(np.median(image[np.isfinite(image)])) if image.size else 0.0
    sigma = max(MIN_SIGMA_VALUE, 0.01 * abs(global_median))
    return {
        "image_sigma_adu": sigma,
        "sigma_method": "fallback",
        "n_pixels_used": 0,
    }


def _iterative_sigma_clip(
    values: np.ndarray, n_sigma: float = 3.0, n_iter: int = 3
) -> np.ndarray:
    """Return the subset surviving ``n_iter`` rounds of ``n_sigma`` MAD clipping."""
    current = values.astype(np.float64, copy=True)
    for _ in range(n_iter):
        if current.size < 10:
            break
        med = np.median(current)
        mad = np.median(np.abs(current - med))
        sigma = 1.4826 * mad
        if sigma <= 0 or not np.isfinite(sigma):
            break
        keep = np.abs(current - med) < n_sigma * sigma
        if keep.all():
            break
        current = current[keep]
    return current
