"""
Multi-band ellipse sampler for the joint isoster fitter.

Wraps :func:`scipy.ndimage.map_coordinates` so a single ellipse path is
computed once per geometry and then sampled across all bands in one
vectorized call. The shared-validity rule is enforced: a sample index is
kept only if every band passes mask, NaN, and (when WLS) variance
checks. This guarantees that the joint design matrix has the same
sample count `N` per band, which the joint solver requires.

See ``docs/agent/plan-2026-04-29-multiband-feasibility.md`` decisions
D6/D7/D9 for the validity rule and D19 for the vectorization choice.
"""

from collections import namedtuple
from typing import List, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates

from ..numba_kernels import compute_ellipse_coords


# Multi-band counterpart of ``isoster.sampling.IsophoteData``. Per-band
# arrays are stored as a 2D ``(B, N_valid)`` ndarray (intens, variances)
# while the shared per-isophote angle and radius arrays are 1D shape
# ``(N_valid,)``. ``valid_count`` records how many of the original
# ``N_samples`` survived shared-validity filtering.
MultiIsophoteData = namedtuple(
    "MultiIsophoteData",
    [
        "angles",        # shape (N_valid,) - psi (EA mode) or phi (regular mode)
        "phi",           # shape (N_valid,) - position angle, always available
        "intens",        # shape (B, N_valid) - per-band intensities
        "radii",         # shape (N_valid,) - constant = sma
        "variances",     # shape (B, N_valid) or None when no variance maps
        "n_samples",     # int - total samples on the ellipse path before filtering
        "valid_count",   # int - samples that survived shared-validity (== N_valid)
    ],
)


def _stack_images(images: Sequence[NDArray[np.floating]]) -> NDArray[np.float64]:
    """Validate and stack a list of B same-shape images into ``(B, H, W)`` float64."""
    if len(images) == 0:
        raise ValueError("images list cannot be empty")
    h, w = images[0].shape
    for i, im in enumerate(images):
        if im.shape != (h, w):
            raise ValueError(
                f"images[{i}] has shape {im.shape}, expected {(h, w)} "
                "(must match images[0])"
            )
    # Cast to float64 so map_coordinates does not silently truncate.
    return np.stack([np.ascontiguousarray(im, dtype=np.float64) for im in images], axis=0)


def _resolve_masks(
    masks: Union[None, NDArray[np.bool_], Sequence[Optional[NDArray[np.bool_]]]],
    n_bands: int,
    h: int,
    w: int,
) -> List[Optional[NDArray[np.float64]]]:
    """
    Normalize masks input into a list of length B.

    Accepts: ``None`` (no masks), a single ``(H, W)`` boolean ndarray
    (broadcast to all bands), or a list of length ``B`` of ndarrays or
    Nones (per-band, where None means "no bad pixels in that band").

    Each returned mask is converted to ``float64`` (1.0 = bad, 0.0 = good)
    so it can be sampled by map_coordinates with ``order=0`` and the
    cval=1.0 (off-image = bad) convention. ``None`` entries remain
    ``None`` so the sampler can skip the call.
    """
    if masks is None:
        return [None] * n_bands

    if isinstance(masks, np.ndarray):
        if masks.shape != (h, w):
            raise ValueError(
                f"masks ndarray shape {masks.shape} does not match images shape {(h, w)}"
            )
        m_f: NDArray[np.float64] = (
            masks.astype(np.float64) if masks.dtype.kind != "f" else masks.astype(np.float64, copy=False)
        )
        return [m_f] * n_bands

    if not hasattr(masks, "__len__") or len(masks) != n_bands:
        raise ValueError(
            f"masks list length must equal n_bands ({n_bands}); got "
            f"{len(masks) if hasattr(masks, '__len__') else type(masks).__name__}"
        )

    out: List[Optional[NDArray[np.float64]]] = []
    for i, m in enumerate(masks):
        if m is None:
            out.append(None)
            continue
        if not isinstance(m, np.ndarray):
            raise TypeError(
                f"masks[{i}] must be a numpy ndarray or None, got {type(m).__name__}"
            )
        if m.shape != (h, w):
            raise ValueError(
                f"masks[{i}] shape {m.shape} does not match images shape {(h, w)}"
            )
        m_arr: NDArray[np.float64] = (
            m.astype(np.float64, copy=False) if m.dtype != np.float64 else m  # type: ignore[assignment]
        )
        out.append(m_arr)
    return out


def _resolve_variance_maps(
    variance_maps: Union[None, NDArray[np.floating], Sequence[NDArray[np.floating]]],
    n_bands: int,
    h: int,
    w: int,
) -> Optional[List[NDArray[np.float64]]]:
    """
    Normalize variance_maps input into a list of length B (or None).

    All-or-nothing per decision D7: either every band has a map (full
    WLS) or none do (full OLS). ``None`` values inside a list are
    rejected even if the list has the right length.

    Returns ``None`` when the user passed ``None`` (OLS mode signal).
    Otherwise returns a list of B ``(H, W)`` float64 arrays.
    """
    if variance_maps is None:
        return None

    if isinstance(variance_maps, np.ndarray):
        if variance_maps.shape != (h, w):
            raise ValueError(
                f"variance_maps ndarray shape {variance_maps.shape} does not match "
                f"images shape {(h, w)}"
            )
        v_f = variance_maps.astype(np.float64)
        return [v_f] * n_bands

    if len(variance_maps) != n_bands:
        raise ValueError(
            f"variance_maps list length must equal n_bands ({n_bands}); got "
            f"{len(variance_maps)}"
        )

    out: List[NDArray[np.float64]] = []
    for i, v in enumerate(variance_maps):
        if v is None:
            raise ValueError(
                f"variance_maps[{i}] is None: variance maps are all-or-nothing. "
                "Pass variance_maps=None for full OLS, or provide a map for every "
                "band."
            )
        if not isinstance(v, np.ndarray):
            raise TypeError(
                f"variance_maps[{i}] must be a numpy ndarray, got {type(v).__name__}"
            )
        if v.shape != (h, w):
            raise ValueError(
                f"variance_maps[{i}] shape {v.shape} does not match images shape {(h, w)}"
            )
        out.append(v.astype(np.float64))
    return out


def _sample_image_stack(
    image_stack: NDArray[np.float64],
    coords: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Sample a (B, H, W) stack along the (y, x) ellipse path in one call.

    ``map_coordinates`` accepts N-D inputs; the 0th axis (band) is
    treated as a non-interpolated dimension by passing a 0-stride
    coordinate of ``np.arange(B)``. Returns a (B, N) array.

    Implementation: loop is acceptable here since N <= a few hundred per
    isophote and B is small (typically 2-5). Profiling driver-end will
    decide whether the per-band Python iteration is the bottleneck.
    """
    n_bands, _, _ = image_stack.shape
    n_samples = coords.shape[1]
    out = np.empty((n_bands, n_samples), dtype=np.float64)
    for b in range(n_bands):
        out[b] = map_coordinates(image_stack[b], coords, order=1, mode="constant", cval=np.nan)
    return out


def extract_isophote_data_multi(
    images: Sequence[NDArray[np.floating]],
    masks: Union[None, NDArray[np.bool_], Sequence[Optional[NDArray[np.bool_]]]],
    x0: float,
    y0: float,
    sma: float,
    eps: float,
    pa: float,
    use_eccentric_anomaly: bool = False,
    variance_maps: Union[None, NDArray[np.floating], Sequence[NDArray[np.floating]]] = None,
) -> MultiIsophoteData:
    """
    Extract per-band intensities along a shared elliptical path.

    Parameters
    ----------
    images : sequence of ndarray
        ``B`` aligned same-shape ``(H, W)`` images. The 0th sequence
        index is the band index used by the joint design matrix.
    masks : None | ndarray | sequence of (ndarray|None)
        Bad-pixel mask. ``None`` = no masking on any band. A single
        ``(H, W)`` boolean ndarray broadcasts to all bands. A sequence
        of length ``B`` allows per-band masks; ``None`` per band means
        "no bad pixels in that band".
    x0, y0 : float
        Ellipse center coordinates.
    sma : float
        Semi-major axis length.
    eps : float
        Ellipticity (1 - b/a).
    pa : float
        Position angle in radians.
    use_eccentric_anomaly : bool, default False
        Sample uniformly in eccentric anomaly (Ciambur 2015) when True.
    variance_maps : None | ndarray | sequence of ndarray
        Per-pixel variance. All-or-nothing per band.

    Returns
    -------
    MultiIsophoteData
        Named tuple with shared ``angles, phi, radii`` and per-band
        ``intens, variances`` (each shape ``(B, N_valid)``). ``variances``
        is ``None`` when ``variance_maps`` is ``None``.

    Notes
    -----
    Shared-validity rule (decision D9): a sample is kept iff *every*
    band's mask, NaN, and (in WLS mode) finite-variance checks pass at
    that location. Off-image samples (``map_coordinates`` returning
    ``cval=NaN``) are dropped from all bands.
    """
    image_stack = _stack_images(images)  # (B, H, W) float64 by construction
    n_bands, h, w = image_stack.shape

    masks_list = _resolve_masks(masks, n_bands, h, w)
    var_list = _resolve_variance_maps(variance_maps, n_bands, h, w)

    n_samples = max(64, int(2.0 * np.pi * sma))

    x_coords, y_coords, psi, phi = compute_ellipse_coords(
        n_samples, sma, eps, pa, x0, y0, use_eccentric_anomaly
    )
    coords = np.vstack([y_coords, x_coords]).astype(np.float64, copy=False)

    intens_full = _sample_image_stack(image_stack, coords)  # (B, N)

    # Sample per-band masks. Order=0 nearest-neighbor with cval=1.0
    # (off-image = bad), to match the single-band convention.
    valid = np.ones(n_samples, dtype=bool)
    for b in range(n_bands):
        valid &= ~np.isnan(intens_full[b])
        m_b = masks_list[b]
        if m_b is not None:
            mask_vals = map_coordinates(m_b, coords, order=0, mode="constant", cval=1.0)
            valid &= mask_vals < 0.5

    var_full: Optional[NDArray[np.float64]] = None
    if var_list is not None:
        var_full = np.empty((n_bands, n_samples), dtype=np.float64)
        for b in range(n_bands):
            v = map_coordinates(var_list[b], coords, order=1, mode="constant", cval=np.nan)
            var_full[b] = v
            valid &= np.isfinite(v) & (v > 0.0)

    n_valid = int(np.sum(valid))
    intens_kept = intens_full[:, valid]
    var_kept: Optional[NDArray[np.float64]]
    var_kept = var_full[:, valid] if var_full is not None else None

    if use_eccentric_anomaly:
        angles_kept = psi[valid]
        phi_kept = phi[valid]
    else:
        angles_kept = phi[valid]
        phi_kept = phi[valid]

    radii_kept = np.full(n_valid, sma, dtype=np.float64)

    return MultiIsophoteData(
        angles=angles_kept,
        phi=phi_kept,
        intens=intens_kept,
        radii=radii_kept,
        variances=var_kept,
        n_samples=n_samples,
        valid_count=n_valid,
    )
