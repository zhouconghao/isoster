"""
Numba-accelerated kernels for multi-band joint design matrix construction.

This module provides JIT-compiled builders for the
``(N × (5 + B))``-shaped joint design matrix used by the multi-band
fitter. The matrix has the structure

    row[i, b]  =  [δ_{0b}, δ_{1b}, …, δ_{(B-1)b},
                   sin(φᵢ), cos(φᵢ), sin(2φᵢ), cos(2φᵢ)]

where the per-band indicator block (`I_b`) parameterizes per-band
nuisance background terms `I0_b`, and the shared trailing block
parameterizes the geometric harmonic coefficients
`(A1, B1, A2, B2)`.

Each column block is repeated `B` times (once per band) along the row
axis. The numba kernel writes the indicator block efficiently as a
band-diagonal pattern; the harmonic block is identical to the
single-band design matrix per row, repeated across bands.

Includes a NumPy fallback when numba is unavailable. Same pattern as
``isoster.numba_kernels``: a ``@njit`` decorator that becomes a no-op
when ``NUMBA_AVAILABLE = False`` or ``NUMBA_DISABLE_JIT=1``.
"""

import os

import numpy as np
from numpy.typing import NDArray

# Mirror the single-band kernel's numba availability detection.
try:
    from numba import njit  # type: ignore[import-not-found]

    NUMBA_AVAILABLE = os.environ.get("NUMBA_DISABLE_JIT", "0") != "1"
except ImportError:
    NUMBA_AVAILABLE = False

if not NUMBA_AVAILABLE:

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


@njit(cache=True)
def _build_joint_design_matrix_numba(phi, n_bands):
    """
    Build the (B*N × (5 + B)) joint design matrix (numba-accelerated).

    For each kept sample index `i` and band index `b`:
        row = b * N + i
        col 0..B-1 : indicator (1.0 if col == b, else 0.0)
        col B+0    : sin(φᵢ)
        col B+1    : cos(φᵢ)
        col B+2    : sin(2 φᵢ)
        col B+3    : cos(2 φᵢ)

    The right-hand side is constructed by the caller as the
    band-stacked intensity vector ``ravel`` over the same row order.

    Parameters
    ----------
    phi : ndarray
        Shape (N,). Angle in radians. The same shared ellipse path is
        used for every band, so phi has only one axis.
    n_bands : int
        Number of bands `B`.

    Returns
    -------
    ndarray
        Shape (B * N, 5 + B). The constant-1 column from the single-band
        design matrix is replaced by `B` per-band indicator columns.
        The shared geometric block ``[sin φ, cos φ, sin 2φ, cos 2φ]``
        sits in the trailing four columns (`5 + B - 4 = B + 1`-…-`B + 4`).

        Note: the column order is `[I_0, I_1, ..., I_{B-1}, A1, B1, A2, B2]`,
        i.e. `B` indicator columns followed by 4 geometric columns,
        for a total of `5 + B` columns. (The 5 here counts as
        `B + 4` from the per-band-`I0` plus 4 geometric coefficients.)
    """
    n_samples = phi.shape[0]
    n_rows = n_bands * n_samples
    n_cols = n_bands + 4
    A = np.zeros((n_rows, n_cols), dtype=np.float64)

    for b in range(n_bands):
        for i in range(n_samples):
            row = b * n_samples + i
            A[row, b] = 1.0
            p = phi[i]
            A[row, n_bands + 0] = np.sin(p)
            A[row, n_bands + 1] = np.cos(p)
            A[row, n_bands + 2] = np.sin(2.0 * p)
            A[row, n_bands + 3] = np.cos(2.0 * p)

    return A


def _build_joint_design_matrix_numpy(phi, n_bands):
    """
    Build the (B*N × (B + 4)) joint design matrix (pure numpy).

    Vectorized: builds the per-band indicator block via ``np.eye``-style
    repeats and the geometric block via broadcasting.
    """
    n_samples = phi.shape[0]
    n_cols = n_bands + 4
    A = np.zeros((n_bands * n_samples, n_cols), dtype=np.float64)

    # Indicator block: a band-diagonal pattern of 1.0s. Row b*N + i
    # has a 1 at column b. Build by tiling np.eye(n_bands) up to n_samples.
    band_idx = np.repeat(np.arange(n_bands), n_samples)  # shape (B*N,)
    row_idx = np.arange(n_bands * n_samples)
    A[row_idx, band_idx] = 1.0

    # Geometric block: the same 4-column tile per band, stacked B times.
    sin1 = np.sin(phi)
    cos1 = np.cos(phi)
    sin2 = np.sin(2.0 * phi)
    cos2 = np.cos(2.0 * phi)
    geom = np.column_stack([sin1, cos1, sin2, cos2])  # (N, 4)
    A[:, n_bands:] = np.tile(geom, (n_bands, 1))

    return A


_build_joint_design_matrix_impl = (
    _build_joint_design_matrix_numba if NUMBA_AVAILABLE else _build_joint_design_matrix_numpy
)


def build_joint_design_matrix(
    phi: NDArray[np.floating], n_bands: int
) -> NDArray[np.floating]:
    """
    Build the joint design matrix for multi-band harmonic fitting.

    Parameters
    ----------
    phi : ndarray
        Shape (N,). Angles in radians, shared across all bands.
    n_bands : int
        Number of bands `B`. Must be >= 1.

    Returns
    -------
    ndarray
        Shape (B * N, B + 4). Column order:
        ``[I_0, I_1, ..., I_{B-1}, A1, B1, A2, B2]`` where the leading
        `B` columns are per-band indicators (one row of 1.0 per band) and
        the trailing 4 columns are the shared geometric harmonic terms.

        The B=1 case reduces to a 5-column matrix
        ``[1, sin φ, cos φ, sin 2φ, cos 2φ]`` numerically identical to
        the single-band design matrix.

    Raises
    ------
    ValueError
        If ``phi`` is empty or ``n_bands < 1``.
    """
    if phi.size == 0:
        raise ValueError("phi array cannot be empty")
    if n_bands < 1:
        raise ValueError(f"n_bands must be >= 1, got {n_bands}")
    return _build_joint_design_matrix_impl(phi, n_bands)


def warmup_numba_mb() -> None:
    """
    Trigger numba JIT compilation for the multi-band kernels.

    Mirrors :func:`isoster.numba_kernels.warmup_numba` so benchmark
    callers can warm up both the single-band and multi-band paths.
    """
    if not NUMBA_AVAILABLE:
        return
    phi = np.linspace(0.0, 2.0 * np.pi, 64)
    _ = _build_joint_design_matrix_numba(phi, 2)
