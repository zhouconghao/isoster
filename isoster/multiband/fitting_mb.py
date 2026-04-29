"""
Multi-band joint per-isophote fitter.

Implements the joint design-matrix solve and the per-isophote iteration
loop that produces a single shared geometry per SMA along with per-band
intensities and per-band harmonic deviations. See
``docs/agent/plan-2026-04-29-multiband-feasibility.md`` decisions
D2 (joint design matrix), D9 (sigma clipping), D10 (combined gradient),
D11 (per-band I0_b), and D12 (band weights).

Out of Stage-1 scope: ISOFIT (`simultaneous_harmonics`), LSB auto-lock,
outer-center regularization. The iteration loop here is a leaner port
of :func:`isoster.fitting.fit_isophote` with those features removed.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ..fitting import (
    _prepare_mask_float,
    compute_aperture_photometry,
    compute_deviations,
    sigma_clip,
)
from ..numba_kernels import build_harmonic_matrix
from .config_mb import IsosterConfigMB
from .numba_kernels_mb import build_joint_design_matrix
from .sampling_mb import (
    MultiIsophoteData,
    extract_isophote_data_multi,
)


# ---------------------------------------------------------------------------
# Joint solve
# ---------------------------------------------------------------------------


def fit_first_and_second_harmonics_joint(
    angles: NDArray[np.float64],
    intens_per_band: NDArray[np.float64],
    band_weights_arr: NDArray[np.float64],
    variances_per_band: Optional[NDArray[np.float64]] = None,
) -> Tuple[NDArray[np.float64], Optional[NDArray[np.float64]], bool]:
    """
    Solve the joint multi-band 1st+2nd harmonic system in WLS or OLS mode.

    Parameters
    ----------
    angles : (N,) float64
        Shared angle array along the ellipse (psi in EA mode, phi in
        regular mode). Same for every band by construction.
    intens_per_band : (B, N) float64
        Per-band intensity samples. Order matches the band order in the
        IsosterConfigMB.bands list.
    band_weights_arr : (B,) float64
        Per-band scalar weights ``w_b`` (already resolved). Must be > 0.
    variances_per_band : (B, N) float64 or None
        Per-pixel variances. ``None`` triggers OLS mode; otherwise WLS.

    Returns
    -------
    coeffs : (B + 4,) float64
        Coefficient vector ``[I0_0, I0_1, ..., I0_{B-1}, A1, B1, A2, B2]``.
    cov : (B + 4, B + 4) float64 or None
        Covariance matrix from the joint solve. WLS: ``(A^T W A)^-1`` is
        the exact covariance. OLS: ``(A^T A)^-1`` (caller must scale by
        residual variance for true covariance). ``None`` on solver failure.
    wls_mode : bool
        True when ``variances_per_band`` was provided.

    Notes
    -----
    Decision D12: ``band_weights`` enter as ``sqrt(w_b)`` row scaling on
    each band's row block. In WLS this composes with the per-pixel
    inverse-variance weight as ``w_b / variance_b(pixel)`` in the
    effective row weight.
    """
    n_bands, n_samples = intens_per_band.shape
    A = build_joint_design_matrix(angles, n_bands)  # (B*N, B+4)

    # Stack the per-band intensities band-by-band into a single RHS.
    y = intens_per_band.reshape(n_bands * n_samples)

    # Per-row effective weights w_eff: in WLS, w_eff = w_b / variance_b(pixel).
    # In OLS, w_eff = w_b. Either way w_eff is a length (B*N) vector with
    # the band's scalar weight applied to every sample of that band.
    w_band_per_row = np.repeat(band_weights_arr, n_samples)  # (B*N,)
    if variances_per_band is not None:
        var_flat = variances_per_band.reshape(n_bands * n_samples)
        w_eff = w_band_per_row / var_flat
        wls_mode = True
    else:
        w_eff = w_band_per_row
        wls_mode = False

    # Exact WLS / scaled OLS solve via the normal equations
    # A^T W A x = A^T W y, with W = diag(w_eff).
    AW = A * w_eff[:, None]
    ATWA = AW.T @ A
    ATWy = AW.T @ y
    try:
        coeffs = np.linalg.solve(ATWA, ATWy)
        cov = np.linalg.inv(ATWA)
        return coeffs, cov, wls_mode
    except np.linalg.LinAlgError:
        # Fallback: per-band means as I0_b, zeros for harmonic coefficients.
        fallback = np.zeros(n_bands + 4, dtype=np.float64)
        for b in range(n_bands):
            fallback[b] = float(np.mean(intens_per_band[b]))
        return fallback, None, wls_mode


def fit_first_and_second_harmonics_ref(
    angles: NDArray[np.float64],
    intens_ref: NDArray[np.float64],
    variances_ref: Optional[NDArray[np.float64]] = None,
) -> Tuple[NDArray[np.float64], Optional[NDArray[np.float64]], bool]:
    """
    Reference-band-only fallback for ``harmonic_combination='ref'``.

    Wraps :func:`isoster.numba_kernels.build_harmonic_matrix` and the
    standard 5-param solve so the calling iteration loop sees the same
    coefficient layout as the joint solver: a ``(B + 4,)`` vector with
    placeholder ``I0_b = mean(intens_b)`` for every non-reference band.

    Returns
    -------
    coeffs : (5,)
        ``[I0_ref, A1, B1, A2, B2]``. Caller widens to ``(B + 4,)`` by
        filling per-band means.
    cov : (5, 5) or None
    wls_mode : bool
    """
    A = build_harmonic_matrix(angles)
    if variances_ref is not None:
        weights = 1.0 / variances_ref
        AW = A * weights[:, None]
        ATWA = AW.T @ A
        ATWy = AW.T @ intens_ref
        try:
            coeffs = np.linalg.solve(ATWA, ATWy)
            cov = np.linalg.inv(ATWA)
            return coeffs, cov, True
        except np.linalg.LinAlgError:
            return np.array([np.mean(intens_ref), 0.0, 0.0, 0.0, 0.0]), None, True
    try:
        coeffs, _residuals, _rank, _sv = np.linalg.lstsq(A, intens_ref, rcond=None)
        cov = np.linalg.inv(A.T @ A)
        return coeffs, cov, False
    except np.linalg.LinAlgError:
        return np.array([np.mean(intens_ref), 0.0, 0.0, 0.0, 0.0]), None, False


def evaluate_joint_model(
    angles: NDArray[np.float64],
    coeffs: NDArray[np.float64],
    n_bands: int,
) -> NDArray[np.float64]:
    """Evaluate the joint model intensities for every band at every angle.

    Returns shape ``(B, N)``: each row b is
    ``I0_b + A1·sin(φ) + B1·cos(φ) + A2·sin(2φ) + B2·cos(2φ)``.
    """
    A1, B1, A2, B2 = coeffs[n_bands], coeffs[n_bands + 1], coeffs[n_bands + 2], coeffs[n_bands + 3]
    geom = (
        A1 * np.sin(angles)
        + B1 * np.cos(angles)
        + A2 * np.sin(2.0 * angles)
        + B2 * np.cos(2.0 * angles)
    )
    out = np.empty((n_bands, len(angles)), dtype=np.float64)
    for b in range(n_bands):
        out[b] = coeffs[b] + geom
    return out


# ---------------------------------------------------------------------------
# Per-band sigma clipping (decision D9)
# ---------------------------------------------------------------------------


def _per_band_sigma_clip(
    angles: NDArray[np.float64],
    phi: NDArray[np.float64],
    intens_per_band: NDArray[np.float64],
    variances_per_band: Optional[NDArray[np.float64]],
    sclip: float,
    nclip: int,
    sclip_low: Optional[float],
    sclip_high: Optional[float],
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    Optional[NDArray[np.float64]],
    int,
]:
    """
    Per-band sigma clipping with shared-validity AND across bands.

    Each band is clipped independently against its own intensity
    statistics; the resulting per-band survivor masks are AND-ed and
    applied to angles, phi, and every band's intens/variance arrays.
    Reduces to the existing single-band clip when B=1.
    """
    n_bands, n_samples = intens_per_band.shape
    if nclip <= 0 or n_samples == 0:
        return angles, phi, intens_per_band, variances_per_band, 0

    keep = np.ones(n_samples, dtype=bool)
    for b in range(n_bands):
        # Use the existing single-band sigma_clip on (angles, intens_b)
        # to get an index mask back. sigma_clip returns clipped arrays,
        # not a mask, so we re-derive by tracking which samples survive
        # via index alignment.
        idx = np.arange(n_samples)
        clipped = sigma_clip(
            idx.astype(np.float64),
            intens_per_band[b].copy(),
            sclip=sclip,
            nclip=nclip,
            sclip_low=sclip_low,
            sclip_high=sclip_high,
        )
        idx_keep = clipped[0].astype(np.int64)
        survivor = np.zeros(n_samples, dtype=bool)
        survivor[idx_keep] = True
        keep &= survivor

    if keep.all():
        return angles, phi, intens_per_band, variances_per_band, 0

    n_clipped = int(n_samples - keep.sum())
    intens_clipped = intens_per_band[:, keep]
    variances_clipped: Optional[NDArray[np.float64]]
    variances_clipped = variances_per_band[:, keep] if variances_per_band is not None else None
    return angles[keep], phi[keep], intens_clipped, variances_clipped, n_clipped


# ---------------------------------------------------------------------------
# Combined gradient (decision D10)
# ---------------------------------------------------------------------------


def compute_joint_gradient(
    images: Sequence[NDArray[np.floating]],
    masks: Union[None, NDArray[np.bool_], Sequence[Optional[NDArray[np.bool_]]]],
    geometry: Dict[str, float],
    config: IsosterConfigMB,
    band_weights_arr: NDArray[np.float64],
    previous_gradient: Optional[float] = None,
    variance_maps: Union[None, NDArray[np.floating], Sequence[NDArray[np.floating]]] = None,
    current_data: Optional[MultiIsophoteData] = None,
) -> Tuple[float, Optional[float], List[float], List[Optional[float]]]:
    """
    Compute one combined-scalar radial gradient from multiple bands.

    Returns
    -------
    gradient_joint : float
        Weighted-mean gradient ``Σ w_b grad_b / Σ w_b``.
    gradient_error_joint : float or None
        ``sqrt(Σ w_b^2 σ_b^2 / (Σ w_b)^2)``, or None when no per-band
        errors are available.
    per_band_gradients : list[float]
        Length B. Each entry is ``grad_b`` (raw per-band gradient).
    per_band_gradient_errors : list[float|None]
        Length B. Each entry is ``σ_b`` (per-band gradient error) or None.
    """
    x0 = geometry["x0"]
    y0 = geometry["y0"]
    sma = geometry["sma"]
    eps = geometry["eps"]
    pa = geometry["pa"]

    # Sample at the current SMA (reuse cached data when available).
    if current_data is not None:
        data_c = current_data
    else:
        data_c = extract_isophote_data_multi(
            images, masks, x0, y0, sma, eps, pa,
            use_eccentric_anomaly=config.use_eccentric_anomaly,
            variance_maps=variance_maps,
        )

    if data_c.intens.shape[1] == 0:
        if previous_gradient is not None:
            return previous_gradient * 0.8, None, [], []
        return -1.0, None, [], []

    # Sample at sma + step (linear or geometric).
    if config.linear_growth:
        gradient_sma = sma + config.astep
    else:
        gradient_sma = sma * (1.0 + config.astep)

    data_g = extract_isophote_data_multi(
        images, masks, x0, y0, gradient_sma, eps, pa,
        use_eccentric_anomaly=config.use_eccentric_anomaly,
        variance_maps=variance_maps,
    )

    if data_g.intens.shape[1] == 0:
        if previous_gradient is not None:
            return previous_gradient * 0.8, None, [], []
        return -1.0, None, [], []

    delta_r = config.astep if config.linear_growth else sma * config.astep
    n_bands = data_c.intens.shape[0]
    per_band_grad: List[float] = []
    per_band_err: List[Optional[float]] = []

    for b in range(n_bands):
        intens_c_b = data_c.intens[b]
        intens_g_b = data_g.intens[b]
        if config.integrator == "median":
            mean_c = float(np.median(intens_c_b))
            mean_g = float(np.median(intens_g_b))
        else:
            mean_c = float(np.mean(intens_c_b))
            mean_g = float(np.mean(intens_g_b))
        grad_b = (mean_g - mean_c) / delta_r
        per_band_grad.append(grad_b)

        if data_c.variances is not None and data_g.variances is not None:
            var_c_b = data_c.variances[b]
            var_g_b = data_g.variances[b]
            var_mean_c = float(np.sum(var_c_b)) / max(len(var_c_b), 1) ** 2
            var_mean_g = float(np.sum(var_g_b)) / max(len(var_g_b), 1) ** 2
            err_b = float(np.sqrt(var_mean_c + var_mean_g)) / delta_r
        else:
            sigma_c_b = float(np.std(intens_c_b))
            sigma_g_b = float(np.std(intens_g_b))
            err_b = (
                float(np.sqrt(sigma_c_b**2 / max(len(intens_c_b), 1) + sigma_g_b**2 / max(len(intens_g_b), 1)))
                / delta_r
            )
        per_band_err.append(err_b)

    # Combine per-band gradients with the user-supplied band weights.
    # gradient_joint = Σ w_b grad_b / Σ w_b
    # σ²_joint = Σ w²_b σ²_b / (Σ w_b)²   (independent measurements)
    w_sum = float(band_weights_arr.sum())
    grad_joint = float(np.sum(band_weights_arr * np.array(per_band_grad))) / w_sum

    if any(e is None for e in per_band_err):
        grad_err_joint: Optional[float] = None
    else:
        per_band_err_arr = np.array([e if e is not None else 0.0 for e in per_band_err], dtype=np.float64)
        var_joint = float(np.sum((band_weights_arr**2) * (per_band_err_arr**2))) / (w_sum**2)
        grad_err_joint = float(np.sqrt(var_joint))

    return grad_joint, grad_err_joint, per_band_grad, per_band_err


# ---------------------------------------------------------------------------
# Parameter errors from the joint covariance matrix
# ---------------------------------------------------------------------------


def _compute_parameter_errors_from_joint(
    coeffs: NDArray[np.float64],
    cov_full: Optional[NDArray[np.float64]],
    n_bands: int,
    sma: float,
    eps: float,
    pa: float,
    gradient: float,
    gradient_error: Optional[float],
    angles: NDArray[np.float64],
    intens_per_band: NDArray[np.float64],
    use_exact_covariance: bool,
    var_residual_floor: Optional[float],
) -> Tuple[float, float, float, float]:
    """
    Map the joint (B+4)x(B+4) covariance into geometric parameter errors.

    The shared geometric coefficients (A1, B1, A2, B2) live at indices
    ``[n_bands : n_bands+4]`` of the joint coefficient vector. Their
    diagonal variances (after residual-variance scaling in OLS mode)
    feed the standard Jedrzejewski-1987 error propagation, identical
    to single-band. Returns (x0_err, y0_err, eps_err, pa_err).
    """
    if cov_full is None or gradient is None or abs(gradient) < 1e-10:
        return 0.0, 0.0, 0.0, 0.0
    n_geom_params = n_bands + 4
    n_pixels = intens_per_band.size
    if n_pixels <= n_geom_params:
        return 0.0, 0.0, 0.0, 0.0

    g_err_sq = gradient_error**2 if gradient_error is not None else 0.0
    g_sq = gradient**2

    try:
        if use_exact_covariance:
            covariance = cov_full
        else:
            # OLS: scale the (A^T A)^-1 covariance by the residual variance
            # of the joint fit. Residuals are flattened band-stacked.
            model_per_band = evaluate_joint_model(angles, coeffs, n_bands)
            residuals = (intens_per_band - model_per_band).reshape(-1)
            ddof = n_geom_params
            if len(residuals) <= ddof:
                return 0.0, 0.0, 0.0, 0.0
            var_residual = float(np.var(residuals, ddof=ddof))
            if var_residual_floor is not None:
                var_residual = max(var_residual, var_residual_floor)
            covariance = cov_full * var_residual
        errors = np.sqrt(np.diagonal(covariance))

        sig_a1_sq = float(errors[n_bands] ** 2)
        sig_b1_sq = float(errors[n_bands + 1] ** 2)
        sig_a2_sq = float(errors[n_bands + 2] ** 2)
        sig_b2_sq = float(errors[n_bands + 3] ** 2)

        a1 = float(coeffs[n_bands])
        b1 = float(coeffs[n_bands + 1])
        a2 = float(coeffs[n_bands + 2])
        b2 = float(coeffs[n_bands + 3])

        var_major = (sig_b1_sq + (b1**2 / g_sq) * g_err_sq) / g_sq
        var_minor = ((1.0 - eps) ** 2 / g_sq) * (sig_a1_sq + (a1**2 / g_sq) * g_err_sq)

        x0_err = float(np.sqrt(var_minor * np.sin(pa) ** 2 + var_major * np.cos(pa) ** 2))
        y0_err = float(np.sqrt(var_minor * np.cos(pa) ** 2 + var_major * np.sin(pa) ** 2))

        var_eps = (2.0 * (1.0 - eps) / (sma * gradient)) ** 2 * (sig_b2_sq + (b2**2 / g_sq) * g_err_sq)
        eps_err = float(np.sqrt(var_eps))

        if abs(eps) > np.finfo(float).resolution:
            denom = (1.0 - eps) ** 2 - 1.0
            if abs(denom) < 1e-10:
                denom = -1e-10
            var_pa = (2.0 * (1.0 - eps) / (sma * gradient * denom)) ** 2 * (
                sig_a2_sq + (a2**2 / g_sq) * g_err_sq
            )
            pa_err = float(np.sqrt(var_pa))
        else:
            pa_err = 0.0

        return x0_err, y0_err, eps_err, pa_err
    except (np.linalg.LinAlgError, ValueError) as e:
        warnings.warn(
            f"_compute_parameter_errors_from_joint failed: {e}. Returning zero errors.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0.0, 0.0, 0.0, 0.0


# ---------------------------------------------------------------------------
# Per-isophote multi-band fit
# ---------------------------------------------------------------------------


def _empty_isophote_dict(
    sma: float,
    x0: float,
    y0: float,
    eps: float,
    pa: float,
    bands: Sequence[str],
    use_eccentric_anomaly: bool,
    stop_code: int,
    niter: int,
    debug: bool,
) -> Dict[str, object]:
    """Build a degenerate isophote row with NaN intensities for every band."""
    row: Dict[str, object] = {
        "sma": sma,
        "x0": x0, "y0": y0, "eps": eps, "pa": pa,
        "x0_err": 0.0, "y0_err": 0.0, "eps_err": 0.0, "pa_err": 0.0,
        "rms": float("nan"),
        "stop_code": stop_code,
        "niter": niter,
        "valid": False,
        "use_eccentric_anomaly": use_eccentric_anomaly,
        "tflux_e": float("nan"),
        "tflux_c": float("nan"),
        "npix_e": 0,
        "npix_c": 0,
        "ndata": 0,
        "nflag": 0,
    }
    for b in bands:
        row[f"intens_{b}"] = float("nan")
        row[f"intens_err_{b}"] = float("nan")
        row[f"rms_{b}"] = float("nan")
        row[f"a3_{b}"] = 0.0
        row[f"b3_{b}"] = 0.0
        row[f"a3_err_{b}"] = 0.0
        row[f"b3_err_{b}"] = 0.0
        row[f"a4_{b}"] = 0.0
        row[f"b4_{b}"] = 0.0
        row[f"a4_err_{b}"] = 0.0
        row[f"b4_err_{b}"] = 0.0
        if debug:
            row[f"grad_{b}"] = float("nan")
            row[f"grad_error_{b}"] = float("nan")
            row[f"grad_r_error_{b}"] = float("nan")
    return row


def fit_isophote_mb(
    images: Sequence[NDArray[np.floating]],
    masks: Union[None, NDArray[np.bool_], Sequence[Optional[NDArray[np.bool_]]]],
    sma: float,
    start_geometry: Dict[str, float],
    config: IsosterConfigMB,
    going_inwards: bool = False,
    previous_geometry: Optional[Dict[str, float]] = None,
    variance_maps: Union[None, NDArray[np.floating], Sequence[NDArray[np.floating]]] = None,
) -> Dict[str, object]:
    """
    Fit a single multi-band isophote at the given semi-major axis.

    Mirrors :func:`isoster.fitting.fit_isophote` but with the joint
    design-matrix solver, the combined-scalar radial gradient, per-band
    intensity / harmonic columns, and the experimental-feature reductions
    documented in the Stage-1 plan.
    """
    n_bands = len(config.bands)
    bands = list(config.bands)
    band_weights = config.resolved_band_weights()
    band_weights_arr = np.array([band_weights[b] for b in bands], dtype=np.float64)
    debug = bool(config.debug)

    # Pre-resolve masks once (to float64) for use across iterations. We
    # piggyback on the single-band helper because mask handling in the
    # sampler accepts any list shape including pre-resolved float arrays.
    if isinstance(masks, np.ndarray):
        masks_resolved: Union[None, NDArray[np.float64], List[Optional[NDArray[np.float64]]]] = (
            _prepare_mask_float(masks)
        )
    elif masks is None:
        masks_resolved = None
    else:
        masks_resolved = [_prepare_mask_float(m) if m is not None else None for m in masks]

    x0 = start_geometry["x0"]
    y0 = start_geometry["y0"]
    eps = start_geometry["eps"]
    pa = start_geometry["pa"]

    stop_code = 0
    niter = 0
    best_geometry: Optional[Dict[str, object]] = None
    converged = False
    min_amplitude = float("inf")
    previous_gradient: Optional[float] = None
    lexceed = False

    if config.convergence_scaling == "sector_area":
        n_samples_for_scale = max(64, int(2.0 * np.pi * sma))
        angular_width = 2.0 * np.pi / n_samples_for_scale
        delta_sma_for_scale = sma * config.astep if not config.linear_growth else config.astep
        convergence_scale = max(1.0, sma * delta_sma_for_scale * angular_width)
    elif config.convergence_scaling == "sqrt_sma":
        convergence_scale = max(1.0, float(np.sqrt(sma)))
    else:
        convergence_scale = 1.0

    prev_geom = (x0, y0, eps, pa)
    stable_count = 0
    cached_gradient: Optional[float] = None
    cached_gradient_error: Optional[float] = None
    cached_per_band_grad: List[float] = []
    cached_per_band_grad_err: List[Optional[float]] = []
    no_improvement_count = 0

    last_data: Optional[MultiIsophoteData] = None
    last_per_band_grad: List[float] = []
    last_per_band_grad_err: List[Optional[float]] = []

    # ISOFIT and forced-photometry handling are out of scope; we always
    # fit a 5-param model in joint or ref form.
    min_points = 6 + n_bands  # Decision D9 for joint mode
    use_ref_only = config.harmonic_combination == "ref"

    for i in range(config.maxit):
        niter = i + 1
        data = extract_isophote_data_multi(
            images, masks_resolved, x0, y0, sma, eps, pa,
            use_eccentric_anomaly=config.use_eccentric_anomaly,
            variance_maps=variance_maps,
        )
        last_data = data
        total_points = data.valid_count

        # Per-band sigma clip + AND across bands.
        angles, phi, intens_per_band, variances_per_band, _n_clipped = _per_band_sigma_clip(
            data.angles, data.phi, data.intens, data.variances,
            config.sclip, config.nclip, config.sclip_low, config.sclip_high,
        )
        actual_points = len(angles)

        if total_points > 0 and actual_points < total_points * (1.0 - config.fflag):
            if best_geometry is not None:
                best_geometry["stop_code"] = 1
                best_geometry["niter"] = niter
                return best_geometry
            stop_code = 1
            break

        # In ref mode we use the reference band's intensity vector for the
        # 5-param solve; in joint mode we use the (B+4)-column joint solve.
        # Both modes share validation against `min_points`.
        ref_min_points = 6
        if use_ref_only:
            if actual_points < ref_min_points:
                stop_code = 3
                break
        else:
            if actual_points < min_points:
                stop_code = 3
                break

        if use_ref_only:
            ref_idx = bands.index(config.reference_band)
            intens_ref = intens_per_band[ref_idx]
            variances_ref = variances_per_band[ref_idx] if variances_per_band is not None else None
            coeffs_ref, cov_ref, wls_mode = fit_first_and_second_harmonics_ref(
                angles, intens_ref, variances_ref
            )
            # Widen ref coeffs to a full (B + 4,) layout so downstream
            # bookkeeping is uniform with joint mode. Per-band I0_b for
            # non-reference bands is taken as the (weighted) mean of that
            # band's intensities.
            coeffs = np.zeros(n_bands + 4, dtype=np.float64)
            for b_idx in range(n_bands):
                coeffs[b_idx] = (
                    float(np.mean(intens_per_band[b_idx]))
                    if b_idx != ref_idx
                    else float(coeffs_ref[0])
                )
            coeffs[n_bands:] = coeffs_ref[1:5]
            # cov: only the harmonic block is meaningful in ref mode.
            if cov_ref is not None:
                cov_full = np.zeros((n_bands + 4, n_bands + 4), dtype=np.float64)
                # Preserve per-band I0 diagonal stub variance (mean variance
                # under WLS, sample variance / N under OLS) so error propagation
                # later does not divide by zero when reading diagonals — but we
                # only ever read the harmonic block, so leaving zeros here is
                # also fine. We choose the safe explicit form.
                for b_idx in range(n_bands):
                    if b_idx == ref_idx:
                        cov_full[b_idx, b_idx] = float(cov_ref[0, 0])
                    else:
                        if variances_per_band is not None:
                            cov_full[b_idx, b_idx] = float(np.mean(variances_per_band[b_idx])) / max(
                                actual_points, 1
                            )
                        else:
                            cov_full[b_idx, b_idx] = float(np.var(intens_per_band[b_idx])) / max(
                                actual_points, 1
                            )
                cov_full[n_bands:, n_bands:] = cov_ref[1:5, 1:5]
            else:
                cov_full = None
        else:
            coeffs, cov_full, wls_mode = fit_first_and_second_harmonics_joint(
                angles, intens_per_band, band_weights_arr, variances_per_band
            )

        A1 = float(coeffs[n_bands])
        B1 = float(coeffs[n_bands + 1])
        A2 = float(coeffs[n_bands + 2])
        B2 = float(coeffs[n_bands + 3])

        # Combined gradient.
        if (
            i == 0
            or not config.use_lazy_gradient
            or no_improvement_count >= 3
            or cached_gradient_error is None
            or lexceed
        ):
            geom = {"x0": x0, "y0": y0, "sma": sma, "eps": eps, "pa": pa}
            grad_joint, grad_err_joint, per_band_grad, per_band_err = compute_joint_gradient(
                images, masks_resolved, geom, config, band_weights_arr,
                previous_gradient=previous_gradient,
                variance_maps=variance_maps,
                current_data=data,
            )
            cached_gradient = grad_joint
            cached_gradient_error = grad_err_joint
            cached_per_band_grad = per_band_grad
            cached_per_band_grad_err = per_band_err
            if no_improvement_count >= 3:
                no_improvement_count = 0
        else:
            grad_joint = cached_gradient
            grad_err_joint = cached_gradient_error
            per_band_grad = cached_per_band_grad
            per_band_err = cached_per_band_grad_err

        last_per_band_grad = per_band_grad
        last_per_band_grad_err = per_band_err

        if grad_err_joint is not None:
            previous_gradient = grad_joint

        if grad_joint == 0.0 or grad_joint is None:
            stop_code = -1
            break

        # Gradient-error gate (mirrors single-band behavior).
        gradient_relative_error: Optional[float]
        if grad_err_joint is not None and grad_joint < 0.0:
            gradient_relative_error = abs(grad_err_joint / grad_joint)
        else:
            gradient_relative_error = None
        if not going_inwards:
            if config.permissive_geometry and gradient_relative_error is None:
                pass
            elif (
                gradient_relative_error is None
                or gradient_relative_error > config.maxgerr
                or grad_joint >= 0.0
            ):
                if lexceed:
                    stop_code = -1
                    break
                lexceed = True

        # RMS of the joint model fit (per-band-band-stacked residuals).
        model_per_band = evaluate_joint_model(angles, coeffs, n_bands)
        residuals_flat = (intens_per_band - model_per_band).reshape(-1)
        rms = float(np.std(residuals_flat))

        harmonics = [A1, B1, A2, B2]
        if config.fix_center:
            harmonics[0] = 0.0
            harmonics[1] = 0.0
        if config.fix_pa:
            harmonics[2] = 0.0
        if config.fix_eps:
            harmonics[3] = 0.0
        max_idx = int(np.argmax(np.abs(harmonics)))
        max_amp = harmonics[max_idx]

        effective_amp = abs(max_amp)
        if effective_amp < min_amplitude:
            min_amplitude = effective_amp
            no_improvement_count = 0

            if config.compute_errors:
                x0_err, y0_err, eps_err, pa_err = _compute_parameter_errors_from_joint(
                    coeffs=coeffs,
                    cov_full=cov_full,
                    n_bands=n_bands,
                    sma=sma, eps=eps, pa=pa,
                    gradient=grad_joint,
                    gradient_error=grad_err_joint if config.use_corrected_errors else None,
                    angles=angles,
                    intens_per_band=intens_per_band,
                    use_exact_covariance=wls_mode,
                    var_residual_floor=config.sigma_bg**2 if config.sigma_bg is not None else None,
                )
            else:
                x0_err = y0_err = eps_err = pa_err = 0.0

            # Per-band reported intensity: use the joint coefficient I0_b
            # (which is the WLS / OLS-fitted background level for that
            # band along this isophote, parallel to single-band's `intens`).
            best_geometry = {
                "sma": sma,
                "x0": x0, "y0": y0, "eps": eps, "pa": pa,
                "x0_err": x0_err, "y0_err": y0_err, "eps_err": eps_err, "pa_err": pa_err,
                "rms": rms,
                "valid": True,
                "use_eccentric_anomaly": config.use_eccentric_anomaly,
                "tflux_e": float("nan"),
                "tflux_c": float("nan"),
                "npix_e": 0,
                "npix_c": 0,
            }
            for b_idx, b in enumerate(bands):
                intens_b = float(coeffs[b_idx])
                # Per-band rms from band b residuals; intens_err_b from
                # diagonal of the joint covariance at row b (already an
                # exact covariance under WLS, residual-scaled under OLS).
                rms_b = float(np.std(intens_per_band[b_idx] - model_per_band[b_idx]))
                if cov_full is not None:
                    if wls_mode:
                        intens_err_b = float(np.sqrt(max(cov_full[b_idx, b_idx], 0.0)))
                    else:
                        # OLS: scale the (A^T A)^-1 diagonal by per-band
                        # residual variance (with ddof=B+4 split equally
                        # across bands is ambiguous; we use a per-band
                        # sample variance estimate as a pragmatic choice).
                        ddof_eff = 1 + 4  # one I0_b + 4 shared geometric
                        if len(intens_per_band[b_idx]) > ddof_eff:
                            var_res_b = float(
                                np.var(intens_per_band[b_idx] - model_per_band[b_idx], ddof=ddof_eff)
                            )
                        else:
                            var_res_b = 0.0
                        intens_err_b = float(np.sqrt(max(cov_full[b_idx, b_idx], 0.0) * var_res_b))
                else:
                    intens_err_b = float("nan")
                best_geometry[f"intens_{b}"] = intens_b
                best_geometry[f"intens_err_{b}"] = intens_err_b
                best_geometry[f"rms_{b}"] = rms_b
                # Harmonic deviation placeholders; computed post-hoc on
                # convergence below. Initialize to zeros for unconverged
                # exit paths (matches single-band convention).
                for n_order in (3, 4):
                    best_geometry[f"a{n_order}_{b}"] = 0.0
                    best_geometry[f"b{n_order}_{b}"] = 0.0
                    best_geometry[f"a{n_order}_err_{b}"] = 0.0
                    best_geometry[f"b{n_order}_err_{b}"] = 0.0
                if debug:
                    grad_b = per_band_grad[b_idx] if b_idx < len(per_band_grad) else float("nan")
                    err_b = per_band_err[b_idx] if b_idx < len(per_band_err) else None
                    grad_r_err_b = (
                        abs(err_b / grad_b)
                        if (err_b is not None and grad_b is not None and grad_b != 0.0)
                        else float("nan")
                    )
                    best_geometry[f"grad_{b}"] = float(grad_b)
                    best_geometry[f"grad_error_{b}"] = float(err_b) if err_b is not None else float("nan")
                    best_geometry[f"grad_r_error_{b}"] = float(grad_r_err_b)
            if debug:
                best_geometry["ndata"] = actual_points
                best_geometry["nflag"] = total_points - actual_points
        else:
            no_improvement_count += 1

        # Effective rms for convergence check (decision D17 — sigma_bg honored
        # in multi-band even though variance maps already encode pixel noise).
        effective_rms = rms
        if config.sigma_bg is not None and len(angles) > 0:
            noise_floor = config.sigma_bg / np.sqrt(len(angles))
            effective_rms = max(rms, noise_floor)

        if abs(max_amp) < config.conver * convergence_scale * effective_rms and i >= config.minit:
            stop_code = 0
            converged = True
            if config.compute_deviations and best_geometry is not None:
                _attach_per_band_harmonics(
                    best_geometry, bands, angles, intens_per_band, variances_per_band,
                    sma, per_band_grad,
                )
            break

        # Geometry update (Jedrzejewski-1987, joint gradient denominator).
        damping = config.geometry_damping
        if grad_err_joint is not None and abs(grad_joint) > 0.0:
            grad_snr = abs(grad_joint / grad_err_joint)
            snr_damping = float(np.clip(grad_snr / 3.0, 0.1, 1.0))
            damping *= snr_damping

        if config.geometry_update_mode == "simultaneous":
            # All four parameters update each iteration.
            if not config.fix_center:
                coeff_c_minor = (1.0 - eps) / grad_joint
                coeff_c_major = 1.0 / grad_joint
                aux_minor = -A1 * coeff_c_minor * damping
                aux_major = -B1 * coeff_c_major * damping
                if config.clip_max_shift is not None:
                    max_iter_shift = max(config.clip_max_shift, 0.05 * sma)
                    shift_len = float(np.sqrt(aux_minor**2 + aux_major**2))
                    if shift_len > max_iter_shift:
                        scale = max_iter_shift / shift_len
                        aux_minor *= scale
                        aux_major *= scale
                x0 += -aux_minor * np.sin(pa) + aux_major * np.cos(pa)
                y0 += aux_minor * np.cos(pa) + aux_major * np.sin(pa)
            if not config.fix_pa:
                denom = (1.0 - eps) ** 2 - 1.0
                if abs(denom) < 1e-10:
                    denom = -1e-10
                coeff_pa = 2.0 * (1.0 - eps) / sma / grad_joint / denom
                pa_corr = A2 * coeff_pa * damping
                if config.clip_max_pa is not None:
                    pa_corr = float(np.clip(pa_corr, -config.clip_max_pa, config.clip_max_pa))
                pa = (pa + pa_corr) % np.pi
            if not config.fix_eps:
                coeff_eps = 2.0 * (1.0 - eps) / sma / grad_joint
                eps_corr = B2 * coeff_eps * damping
                if config.clip_max_eps is not None:
                    eps_corr = float(np.clip(eps_corr, -config.clip_max_eps, config.clip_max_eps))
                eps = min(eps - eps_corr, 0.95)
                if eps < 0.0:
                    eps = min(-eps, 0.95)
                    pa = (pa + np.pi / 2) % np.pi
                if eps == 0.0:
                    eps = 0.05
        else:
            # 'largest' mode: update only the geometry parameter with the
            # largest |harmonic|.
            if max_idx == 0 and not config.fix_center:
                coeff = (1.0 - eps) / grad_joint
                aux = -max_amp * coeff * damping
                if config.clip_max_shift is not None:
                    aux = float(np.clip(aux, -config.clip_max_shift, config.clip_max_shift))
                x0 -= aux * np.sin(pa)
                y0 += aux * np.cos(pa)
            elif max_idx == 1 and not config.fix_center:
                coeff = 1.0 / grad_joint
                aux = -max_amp * coeff * damping
                if config.clip_max_shift is not None:
                    aux = float(np.clip(aux, -config.clip_max_shift, config.clip_max_shift))
                x0 += aux * np.cos(pa)
                y0 += aux * np.sin(pa)
            elif max_idx == 2 and not config.fix_pa:
                denom = (1.0 - eps) ** 2 - 1.0
                if abs(denom) < 1e-10:
                    denom = -1e-10
                coeff = 2.0 * (1.0 - eps) / sma / grad_joint / denom
                pa_corr = max_amp * coeff * damping
                if config.clip_max_pa is not None:
                    pa_corr = float(np.clip(pa_corr, -config.clip_max_pa, config.clip_max_pa))
                pa = (pa + pa_corr) % np.pi
            elif max_idx == 3 and not config.fix_eps:
                coeff = 2.0 * (1.0 - eps) / sma / grad_joint
                eps_corr = max_amp * coeff * damping
                if config.clip_max_eps is not None:
                    eps_corr = float(np.clip(eps_corr, -config.clip_max_eps, config.clip_max_eps))
                eps = min(eps - eps_corr, 0.95)
                if eps < 0.0:
                    eps = min(-eps, 0.95)
                    pa = (pa + np.pi / 2) % np.pi
                if eps == 0.0:
                    eps = 0.05

        if config.geometry_convergence and i >= config.minit:
            gx0, gy0, geps, gpa = prev_geom
            delta_x0 = abs(x0 - gx0) / max(sma, 1.0)
            delta_y0 = abs(y0 - gy0) / max(sma, 1.0)
            delta_eps = abs(eps - geps)
            delta_pa_raw = abs(pa - gpa)
            delta_pa = min(delta_pa_raw, np.pi - delta_pa_raw) / np.pi
            max_delta = max(delta_x0, delta_y0, delta_eps, delta_pa)
            if max_delta < config.geometry_tolerance:
                stable_count += 1
            else:
                stable_count = 0
            if stable_count >= config.geometry_stable_iters:
                stop_code = 0
                converged = True
                if config.compute_deviations and best_geometry is not None:
                    _attach_per_band_harmonics(
                        best_geometry, bands, angles, intens_per_band, variances_per_band,
                        sma, per_band_grad,
                    )
                break

        prev_geom = (x0, y0, eps, pa)

    # Wrap up.
    if best_geometry is None:
        best_geometry = _empty_isophote_dict(
            sma, x0, y0, eps, pa, bands, config.use_eccentric_anomaly,
            stop_code if stop_code != 0 else 2, niter, debug,
        )

    if niter >= config.maxit and stop_code == 0 and not converged:
        stop_code = 2
        # Best-effort post-hoc harmonics from the final iteration's data.
        if (
            config.compute_deviations
            and last_data is not None
            and last_data.valid_count > 6
        ):
            _attach_per_band_harmonics(
                best_geometry,
                bands,
                last_data.angles,
                last_data.intens,
                last_data.variances,
                sma,
                last_per_band_grad if last_per_band_grad else [0.0] * n_bands,
            )

    if config.full_photometry:
        # Per-band aperture totals: same elliptical aperture, B independent
        # photometric integrations. tflux_e / tflux_c columns are written
        # per band as `tflux_e_<b>`, `tflux_c_<b>`.
        for b_idx, b in enumerate(bands):
            mask_b = None
            if isinstance(masks, np.ndarray):
                mask_b = masks
            elif masks is not None:
                mask_b = masks[b_idx]
            tflux_e_b, tflux_c_b, npix_e_b, npix_c_b = compute_aperture_photometry(
                np.asarray(images[b_idx]), mask_b,
                float(best_geometry["x0"]), float(best_geometry["y0"]),
                float(best_geometry["sma"]),
                float(best_geometry["eps"]), float(best_geometry["pa"]),
            )
            best_geometry[f"tflux_e_{b}"] = float(tflux_e_b)
            best_geometry[f"tflux_c_{b}"] = float(tflux_c_b)
            best_geometry[f"npix_e_{b}"] = int(npix_e_b)
            best_geometry[f"npix_c_{b}"] = int(npix_c_b)

    best_geometry["stop_code"] = stop_code
    best_geometry["niter"] = niter
    return best_geometry


def _attach_per_band_harmonics(
    geom: Dict[str, object],
    bands: Sequence[str],
    angles: NDArray[np.float64],
    intens_per_band: NDArray[np.float64],
    variances_per_band: Optional[NDArray[np.float64]],
    sma: float,
    per_band_grad: Sequence[float],
) -> None:
    """Compute per-band a3, b3, a4, b4 and write them into ``geom``.

    Each band uses its own intensity vector and its own gradient,
    matching the single-band ``compute_deviations`` path. Bender
    normalization happens at plotting time (decision D16).
    """
    for b_idx, b in enumerate(bands):
        intens_b = intens_per_band[b_idx]
        var_b = variances_per_band[b_idx] if variances_per_band is not None else None
        grad_b = float(per_band_grad[b_idx]) if b_idx < len(per_band_grad) else 0.0
        for n_order in (3, 4):
            a, c, a_err, b_err = compute_deviations(
                angles, intens_b, sma, grad_b, n_order, variances=var_b,
            )
            geom[f"a{n_order}_{b}"] = float(a)
            geom[f"b{n_order}_{b}"] = float(c)
            geom[f"a{n_order}_err_{b}"] = float(a_err)
            geom[f"b{n_order}_err_{b}"] = float(b_err)


# ---------------------------------------------------------------------------
# Forced multi-band photometry helper (used by the driver's central pixel
# and template fallback paths)
# ---------------------------------------------------------------------------


def extract_forced_photometry_mb(
    images: Sequence[NDArray[np.floating]],
    masks: Union[None, NDArray[np.bool_], Sequence[Optional[NDArray[np.bool_]]]],
    x0: float,
    y0: float,
    sma: float,
    eps: float,
    pa: float,
    bands: Sequence[str],
    config: IsosterConfigMB,
    variance_maps: Union[None, NDArray[np.floating], Sequence[NDArray[np.floating]]] = None,
) -> Dict[str, object]:
    """
    Single-isophote forced multi-band extraction (no fitting).

    Used by the driver as the central-pixel record for ``minsma == 0.0``
    growth and as a defensive fallback when an isophote fails the
    iterative fit. Produces the same per-isophote dict layout as
    :func:`fit_isophote_mb` with ``stop_code=0`` and ``niter=0``.
    """
    debug = bool(config.debug)
    band_list = list(bands)
    data = extract_isophote_data_multi(
        images, masks, x0, y0, sma, eps, pa,
        use_eccentric_anomaly=config.use_eccentric_anomaly,
        variance_maps=variance_maps,
    )
    if data.valid_count == 0:
        return _empty_isophote_dict(
            sma, x0, y0, eps, pa, band_list, config.use_eccentric_anomaly,
            stop_code=3, niter=0, debug=debug,
        )

    geom: Dict[str, object] = {
        "sma": sma,
        "x0": x0, "y0": y0, "eps": eps, "pa": pa,
        "x0_err": 0.0, "y0_err": 0.0, "eps_err": 0.0, "pa_err": 0.0,
        "rms": float("nan"),
        "stop_code": 0,
        "niter": 0,
        "valid": True,
        "use_eccentric_anomaly": config.use_eccentric_anomaly,
        "tflux_e": float("nan"),
        "tflux_c": float("nan"),
        "npix_e": 0,
        "npix_c": 0,
    }
    if debug:
        geom["ndata"] = data.valid_count
        geom["nflag"] = data.n_samples - data.valid_count

    for b_idx, b in enumerate(band_list):
        intens_b = data.intens[b_idx]
        if data.variances is not None:
            v_b = data.variances[b_idx]
            weights = 1.0 / v_b
            sum_w = float(weights.sum())
            intens_val = float((weights * intens_b).sum() / sum_w)
            intens_err = float(1.0 / np.sqrt(sum_w))
        elif config.integrator == "median":
            intens_val = float(np.median(intens_b))
            intens_err = float(np.std(intens_b) / np.sqrt(len(intens_b)))
        else:
            intens_val = float(np.mean(intens_b))
            intens_err = float(np.std(intens_b) / np.sqrt(len(intens_b)))
        rms_b = float(np.std(intens_b))
        geom[f"intens_{b}"] = intens_val
        geom[f"intens_err_{b}"] = intens_err
        geom[f"rms_{b}"] = rms_b
        for n_order in (3, 4):
            geom[f"a{n_order}_{b}"] = 0.0
            geom[f"b{n_order}_{b}"] = 0.0
            geom[f"a{n_order}_err_{b}"] = 0.0
            geom[f"b{n_order}_err_{b}"] = 0.0
        if debug:
            geom[f"grad_{b}"] = float("nan")
            geom[f"grad_error_{b}"] = float("nan")
            geom[f"grad_r_error_{b}"] = float("nan")
    return geom
