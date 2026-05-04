"""
Multi-band isoster driver (``fit_image_multiband``).

Single public entry point for joint multi-band free fitting. Mirrors the
single-band :func:`isoster.driver.fit_image` orchestration:

- Central pixel record at SMA = 0 (when ``minsma <= 0``).
- First-isophote retry schedule.
- Inward growth from the anchor down to ``minsma``.
- Outward growth from the anchor out to ``maxsma``.
- Final result list assembled in ascending-SMA order.

Stage-1 omits LSB auto-lock and outer-center regularization (see
``docs/agent/plan-2026-04-29-multiband-feasibility.md`` decision D13).
B=1 input delegates to the existing :func:`isoster.driver.fit_image`
and returns the legacy single-band schema unmodified (decision D14).
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from ..config import IsosterConfig
from ..driver import fit_image
from .config_mb import IsosterConfigMB
from .fitting_mb import (
    _PER_BAND_DEBUG_KEYS,
    extract_forced_photometry_mb,
    fit_isophote_mb,
)
from .sampling_mb import prepare_inputs

ACCEPTABLE_STOP_CODES = {0, 1, 2}


# ---------------------------------------------------------------------------
# Lightweight helpers
# ---------------------------------------------------------------------------


def _is_acceptable(iso: Dict[str, object]) -> bool:
    return int(iso.get("stop_code", -1)) in ACCEPTABLE_STOP_CODES  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Stage-3 Stage-C: LSB auto-lock helpers (port of isoster.driver._is_lsb_isophote
# / _mark_lsb_lock_state / _build_locked_cfg, adapted to multi-band)
# ---------------------------------------------------------------------------


def _is_lsb_isophote_mb(iso: Dict[str, object], maxgerr_thresh: float) -> bool:
    """Lock-trigger predicate for the multi-band outward sweep.

    Plan section 7 S3: the trigger surface is the **joint combined
    gradient** (``grad_joint`` / ``grad_err_joint``), exposed by the
    fitter as top-level ``grad`` / ``grad_error`` row keys when
    ``debug=True``. A relative joint gradient error above
    ``maxgerr_thresh``, OR a stop_code=-1 (gradient-error stop), OR a
    non-negative joint gradient (galaxies have negative dI/da; positive
    values mean the fit has lost the profile) all count as triggering.

    None / non-finite values mean "cannot assess" and return False —
    the next isophote will get another chance to trigger.
    """
    if iso.get("stop_code") == -1:
        return True
    grad = iso.get("grad")
    grad_err = iso.get("grad_error")
    if grad is None or grad_err is None:
        return False
    try:
        grad_f = float(grad)
        grad_err_f = float(grad_err)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(grad_f) or not np.isfinite(grad_err_f) or grad_err_f == 0.0:
        return False
    if grad_f >= 0.0:
        return True
    return abs(grad_err_f / grad_f) > maxgerr_thresh


def _mark_lsb_lock_state_mb(
    iso: Dict[str, object], locked: bool, is_anchor: bool = False,
) -> None:
    """Stamp ``lsb_locked`` and (optionally) ``lsb_auto_lock_anchor``
    onto a multi-band result row. Centralized so future loop reorders
    cannot accidentally write the wrong flag."""
    iso["lsb_locked"] = bool(locked)
    if is_anchor:
        iso["lsb_auto_lock_anchor"] = True


def _build_locked_cfg_mb(
    cfg: IsosterConfigMB, anchor_iso: Dict[str, object], locked_integrator: str,
) -> IsosterConfigMB:
    """Clone the multi-band config and freeze geometry to a clean anchor.

    Mirrors :func:`isoster.driver._build_locked_cfg` with two multi-
    band-specific tweaks:

    1. When ``locked_integrator='median'``, also flip
       ``fit_per_band_intens_jointly=False`` on the clone so Stage A's
       S1 validator accepts the result. The config-time validator
       already rejects ``lsb_auto_lock_integrator='median'`` ∧
       ``fit_per_band_intens_jointly=True`` (S1 × S3 composition);
       this branch makes the lock-fire path consistent regardless of
       what the user originally set.
    2. Disable outer-region damping on the clone — it makes no sense
       once geometry is hard-locked, and would consume the outer
       reference geom for nothing.
    """
    locked = cfg.model_copy(deep=True)
    locked.x0 = float(anchor_iso["x0"])  # type: ignore[arg-type]
    locked.y0 = float(anchor_iso["y0"])  # type: ignore[arg-type]
    locked.eps = float(anchor_iso["eps"])  # type: ignore[arg-type]
    locked.pa = float(anchor_iso["pa"])  # type: ignore[arg-type]
    locked.fix_center = True
    locked.fix_pa = True
    locked.fix_eps = True
    locked.integrator = locked_integrator  # type: ignore[assignment]
    if locked_integrator == "median":
        # Median requires the decoupled intercept mode (Stage A S1).
        locked.fit_per_band_intens_jointly = False
    # The auto-lock has already committed; the clone should not try to
    # re-detect on its own isophotes.
    locked.lsb_auto_lock = False
    # Outer-region damping is meaningless once geometry is frozen.
    locked.use_outer_center_regularization = False
    return locked


def _build_outer_reference_mb(
    inwards_results: Sequence[Dict[str, object]],
    anchor_iso: Dict[str, object],
    config: IsosterConfigMB,
) -> Dict[str, float]:
    """Frozen reference geometry for the multi-band outer-region damping.

    Mirrors :func:`isoster.driver._build_outer_reference`: takes a flux-
    weighted mean of inward isophotes (plus the anchor) within a sma
    window of ``sma0 * config.outer_reg_ref_sma_factor``. Multi-band
    geometry is shared, so only the flux weight differs from single-band:
    we use ``intens_<reference_band>`` since the geometry was driven
    against that band's diagnostics. The reference carries x0/y0/eps/pa;
    pa uses a flux-weighted circular mean on 2*pa (axis-like angles).
    """
    ref_band = config.reference_band
    intens_key = f"intens_{ref_band}"

    def _intens(iso: Dict[str, object]) -> Optional[float]:
        val = iso.get(intens_key)
        if val is None:
            return None
        v = float(val)
        return v if np.isfinite(v) else None

    anchor_ref = {
        "x0": float(anchor_iso["x0"]),  # type: ignore[arg-type]
        "y0": float(anchor_iso["y0"]),  # type: ignore[arg-type]
        "eps": float(anchor_iso["eps"]),  # type: ignore[arg-type]
        "pa": float(anchor_iso["pa"]),  # type: ignore[arg-type]
    }

    sma_hi = float(anchor_iso["sma"]) * config.outer_reg_ref_sma_factor  # type: ignore[arg-type]

    candidates: List[Dict[str, object]] = [anchor_iso]
    for iso in inwards_results:
        if not _is_acceptable(iso):
            continue
        sma_v = iso.get("sma")
        if sma_v is None or not np.isfinite(float(sma_v)):  # type: ignore[arg-type]
            continue
        if float(sma_v) > sma_hi:  # type: ignore[arg-type]
            continue
        intens_v = _intens(iso)
        if intens_v is None:
            continue
        for k in ("x0", "y0", "eps", "pa"):
            v = iso.get(k)
            if v is None or not np.isfinite(float(v)):  # type: ignore[arg-type]
                break
        else:
            candidates.append(iso)

    x0s = np.array([float(iso["x0"]) for iso in candidates])  # type: ignore[arg-type]
    y0s = np.array([float(iso["y0"]) for iso in candidates])  # type: ignore[arg-type]
    eps_arr = np.array([float(iso["eps"]) for iso in candidates])  # type: ignore[arg-type]
    pa_arr = np.array([float(iso["pa"]) for iso in candidates])  # type: ignore[arg-type]
    weights = np.array(
        [max(_intens(iso) or 1e-6, 1e-6) for iso in candidates], dtype=np.float64,
    )
    if weights.sum() <= 0.0 or not np.all(np.isfinite(weights)):
        return anchor_ref

    x0_ref = float(np.average(x0s, weights=weights))
    y0_ref = float(np.average(y0s, weights=weights))
    eps_ref = float(np.average(eps_arr, weights=weights))

    cos_sum = float(np.sum(weights * np.cos(2.0 * pa_arr)))
    sin_sum = float(np.sum(weights * np.sin(2.0 * pa_arr)))
    w_sum = float(weights.sum())
    resultant = np.hypot(cos_sum, sin_sum) / w_sum if w_sum > 0.0 else 0.0
    if resultant >= 0.1:
        pa_ref = 0.5 * float(np.arctan2(sin_sum, cos_sum))
        pa_ref = pa_ref % np.pi
    else:
        pa_ref = float(anchor_iso["pa"]) % np.pi  # type: ignore[arg-type]

    if not all(np.isfinite([x0_ref, y0_ref, eps_ref, pa_ref])):
        return anchor_ref
    return {"x0": x0_ref, "y0": y0_ref, "eps": eps_ref, "pa": pa_ref}


def _validate_inputs(
    images: Sequence[NDArray[np.floating]],
    masks: Union[None, NDArray[np.bool_], Sequence[Optional[NDArray[np.bool_]]]],
    variance_maps: Union[None, NDArray[np.floating], Sequence[NDArray[np.floating]]],
    config: IsosterConfigMB,
) -> None:
    """Cross-check shapes and band counts before any sampling happens."""
    if len(images) != len(config.bands):
        raise ValueError(
            f"len(images) ({len(images)}) does not match len(config.bands) "
            f"({len(config.bands)}). Each image must correspond to a band name."
        )
    h, w = images[0].shape
    for i, im in enumerate(images):
        if im.shape != (h, w):
            raise ValueError(
                f"images[{i}] shape {im.shape} does not match images[0] shape {(h, w)}."
            )
    # Mask shape consistency is delegated to the sampler. Variance map
    # all-or-nothing semantics likewise. We validate band counts here so
    # the user gets a clear error before any expensive call.
    n_bands = len(config.bands)
    if variance_maps is not None and not isinstance(variance_maps, np.ndarray):
        try:
            n_var = len(variance_maps)
        except TypeError as exc:
            raise TypeError(
                "variance_maps must be None, a single ndarray, or a sequence "
                f"of length {n_bands}; got non-sequence {type(variance_maps).__name__}."
            ) from exc
        if n_var != n_bands:
            raise ValueError(
                f"len(variance_maps) ({n_var}) does not match "
                f"len(config.bands) ({n_bands})."
            )
    if masks is not None and not isinstance(masks, np.ndarray):
        try:
            n_masks = len(masks)
        except TypeError as exc:
            raise TypeError(
                "masks must be None, a single boolean ndarray, or a sequence "
                f"of length {n_bands}; got non-sequence {type(masks).__name__}."
            ) from exc
        if n_masks != n_bands:
            raise ValueError(
                f"len(masks) ({n_masks}) does not match "
                f"len(config.bands) ({n_bands})."
            )


def _delegate_single_band(
    images: Sequence[NDArray[np.floating]],
    masks: Union[None, NDArray[np.bool_], Sequence[Optional[NDArray[np.bool_]]]],
    variance_maps: Union[None, NDArray[np.floating], Sequence[NDArray[np.floating]]],
    config_mb: IsosterConfigMB,
) -> Dict[str, object]:
    """Decision D14: B=1 input delegates to single-band fit_image."""
    warnings.warn(
        "Single-band input (len(bands)==1): delegating to "
        "isoster.fit_image. The legacy single-band schema is produced "
        "(no _<band> column suffixes, no multi-band top-level keys).",
        UserWarning,
        stacklevel=3,
    )
    image = np.asarray(images[0])
    if isinstance(masks, np.ndarray):
        mask = masks
    elif masks is None:
        mask = None
    else:
        mask = masks[0]
    if isinstance(variance_maps, np.ndarray):
        var = variance_maps
    elif variance_maps is None:
        var = None
    else:
        var = variance_maps[0]

    # Build a single-band IsosterConfig that mirrors the multi-band knobs
    # we share. Multi-band-only fields are dropped; ISOFIT/LSB-lock
    # defaults stay off.
    sb_kwargs = dict(
        x0=config_mb.x0, y0=config_mb.y0, eps=config_mb.eps, pa=config_mb.pa,
        sma0=config_mb.sma0, minsma=config_mb.minsma, maxsma=config_mb.maxsma,
        astep=config_mb.astep, linear_growth=config_mb.linear_growth,
        maxit=config_mb.maxit, minit=config_mb.minit, conver=config_mb.conver,
        convergence_scaling=config_mb.convergence_scaling,
        sigma_bg=config_mb.sigma_bg,
        use_corrected_errors=config_mb.use_corrected_errors,
        use_lazy_gradient=config_mb.use_lazy_gradient,
        geometry_damping=config_mb.geometry_damping,
        clip_max_shift=config_mb.clip_max_shift,
        clip_max_pa=config_mb.clip_max_pa,
        clip_max_eps=config_mb.clip_max_eps,
        geometry_convergence=config_mb.geometry_convergence,
        geometry_tolerance=config_mb.geometry_tolerance,
        geometry_stable_iters=config_mb.geometry_stable_iters,
        geometry_update_mode=config_mb.geometry_update_mode,
        sclip=config_mb.sclip, nclip=config_mb.nclip,
        sclip_low=config_mb.sclip_low, sclip_high=config_mb.sclip_high,
        fflag=config_mb.fflag, maxgerr=config_mb.maxgerr,
        fix_center=config_mb.fix_center, fix_pa=config_mb.fix_pa,
        fix_eps=config_mb.fix_eps,
        compute_errors=config_mb.compute_errors,
        compute_deviations=config_mb.compute_deviations,
        full_photometry=config_mb.full_photometry,
        debug=config_mb.debug,
        integrator=config_mb.integrator,
        use_eccentric_anomaly=config_mb.use_eccentric_anomaly,
        permissive_geometry=config_mb.permissive_geometry,
        max_retry_first_isophote=config_mb.max_retry_first_isophote,
        first_isophote_fail_count=config_mb.first_isophote_fail_count,
    )
    cfg_sb = IsosterConfig(**sb_kwargs)
    return fit_image(image, mask=mask, config=cfg_sb, variance_map=var)


def _first_isophote_perturbations(
    sma0: float, eps: float, pa: float, max_retries: int,
) -> List[tuple]:
    """Mirror :func:`isoster.driver._first_isophote_perturbations`."""
    schedule = [
        (0.8, eps, pa),
        (1.3, eps, pa),
        (0.6, 0.05, pa),
        (1.5, eps, pa + np.pi / 4),
        (0.5, 0.05, pa + np.pi / 2),
    ]
    extended_factors = [0.4, 0.7, 1.1, 1.6, 2.0]
    for i, factor in enumerate(extended_factors):
        ext_eps = 0.05 if i % 2 == 0 else eps
        ext_pa = pa + (i + 1) * np.pi / 6
        schedule.append((factor, ext_eps, ext_pa))
    out: List[tuple] = []
    for i in range(min(max_retries, len(schedule))):
        sma_factor, eps_new, pa_new = schedule[i]
        sma_new = max(1.0, sma0 * sma_factor)
        eps_new = min(max(0.0, eps_new), 0.99)
        out.append((sma_new, eps_new, pa_new))
    return out


def _validate_non_negative_error_fields(isophotes: Sequence[Dict[str, object]]) -> None:
    """Mirror the single-band validator: any negative error => hard error."""
    for idx, iso in enumerate(isophotes):
        for field, value in iso.items():
            if not (field.endswith("_err") or field.endswith("_error")):
                continue
            try:
                v = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            if np.isfinite(v) and v < 0.0:
                raise ValueError(
                    f"multiband isoster produced negative error value "
                    f"(field={field}, index={idx}, value={v})"
                )


# ---------------------------------------------------------------------------
# Central pixel (multi-band)
# ---------------------------------------------------------------------------


def _fit_central_pixel_mb(
    images: Sequence[NDArray[np.floating]],
    masks: Union[None, NDArray[np.bool_], Sequence[Optional[NDArray[np.bool_]]]],
    x0: float, y0: float,
    config: IsosterConfigMB,
) -> Dict[str, object]:
    """Per-band central-pixel record at SMA=0."""
    bands = list(config.bands)
    debug = bool(config.debug)
    h, w = np.asarray(images[0]).shape
    iy = int(np.round(y0))
    ix = int(np.round(x0))
    in_bounds = 0 <= iy < h and 0 <= ix < w

    row: Dict[str, object] = {
        "sma": 0.0,
        "x0": x0, "y0": y0, "eps": 0.0, "pa": 0.0,
        "x0_err": 0.0, "y0_err": 0.0, "eps_err": 0.0, "pa_err": 0.0,
        "rms": 0.0,
        "stop_code": 0 if in_bounds else -1,
        "niter": 0,
        "valid": in_bounds,
        "use_eccentric_anomaly": config.use_eccentric_anomaly,
        "tflux_e": float("nan"),
        "tflux_c": float("nan"),
        "npix_e": 0,
        "npix_c": 0,
    }
    if debug:
        row["ndata"] = 1 if in_bounds else 0
        row["nflag"] = 0
    for b_idx, b in enumerate(bands):
        if in_bounds:
            mask_b = None
            if isinstance(masks, np.ndarray):
                mask_b = masks
            elif masks is not None:
                mask_b = masks[b_idx]
            val = float(np.asarray(images[b_idx])[iy, ix])
            valid_b = True
            if mask_b is not None and bool(mask_b[iy, ix]):
                valid_b = False
                val = float("nan")
        else:
            val = float("nan")
            valid_b = False
        row[f"intens_{b}"] = val if valid_b else float("nan")
        row[f"intens_err_{b}"] = 0.0
        row[f"rms_{b}"] = 0.0
        for n_order in config.harmonic_orders:
            for prefix in ("a", "b"):
                row[f"{prefix}{int(n_order)}_{b}"] = 0.0
                row[f"{prefix}{int(n_order)}_err_{b}"] = 0.0
        if debug:
            for key in _PER_BAND_DEBUG_KEYS:
                row[f"{key}_{b}"] = float("nan")
        # D9 backport: per-band surviving-sample count. The central pixel
        # is a single point, not a ring, so a value of 1 (or 0 if masked)
        # is the most honest report.
        row[f"n_valid_{b}"] = 1 if valid_b else 0
    return row


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def fit_image_multiband(
    images: Sequence[NDArray[np.floating]],
    masks: Union[None, NDArray[np.bool_], Sequence[Optional[NDArray[np.bool_]]]] = None,
    config: Optional[IsosterConfigMB] = None,
    variance_maps: Union[None, NDArray[np.floating], Sequence[NDArray[np.floating]]] = None,
) -> Dict[str, object]:
    """
    Fit isophotes jointly across multiple aligned same-pixel-grid images.

    Single shared geometry per SMA, per-band intensities and per-band
    harmonic deviations. Replaces the traditional forced-photometry
    workflow (decision D1).

    Parameters
    ----------
    images : sequence of ndarray
        ``B`` aligned images of shape ``(H, W)``. The order must match
        ``config.bands``.
    masks : None | ndarray | sequence of (ndarray|None), optional
        Bad-pixel masks. See :func:`isoster.multiband.sampling_mb.extract_isophote_data_multi`.
    config : IsosterConfigMB
        Multi-band configuration. ``bands`` and ``reference_band`` are
        required; everything else has sensible defaults.
    variance_maps : None | ndarray | sequence of ndarray, optional
        Per-pixel variance for WLS. All-or-nothing.

    Returns
    -------
    result : dict
        ``result['isophotes']`` is a list of per-isophote dicts in
        ascending SMA order. ``result['config']`` is the resolved
        :class:`IsosterConfigMB`. Multi-band top-level keys (decision D14):
        ``'bands'``, ``'multiband'``, ``'harmonic_combination'``,
        ``'reference_band'``, ``'band_weights'``, ``'variance_mode'``.

    Notes
    -----
    When ``len(config.bands) == 1`` this delegates to
    :func:`isoster.fit_image` and returns the legacy single-band schema
    unmodified (with an informational warning).
    """
    if config is None:
        raise ValueError("config is required (IsosterConfigMB with bands and reference_band)")
    _validate_inputs(images, masks, variance_maps, config)

    # B=1 fallback (decision D14).
    if len(config.bands) == 1:
        return _delegate_single_band(images, masks, variance_maps, config)

    # Surface the variance mode on the resolved config so it lands in
    # the FITS CONFIG HDU later.
    variance_mode = "wls" if variance_maps is not None else "ols"
    config.variance_mode = variance_mode

    # Pre-resolve image / mask / variance arrays exactly once and reuse
    # across every per-isophote call. This is the dominant performance
    # win identified in Stage-1 benchmarking (decision D19).
    image_stack, masks_resolved, var_stack = prepare_inputs(
        images, masks, variance_maps,
    )

    h, w = np.asarray(images[0]).shape
    x0 = config.x0 if config.x0 is not None else w / 2.0
    y0 = config.y0 if config.y0 is not None else h / 2.0
    sma0 = config.sma0
    minsma = config.minsma
    maxsma = config.maxsma if config.maxsma is not None else float(np.hypot(h, w) / 2.0)
    astep = config.astep
    linear_growth = config.linear_growth

    # Central-pixel record (when minsma <= 0).
    central_result = _fit_central_pixel_mb(images, masks, x0, y0, config)

    # First isophote at sma0.
    start_geometry = {"x0": x0, "y0": y0, "eps": config.eps, "pa": config.pa}
    first_iso = fit_isophote_mb(
        images, masks, sma0, start_geometry, config, variance_maps=variance_maps,
        image_stack=image_stack, masks_resolved=masks_resolved, var_stack=var_stack,
    )

    retry_log: List[Dict[str, object]] = []
    if not _is_acceptable(first_iso) and config.max_retry_first_isophote > 0:
        perturbations = _first_isophote_perturbations(
            sma0, config.eps, config.pa, config.max_retry_first_isophote,
        )
        for attempt_idx, (sma0_try, eps_try, pa_try) in enumerate(perturbations, start=1):
            retry_geom = {"x0": x0, "y0": y0, "eps": eps_try, "pa": pa_try}
            cand = fit_isophote_mb(
                images, masks, sma0_try, retry_geom, config, variance_maps=variance_maps,
                image_stack=image_stack, masks_resolved=masks_resolved, var_stack=var_stack,
            )
            retry_log.append({
                "attempt": attempt_idx,
                "sma0": sma0_try, "eps": eps_try, "pa": pa_try,
                "stop_code": cand["stop_code"],
            })
            if _is_acceptable(cand):
                first_iso = cand
                sma0 = sma0_try
                break

    anchor_iso: Optional[Dict[str, object]] = None
    first_isophote_failure = False
    if _is_acceptable(first_iso):
        anchor_iso = first_iso
    else:
        failed_initial = [first_iso]
        probe_sma = sma0
        n_extra_probes = config.first_isophote_fail_count - 1
        for _ in range(n_extra_probes):
            probe_sma = probe_sma + astep if linear_growth else probe_sma * (1.0 + astep)
            if probe_sma > maxsma:
                break
            probe_iso = fit_isophote_mb(
                images, masks, probe_sma, start_geometry, config,
                variance_maps=variance_maps,
                image_stack=image_stack, masks_resolved=masks_resolved, var_stack=var_stack,
            )
            failed_initial.append(probe_iso)
            if _is_acceptable(probe_iso):
                anchor_iso = probe_iso
                sma0 = probe_sma
                break
        if anchor_iso is None:
            first_isophote_failure = True
            stop_codes = [iso["stop_code"] for iso in failed_initial]
            warnings.warn(
                f"FIRST_FEW_ISOPHOTE_FAILURE (multiband): the first {len(failed_initial)} "
                f"isophotes (starting sma0={config.sma0:.2f}) all failed with stop "
                f"codes {stop_codes}. Consider adjusting sma0, initial geometry, or "
                f"max_retry_first_isophote.",
                RuntimeWarning,
                stacklevel=2,
            )

    # Inward growth.
    inwards_results: List[Dict[str, object]] = []
    if minsma < sma0 and anchor_iso is not None:
        current_iso: Dict[str, object] = anchor_iso
        current_sma = float(anchor_iso["sma"])  # type: ignore[arg-type]
        while True:
            next_sma = current_sma - astep if linear_growth else current_sma / (1.0 + astep)
            limit_sma = max(minsma, 0.5)
            if next_sma < limit_sma:
                break
            current_sma = next_sma
            current_geom = {
                "x0": float(current_iso["x0"]),  # type: ignore[arg-type]
                "y0": float(current_iso["y0"]),  # type: ignore[arg-type]
                "eps": float(current_iso["eps"]),  # type: ignore[arg-type]
                "pa": float(current_iso["pa"]),  # type: ignore[arg-type]
            }
            next_iso = fit_isophote_mb(
                images, masks, current_sma, current_geom, config,
                going_inwards=True,
                previous_geometry=current_geom,
                variance_maps=variance_maps,
                image_stack=image_stack, masks_resolved=masks_resolved, var_stack=var_stack,
            )
            inwards_results.append(next_iso)
            if _is_acceptable(next_iso) or config.permissive_geometry:
                current_iso = next_iso

    # Build the frozen outer-region reference geometry once, when the
    # outer damping feature is on and we have an anchor (Stage-3 Stage-B).
    outer_ref_geom: Optional[Dict[str, float]] = None
    if config.use_outer_center_regularization and anchor_iso is not None:
        outer_ref_geom = _build_outer_reference_mb(
            inwards_results, anchor_iso, config,
        )

    # Outward growth.
    outwards_results: List[Dict[str, object]] = []
    # Stage-3 Stage-C: LSB auto-lock state. Only mutated when
    # config.lsb_auto_lock=True; the dict is always initialized so
    # the post-loop metadata block stays uniform. Mirrors single-band.
    lsb_state: Dict[str, object] = {
        "locked": False,
        "consec": 0,
        "transition_sma": None,
        "anchor_index": None,
    }
    active_cfg: IsosterConfigMB = config
    if anchor_iso is not None:
        if config.lsb_auto_lock:
            _mark_lsb_lock_state_mb(anchor_iso, locked=False)
        outwards_results.append(anchor_iso)
        current_iso = anchor_iso
        current_sma = float(anchor_iso["sma"])  # type: ignore[arg-type]
        while True:
            next_sma = current_sma + astep if linear_growth else current_sma * (1.0 + astep)
            if next_sma > maxsma:
                break
            current_sma = next_sma
            current_geom = {
                "x0": float(current_iso["x0"]),  # type: ignore[arg-type]
                "y0": float(current_iso["y0"]),  # type: ignore[arg-type]
                "eps": float(current_iso["eps"]),  # type: ignore[arg-type]
                "pa": float(current_iso["pa"]),  # type: ignore[arg-type]
            }
            next_iso = fit_isophote_mb(
                images, masks, current_sma, current_geom, active_cfg,
                previous_geometry=current_geom,
                variance_maps=variance_maps,
                image_stack=image_stack, masks_resolved=masks_resolved, var_stack=var_stack,
                outer_reference_geom=outer_ref_geom,
            )
            if config.lsb_auto_lock:
                _mark_lsb_lock_state_mb(next_iso, locked=bool(lsb_state["locked"]))
            outwards_results.append(next_iso)
            if _is_acceptable(next_iso) or active_cfg.permissive_geometry:
                current_iso = next_iso

            # LSB auto-lock detector: free-mode only. Once locked, stays
            # locked for the rest of outward growth (one-way state
            # machine). Trigger surface is the joint combined gradient
            # (plan section 7 S3) via _is_lsb_isophote_mb.
            if config.lsb_auto_lock and not lsb_state["locked"]:
                triggered = _is_lsb_isophote_mb(
                    next_iso, config.lsb_auto_lock_maxgerr,
                )
                if triggered:
                    lsb_state["consec"] = int(lsb_state["consec"]) + 1
                    if int(lsb_state["consec"]) >= config.lsb_auto_lock_debounce:
                        # Anchor = isophote immediately BEFORE the streak,
                        # NOT the trigger isophote itself. The trigger
                        # isophotes are already partially in LSB and their
                        # geometries may have begun drifting.
                        anchor_local_idx = (
                            len(outwards_results) - 1 - config.lsb_auto_lock_debounce
                        )
                        anchor_local_idx = max(0, anchor_local_idx)
                        lock_anchor = outwards_results[anchor_local_idx]
                        active_cfg = _build_locked_cfg_mb(
                            config, lock_anchor, config.lsb_auto_lock_integrator,
                        )
                        lsb_state["locked"] = True
                        lsb_state["transition_sma"] = float(next_iso["sma"])  # type: ignore[arg-type]
                        lsb_state["anchor_index"] = anchor_local_idx
                        _mark_lsb_lock_state_mb(next_iso, locked=True, is_anchor=True)
                        # Restart the geometry carry-forward from the
                        # clean anchor so the next isophote is seeded by
                        # it, not by the (degraded) trigger isophote.
                        current_iso = lock_anchor
                else:
                    lsb_state["consec"] = 0

    # Assemble final list (ascending SMA).
    if minsma <= 0.0:
        final_list: List[Dict[str, object]] = [central_result] + inwards_results[::-1] + outwards_results
    else:
        final_list = inwards_results[::-1] + outwards_results

    _validate_non_negative_error_fields(final_list)

    # Stage-3 Stage-D: per-band curve-of-growth photometry. Runs once
    # over the assembled isophote list (ascending sma) and stamps
    # per-band ``cog_<b>`` / ``cog_annulus_<b>`` plus shared
    # ``area_annulus`` / ``flag_cross`` / ``flag_negative_area`` onto
    # each row dict. Schema 1 round-trip handles the new columns
    # automatically via ``Table(rows=isophotes)`` auto-inference. See
    # plan section 7 S7.
    if config.compute_cog and final_list:
        from .cog_mb import add_cog_mb_to_isophotes, compute_cog_mb

        cog_results = compute_cog_mb(
            final_list,
            bands=list(config.bands),
            fix_center=config.fix_center,
            fix_geometry=config.fix_center and config.fix_pa and config.fix_eps,
        )
        add_cog_mb_to_isophotes(final_list, list(config.bands), cog_results)

    # Suppress the forced-photometry helper for unused imports — the
    # central pixel uses _fit_central_pixel_mb above. Kept available for
    # callers that want to do extra single-isophote forced extractions.
    _ = extract_forced_photometry_mb  # noqa: F841

    result: Dict[str, object] = {
        "isophotes": final_list,
        "config": config,
        "bands": list(config.bands),
        "multiband": True,
        "harmonic_combination": config.harmonic_combination,
        "reference_band": config.reference_band,
        "band_weights": config.resolved_band_weights(),
        "variance_mode": variance_mode,
        "fit_per_band_intens_jointly": config.fit_per_band_intens_jointly,
        "loose_validity": config.loose_validity,
        # Section 6: higher-order harmonics mode + orders. ``harmonics_shared``
        # is a derived convenience flag (True iff a non-``independent`` mode
        # is active) so downstream readers can short-circuit without parsing
        # the enum.
        "multiband_higher_harmonics": config.multiband_higher_harmonics,
        "harmonic_orders": list(config.harmonic_orders),
        "harmonics_shared": config.multiband_higher_harmonics != "independent",
    }
    if first_isophote_failure:
        result["first_isophote_failure"] = True
    if retry_log:
        result["first_isophote_retry_log"] = retry_log
    # Stage-3 Stage-C: surface lsb_auto_lock metadata when enabled.
    if config.lsb_auto_lock:
        result["lsb_auto_lock"] = True
        result["lsb_auto_lock_sma"] = lsb_state["transition_sma"]
        result["lsb_auto_lock_count"] = sum(
            1 for iso in final_list if bool(iso.get("lsb_locked", False))
        )
    return result
