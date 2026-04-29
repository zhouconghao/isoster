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
    if isinstance(variance_maps, list) and len(variance_maps) != len(config.bands):
        raise ValueError(
            f"len(variance_maps) ({len(variance_maps)}) does not match "
            f"len(config.bands) ({len(config.bands)})."
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
        for n_order in (3, 4):
            row[f"a{n_order}_{b}"] = 0.0
            row[f"b{n_order}_{b}"] = 0.0
            row[f"a{n_order}_err_{b}"] = 0.0
            row[f"b{n_order}_err_{b}"] = 0.0
        if debug:
            row[f"grad_{b}"] = float("nan")
            row[f"grad_error_{b}"] = float("nan")
            row[f"grad_r_error_{b}"] = float("nan")
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

    # Outward growth.
    outwards_results: List[Dict[str, object]] = []
    if anchor_iso is not None:
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
                images, masks, current_sma, current_geom, config,
                previous_geometry=current_geom,
                variance_maps=variance_maps,
                image_stack=image_stack, masks_resolved=masks_resolved, var_stack=var_stack,
            )
            outwards_results.append(next_iso)
            if _is_acceptable(next_iso) or config.permissive_geometry:
                current_iso = next_iso

    # Assemble final list (ascending SMA).
    if minsma <= 0.0:
        final_list: List[Dict[str, object]] = [central_result] + inwards_results[::-1] + outwards_results
    else:
        final_list = inwards_results[::-1] + outwards_results

    _validate_non_negative_error_fields(final_list)

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
    }
    if first_isophote_failure:
        result["first_isophote_failure"] = True
    if retry_log:
        result["first_isophote_retry_log"] = retry_log
    return result
