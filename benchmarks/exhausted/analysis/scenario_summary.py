"""Shared primitives for per-campaign scenario-level summaries.

Every exhausted-benchmark campaign writes a regular tree of per-arm
``profile.fits`` / ``inventory.fits`` / ``run_record.json`` and a per-
galaxy ``MANIFEST.json``. This module exposes cheap loaders and a
single metrics entry point that downstream analysis drivers compose.

Phase F.0 is the first consumer: it audits the ``clean_z005`` scenario
against the priors implied by the axisymmetric multi-Sersic mocks. See
``clean_z005_audit.py`` for the driver.

The three radial zones (``inner`` / ``mid`` / ``outer``) are taken from
``residual_zones.py`` so the same bins used for image residual RMS are
reused for 1-D isophote metrics. Because every pixel on an isophote
satisfies ``r_ell == sma`` (by construction), the zone thresholds in
``sma`` space match the 2-D pixel thresholds exactly.

Design goals:

- Picklable, I/O only helpers (so a multiprocess driver can fan out
  across galaxies later without adapter-class pickling concerns).
- Pure-function metrics that accept an ``astropy.Table`` profile plus
  a primitive manifest dict, so the same code can score isoster,
  photutils, or autoprof profiles without tool-specific branches.
- Zone masks are computed once per profile and passed to every metric
  so callers never recompute them and drift.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.table import Table


# ---------------------------------------------------------------------------
# Manifest + profile loaders
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GalaxyManifest:
    """Flat view of ``<galaxy_dir>/MANIFEST.json``.

    ``re_outer_pix`` is the flux-independent upper envelope of the
    per-component effective radii (max over ``RE_PX{i}``); retained as
    a diagnostic even though the current prior-2/3 zone definitions
    are driven by ``effective_Re_pix`` through ``residual_zones``.
    """

    galaxy_id: str
    galaxy_name: str
    scenario: str
    pixel_scale_arcsec: float
    sb_zeropoint: float
    psf_fwhm_arcsec: float | None
    psf_fwhm_pix: float | None
    image_shape: tuple[int, int]
    image_sigma_adu: float
    initial_x0: float
    initial_y0: float
    effective_Re_pix: float | None
    truth_components: tuple[dict[str, float], ...]
    re_outer_pix: float | None

    @property
    def half_extent(self) -> int:
        return min(self.image_shape) // 2

    @property
    def true_center(self) -> tuple[float, float]:
        """Mocks share a single center at (initial_x0, initial_y0)."""
        return (self.initial_x0, self.initial_y0)


def load_galaxy_manifest(galaxy_dir: Path) -> GalaxyManifest:
    """Read ``<galaxy_dir>/MANIFEST.json`` into a :class:`GalaxyManifest`."""
    with (Path(galaxy_dir) / "MANIFEST.json").open() as handle:
        data = json.load(handle)
    pix_scale = float(data["pixel_scale_arcsec"])
    extra = data.get("extra", {}) or {}
    psf_arcsec = extra.get("psf_fwhm_arcsec")
    psf_pix = float(psf_arcsec) / pix_scale if psf_arcsec else None
    truth = tuple(extra.get("truth_components", []) or [])
    re_pxs = [float(c["re_px"]) for c in truth if c.get("re_px") is not None]
    re_outer = max(re_pxs) if re_pxs else None
    shape = tuple(data["image_shape"])
    if len(shape) != 2:
        raise ValueError(f"{galaxy_dir}: MANIFEST image_shape must be 2D")
    return GalaxyManifest(
        galaxy_id=data["galaxy_id"],
        galaxy_name=extra.get("galaxy_name", ""),
        scenario=extra.get("scenario", ""),
        pixel_scale_arcsec=pix_scale,
        sb_zeropoint=float(data["sb_zeropoint"]),
        psf_fwhm_arcsec=float(psf_arcsec) if psf_arcsec else None,
        psf_fwhm_pix=psf_pix,
        image_shape=shape,  # type: ignore[arg-type]
        image_sigma_adu=float(data["image_sigma"]["image_sigma_adu"]),
        initial_x0=float(data["initial_geometry"]["x0"]),
        initial_y0=float(data["initial_geometry"]["y0"]),
        effective_Re_pix=(
            float(data["effective_Re_pix"])
            if data.get("effective_Re_pix") is not None
            else None
        ),
        truth_components=truth,
        re_outer_pix=re_outer,
    )


def load_arm_profile(galaxy_dir: Path, tool: str, arm_id: str) -> Table | None:
    """Load the ``ISOPHOTES`` HDU of one arm's ``profile.fits``."""
    path = Path(galaxy_dir) / tool / "arms" / arm_id / "profile.fits"
    if not path.is_file():
        return None
    with fits.open(path) as hdul:
        return Table(hdul[1].data)  # type: ignore[index]


def load_arm_record(galaxy_dir: Path, tool: str, arm_id: str) -> dict[str, Any] | None:
    path = Path(galaxy_dir) / tool / "arms" / arm_id / "run_record.json"
    if not path.is_file():
        return None
    with path.open() as handle:
        return json.load(handle)


def list_campaign_galaxies(campaign_dir: Path, dataset: str) -> list[Path]:
    """List ``<campaign_dir>/<dataset>/<safe_galaxy_id>/`` directories.

    Safe ids carry the double-underscore substitution of ``/`` (see
    ``adapters.base.safe_galaxy_id``). Returns a sorted list of Path.
    """
    ds_dir = Path(campaign_dir) / dataset
    if not ds_dir.is_dir():
        raise FileNotFoundError(f"dataset dir missing: {ds_dir}")
    return sorted(p for p in ds_dir.iterdir() if p.is_dir() and "__" in p.name)


def list_arms(galaxy_dir: Path, tool: str) -> list[str]:
    ap = Path(galaxy_dir) / tool / "arms"
    if not ap.is_dir():
        return []
    return sorted(p.name for p in ap.iterdir() if p.is_dir())


# ---------------------------------------------------------------------------
# Zone helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SmaZones:
    """Inner / mid / outer sma-masks plus the reference length used.

    Matches :mod:`residual_zones` exactly: ``inner`` is ``r_ell <
    0.5 * R_ref``, ``mid`` is ``[0.5, 2) * R_ref``, ``outer`` is
    ``>= 2 * R_ref``.  Because ``r_ell == sma`` on an isophote the
    same thresholds apply here.
    """

    R_ref_pix: float
    fallback: str
    inner: np.ndarray
    mid: np.ndarray
    outer: np.ndarray

    @property
    def inner_end(self) -> float:
        return 0.5 * self.R_ref_pix

    @property
    def mid_end(self) -> float:
        return 2.0 * self.R_ref_pix


def _reference_length(manifest: GalaxyManifest, sma: np.ndarray) -> tuple[float, str]:
    """Pick ``R_ref`` per residual_zones convention.

    Uses ``effective_Re_pix`` when finite and positive; otherwise falls
    back to ``max(1.0, 0.25 * maxsma)`` where ``maxsma`` is the largest
    observed sma in the profile (the per-arm top of the ladder).
    """
    re_px = manifest.effective_Re_pix
    if re_px is not None and np.isfinite(re_px) and re_px > 0:
        return float(re_px), ""
    maxsma = float(np.nanmax(sma)) if sma.size else float(manifest.half_extent)
    if not np.isfinite(maxsma) or maxsma <= 0:
        maxsma = float(manifest.half_extent)
    return max(1.0, 0.25 * maxsma), "Re_missing"


def sma_zones(
    profile: Table,
    manifest: GalaxyManifest,
    *,
    ok_mask: np.ndarray | None = None,
) -> SmaZones:
    """Build inner / mid / outer boolean masks aligned to the profile rows.

    Masks are AND'ed with ``ok_mask`` (defaulting to ``stop_code == 0``)
    so metrics iterate only over valid isophotes.
    """
    sma = np.asarray(profile["sma"], dtype=float)
    R_ref, fallback = _reference_length(manifest, sma)
    if ok_mask is None:
        ok = np.asarray(np.asarray(profile["stop_code"]) == 0, dtype=bool)
    else:
        ok = np.asarray(ok_mask, dtype=bool)
    inner = ok & (sma < 0.5 * R_ref)
    outer = ok & (sma >= 2.0 * R_ref)
    mid = ok & ~(sma < 0.5 * R_ref) & ~(sma >= 2.0 * R_ref)
    return SmaZones(
        R_ref_pix=R_ref,
        fallback=fallback,
        inner=inner,
        mid=mid,
        outer=outer,
    )


# ---------------------------------------------------------------------------
# Harmonic normalization + local-gradient helpers
# ---------------------------------------------------------------------------


def _col_or_nan(profile: Table, name: str) -> np.ndarray:
    """Return ``profile[name]`` as float64, or an all-NaN array if absent."""
    if name in profile.colnames:
        return np.asarray(profile[name], dtype=float)
    return np.full(len(profile), np.nan, dtype=float)


def profile_local_grad(sma: np.ndarray, intens: np.ndarray) -> np.ndarray:
    """Finite-difference ``dI/da`` from a 1-D profile.

    Used as a fallback when a fitter's ``grad`` column is missing or
    entirely NaN. ``np.gradient`` handles non-uniform ``sma`` spacing.
    """
    sma = np.asarray(sma, dtype=float)
    intens = np.asarray(intens, dtype=float)
    if sma.size < 2:
        return np.full_like(sma, np.nan, dtype=float)
    good = np.isfinite(sma) & np.isfinite(intens)
    if good.sum() < 2:
        return np.full_like(sma, np.nan, dtype=float)
    out = np.full_like(sma, np.nan, dtype=float)
    order = np.argsort(sma[good])
    s = sma[good][order]
    v = intens[good][order]
    # Guard strict monotonicity
    uniq = np.concatenate(([True], np.diff(s) > 0))
    s = s[uniq]
    v = v[uniq]
    if s.size < 2:
        return out
    g = np.gradient(v, s)
    # Map back to original index space
    idx = np.where(good)[0][order][uniq]
    out[idx] = g
    return out


def effective_grad(
    profile: Table,
    *,
    prefer_column: str = "grad",
) -> np.ndarray:
    """Return the intensity gradient with on-the-fly fallback.

    Uses the profile's ``grad`` column where it is finite; falls back
    to :func:`profile_local_grad` on rows where the column is missing
    or NaN. This is the gradient consumed by the harmonic-normalization
    step.
    """
    sma = _col_or_nan(profile, "sma")
    intens = _col_or_nan(profile, "intens")
    grad = _col_or_nan(profile, prefer_column)
    fallback = profile_local_grad(sma, intens)
    out = np.where(np.isfinite(grad), grad, fallback)
    return out


def normalize_harmonic(
    harm: np.ndarray,
    sma: np.ndarray,
    grad: np.ndarray,
    *,
    min_abs_agrad_frac: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(-harm / (sma * grad), valid_mask)`` per Bender convention.

    Points where ``|sma * grad|`` is below ``min_abs_agrad_frac`` of its
    zone-wise peak (or where the quantities are non-finite) are masked
    to NaN to avoid blowing up near profile turnovers.
    """
    sma = np.asarray(sma, dtype=float)
    grad = np.asarray(grad, dtype=float)
    harm = np.asarray(harm, dtype=float)
    denom = sma * grad
    abs_denom = np.abs(denom)
    finite = np.isfinite(denom) & np.isfinite(harm) & (abs_denom > 0)
    if not finite.any():
        return np.full_like(harm, np.nan), finite
    peak = float(np.nanmax(abs_denom[finite])) if finite.any() else 0.0
    floor = max(peak * min_abs_agrad_frac, np.finfo(float).tiny)
    valid = finite & (abs_denom > floor)
    out = np.full_like(harm, np.nan, dtype=float)
    out[valid] = -harm[valid] / denom[valid]
    return out, valid


# ---------------------------------------------------------------------------
# Intensity-weighted centroid drift
# ---------------------------------------------------------------------------


def zone_iw_centroid_drift(
    x0: np.ndarray,
    y0: np.ndarray,
    intens: np.ndarray,
    mask: np.ndarray,
    true_x: float,
    true_y: float,
) -> float:
    """Intensity-weighted |(xbar, ybar) - (true_x, true_y)| within ``mask``."""
    if mask.sum() == 0:
        return float("nan")
    w = np.asarray(intens, dtype=float)[mask]
    xv = np.asarray(x0, dtype=float)[mask]
    yv = np.asarray(y0, dtype=float)[mask]
    good = np.isfinite(w) & (w > 0) & np.isfinite(xv) & np.isfinite(yv)
    if not good.any():
        return float("nan")
    w = w[good]
    xv = xv[good]
    yv = yv[good]
    wsum = float(np.sum(w))
    if wsum <= 0:
        return float("nan")
    xbar = float(np.sum(xv * w) / wsum)
    ybar = float(np.sum(yv * w) / wsum)
    return float(np.hypot(xbar - true_x, ybar - true_y))


# ---------------------------------------------------------------------------
# Local three-point residual (prior 3)
# ---------------------------------------------------------------------------


def _wrap_pa_diff(delta: np.ndarray) -> np.ndarray:
    """Wrap a PA difference onto ``(-pi/2, pi/2]`` (PA is mod pi)."""
    return (delta + np.pi / 2.0) % np.pi - np.pi / 2.0


def local_triplet_residual(
    x: np.ndarray,
    y: np.ndarray,
    *,
    circular_pi: bool = False,
) -> np.ndarray:
    """Deviation of ``y[i]`` from the linear interpolation of its neighbors.

    For ``i`` in ``1 .. n-2`` with monotone ``x``:
        ``t = (x[i] - x[i-1]) / (x[i+1] - x[i-1])``
        ``resid[i] = y[i] - (y[i-1] + t * (y[i+1] - y[i-1]))``

    The endpoint rows and any with ``x[i+1] == x[i-1]`` are NaN.
    When ``circular_pi`` is set, the residual is wrapped onto
    ``(-pi/2, pi/2]`` so a 90 deg flip produces a small residual only
    if both neighbors already sit across the branch cut.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    out = np.full(n, np.nan, dtype=float)
    if n < 3:
        return out
    for i in range(1, n - 1):
        dx = x[i + 1] - x[i - 1]
        if not np.isfinite(dx) or dx == 0:
            continue
        t = (x[i] - x[i - 1]) / dx
        if circular_pi:
            pa0 = y[i - 1]
            d1 = _wrap_pa_diff(np.array([y[i] - pa0]))[0]
            d2 = _wrap_pa_diff(np.array([y[i + 1] - pa0]))[0]
            resid = _wrap_pa_diff(np.array([d1 - t * d2]))[0]
        else:
            y_interp = y[i - 1] + t * (y[i + 1] - y[i - 1])
            resid = y[i] - y_interp
        out[i] = resid
    return out


# ---------------------------------------------------------------------------
# Prior-violation metrics
# ---------------------------------------------------------------------------


def _err_ratio(amp: np.ndarray, err: np.ndarray, mask: np.ndarray) -> float:
    a = amp[mask]
    e = err[mask]
    if a.size == 0:
        return float("nan")
    good = np.isfinite(a) & np.isfinite(e) & (e > 0)
    if not np.any(good):
        return float("nan")
    return float(np.median(np.abs(a[good]) / e[good]))


def _nanmax_abs(arr: np.ndarray, mask: np.ndarray) -> float:
    v = arr[mask]
    v = v[np.isfinite(v)]
    return float(np.max(np.abs(v))) if v.size else float("nan")


def compute_prior_metrics(
    profile: Table,
    manifest: GalaxyManifest,
) -> dict[str, Any]:
    """Per-arm F.0 prior-violation metrics (overhauled, three priors).

    Prior 1 — intensity-weighted centroid drift per zone (inner / mid /
              outer).
    Prior 2 — normalized harmonic ``|A_n_norm| / err(A_n_norm)`` in the
              outer zone.
    Prior 3 — 3-point local residual of ``eps`` and ``pa`` in the outer
              zone.

    All returned values are plain floats / ints (NaN-safe) so the
    result dict can be added directly to a long-form DataFrame row.
    """
    out: dict[str, Any] = {}
    if profile is None or len(profile) == 0:
        return {"n_iso_total": 0, "n_iso_ok": 0}

    sma = _col_or_nan(profile, "sma")
    ok = np.asarray(profile["stop_code"]) == 0
    x0 = _col_or_nan(profile, "x0")
    y0 = _col_or_nan(profile, "y0")
    eps = _col_or_nan(profile, "eps")
    pa = _col_or_nan(profile, "pa")
    intens = _col_or_nan(profile, "intens")

    zones = sma_zones(profile, manifest, ok_mask=ok)

    true_x, true_y = manifest.true_center
    drift_per_iso = np.hypot(x0 - true_x, y0 - true_y)

    # ---- Prior 1: intensity-weighted centroid drift per zone --------
    drift_inner = zone_iw_centroid_drift(x0, y0, intens, zones.inner, true_x, true_y)
    drift_mid = zone_iw_centroid_drift(x0, y0, intens, zones.mid, true_x, true_y)
    drift_outer = zone_iw_centroid_drift(x0, y0, intens, zones.outer, true_x, true_y)

    # ---- Prior 2: normalized harmonics in outer zone ----------------
    grad = effective_grad(profile)
    harm_fields = ("a3", "b3", "a4", "b4")
    harm_norm: dict[str, np.ndarray] = {}
    harm_norm_valid: dict[str, np.ndarray] = {}
    harm_norm_err: dict[str, np.ndarray] = {}
    for name in harm_fields:
        raw = _col_or_nan(profile, name)
        err = _col_or_nan(profile, f"{name}_err")
        norm_val, valid = normalize_harmonic(raw, sma, grad)
        # Normalized error propagates with the same 1 / |a*grad| factor.
        denom_abs = np.abs(sma * grad)
        with np.errstate(divide="ignore", invalid="ignore"):
            norm_err = np.where(valid & (denom_abs > 0), err / denom_abs, np.nan)
        harm_norm[name] = norm_val
        harm_norm_valid[name] = valid
        harm_norm_err[name] = norm_err

    def _harm_stat(name: str) -> dict[str, float]:
        valid = harm_norm_valid[name] & zones.outer
        ratio = _err_ratio(
            np.abs(harm_norm[name]), harm_norm_err[name], valid
        )
        vals = harm_norm[name][valid]
        vals = vals[np.isfinite(vals)]
        abs_median = float(np.median(np.abs(vals))) if vals.size else float("nan")
        return {"err_ratio": ratio, "abs_median": abs_median, "n": int(valid.sum())}

    harm_stats = {name: _harm_stat(name) for name in harm_fields}

    # Prior 2 applicability: when both low-order harmonics (A3 and A4)
    # have zero valid outer-zone points, the arm did not fit harmonics
    # there (e.g. `harm_disabled`, or sweeps that replace the default
    # [3, 4] orders). Mark Prior 2 N/A so the arm is neither credited
    # with a clean pass nor flagged as violating.
    prior2_applicable = bool(
        (harm_stats["a3"]["n"] > 0) or (harm_stats["a4"]["n"] > 0)
    )

    # ---- Prior 3: 3-point local residual in outer zone --------------
    # Use only outer-zone rows so the triplet is built over contiguous
    # outer isophotes. The per-row residual is only defined where both
    # neighbors are also in the mask.
    outer_idx = np.where(zones.outer)[0]
    if outer_idx.size >= 3:
        s_outer = sma[outer_idx]
        eps_resid = local_triplet_residual(s_outer, eps[outer_idx])
        pa_resid = local_triplet_residual(s_outer, pa[outer_idx], circular_pi=True)
        max_eps_res = float(np.nanmax(np.abs(eps_resid))) if np.any(np.isfinite(eps_resid)) else float("nan")
        max_pa_res_rad = float(np.nanmax(np.abs(pa_resid))) if np.any(np.isfinite(pa_resid)) else float("nan")
        max_pa_res_deg = float(np.degrees(max_pa_res_rad)) if np.isfinite(max_pa_res_rad) else float("nan")
        n_triplets = int(np.sum(np.isfinite(eps_resid)))
    else:
        max_eps_res = float("nan")
        max_pa_res_deg = float("nan")
        n_triplets = 0

    out.update(
        {
            "n_iso_total": int(len(profile)),
            "n_iso_ok": int(ok.sum()),
            "R_ref_pix": zones.R_ref_pix,
            "zone_fallback_ref": zones.fallback,
            "sma_zone_inner_end": zones.inner_end,
            "sma_zone_mid_end": zones.mid_end,
            "n_iso_inner": int(zones.inner.sum()),
            "n_iso_mid": int(zones.mid.sum()),
            "n_iso_outer": int(zones.outer.sum()),
            # Prior 1 (primary) — intensity-weighted drift per zone
            "drift_iw_inner_pix": drift_inner,
            "drift_iw_mid_pix": drift_mid,
            "drift_iw_outer_pix": drift_outer,
            # Prior 1 diagnostics (not scored)
            "max_drift_pix_all_ok": _nanmax_abs(drift_per_iso, ok),
            "max_drift_pix_outer": _nanmax_abs(drift_per_iso, zones.outer),
            # Prior 2 — normalized harmonics, outer zone only
            "a3n_err_ratio_outer": harm_stats["a3"]["err_ratio"],
            "b3n_err_ratio_outer": harm_stats["b3"]["err_ratio"],
            "a4n_err_ratio_outer": harm_stats["a4"]["err_ratio"],
            "b4n_err_ratio_outer": harm_stats["b4"]["err_ratio"],
            "abs_a3n_median_outer": harm_stats["a3"]["abs_median"],
            "abs_b3n_median_outer": harm_stats["b3"]["abs_median"],
            "abs_a4n_median_outer": harm_stats["a4"]["abs_median"],
            "abs_b4n_median_outer": harm_stats["b4"]["abs_median"],
            "n_harm_outer": harm_stats["a4"]["n"],
            "prior2_applicable": prior2_applicable,
            # Prior 3 — 3-point local residual, outer zone only
            "max_local_resid_eps_outer": max_eps_res,
            "max_local_resid_pa_outer_deg": max_pa_res_deg,
            "n_triplets_outer": n_triplets,
        }
    )
    return out


# ---------------------------------------------------------------------------
# Default violation thresholds
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: dict[str, float] = {
    # Prior 1 — intensity-weighted centroid, per zone.
    "drift_iw_inner_pix": 0.5,
    "drift_iw_mid_pix": 0.5,
    "drift_iw_outer_pix": 0.5,  # calibrated on F.a: ref_default p95 hits ~1.15 px at wide_z050; 0.5 is a defensible "drift is visible" bar
    # Prior 2 — outer-zone normalized harmonics vs formal errors.
    # A violation requires BOTH the err-ratio threshold AND the absolute
    # |A_n_norm| floor to fire, so tiny |A4n| with optimistic formal
    # errors no longer triggers a false positive.
    "a3n_err_ratio_outer": 3.0,
    "b3n_err_ratio_outer": 3.0,
    "a4n_err_ratio_outer": 3.0,
    "b4n_err_ratio_outer": 3.0,
    "abs_a3n_median_outer": 0.01,
    "abs_b3n_median_outer": 0.01,
    "abs_a4n_median_outer": 0.01,
    "abs_b4n_median_outer": 0.01,
    # Prior 3 — 3-point local residual beyond 2 * R_ref.
    "max_local_resid_eps_outer": 0.05,
    "max_local_resid_pa_outer_deg": 8.0,
}


def classify_violations(
    metrics: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Map metric values onto ``violates_<prior>`` flags.

    Returns booleans for Priors 1 and 3. ``violates_harmonics_outer``
    is ``None`` when Prior 2 does not apply (``prior2_applicable ==
    False``, e.g. an arm that fit no A3/A4 harmonics). Treating N/A
    as ``None`` lets downstream drivers skip that arm in the Prior 2
    clean-fraction numerator and denominator without crediting a
    free pass. Missing finite values (NaN) still count as
    non-violation.
    """
    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        th.update(thresholds)

    def _gt(key: str) -> bool:
        val = metrics.get(key)
        if val is None:
            return False
        try:
            v = float(val)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(v):
            return False
        return v > th[key]

    prior2_applicable = bool(metrics.get("prior2_applicable", True))
    harmonics_viol: bool | None
    if prior2_applicable:
        harmonics_viol = any(
            _gt(f"{name}_err_ratio_outer")
            and _gt(f"abs_{name}_median_outer")
            for name in ("a3n", "b3n", "a4n", "b4n")
        )
    else:
        harmonics_viol = None

    return {
        "violates_drift_inner": _gt("drift_iw_inner_pix"),
        "violates_drift_mid": _gt("drift_iw_mid_pix"),
        "violates_drift_outer": _gt("drift_iw_outer_pix"),
        "violates_drift_any": (
            _gt("drift_iw_inner_pix")
            or _gt("drift_iw_mid_pix")
            or _gt("drift_iw_outer_pix")
        ),
        "violates_harmonics_outer": harmonics_viol,
        "violates_geometry_outer": (
            _gt("max_local_resid_eps_outer")
            or _gt("max_local_resid_pa_outer_deg")
        ),
    }
