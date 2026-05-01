"""
Multi-band isoster configuration (``IsosterConfigMB``).

Sibling of :class:`isoster.config.IsosterConfig` with a deliberately
reduced field set, plus multi-band-only fields. There is **no
inheritance** between the two classes: changes to ``IsosterConfig`` do
not propagate automatically. This is intentional — the multi-band path
is experimental and we want it free to evolve independently of the
stable single-band path. See
``docs/agent/plan-2026-04-29-multiband-feasibility.md`` (decision D23)
for the rationale.

Excluded fields (deliberately not copied; Stage 2+ may add multi-band
variants):

- ``harmonic_orders`` (locked to ``[3, 4]`` in Stage 1)
- ``simultaneous_harmonics`` and ``isofit_mode`` (ISOFIT)
- All ``lsb_auto_lock_*`` fields
- All ``outer_reg_*`` and ``use_outer_center_regularization`` fields
- ``compute_cog`` (multi-band CoG attachment is out of Stage-1 scope)
- ``central_reg_*`` (single-band-only LSB knobs)
- ``lsb_sma_threshold`` and the ``adaptive`` integrator option
"""

import re
import warnings
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


# Band-name regex: must start with a letter, then letters/digits/underscores
# only. Hyphens and other punctuation are rejected so that band strings
# can be used verbatim as FITS column suffixes (e.g. ``intens_g``,
# ``intens_HSC_G``). See decision D8.
_BAND_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class IsosterConfigMB(BaseModel):
    """
    Configuration for multi-band isoster (Stage-1, experimental).

    The fitter consumes multiple aligned same-pixel-grid images of the
    same target, fits a single shared geometry per SMA, and reports
    per-band intensities and per-band harmonic deviations. See
    ``docs/10-multiband.md`` for the full user-facing reference.

    **Stop codes** are inherited from single-band isoster (see
    :class:`isoster.config.IsosterConfig`).
    """

    # --- Multi-band-only fields ---

    bands: List[str] = Field(
        ...,
        min_length=1,
        description="Ordered list of band names. Must match the input images "
        "list element-for-element. Each name must match the regex "
        "``^[A-Za-z][A-Za-z0-9_]*$`` (letter then letters/digits/underscores). "
        "No duplicates. The strings appear verbatim as FITS column suffixes, "
        "e.g. ``bands=['g', 'r']`` produces ``intens_g`` and ``intens_r``.",
    )
    reference_band: str = Field(
        ...,
        description="Name of the band used for diagnostic and driver scalar "
        "decisions: ``sma0`` initial-guess fallback, first-isophote-retry "
        "intensity interpretation, single-scalar gradient/SNR for QA "
        "plotting. Must be one of ``bands``. Does not affect joint geometry "
        "or per-band intensity columns.",
    )
    band_weights: Optional[Union[Dict[str, float], List[float]]] = Field(
        default=None,
        description="Per-band scalar weights ``w_b``, applied multiplicatively "
        "to each band's row block in the joint design matrix as ``sqrt(w_b)``. "
        "Either a dict keyed by band name (every band must have a key) or a "
        "list aligned with ``bands`` (length ``B``). Each weight must be "
        "positive and finite. ``None`` (default) means uniform weights "
        "``w_b = 1.0``. Composes with per-pixel inverse-variance weights in "
        "WLS mode as ``w_b / variance_b(pixel)``.",
    )
    harmonic_combination: Literal["joint", "ref"] = Field(
        default="joint",
        description="Strategy for combining per-band harmonic information into "
        "the geometry update. ``'joint'`` (default): solve a single "
        "``(5 + B)``-column design matrix once per iteration, with per-band "
        "``I0_b`` nuisance parameters and shared ``(A1, B1, A2, B2)`` "
        "geometric coefficients. ``'ref'``: drive geometry from the reference "
        "band only; other bands are passive (post-hoc intensity and harmonic "
        "extraction). Use ``'ref'`` for debugging or when the joint solver "
        "is suspected of misbehaving.",
    )
    fix_per_band_background_to_zero: bool = Field(
        default=False,
        description="When True, drop the leading ``B`` per-band intercept "
        "columns from the joint design matrix. The solve becomes a "
        "4-column ``(A1, B1, A2, B2)`` system shared across all bands; "
        "per-band ``intens_<b>`` is then reported as the band's own "
        "ring-mean intensity along the fitted ellipse rather than as a "
        "free nuisance parameter. Use this when the input images have "
        "perfectly subtracted sky and the per-band ``I0_b`` plateau "
        "visible at the LSB transition is being driven by sky residual "
        "rather than real galaxy flux. Mutually exclusive with the "
        "``'ref'`` ``harmonic_combination``: ref-mode already side-steps "
        "the joint solve. Decision D11 (Stage-2 backport).",
    )

    # --- D9 backport: per-band loose validity ---
    loose_validity: bool = Field(
        default=False,
        description="Relax the cross-band shared-validity AND. With "
        "``False`` (default), a sample is dropped from every band if any "
        "band fails (mask, NaN, non-positive variance) at that location. "
        "With ``True``, each band keeps its own surviving samples; the "
        "joint design matrix becomes block-diagonal in the per-band "
        "intercept columns and uses each band's own kept angles. A band "
        "that falls below the per-band thresholds at a given isophote is "
        "dropped from the joint solve at that isophote (its ``intens_<b>`` "
        "is reported as NaN); the surviving bands still constrain the "
        "shared geometry. Whole-isophote ``stop_code=3`` only fires when "
        "fewer than 2 bands survive. Decision D9 backport (locked "
        "2026-05-01).",
    )
    loose_validity_min_per_band_count: int = Field(
        default=6,
        ge=1,
        description="Per-band absolute minimum surviving-sample count "
        "under loose validity. A band with fewer than this many kept "
        "samples at a given isophote is dropped from the joint solve at "
        "that isophote. Default 6 mirrors the single-band 5-parameter "
        "minimum + 1 sample of slack. Ignored when ``loose_validity=False``.",
    )
    loose_validity_min_per_band_frac: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Per-band minimum surviving-sample fraction under "
        "loose validity (kept / sampled). A band below this fraction at a "
        "given isophote is dropped from the joint solve at that isophote. "
        "Default 0.2. Ignored when ``loose_validity=False``.",
    )
    loose_validity_band_normalization: Literal["none", "per_band_count"] = Field(
        default="none",
        description="Per-band normalization of the joint design matrix "
        "and the combined-gradient combiner under loose validity. "
        "``'none'`` (default): each band's row block contributes "
        "proportionally to its own ``N_b``; ``w_b`` multiplies every row. "
        "Combined gradient = ``Σ w_b · grad_b / Σ w_b`` over surviving "
        "bands. ``'per_band_count'``: each band's design-matrix block is "
        "row-scaled by ``√(1/N_b)`` so its total contribution equals "
        "``w_b`` regardless of ``N_b``; combined gradient is weighted by "
        "``(w_b · N_b)``. Use ``'per_band_count'`` to preserve the "
        "user-specified ``band_weights`` semantics across mask-induced "
        "per-band sample-count differences. Requires ``loose_validity=True`` "
        "(rejected at construction otherwise — meaningless under shared "
        "validity since all ``N_b`` are identical).",
    )

    # --- Geometry initialization (copied from IsosterConfig) ---
    x0: Optional[float] = Field(None, description="Initial center x coordinate. If None, uses image center.")
    y0: Optional[float] = Field(None, description="Initial center y coordinate. If None, uses image center.")
    eps: float = Field(0.2, ge=0.0, lt=1.0, description="Initial ellipticity.")
    pa: float = Field(0.0, description="Initial position angle in radians.")

    # --- SMA grid (copied) ---
    sma0: float = Field(10.0, gt=0.0, description="Starting semi-major axis length.")
    minsma: float = Field(0.0, ge=0.0, description="Minimum SMA to fit.")
    maxsma: Optional[float] = Field(None, gt=0.0, description="Maximum SMA to fit. If None, uses image hypotenuse/2.")
    astep: float = Field(0.1, gt=0.0, description="Step size for SMA growth.")
    linear_growth: bool = Field(False, description="If True, sma = sma + astep. Else sma = sma * (1 + astep).")

    # --- Iteration / convergence (copied) ---
    maxit: int = Field(50, gt=0, description="Maximum iterations per isophote.")
    minit: int = Field(6, gt=0, description="Minimum iterations per isophote.")
    conver: float = Field(0.05, gt=0.0, description="Convergence threshold (max harmonic amplitude / rms).")
    convergence_scaling: Literal["none", "sector_area", "sqrt_sma"] = Field(
        default="sector_area",
        description="Scale convergence threshold with SMA. See "
        ":class:`isoster.config.IsosterConfig` for option meanings.",
    )
    sigma_bg: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Explicit background noise level (sigma) shared across bands. "
        "If provided, establishes a hard lower bound on the convergence "
        "threshold to prevent overfitting in LSB regions. Stage 1 uses a "
        "single shared sigma_bg; per-band sigma_bg is a Stage-2+ revisit.",
    )
    use_corrected_errors: bool = Field(
        default=True,
        description="Include the gradient uncertainty term in geometric error "
        "propagation. Same semantics as single-band.",
    )
    use_lazy_gradient: bool = Field(
        default=True,
        description="Use lazy gradient evaluation (gradient cached across "
        "iterations unless convergence stalls). Same semantics as single-band.",
    )

    # --- Geometry update control (copied) ---
    geometry_damping: float = Field(0.7, gt=0.0, le=1.0, description="Damping factor for geometry updates.")
    clip_max_shift: Optional[float] = Field(5.0, gt=0.0, description="Max center shift per iteration (pixels).")
    clip_max_pa: Optional[float] = Field(0.5, gt=0.0, description="Max PA change per iteration (radians).")
    clip_max_eps: Optional[float] = Field(0.1, gt=0.0, description="Max ellipticity change per iteration.")
    geometry_convergence: bool = Field(
        default=False,
        description="Enable secondary convergence based on geometry stability.",
    )
    geometry_tolerance: float = Field(0.01, gt=0.0, description="Threshold for geometry convergence.")
    geometry_stable_iters: int = Field(3, ge=2, description="Consecutive stable iterations required.")
    geometry_update_mode: Literal["largest", "simultaneous"] = Field(
        default="largest",
        description="Geometry update strategy: ``'largest'`` (coordinate descent) "
        "or ``'simultaneous'`` (update all four geometry parameters per iteration).",
    )

    # --- Quality control (copied) ---
    sclip: float = Field(3.0, gt=0.0, description="Sigma clipping threshold.")
    nclip: int = Field(1, ge=0, description="Number of sigma clipping iterations.")
    sclip_low: Optional[float] = Field(None, description="Lower sigma clipping threshold.")
    sclip_high: Optional[float] = Field(None, description="Upper sigma clipping threshold.")
    fflag: float = Field(0.5, ge=0.0, le=1.0, description="Maximum fraction of flagged data.")
    maxgerr: float = Field(0.5, gt=0.0, description="Maximum relative error in local radial gradient.")

    # --- Geometry constraints (copied) ---
    fix_center: bool = Field(False, description="Fix center coordinates during fitting.")
    fix_pa: bool = Field(False, description="Fix position angle during fitting.")
    fix_eps: bool = Field(False, description="Fix ellipticity during fitting.")

    # --- Outputs / features (copied, with multi-band restrictions) ---
    compute_errors: bool = Field(True, description="Calculate parameter errors.")
    compute_deviations: bool = Field(True, description="Calculate higher-order harmonic deviations (a3, b3, a4, b4).")
    full_photometry: bool = Field(False, description="Calculate flux integration metrics (tflux_e, etc.).")
    debug: bool = Field(False, description="Include debug info in results (per-band grad columns, ndata, nflag).")

    # --- Integrator (restricted: no 'adaptive' since LSB auto-lock is out) ---
    integrator: Literal["mean", "median"] = Field(
        default="mean",
        description="Integration method for flux calculation. ``'adaptive'`` "
        "is intentionally not supported in Stage 1 because LSB auto-lock is "
        "out of scope; use ``'median'`` directly if median statistics are "
        "needed in the LSB regime.",
    )

    # --- Eccentric anomaly sampling (copied) ---
    use_eccentric_anomaly: bool = Field(
        default=False,
        description="Use eccentric anomaly for uniform ellipse sampling "
        "(recommended for ε > 0.3). Same semantics as single-band.",
    )

    # --- Permissive geometry mode (copied) ---
    permissive_geometry: bool = Field(
        default=False,
        description="Enable permissive geometry updates (photutils-style). "
        "Same semantics as single-band.",
    )

    # --- First-isophote robustness (copied) ---
    max_retry_first_isophote: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Maximum retry attempts for the first isophote. Same "
        "semantics as single-band.",
    )
    first_isophote_fail_count: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Consecutive initial-isophote failures before declaring "
        "FIRST_FEW_ISOPHOTE_FAILURE. Same semantics as single-band.",
    )

    # --- Read-only field surfaced after driver entry ---
    # ``variance_mode`` is auto-derived from whether ``variance_maps`` is
    # provided to ``fit_image_multiband``. It is exposed as a config field
    # so the resolved-config record (saved to the FITS CONFIG HDU)
    # captures it. Users do not set it directly; the driver assigns it
    # before serialization.
    variance_mode: Optional[Literal["wls", "ols"]] = Field(
        default=None,
        description="Auto-derived: 'wls' when variance maps are provided to "
        "the driver, 'ols' otherwise. Users should not set this directly.",
    )

    @model_validator(mode="after")
    def _check_multiband_consistency(self):
        """Validate multi-band-specific consistency rules."""
        # --- bands: regex + uniqueness (D8) ---
        for band in self.bands:
            if not _BAND_NAME_RE.match(band):
                raise ValueError(
                    f"band name {band!r} does not match the required regex "
                    f"'^[A-Za-z][A-Za-z0-9_]*$'. Band names must start with a "
                    f"letter and contain only letters, digits, and underscores. "
                    f"Hyphens and other punctuation are not allowed because "
                    f"band names appear verbatim as FITS column suffixes."
                )
        if len(set(self.bands)) != len(self.bands):
            seen = set()
            duplicates = [b for b in self.bands if b in seen or seen.add(b)]
            raise ValueError(
                f"bands contains duplicates: {duplicates}. Each band name "
                f"must be unique."
            )

        # --- reference_band must be in bands (D3) ---
        if self.reference_band not in self.bands:
            raise ValueError(
                f"reference_band {self.reference_band!r} must be one of bands "
                f"{self.bands!r}."
            )

        # --- band_weights validation (D12) ---
        if self.band_weights is not None:
            if isinstance(self.band_weights, dict):
                missing = set(self.bands) - set(self.band_weights.keys())
                extra = set(self.band_weights.keys()) - set(self.bands)
                if missing:
                    raise ValueError(
                        f"band_weights is missing keys for bands "
                        f"{sorted(missing)}. When band_weights is a dict, "
                        f"every band in `bands` must have a corresponding key."
                    )
                if extra:
                    raise ValueError(
                        f"band_weights contains keys {sorted(extra)} that are "
                        f"not in `bands` {self.bands!r}."
                    )
                weight_values = [self.band_weights[b] for b in self.bands]
            else:
                if len(self.band_weights) != len(self.bands):
                    raise ValueError(
                        f"band_weights list has length {len(self.band_weights)} "
                        f"but `bands` has length {len(self.bands)}. The list "
                        f"form of band_weights must align with `bands` "
                        f"element-for-element."
                    )
                weight_values = list(self.band_weights)
            for band_name, w in zip(self.bands, weight_values):
                if not (isinstance(w, (int, float)) and w > 0 and w == w and w != float("inf")):
                    raise ValueError(
                        f"band_weights[{band_name!r}] = {w!r} is not a "
                        f"positive finite float. Each weight must be > 0 and "
                        f"finite."
                    )

        # --- iteration consistency (copied from single-band) ---
        if self.maxsma is not None and self.maxsma < self.minsma:
            raise ValueError(f"maxsma ({self.maxsma}) must be greater than minsma ({self.minsma}).")
        if self.minit > self.maxit:
            raise ValueError(f"minit ({self.minit}) must be <= maxit ({self.maxit}).")

        # --- D11 backport: fix_per_band_background_to_zero is incompatible
        # with ref-mode (ref-mode does not exercise the joint solver, so
        # the column-drop has no effect there).
        if self.fix_per_band_background_to_zero and self.harmonic_combination == "ref":
            raise ValueError(
                "fix_per_band_background_to_zero is incompatible with "
                "harmonic_combination='ref': the ref mode bypasses the joint "
                "design matrix entirely. Pick one or the other."
            )

        # --- D9 backport: per-band-count normalization only makes sense
        # under loose validity (all N_b are identical under shared validity).
        if (
            self.loose_validity_band_normalization == "per_band_count"
            and not self.loose_validity
        ):
            raise ValueError(
                "loose_validity_band_normalization='per_band_count' requires "
                "loose_validity=True. Under shared validity all bands have "
                "the same N_b and the per-band-count renormalization is a "
                "no-op."
            )

        # --- soft warnings parallel to single-band V2/V3 ---
        if self.maxsma is not None and self.maxsma < self.sma0:
            warnings.warn(
                f"maxsma ({self.maxsma}) < sma0 ({self.sma0}): only one isophote + inward sweep will be produced.",
                UserWarning,
                stacklevel=2,
            )
        if self.minsma >= self.sma0:
            warnings.warn(
                f"minsma ({self.minsma}) >= sma0 ({self.sma0}): inward loop will not run.",
                UserWarning,
                stacklevel=2,
            )
        if self.geometry_update_mode == "simultaneous" and self.geometry_damping > 0.7:
            warnings.warn(
                "geometry_damping > 0.7 with geometry_update_mode='simultaneous' may cause oscillations; consider 0.5.",
                UserWarning,
                stacklevel=2,
            )
        if self.geometry_convergence and self.maxit < self.minit + self.geometry_stable_iters:
            warnings.warn(
                f"maxit ({self.maxit}) < minit + geometry_stable_iters "
                f"({self.minit + self.geometry_stable_iters}): "
                f"geometry convergence can never trigger.",
                UserWarning,
                stacklevel=2,
            )

        return self

    def resolved_band_weights(self) -> Dict[str, float]:
        """
        Return ``band_weights`` as a normalized ``{band: weight}`` dict.

        When ``band_weights`` is ``None`` returns uniform 1.0 for every
        band. Useful for downstream code that wants a single canonical
        form regardless of how the user constructed the config.
        """
        if self.band_weights is None:
            return {b: 1.0 for b in self.bands}
        if isinstance(self.band_weights, dict):
            return {b: float(self.band_weights[b]) for b in self.bands}
        return {b: float(w) for b, w in zip(self.bands, self.band_weights)}
