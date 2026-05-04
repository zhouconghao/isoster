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

- ``simultaneous_harmonics`` and ``isofit_mode`` (single-band ISOFIT
  API; the multi-band lift uses the ``multiband_higher_harmonics``
  enum instead, see plan section 6).
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
    fit_per_band_intens_jointly: bool = Field(
        default=True,
        description="Controls how per-band ``intens_<b>`` (a.k.a. the "
        "per-band intercept ``I0_b``) is computed at each isophote.\n"
        "\n"
        "When ``True`` (default, ``'joint'`` / coupled mode), the joint "
        "design matrix carries ``B`` per-band intercept columns alongside "
        "the shared ``(A1, B1, A2, B2)`` geometric harmonics, giving a "
        "``(B*N, B + 4)`` weighted-least-squares system. ``intens_<b>`` "
        "comes directly from the matrix solve, so the per-band intercept "
        "is co-fit with the harmonic deformations: on partial rings "
        "(masked sectors, loose validity) the joint solve absorbs the "
        "harmonic-shape coupling between bands and corrects each band's "
        "intercept accordingly. On full rings, ``sin(nφ)`` and ``cos(nφ)`` "
        "are mathematically orthogonal to the constant column over "
        "``[0, 2π]`` and the joint coefficient is numerically equivalent "
        "to the per-band inverse-variance-weighted mean.\n"
        "\n"
        "When ``False`` (``'ring_mean'`` / decoupled mode), the per-band "
        "intercept columns are dropped from the design matrix. "
        "``intens_<b>`` is computed independently as the per-band "
        "inverse-variance-weighted ring mean (or simple mean under OLS) "
        "and the geometric ``(A1, B1, A2, B2)`` solve runs on "
        "``intens - mean`` residuals. Use this when you have already "
        "subtracted sky upstream and want to interpret each band's "
        "intercept as a pure ring statistic decoupled from the harmonic "
        "fit. Mutually exclusive with ``harmonic_combination='ref'`` "
        "(ref-mode side-steps the joint solve entirely)."
        "\n\n"
        "**Note on integrator scope (Stage-2):** the multi-band path "
        "currently always uses an inverse-variance-weighted mean (WLS) "
        "or simple mean (OLS) for ``intens_<b>``, in both modes. The "
        "``integrator`` field on this config does NOT affect "
        "``intens_<b>`` reporting at converged isophotes — it only "
        "applies to per-band gradient computation and the forced-"
        "photometry fallback. Median-integrator support for "
        "``intens_<b>`` will land when the single-band integrator "
        "features are backported (Stage-3).",
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

    # --- Higher-order harmonics (multiband_higher_harmonics enum) ---
    multiband_higher_harmonics: Literal[
        "independent",
        "shared",
        "simultaneous_in_loop",
        "simultaneous_original",
    ] = Field(
        default="independent",
        description="Strategy for higher-order (n>=3) harmonic fitting "
        "across bands. ``'independent'`` (default): per-band post-hoc "
        "fits, one per band per order, uncoupled across bands "
        "(reproduces Stage-1 behavior bit-identically). ``'shared'``: "
        "after joint geometry has converged, run ONE post-hoc joint "
        "WLS/OLS solve for higher-order coefficients shared across all "
        "bands; (A1, B1, A2, B2) and per-band ``I0_b`` stay frozen at "
        "their converged-loop values. ``'simultaneous_in_loop'``: "
        "extend the joint design matrix every iteration to "
        "``(B*N, B + 4 + 2*len(harmonic_orders))`` with shared "
        "higher-order columns; all coefficients fit jointly inside the "
        "iteration loop (Ciambur 2015 in-loop variant). "
        "``'simultaneous_original'``: standard 5-param iteration loop, "
        "then ONE post-hoc joint solve over all orders simultaneously "
        "(Ciambur 2015 original variant). All non-``'independent'`` "
        "modes share higher-order coefficients across bands; per-band "
        "Schema-1 columns (``a3_<b>``, ...) carry the identical shared "
        "value. Section 6 of plan-2026-04-29-multiband-feasibility.md.",
    )
    harmonic_orders: List[int] = Field(
        default_factory=lambda: [3, 4],
        description="Higher-order harmonic orders fit for isophote "
        "deviations under all ``multiband_higher_harmonics`` modes. "
        "Default ``[3, 4]`` matches the Stage-1 hardcoded behavior. "
        "Each entry must be an int >= 3 (orders 1 and 2 are reserved "
        "for the geometric harmonics driving the iteration loop). "
        "Duplicates are rejected; the list is unique-sorted on "
        "construction. Mirrors the single-band "
        "``IsosterConfig.harmonic_orders`` field.",
    )

    # --- Outputs / features (copied, with multi-band restrictions) ---
    compute_errors: bool = Field(True, description="Calculate parameter errors.")
    compute_deviations: bool = Field(True, description="Calculate higher-order harmonic deviations (a3, b3, a4, b4).")
    full_photometry: bool = Field(False, description="Calculate flux integration metrics (tflux_e, etc.).")
    debug: bool = Field(False, description="Include debug info in results (per-band grad columns, ndata, nflag).")

    # --- Integrator (restricted: no 'adaptive' since LSB auto-lock is out) ---
    integrator: Literal["mean", "median"] = Field(
        default="mean",
        description="Integration method for flux statistics derived from a "
        "single ring of pixels. ``'adaptive'`` is intentionally not "
        "supported in Stage 1 because LSB auto-lock is out of scope.\n"
        "\n"
        "**Affects three distinct quantities** (Stage-3, plan Section 7 S1–S2):\n"
        "  (a) per-band gradient computation (``compute_joint_gradient``);\n"
        "  (b) forced-photometry fallback used by the driver for the "
        "central pixel and as a fail-safe when the iterative fit cannot "
        "converge;\n"
        "  (c) per-band ``intens_<b>`` at converged isophotes — but ONLY "
        "in the decoupled intercept mode (``fit_per_band_intens_jointly="
        "False``). The matrix-mode joint LS solve cannot host a median "
        "(non-linear), so ``integrator='median'`` is gated to the "
        "decoupled path: the validator hard-errors on "
        "``integrator='median' ∧ fit_per_band_intens_jointly=True``. In "
        "the decoupled path with ``integrator='median'``, "
        "``intens_<b>`` is the per-band median of the ring samples "
        "(plain ``np.median``; sample sigma-clipping has already been "
        "applied upstream by the sclip/nclip pipeline). With "
        "``integrator='mean'``, both intercept modes give a "
        "(weighted) ring mean — full rings are numerically identical "
        "since ``sin(nφ)``/``cos(nφ)`` are orthogonal to the constant "
        "column over ``[0, 2π]``.",
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

    @model_validator(mode="before")
    @classmethod
    def _reject_renamed_fields(cls, data):
        """Hard-error on the old ``fix_per_band_background_to_zero`` field.

        Renamed to :attr:`fit_per_band_intens_jointly` (with inverted
        polarity) in the Section 6 cleanup. The multi-band path is
        pre-production so we take a clean break: the old name is no
        longer accepted and there's no auto-translation. Users see a
        clear error pointing at the new field rather than a silent
        config drop (pydantic's default ``extra='ignore'`` would otherwise
        silently swallow it).
        """
        if isinstance(data, dict) and "fix_per_band_background_to_zero" in data:
            old = data["fix_per_band_background_to_zero"]
            new = "fit_per_band_intens_jointly={}".format(not bool(old))
            raise ValueError(
                f"`fix_per_band_background_to_zero` has been renamed to "
                f"`fit_per_band_intens_jointly` with inverted polarity. "
                f"Replace `fix_per_band_background_to_zero={old!r}` with "
                f"`{new}`. See Section 6 of "
                f"docs/agent/plan-2026-04-29-multiband-feasibility.md "
                f"for the rationale."
            )
        return data

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

        # --- D11 backport: the ``ring_mean`` intercept mode
        # (fit_per_band_intens_jointly=False) is incompatible with
        # ref-mode harmonic combination: ref-mode bypasses the joint
        # solver entirely so the column-drop is meaningless.
        if (not self.fit_per_band_intens_jointly) and self.harmonic_combination == "ref":
            raise ValueError(
                "fit_per_band_intens_jointly=False is incompatible with "
                "harmonic_combination='ref': ref-mode bypasses the joint "
                "design matrix entirely, so the per-band-intercept-column "
                "drop has nothing to act on. Either set "
                "fit_per_band_intens_jointly=True or pick "
                "harmonic_combination='joint'."
            )

        # --- Stage-3 S1: integrator='median' requires the decoupled
        # intercept mode. The matrix-mode joint LS cannot host a median
        # (non-linear); see plan section 7.2 S1 for the rejected Path-B
        # alternative (post-hoc median replacement) and why Path A was
        # chosen instead.
        if self.integrator == "median" and self.fit_per_band_intens_jointly:
            raise ValueError(
                "integrator='median' requires fit_per_band_intens_jointly="
                "False. The matrix-mode joint LS solve carries per-band "
                "intercept columns whose values come from a linear "
                "least-squares solve, which cannot represent a median. "
                "Use fit_per_band_intens_jointly=False to switch to the "
                "decoupled intercept mode where intens_<b> is computed "
                "as a per-band median of the ring samples, or stay on "
                "integrator='mean' to keep the matrix-mode solve."
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

        # --- Section 6: multiband_higher_harmonics validation ---
        # harmonic_orders must be a non-empty list of ints >= 3 with no
        # duplicates; unique-sort in place so downstream code can rely on
        # ascending order.
        if not isinstance(self.harmonic_orders, list) or len(self.harmonic_orders) == 0:
            raise ValueError(
                f"harmonic_orders must be a non-empty list of ints >= 3, got "
                f"{self.harmonic_orders!r}."
            )
        for n in self.harmonic_orders:
            if not isinstance(n, int) or isinstance(n, bool) or n < 3:
                raise ValueError(
                    f"harmonic_orders entries must be ints >= 3 (orders 1 "
                    f"and 2 are reserved for geometric harmonics); got "
                    f"{self.harmonic_orders!r}."
                )
        if len(set(self.harmonic_orders)) != len(self.harmonic_orders):
            seen: set = set()
            duplicates = [n for n in self.harmonic_orders if n in seen or seen.add(n)]
            raise ValueError(
                f"harmonic_orders contains duplicates {duplicates}; each "
                f"order must appear at most once."
            )
        # Unique-sort in place (mutating the validated list is fine — pydantic
        # has already accepted the field).
        self.harmonic_orders = sorted(self.harmonic_orders)

        # Hard-error: shared/simultaneous_* modes are incompatible with
        # ref-mode harmonic combination, which bypasses the joint solver.
        if (
            self.multiband_higher_harmonics != "independent"
            and self.harmonic_combination == "ref"
        ):
            raise ValueError(
                f"multiband_higher_harmonics={self.multiband_higher_harmonics!r} "
                f"is incompatible with harmonic_combination='ref': ref-mode "
                f"bypasses the joint design matrix entirely, so the "
                f"shared/simultaneous higher-order modes have nothing to "
                f"act on. Either set multiband_higher_harmonics='independent' "
                f"or pick harmonic_combination='joint'."
            )

        # Soft warning when a simultaneous_* mode is selected: single-band
        # benchmarks have flagged in-loop joint harmonic fitting as
        # unreliable; users should validate on PGC006669 + asteris before
        # trusting the output.
        if self.multiband_higher_harmonics in (
            "simultaneous_in_loop",
            "simultaneous_original",
        ):
            warnings.warn(
                f"multiband_higher_harmonics={self.multiband_higher_harmonics!r} "
                f"is experimental: the single-band equivalent "
                f"(IsosterConfig.simultaneous_harmonics=True) has shown "
                f"benchmark regressions. Multi-band may behave better thanks "
                f"to the joint constraint, but validate on the asteris and "
                f"PGC006669 demos before trusting the output. Consider "
                f"multiband_higher_harmonics='shared' as the lower-risk "
                f"alternative.",
                UserWarning,
                stacklevel=2,
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
