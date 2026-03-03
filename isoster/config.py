import warnings
from typing import Optional, List
from pydantic import BaseModel, Field, model_validator

class IsosterConfig(BaseModel):
    """
    Configuration for isoster fitting.

    This Pydantic model defines all tunable parameters for the isophote fitting algorithm,
    including geometry initialization, fitting control, quality control, and output options.

    **Stop Codes:**
    Fitting results include a 'stop_code' field indicating termination condition:
    - 0: Success (converged)
    - 1: Too many flagged pixels (>fflag threshold)
    - 2: Maximum iteration count reached without convergence
    - 3: Too few points (<6)
    - -1: Gradient error (invalid or unreliable)

    See docs/user-guide.md (Stop Codes (Canonical Reference)) for detailed documentation.
    """
    # Geometry initialization
    x0: Optional[float] = Field(None, description="Initial center x coordinate. If None, uses image center.")
    y0: Optional[float] = Field(None, description="Initial center y coordinate. If None, uses image center.")
    eps: float = Field(0.2, ge=0.0, lt=1.0, description="Initial ellipticity.")
    pa: float = Field(0.0, description="Initial position angle in radians.")
    
    # SMA control
    sma0: float = Field(10.0, gt=0.0, description="Starting semi-major axis length.")
    minsma: float = Field(0.0, ge=0.0, description="Minimum SMA to fit.")
    maxsma: Optional[float] = Field(None, gt=0.0, description="Maximum SMA to fit. If None, uses image size/2.")
    astep: float = Field(0.1, gt=0.0, description="Step size for SMA growth.")
    linear_growth: bool = Field(False, description="If True, sma = sma + astep. Else sma = sma * (1 + astep).")
    
    # Fitting control
    maxit: int = Field(50, gt=0, description="Maximum iterations per isophote.")
    minit: int = Field(10, gt=0, description="Minimum iterations per isophote.")
    conver: float = Field(0.05, gt=0.0, description="Convergence threshold (max harmonic amplitude / rms).")
    convergence_scaling: str = Field(
        default='sector_area',
        pattern='^(none|sector_area|sqrt_sma)$',
        description="Scale convergence threshold with SMA. "
                    "'sector_area': multiply by approximate sector area (matches photutils behavior, default). "
                    "'sqrt_sma': multiply by sqrt(sma). "
                    "'none': constant threshold (legacy behavior)."
    )
    use_lazy_gradient: bool = Field(
        default=True,
        description="If True, use 'Lazy Gradient Evaluation' (Modified Newton Method). "
                    "The radial gradient is computed only on the first iteration and reused "
                    "for subsequent iterations unless convergence stalls. This significantly "
                    "reduces image sampling overhead."
    )

    # Geometry update control
    geometry_damping: float = Field(
        default=0.7,
        gt=0.0,
        le=1.0,
        description="Damping factor for geometry updates (0 < d <= 1). "
                    "Each geometry correction is multiplied by this factor. "
                    "Default 0.7 stabilizes oscillations at outer isophotes. "
                    "Use 1.0 for no damping (legacy behavior)."
    )
    geometry_convergence: bool = Field(
        default=False,
        description="Enable secondary convergence based on geometry stability. "
                    "Declares convergence when geometry changes fall below tolerance "
                    "for consecutive iterations, even if harmonic criterion is not met."
    )
    geometry_tolerance: float = Field(
        default=0.01,
        gt=0.0,
        description="Threshold for geometry convergence. Convergence declared when "
                    "max(|delta_eps|, |delta_pa/pi|, |delta_x0/sma|, |delta_y0/sma|) "
                    "< geometry_tolerance for geometry_stable_iters consecutive iterations."
    )
    geometry_stable_iters: int = Field(
        default=3,
        ge=2,
        description="Number of consecutive iterations with small geometry changes "
                    "required to trigger geometry-based convergence."
    )

    # Quality control
    sclip: float = Field(3.0, gt=0.0, description="Sigma clipping threshold.")
    nclip: int = Field(0, ge=0, description="Number of sigma clipping iterations.")
    sclip_low: Optional[float] = Field(None, description="Lower sigma clipping threshold.")
    sclip_high: Optional[float] = Field(None, description="Upper sigma clipping threshold.")
    fflag: float = Field(0.5, ge=0.0, le=1.0, description="Maximum fraction of flagged data (masked + clipped).")
    maxgerr: float = Field(0.5, gt=0.0, description="Maximum relative error in local radial gradient.")
    
    # Constraints
    fix_center: bool = Field(False, description="Fix center coordinates during fitting.")
    fix_pa: bool = Field(False, description="Fix position angle during fitting.")
    fix_eps: bool = Field(False, description="Fix ellipticity during fitting.")
    
    # Outputs & Features
    compute_errors: bool = Field(True, description="Calculate parameter errors.")
    compute_deviations: bool = Field(True, description="Calculate higher-order harmonic deviations (a3, b3, etc.).")
    full_photometry: bool = Field(False, description="Calculate flux integration metrics (tflux_e, etc.).")
    compute_cog: bool = Field(False, description="Calculate curve-of-growth photometry.")
    debug: bool = Field(False, description="Include debug info in results and enable verbose calculation.")

    # Integration Mode
    integrator: str = Field(default='mean', pattern='^(mean|median|adaptive)$', description="Integration method for flux calculation.")
    lsb_sma_threshold: Optional[float] = Field(None, gt=0.0, description="SMA threshold for switching to median integrator in adaptive mode.")
    
    # Eccentric Anomaly Sampling
    use_eccentric_anomaly: bool = Field(
        False,
        description="Use eccentric anomaly for uniform ellipse sampling (recommended for ε > 0.3). "
                    "Provides better sampling for high-ellipticity cases. ε = 1 - b/a."
    )

    # Higher-Order Harmonics Fitting
    simultaneous_harmonics: bool = Field(
        False,
        description="Enable true ISOFIT simultaneous harmonic fitting (Ciambur 2015). "
                    "When True, higher-order harmonics (specified by harmonic_orders) are fitted "
                    "jointly with the geometry harmonics (orders 1-2) inside the iteration loop, "
                    "accounting for cross-correlations and yielding cleaner RMS estimates. "
                    "Falls back to 5-param fit when insufficient sample points. "
                    "When False, harmonics are fitted post-hoc after geometry convergence."
    )

    harmonic_orders: List[int] = Field(
        default_factory=lambda: [3, 4],
        description="List of harmonic orders to fit for isophote deviations. "
                    "Default [3, 4] fits 3rd and 4th order. Can extend to [3, 4, 5, 6] etc."
    )

    isofit_mode: str = Field(
        default='in_loop',
        pattern='^(in_loop|original)$',
        description="ISOFIT algorithm variant (only meaningful when simultaneous_harmonics=True). "
                    "'in_loop': fit all harmonic orders simultaneously inside the iteration loop "
                    "(current isoster behavior, more aggressive than Ciambur 2015). "
                    "'original': match Ciambur 2015 — use 5-param fit inside the loop for geometry, "
                    "then fit all higher-order harmonics simultaneously post-hoc after convergence."
    )

    # Geometry update strategy
    geometry_update_mode: str = Field(
        default='largest',
        pattern='^(largest|simultaneous)$',
        description="Geometry update strategy per iteration. "
                    "'largest': update only the geometry parameter with the largest harmonic amplitude "
                    "(coordinate descent, traditional isofit/photutils behavior). "
                    "'simultaneous': update all four geometry parameters (x0, y0, PA, eps) each "
                    "iteration using their respective harmonic corrections. Typically converges in "
                    "fewer iterations but may need lower damping (e.g., 0.5)."
    )

    # Central Region Geometry Regularization
    use_central_regularization: bool = Field(
        False,
        description="Enable geometry regularization for central region to stabilize fitting at low SMA. "
                    "Adds penalty for large geometry changes at SMA < threshold."
    )
    
    central_reg_sma_threshold: float = Field(
        5.0,
        gt=0.0,
        description="SMA threshold for central regularization (pixels). "
                    "Regularization strength decays exponentially with distance from center."
    )
    
    central_reg_strength: float = Field(
        1.0,
        ge=0.0,
        description="Maximum regularization strength at SMA=0. "
                    "0=no regularization, 1=moderate, 10=strong. Typical range: 0.1-10."
    )
    
    central_reg_weights: dict = Field(
        default_factory=lambda: {'eps': 1.0, 'pa': 1.0, 'center': 1.0},
        description="Relative weights for regularization penalties on ellipticity, PA, and center. "
                    "Dict with keys: 'eps', 'pa', 'center'."
    )

    # Permissive Geometry Mode (photutils-style)
    permissive_geometry: bool = Field(
        False,
        description="Enable permissive geometry updates (photutils-style). "
                    "When True, geometry is always updated even from failed fits to prevent "
                    "cascading failures, and None gradient errors are treated as acceptable. "
                    "This matches photutils's 'best effort' approach vs isoster's stricter "
                    "'convergence-required' default behavior."
    )

    @model_validator(mode='after')
    def check_config_consistency(self):
        """Validate config consistency and emit warnings for likely misconfigurations."""
        # --- Hard errors ---
        if self.maxsma is not None and self.maxsma < self.minsma:
            raise ValueError(f"maxsma ({self.maxsma}) must be greater than minsma ({self.minsma})")
        if self.minit > self.maxit:
            raise ValueError(f"minit ({self.minit}) must be <= maxit ({self.maxit})")
        if self.integrator == 'adaptive' and self.lsb_sma_threshold is None:
            raise ValueError("lsb_sma_threshold must be provided when integrator='adaptive'")
        if any(order < 3 for order in self.harmonic_orders):
            raise ValueError(
                f"harmonic_orders must all be >= 3 (orders 1 and 2 are used internally "
                f"for geometry fitting), got {self.harmonic_orders}"
            )
        # V7: central_reg_weights must use valid keys
        valid_reg_keys = {'eps', 'pa', 'center'}
        unknown_keys = set(self.central_reg_weights.keys()) - valid_reg_keys
        if unknown_keys:
            raise ValueError(
                f"central_reg_weights contains unknown keys: {unknown_keys}. "
                f"Valid keys: {valid_reg_keys}"
            )

        # --- Soft warnings for likely misconfigurations ---

        # V1: isofit_mode is a no-op without simultaneous_harmonics
        if self.isofit_mode != 'in_loop' and not self.simultaneous_harmonics:
            warnings.warn(
                "isofit_mode has no effect when simultaneous_harmonics=False",
                UserWarning, stacklevel=2
            )

        # V2: maxsma < sma0 means only one isophote + inward sweep
        if self.maxsma is not None and self.maxsma < self.sma0:
            warnings.warn(
                f"maxsma ({self.maxsma}) < sma0 ({self.sma0}): "
                f"only one isophote + inward sweep will be produced",
                UserWarning, stacklevel=2
            )

        # V3: minsma >= sma0 means inward loop never runs
        if self.minsma >= self.sma0:
            warnings.warn(
                f"minsma ({self.minsma}) >= sma0 ({self.sma0}): "
                f"inward loop will not run",
                UserWarning, stacklevel=2
            )

        # V4: simultaneous geometry update with high damping may oscillate
        if (self.geometry_update_mode == 'simultaneous'
                and self.geometry_damping > 0.7):
            warnings.warn(
                "geometry_damping > 0.7 with geometry_update_mode='simultaneous' "
                "may cause oscillations; consider 0.5",
                UserWarning, stacklevel=2
            )

        # V10: geometry convergence can never trigger if maxit too small
        if (self.geometry_convergence
                and self.maxit < self.minit + self.geometry_stable_iters):
            warnings.warn(
                f"maxit ({self.maxit}) < minit + geometry_stable_iters "
                f"({self.minit + self.geometry_stable_iters}): "
                f"geometry convergence can never trigger",
                UserWarning, stacklevel=2
            )

        # V11: lsb_sma_threshold with non-adaptive integrator is ignored
        if self.integrator != 'adaptive' and self.lsb_sma_threshold is not None:
            warnings.warn(
                f"lsb_sma_threshold is set but integrator='{self.integrator}' "
                f"(not 'adaptive'); threshold will be ignored",
                UserWarning, stacklevel=2
            )

        return self
