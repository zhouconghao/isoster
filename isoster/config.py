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
    
    # Forced Mode
    forced: bool = Field(False, description="Enable pure forced photometry mode (no fitting, just sampling).")
    forced_sma: Optional[List[float]] = Field(None, description="List of SMA values for forced mode. Required if forced=True.")
    
    # Eccentric Anomaly Sampling
    use_eccentric_anomaly: bool = Field(
        False,
        description="Use eccentric anomaly for uniform ellipse sampling (recommended for ε > 0.3). "
                    "Provides better sampling for high-ellipticity cases. ε = 1 - b/a."
    )

    # Higher-Order Harmonics Fitting
    simultaneous_harmonics: bool = Field(
        False,
        description="Use simultaneous fitting for higher-order harmonics (n >= 3). "
                    "ISOFIT-style approach from Ciambur 2015 that accounts for cross-correlations "
                    "between harmonics and provides better error estimates."
    )

    harmonic_orders: List[int] = Field(
        default_factory=lambda: [3, 4],
        description="List of harmonic orders to fit for isophote deviations. "
                    "Default [3, 4] fits 3rd and 4th order. Can extend to [3, 4, 5, 6] etc."
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
    def check_sma_consistency(self):
        if self.maxsma is not None and self.maxsma < self.minsma:
            raise ValueError(f"maxsma ({self.maxsma}) must be greater than minsma ({self.minsma})")
        if self.integrator == 'adaptive' and self.lsb_sma_threshold is None:
            raise ValueError("lsb_sma_threshold must be provided when integrator='adaptive'")
        if self.forced and (self.forced_sma is None or len(self.forced_sma) == 0):
            raise ValueError("forced_sma must be provided when forced=True")
        return self
