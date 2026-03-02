"""Configuration registry for the exhaustive isoster parameter sweep.

Defines all 39 configurations (P00 + S00-S23 + C01-C12) as a list of
(config_id, description, overrides_dict) tuples.  Imported by run_benchmark.py.

Galaxy geometry (x0, y0, eps, pa, sma0, maxsma) is NOT defined here —
it must be supplied at runtime via CLI arguments or auto-detected from the image.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Fitting parameter baseline
# ---------------------------------------------------------------------------
# These non-geometry fitting parameters are applied to every configuration.
# Geometry (x0, y0, eps, pa, sma0, maxsma) is injected by run_benchmark.py
# after parsing CLI arguments.
# ---------------------------------------------------------------------------

PARAMETER_BASELINE: dict = {
    "nclip": 2,
    "maxit": 100,
    "astep": 0.1,
    "maxgerr": 0.5,
    "conver": 0.05,
    "convergence_scaling": "sector_area",
    "geometry_damping": 0.7,
    "geometry_update_mode": "largest",
    "compute_errors": True,
    "compute_deviations": True,
}

# Photutils-equivalent fitting parameters used by run_photutils_baseline().
# Geometry is merged in at runtime from CLI args.
PHOTUTILS_PARAMETER_CONFIG: dict = {
    "minsma": 1.0,
    "nclip": 2,
    "astep": 0.1,
    "maxgerr": 0.5,
    "sclip": 3.0,
    "integrmode": "bilinear",
}

# ---------------------------------------------------------------------------
# Isoster configurations: (config_id, description, overrides_from_S00)
# S00 uses PARAMETER_BASELINE + geometry as-is (no extra overrides).
# ---------------------------------------------------------------------------

CONFIGURATIONS: list[tuple[str, str, dict]] = [
    # === Baseline ===
    ("S00", "Isoster baseline (defaults)", {}),

    # === Single-parameter sweeps ===
    ("S01", "convergence_scaling=none", {
        "convergence_scaling": "none",
    }),
    ("S02", "convergence_scaling=sqrt_sma", {
        "convergence_scaling": "sqrt_sma",
    }),
    ("S03", "geometry_damping=0.5", {
        "geometry_damping": 0.5,
    }),
    ("S04", "geometry_damping=1.0 (no damping)", {
        "geometry_damping": 1.0,
    }),
    ("S05", "geometry_update_mode=simultaneous (+damping=0.5)", {
        "geometry_update_mode": "simultaneous",
        "geometry_damping": 0.5,
    }),
    ("S06", "geometry_convergence=True", {
        "geometry_convergence": True,
    }),
    ("S07", "permissive_geometry=True", {
        "permissive_geometry": True,
    }),
    ("S08", "use_eccentric_anomaly=True", {
        "use_eccentric_anomaly": True,
    }),
    ("S09", "simultaneous_harmonics=True (in_loop)", {
        "simultaneous_harmonics": True,
        "isofit_mode": "in_loop",
    }),
    ("S10", "simultaneous_harmonics=True (original)", {
        "simultaneous_harmonics": True,
        "isofit_mode": "original",
    }),
    ("S11", "harmonic_orders=[3,4,5,6,7]", {
        "harmonic_orders": [3, 4, 5, 6, 7],
    }),
    ("S12", "use_central_regularization=True", {
        "use_central_regularization": True,
    }),
    ("S13", "integrator=median", {
        "integrator": "median",
    }),
    ("S14", "integrator=adaptive (lsb_sma=100)", {
        "integrator": "adaptive",
        "lsb_sma_threshold": 100.0,
    }),
    ("S15", "conver=0.02 (strict)", {
        "conver": 0.02,
    }),
    ("S16", "conver=0.10 (loose)", {
        "conver": 0.10,
    }),
    ("S17", "maxit=50", {
        "maxit": 50,
    }),
    ("S18", "maxit=300 (extreme)", {
        "maxit": 300,
    }),
    ("S19", "sclip=2.0 (strict)", {
        "sclip": 2.0,
    }),
    ("S20", "sclip_low=3.0 + sclip_high=2.0 (asymmetric)", {
        "sclip_low": 3.0,
        "sclip_high": 2.0,
    }),
    ("S21", "full_photometry + compute_cog", {
        "full_photometry": True,
        "compute_cog": True,
    }),
    # S22 and S23 need reference geometry from photutils — handled at runtime
    ("S22", "fix_center=True (photutils median)", {
        "fix_center": True,
        # x0, y0 overridden at runtime from photutils reference geometry
    }),
    ("S23", "fix_center + fix_pa + fix_eps (photutils medians)", {
        "fix_center": True,
        "fix_pa": True,
        "fix_eps": True,
        # x0, y0, pa, eps overridden at runtime from photutils reference geometry
    }),

    # === Combination configs ===
    ("C01", "pure legacy (scaling=none, damping=1.0)", {
        "convergence_scaling": "none",
        "geometry_damping": 1.0,
    }),
    ("C02", "permissive + geometry_convergence", {
        "permissive_geometry": True,
        "geometry_convergence": True,
    }),
    ("C03", "EA + ISOFIT in_loop", {
        "use_eccentric_anomaly": True,
        "simultaneous_harmonics": True,
        "isofit_mode": "in_loop",
    }),
    ("C04", "EA + extended harmonics [3-7]", {
        "use_eccentric_anomaly": True,
        "harmonic_orders": [3, 4, 5, 6, 7],
    }),
    ("C05", "ISOFIT in_loop + extended harmonics [3-7]", {
        "simultaneous_harmonics": True,
        "isofit_mode": "in_loop",
        "harmonic_orders": [3, 4, 5, 6, 7],
    }),
    ("C06", "simultaneous_geom + damping=0.5 + geom_conv", {
        "geometry_update_mode": "simultaneous",
        "geometry_damping": 0.5,
        "geometry_convergence": True,
    }),
    ("C07", "permissive + scaling=none", {
        "permissive_geometry": True,
        "convergence_scaling": "none",
    }),
    ("C08", "central_reg + damping=0.5", {
        "use_central_regularization": True,
        "geometry_damping": 0.5,
    }),
    ("C09", "EA + ISOFIT original + extended [3-7] (full Ciambur 2015)", {
        "use_eccentric_anomaly": True,
        "simultaneous_harmonics": True,
        "isofit_mode": "original",
        "harmonic_orders": [3, 4, 5, 6, 7],
    }),
    ("C10", "permissive + EA + ISOFIT in_loop (kitchen sink)", {
        "permissive_geometry": True,
        "use_eccentric_anomaly": True,
        "simultaneous_harmonics": True,
        "isofit_mode": "in_loop",
    }),
    ("C11", "conver=0.02 + maxit=50 (stress test)", {
        "conver": 0.02,
        "maxit": 50,
    }),
    ("C12", "sector_area + damping=0.5 + permissive", {
        "convergence_scaling": "sector_area",
        "geometry_damping": 0.5,
        "permissive_geometry": True,
    }),
]

# Configs that use extended harmonics and should get extended QA figures
EXTENDED_HARMONIC_CONFIGS = {"S11", "C04", "C05", "C09"}

# Configs that need runtime reference geometry from photutils
NEEDS_REFERENCE_GEOMETRY = {"S22", "S23"}
