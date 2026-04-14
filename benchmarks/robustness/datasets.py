"""Dataset loaders for the robustness benchmark.

Four tiers, ordered easy → hard:

1. ``mocks``     — clean synthetic Sersic images generated on demand via
   ``benchmarks.utils.sersic_model``. Pure-geometry controlled test.
2. ``huang2013`` — Huang 2013 external libprofit mock galaxies (one
   galaxy, four mock IDs at present), with header-driven initial
   geometry via ``examples/example_huang2013/huang2013_shared``.
3. ``highorder`` — high-order-harmonic LegacySurvey galaxies
   (``eso243-49``, ``ngc3610``) from ``data/`` at the project root,
   reusing ``examples/example_ls_highorder_harmonic/shared.py`` helpers.
4. ``hsc``       — 6 HSC edge-case galaxies from
   ``examples/example_hsc_edgecases/data/`` with pre-packaged
   ``image``/``mask``/``variance`` FITS files in the ``HSC_I`` band.
   Fiducial starting conditions mirror ``run_lsb_mode_sweep`` so the
   robustness reference fit matches the shipped LSB sweep's ``B_std``
   arm.

Each tier provides ``list_galaxies(tier)`` → list of ``GalaxySpec``
objects and ``load_galaxy(spec)`` → ``GalaxyData``. The sweep script
imports these two functions and does not depend on the individual
loader implementations.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.utils.sersic_model import (  # noqa: E402
    create_sersic_image_vectorized,
    get_true_profile_at_sma,
)

TIERS = ("mocks", "huang2013", "highorder", "hsc")


@dataclass
class GalaxySpec:
    """Lightweight descriptor for a galaxy in the sweep."""

    tier: str
    obj_id: str
    description: str
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GalaxyData:
    """Loaded image plus fiducial starting conditions for a galaxy.

    ``config_overrides`` carries tier-specific knobs (e.g., the
    ``permissive_geometry`` / ``geometry_damping`` /
    ``convergence_scaling`` set that real LegacySurvey galaxies need)
    that run_sweep merges into the per-arm IsosterConfig. Mocks leave
    it empty.
    """

    spec: GalaxySpec
    image: np.ndarray
    mask: Optional[np.ndarray]
    variance_map: Optional[np.ndarray]
    fiducial_sma0: float
    fiducial_eps: float
    fiducial_pa: float
    fiducial_x0: float
    fiducial_y0: float
    maxsma: float
    minsma: float = 0.0
    truth: Optional[Dict[str, Any]] = None
    config_overrides: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tier 1: clean synthetic Sersic mocks
# ---------------------------------------------------------------------------

MOCK_PROFILES: Dict[str, Dict[str, Any]] = {
    "mock_disk_low_n": {
        "description": "disk, moderate ellipticity",
        "n": 1.0,
        "r_eff": 30.0,
        "i_eff": 100.0,
        "eps": 0.4,
        "pa": 0.0,
        "shape": (256, 256),
    },
    "mock_disk_elong": {
        "description": "highly elongated, rotated",
        "n": 1.0,
        "r_eff": 30.0,
        "i_eff": 100.0,
        "eps": 0.7,
        "pa": np.pi / 4.0,
        "shape": (256, 256),
    },
    "mock_bulge_high_n": {
        "description": "compact bulge, near-circular",
        "n": 4.0,
        "r_eff": 15.0,
        "i_eff": 200.0,
        "eps": 0.1,
        "pa": 0.0,
        "shape": (192, 192),
    },
    "mock_exp_large": {
        "description": "large low-surface-brightness exponential",
        "n": 1.5,
        "r_eff": 60.0,
        "i_eff": 30.0,
        "eps": 0.3,
        "pa": np.pi / 3.0,
        "shape": (384, 384),
    },
}


def _list_mocks() -> List[GalaxySpec]:
    return [
        GalaxySpec(
            tier="mocks",
            obj_id=name,
            description=profile["description"],
            extras={"profile_key": name},
        )
        for name, profile in MOCK_PROFILES.items()
    ]


def _load_mock(spec: GalaxySpec) -> GalaxyData:
    profile = MOCK_PROFILES[spec.obj_id]
    image, params = create_sersic_image_vectorized(
        n=profile["n"],
        R_e=profile["r_eff"],
        I_e=profile["i_eff"],
        eps=profile["eps"],
        pa=profile["pa"],
        shape=profile["shape"],
        oversample=10,
        oversample_radius=5.0,
    )
    height, width = profile["shape"]
    maxsma = float(min(height, width) / 2.0 - 5.0)
    # Truth profile on a dense SMA grid, for optional sanity checks.
    dense_sma = np.linspace(1.0, maxsma, 256)
    truth = {
        "params": params,
        "profile": get_true_profile_at_sma(dense_sma, params),
    }
    return GalaxyData(
        spec=spec,
        image=image,
        mask=None,
        variance_map=None,
        fiducial_sma0=float(profile["r_eff"]),
        fiducial_eps=float(profile["eps"]),
        fiducial_pa=float(profile["pa"]),
        fiducial_x0=float(params["x0"]),
        fiducial_y0=float(params["y0"]),
        maxsma=maxsma,
        minsma=0.0,
        truth=truth,
    )


# ---------------------------------------------------------------------------
# Tier 2: Huang2013 libprofit mock galaxies
# ---------------------------------------------------------------------------

#: Default location of Huang2013 mock FITS inputs. These are external,
#: large (~27 MB each as float64), and do not ship with the repo. Set
#: ``HUANG2013_DATA_ROOT`` to override — the loader reads the directory
#: tree ``<root>/<GALAXY>/<GALAXY>_mock<ID>.fits`` identical to the
#: layout used by ``examples/example_huang2013``.
_HUANG2013_DEFAULT_DATA_ROOT = Path(
    "/Users/shuang/Library/CloudStorage/Dropbox/work/project/otters/"
    "isophote_test/output/huang2013"
)


def _huang2013_data_root() -> Path:
    env_root = os.environ.get("HUANG2013_DATA_ROOT")
    if env_root:
        return Path(env_root)
    return _HUANG2013_DEFAULT_DATA_ROOT


#: One galaxy with four independent libprofit mocks. Each mock shares
#: the same OBJECT/REDSHIFT/NCOMP structure but varies the component
#: realizations, so they exercise the initial-condition capture radius
#: on four slightly different surface-brightness profiles.
_HUANG2013_GALAXIES: tuple[tuple[str, int], ...] = (
    ("IC2597", 1),
    ("IC2597", 2),
    ("IC2597", 3),
    ("IC2597", 4),
)


def _huang2013_obj_id(galaxy: str, mock_id: int) -> str:
    return f"{galaxy}_mock{mock_id}"


def _list_huang2013() -> List[GalaxySpec]:
    return [
        GalaxySpec(
            tier="huang2013",
            obj_id=_huang2013_obj_id(galaxy, mock_id),
            description=f"Huang2013 libprofit mock: {galaxy} mock{mock_id}",
            extras={"galaxy": galaxy, "mock_id": mock_id},
        )
        for galaxy, mock_id in _HUANG2013_GALAXIES
    ]


def _load_huang2013(spec: GalaxySpec) -> GalaxyData:
    """Load a Huang2013 libprofit mock (clean image, header-driven geometry).

    Reuses ``infer_initial_geometry`` and ``infer_default_maxsma`` from
    the ``examples/example_huang2013`` helpers so the robustness sweep's
    fiducial start matches the shipped Huang2013 campaign. No mask or
    variance map — these mocks are noise-free by construction (header
    flag ``NOISE = False``).
    """
    from astropy.io import fits

    example_dir = PROJECT_ROOT / "examples" / "example_huang2013"
    if not example_dir.exists():
        raise FileNotFoundError(
            f"expected example directory not found: {example_dir}"
        )
    if str(example_dir) not in sys.path:
        sys.path.insert(0, str(example_dir))

    # Lazy imports — only paid when the huang2013 tier is touched.
    from huang2013_shared import (  # type: ignore[import-not-found]
        infer_initial_geometry,
    )
    from run_huang2013_profile_extraction import (  # type: ignore[import-not-found]
        infer_default_maxsma,
    )

    galaxy = str(spec.extras["galaxy"])
    mock_id = int(spec.extras["mock_id"])

    data_root = _huang2013_data_root()
    fits_path = data_root / galaxy / f"{_huang2013_obj_id(galaxy, mock_id)}.fits"
    if not fits_path.exists():
        raise FileNotFoundError(
            f"huang2013 FITS file not found: {fits_path}. "
            f"Set HUANG2013_DATA_ROOT to point at the "
            f"``<GALAXY>/<GALAXY>_mock<ID>.fits`` tree."
        )

    image = np.asarray(fits.getdata(str(fits_path)), dtype=np.float64)
    header = fits.getheader(str(fits_path))

    fiducial = infer_initial_geometry(header, image.shape)
    fiducial_pa_rad = float(np.deg2rad(fiducial["pa_deg"]))
    maxsma = float(infer_default_maxsma(header, image.shape))

    return GalaxyData(
        spec=spec,
        image=image,
        mask=None,
        variance_map=None,
        fiducial_sma0=float(fiducial["sma0"]),
        fiducial_eps=float(fiducial["eps"]),
        fiducial_pa=fiducial_pa_rad,
        fiducial_x0=float(fiducial["x0"]),
        fiducial_y0=float(fiducial["y0"]),
        maxsma=maxsma,
        minsma=0.0,
        truth=None,
        # No overrides: clean libprofit mocks converge with defaults.
    )


# ---------------------------------------------------------------------------
# Tier 3: LegacySurvey high-order-harmonic real galaxies
# ---------------------------------------------------------------------------

#: Shared config overrides for real LegacySurvey galaxies. These match
#: the knobs used by ``examples/example_ls_highorder_harmonic/shared.py::
#: make_isoster_configs`` and are the minimum needed to fit real images
#: cleanly regardless of arm.
_HIGHORDER_CONFIG_OVERRIDES: Dict[str, Any] = {
    "convergence_scaling": "sector_area",
    "geometry_damping": 0.7,
    "permissive_geometry": True,
}

#: Band index to extract from the 3-D LegacySurvey cube. Both galaxies
#: use r-band at index 1, matching the example campaign default.
_HIGHORDER_BAND_INDEX = 1

_HIGHORDER_SPECS: tuple[GalaxySpec, ...] = (
    GalaxySpec(
        tier="highorder",
        obj_id="eso243-49",
        description="LegacySurvey ESO243-49 r-band (high-order harmonic target)",
        extras={"band_index": _HIGHORDER_BAND_INDEX},
    ),
    GalaxySpec(
        tier="highorder",
        obj_id="ngc3610",
        description="LegacySurvey NGC3610 r-band (high-order harmonic target)",
        extras={"band_index": _HIGHORDER_BAND_INDEX},
    ),
)


def _list_highorder() -> List[GalaxySpec]:
    return list(_HIGHORDER_SPECS)


def _highorder_mask_cache_path(galaxy: str, band_index: int) -> Path:
    cache_dir = (
        PROJECT_ROOT
        / "outputs"
        / "benchmark_robustness"
        / "cache"
        / "highorder"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{galaxy}_band{band_index}_mask.npz"


def _load_highorder(spec: GalaxySpec) -> GalaxyData:
    """Load a LegacySurvey high-order-harmonic galaxy (real image + mask).

    The example folder (``examples/example_ls_highorder_harmonic``) is
    added to ``sys.path`` lazily so the robustness module does not depend
    on photutils/astropy-convolution unless this tier is actually used.
    """
    import warnings

    example_dir = PROJECT_ROOT / "examples" / "example_ls_highorder_harmonic"
    if not example_dir.exists():
        raise FileNotFoundError(
            f"expected example directory not found: {example_dir}"
        )
    if str(example_dir) not in sys.path:
        sys.path.insert(0, str(example_dir))

    # Lazy imports — only paid when the highorder tier is touched.
    from masking import make_object_mask  # type: ignore[import-not-found]
    from shared import (  # type: ignore[import-not-found]
        FITS_FILENAME,
        INITIAL_SMA,
        MASK_PARAMS,
        load_legacysurvey_fits,
    )

    galaxy = spec.obj_id
    band_index = int(spec.extras.get("band_index", _HIGHORDER_BAND_INDEX))

    fits_path = PROJECT_ROOT / "data" / FITS_FILENAME[galaxy]
    if not fits_path.exists():
        raise FileNotFoundError(
            f"highorder FITS file not found: {fits_path}. "
            f"See examples/example_ls_highorder_harmonic/README.md."
        )

    image, _ = load_legacysurvey_fits(fits_path, band_index)
    height, width = image.shape

    # Mask is deterministic from (image, params); cache it to avoid the
    # several-second photutils segmentation cost on every run.
    cache_file = _highorder_mask_cache_path(galaxy, band_index)
    fits_mtime = fits_path.stat().st_mtime
    if cache_file.exists() and cache_file.stat().st_mtime >= fits_mtime:
        with np.load(cache_file) as loaded:
            mask = loaded["mask"].astype(bool)
    else:
        mask_kwargs = MASK_PARAMS.get(galaxy, {})
        center_xy = ((width - 1) / 2.0, (height - 1) / 2.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask = make_object_mask(
                image, center_xy=center_xy, **mask_kwargs
            )
        np.savez_compressed(cache_file, mask=mask.astype(bool))

    fiducial_sma0 = float(INITIAL_SMA[galaxy])
    x0 = (width - 1) / 2.0
    y0 = (height - 1) / 2.0
    # Leave a 5-pixel buffer from the nearest edge; isoster's outward
    # walk stops earlier on real data anyway.
    maxsma = float(min(height, width) / 2.0 - 5.0)

    return GalaxyData(
        spec=spec,
        image=image,
        mask=mask,
        variance_map=None,
        fiducial_sma0=fiducial_sma0,
        fiducial_eps=0.0,
        fiducial_pa=0.0,
        fiducial_x0=x0,
        fiducial_y0=y0,
        maxsma=maxsma,
        minsma=0.0,
        truth=None,
        config_overrides=dict(_HIGHORDER_CONFIG_OVERRIDES),
    )


# ---------------------------------------------------------------------------
# Tier 4: HSC edge-case galaxies
# ---------------------------------------------------------------------------

#: 6 HSC edge-case galaxies from ``examples/example_hsc_edgecases``.
#: Order and descriptions mirror ``run_lsb_mode_sweep.GALAXIES`` so rows
#: of this tier are directly comparable to the shipped LSB sweep.
_HSC_GALAXIES: tuple[tuple[str, str], ...] = (
    ("10140088", "clear case"),
    ("10140002", "nearby bright star"),
    ("10140006", "nearby large galaxy"),
    ("10140009", "blending bright star"),
    ("10140056", "artifact"),
    ("10140093", "small blending source"),
)

#: HSC band used for the robustness sweep. Matches ``run_lsb_mode_sweep``.
_HSC_BAND = "HSC_I"

#: Fiducial starting conditions lifted from
#: ``examples/example_hsc_edgecases/run_lsb_mode_sweep.BASE_CONFIG``.
#: eps/pa live on ``GalaxyData`` because the robustness sweep perturbs
#: them; the rest (astep, sclip, nclip, etc.) is already captured by
#: ``run_sweep.BASE_CONFIG``, so no per-galaxy ``config_overrides`` is
#: required for this tier.
_HSC_FIDUCIAL_SMA0 = 10.0
_HSC_FIDUCIAL_EPS = 0.2
_HSC_FIDUCIAL_PA = 0.0


def _list_hsc() -> List[GalaxySpec]:
    return [
        GalaxySpec(
            tier="hsc",
            obj_id=obj_id,
            description=f"HSC edge case: {label}",
            extras={"band": _HSC_BAND, "label": label},
        )
        for obj_id, label in _HSC_GALAXIES
    ]


def _load_hsc(spec: GalaxySpec) -> GalaxyData:
    """Load a pre-packaged HSC edge-case galaxy (image + variance + mask)."""
    from astropy.io import fits

    band = spec.extras.get("band", _HSC_BAND)
    galaxy_dir = (
        PROJECT_ROOT / "examples" / "example_hsc_edgecases" / "data" / spec.obj_id
    )
    if not galaxy_dir.is_dir():
        raise FileNotFoundError(
            f"HSC galaxy directory not found: {galaxy_dir}. "
            f"See examples/example_hsc_edgecases/README.md."
        )

    image_path = galaxy_dir / f"{spec.obj_id}_{band}_image.fits"
    mask_path = galaxy_dir / f"{spec.obj_id}_{band}_mask.fits"
    variance_path = galaxy_dir / f"{spec.obj_id}_{band}_variance.fits"

    for path in (image_path, mask_path, variance_path):
        if not path.exists():
            raise FileNotFoundError(f"HSC file missing: {path}")

    image = np.asarray(fits.getdata(str(image_path))).astype(np.float64)
    # Mask convention: non-zero → bad, matching isoster's boolean mask.
    mask = np.asarray(fits.getdata(str(mask_path))).astype(bool)
    variance = np.asarray(fits.getdata(str(variance_path))).astype(np.float64)

    height, width = image.shape
    x0 = (width - 1) / 2.0
    y0 = (height - 1) / 2.0
    # HSC cutouts are ~1192 px per side; leave a 10-pixel buffer so the
    # outward walk does not sample partially outside the frame.
    maxsma = float(min(height, width) / 2.0 - 10.0)

    return GalaxyData(
        spec=spec,
        image=image,
        mask=mask,
        variance_map=variance,
        fiducial_sma0=_HSC_FIDUCIAL_SMA0,
        fiducial_eps=_HSC_FIDUCIAL_EPS,
        fiducial_pa=_HSC_FIDUCIAL_PA,
        fiducial_x0=x0,
        fiducial_y0=y0,
        maxsma=maxsma,
        minsma=0.0,
        truth=None,
        # No overrides: run_sweep.BASE_CONFIG already matches the HSC
        # LSB-sweep BASE_CONFIG on every shared knob (astep, sclip, etc.).
    )


# ---------------------------------------------------------------------------
# Tier dispatch
# ---------------------------------------------------------------------------


def list_galaxies(tier: str) -> List[GalaxySpec]:
    """Return the list of galaxies registered for a tier."""
    if tier == "mocks":
        return _list_mocks()
    if tier == "huang2013":
        return _list_huang2013()
    if tier == "highorder":
        return _list_highorder()
    if tier == "hsc":
        return _list_hsc()
    raise ValueError(f"unknown tier: {tier!r} (valid: {TIERS})")


def load_galaxy(spec: GalaxySpec) -> GalaxyData:
    """Materialize a galaxy's image, mask, variance, and fiducial config."""
    if spec.tier == "mocks":
        return _load_mock(spec)
    if spec.tier == "huang2013":
        return _load_huang2013(spec)
    if spec.tier == "highorder":
        return _load_highorder(spec)
    if spec.tier == "hsc":
        return _load_hsc(spec)
    raise NotImplementedError(
        f"loader for tier {spec.tier!r} is not implemented yet"
    )
