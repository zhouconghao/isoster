"""Local-FITS adapter for explicit paths declared in the campaign YAML.

Use this when you have a handful of already-downloaded FITS files and
don't want to impose a ``<galaxy>/<galaxy>_mockNNN.fits`` directory
convention. Each entry in ``datasets.local_fits.files`` is either a
bare path (``str``) or a mapping::

    datasets:
      local_fits:
        enabled: true
        adapter: local_fits
        root: "data"                    # optional; prepended to relative paths
        pixel_scale_arcsec: 0.168       # optional; default from FITS header PIXSCALE
        sb_zeropoint: 27.0              # optional; default from FITS header MAGZERO
        files:
          - IC3370_mock2.fits
          - path: ngc3610.fits
            galaxy_id: ngc3610_r
            band_index: 1                # 3D cube: pick band 1 (r)
            variance: null               # optional
            mask: null                   # optional
            eps: 0.3                     # optional override
            pa_deg: 45.0                 # optional (degrees; converted to radians)
            sma0: 6.0                    # optional
            maxsma: 100.0                # optional

Header parsing mirrors the Huang2013 adapter when ``NCOMP`` is present
(flux-weighted ELLIP / PA / RE_PX), otherwise falls back to image
center + per-entry overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from .base import DatasetAdapter, GalaxyBundle, GalaxyMetadata, expand_root
from .huang2013 import DEFAULT_INITIAL_SMA_PIX, HUANG2013_PA_OFFSET_DEG


@dataclass
class LocalFitsEntry:
    path: Path
    galaxy_id: str
    band_index: int = 0
    variance_path: Path | None = None
    mask_path: Path | None = None
    pixel_scale_arcsec: float | None = None
    sb_zeropoint: float | None = None
    overrides: dict[str, float] = field(default_factory=dict)


class LocalFitsAdapter:
    """Adapter reading explicit FITS paths listed in the campaign YAML."""

    dataset_name = "local_fits"

    def __init__(
        self,
        root: str | Path | None = None,
        files: list[Any] | None = None,
        pixel_scale_arcsec: float | None = None,
        sb_zeropoint: float | None = None,
    ) -> None:
        if files is None:
            raise ValueError(
                "LocalFitsAdapter requires a `files:` list in the dataset YAML "
                "(each entry is a path or a mapping with 'path' + optional keys)."
            )
        self.root = expand_root(root) if root is not None else Path.cwd()
        self.default_pixel_scale = pixel_scale_arcsec
        self.default_sb_zeropoint = sb_zeropoint
        self._entries = [
            self._normalize_entry(entry, index) for index, entry in enumerate(files)
        ]

        # Fail fast: every file must exist now so `list_galaxies` returns a
        # trustworthy set for dry-run.
        for entry in self._entries:
            if not entry.path.is_file():
                raise FileNotFoundError(
                    f"local_fits: path does not exist: {entry.path} "
                    f"(resolved from root={self.root})"
                )

    def list_galaxies(self) -> list[str]:
        return [entry.galaxy_id for entry in self._entries]

    def load_galaxy(self, galaxy_id: str) -> GalaxyBundle:
        entry = self._find_entry(galaxy_id)
        image, header = _load_image(entry.path, entry.band_index)
        variance = _load_optional(entry.variance_path, entry.band_index) \
            if entry.variance_path is not None else None
        mask = _load_optional_bool(entry.mask_path, entry.band_index) \
            if entry.mask_path is not None else None

        pixel_scale = (
            entry.pixel_scale_arcsec
            or self.default_pixel_scale
            or _maybe_float(header, "PIXSCALE")
            or 0.168
        )
        sb_zeropoint = (
            entry.sb_zeropoint
            or self.default_sb_zeropoint
            or _maybe_float(header, "MAGZERO")
            or 27.0
        )

        initial_geometry = self._infer_geometry(header, image.shape, entry.overrides)
        effective_Re = self._infer_effective_Re(header, pixel_scale)
        # User overrides always win.
        if "Re_pix" in entry.overrides:
            effective_Re = float(entry.overrides["Re_pix"])

        metadata = GalaxyMetadata(
            galaxy_id=entry.galaxy_id,
            dataset=self.dataset_name,
            pixel_scale_arcsec=float(pixel_scale),
            sb_zeropoint=float(sb_zeropoint),
            effective_Re_pix=effective_Re,
            redshift=_maybe_float(header, "REDSHIFT"),
            extra={
                "fits_path": str(entry.path),
                "band_index": entry.band_index,
                "object": str(header.get("OBJECT", "")),
                "ncomp": int(header.get("NCOMP", 0)),
            },
        )
        return GalaxyBundle(
            metadata=metadata,
            image=image,
            variance=variance,
            mask=mask,
            initial_geometry=initial_geometry,
            truth_profile=None,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _normalize_entry(self, raw: Any, index: int) -> LocalFitsEntry:
        if isinstance(raw, str):
            raw = {"path": raw}
        if not isinstance(raw, dict):
            raise TypeError(
                f"local_fits file entry #{index} must be a path string or mapping, "
                f"got {type(raw).__name__}"
            )
        path_raw = raw.get("path")
        if path_raw is None:
            raise ValueError(
                f"local_fits file entry #{index} missing required 'path' field"
            )
        path = Path(str(path_raw)).expanduser()
        if not path.is_absolute():
            path = (self.root / path).resolve()

        galaxy_id = str(raw.get("galaxy_id") or path.stem)
        band_index = int(raw.get("band_index", 0))

        variance_path = _resolve_optional_path(raw.get("variance"), self.root)
        mask_path = _resolve_optional_path(raw.get("mask"), self.root)

        overrides: dict[str, float] = {}
        for key in ("eps", "sma0", "maxsma", "Re_pix"):
            if raw.get(key) is not None:
                overrides[key] = float(raw[key])
        if raw.get("pa_deg") is not None:
            overrides["pa"] = float(np.deg2rad(float(raw["pa_deg"])))
        if raw.get("pa") is not None:
            # explicit radians
            overrides["pa"] = float(raw["pa"])
        if raw.get("x0") is not None:
            overrides["x0"] = float(raw["x0"])
        if raw.get("y0") is not None:
            overrides["y0"] = float(raw["y0"])

        return LocalFitsEntry(
            path=path,
            galaxy_id=galaxy_id,
            band_index=band_index,
            variance_path=variance_path,
            mask_path=mask_path,
            pixel_scale_arcsec=_optional_float(raw.get("pixel_scale_arcsec")),
            sb_zeropoint=_optional_float(raw.get("sb_zeropoint")),
            overrides=overrides,
        )

    def _find_entry(self, galaxy_id: str) -> LocalFitsEntry:
        for entry in self._entries:
            if entry.galaxy_id == galaxy_id:
                return entry
        raise KeyError(f"local_fits: galaxy_id {galaxy_id!r} not found")

    def _infer_geometry(
        self,
        header: fits.Header,
        image_shape: tuple[int, int],
        overrides: dict[str, float],
    ) -> dict[str, float]:
        """Start from image-center defaults, refine with Huang2013-style
        component headers when present, then apply per-entry overrides."""
        center_x = (image_shape[1] - 1) / 2.0
        center_y = (image_shape[0] - 1) / 2.0
        eps = 0.2
        pa_rad = 0.0

        ncomp = int(header.get("NCOMP", 0))
        if ncomp > 0:
            eps_values: list[float] = []
            pa_values: list[float] = []
            weight_values: list[float] = []
            for i in range(1, ncomp + 1):
                if f"ELLIP{i}" not in header or f"PA{i}" not in header:
                    continue
                eps_values.append(float(header[f"ELLIP{i}"]))
                pa_values.append(float(header[f"PA{i}"]))
                if f"APPMAG{i}" in header:
                    weight_values.append(10.0 ** (-0.4 * float(header[f"APPMAG{i}"])))
                else:
                    weight_values.append(1.0)
            if eps_values:
                weights = np.asarray(weight_values, dtype=float)
                weights /= float(weights.sum())
                eps = float(np.clip(np.sum(np.asarray(eps_values) * weights), 0.0, 0.95))
                pa_array_rad = np.deg2rad(np.asarray(pa_values, dtype=float))
                pa_deg = np.rad2deg(
                    np.arctan2(
                        np.sum(weights * np.sin(2.0 * pa_array_rad)),
                        np.sum(weights * np.cos(2.0 * pa_array_rad)),
                    )
                    / 2.0
                )
                pa_rad = float(np.deg2rad(pa_deg + HUANG2013_PA_OFFSET_DEG))

        sma0 = float(max(DEFAULT_INITIAL_SMA_PIX, 3.0))
        maxsma = float(min(image_shape) // 2)

        geometry = {
            "x0": float(center_x),
            "y0": float(center_y),
            "eps": eps,
            "pa": pa_rad,
            "sma0": sma0,
            "maxsma": maxsma,
        }
        geometry.update(overrides)
        return geometry

    def _infer_effective_Re(
        self, header: fits.Header, pixel_scale_arcsec: float
    ) -> float | None:
        for key in ("REHALF", "R_E_PX", "REFF_PIX", "RE_PIX", "SMA_E"):
            if key in header:
                try:
                    value = float(header[key])
                except (TypeError, ValueError):
                    continue
                if value > 0.0 and np.isfinite(value):
                    return value
        ncomp = int(header.get("NCOMP", 0))
        if ncomp > 0:
            re_values: list[float] = []
            weights: list[float] = []
            for i in range(1, ncomp + 1):
                re_key = f"RE_PX{i}"
                if re_key not in header:
                    continue
                try:
                    re_value = float(header[re_key])
                except (TypeError, ValueError):
                    continue
                if not (re_value > 0.0 and np.isfinite(re_value)):
                    continue
                re_values.append(re_value)
                if f"APPMAG{i}" in header:
                    try:
                        weights.append(10.0 ** (-0.4 * float(header[f"APPMAG{i}"])))
                    except (TypeError, ValueError):
                        weights.append(1.0)
                else:
                    weights.append(1.0)
            if re_values:
                weight_array = np.asarray(weights, dtype=float)
                weight_array /= float(weight_array.sum())
                return float(np.sum(np.asarray(re_values) * weight_array))
        return None


def _load_image(path: Path, band_index: int) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
        header = hdul[0].header.copy()
    if data.ndim == 2:
        return data, header
    if data.ndim == 3:
        if band_index < 0 or band_index >= data.shape[0]:
            raise IndexError(
                f"{path}: band_index={band_index} out of range for cube shape {data.shape}"
            )
        return data[band_index], header
    raise ValueError(f"{path}: unsupported image rank {data.ndim}")


def _load_optional(path: Path, band_index: int) -> np.ndarray:
    data, _ = _load_image(path, band_index)
    return data


def _load_optional_bool(path: Path, band_index: int) -> np.ndarray:
    data, _ = _load_image(path, band_index)
    return data.astype(bool)


def _resolve_optional_path(raw: Any, root: Path) -> Path | None:
    if raw is None:
        return None
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (root / path).resolve()
    return path


def _optional_float(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _maybe_float(header: fits.Header, key: str) -> float | None:
    if key not in header:
        return None
    try:
        return float(header[key])
    except (TypeError, ValueError):
        return None


ADAPTER_CLASS: type[DatasetAdapter] = LocalFitsAdapter
