"""Huang2013 multi-component Sersic mock adapter.

Expected on-disk layout (configurable via the campaign YAML ``root``):

    <root>/
      <galaxy_name>/
        <galaxy_name>_mock001.fits
        <galaxy_name>_mock002.fits
        ...

Each FITS is a single-HDU image with header keys describing the
multi-component Sersic truth: ``NCOMP``, ``ELLIP{i}``, ``PA{i}``,
``APPMAG{i}``, and (when present) a half-light radius key we try to
read in that priority order: ``REHALF``, ``R_E_PX``, ``REFF_PIX``,
``RE_PIX``, ``REKPC`` combined with pixel scale.

The mocks do not ship a variance map or an object mask; those are
returned as ``None`` so the fitter falls back to OLS and no-mask.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from .base import DatasetAdapter, GalaxyBundle, GalaxyMetadata, expand_root

HUANG2013_PA_OFFSET_DEG = -90.0
DEFAULT_INITIAL_SMA_PIX = 6.0
DEFAULT_PIXEL_SCALE_ARCSEC = 0.168   # HSC coadd convention, matches CLAUDE.md
DEFAULT_ZEROPOINT = 27.0              # HSC coadd convention


class Huang2013Adapter:
    """Adapter for Huang (2013) multi-Sersic mocks."""

    dataset_name = "huang2013"

    def __init__(
        self,
        root: str | Path,
        pixel_scale_arcsec: float = DEFAULT_PIXEL_SCALE_ARCSEC,
        sb_zeropoint: float = DEFAULT_ZEROPOINT,
    ) -> None:
        self.root = expand_root(root)
        self.pixel_scale_arcsec = pixel_scale_arcsec
        self.sb_zeropoint = sb_zeropoint

        if not self.root.is_dir():
            raise FileNotFoundError(
                f"Huang2013 root not found: {self.root}\n"
                f"Expected layout: <root>/<galaxy>/<galaxy>_mockNNN.fits\n"
                f"Set `datasets.huang2013.root` in the campaign YAML "
                f"to the directory containing per-galaxy subdirectories."
            )

    def list_galaxies(self) -> list[str]:
        """Return ``<galaxy>/mockNNN`` identifiers sorted lexically."""
        galaxy_ids: list[str] = []
        for galaxy_dir in sorted(self.root.iterdir()):
            if not galaxy_dir.is_dir() or galaxy_dir.name.startswith("."):
                continue
            for mock_path in sorted(galaxy_dir.glob(f"{galaxy_dir.name}_mock*.fits")):
                mock_id = mock_path.stem.split("_mock", 1)[1]
                galaxy_ids.append(f"{galaxy_dir.name}/mock{mock_id}")
        return galaxy_ids

    def load_galaxy(self, galaxy_id: str) -> GalaxyBundle:
        fits_path = self._resolve_fits_path(galaxy_id)
        with fits.open(fits_path) as hdul:
            image = np.asarray(hdul[0].data, dtype=np.float64)
            header = hdul[0].header.copy()
        if image.ndim != 2:
            raise ValueError(f"{fits_path}: expected 2D image, got shape {image.shape}")

        initial_geometry = self._infer_initial_geometry(header, image.shape)
        effective_Re = self._read_effective_Re_pix(header)

        metadata = GalaxyMetadata(
            galaxy_id=galaxy_id,
            dataset=self.dataset_name,
            pixel_scale_arcsec=self.pixel_scale_arcsec,
            sb_zeropoint=self.sb_zeropoint,
            effective_Re_pix=effective_Re,
            redshift=_maybe_float(header, "REDSHIFT"),
            extra={
                "ncomp": int(header.get("NCOMP", 0)),
                "fits_path": str(fits_path),
            },
        )
        return GalaxyBundle(
            metadata=metadata,
            image=image,
            variance=None,
            mask=None,
            initial_geometry=initial_geometry,
            truth_profile=None,  # Phase C writes a reader when needed
        )

    def _resolve_fits_path(self, galaxy_id: str) -> Path:
        try:
            galaxy_name, mock_token = galaxy_id.split("/", 1)
        except ValueError as exc:
            raise ValueError(
                f"Huang2013 galaxy_id must be '<galaxy>/mockNNN', got '{galaxy_id}'"
            ) from exc
        if not mock_token.startswith("mock"):
            raise ValueError(
                f"Huang2013 mock token must start with 'mock', got '{mock_token}'"
            )
        mock_id = mock_token[len("mock"):]
        fits_path = self.root / galaxy_name / f"{galaxy_name}_mock{mock_id}.fits"
        if not fits_path.is_file():
            raise FileNotFoundError(f"Huang2013 mock FITS not found: {fits_path}")
        return fits_path

    def _infer_initial_geometry(
        self, header: fits.Header, image_shape: tuple[int, int]
    ) -> dict[str, float]:
        """Flux-weighted average of per-component ELLIP/PA, center at image middle.

        Mirrors ``examples/example_huang2013/huang2013_shared.py:infer_initial_geometry``
        so the campaign stays consistent with prior Huang2013 runs.
        """
        center_x = (image_shape[1] - 1) / 2.0
        center_y = (image_shape[0] - 1) / 2.0
        component_count = int(header.get("NCOMP", 0))

        eps_values: list[float] = []
        pa_values: list[float] = []
        weight_values: list[float] = []
        for i in range(1, component_count + 1):
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
            weights /= np.sum(weights)
            eps_initial = float(np.sum(np.asarray(eps_values) * weights))
            pa_rad = np.deg2rad(np.asarray(pa_values, dtype=float))
            pa_initial_deg = np.rad2deg(
                np.arctan2(
                    np.sum(weights * np.sin(2.0 * pa_rad)),
                    np.sum(weights * np.cos(2.0 * pa_rad)),
                )
                / 2.0
            )
        else:
            eps_initial = float(header.get("ELLIP1", 0.2))
            pa_initial_deg = float(header.get("PA1", 0.0))

        pa_initial_deg = float(pa_initial_deg + HUANG2013_PA_OFFSET_DEG)
        eps_initial = float(np.clip(eps_initial, 0.0, 0.95))

        sma0 = float(max(DEFAULT_INITIAL_SMA_PIX, 3.0))
        maxsma = float(min(image_shape) // 2)

        return {
            "x0": center_x,
            "y0": center_y,
            "eps": eps_initial,
            "pa": float(np.deg2rad(pa_initial_deg)),
            "sma0": sma0,
            "maxsma": maxsma,
        }

    def _read_effective_Re_pix(self, header: fits.Header) -> float | None:
        """Try several header-key conventions before giving up.

        The Huang2013 mock headers do not carry a uniform Re key; when
        all candidates are missing we return ``None`` so the runner
        skips the adaptive-integrator arms for this galaxy.
        """
        for key in ("REHALF", "R_E_PX", "REFF_PIX", "RE_PIX", "SMA_E"):
            if key in header:
                try:
                    value = float(header[key])
                except (TypeError, ValueError):
                    continue
                if value > 0.0 and np.isfinite(value):
                    return value
        if "REKPC" in header and "REDSHIFT" in header:
            try:
                from astropy import units
                from astropy.cosmology import Planck18
                re_kpc = float(header["REKPC"])
                z = float(header["REDSHIFT"])
                kpc_per_arcsec = Planck18.kpc_proper_per_arcmin(z).to_value(
                    units.kpc / units.arcsec
                )
                re_arcsec = re_kpc / kpc_per_arcsec
                return float(re_arcsec / self.pixel_scale_arcsec)
            except Exception:  # noqa: BLE001 - last-resort fallback
                return None
        return None


def _maybe_float(header: fits.Header, key: str) -> float | None:
    if key not in header:
        return None
    try:
        return float(header[key])
    except (TypeError, ValueError):
        return None


# Register with the adapter name used in the campaign YAML.
ADAPTER_CLASS: type[DatasetAdapter] = Huang2013Adapter
