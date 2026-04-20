"""DatasetAdapter protocol and supporting dataclasses.

Every dataset (Huang2013 mocks, HSC edge-real BCGs, SGA cutouts, ...)
implements this protocol so the orchestrator can iterate galaxies
without knowing the on-disk layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class GalaxyMetadata:
    """Dataset-supplied static facts about one galaxy.

    ``effective_Re_pix`` may be ``None`` when the dataset does not
    provide a half-light radius; adaptive-integrator arms are then
    skipped for the galaxy.
    """

    galaxy_id: str
    dataset: str
    pixel_scale_arcsec: float
    sb_zeropoint: float
    effective_Re_pix: float | None = None
    redshift: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GalaxyBundle:
    """Everything a fitter needs to process one galaxy.

    ``variance`` and ``mask`` are optional; ``mask`` follows the
    isoster convention ``True`` = masked / bad pixel.

    ``initial_geometry`` is a dict with keys ``x0, y0, eps, pa, sma0,
    maxsma``; all floats (pixel units; ``pa`` in radians to match
    ``IsosterConfig``).
    """

    metadata: GalaxyMetadata
    image: Any              # np.ndarray, 2D float64
    variance: Any | None    # np.ndarray or None
    mask: Any | None        # np.ndarray (bool) or None
    initial_geometry: dict[str, float]
    truth_profile: dict[str, Any] | None = None  # set when mocks expose ground truth


class DatasetAdapter(Protocol):
    """Protocol every per-dataset adapter must implement.

    Adapters are cheap to construct (do not load all galaxies at
    init); ``load_galaxy`` does the I/O on demand so the orchestrator
    can parallelize across galaxies.
    """

    dataset_name: str

    def list_galaxies(self) -> list[str]:
        """Return a stable-ordered list of ``galaxy_id`` strings found
        under the adapter's data root."""
        ...

    def load_galaxy(self, galaxy_id: str) -> GalaxyBundle:
        """Load one galaxy's image / variance / mask / initial geometry.

        Must raise a clear exception with the missing path when the
        files cannot be found; the orchestrator records the error in
        the inventory and moves on."""
        ...


def safe_galaxy_id(galaxy_id: str) -> str:
    """Sanitize a galaxy_id for use in filesystem paths.

    Replaces path separators with ``__`` so a nested id like
    ``NGC_0596/mock001`` becomes a single output directory name
    ``NGC_0596__mock001``.
    """
    return galaxy_id.replace("/", "__").replace("\\", "__")


def expand_root(root: str | Path) -> Path:
    """Expand ``~`` and environment variables, return absolute Path."""
    import os

    return Path(os.path.expandvars(str(root))).expanduser().resolve()
