"""Per-galaxy per-arm QA wrapper.

Thin adapter around :func:`isoster.plotting.plot_qa_summary` that
supplies campaign-standard kwargs: surface-brightness zeropoint +
pixel scale come from the galaxy metadata so every arm renders on the
same physical axes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from isoster import build_isoster_model
from isoster.plotting import plot_qa_summary

from ..adapters.base import GalaxyBundle


def render_per_arm_qa(
    bundle: GalaxyBundle,
    arm_id: str,
    results: dict[str, Any],
    output_path: Path,
    *,
    relative_residual: bool = False,
) -> None:
    """Write a 6-panel QA PNG for one ``(galaxy, arm)`` pair.

    The isoster 2D model is built from ``results['isophotes']``; the
    residual panel shows ``image - model`` (or fractional residual
    when ``relative_residual=True``).
    """
    image = np.asarray(bundle.image, dtype=np.float64)
    isophotes = results.get("isophotes", [])
    model = build_isoster_model(image.shape, isophotes)
    title = f"{bundle.metadata.galaxy_id}  |  {arm_id}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_qa_summary(
        title=title,
        image=image,
        isoster_model=model,
        isoster_res=isophotes,
        mask=bundle.mask,
        filename=str(output_path),
        relative_residual=relative_residual,
        sb_zeropoint=bundle.metadata.sb_zeropoint,
        pixel_scale_arcsec=bundle.metadata.pixel_scale_arcsec,
    )


def build_model_cube(
    bundle: GalaxyBundle, results: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(model, residual)`` 2D arrays for FITS output."""
    image = np.asarray(bundle.image, dtype=np.float64)
    isophotes = results.get("isophotes", [])
    model = build_isoster_model(image.shape, isophotes)
    residual = image - model
    return model, residual
