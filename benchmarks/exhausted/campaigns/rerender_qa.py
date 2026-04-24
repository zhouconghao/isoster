"""Bulk re-render of campaign QA figures from existing profile / model FITS.

No fitting is re-run. For every galaxy on disk under a campaign root,
this driver re-emits:

- ``<galaxy>/<tool>/arms/<arm>/qa.png`` from the arm's
  ``profile.fits`` + ``model.fits`` + the galaxy's source image.
- ``<galaxy>/<tool>/cross_arm_overlay.png`` from the tool's
  ``inventory.fits`` (which already lists each arm's profile path).
- ``<galaxy>/cross/cross_tool_comparison.png`` from each tool's
  default-arm ``profile.fits`` + ``model.fits``.

The driver is intended for cases where only the *plotting* code
changed (e.g. a new harmonic convention). The profiles themselves
are not touched.

Existing output files are overwritten in place. Any profile that
fails to load is logged and skipped; the rest of the galaxy still
renders. A per-galaxy failure record is written next to the figure
as ``<fig>.err.txt``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.table import Table

# Safety: headless matplotlib.
import matplotlib
matplotlib.use("Agg")

from isoster import build_isoster_model
from isoster.plotting import (
    build_method_profile,
    plot_comparison_qa_figure,
    plot_qa_summary,
)

from benchmarks.exhausted.plotting.cross_arm_overlay import plot_cross_arm_overlay
from benchmarks.exhausted.plotting.cross_tool_comparison import DEFAULT_ARM_PER_TOOL


TOOLS = ("isoster", "photutils", "autoprof")


@dataclass(frozen=True)
class GalaxyJob:
    """All inputs needed to rebuild one galaxy's figures."""

    galaxy_dir: Path

    @property
    def galaxy_tag(self) -> str:
        return self.galaxy_dir.name


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_manifest(galaxy_dir: Path) -> dict[str, Any]:
    with (galaxy_dir / "MANIFEST.json").open() as handle:
        return json.load(handle)


def _load_profile_as_iso_list(path: Path) -> list[dict[str, Any]] | None:
    """Return per-row dicts (the shape ``plot_qa_summary`` expects for isoster/photutils)."""
    try:
        with fits.open(path) as hdul:
            tbl = _find_table_hdu(hdul)
            if tbl is None:
                return None
            t = Table(tbl.data)
    except (OSError, ValueError):
        return None
    return [{name: row[name] for name in t.colnames} for row in t]


def _load_profile_as_dict(path: Path) -> dict[str, np.ndarray] | None:
    """Return col->array (the shape ``build_method_profile`` wants for
    cross-tool comparison's autoprof path)."""
    try:
        with fits.open(path) as hdul:
            tbl = _find_table_hdu(hdul)
            if tbl is None:
                return None
            return {c.lower(): np.asarray(tbl.data[c]) for c in tbl.data.names}
    except (OSError, ValueError):
        return None


def _find_table_hdu(hdul):
    for hdu in hdul:
        if getattr(hdu, "name", "").upper() == "ISOPHOTES" and hasattr(hdu, "data"):
            return hdu
    for hdu in hdul[1:]:
        if hasattr(hdu, "data") and hdu.data is not None:
            return hdu
    return None


def _load_image(manifest: dict[str, Any]) -> np.ndarray | None:
    """Pull the raw image path from ``manifest.extra.fits_path`` and load it."""
    extra = manifest.get("extra") or {}
    path = extra.get("fits_path")
    if not path:
        return None
    p = Path(str(path))
    if not p.is_file():
        return None
    try:
        with fits.open(p) as hdul:
            data = hdul[0].data
        return np.asarray(data, dtype=np.float64)
    except (OSError, ValueError):
        return None


def _load_model(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        with fits.open(path) as hdul:
            data = hdul[0].data
        return np.asarray(data, dtype=np.float64) if data is not None else None
    except (OSError, ValueError):
        return None


def _load_inventory_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        with fits.open(path) as hdul:
            t = Table(hdul[1].data)
    except (OSError, ValueError):
        return []
    return [{name: row[name] for name in t.colnames} for row in t]


# ---------------------------------------------------------------------------
# Per-figure renderers (each returns (rendered, errored) counts)
# ---------------------------------------------------------------------------


def _rebuild_arm_qa(
    arm_dir: Path,
    *,
    image: np.ndarray,
    manifest: dict[str, Any],
    tool: str,
    galaxy_id: str,
    arm_id: str,
) -> tuple[int, int]:
    profile_path = arm_dir / "profile.fits"
    if not profile_path.is_file():
        return 0, 0
    iso = _load_profile_as_iso_list(profile_path)
    if not iso:
        return 0, 0
    # Reconstruct isoster-style 2-D model: prefer on-disk model.fits so
    # photutils/autoprof variants keep the tool's own model convention.
    model = _load_model(arm_dir / "model.fits")
    if model is None:
        # Fallback: synthesize from the isophote list (isoster only).
        if tool == "isoster":
            model = build_isoster_model(image.shape, iso)
        else:
            return 0, 0
    qa_path = arm_dir / "qa.png"
    title = f"{galaxy_id}  |  {tool}  |  {arm_id}"
    try:
        plot_qa_summary(
            title=title,
            image=image,
            isoster_model=model,
            isoster_res=iso,
            filename=str(qa_path),
            relative_residual=False,
            sb_zeropoint=float(manifest["sb_zeropoint"]),
            pixel_scale_arcsec=float(manifest["pixel_scale_arcsec"]),
        )
        return 1, 0
    except Exception as exc:  # noqa: BLE001 — per-arm render isolation
        qa_path.with_suffix(".png.err.txt").write_text(
            f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
        )
        return 0, 1


def _rebuild_cross_arm_overlay(
    tool_dir: Path,
    *,
    manifest: dict[str, Any],
    tool: str,
    galaxy_id: str,
) -> tuple[int, int]:
    inv_path = tool_dir / "inventory.fits"
    rows = _load_inventory_rows(inv_path)
    if not rows:
        return 0, 0
    overlay_path = tool_dir / "cross_arm_overlay.png"
    try:
        plot_cross_arm_overlay(
            rows,
            overlay_path,
            sb_zeropoint=float(manifest["sb_zeropoint"]),
            pixel_scale_arcsec=float(manifest["pixel_scale_arcsec"]),
            title=f"{galaxy_id}  |  {tool}",
            tool_name=tool,
        )
        return 1, 0
    except Exception as exc:  # noqa: BLE001
        overlay_path.with_suffix(".png.err.txt").write_text(
            f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
        )
        return 0, 1


def _rebuild_cross_tool(
    galaxy_dir: Path,
    *,
    image: np.ndarray,
    manifest: dict[str, Any],
    galaxy_id: str,
) -> tuple[int, int]:
    profiles: dict[str, dict[str, np.ndarray] | None] = {}
    for tool in TOOLS:
        arm = DEFAULT_ARM_PER_TOOL.get(tool)
        if arm is None:
            continue
        p = galaxy_dir / tool / "arms" / arm / "profile.fits"
        if not p.is_file():
            profiles[tool] = None
            continue
        data = _load_profile_as_dict(p)
        if data is None:
            profiles[tool] = None
            continue
        profiles[tool] = build_method_profile(data)
    # Keep only tools that have a usable profile.
    usable = {k: v for k, v in profiles.items() if v is not None}
    if not usable:
        return 0, 0
    cross_dir = galaxy_dir / "cross"
    cross_dir.mkdir(exist_ok=True)
    out = cross_dir / "cross_tool_comparison.png"
    try:
        plot_comparison_qa_figure(
            image=image,
            profiles=usable,
            title=galaxy_id,
            output_path=str(out),
            relative_residual=False,
        )
        return 1, 0
    except Exception as exc:  # noqa: BLE001
        out.with_suffix(".png.err.txt").write_text(
            f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
        )
        return 0, 1


# ---------------------------------------------------------------------------
# Per-galaxy pipeline (unit of parallelism)
# ---------------------------------------------------------------------------


@dataclass
class GalaxyStats:
    galaxy_tag: str
    qa_ok: int = 0
    qa_err: int = 0
    overlay_ok: int = 0
    overlay_err: int = 0
    cross_ok: int = 0
    cross_err: int = 0
    fatal: str = ""

    def render_summary(self) -> str:
        return (
            f"{self.galaxy_tag}: qa {self.qa_ok}/{self.qa_ok + self.qa_err}  "
            f"overlay {self.overlay_ok}/{self.overlay_ok + self.overlay_err}  "
            f"cross {self.cross_ok}/{self.cross_ok + self.cross_err}"
            + (f"  FATAL: {self.fatal}" if self.fatal else "")
        )


def rebuild_one_galaxy(galaxy_dir: Path) -> GalaxyStats:
    stats = GalaxyStats(galaxy_tag=galaxy_dir.name)
    try:
        manifest = _load_manifest(galaxy_dir)
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        stats.fatal = f"manifest load failed: {exc}"
        return stats
    image = _load_image(manifest)
    if image is None:
        stats.fatal = "source image missing or unreadable"
        return stats
    galaxy_id = manifest["galaxy_id"]

    # Per-arm qa.png for every tool.
    for tool in TOOLS:
        arms_dir = galaxy_dir / tool / "arms"
        if not arms_dir.is_dir():
            continue
        for arm_dir in sorted(arms_dir.iterdir()):
            if not arm_dir.is_dir():
                continue
            ok, err = _rebuild_arm_qa(
                arm_dir,
                image=image,
                manifest=manifest,
                tool=tool,
                galaxy_id=galaxy_id,
                arm_id=arm_dir.name,
            )
            stats.qa_ok += ok
            stats.qa_err += err

        ok, err = _rebuild_cross_arm_overlay(
            galaxy_dir / tool,
            manifest=manifest,
            tool=tool,
            galaxy_id=galaxy_id,
        )
        stats.overlay_ok += ok
        stats.overlay_err += err

    ok, err = _rebuild_cross_tool(
        galaxy_dir,
        image=image,
        manifest=manifest,
        galaxy_id=galaxy_id,
    )
    stats.cross_ok += ok
    stats.cross_err += err
    return stats


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _enumerate_galaxies(
    campaign_root: Path, dataset: str, only: list[str] | None
) -> list[Path]:
    """Every galaxy dir across all campaign subdirs.

    The campaign tree is ``<root>/huang2013_*_z*/<dataset>/<galaxy>`` for
    huang2013; generalizing by globbing ``*/<dataset>/*``.
    """
    galaxies: list[Path] = []
    for campaign_dir in sorted(campaign_root.iterdir()):
        if not campaign_dir.is_dir():
            continue
        if campaign_dir.name.startswith("_"):
            continue  # skip _analysis etc.
        ds_dir = campaign_dir / dataset
        if not ds_dir.is_dir():
            continue
        for gdir in sorted(ds_dir.iterdir()):
            if not gdir.is_dir() or "__" not in gdir.name:
                continue
            if only and gdir.name not in only:
                continue
            galaxies.append(gdir)
    return galaxies


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rerender_qa",
        description="Bulk re-emit campaign QA figures from existing profile FITS.",
    )
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=Path("/Volumes/galaxy/isophote/huang2013/_campaigns"),
        help="Directory containing huang2013_*_z* subdirs.",
    )
    parser.add_argument("--dataset", default="huang2013")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional filter: only re-render these galaxy directory names "
        "(e.g. IC1459__clean_z005).",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help="Number of galaxy workers (default: half of CPU count).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print an aggregate status line every N galaxies.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Enumerate galaxies and exit without rendering.",
    )
    args = parser.parse_args(argv)

    galaxies = _enumerate_galaxies(args.campaign_root, args.dataset, args.only)
    print(
        f"Found {len(galaxies)} galaxies under {args.campaign_root} / "
        f"{args.dataset}",
        flush=True,
    )
    if args.dry_run:
        for g in galaxies[:10]:
            print(f"  {g}")
        if len(galaxies) > 10:
            print(f"  ... and {len(galaxies) - 10} more")
        return 0
    if not galaxies:
        return 1

    total = len(galaxies)
    done = 0
    totals = GalaxyStats(galaxy_tag="TOTAL")
    any_fatal = False
    with ProcessPoolExecutor(max_workers=max(1, args.max_parallel)) as ex:
        future_to_tag = {
            ex.submit(rebuild_one_galaxy, g): g.name for g in galaxies
        }
        for fut in as_completed(future_to_tag):
            tag = future_to_tag[fut]
            try:
                s = fut.result()
            except Exception as exc:  # noqa: BLE001
                s = GalaxyStats(galaxy_tag=tag, fatal=f"worker crash: {exc}")
            totals.qa_ok += s.qa_ok
            totals.qa_err += s.qa_err
            totals.overlay_ok += s.overlay_ok
            totals.overlay_err += s.overlay_err
            totals.cross_ok += s.cross_ok
            totals.cross_err += s.cross_err
            if s.fatal:
                any_fatal = True
                print(f"  [FATAL] {s.render_summary()}", file=sys.stderr, flush=True)
            done += 1
            if done % args.progress_every == 0 or done == total:
                print(
                    f"  [{done}/{total}] qa {totals.qa_ok}  overlay "
                    f"{totals.overlay_ok}  cross {totals.cross_ok}  "
                    f"errors qa={totals.qa_err} ov={totals.overlay_err} "
                    f"cr={totals.cross_err}",
                    flush=True,
                )

    print("=" * 60)
    print(f"Re-rendered {done} galaxies from {args.campaign_root}")
    print(
        f"qa.png:                {totals.qa_ok} ok, {totals.qa_err} errors"
    )
    print(
        f"cross_arm_overlay.png: {totals.overlay_ok} ok, {totals.overlay_err} errors"
    )
    print(
        f"cross_tool_comparison.png: {totals.cross_ok} ok, {totals.cross_err} errors"
    )
    return 0 if (totals.qa_err + totals.overlay_err + totals.cross_err == 0
                 and not any_fatal) else 1


if __name__ == "__main__":
    raise SystemExit(main())
