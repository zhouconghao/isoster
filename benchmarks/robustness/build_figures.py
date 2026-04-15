#!/usr/bin/env python3
"""Build the robustness figures from persisted sweep outputs.

Reads the FITS tables written by ``run_sweep.py`` (via
``benchmarks.robustness.persist``) plus the scalar metrics in each
tier's ``results.json`` and produces two families of figures under
``outputs/benchmark_robustness/{tier}/figures/``:

1. **Profile overlays** (``{tier}/figures/profiles/{galaxy}/{arm}_{axis}.png``):
   2×3 panel figure showing ``intens(sma)``, ``(1-eps)(sma)``,
   ``pa(sma)``, ``x0 - x0_ref``, ``y0 - y0_ref``, and center-drift
   magnitude. All perturbations are plotted; inliers are drawn as
   thin grey traces in the background while outliers (top-N by
   ``profile_rel_rms``) are promoted to saturated viridis colors,
   thicker linewidth, and carry labels. Reference is overlaid as a
   thick black line on every panel.

2. **Outlier QA figures**
   (``{tier}/figures/outliers/{galaxy}/{arm}_{axis}_{value}_qa.png``):
   3-panel figure (image, reconstructed model, residual) for each
   top-K outlier. Uses ``isoster.build_isoster_model`` directly from
   the persisted isophote FITS — no need to re-run the fit.

Usage
-----
::

    uv run python benchmarks/robustness/build_figures.py
    uv run python benchmarks/robustness/build_figures.py \\
        --input outputs/benchmark_robustness \\
        --galaxies ngc3610

The script is idempotent — regenerating the figures does not require a
re-run of the sweep. Every tier subfolder containing a ``results.json``
is processed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import benchmarks.robustness.datasets as datasets  # noqa: E402
from benchmarks.robustness.persist import (  # noqa: E402
    _reference_path,
    _row_path,
    read_isophote_fits,
)
from isoster import build_isoster_model  # noqa: E402

matplotlib.use("Agg")


OUTLIER_TOP_K = 3
INLIER_COLOR = "#b0b0b0"
INLIER_ALPHA = 0.55
INLIER_LINEWIDTH = 0.9
OUTLIER_LINEWIDTH = 2.1
AXIS_LABELS: Dict[str, str] = {
    "sma0": "sma0 factor",
    "eps": "start eps",
    "pa": "start pa (rad)",
}


@dataclass
class RowRecord:
    """Slim view of one results.json row — enough for figure selection."""

    tier: str
    obj_id: str
    arm: str
    axis: str
    value: float
    profile_rel_rms: float
    success: bool
    first_iso_failure: bool
    n_iso_accepted: int
    n_iso_total: int


def load_rows(results_json: Path) -> List[RowRecord]:
    body = json.loads(results_json.read_text())
    out: List[RowRecord] = []
    for r in body.get("rows", []):
        comp = r.get("comparison") or {}
        rel = comp.get("profile_rel_rms")
        out.append(
            RowRecord(
                tier=r["tier"],
                obj_id=r["obj_id"],
                arm=r["arm"],
                axis=r["axis"],
                value=float(r["value"]),
                profile_rel_rms=float(rel) if rel is not None else float("nan"),
                success=bool(r.get("success", True)),
                first_iso_failure=bool(r.get("first_iso_failure", False)),
                n_iso_accepted=int(r.get("n_iso_accepted", 0)),
                n_iso_total=int(r.get("n_iso_total", 0)),
            )
        )
    return out


def _is_outlier(rec: RowRecord, axis_rows: Sequence[RowRecord]) -> bool:
    """Outlier rule: top-K by rel_rms OR first-iso failure OR n_accepted<3."""
    if rec.first_iso_failure or rec.n_iso_accepted < 3:
        return True
    scored = sorted(axis_rows, key=lambda r: -r.profile_rel_rms)
    top = {(r.obj_id, r.arm, r.axis, r.value) for r in scored[:OUTLIER_TOP_K]}
    return (rec.obj_id, rec.arm, rec.axis, rec.value) in top


def _load_reference(output_root: Path, tier: str, obj_id: str) -> Dict[str, np.ndarray]:
    path = _reference_path(output_root, tier, obj_id)
    cols, _meta = read_isophote_fits(path)
    return cols


def _load_row_cols(
    output_root: Path,
    tier: str,
    arm: str,
    obj_id: str,
    axis: str,
    value: float,
) -> Optional[Dict[str, np.ndarray]]:
    path = _row_path(output_root, tier, arm, obj_id, axis, value)
    if not path.exists():
        return None
    cols, _meta = read_isophote_fits(path)
    return cols


def _keep_valid(cols: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Drop isophotes with stop_code indicating rejection, keep code in {0,1,2}."""
    keep = np.isin(cols["stop_code"], (0, 1, 2))
    if not np.any(keep):
        return {k: v[:0] for k, v in cols.items()}
    return {k: v[keep] for k, v in cols.items()}


def _colormap_norm(values: Sequence[float]) -> Tuple[Any, Any]:
    vmin = float(min(values))
    vmax = float(max(values))
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    return plt.get_cmap("viridis"), norm


def build_profile_figure(
    output_root: Path,
    galaxy_rows: Sequence[RowRecord],
    tier: str,
    obj_id: str,
    arm: str,
    axis: str,
    out_path: Path,
) -> Optional[Path]:
    """Build one 2×3 profile overlay figure for one (galaxy, arm, axis)."""
    axis_rows = [r for r in galaxy_rows if r.arm == arm and r.axis == axis]
    if not axis_rows:
        return None

    axis_rows = sorted(axis_rows, key=lambda r: r.value)
    ref_cols = _load_reference(output_root, tier, obj_id)
    ref_cols = _keep_valid(ref_cols)
    if ref_cols["sma"].size == 0:
        return None

    ref_sma = ref_cols["sma"]
    ref_intens = ref_cols["intens"]
    ref_eps = ref_cols["eps"]
    ref_pa_deg = np.rad2deg(ref_cols["pa"])
    ref_x0 = ref_cols["x0"]
    ref_y0 = ref_cols["y0"]

    # Compute a robust median center so we plot drift w.r.t. the
    # reference fit's own center, not an arbitrary isophote.
    ref_center_x = float(np.median(ref_x0))
    ref_center_y = float(np.median(ref_y0))

    cmap, norm = _colormap_norm([r.value for r in axis_rows])

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    (ax_intens, ax_ba, ax_pa), (ax_x0, ax_y0, ax_drift) = axes

    # Two-pass drawing so outlier traces land on top of the inlier
    # background and are not hidden by later iterations.
    loaded: List[Tuple[RowRecord, Dict[str, np.ndarray], bool]] = []
    for rec in axis_rows:
        cols = _load_row_cols(output_root, tier, arm, obj_id, axis, rec.value)
        if cols is None:
            continue
        cols = _keep_valid(cols)
        if cols["sma"].size == 0:
            continue
        loaded.append((rec, cols, _is_outlier(rec, axis_rows)))

    def _plot_row(
        rec: RowRecord,
        cols: Dict[str, np.ndarray],
        is_outlier: bool,
    ) -> None:
        sma = cols["sma"]
        if is_outlier:
            color = cmap(norm(rec.value))
            ls = "-"
            alpha = 1.0
            lw = OUTLIER_LINEWIDTH
            zorder = 6
            label = (
                f"{axis}={rec.value:+.2f} (rel_rms={rec.profile_rel_rms:.2f})"
            )
        else:
            color = INLIER_COLOR
            ls = "-"
            alpha = INLIER_ALPHA
            lw = INLIER_LINEWIDTH
            zorder = 2
            label = None
        ax_intens.plot(sma, cols["intens"], color=color, lw=lw, ls=ls,
                       alpha=alpha, zorder=zorder)
        ax_ba.plot(sma, 1.0 - cols["eps"], color=color, lw=lw, ls=ls,
                   alpha=alpha, zorder=zorder)
        ax_pa.plot(sma, np.rad2deg(cols["pa"]), color=color, lw=lw, ls=ls,
                   alpha=alpha, zorder=zorder)
        ax_x0.plot(sma, cols["x0"] - ref_center_x, color=color, lw=lw, ls=ls,
                   alpha=alpha, zorder=zorder)
        ax_y0.plot(sma, cols["y0"] - ref_center_y, color=color, lw=lw, ls=ls,
                   alpha=alpha, zorder=zorder)
        drift = np.hypot(cols["x0"] - ref_center_x, cols["y0"] - ref_center_y)
        ax_drift.plot(sma, drift, color=color, lw=lw, ls=ls, alpha=alpha,
                      zorder=zorder, label=label)

    for rec, cols, is_out in loaded:
        if not is_out:
            _plot_row(rec, cols, False)
    for rec, cols, is_out in loaded:
        if is_out:
            _plot_row(rec, cols, True)

    # Reference overlay (bold black) on every panel.
    for ax, yvals in (
        (ax_intens, ref_intens),
        (ax_ba, 1.0 - ref_eps),
        (ax_pa, ref_pa_deg),
        (ax_x0, ref_x0 - ref_center_x),
        (ax_y0, ref_y0 - ref_center_y),
        (
            ax_drift,
            np.hypot(ref_x0 - ref_center_x, ref_y0 - ref_center_y),
        ),
    ):
        ax.plot(ref_sma, yvals, color="black", lw=2.0, ls="-", zorder=10, label="reference")

    ax_intens.set_yscale("log")
    ax_intens.set_ylabel("intens (counts/pix)")
    ax_intens.set_title("surface brightness")
    ax_ba.set_ylabel("1 - eps  (b/a)")
    ax_ba.set_title("axis ratio")
    ax_pa.set_ylabel("pa (deg)")
    ax_pa.set_title("position angle")
    ax_x0.set_ylabel("x0 - x0_ref (pix)")
    ax_x0.set_title("centroid x drift")
    ax_y0.set_ylabel("y0 - y0_ref (pix)")
    ax_y0.set_title("centroid y drift")
    ax_drift.set_ylabel("|center drift| (pix)")
    ax_drift.set_title("centroid drift magnitude")

    for ax in (ax_intens, ax_ba, ax_pa, ax_x0, ax_y0, ax_drift):
        ax.set_xlabel("sma (pix)")
        ax.grid(True, alpha=0.3)

    # Zero lines on the drift panels make outliers pop.
    for ax in (ax_x0, ax_y0):
        ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)

    # Colorbar for the axis-value encoding.
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
    cbar.set_label(AXIS_LABELS.get(axis, axis))

    # Legend on the drift panel for the outliers (plus the reference).
    handles, labels = ax_drift.get_legend_handles_labels()
    if handles:
        ax_drift.legend(
            handles, labels, fontsize=7, loc="upper left", framealpha=0.85
        )

    fig.suptitle(
        f"{obj_id} — arm={arm} — axis={axis}\n"
        f"reference = {len(ref_sma)} isophotes  (black, thick)",
        fontsize=12,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _discover_tier_dirs(output_root: Path) -> List[Path]:
    """Return every immediate subdir of ``output_root`` with a results.json."""
    if not output_root.exists():
        return []
    return sorted(
        p for p in output_root.iterdir()
        if p.is_dir() and (p / "results.json").exists()
    )


def build_all_profile_figures(
    output_root: Path,
    galaxy_filter: Optional[Sequence[str]] = None,
    arm_filter: Optional[Sequence[str]] = None,
    axis_filter: Optional[Sequence[str]] = None,
) -> List[Path]:
    """Build every profile overlay figure across tier subfolders."""
    tier_dirs = _discover_tier_dirs(output_root)
    if not tier_dirs:
        raise FileNotFoundError(
            f"no tier subfolders with results.json under {output_root}"
        )

    written: List[Path] = []
    for tier_dir in tier_dirs:
        results_json = tier_dir / "results.json"
        rows = load_rows(results_json)
        if galaxy_filter:
            wanted_g = set(galaxy_filter)
            rows = [r for r in rows if r.obj_id in wanted_g]
        if arm_filter:
            wanted_a = set(arm_filter)
            rows = [r for r in rows if r.arm in wanted_a]
        if axis_filter:
            wanted_x = set(axis_filter)
            rows = [r for r in rows if r.axis in wanted_x]
        if not rows:
            continue

        tiers_by_galaxy: Dict[str, str] = {r.obj_id: r.tier for r in rows}
        figures_root = tier_dir / "figures" / "profiles"

        for obj_id in sorted(tiers_by_galaxy):
            galaxy_rows = [r for r in rows if r.obj_id == obj_id]
            tier = tiers_by_galaxy[obj_id]
            arms_in = sorted({r.arm for r in galaxy_rows})
            axes_in = sorted({r.axis for r in galaxy_rows})
            for arm in arms_in:
                for axis in axes_in:
                    out_path = figures_root / obj_id / f"{arm}_{axis}.png"
                    result = build_profile_figure(
                        output_root=output_root,
                        galaxy_rows=galaxy_rows,
                        tier=tier,
                        obj_id=obj_id,
                        arm=arm,
                        axis=axis,
                        out_path=out_path,
                    )
                    if result is not None:
                        written.append(result)
                        print(f"  wrote {result.relative_to(output_root)}")
    return written


# ---------------------------------------------------------------------------
# Outlier QA figures: image / reconstructed model / residual
# ---------------------------------------------------------------------------


def _cols_to_isophote_list(cols: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """Convert persist column arrays back to the dict list build_isoster_model wants."""
    n = cols["sma"].size
    out: List[Dict[str, Any]] = []
    for i in range(n):
        iso: Dict[str, Any] = {}
        for key, arr in cols.items():
            iso[key] = arr[i].item() if hasattr(arr[i], "item") else arr[i]
        out.append(iso)
    return out


def _pick_outliers(
    rows: Sequence[RowRecord],
    top_k: int = OUTLIER_TOP_K,
) -> List[RowRecord]:
    """Return the top-K rows by ``profile_rel_rms`` from a pool."""
    finite = [r for r in rows if np.isfinite(r.profile_rel_rms)]
    return sorted(finite, key=lambda r: -r.profile_rel_rms)[:top_k]


def _log_stretch(image: np.ndarray, percentile_lo: float = 20.0, percentile_hi: float = 99.5) -> Tuple[float, float]:
    """Simple percentile-based vmin/vmax for a log-style display."""
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(finite[finite > 0], percentile_lo)) if np.any(finite > 0) else 1e-3
    vmax = float(np.percentile(finite, percentile_hi))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def build_outlier_qa_figure(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    cols: Dict[str, np.ndarray],
    rec: RowRecord,
    out_path: Path,
) -> Optional[Path]:
    """Build one 3-panel image/model/residual QA figure for a single outlier row."""
    isos = _cols_to_isophote_list(cols)
    if not isos:
        return None
    try:
        model = build_isoster_model(image.shape, isos, fill=0.0)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"    skip {rec.obj_id}/{rec.arm}/{rec.axis}={rec.value:+.2f}: {exc}")
        return None
    residual = image - model

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    ax_img, ax_mod, ax_res = axes

    vmin, vmax = _log_stretch(image)
    norm = matplotlib.colors.LogNorm(vmin=max(vmin, 1e-6), vmax=vmax)
    ax_img.imshow(np.clip(image, 1e-6, None), origin="lower", cmap="magma", norm=norm)
    ax_img.set_title("image (log stretch)")
    if mask is not None:
        mask_overlay = np.ma.masked_where(~mask.astype(bool), np.ones_like(image))
        ax_img.imshow(
            mask_overlay, origin="lower", cmap="Reds", alpha=0.25, vmin=0, vmax=1
        )

    ax_mod.imshow(np.clip(model, 1e-6, None), origin="lower", cmap="magma", norm=norm)
    ax_mod.set_title(f"reconstructed model ({len(isos)} isophotes)")

    # Residual: symmetric log-ish colormap centered at zero.
    finite_res = residual[np.isfinite(residual)]
    if finite_res.size:
        rlim = float(np.percentile(np.abs(finite_res), 99))
    else:
        rlim = 1.0
    ax_res.imshow(residual, origin="lower", cmap="RdBu_r", vmin=-rlim, vmax=rlim)
    ax_res.set_title(f"residual (|p99|={rlim:.2f})")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    stop_codes = _stop_code_histogram(cols["stop_code"])
    fail_tag = " (first_iso_fail)" if rec.first_iso_failure else ""
    fig.suptitle(
        f"{rec.obj_id} — arm={rec.arm} — axis={rec.axis} value={rec.value:+.2f}{fail_tag}\n"
        f"rel_rms={rec.profile_rel_rms:.2f}  n_accepted={rec.n_iso_accepted}/{rec.n_iso_total}"
        f"  stop_codes={stop_codes}",
        fontsize=11,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _stop_code_histogram(stop_codes: np.ndarray) -> str:
    """Format a compact ``code:count,code:count`` summary."""
    codes, counts = np.unique(stop_codes.astype(int), return_counts=True)
    return ",".join(f"{int(c)}:{int(n)}" for c, n in zip(codes, counts))


def build_all_outlier_qa_figures(
    output_root: Path,
    galaxy_filter: Optional[Sequence[str]] = None,
    arm_filter: Optional[Sequence[str]] = None,
    top_k: int = OUTLIER_TOP_K,
) -> List[Path]:
    """Build the outlier QA figure set per tier × galaxy × arm.

    Only the top-``top_k`` rows by ``profile_rel_rms`` per (galaxy, arm)
    get a figure. Rows with ``profile_rel_rms`` NaN or too few isophotes
    are skipped.
    """
    tier_dirs = _discover_tier_dirs(output_root)
    if not tier_dirs:
        raise FileNotFoundError(
            f"no tier subfolders with results.json under {output_root}"
        )

    galaxy_cache: Dict[str, datasets.GalaxyData] = {}

    def _get_galaxy(tier: str, obj_id: str) -> datasets.GalaxyData:
        key = f"{tier}/{obj_id}"
        if key not in galaxy_cache:
            spec = next(
                (s for s in datasets.list_galaxies(tier) if s.obj_id == obj_id),
                None,
            )
            if spec is None:
                raise RuntimeError(f"no galaxy spec for {tier}/{obj_id}")
            galaxy_cache[key] = datasets.load_galaxy(spec)
        return galaxy_cache[key]

    written: List[Path] = []
    for tier_dir in tier_dirs:
        results_json = tier_dir / "results.json"
        rows = load_rows(results_json)
        if galaxy_filter:
            wanted_g = set(galaxy_filter)
            rows = [r for r in rows if r.obj_id in wanted_g]
        if arm_filter:
            wanted_a = set(arm_filter)
            rows = [r for r in rows if r.arm in wanted_a]
        if not rows:
            continue

        tiers_by_galaxy: Dict[str, str] = {r.obj_id: r.tier for r in rows}
        galaxies_in = sorted(tiers_by_galaxy)
        outlier_root = tier_dir / "figures" / "outliers"

        for obj_id in galaxies_in:
            galaxy_rows = [r for r in rows if r.obj_id == obj_id]
            tier = tiers_by_galaxy[obj_id]
            arms_in = sorted({r.arm for r in galaxy_rows})
            for arm in arms_in:
                arm_rows = [r for r in galaxy_rows if r.arm == arm]
                outliers = _pick_outliers(arm_rows, top_k=top_k)
                if not outliers:
                    continue
                galaxy = _get_galaxy(tier, obj_id)
                for rec in outliers:
                    cols = _load_row_cols(
                        output_root, tier, arm, obj_id, rec.axis, rec.value
                    )
                    if cols is None:
                        print(
                            f"    skip {obj_id}/{arm}/{rec.axis}="
                            f"{rec.value:+.2f}: no fits file"
                        )
                        continue
                    cols = _keep_valid(cols)
                    out_path = (
                        outlier_root
                        / obj_id
                        / f"{arm}_{rec.axis}_{rec.value:+.2f}_qa.png"
                    )
                    result = build_outlier_qa_figure(
                        image=galaxy.image,
                        mask=galaxy.mask,
                        cols=cols,
                        rec=rec,
                        out_path=out_path,
                    )
                    if result is not None:
                        written.append(result)
                        print(f"  wrote {result.relative_to(output_root)}")
    return written


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build phase-2 robustness figures from sweep outputs."
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "benchmark_robustness",
        help="Directory with results.json, sweep/, and reference/ subdirs.",
    )
    ap.add_argument(
        "--galaxies",
        nargs="+",
        default=None,
        help="Restrict to these galaxy IDs.",
    )
    ap.add_argument(
        "--arms",
        nargs="+",
        default=None,
        help="Restrict to these arm names.",
    )
    ap.add_argument(
        "--axes",
        nargs="+",
        default=None,
        help="Restrict to these axis names (profile overlays only).",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=OUTLIER_TOP_K,
        help="Number of top-rel_rms outliers per (galaxy, arm) to render QA for.",
    )
    ap.add_argument(
        "--skip-profiles",
        action="store_true",
        help="Skip the profile overlay figures.",
    )
    ap.add_argument(
        "--skip-outliers",
        action="store_true",
        help="Skip the outlier QA figures.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    written_all: List[Path] = []
    if not args.skip_profiles:
        print(
            f"Building profile overlay figures under {args.input}/"
            "{tier}/figures/profiles"
        )
        written_profiles = build_all_profile_figures(
            output_root=args.input,
            galaxy_filter=args.galaxies,
            arm_filter=args.arms,
            axis_filter=args.axes,
        )
        written_all.extend(written_profiles)
        print(f"  profile figures: {len(written_profiles)}")
    if not args.skip_outliers:
        print(
            f"Building outlier QA figures under {args.input}/"
            "{tier}/figures/outliers"
        )
        written_outliers = build_all_outlier_qa_figures(
            output_root=args.input,
            galaxy_filter=args.galaxies,
            arm_filter=args.arms,
            top_k=args.top_k,
        )
        written_all.extend(written_outliers)
        print(f"  outlier QA figures: {len(written_outliers)}")
    print(f"Wrote {len(written_all)} figures total")


if __name__ == "__main__":
    main()
