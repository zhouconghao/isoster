"""AutoProf campaign fitter.

Runs AutoProf in an isolated venv via subprocess. When the venv is
missing or autoprof cannot import, the arm is reported as
``status="skipped"`` with a clear ``error_msg`` — never as a hard
failure. The campaign YAML's ``tools.autoprof.venv_python`` points at
the isolated python; we probe it lazily on the first arm of each
galaxy.

Output schema matches the other fitters: ``profile.fits`` / ``model.fits``
/ ``qa.png`` / ``run_record.json`` / ``config.yaml`` with an extra
``raw/`` subdirectory for AutoProf's native ``.prof`` / ``.aux`` /
``_genmodel.fits``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from astropy.io import fits

from isoster import build_isoster_model
from isoster.plotting import plot_qa_summary
from isoster.utils import isophote_results_to_fits

from ..adapters.base import GalaxyBundle
from ..analysis.inventory import INVENTORY_COLUMNS
from ..analysis.metrics import summarize_fit
from ..analysis.quality_flags import evaluate_flags
from ..analysis.residual_zones import residual_zone_stats


_VENV_PROBE_CACHE: dict[str, str] = {}


def run_one_arm(
    bundle: GalaxyBundle,
    arm_id: str,
    arm_delta: dict[str, Any],
    output_dir: Path,
    *,
    write_qa: bool = True,
    write_model_fits: bool = True,
    venv_python: str = "/tmp/autoprof_venv/bin/python",
    timeout: int = 300,
) -> dict[str, Any]:
    """Run one ``(galaxy, autoprof-arm)`` pair. Returns an inventory row."""
    total_start = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    row = _empty_inventory_row(bundle.metadata.galaxy_id, arm_id)

    # 1) Venv probe. Skip gracefully if unusable.
    probe_reason = _probe_venv(venv_python)
    if probe_reason is not None:
        row["status"] = "skipped"
        row["error_msg"] = probe_reason
        _write_run_record(output_dir, {"status": "skipped", "reason": probe_reason})
        return row

    # 2) Stage inputs — AutoProf needs a plain-HDU FITS on disk.
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    image = np.asarray(bundle.image, dtype=np.float64)
    galaxy_tag = bundle.metadata.galaxy_id.replace("/", "__")
    image_path = temp_dir / f"{galaxy_tag}_image.fits"
    fits.PrimaryHDU(data=image).writeto(image_path, overwrite=True)
    mask_path: Path | None = None
    if bundle.mask is not None:
        mask_path = temp_dir / f"{galaxy_tag}_mask.fits"
        # AutoProf convention: 0 = good, >0 = bad (matches isoster bool mask)
        fits.PrimaryHDU(
            data=np.asarray(bundle.mask, dtype=np.int16)
        ).writeto(mask_path, overwrite=True)

    # 3) Build options JSON.
    geom = bundle.initial_geometry
    options = _build_options(
        bundle=bundle,
        arm_delta=arm_delta,
        image_path=image_path,
        mask_path=mask_path,
        save_dir=str(raw_dir),
        galaxy_tag=galaxy_tag,
    )
    json_path = temp_dir / f"{galaxy_tag}_options.json"
    with json_path.open("w") as handle:
        json.dump(options, handle, indent=2)

    # 4) Subprocess invocation.
    worker_script = Path(__file__).with_name("autoprof_worker.py")
    fit_start = time.perf_counter()
    try:
        proc = subprocess.run(
            [venv_python, str(worker_script), str(json_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        rc = proc.returncode
        stderr = proc.stderr
    except subprocess.TimeoutExpired:
        rc, stderr = -1, f"timeout after {timeout}s"
    wall_fit = time.perf_counter() - fit_start

    status_path = raw_dir / f"{galaxy_tag}_status.json"
    if rc == -1:
        row["status"] = "failed"
        row["error_msg"] = stderr
        row["wall_time_fit_s"] = float(wall_fit)
        row["wall_time_total_s"] = float(time.perf_counter() - total_start)
        _write_run_record(
            output_dir,
            {"status": "failed", "error_msg": stderr, "wall_time_fit_s": float(wall_fit)},
        )
        return row
    if rc != 0 and not status_path.is_file():
        row["status"] = "failed"
        row["error_msg"] = (stderr or f"returncode={rc}")[:500]
        row["wall_time_fit_s"] = float(wall_fit)
        row["wall_time_total_s"] = float(time.perf_counter() - total_start)
        _write_run_record(
            output_dir,
            {"status": "failed", "error_msg": row["error_msg"]},
        )
        return row

    if status_path.is_file():
        with status_path.open() as handle:
            status_data = json.load(handle)
        if status_data.get("status") != "ok":
            row["status"] = "failed"
            row["error_msg"] = str(status_data.get("error_msg", "unknown autoprof error"))
            row["wall_time_fit_s"] = float(wall_fit)
            row["wall_time_total_s"] = float(time.perf_counter() - total_start)
            _write_run_record(output_dir, {"status": "failed", **status_data})
            return row

    # 5) Parse AutoProf outputs.
    prof_path = raw_dir / f"{galaxy_tag}.prof"
    aux_path = raw_dir / f"{galaxy_tag}.aux"
    isophotes, n_raw, n_filtered = _parse_prof_file(
        prof_path, pixel_scale_arcsec=bundle.metadata.pixel_scale_arcsec,
        sb_zeropoint=bundle.metadata.sb_zeropoint,
    )
    aux_info = _parse_aux_file(aux_path)

    if not isophotes:
        row["status"] = "failed"
        row["error_msg"] = f"empty profile (n_raw={n_raw}, filtered={n_filtered})"
        row["wall_time_fit_s"] = float(wall_fit)
        row["wall_time_total_s"] = float(time.perf_counter() - total_start)
        _write_run_record(
            output_dir,
            {"status": "failed", "error_msg": row["error_msg"]},
        )
        return row

    # Stamp centers onto every isophote (AutoProf reports a single center).
    cx = float(aux_info.get("center_x", np.nan))
    cy = float(aux_info.get("center_y", np.nan))
    for iso in isophotes:
        if "x0" not in iso or not np.isfinite(iso.get("x0", np.nan)):
            iso["x0"] = cx
        if "y0" not in iso or not np.isfinite(iso.get("y0", np.nan)):
            iso["y0"] = cy

    results: dict[str, Any] = {
        "isophotes": isophotes,
        "tool": "autoprof",
        "config": dict(arm_delta or {}),
        "first_isophote_failure": False,
        "first_isophote_retry_log": [],
        "autoprof_aux": aux_info,
        "autoprof_n_raw": n_raw,
        "autoprof_n_filtered": n_filtered,
    }

    profile_path = output_dir / "profile.fits"
    isophote_results_to_fits(results, str(profile_path), overwrite=True)

    config_path = output_dir / "config.yaml"
    with config_path.open("w") as handle:
        yaml.safe_dump(
            {"tool": "autoprof", "arm_id": arm_id, **(arm_delta or {})},
            handle,
            sort_keys=False,
        )

    # 6) Model: prefer AutoProf's genmodel; fall back to our rebuild.
    genmodel_path = raw_dir / f"{galaxy_tag}_genmodel.fits"
    model: np.ndarray
    if genmodel_path.is_file():
        with fits.open(genmodel_path) as hdul:
            # AutoProf places model in HDU 1
            model_hdu = hdul[1] if len(hdul) > 1 else hdul[0]
            model = np.asarray(model_hdu.data, dtype=np.float64)
    else:
        model = build_isoster_model(image.shape, isophotes)

    model_path = None
    if write_model_fits:
        model_path = output_dir / "model.fits"
        residual = image - model
        primary = fits.PrimaryHDU(
            data=model.astype(np.float32),
            header=fits.Header({"EXTNAME": "MODEL", "ARM_ID": arm_id}),
        )
        resid_hdu = fits.ImageHDU(
            data=residual.astype(np.float32),
            header=fits.Header({"EXTNAME": "RESIDUAL", "ARM_ID": arm_id}),
        )
        fits.HDUList([primary, resid_hdu]).writeto(model_path, overwrite=True)

    qa_path = None
    if write_qa:
        qa_path = output_dir / "qa.png"
        try:
            plot_qa_summary(
                title=f"{bundle.metadata.galaxy_id}  |  autoprof::{arm_id}",
                image=image,
                isoster_model=model,
                isoster_res=isophotes,
                mask=bundle.mask,
                filename=str(qa_path),
                relative_residual=False,
                sb_zeropoint=bundle.metadata.sb_zeropoint,
                pixel_scale_arcsec=bundle.metadata.pixel_scale_arcsec,
            )
        except Exception as exc:  # noqa: BLE001
            qa_path.with_suffix(".png.err.txt").write_text(
                f"{type(exc).__name__}: {exc}\n"
            )
            qa_path = None

    # 7) Metrics + residual zones + flags (shared pipeline).
    metrics = summarize_fit(results, sma0=float(geom.get("sma0", 1.0)))
    zone_stats = residual_zone_stats(
        image,
        model,
        x0=float(geom["x0"]),
        y0=float(geom["y0"]),
        eps=float(geom.get("eps", 0.2)),
        pa=float(geom.get("pa", 0.0)),
        R_ref=bundle.metadata.effective_Re_pix,
        maxsma=float(geom.get("maxsma", min(image.shape) // 2)),
        mask=bundle.mask,
    )
    zone_row = {
        k: zone_stats[k]
        for k in (
            "resid_rms_inner",
            "resid_rms_mid",
            "resid_rms_outer",
            "resid_median_inner",
            "resid_median_mid",
            "resid_median_outer",
            "frac_above_3sigma_outer",
        )
    }

    row.update(
        {
            "status": "ok",
            "error_msg": "",
            "wall_time_fit_s": float(wall_fit),
            "wall_time_total_s": float(time.perf_counter() - total_start),
            **metrics,
            **zone_row,
            "first_isophote_failure": False,
            "first_isophote_retry_attempts": 0,
            "first_isophote_retry_stop_codes": "",
            "qa_path": str(qa_path) if qa_path else "",
            "profile_path": str(profile_path),
            "model_path": str(model_path) if model_path else "",
            "config_path": str(config_path),
        }
    )
    row.update(evaluate_flags(row))
    _write_run_record(
        output_dir,
        {
            "status": "ok",
            "wall_time_fit_s": float(wall_fit),
            "wall_time_total_s": row["wall_time_total_s"],
            "metrics": {**metrics, **zone_stats},
            "flags": row.get("flags", ""),
            "flag_severity_max": row.get("flag_severity_max", 0.0),
            "arm_delta": arm_delta,
            "autoprof_aux": aux_info,
            "autoprof_n_raw": n_raw,
            "autoprof_n_filtered": n_filtered,
        },
    )
    return row


# ---------------------------------------------------------------------------
# Venv probe
# ---------------------------------------------------------------------------


def _probe_venv(venv_python: str) -> str | None:
    """Return ``None`` if the venv can import autoprof; otherwise a skip reason.

    Result cached per ``venv_python`` path so only the first arm on the
    first galaxy pays the probe cost.
    """
    cached = _VENV_PROBE_CACHE.get(venv_python)
    if cached == "OK":
        return None
    if cached is not None:
        return cached
    if not Path(venv_python).is_file():
        reason = (
            f"autoprof venv python not found: {venv_python}. "
            f"Create it with `python -m venv {Path(venv_python).parents[1]}` "
            f"and `pip install autoprof`, then re-run."
        )
        _VENV_PROBE_CACHE[venv_python] = reason
        return reason
    try:
        proc = subprocess.run(
            [venv_python, "-c", "import autoprof; print('ok')"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        reason = f"autoprof probe crashed: {exc}"
        _VENV_PROBE_CACHE[venv_python] = reason
        return reason
    if proc.returncode != 0:
        reason = (
            "autoprof import failed inside venv "
            f"({venv_python}): {proc.stderr.strip()[:200]}"
        )
        _VENV_PROBE_CACHE[venv_python] = reason
        return reason
    _VENV_PROBE_CACHE[venv_python] = "OK"
    return None


# ---------------------------------------------------------------------------
# Options builder
# ---------------------------------------------------------------------------

AUTOPROF_DEFAULTS: dict[str, Any] = {
    "ap_isoclip": True,
    "ap_isoclip_nsigma": 5.0,
    "ap_isoaverage_method": "median",
    "ap_regularize_scale": 1.0,
    "ap_fit_limit": 2.0,
    "ap_samplegeometricscale": 0.1,
    "ap_iso_interpolate_start": 5,
    "ap_iso_measurecoefs": (3, 4),
}


def _build_options(
    *,
    bundle: GalaxyBundle,
    arm_delta: dict[str, Any],
    image_path: Path,
    mask_path: Path | None,
    save_dir: str,
    galaxy_tag: str,
) -> dict[str, Any]:
    cfg = dict(AUTOPROF_DEFAULTS)
    cfg.update(arm_delta or {})
    geom = bundle.initial_geometry
    md = bundle.metadata

    options: dict[str, Any] = {
        "ap_image_file": str(image_path),
        "ap_name": galaxy_tag,
        "ap_pixscale": md.pixel_scale_arcsec,
        "ap_zeropoint": md.sb_zeropoint,
        "ap_process_mode": "image",
        "ap_doplot": False,
        "ap_saveto": save_dir + "/",
        "ap_plotpath": save_dir + "/",
        "ap_iso_measurecoefs": list(cfg["ap_iso_measurecoefs"]),
        "ap_isoclip": bool(cfg["ap_isoclip"]),
        "ap_isoaverage_method": cfg["ap_isoaverage_method"],
        "ap_regularize_scale": float(cfg["ap_regularize_scale"]),
        "ap_fit_limit": float(cfg["ap_fit_limit"]),
        "ap_samplegeometricscale": float(cfg["ap_samplegeometricscale"]),
        "ap_iso_interpolate_start": int(cfg["ap_iso_interpolate_start"]),
    }
    if cfg["ap_isoclip"]:
        options["ap_isoclip_nsigma"] = float(cfg["ap_isoclip_nsigma"])
    if mask_path is not None:
        options["ap_mask_file"] = str(mask_path)
        options["ap_mask_hdu"] = 0
    # Seed AutoProf's center with the adapter's initial x0/y0. That's
    # the same convention our other fitters use.
    options["ap_set_center"] = {"x": float(geom["x0"]), "y": float(geom["y0"])}
    # Optional PA / ellipticity seeds from the adapter.
    options["ap_isoinit_pa_set"] = float(np.degrees(geom.get("pa", 0.0)))
    options["ap_isoinit_ellip_set"] = float(geom.get("eps", 0.2))
    return options


# ---------------------------------------------------------------------------
# .prof / .aux parsing
# ---------------------------------------------------------------------------


def _parse_prof_file(
    prof_path: Path,
    *,
    pixel_scale_arcsec: float,
    sb_zeropoint: float,
) -> tuple[list[dict[str, Any]], int, int]:
    """Convert an AutoProf .prof into an isoster-compatible list of dicts."""
    if not prof_path.is_file():
        return [], 0, 0
    data = np.genfromtxt(prof_path, delimiter=",", names=True, skip_header=1)
    if data.ndim == 0:
        data = np.array([data])
    n_raw = len(data)
    valid = data["SB"] < 90.0
    n_filtered = int(np.sum(~valid))
    data = data[valid]
    if len(data) == 0:
        return [], n_raw, n_filtered

    # SMA in pixels
    sma = data["R"] / pixel_scale_arcsec
    sb = data["SB"]
    sb_err = data["SB_e"]
    intens_arcsec2 = 10.0 ** (-(sb - sb_zeropoint) / 2.5)
    intens = intens_arcsec2 * pixel_scale_arcsec ** 2
    intens_err = intens * np.log(10.0) / 2.5 * np.abs(sb_err)

    eps = data["ellip"]
    eps_err = data["ellip_e"]
    pa_rad = np.deg2rad(data["pa"] - 90.0) % np.pi
    pa_err = np.deg2rad(data["pa_e"])

    ndata = (
        np.asarray(data["pixels"], dtype=np.int32)
        if "pixels" in data.dtype.names
        else np.zeros(len(data), dtype=np.int32)
    )
    nflag = (
        np.asarray(data["maskedpixels"], dtype=np.int32)
        if "maskedpixels" in data.dtype.names
        else np.zeros(len(data), dtype=np.int32)
    )
    harmonics: dict[str, np.ndarray] = {}
    for key in ("a3", "b3", "a4", "b4"):
        if key in data.dtype.names:
            harmonics[key] = np.asarray(data[key], dtype=np.float64)

    rows: list[dict[str, Any]] = []
    for i in range(len(data)):
        iso = {
            "sma": float(sma[i]),
            "intens": float(intens[i]),
            "intens_err": float(intens_err[i]),
            "eps": float(eps[i]),
            "eps_err": float(eps_err[i]),
            "pa": float(pa_rad[i]),
            "pa_err": float(pa_err[i]),
            "x0": float("nan"),
            "y0": float("nan"),
            "x0_err": 0.0,
            "y0_err": 0.0,
            "rms": float("nan"),
            "stop_code": 0,
            "ndata": int(ndata[i]),
            "nflag": int(nflag[i]),
            "niter": 0,
            "grad": float("nan"),
            "grad_error": float("nan"),
            "grad_r_error": float("nan"),
            "tflux_e": 0.0,
            "tflux_c": 0.0,
            "npix_e": 0,
            "npix_c": 0,
            "lsb_locked": False,
        }
        for key, arr in harmonics.items():
            iso[key] = float(arr[i])
        rows.append(iso)
    return rows, n_raw, n_filtered


def _parse_aux_file(aux_path: Path) -> dict[str, float]:
    info = {
        "center_x": float("nan"),
        "center_y": float("nan"),
        "background": float("nan"),
        "background_noise": float("nan"),
        "psf_fwhm": float("nan"),
    }
    if not aux_path.is_file():
        return info
    for raw_line in aux_path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("center x:"):
            try:
                parts = line.split(",")
                info["center_x"] = float(parts[0].split(":")[1].strip().split()[0])
                info["center_y"] = float(parts[1].split(":")[1].strip().split()[0])
            except Exception:  # noqa: BLE001
                pass
        elif line.startswith("background:"):
            try:
                after = line.split(":", 1)[1].strip()
                info["background"] = float(after.split()[0])
                if "noise:" in line:
                    info["background_noise"] = float(
                        line.split("noise:")[1].strip().split()[0]
                    )
            except (ValueError, IndexError):
                pass
        elif line.startswith("psf fwhm:"):
            try:
                info["psf_fwhm"] = float(
                    line.split(":")[1].strip().split()[0]
                )
            except (ValueError, IndexError):
                pass
    return info


# ---------------------------------------------------------------------------
# Inventory row skeleton + run_record helper (mirrors photutils_fitter)
# ---------------------------------------------------------------------------


def _empty_inventory_row(galaxy_id: str, arm_id: str) -> dict[str, Any]:
    row: dict[str, Any] = {col: "" for col in INVENTORY_COLUMNS}
    row["galaxy_id"] = galaxy_id
    row["tool"] = "autoprof"
    row["arm_id"] = arm_id
    row["status"] = "pending"
    for int_col in (
        "n_iso", "n_stop_0", "n_stop_1", "n_stop_2", "n_stop_m1", "n_locked",
        "first_isophote_retry_attempts", "n_iso_ref_used",
    ):
        row[int_col] = 0
    row["first_isophote_failure"] = False
    for num_col in (
        "wall_time_fit_s",
        "wall_time_total_s",
        "frac_stop_nonzero",
        "combined_drift_pix",
        "spline_rms_center",
        "max_dpa_deg",
        "max_deps",
        "outer_gerr_median",
        "outward_drift_x",
        "outward_drift_y",
        "locked_drift_x",
        "locked_drift_y",
        "resid_rms_inner",
        "resid_rms_mid",
        "resid_rms_outer",
        "resid_median_inner",
        "resid_median_mid",
        "resid_median_outer",
        "frac_above_3sigma_outer",
        "image_sigma_adu",
        "flag_severity_max",
        "composite_score",
    ):
        row[num_col] = float("nan")
    return row


def _write_run_record(output_dir: Path, record: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_record.json").open("w") as handle:
        json.dump(record, handle, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


__all__ = ["run_one_arm"]
# keep shutil import used for future cleanup if needed
_ = shutil
