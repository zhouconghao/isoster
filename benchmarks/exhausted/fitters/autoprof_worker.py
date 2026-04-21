#!/usr/bin/env python
"""AutoProf subprocess worker — runs inside the isolated AutoProf venv.

Reads a JSON options file, runs the AutoProf pipeline on one galaxy,
writes a status JSON with timing and error info. Must be invoked with
the AutoProf venv python:

    <venv>/bin/python autoprof_worker.py options.json

Ported from sga_isoster/scripts/autoprof_worker.py.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: autoprof_worker.py <options.json>", file=sys.stderr)
        sys.exit(2)

    options_path = sys.argv[1]
    with open(options_path) as handle:
        options = json.load(handle)

    galaxy_name = options.get("ap_name", "unknown")
    save_dir = options.get("ap_saveto", "./")
    status_path = os.path.join(save_dir, f"{galaxy_name}_status.json")

    # Tuple conversion (JSON has no tuples)
    if "ap_iso_measurecoefs" in options:
        options["ap_iso_measurecoefs"] = tuple(options["ap_iso_measurecoefs"])
    if "ap_set_center" in options and options["ap_set_center"] is not None:
        center = options["ap_set_center"]
        options["ap_set_center"] = {"x": float(center["x"]), "y": float(center["y"])}
    pipeline_steps = options.pop("pipeline_steps", None)

    try:
        from autoprof.Pipeline import Isophote_Pipeline  # type: ignore

        log_path = os.path.join(save_dir, f"{galaxy_name}_autoprof.log")
        pipeline = Isophote_Pipeline(loggername=log_path)

        if pipeline_steps:
            pipeline.UpdatePipeline(new_pipeline_steps=pipeline_steps)
        else:
            pipeline.UpdatePipeline(
                new_pipeline_steps=[
                    "background", "psf", "center",
                    "isophoteinit", "isophotefit", "isophoteextract",
                    "checkfit", "ellipsemodel", "writeprof",
                ]
            )

        t_start = time.perf_counter()
        result = pipeline.Process_Image(options=options)
        wall = time.perf_counter() - t_start

        if result == 1:
            status = {
                "status": "error",
                "error_msg": "AutoProf Process_Image returned 1",
                "wall_time": wall,
            }
        else:
            status = {
                "status": "ok",
                "wall_time": wall,
                "timing": {k: float(v) for k, v in result.items()}
                if isinstance(result, dict) else {},
            }
    except Exception as exc:  # noqa: BLE001
        status = {
            "status": "error",
            "error_msg": str(exc),
            "traceback": traceback.format_exc(),
            "wall_time": 0.0,
        }

    with open(status_path, "w") as handle:
        json.dump(status, handle, indent=2)

    if status["status"] != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
