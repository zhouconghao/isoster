"""Per-tool single-galaxy fit drivers.

Each fitter exposes a ``fit_galaxy(bundle, arm_config, output_dir)``
function that writes ``profile.fits``, ``model.fits`` (optional),
``qa.png``, ``run_record.json``, and returns an inventory row dict.
"""
