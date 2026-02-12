"""CLI smoke tests for isoster command-line entrypoints."""

import sys

import numpy as np
import yaml
from astropy.io import fits
from astropy.table import Table

from isoster.cli import main


def create_cli_test_image(shape=(80, 80)):
    """Create a smooth image suitable for fast CLI smoke tests."""
    y_index, x_index = np.mgrid[: shape[0], : shape[1]].astype(np.float64)
    x0 = shape[1] / 2.0
    y0 = shape[0] / 2.0
    image = 1500.0 * np.exp(-0.5 * (((x_index - x0) / 12.0) ** 2 + ((y_index - y0) / 10.0) ** 2))
    return image


def test_cli_smoke_csv_output(tmp_path, monkeypatch):
    """Run CLI end-to-end on a synthetic FITS image and validate CSV output."""
    image_path = tmp_path / "cli_image.fits"
    config_path = tmp_path / "cli_config.yaml"
    output_path = tmp_path / "cli_isophotes.csv"

    image = create_cli_test_image()
    fits.writeto(image_path, image, overwrite=True)

    config = {
        "x0": image.shape[1] / 2.0,
        "y0": image.shape[0] / 2.0,
        "sma0": 8.0,
        "minsma": 3.0,
        "maxsma": 24.0,
        "astep": 0.15,
        "eps": 0.2,
        "pa": 0.1,
        "minit": 8,
        "maxit": 30,
        "conver": 0.05,
        "fix_center": True,
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "isoster",
            str(image_path),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
            "--fix_center",
        ],
    )

    main()

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    table = Table.read(output_path, format="ascii.csv")
    assert len(table) > 0
    assert "sma" in table.colnames
    assert "intens" in table.colnames
