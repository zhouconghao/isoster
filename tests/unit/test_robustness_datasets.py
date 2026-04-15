"""Smoke tests for ``benchmarks.robustness.datasets`` tier dispatch.

Covers the lightweight ``list_galaxies`` side of each tier and the
huang2013 loader when its external data root is present locally. The
heavy loaders (highorder, hsc) are already exercised end-to-end by the
sweep smoke runs, so they are not duplicated here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.robustness import datasets  # noqa: E402


def test_tier_tuple_contains_all_four_tiers() -> None:
    assert datasets.TIERS == ("mocks", "huang2013", "highorder", "hsc")


def test_list_galaxies_huang2013_returns_four_ic2597_mocks() -> None:
    specs = datasets.list_galaxies("huang2013")
    assert [spec.obj_id for spec in specs] == [
        "IC2597_mock1",
        "IC2597_mock2",
        "IC2597_mock3",
        "IC2597_mock4",
    ]
    for spec in specs:
        assert spec.tier == "huang2013"
        assert spec.extras["galaxy"] == "IC2597"
        assert spec.extras["mock_id"] in (1, 2, 3, 4)


def test_list_galaxies_unknown_tier_raises() -> None:
    with pytest.raises(ValueError, match="unknown tier"):
        datasets.list_galaxies("does_not_exist")


def _huang2013_data_available() -> bool:
    specs = datasets.list_galaxies("huang2013")
    if not specs:
        return False
    galaxy = specs[0].extras["galaxy"]
    mock_id = specs[0].extras["mock_id"]
    data_root = datasets._huang2013_data_root()
    return (data_root / galaxy / f"{galaxy}_mock{mock_id}.fits").exists()


@pytest.mark.skipif(
    not _huang2013_data_available(),
    reason="huang2013 external data root not available on this machine",
)
def test_load_huang2013_first_mock_exposes_header_driven_fiducial() -> None:
    specs = datasets.list_galaxies("huang2013")
    data = datasets.load_galaxy(specs[0])

    assert data.image.ndim == 2
    height, width = data.image.shape
    assert height == width  # IC2597 mocks are square
    # infer_initial_geometry anchors the start at the image center and
    # returns the DEFAULT_INITIAL_SMA_PIX (6.0) unless overridden.
    assert data.fiducial_x0 == pytest.approx((width - 1) / 2.0)
    assert data.fiducial_y0 == pytest.approx((height - 1) / 2.0)
    assert data.fiducial_sma0 == pytest.approx(6.0)
    assert 0.0 <= data.fiducial_eps <= 0.95
    # infer_default_maxsma clamps to min(0.48 * size, 8 * max_component_re).
    assert 30.0 < data.maxsma < 0.48 * min(height, width) + 1.0
    assert data.mask is None
    assert data.variance_map is None
    assert data.config_overrides == {}
