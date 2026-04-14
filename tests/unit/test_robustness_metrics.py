"""Unit tests for ``benchmarks.robustness.metrics``.

These guard the math that the robustness benchmark depends on — the
relative-intensity deviation, the pi-periodic angular MAD, and the
interpolation-based ``compare_to_reference`` entry point. The sweep
script itself has no unit test; these helpers are the load-bearing core.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.robustness import metrics  # noqa: E402


def _isophotes_from_arrays(
    sma: np.ndarray,
    intens: np.ndarray,
    eps: float = 0.3,
    pa: float = 0.0,
    x0: float = 100.0,
    y0: float = 100.0,
) -> list[dict]:
    """Build a minimal isophote-dict list for metric tests."""
    return [
        {
            "sma": float(sma[i]),
            "intens": float(intens[i]),
            "eps": float(eps),
            "pa": float(pa),
            "x0": float(x0),
            "y0": float(y0),
        }
        for i in range(len(sma))
    ]


def test_relative_intensity_deviation_matching_arrays() -> None:
    ref = np.array([10.0, 20.0, 30.0])
    rms, rel_max, n_low = metrics.relative_intensity_deviation(
        ref, ref.copy(), low_signal_threshold=1e-9
    )
    assert rms == pytest.approx(0.0, abs=1e-12)
    assert rel_max == pytest.approx(0.0, abs=1e-12)
    assert n_low == 0


def test_relative_intensity_deviation_known_deviation() -> None:
    ref = np.array([100.0, 100.0, 100.0])
    pert = np.array([105.0, 95.0, 110.0])  # +5%, -5%, +10%
    rms, rel_max, n_low = metrics.relative_intensity_deviation(
        ref, pert, low_signal_threshold=1e-9
    )
    expected_rms = float(np.sqrt(np.mean([0.05**2, 0.05**2, 0.10**2])))
    assert rms == pytest.approx(expected_rms, rel=1e-9)
    assert rel_max == pytest.approx(0.10, rel=1e-9)
    assert n_low == 0


def test_relative_intensity_deviation_low_signal_points_dropped() -> None:
    ref = np.array([1e-12, 100.0, 100.0])
    pert = np.array([1.0, 105.0, 95.0])
    rms, rel_max, n_low = metrics.relative_intensity_deviation(
        ref, pert, low_signal_threshold=1e-6
    )
    assert n_low == 1
    expected_rms = float(np.sqrt(np.mean([0.05**2, 0.05**2])))
    assert rms == pytest.approx(expected_rms, rel=1e-9)
    assert rel_max == pytest.approx(0.05, rel=1e-9)


def test_angular_mad_pi_periodicity() -> None:
    # pa differences of pi should wrap to zero (isophotes are pi-periodic).
    ref = np.array([0.1, -0.2, 0.5])
    pert = ref + np.pi
    assert metrics.angular_mad(ref, pert) == pytest.approx(0.0, abs=1e-12)


def test_angular_mad_halfpi_midpoint() -> None:
    ref = np.array([0.0])
    pert = np.array([np.pi / 4.0])
    assert metrics.angular_mad(ref, pert) == pytest.approx(np.pi / 4.0, abs=1e-12)


def test_compare_to_reference_identical_lists_gives_zero_motion() -> None:
    sma = np.linspace(5.0, 50.0, 20)
    intens = 100.0 * np.exp(-sma / 15.0)
    ref = _isophotes_from_arrays(sma, intens)
    pert = _isophotes_from_arrays(sma.copy(), intens.copy())
    comparison = metrics.compare_to_reference(
        ref, pert, low_signal_threshold=1e-9
    )
    assert comparison.frac_overlap == pytest.approx(1.0)
    assert comparison.profile_rel_rms == pytest.approx(0.0, abs=1e-10)
    assert comparison.profile_rel_max == pytest.approx(0.0, abs=1e-10)
    assert comparison.eps_mad == pytest.approx(0.0, abs=1e-12)
    assert comparison.pa_mad_deg == pytest.approx(0.0, abs=1e-12)
    assert comparison.center_max_drift == pytest.approx(0.0, abs=1e-12)


def test_compare_to_reference_perturbed_outside_support_reduces_overlap() -> None:
    sma_ref = np.linspace(5.0, 50.0, 20)
    intens_ref = 100.0 * np.exp(-sma_ref / 15.0)
    ref = _isophotes_from_arrays(sma_ref, intens_ref)
    # Perturbed SMAs stretch past the reference support on both ends.
    sma_pert = np.linspace(2.0, 70.0, 25)
    intens_pert = 100.0 * np.exp(-sma_pert / 15.0)
    pert = _isophotes_from_arrays(sma_pert, intens_pert)
    comparison = metrics.compare_to_reference(
        ref, pert, low_signal_threshold=1e-9
    )
    assert comparison.n_pert == 25
    assert comparison.n_overlap < 25
    assert comparison.frac_overlap < 1.0
    # In-support region uses linear interpolation across the exponential
    # reference, so the rel_rms is dominated by interpolation curvature
    # error, not by a real profile disagreement. Keep a loose bound.
    assert comparison.profile_rel_rms < 0.01


def test_bin_metrics_bins_a_clean_fit_as_good() -> None:
    comp = metrics.ProfileComparison(
        n_ref=50,
        n_pert=50,
        n_overlap=50,
        frac_overlap=1.0,
        profile_rel_rms=0.005,
        profile_rel_max=0.01,
        n_low_signal=0,
        eps_mad=0.005,
        pa_mad_deg=0.5,
        center_max_drift=0.1,
    )
    bins = metrics.bin_metrics(comp, first_iso_failure=False)
    assert metrics.worst_bin(bins) == "good"


def test_bin_metrics_marks_first_iso_failure_as_worst_case_fail() -> None:
    comp = metrics.ProfileComparison(
        n_ref=0,
        n_pert=0,
        n_overlap=0,
        frac_overlap=float("nan"),
        profile_rel_rms=float("nan"),
        profile_rel_max=float("nan"),
        n_low_signal=0,
        eps_mad=float("nan"),
        pa_mad_deg=float("nan"),
        center_max_drift=float("nan"),
    )
    bins = metrics.bin_metrics(comp, first_iso_failure=True)
    assert bins["first_iso_failure"] == "fail"
    assert metrics.worst_bin(bins) == "fail"
