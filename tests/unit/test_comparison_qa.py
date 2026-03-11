"""Tests for multi-method comparison QA figure helpers."""

from pathlib import Path

import numpy as np
import pytest

from isoster.plotting import (
    METHOD_STYLES,
    build_method_profile,
    normalize_pa_degrees,
    plot_comparison_qa_figure,
)


def _make_isophote_list(n=10, pa=0.5):
    """Create a minimal isophote list for testing."""
    return [
        {
            "sma": float(i + 1),
            "intens": 100.0 / (i + 1),
            "eps": 0.3,
            "pa": pa,
            "x0": 50.0,
            "y0": 50.0,
            "stop_code": 0,
            "intens_err": 1.0,
            "rms": 2.0,
        }
        for i in range(n)
    ]


def _make_image(shape=(64, 64), seed=42):
    return np.random.default_rng(seed).normal(100, 10, shape)


# ---------------------------------------------------------------------------
# METHOD_STYLES
# ---------------------------------------------------------------------------


class TestMethodStyles:
    def test_has_required_methods(self):
        assert "isoster" in METHOD_STYLES
        assert "photutils" in METHOD_STYLES
        assert "autoprof" in METHOD_STYLES

    def test_style_keys(self):
        for method, style in METHOD_STYLES.items():
            assert "color" in style
            assert "marker" in style
            assert "label" in style


# ---------------------------------------------------------------------------
# build_method_profile
# ---------------------------------------------------------------------------


class TestBuildMethodProfile:
    def test_from_isophote_list(self):
        isos = _make_isophote_list()
        prof = build_method_profile(isos)
        assert prof is not None
        assert "sma" in prof
        assert "x_axis" in prof
        assert len(prof["sma"]) == 10
        np.testing.assert_allclose(prof["x_axis"], prof["sma"] ** 0.25)

    def test_from_isophote_list_with_stop_codes(self):
        isos = _make_isophote_list()
        isos[5]["stop_code"] = 2
        prof = build_method_profile(isos)
        assert "stop_codes" in prof
        assert prof["stop_codes"][5] == 2

    def test_optional_fields_included(self):
        isos = _make_isophote_list()
        prof = build_method_profile(isos)
        assert "x0" in prof
        assert "y0" in prof
        assert "intens_err" in prof
        assert "rms" in prof

    def test_from_array_dict(self):
        """AutoProf-style: dict with numpy array values."""
        sma = np.arange(1.0, 11.0)
        prof = build_method_profile({
            "sma": sma,
            "intens": 100.0 / sma,
            "eps": np.full(10, 0.3),
            "pa": np.full(10, 0.5),
        })
        assert prof is not None
        assert len(prof["sma"]) == 10
        assert "stop_codes" not in prof

    def test_empty_list_returns_none(self):
        assert build_method_profile([]) is None

    def test_empty_dict_returns_none(self):
        assert build_method_profile({"sma": np.array([])}) is None


# ---------------------------------------------------------------------------
# plot_comparison_qa_figure — layout modes
# ---------------------------------------------------------------------------


class TestPlotComparisonQaFigure:
    """Smoke tests for the three layout modes."""

    def test_mode1_solo_with_model(self, tmp_path):
        """Mode 1: single method with model -> image, model, residual."""
        image = _make_image()
        model = np.full((64, 64), 100.0)
        isos = _make_isophote_list(20)
        out = tmp_path / "mode1.png"
        plot_comparison_qa_figure(
            image=image,
            profiles={"isoster": build_method_profile(isos)},
            models={"isoster": model},
            title="mode 1 solo",
            output_path=out,
        )
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_mode1_solo_no_model(self, tmp_path):
        """Mode 1: single method without model -> blank model/residual."""
        image = _make_image()
        isos = _make_isophote_list(20)
        out = tmp_path / "mode1_nomodel.png"
        plot_comparison_qa_figure(
            image=image,
            profiles={"isoster": build_method_profile(isos)},
            title="mode 1 no model",
            output_path=out,
        )
        assert out.exists()

    def test_mode2_one_on_one(self, tmp_path):
        """Mode 2: two methods with models -> image, 2x residual+overlays."""
        image = _make_image()
        model = np.full((64, 64), 100.0)
        isos = _make_isophote_list(15)
        out = tmp_path / "mode2.png"
        plot_comparison_qa_figure(
            image=image,
            profiles={
                "isoster": build_method_profile(isos),
                "photutils": build_method_profile(isos),
            },
            models={"isoster": model, "photutils": model},
            title="mode 2 one-on-one",
            output_path=out,
        )
        assert out.exists()

    def test_mode3_three_way(self, tmp_path):
        """Mode 3: three methods -> image, 3x residual+overlays."""
        image = _make_image()
        model = np.full((64, 64), 100.0)
        isos = _make_isophote_list(15)
        sma = np.arange(1.0, 16.0)
        autoprof_prof = build_method_profile({
            "sma": sma,
            "intens": 100.0 / sma,
            "eps": np.full(15, 0.3),
            "pa": np.full(15, 0.5),
        })
        out = tmp_path / "mode3.png"
        plot_comparison_qa_figure(
            image=image,
            profiles={
                "isoster": build_method_profile(isos),
                "photutils": build_method_profile(isos),
                "autoprof": autoprof_prof,
            },
            models={"isoster": model, "photutils": model, "autoprof": model},
            title="mode 3 three-way",
            output_path=out,
        )
        assert out.exists()

    def test_empty_profiles_no_crash(self, tmp_path):
        """Empty profiles dict should not crash."""
        image = _make_image()
        out = tmp_path / "empty.png"
        plot_comparison_qa_figure(
            image=image, profiles={}, title="empty", output_path=out,
        )
        assert out.exists()

    def test_mask_overlay(self, tmp_path):
        """Mask is overlaid on the data image."""
        image = _make_image()
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:20, 10:20] = True
        isos = _make_isophote_list(15)
        out = tmp_path / "mask.png"
        plot_comparison_qa_figure(
            image=image,
            profiles={"isoster": build_method_profile(isos)},
            mask=mask,
            title="with mask",
            output_path=out,
        )
        assert out.exists()

    def test_relative_residual(self, tmp_path):
        """relative_residual=True produces a valid figure."""
        image = _make_image()
        model = np.full((64, 64), 100.0)
        isos = _make_isophote_list(15)
        out = tmp_path / "relative.png"
        plot_comparison_qa_figure(
            image=image,
            profiles={"isoster": build_method_profile(isos)},
            models={"isoster": model},
            relative_residual=True,
            title="relative residual",
            output_path=out,
        )
        assert out.exists()

    def test_method_without_stop_codes(self, tmp_path):
        """Method without stop_codes uses plain scatter."""
        image = _make_image()
        sma = np.arange(1.0, 21.0)
        prof = build_method_profile({
            "sma": sma,
            "intens": 100.0 / sma,
            "eps": np.full(20, 0.3),
            "pa": np.full(20, 0.5),
        })
        out = tmp_path / "no_stop.png"
        plot_comparison_qa_figure(
            image=image,
            profiles={"autoprof": prof},
            title="no stop codes",
            output_path=out,
        )
        assert out.exists()

    def test_errorbars_rendered(self, tmp_path):
        """Method with intens_err renders errorbars."""
        image = _make_image()
        isos = _make_isophote_list(15)
        out = tmp_path / "errorbars.png"
        plot_comparison_qa_figure(
            image=image,
            profiles={"isoster": build_method_profile(isos)},
            title="with errorbars",
            output_path=out,
        )
        assert out.exists()
        assert out.stat().st_size > 1000


# ---------------------------------------------------------------------------
# Cross-method PA normalization
# ---------------------------------------------------------------------------


class TestCrossMethodPaNormalization:
    """Verify PA profiles from different methods are anchored consistently."""

    def test_pa_offset_by_180_is_resolved(self):
        """Two methods with PAs ~180 deg apart should be normalized close."""
        # Method A: PA ~ 10 deg (in radians)
        pa_a = np.full(10, np.radians(10.0))
        # Method B: PA ~ 190 deg -> same physical direction as 10 deg
        pa_b = np.full(10, np.radians(190.0))

        pa_a_deg = np.degrees(pa_a)
        pa_b_deg = np.degrees(pa_b)

        # Normalize A first (no anchor)
        norm_a = normalize_pa_degrees(pa_a_deg)
        ref_median = float(np.nanmedian(norm_a))

        # Normalize B anchored to A's median
        norm_b = normalize_pa_degrees(pa_b_deg, anchor=ref_median)

        # They should now be within ~5 deg of each other (not 180 apart)
        diff = np.abs(np.nanmedian(norm_a) - np.nanmedian(norm_b))
        assert diff < 10.0, (
            f"PA profiles should be close after anchoring, got diff={diff:.1f} deg"
        )

    def test_pa_no_offset_preserved(self):
        """Two methods with similar PAs should stay similar after anchoring."""
        pa_a = np.full(10, np.radians(45.0))
        pa_b = np.full(10, np.radians(47.0))

        norm_a = normalize_pa_degrees(np.degrees(pa_a))
        ref_median = float(np.nanmedian(norm_a))
        norm_b = normalize_pa_degrees(np.degrees(pa_b), anchor=ref_median)

        diff = np.abs(np.nanmedian(norm_a) - np.nanmedian(norm_b))
        assert diff < 5.0, f"Similar PAs should remain close, got diff={diff:.1f}"
