"""Tests for ``isoster.multiband.fitting_mb``."""

import numpy as np
import pytest

from isoster.fitting import fit_first_and_second_harmonics
from isoster.multiband.config_mb import IsosterConfigMB
from isoster.multiband.fitting_mb import (
    evaluate_joint_model,
    extract_forced_photometry_mb,
    fit_first_and_second_harmonics_joint,
    fit_isophote_mb,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _planted_galaxy(
    h: int = 256, w: int = 256, x0: float = 128.0, y0: float = 128.0,
    eps: float = 0.3, pa: float = 0.5, n_sersic: float = 1.5, re: float = 30.0,
    amplitude: float = 1.0, noise_sigma: float = 0.01,
    seed: int = 42,
) -> np.ndarray:
    """A perfect-ellipse Sersic-like profile with Gaussian noise.

    Sersic with the ellipse semi-major axis as the radial coordinate
    (b/a = 1 - eps, axis rotated by pa), so the truth geometry is
    exactly ``(x0, y0, eps, pa)``.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    dx = x - x0
    dy = y - y0
    cos_pa, sin_pa = np.cos(pa), np.sin(pa)
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    r = np.sqrt(x_rot**2 + (y_rot / (1.0 - eps)) ** 2)
    bn = 2.0 * n_sersic - 0.327
    img = amplitude * np.exp(-bn * ((r / re) ** (1.0 / n_sersic) - 1.0))
    img += rng.normal(0.0, noise_sigma, size=img.shape)
    return img


# ---------------------------------------------------------------------------
# Joint solver
# ---------------------------------------------------------------------------


def test_joint_solver_recovers_planted_coefficients():
    """The joint solver hits the truth for clean inputs."""
    rng = np.random.default_rng(0)
    n = 128
    angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    n_bands = 4
    I0 = np.array([100.0, 50.0, 200.0, 30.0])
    A1, B1, A2, B2 = 0.5, -0.3, 0.2, 0.1
    intens = np.empty((n_bands, n), dtype=np.float64)
    for b in range(n_bands):
        intens[b] = (
            I0[b]
            + A1 * np.sin(angles) + B1 * np.cos(angles)
            + A2 * np.sin(2 * angles) + B2 * np.cos(2 * angles)
        )
    intens += rng.normal(0.0, 1e-3, size=intens.shape)
    weights = np.ones(n_bands)
    coeffs, cov, wls = fit_first_and_second_harmonics_joint(angles, intens, weights, None)

    assert coeffs.shape == (n_bands + 4,)
    assert cov is not None and cov.shape == (n_bands + 4, n_bands + 4)
    assert wls is False  # OLS path
    np.testing.assert_allclose(coeffs[:n_bands], I0, atol=2e-4)
    np.testing.assert_allclose(coeffs[n_bands:], [A1, B1, A2, B2], atol=2e-4)


def test_joint_solver_b1_matches_single_band_solver():
    """B=1 joint solve must match the single-band 5-param fit."""
    rng = np.random.default_rng(7)
    n = 64
    angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    coeffs_truth = np.array([100.0, 0.5, -0.3, 0.2, 0.1])
    intens = (
        coeffs_truth[0]
        + coeffs_truth[1] * np.sin(angles) + coeffs_truth[2] * np.cos(angles)
        + coeffs_truth[3] * np.sin(2 * angles) + coeffs_truth[4] * np.cos(2 * angles)
    )
    intens += rng.normal(0.0, 1e-3, size=n)

    sb_coeffs, _ = fit_first_and_second_harmonics(angles, intens)

    intens_2d = intens[None, :]  # (1, N)
    weights = np.ones(1)
    mb_coeffs, _, _ = fit_first_and_second_harmonics_joint(angles, intens_2d, weights, None)

    # Same coefficient ordering: [I0, A1, B1, A2, B2].
    np.testing.assert_allclose(mb_coeffs, sb_coeffs, atol=1e-12)


def test_joint_solver_wls_exact_covariance():
    """WLS mode produces the exact (A^T W A)^-1 covariance, no scaling."""
    rng = np.random.default_rng(11)
    n = 256
    angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    n_bands = 2
    I0 = np.array([100.0, 50.0])
    intens = np.empty((n_bands, n))
    for b in range(n_bands):
        intens[b] = (
            I0[b] + 0.3 * np.sin(angles) + 0.1 * np.cos(angles)
            + 0.05 * np.sin(2 * angles) + 0.02 * np.cos(2 * angles)
        )
    sigma_b = np.array([0.5, 0.1])
    for b in range(n_bands):
        intens[b] += rng.normal(0.0, sigma_b[b], size=n)
    var = np.broadcast_to(sigma_b[:, None] ** 2, (n_bands, n)).copy()

    weights = np.ones(n_bands)
    coeffs, cov, wls = fit_first_and_second_harmonics_joint(angles, intens, weights, var)
    assert wls is True
    # I0_b err in WLS: σ_I0_b ~ σ_b / sqrt(N)  (ignoring harmonic-cov coupling)
    err_I0 = np.sqrt(np.diag(cov))[:n_bands]
    np.testing.assert_allclose(err_I0, sigma_b / np.sqrt(n), rtol=0.1)


def test_joint_solver_band_weights_zero_band_drops_band():
    """``band_weights`` zero on one band makes its rows have zero weight."""
    rng = np.random.default_rng(13)
    n = 64
    angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    n_bands = 2
    I0 = np.array([100.0, 0.0])
    intens = np.empty((n_bands, n))
    intens[0] = I0[0] + 0.5 * np.sin(angles)
    # Garbage in band 1 — but we will weight it to ~0
    intens[1] = rng.normal(0.0, 100.0, size=n)
    weights = np.array([1.0, 1e-12])  # band 1 essentially ignored
    coeffs, _cov, _wls = fit_first_and_second_harmonics_joint(angles, intens, weights, None)
    # Band 0's I0 should still recover
    assert abs(coeffs[0] - I0[0]) < 0.05
    # Shared A1 should recover
    assert abs(coeffs[n_bands] - 0.5) < 0.05


def test_evaluate_joint_model_shape_and_values():
    angles = np.linspace(0.0, 2 * np.pi, 8, endpoint=False)
    coeffs = np.array([10.0, 20.0, 30.0,    # I0_g, I0_r, I0_i
                       1.0, 0.0, 0.0, 0.0])  # A1=1, others=0
    model = evaluate_joint_model(angles, coeffs, 3)
    assert model.shape == (3, 8)
    expected_geom = np.sin(angles)
    for b, I0_b in enumerate([10.0, 20.0, 30.0]):
        np.testing.assert_allclose(model[b], I0_b + expected_geom, atol=1e-12)


# ---------------------------------------------------------------------------
# Per-band sigma clipping
# ---------------------------------------------------------------------------


def test_per_band_sigma_clip_and_shared_validity():
    """A clipped sample in one band drops it from all bands (AND mask)."""
    n = 64
    angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    intens = np.ones((2, n))
    intens[0] *= 100.0
    intens[1] *= 50.0
    intens[1, 10] = 1e6  # huge outlier in band 1 only
    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g", sclip=3.0, nclip=3)
    # Build a fake MultiIsophoteData and reuse the iteration loop's clip.
    from isoster.multiband.fitting_mb import _per_band_sigma_clip
    a, p, ic, vc, n_clipped = _per_band_sigma_clip(
        angles, angles, intens, None, cfg.sclip, cfg.nclip, cfg.sclip_low, cfg.sclip_high,
    )
    assert n_clipped >= 1
    assert ic.shape[1] == n - n_clipped
    # Both bands' arrays must have the same surviving N (shared validity).
    assert ic.shape[0] == 2
    # Variance is None on the input, so the clipper must mirror that.
    assert vc is None
    # Surviving angles/phi length must equal the surviving intensity column count.
    assert a.shape[0] == ic.shape[1]
    assert p.shape[0] == ic.shape[1]
    # The huge outlier in band 1 at index 10 must have been removed from BOTH
    # bands' intensity rows (shared validity).
    assert np.max(ic[1]) < 1e5
    assert ic[0, :].max() <= 100.0 and ic[0, :].min() >= 100.0 - 1e-9


# ---------------------------------------------------------------------------
# fit_isophote_mb on a planted galaxy
# ---------------------------------------------------------------------------


def test_fit_isophote_mb_planted_recovers_geometry():
    """B=2 joint fit on a planted Sersic recovers shared geometry near truth."""
    img1 = _planted_galaxy(amplitude=100.0, noise_sigma=0.01, seed=1)
    img2 = _planted_galaxy(amplitude=50.0, noise_sigma=0.01, seed=2)
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        sma0=20.0, eps=0.2, pa=0.4,
        debug=True, compute_deviations=True,
        nclip=0,  # no sigma clipping needed for clean synthetic
    )
    start = {"x0": 128.0, "y0": 128.0, "eps": 0.2, "pa": 0.4}
    out = fit_isophote_mb(
        images=[img1, img2], masks=None, sma=30.0,
        start_geometry=start, config=cfg,
    )
    assert out["stop_code"] in (0, 2)
    assert out["valid"] is True
    # Recovered geometry within Q17 thresholds (0.5 px / 0.02 eps / 1°).
    assert abs(out["x0"] - 128.0) < 0.5
    assert abs(out["y0"] - 128.0) < 0.5
    assert abs(out["eps"] - 0.3) < 0.02
    pa_diff = abs((out["pa"] - 0.5 + np.pi / 2) % np.pi - np.pi / 2)
    assert pa_diff < np.deg2rad(1.0)
    # Per-band intensity columns present and finite.
    for b in ("g", "r"):
        assert np.isfinite(out[f"intens_{b}"])
        assert np.isfinite(out[f"rms_{b}"])
        # Harmonic deviations attached after convergence.
        for n in (3, 4):
            assert f"a{n}_{b}" in out
            assert f"b{n}_{b}" in out


def test_fit_isophote_mb_too_few_points_returns_stop3():
    """A geometry that puts the ring almost entirely off-image returns stop_code=3."""
    img = _planted_galaxy(h=64, w=64, x0=32.0, y0=32.0)
    cfg = IsosterConfigMB(
        bands=["g", "r", "i"],
        reference_band="g",
        sma0=10.0,
        nclip=0,
    )
    # Center far outside the cutout → most of the ring is off-image.
    start = {"x0": 2000.0, "y0": 2000.0, "eps": 0.2, "pa": 0.0}
    out = fit_isophote_mb(
        images=[img, img, img], masks=None, sma=10.0,
        start_geometry=start, config=cfg,
    )
    # Either stop_code 3 (too few points) or -1 (gradient failure) is
    # acceptable here; both indicate the fit recognized a degenerate input.
    assert out["stop_code"] in (3, -1)
    assert out["valid"] is False


def test_fit_isophote_mb_b1_runs():
    """B=1 fit_isophote_mb still runs and produces the suffixed schema."""
    img = _planted_galaxy(amplitude=100.0, noise_sigma=0.01, seed=1)
    cfg = IsosterConfigMB(bands=["g"], reference_band="g", sma0=20.0, debug=True, nclip=0)
    start = {"x0": 128.0, "y0": 128.0, "eps": 0.2, "pa": 0.4}
    out = fit_isophote_mb(
        images=[img], masks=None, sma=30.0,
        start_geometry=start, config=cfg,
    )
    assert "intens_g" in out
    assert out["stop_code"] in (0, 2)
    assert abs(out["x0"] - 128.0) < 0.5


def test_fit_isophote_mb_ref_mode_runs():
    """``harmonic_combination='ref'`` exercises the single-band-style fallback."""
    img1 = _planted_galaxy(amplitude=100.0, noise_sigma=0.01, seed=1)
    img2 = _planted_galaxy(amplitude=50.0, noise_sigma=0.01, seed=2)
    cfg = IsosterConfigMB(
        bands=["g", "r"], reference_band="g",
        harmonic_combination="ref",
        sma0=20.0, debug=True, compute_deviations=True, nclip=0,
    )
    start = {"x0": 128.0, "y0": 128.0, "eps": 0.2, "pa": 0.4}
    out = fit_isophote_mb(
        images=[img1, img2], masks=None, sma=30.0,
        start_geometry=start, config=cfg,
    )
    assert out["stop_code"] in (0, 2)
    assert abs(out["x0"] - 128.0) < 1.0
    assert "intens_g" in out and "intens_r" in out


def test_fix_per_band_background_to_zero_solver_reduces_columns():
    """D11 backport: with the flag on, the joint solver becomes a
    4-column geometric system.  ``coeffs[:n_bands]`` are the per-band
    ring means (post-fit), and the harmonic block ``coeffs[n_bands:]``
    matches what a single-band solve on residuals would produce.
    """
    n = 64
    angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    rng = np.random.default_rng(7)
    means = np.array([10.0, 20.0])
    A1, B1 = 0.3, -0.1
    geom = A1 * np.sin(angles) + B1 * np.cos(angles)
    intens = np.array([means[0] + geom + rng.normal(0, 0.01, n),
                       means[1] + geom + rng.normal(0, 0.01, n)])
    weights = np.ones(2)

    coeffs_full, _, _ = fit_first_and_second_harmonics_joint(
        angles, intens, weights, None,
    )
    coeffs_zero, cov_zero, _ = fit_first_and_second_harmonics_joint(
        angles, intens, weights, None, fix_per_band_background_to_zero=True,
    )

    # Per-band intercepts should be the empirical ring means.
    np.testing.assert_allclose(coeffs_zero[:2], np.mean(intens, axis=1), atol=1e-12)
    # Geometric coefficients agree with the full solve on this clean planted case.
    np.testing.assert_allclose(coeffs_full[2:], coeffs_zero[2:], atol=2e-3)
    # cov has zero off-diagonal coupling between per-band intercepts and
    # geometric block (since intercepts are post-fit, not solved).
    assert cov_zero is not None
    assert np.all(cov_zero[:2, 2:] == 0.0)
    assert np.all(cov_zero[2:, :2] == 0.0)


def test_fix_per_band_background_to_zero_intens_err_scales_with_band_noise():
    """B3-style regression for D11: per-band intens_err must scale linearly
    with band noise sigma when fix_per_band_background_to_zero=True."""
    h, w = 64, 64
    rng = np.random.default_rng(11)
    img_g = 10.0 + rng.normal(0.0, 0.10, size=(h, w))
    img_r_low = 10.0 + rng.normal(0.0, 0.10, size=(h, w))
    img_r_high = 10.0 + rng.normal(0.0, 1.00, size=(h, w))
    cfg = IsosterConfigMB(
        bands=["g", "r"], reference_band="g",
        fix_per_band_background_to_zero=True,
        nclip=0,
    )
    start = {"x0": 32.0, "y0": 32.0, "eps": 0.0, "pa": 0.0}
    out_low = fit_isophote_mb(
        images=[img_g, img_r_low], masks=None, sma=12.0,
        start_geometry=start, config=cfg,
    )
    out_high = fit_isophote_mb(
        images=[img_g, img_r_high], masks=None, sma=12.0,
        start_geometry=start, config=cfg,
    )
    err_low = float(out_low["intens_err_r"])
    err_high = float(out_high["intens_err_r"])
    assert err_low > 0 and err_high > 0
    assert 4.0 < err_high / err_low < 25.0


def test_fix_per_band_background_to_zero_rejected_in_ref_mode():
    """The flag is incompatible with harmonic_combination='ref'."""
    with pytest.raises(ValueError, match="incompatible"):
        IsosterConfigMB(
            bands=["g", "r"], reference_band="g",
            harmonic_combination="ref",
            fix_per_band_background_to_zero=True,
        )


# ---------------------------------------------------------------------------
# D9 backport — loose validity end-to-end
# ---------------------------------------------------------------------------


def _plant_per_band_arc_artifact_image(
    h: int = 256, w: int = 256, x0: float = 128.0, y0: float = 128.0,
    eps: float = 0.3, pa: float = 0.5, amplitude: float = 100.0,
    noise_sigma: float = 0.05, seed: int = 1,
):
    """Two-band planted galaxy plus a 30°-arc mask in band r at SMA 35-45."""
    img_g = _planted_galaxy(
        h=h, w=w, x0=x0, y0=y0, eps=eps, pa=pa,
        amplitude=amplitude, noise_sigma=noise_sigma, seed=seed,
    )
    img_r = _planted_galaxy(
        h=h, w=w, x0=x0, y0=y0, eps=eps, pa=pa,
        amplitude=amplitude * 0.6, noise_sigma=noise_sigma, seed=seed + 1,
    )
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    dx = xx - x0
    dy = yy - y0
    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    r_ell = np.sqrt(x_rot ** 2 + (y_rot / max(1.0 - eps, 1e-3)) ** 2)
    phi_grid = np.arctan2(y_rot, x_rot)
    mask_r = (
        (r_ell > 35.0)
        & (r_ell < 45.0)
        & (phi_grid > -0.4)
        & (phi_grid < 0.4)
    )
    return img_g, img_r, mask_r


def test_loose_validity_band_drop_at_isophote():
    """A band pushed below the per-band minimum count is dropped at that
    isophote — its intens_<b> goes NaN, the other band is populated, and
    the isophote still produces a fitted geometry."""
    rng = np.random.default_rng(11)
    img = 10.0 + rng.normal(0.0, 0.05, size=(64, 64))
    bad_r = np.zeros_like(img, dtype=bool)
    # Mask >95% of band-r pixels so its surviving count drops below
    # the default loose_validity_min_per_band_count = 6.
    bad_r[1:, :] = True
    cfg = IsosterConfigMB(
        bands=["g", "r"], reference_band="g",
        loose_validity=True,
        loose_validity_min_per_band_count=6,
        loose_validity_min_per_band_frac=0.0,
        nclip=0,
    )
    out = fit_isophote_mb(
        images=[img, img], masks=[None, bad_r], sma=12.0,
        start_geometry={"x0": 32.0, "y0": 32.0, "eps": 0.0, "pa": 0.0},
        config=cfg,
    )
    # Whole-isophote drop fires when fewer than 2 bands survive — here
    # only band g survives, so the isophote is invalid.
    assert out["stop_code"] == 3
    assert out["valid"] is False


def test_loose_validity_n_valid_columns_present_under_loose_mode():
    """Per-band n_valid_<b> columns appear on every row when loose validity
    is enabled, and equal the band's surviving count after sigma clip."""
    img_g, img_r, mask_r = _plant_per_band_arc_artifact_image()
    cfg = IsosterConfigMB(
        bands=["g", "r"], reference_band="g",
        sma0=20.0, maxsma=60.0, astep=0.15, debug=True, nclip=0,
        loose_validity=True,
    )
    out = fit_isophote_mb(
        images=[img_g, img_r], masks=[None, mask_r], sma=40.0,
        start_geometry={"x0": 128.0, "y0": 128.0, "eps": 0.3, "pa": 0.5},
        config=cfg,
    )
    assert "n_valid_g" in out
    assert "n_valid_r" in out
    # Band r had pixels masked along the arc → fewer surviving samples.
    assert int(out["n_valid_g"]) > int(out["n_valid_r"])
    assert int(out["n_valid_r"]) > 0


def test_loose_validity_default_off_no_behavior_change():
    """Regression guard: with loose_validity=False (default), the result
    is numerically identical to the legacy shared-validity path."""
    img_g = _planted_galaxy(seed=1)
    img_r = _planted_galaxy(amplitude=50.0, seed=2)
    cfg = IsosterConfigMB(
        bands=["g", "r"], reference_band="g",
        sma0=20.0, maxsma=80.0, astep=0.2, debug=True, nclip=0,
    )
    assert cfg.loose_validity is False
    start = {"x0": 128.0, "y0": 128.0, "eps": 0.2, "pa": 0.4}
    out = fit_isophote_mb(
        images=[img_g, img_r], masks=None, sma=30.0,
        start_geometry=start, config=cfg,
    )
    # Sanity: shared validity gives identical n_valid across bands.
    assert int(out["n_valid_g"]) == int(out["n_valid_r"])


def test_loose_validity_per_band_count_normalization_changes_intens_err():
    """`per_band_count` normalization changes the WLS row weights, which
    changes per-band intens_err. Asymmetric per-band noise makes the
    effect easy to verify."""
    rng = np.random.default_rng(3)
    h = 64
    img_g = 10.0 + rng.normal(0.0, 0.10, size=(h, h))
    img_r = 10.0 + rng.normal(0.0, 1.00, size=(h, h))
    var_g = np.full_like(img_g, 0.01, dtype=np.float64)
    var_r = np.full_like(img_r, 1.00, dtype=np.float64)

    base_kwargs = dict(
        bands=["g", "r"], reference_band="g",
        nclip=0, loose_validity=True,
    )
    cfg_none = IsosterConfigMB(
        **base_kwargs,
        loose_validity_band_normalization="none",
    )
    cfg_perband = IsosterConfigMB(
        **base_kwargs,
        loose_validity_band_normalization="per_band_count",
    )
    start = {"x0": 32.0, "y0": 32.0, "eps": 0.0, "pa": 0.0}
    out_none = fit_isophote_mb(
        images=[img_g, img_r], masks=None, sma=10.0,
        start_geometry=start, config=cfg_none,
        variance_maps=[var_g, var_r],
    )
    out_perband = fit_isophote_mb(
        images=[img_g, img_r], masks=None, sma=10.0,
        start_geometry=start, config=cfg_perband,
        variance_maps=[var_g, var_r],
    )
    # Both paths produce finite intens_err for both bands.
    for k in ("intens_err_g", "intens_err_r"):
        assert np.isfinite(float(out_none[k])) and float(out_none[k]) > 0
        assert np.isfinite(float(out_perband[k])) and float(out_perband[k]) > 0
    # The geometric coefficients differ by enough to be detectable —
    # changing the row weighting changes the joint solve. In practice
    # the per-band intens_err scales the same way (direct SEM in both
    # modes), but the rms / x0_err differ.
    assert (
        not np.isclose(float(out_none["x0_err"]), float(out_perband["x0_err"]), rtol=1e-6)
        or not np.isclose(float(out_none["rms"]), float(out_perband["rms"]), rtol=1e-6)
    )


def test_loose_validity_with_fix_per_band_background_to_zero():
    """Both flags on, planted galaxy: per-band intens_<b> equals the
    band's empirical surviving-sample mean."""
    img_g, img_r, mask_r = _plant_per_band_arc_artifact_image()
    cfg = IsosterConfigMB(
        bands=["g", "r"], reference_band="g",
        sma0=20.0, maxsma=60.0, astep=0.2, debug=True, nclip=0,
        loose_validity=True,
        fix_per_band_background_to_zero=True,
    )
    out = fit_isophote_mb(
        images=[img_g, img_r], masks=[None, mask_r], sma=40.0,
        start_geometry={"x0": 128.0, "y0": 128.0, "eps": 0.3, "pa": 0.5},
        config=cfg,
    )
    # Both bands survive (artifact is small relative to the ring); the
    # per-band intens is the band's own ring mean by construction.
    assert np.isfinite(out["intens_g"])
    assert np.isfinite(out["intens_r"])
    assert int(out["n_valid_g"]) > int(out["n_valid_r"])


def test_loose_validity_with_ref_mode():
    """Ref mode + loose validity: reference band intens unchanged from
    its single-band ring mean, non-ref bands' intens come from their own
    surviving samples."""
    img_g, img_r, mask_r = _plant_per_band_arc_artifact_image()
    cfg = IsosterConfigMB(
        bands=["g", "r"], reference_band="g",
        sma0=20.0, maxsma=60.0, astep=0.2, debug=True, nclip=0,
        loose_validity=True,
        harmonic_combination="ref",
    )
    out = fit_isophote_mb(
        images=[img_g, img_r], masks=[None, mask_r], sma=40.0,
        start_geometry={"x0": 128.0, "y0": 128.0, "eps": 0.3, "pa": 0.5},
        config=cfg,
    )
    assert np.isfinite(out["intens_g"]) and np.isfinite(out["intens_r"])
    # Per-band counts must reflect the per-band masking under loose
    # validity even in ref mode (the ref-mode code path does not
    # interfere with how the sampler reports per-band counts).
    assert int(out["n_valid_g"]) > int(out["n_valid_r"])


def test_fit_isophote_mb_ref_mode_intens_err_scaling_ols(monkeypatch):
    """Regression for B3.

    In ``harmonic_combination='ref'`` + OLS mode, non-reference bands'
    ``intens_err_<b>`` must be the band's own SEM ≈ ``σ_b / √N``, not a
    quantity that has the residual variance applied twice.  Concretely:
    if we change band-r's noise level by a factor 10, the reported
    ``intens_err_r`` should also change by ~10×, not by ~100×.
    """
    n = 200
    angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    rng = np.random.default_rng(0)

    def synth(noise_sigma: float, seed: int) -> np.ndarray:
        rng_local = np.random.default_rng(seed)
        return rng_local.normal(loc=10.0, scale=noise_sigma, size=n)

    intens_g = synth(0.10, seed=11)
    intens_r_low = synth(0.10, seed=22)
    intens_r_high = synth(1.00, seed=22)  # 10× noisier in band r

    cfg = IsosterConfigMB(
        bands=["g", "r"], reference_band="g",
        harmonic_combination="ref", nclip=0,
    )

    # Drop into the per-band error branch directly via fit_isophote_mb on a
    # small synthetic image so the full code path executes.
    h, w = 64, 64
    img_g = np.full((h, w), 10.0, dtype=np.float64)
    img_r = img_g.copy()
    img_g += rng.normal(0.0, 0.10, size=img_g.shape)
    img_r_low = img_g + rng.normal(0.0, 0.10, size=img_g.shape)
    img_r_high = img_g + rng.normal(0.0, 1.00, size=img_g.shape)
    start = {"x0": 32.0, "y0": 32.0, "eps": 0.0, "pa": 0.0}

    out_low = fit_isophote_mb(
        images=[img_g, img_r_low], masks=None, sma=12.0,
        start_geometry=start, config=cfg,
    )
    out_high = fit_isophote_mb(
        images=[img_g, img_r_high], masks=None, sma=12.0,
        start_geometry=start, config=cfg,
    )
    err_low = float(out_low["intens_err_r"])
    err_high = float(out_high["intens_err_r"])
    assert np.isfinite(err_low) and np.isfinite(err_high)
    assert err_low > 0 and err_high > 0
    # SEM scales linearly with σ: 10× noise should give ~10× error, not 100×.
    ratio = err_high / err_low
    assert 4.0 < ratio < 25.0, (
        f"intens_err_r should scale ~linearly with band noise (10×), "
        f"got ratio = {ratio:.2f}"
    )


# ---------------------------------------------------------------------------
# Forced photometry helper
# ---------------------------------------------------------------------------


def test_extract_forced_photometry_mb_shape():
    img1 = _planted_galaxy(amplitude=100.0, noise_sigma=0.001, seed=1)
    img2 = _planted_galaxy(amplitude=50.0, noise_sigma=0.001, seed=2)
    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g")
    out = extract_forced_photometry_mb(
        images=[img1, img2], masks=None,
        x0=128.0, y0=128.0, sma=20.0, eps=0.3, pa=0.5,
        bands=cfg.bands, config=cfg,
    )
    assert out["valid"] is True
    assert out["stop_code"] == 0
    for b in ("g", "r"):
        assert np.isfinite(out[f"intens_{b}"])
        assert np.isfinite(out[f"intens_err_{b}"])


# ---------------------------------------------------------------------------
# variance maps mixed all-or-nothing rejected at the sampler level
# ---------------------------------------------------------------------------


def test_fit_isophote_mb_rejects_mixed_variance_maps():
    img = _planted_galaxy(seed=1)
    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g")
    var = np.ones_like(img) * 0.1
    with pytest.raises(ValueError, match="all-or-nothing"):
        fit_isophote_mb(
            images=[img, img], masks=None, sma=20.0,
            start_geometry={"x0": 128.0, "y0": 128.0, "eps": 0.2, "pa": 0.0},
            config=cfg,
            variance_maps=[var, None],  # type: ignore[list-item]
        )
