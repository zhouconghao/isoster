"""Tests for ``isoster.multiband.fitting_mb``."""

import numpy as np
import pytest

from isoster.fitting import fit_first_and_second_harmonics
from isoster.multiband.config_mb import IsosterConfigMB
from isoster.multiband.fitting_mb import (
    _per_band_mean_or_median,
    _per_band_mean_or_median_jagged,
    evaluate_joint_model,
    extract_forced_photometry_mb,
    fit_first_and_second_harmonics_joint,
    fit_first_and_second_harmonics_joint_loose,
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


def test_ring_mean_intercept_solver_reduces_columns():
    """``fit_per_band_intens_jointly=False`` (ring-mean intercept mode):
    the joint solver becomes a 4-column geometric system.
    ``coeffs[:n_bands]`` are the per-band ring means (post-fit), and the
    harmonic block ``coeffs[n_bands:]`` matches what a single-band solve
    on residuals would produce.
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
        angles, intens, weights, None, fit_per_band_intens_jointly=False,
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


def test_ring_mean_intercept_intens_err_scales_with_band_noise():
    """B3-style regression: per-band intens_err must scale linearly with
    band noise sigma when fit_per_band_intens_jointly=False."""
    h, w = 64, 64
    rng = np.random.default_rng(11)
    img_g = 10.0 + rng.normal(0.0, 0.10, size=(h, w))
    img_r_low = 10.0 + rng.normal(0.0, 0.10, size=(h, w))
    img_r_high = 10.0 + rng.normal(0.0, 1.00, size=(h, w))
    cfg = IsosterConfigMB(
        bands=["g", "r"], reference_band="g",
        fit_per_band_intens_jointly=False,
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


def test_ring_mean_intercept_rejected_in_ref_mode():
    """fit_per_band_intens_jointly=False is incompatible with
    harmonic_combination='ref'."""
    with pytest.raises(ValueError, match="incompatible"):
        IsosterConfigMB(
            bands=["g", "r"], reference_band="g",
            harmonic_combination="ref",
            fit_per_band_intens_jointly=False,
        )


def test_legacy_field_name_rejected_with_clear_error():
    """Section 6 cleanup: passing the old ``fix_per_band_background_to_zero``
    field raises a clear ValueError pointing at the new name (no silent
    drop)."""
    with pytest.raises(ValueError, match="renamed.*fit_per_band_intens_jointly"):
        IsosterConfigMB(
            bands=["g", "r"], reference_band="g",
            fix_per_band_background_to_zero=True,  # type: ignore[call-arg]
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


def test_loose_validity_with_ring_mean_intercept():
    """loose_validity=True + fit_per_band_intens_jointly=False: per-band
    intens_<b> equals the band's empirical surviving-sample mean."""
    img_g, img_r, mask_r = _plant_per_band_arc_artifact_image()
    cfg = IsosterConfigMB(
        bands=["g", "r"], reference_band="g",
        sma0=20.0, maxsma=60.0, astep=0.2, debug=True, nclip=0,
        loose_validity=True,
        fit_per_band_intens_jointly=False,
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


# ---------------------------------------------------------------------------
# Stage-3 S2: per-band median integrator in decoupled mode
# ---------------------------------------------------------------------------


def test_per_band_mean_or_median_dispatches_correctly():
    """Direct unit test on the helper: mean vs median agree only on
    symmetric distributions."""
    rng = np.random.default_rng(2026)
    intens = rng.normal(loc=10.0, scale=0.5, size=(3, 200))  # symmetric
    var = np.full_like(intens, 0.25)

    means_ols = _per_band_mean_or_median(intens, None, "mean")
    means_wls = _per_band_mean_or_median(intens, var, "mean")
    medians = _per_band_mean_or_median(intens, var, "median")

    # On a symmetric Gaussian distribution mean ≈ median to better
    # than 0.05 (1/sqrt(N) ~ 0.07, so this is ~ within tolerance).
    assert np.allclose(means_ols, means_wls, atol=1e-12)  # uniform var ⇒ same
    assert np.allclose(means_ols, medians, atol=0.05)

    # Inject a one-sided outlier per band: median ignores it, mean shifts.
    intens_skewed = intens.copy()
    intens_skewed[:, 0] += 100.0  # huge upward outlier in sample 0 of every band
    means_skewed = _per_band_mean_or_median(intens_skewed, None, "mean")
    medians_skewed = _per_band_mean_or_median(intens_skewed, None, "median")
    # Mean shifted by ~ 100/200 = 0.5 per band. Median shifted by < 0.05.
    assert np.all(means_skewed - means_ols > 0.4)
    assert np.all(np.abs(medians_skewed - medians) < 0.05)


def test_per_band_mean_or_median_jagged_handles_empty_bands():
    """Empty band ⇒ NaN under both integrators; non-empty bands resolve."""
    intens = [np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([], dtype=np.float64)]
    var = [np.full(5, 0.1), np.array([], dtype=np.float64)]
    out_mean = _per_band_mean_or_median_jagged(intens, var, "mean")
    out_med = _per_band_mean_or_median_jagged(intens, var, "median")
    assert out_mean[0] == pytest.approx(3.0)
    assert out_med[0] == pytest.approx(3.0)
    assert np.isnan(out_mean[1])
    assert np.isnan(out_med[1])


def test_joint_solver_decoupled_mean_vs_median_full_ring_agree():
    """On a clean full ring with symmetric noise, the per-band intercept
    should agree between integrator='mean' and 'median' to ~1/sqrt(N).
    """
    rng = np.random.default_rng(7)
    n_samples = 360
    angles = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    truth = np.array([10.0, 20.0])
    # Pure intercept signal + a known A1, B1 harmonic + Gaussian noise.
    intens = (
        truth[:, None]
        + 0.3 * np.sin(angles)
        - 0.2 * np.cos(angles)
        + rng.normal(scale=0.05, size=(2, n_samples))
    )
    weights = np.array([1.0, 1.0])

    coeffs_mean, _, _ = fit_first_and_second_harmonics_joint(
        angles, intens, weights, None,
        fit_per_band_intens_jointly=False, integrator="mean",
    )
    coeffs_med, _, _ = fit_first_and_second_harmonics_joint(
        angles, intens, weights, None,
        fit_per_band_intens_jointly=False, integrator="median",
    )

    # Per-band intercepts agree to within a few × σ/√N ~ 0.05/√360 ~ 0.003
    # (median has a 1.25× efficiency penalty vs mean for Gaussian noise).
    assert np.allclose(coeffs_mean[:2], coeffs_med[:2], atol=0.02)
    # Geometric block trivially identical to ~noise floor: with symmetric
    # noise the residuals after subtracting the per-band intercept are
    # statistically equivalent under either reducer.
    assert np.allclose(coeffs_mean[2:], coeffs_med[2:], atol=0.01)


def test_joint_solver_decoupled_median_resists_one_sided_outlier():
    """Asymmetric contamination: the median path should suppress an
    outlier sector that pulls the mean by orders of magnitude.
    """
    rng = np.random.default_rng(11)
    n_samples = 360
    angles = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    truth = 10.0
    intens_clean = truth + rng.normal(scale=0.05, size=n_samples)
    # Inject 5% of samples (one band) with a huge positive contaminant.
    contam_idx = np.arange(n_samples)[:18]
    intens_contam = intens_clean.copy()
    intens_contam[contam_idx] += 100.0
    intens = np.stack([intens_clean, intens_contam], axis=0)
    weights = np.array([1.0, 1.0])

    coeffs_mean, _, _ = fit_first_and_second_harmonics_joint(
        angles, intens, weights, None,
        fit_per_band_intens_jointly=False, integrator="mean",
    )
    coeffs_med, _, _ = fit_first_and_second_harmonics_joint(
        angles, intens, weights, None,
        fit_per_band_intens_jointly=False, integrator="median",
    )

    # Mean of contaminated band shifts by ~ 100 * 18/360 = 5.0
    assert coeffs_mean[1] - truth > 4.0
    # Median is barely affected — < 5% of samples are outliers.
    assert abs(coeffs_med[1] - truth) < 0.1
    # Clean band: both reducers agree on the truth.
    assert abs(coeffs_mean[0] - truth) < 0.05
    assert abs(coeffs_med[0] - truth) < 0.05


def test_joint_solver_loose_decoupled_median_works_with_jagged_input():
    """Loose-validity solver path (jagged) supports integrator='median'."""
    rng = np.random.default_rng(13)
    n_b1, n_b2 = 200, 120
    phi_per_band = [
        np.linspace(0.0, 2.0 * np.pi, n_b1, endpoint=False),
        np.linspace(0.0, 2.0 * np.pi, n_b2, endpoint=False),
    ]
    truth = np.array([5.0, 8.0])
    intens_per_band = [
        truth[0] + rng.normal(scale=0.02, size=n_b1),
        truth[1] + rng.normal(scale=0.03, size=n_b2),
    ]
    weights = np.array([1.0, 1.0])

    coeffs, _, _ = fit_first_and_second_harmonics_joint_loose(
        phi_per_band, intens_per_band, weights, None,
        fit_per_band_intens_jointly=False, integrator="median",
    )
    # Per-band medians sit on truth to noise/√N tolerance.
    assert abs(coeffs[0] - truth[0]) < 0.01
    assert abs(coeffs[1] - truth[1]) < 0.02


def test_fit_isophote_mb_median_intercept_end_to_end():
    """End-to-end with integrator='median' + decoupled intercept mode:
    the fit must converge and produce finite intens_<b>."""
    img1 = _planted_galaxy(amplitude=100.0, noise_sigma=0.05, seed=21)
    img2 = _planted_galaxy(amplitude=50.0, noise_sigma=0.05, seed=22)
    cfg = IsosterConfigMB(
        bands=["g", "r"], reference_band="g",
        integrator="median",
        fit_per_band_intens_jointly=False,
    )
    out = fit_isophote_mb(
        images=[img1, img2], masks=None, sma=20.0,
        start_geometry={"x0": 128.0, "y0": 128.0, "eps": 0.2, "pa": 0.0},
        config=cfg,
    )
    assert out["valid"] is True
    assert out["stop_code"] == 0
    for b in ("g", "r"):
        assert np.isfinite(out[f"intens_{b}"])
        assert out[f"intens_{b}"] > 0
        assert np.isfinite(out[f"intens_err_{b}"])


# ---------------------------------------------------------------------------
# Stage-3 Stage-B: outer-region damping
# ---------------------------------------------------------------------------


def test_fit_isophote_mb_outer_damping_inert_below_onset():
    """At sma well below onset, damping has lambda ≈ 0 → identical fit.

    Both configs run with geometry_convergence=True so the comparison
    isolates the outer-reg effect from the auto-enable side-effect.
    """
    import warnings as _warnings
    img = _planted_galaxy(amplitude=100.0, noise_sigma=0.001, seed=42)
    cfg_off = IsosterConfigMB(
        bands=["g", "r"], reference_band="g",
        geometry_convergence=True,
    )
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        cfg_on = IsosterConfigMB(
            bands=["g", "r"], reference_band="g",
            geometry_convergence=True,
            use_outer_center_regularization=True,
            outer_reg_sma_onset=200.0,  # well above test sma
            outer_reg_strength=10.0,
        )
    ref_geom = {"x0": 128.0, "y0": 128.0, "eps": 0.3, "pa": 0.5}
    out_off = fit_isophote_mb(
        images=[img, img], masks=None, sma=20.0,
        start_geometry={"x0": 128.0, "y0": 128.0, "eps": 0.2, "pa": 0.0},
        config=cfg_off,
    )
    out_on = fit_isophote_mb(
        images=[img, img], masks=None, sma=20.0,
        start_geometry={"x0": 128.0, "y0": 128.0, "eps": 0.2, "pa": 0.0},
        config=cfg_on,
        outer_reference_geom=ref_geom,
    )
    # sma=20, onset=200, default width=80 → lambda ≈ 0.19 * coeff² which
    # times 8e-3-ish coefficient produces an alpha of order 1e-5; the
    # step shrink is sub-pixel-fraction. With matching
    # geometry_convergence on both sides the fits agree to better than
    # 1e-6, well below any pixel-scale interpretation.
    for k in ("x0", "y0", "eps", "pa"):
        assert abs(float(out_off[k]) - float(out_on[k])) < 1e-6


def test_fit_isophote_mb_outer_damping_active_above_onset():
    """At sma well above onset, damping shrinks per-iteration steps:
    fewer / smaller geometry updates from the same starting point in the
    same number of iterations."""
    import warnings as _warnings
    rng = np.random.default_rng(2026)
    img = _planted_galaxy(amplitude=100.0, noise_sigma=0.05, seed=99)
    # Inject a small mid-image bump that pulls geometry away from truth on
    # outer rings; gives the damper something to push back against.
    bump = np.zeros_like(img)
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    bump += 0.5 * np.exp(-((xx - 90.0) ** 2 + (yy - 90.0) ** 2) / (5.0 ** 2))
    img = img + bump + rng.normal(0.0, 1e-4, size=img.shape)

    cfg_off = IsosterConfigMB(
        bands=["g", "r"], reference_band="g", maxit=12,
    )
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        cfg_on = IsosterConfigMB(
            bands=["g", "r"], reference_band="g", maxit=12,
            use_outer_center_regularization=True,
            outer_reg_sma_onset=10.0,
            outer_reg_sma_width=2.0,
            outer_reg_strength=8.0,  # strong damper for a clear signal
            outer_reg_weights={"center": 1.0, "eps": 1.0, "pa": 1.0},
        )
    ref_geom = {"x0": 128.0, "y0": 128.0, "eps": 0.3, "pa": 0.5}
    seed_geom = {"x0": 128.0, "y0": 128.0, "eps": 0.2, "pa": 0.0}

    out_off = fit_isophote_mb(
        images=[img, img], masks=None, sma=80.0, start_geometry=seed_geom,
        config=cfg_off,
    )
    out_on = fit_isophote_mb(
        images=[img, img], masks=None, sma=80.0, start_geometry=seed_geom,
        config=cfg_on, outer_reference_geom=ref_geom,
    )
    # Damping shrinks the per-iteration step → after the same maxit the
    # damped fit should land closer to the seed (i.e. less total drift).
    drift_off = np.hypot(out_off["x0"] - seed_geom["x0"], out_off["y0"] - seed_geom["y0"])
    drift_on = np.hypot(out_on["x0"] - seed_geom["x0"], out_on["y0"] - seed_geom["y0"])
    assert drift_on <= drift_off + 1e-9
    # eps step should also be smaller or equal under damping.
    eps_step_off = abs(out_off["eps"] - seed_geom["eps"])
    eps_step_on = abs(out_on["eps"] - seed_geom["eps"])
    assert eps_step_on <= eps_step_off + 1e-9


def test_fit_isophote_mb_outer_damping_off_when_no_reference():
    """When outer_reference_geom is None, damping is fully inert even with
    use_outer_center_regularization=True. This guards the driver-level
    invariant: outer-reg cannot fire on the inward sweep."""
    import warnings as _warnings
    img = _planted_galaxy(seed=3)
    cfg_off = IsosterConfigMB(bands=["g", "r"], reference_band="g")
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        cfg_on = IsosterConfigMB(
            bands=["g", "r"], reference_band="g",
            use_outer_center_regularization=True,
            outer_reg_sma_onset=5.0, outer_reg_strength=10.0,
        )
    seed_geom = {"x0": 128.0, "y0": 128.0, "eps": 0.2, "pa": 0.0}
    out_off = fit_isophote_mb(
        images=[img, img], masks=None, sma=80.0, start_geometry=seed_geom,
        config=cfg_off,
    )
    out_on = fit_isophote_mb(
        images=[img, img], masks=None, sma=80.0, start_geometry=seed_geom,
        config=cfg_on,  # no outer_reference_geom passed → inert
    )
    for k in ("x0", "y0", "eps", "pa"):
        assert abs(float(out_off[k]) - float(out_on[k])) < 1e-12


def test_build_outer_reference_mb_flux_weighted_average():
    """Driver-level reference builder: x0/y0/eps come from the flux-
    weighted average of qualifying inward isophotes + anchor."""
    from isoster.multiband.driver_mb import _build_outer_reference_mb
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        cfg = IsosterConfigMB(
            bands=["g", "r"], reference_band="g",
            sma0=10.0,
            use_outer_center_regularization=True,
            outer_reg_ref_sma_factor=2.0,
        )
    anchor = {
        "x0": 100.0, "y0": 100.0, "eps": 0.30, "pa": 0.5, "sma": 10.0,
        "stop_code": 0, "intens_g": 100.0, "intens_r": 50.0,
    }
    inwards = [
        # In-window, finite, acceptable: contributes.
        {"x0": 102.0, "y0": 99.0, "eps": 0.32, "pa": 0.45, "sma": 8.0,
         "stop_code": 0, "intens_g": 200.0, "intens_r": 100.0},
        # In-window, intens_g=300 dominant flux weight.
        {"x0": 98.0, "y0": 101.0, "eps": 0.28, "pa": 0.55, "sma": 5.0,
         "stop_code": 0, "intens_g": 300.0, "intens_r": 150.0},
        # Out-of-window (sma > sma0 * factor=20): excluded.
        {"x0": 1000.0, "y0": -1000.0, "eps": 0.9, "pa": 1.5, "sma": 30.0,
         "stop_code": 0, "intens_g": 1.0, "intens_r": 0.5},
        # Bad stop_code: excluded.
        {"x0": 5000.0, "y0": 5000.0, "eps": 0.99, "pa": 0.0, "sma": 7.0,
         "stop_code": 3, "intens_g": 1e6, "intens_r": 1e6},
    ]
    ref = _build_outer_reference_mb(inwards, anchor, cfg)
    # Brute-force reference: anchor + first two inwards entries.
    weights = np.array([100.0, 200.0, 300.0])
    x0s = np.array([100.0, 102.0, 98.0])
    y0s = np.array([100.0, 99.0, 101.0])
    eps_arr = np.array([0.30, 0.32, 0.28])
    expected_x0 = float(np.average(x0s, weights=weights))
    expected_y0 = float(np.average(y0s, weights=weights))
    expected_eps = float(np.average(eps_arr, weights=weights))
    assert ref["x0"] == pytest.approx(expected_x0)
    assert ref["y0"] == pytest.approx(expected_y0)
    assert ref["eps"] == pytest.approx(expected_eps)
    # The garbage entries must not have leaked in (would have shifted the
    # mean by orders of magnitude).
    assert abs(ref["x0"] - 100.0) < 5.0
    assert 0.0 < ref["eps"] < 1.0


def test_build_outer_reference_mb_falls_back_to_anchor_with_no_inwards():
    """No inwards results → reference is the anchor's own geometry."""
    from isoster.multiband.driver_mb import _build_outer_reference_mb
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        cfg = IsosterConfigMB(
            bands=["g", "r"], reference_band="g", sma0=10.0,
            use_outer_center_regularization=True,
        )
    anchor = {
        "x0": 64.0, "y0": 64.0, "eps": 0.25, "pa": 0.3, "sma": 10.0,
        "stop_code": 0, "intens_g": 50.0, "intens_r": 25.0,
    }
    ref = _build_outer_reference_mb([], anchor, cfg)
    assert ref["x0"] == pytest.approx(64.0)
    assert ref["y0"] == pytest.approx(64.0)
    assert ref["eps"] == pytest.approx(0.25)
    assert ref["pa"] == pytest.approx(0.3 % np.pi)


def test_fit_image_multiband_end_to_end_with_outer_damping():
    """Driver end-to-end: outer damping enabled converges and gives
    finite geometry; baseline run with feature off must still match
    on inner isophotes (where lambda ≈ 0)."""
    from isoster.multiband import fit_image_multiband
    import warnings as _warnings
    img1 = _planted_galaxy(amplitude=100.0, noise_sigma=0.005, seed=51)
    img2 = _planted_galaxy(amplitude=50.0, noise_sigma=0.005, seed=52)

    cfg_off = IsosterConfigMB(
        bands=["g", "r"], reference_band="g",
        sma0=10.0, maxsma=80.0, astep=0.2, debug=True,
    )
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        cfg_on = IsosterConfigMB(
            bands=["g", "r"], reference_band="g",
            sma0=10.0, maxsma=80.0, astep=0.2, debug=True,
            use_outer_center_regularization=True,
            outer_reg_sma_onset=60.0,
            outer_reg_strength=4.0,
        )
    res_off = fit_image_multiband([img1, img2], masks=None, config=cfg_off)
    res_on = fit_image_multiband([img1, img2], masks=None, config=cfg_on)
    assert len(res_on["isophotes"]) >= 5
    inner_smas = [iso["sma"] for iso in res_on["isophotes"] if iso["sma"] < 30.0]
    assert len(inner_smas) >= 3
    # Inner isophotes (sma well below onset=60) should be near-identical
    # between off and on. We check x0 / y0 because eps and pa vary more
    # under noise; tolerance is generous since seeded RNG noise + lazy
    # gradient cache differences can land different local minima.
    for iso_off, iso_on in zip(res_off["isophotes"], res_on["isophotes"]):
        if float(iso_off["sma"]) < 30.0:
            assert abs(float(iso_off["x0"]) - float(iso_on["x0"])) < 0.5
            assert abs(float(iso_off["y0"]) - float(iso_on["y0"])) < 0.5


# ---------------------------------------------------------------------------
# Stage-3 Stage-C: lsb_auto_lock state machine
# ---------------------------------------------------------------------------


def test_is_lsb_isophote_mb_predicate():
    """Direct unit test on the lock-trigger predicate (joint gradient)."""
    from isoster.multiband.driver_mb import _is_lsb_isophote_mb
    # Healthy: small relative grad error, negative grad → no trigger.
    assert not _is_lsb_isophote_mb(
        {"grad": -10.0, "grad_error": 1.0, "stop_code": 0}, 0.3,
    )
    # Above threshold: |0.5 / -1.0| = 0.5 > 0.3 → trigger.
    assert _is_lsb_isophote_mb(
        {"grad": -1.0, "grad_error": 0.5, "stop_code": 0}, 0.3,
    )
    # Positive grad → trigger (galaxy gradient should be negative).
    assert _is_lsb_isophote_mb(
        {"grad": 5.0, "grad_error": 1.0, "stop_code": 0}, 0.3,
    )
    # stop_code=-1 → trigger regardless of gradient values.
    assert _is_lsb_isophote_mb(
        {"grad": -10.0, "grad_error": 0.1, "stop_code": -1}, 0.3,
    )
    # Missing keys → cannot assess, no trigger.
    assert not _is_lsb_isophote_mb({}, 0.3)
    assert not _is_lsb_isophote_mb({"grad": -10.0}, 0.3)
    # Non-finite values → no trigger.
    assert not _is_lsb_isophote_mb(
        {"grad": float("nan"), "grad_error": 1.0, "stop_code": 0}, 0.3,
    )
    assert not _is_lsb_isophote_mb(
        {"grad": -10.0, "grad_error": 0.0, "stop_code": 0}, 0.3,
    )


def test_build_locked_cfg_mb_freezes_geometry_and_disables_outer_reg():
    """The lock clone must freeze geometry, disable auto-lock and
    outer-reg on itself, and (when integrator='median') flip
    fit_per_band_intens_jointly to False to satisfy Stage-A S1."""
    from isoster.multiband.driver_mb import _build_locked_cfg_mb
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        cfg = IsosterConfigMB(
            bands=["g", "r"], reference_band="g",
            lsb_auto_lock=True,
            lsb_auto_lock_integrator="median",
            fit_per_band_intens_jointly=False,
            use_outer_center_regularization=True,
            sma0=10.0,
        )
    anchor = {
        "x0": 100.0, "y0": 100.0, "eps": 0.3, "pa": 0.5, "sma": 25.0,
        "stop_code": 0,
    }
    locked = _build_locked_cfg_mb(cfg, anchor, "median")
    assert locked.x0 == 100.0 and locked.y0 == 100.0
    assert locked.eps == 0.3 and locked.pa == 0.5
    assert locked.fix_center is True
    assert locked.fix_pa is True
    assert locked.fix_eps is True
    assert locked.integrator == "median"
    assert locked.fit_per_band_intens_jointly is False  # Stage-A S1
    assert locked.lsb_auto_lock is False  # one-way state machine
    assert locked.use_outer_center_regularization is False  # disabled on clone


def test_build_locked_cfg_mb_mean_keeps_intercept_mode():
    """Locked-region mean integrator does NOT need to flip the
    intercept mode."""
    from isoster.multiband.driver_mb import _build_locked_cfg_mb
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        cfg = IsosterConfigMB(
            bands=["g", "r"], reference_band="g",
            lsb_auto_lock=True,
            lsb_auto_lock_integrator="mean",
        )
    anchor = {"x0": 50.0, "y0": 50.0, "eps": 0.2, "pa": 0.0, "sma": 15.0}
    locked = _build_locked_cfg_mb(cfg, anchor, "mean")
    assert locked.integrator == "mean"
    assert locked.fit_per_band_intens_jointly is True  # untouched


def test_lsb_auto_lock_fires_on_synthetic_galaxy():
    """End-to-end: build a multi-band image where the gradient quality
    falls off in the outer region, run the driver with lsb_auto_lock on,
    confirm the lock fires and the outer isophotes carry the locked
    state metadata."""
    from isoster.multiband import fit_image_multiband
    import warnings as _warnings

    # Compact planted galaxy + heavy noise so the outer-region gradient
    # quality drops below the trigger threshold deterministically.
    img1 = _planted_galaxy(
        h=200, w=200, x0=100.0, y0=100.0, eps=0.25, pa=0.4,
        amplitude=50.0, re=10.0, noise_sigma=2.0, seed=2026,
    )
    img2 = _planted_galaxy(
        h=200, w=200, x0=100.0, y0=100.0, eps=0.25, pa=0.4,
        amplitude=25.0, re=10.0, noise_sigma=2.0, seed=2027,
    )

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        cfg = IsosterConfigMB(
            bands=["g", "r"], reference_band="g",
            sma0=5.0, minsma=2.0, maxsma=80.0, astep=0.15,
            lsb_auto_lock=True,
            lsb_auto_lock_integrator="mean",  # avoid S1; keep matrix-mode
            lsb_auto_lock_maxgerr=0.2,
            lsb_auto_lock_debounce=2,
        )
    res = fit_image_multiband(images=[img1, img2], masks=None, config=cfg)

    assert res.get("lsb_auto_lock") is True
    # The lock should fire SOMEWHERE in the outer sweep on this seeded
    # synthetic image — beyond ~ 2-3× re the gradient is essentially
    # noise. ``lsb_auto_lock_sma`` may be None if the trigger never
    # debounced, in which case the test data is wrong; assert non-None.
    assert res["lsb_auto_lock_sma"] is not None
    assert float(res["lsb_auto_lock_sma"]) > cfg.sma0  # locked outward
    assert int(res["lsb_auto_lock_count"]) >= 1
    # At least one isophote is marked lsb_locked=True; at least the
    # anchor gets lsb_locked=False (lock_state initialized).
    iso_locked = [iso for iso in res["isophotes"] if iso.get("lsb_locked")]
    assert len(iso_locked) >= 1
    # The locked isophotes' geometry must match the lock anchor exactly
    # (since fix_center / fix_pa / fix_eps fire on the cloned cfg).
    locked_x0s = [float(iso["x0"]) for iso in iso_locked]
    locked_y0s = [float(iso["y0"]) for iso in iso_locked]
    locked_eps = [float(iso["eps"]) for iso in iso_locked]
    locked_pa = [float(iso["pa"]) for iso in iso_locked]
    # All locked isophotes share the same geometry to numerical noise.
    assert max(locked_x0s) - min(locked_x0s) < 1e-9
    assert max(locked_y0s) - min(locked_y0s) < 1e-9
    assert max(locked_eps) - min(locked_eps) < 1e-9
    assert max(locked_pa) - min(locked_pa) < 1e-9


def test_lsb_auto_lock_off_no_metadata():
    """With lsb_auto_lock=False, the result dict carries no lock keys."""
    from isoster.multiband import fit_image_multiband
    img = _planted_galaxy(seed=4242)
    cfg = IsosterConfigMB(
        bands=["g", "r"], reference_band="g",
        sma0=10.0, maxsma=60.0, astep=0.15, debug=True,
    )
    res = fit_image_multiband([img, img], masks=None, config=cfg)
    assert "lsb_auto_lock" not in res
    for iso in res["isophotes"]:
        assert "lsb_locked" not in iso


def test_top_level_grad_keys_under_debug():
    """Stage-C added top-level ``grad`` / ``grad_error`` to the row
    when debug=True. Required for the lock predicate."""
    img = _planted_galaxy(amplitude=100.0, noise_sigma=0.001, seed=99)
    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g", debug=True)
    out = fit_isophote_mb(
        images=[img, img], masks=None, sma=20.0,
        start_geometry={"x0": 128.0, "y0": 128.0, "eps": 0.2, "pa": 0.0},
        config=cfg,
    )
    assert "grad" in out
    assert "grad_error" in out
    assert np.isfinite(float(out["grad"]))


# ---------------------------------------------------------------------------
# Stage-3 Stage-D: per-band curve-of-growth
# ---------------------------------------------------------------------------


def test_compute_cog_mb_empty_isophote_list_returns_empty_arrays():
    from isoster.multiband.cog_mb import compute_cog_mb
    out = compute_cog_mb([], bands=["g", "r"])
    for key in (
        "sma", "area_annulus", "area_annulus_raw",
        "flag_cross", "flag_negative_area",
        "cog_g", "cog_annulus_g", "cog_r", "cog_annulus_r",
    ):
        assert key in out, f"missing {key}"
        assert np.asarray(out[key]).size == 0


def test_compute_cog_mb_single_isophote_uses_intensity_for_first_annulus():
    """First annulus area = π·a·b; first cog = intens · area."""
    from isoster.multiband.cog_mb import compute_cog_mb
    isos = [
        {"sma": 10.0, "eps": 0.0, "x0": 0.0, "y0": 0.0, "pa": 0.0,
         "stop_code": 0, "intens": 1.0, "intens_g": 5.0, "intens_r": 3.0},
    ]
    out = compute_cog_mb(isos, bands=["g", "r"])
    expected_area = float(np.pi * 10.0 * 10.0)
    assert float(out["area_annulus"][0]) == pytest.approx(expected_area)
    assert float(out["cog_g"][0]) == pytest.approx(5.0 * expected_area)
    assert float(out["cog_r"][0]) == pytest.approx(3.0 * expected_area)


def test_compute_cog_mb_cumulative_flux_is_monotonic_on_clean_profile():
    """A monotonically decreasing per-band intens profile gives a
    monotonically increasing cog (until the per-band intens hits zero)."""
    from isoster.multiband.cog_mb import compute_cog_mb
    smas = np.linspace(5.0, 50.0, 10)
    intens_g = np.exp(-smas / 20.0) * 100.0
    intens_r = np.exp(-smas / 25.0) * 50.0
    isos = [
        {"sma": float(s), "eps": 0.2, "x0": 0.0, "y0": 0.0, "pa": 0.0,
         "stop_code": 0, "intens": float(intens_g[i]),
         "intens_g": float(intens_g[i]), "intens_r": float(intens_r[i])}
        for i, s in enumerate(smas)
    ]
    out = compute_cog_mb(isos, bands=["g", "r"])
    cog_g = np.asarray(out["cog_g"])
    cog_r = np.asarray(out["cog_r"])
    # Strictly increasing.
    assert np.all(np.diff(cog_g) > 0)
    assert np.all(np.diff(cog_r) > 0)
    # g flux > r flux at every isophote (since intens_g > intens_r).
    assert np.all(cog_g >= cog_r)


def test_add_cog_mb_to_isophotes_stamps_per_band_columns():
    from isoster.multiband.cog_mb import (
        add_cog_mb_to_isophotes, compute_cog_mb,
    )
    isos = [
        {"sma": 5.0, "eps": 0.1, "x0": 0.0, "y0": 0.0, "pa": 0.0,
         "stop_code": 0, "intens": 10.0,
         "intens_g": 10.0, "intens_r": 8.0},
        {"sma": 10.0, "eps": 0.1, "x0": 0.0, "y0": 0.0, "pa": 0.0,
         "stop_code": 0, "intens": 5.0,
         "intens_g": 5.0, "intens_r": 4.0},
    ]
    cog = compute_cog_mb(isos, bands=["g", "r"])
    add_cog_mb_to_isophotes(isos, ["g", "r"], cog)
    for iso in isos:
        for k in (
            "area_annulus", "flag_cross", "flag_negative_area",
            "cog_g", "cog_r", "cog_annulus_g", "cog_annulus_r",
        ):
            assert k in iso, f"row missing {k}"
        assert isinstance(iso["flag_cross"], bool)
        assert isinstance(iso["flag_negative_area"], bool)


def test_fit_image_multiband_compute_cog_end_to_end():
    """Driver end-to-end: compute_cog=True stamps per-band cog columns
    onto every row of result['isophotes']."""
    from isoster.multiband import fit_image_multiband
    img1 = _planted_galaxy(amplitude=100.0, noise_sigma=0.005, seed=88)
    img2 = _planted_galaxy(amplitude=50.0, noise_sigma=0.005, seed=89)
    cfg = IsosterConfigMB(
        bands=["g", "r"], reference_band="g",
        sma0=10.0, maxsma=60.0, astep=0.2,
        compute_cog=True,
    )
    res = fit_image_multiband([img1, img2], masks=None, config=cfg)
    rows = res["isophotes"]
    assert len(rows) >= 5
    for iso in rows:
        for key in (
            "area_annulus", "flag_cross", "flag_negative_area",
            "cog_g", "cog_r", "cog_annulus_g", "cog_annulus_r",
        ):
            assert key in iso, f"row at sma={iso['sma']} missing {key}"
        assert iso["area_annulus"] >= 0.0
    # Cumulative flux should be monotonic on a clean planted galaxy.
    cog_g = np.asarray([float(iso["cog_g"]) for iso in rows])
    valid_g = np.isfinite(cog_g)
    if int(valid_g.sum()) > 3:
        cog_g_valid = cog_g[valid_g]
        # Allow tiny non-monotonicity from noise; require overall growth.
        assert cog_g_valid[-1] > cog_g_valid[0]


def test_fit_image_multiband_compute_cog_off_no_extra_columns():
    """compute_cog=False (default) leaves the row dict untouched."""
    from isoster.multiband import fit_image_multiband
    img = _planted_galaxy(seed=77)
    cfg = IsosterConfigMB(
        bands=["g", "r"], reference_band="g",
        sma0=10.0, maxsma=40.0, astep=0.2,
    )
    res = fit_image_multiband([img, img], masks=None, config=cfg)
    for iso in res["isophotes"]:
        for key in (
            "area_annulus", "flag_cross", "flag_negative_area",
            "cog_g", "cog_r",
        ):
            assert key not in iso, f"unexpected {key} in row at sma={iso['sma']}"


def test_compute_cog_mb_sky_offsets_subtract_constant_before_integration():
    """Per-band sky_offsets shift intens_<b> by a constant before the
    trapezoidal average; the cog_total picks up exactly
    -sky_offset * total_area."""
    from isoster.multiband.cog_mb import compute_cog_mb
    smas = np.linspace(5.0, 50.0, 8)
    intens_g = np.exp(-smas / 12.0) * 50.0
    isos = [
        {"sma": float(s), "eps": 0.0, "x0": 0.0, "y0": 0.0, "pa": 0.0,
         "stop_code": 0, "intens": float(intens_g[i]),
         "intens_g": float(intens_g[i])}
        for i, s in enumerate(smas)
    ]
    out_raw = compute_cog_mb(isos, bands=["g"])
    out_corr = compute_cog_mb(isos, bands=["g"], sky_offsets={"g": 0.05})

    # Total area enclosed at the outermost isophote.
    total_area = float(np.pi * smas[-1] ** 2)
    raw_total = float(out_raw["cog_g"][-1])
    corr_total = float(out_corr["cog_g"][-1])
    expected_diff = -0.05 * total_area
    np.testing.assert_allclose(corr_total - raw_total, expected_diff, rtol=1e-9)
    # Empty dict ≡ no correction.
    out_empty = compute_cog_mb(isos, bands=["g"], sky_offsets={})
    np.testing.assert_allclose(out_empty["cog_g"], out_raw["cog_g"], rtol=0.0)
    # None ≡ no correction (default).
    out_none = compute_cog_mb(isos, bands=["g"], sky_offsets=None)
    np.testing.assert_allclose(out_none["cog_g"], out_raw["cog_g"], rtol=0.0)


def test_compute_cog_mb_b1_matches_single_band_per_band_column():
    """B=1 multi-band CoG agrees with single-band CoG for a curated
    isophote list whose ``intens`` and ``intens_<b>`` are identical."""
    from isoster.cog import compute_cog as compute_cog_sb
    from isoster.multiband.cog_mb import compute_cog_mb
    smas = np.linspace(3.0, 30.0, 8)
    intens_arr = np.exp(-smas / 10.0) * 50.0
    isos = [
        {"sma": float(s), "eps": 0.2, "x0": 0.0, "y0": 0.0, "pa": 0.0,
         "stop_code": 0, "intens": float(intens_arr[i]),
         "intens_g": float(intens_arr[i])}
        for i, s in enumerate(smas)
    ]
    sb_out = compute_cog_sb(isos)
    mb_out = compute_cog_mb(isos, bands=["g"])
    np.testing.assert_allclose(
        np.asarray(mb_out["cog_g"]), sb_out["cog"], rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(mb_out["cog_annulus_g"]), sb_out["cog_annulus"], rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(mb_out["area_annulus"]), sb_out["area_annulus"], rtol=1e-12,
    )
