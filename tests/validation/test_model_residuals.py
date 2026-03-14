"""Test script to examine build_ellipse_model() residuals."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from isoster import fit_image
from isoster.config import IsosterConfig
from isoster.model import build_isoster_model
from isoster.output_paths import resolve_output_directory
from tests.fixtures import create_sersic_model as _create_sersic_model


def compute_residual_statistics(data, model, R_e, x0, y0):
    """Compute residual statistics in different radial ranges.

    Per CLAUDE.md:
    - Fractional residual: 100.0 * (model - data) / data
    - Ranges: <0.5 Re, 0.5-4 Re, 4-8 Re
    """
    h, w = data.shape
    y = np.arange(h)
    x = np.arange(w)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    r = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)

    # Avoid division by zero
    data_safe = np.where(np.abs(data) > 1e-10, data, np.nan)

    # Fractional residual (%)
    frac_resid = 100.0 * (model - data) / data_safe
    frac_abs_resid = 100.0 * np.abs(model - data) / data_safe

    # Define radial ranges
    mask_inner = r < 0.5 * R_e
    mask_mid = (r >= 0.5 * R_e) & (r < 4 * R_e)
    mask_outer = (r >= 4 * R_e) & (r < 8 * R_e)

    stats = {}
    for name, mask in [("inner", mask_inner), ("mid", mask_mid), ("outer", mask_outer)]:
        if np.any(mask):
            valid = mask & np.isfinite(frac_resid)
            if np.any(valid):
                stats[f"{name}_median_frac"] = np.nanmedian(frac_resid[valid])
                stats[f"{name}_max_abs_frac"] = np.nanmax(np.abs(frac_resid[valid]))
                stats[f"{name}_median_abs_frac"] = np.nanmedian(frac_abs_resid[valid])
            else:
                stats[f"{name}_median_frac"] = np.nan
                stats[f"{name}_max_abs_frac"] = np.nan
                stats[f"{name}_median_abs_frac"] = np.nan
        else:
            stats[f"{name}_median_frac"] = np.nan
            stats[f"{name}_max_abs_frac"] = np.nan
            stats[f"{name}_median_abs_frac"] = np.nan

    return stats, frac_resid


def plot_residual_analysis(data, model, R_e, x0, y0, stats, output_path):
    """Create diagnostic plot for model residuals."""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Original data
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(data, origin="lower", cmap="viridis")
    ax1.set_title("Original Data")
    ax1.axhline(y0, color="red", ls="--", alpha=0.3)
    ax1.axvline(x0, color="red", ls="--", alpha=0.3)
    plt.colorbar(im1, ax=ax1)

    # 2. Model
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(model, origin="lower", cmap="viridis")
    ax2.set_title("Reconstructed Model")
    ax2.axhline(y0, color="red", ls="--", alpha=0.3)
    ax2.axvline(x0, color="red", ls="--", alpha=0.3)
    plt.colorbar(im2, ax=ax2)

    # 3. Residual (data - model)
    ax3 = fig.add_subplot(gs[0, 2])
    residual = data - model
    vmax = np.nanpercentile(np.abs(residual), 99)
    im3 = ax3.imshow(residual, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax3.set_title("Residual (Data - Model)")
    plt.colorbar(im3, ax=ax3)

    # 4. Fractional residual
    ax4 = fig.add_subplot(gs[1, 0])
    _, frac_resid = compute_residual_statistics(data, model, R_e, x0, y0)
    vmax_frac = np.nanpercentile(np.abs(frac_resid), 99)
    im4 = ax4.imshow(frac_resid, origin="lower", cmap="RdBu_r", vmin=-vmax_frac, vmax=vmax_frac)
    ax4.set_title("Fractional Residual (%)")
    plt.colorbar(im4, ax=ax4, label="%")

    # 5. Radial profile comparison
    ax5 = fig.add_subplot(gs[1, 1])
    h, w = data.shape
    y = np.arange(h)
    x = np.arange(w)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    r = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)

    # Bin by radius
    r_bins = np.linspace(0, 8 * R_e, 50)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    data_profile = []
    model_profile = []

    for i in range(len(r_bins) - 1):
        mask = (r >= r_bins[i]) & (r < r_bins[i + 1])
        if np.any(mask):
            data_profile.append(np.nanmedian(data[mask]))
            model_profile.append(np.nanmedian(model[mask]))
        else:
            data_profile.append(np.nan)
            model_profile.append(np.nan)

    ax5.plot(r_centers / R_e, data_profile, "o-", label="Data", alpha=0.7)
    ax5.plot(r_centers / R_e, model_profile, "s-", label="Model", alpha=0.7)
    ax5.set_xlabel("r / Re")
    ax5.set_ylabel("Intensity")
    ax5.set_yscale("log")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_title("Radial Profile")

    # 6. Statistics summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")

    text = "Residual Statistics:\n\n"
    text += "<0.5 Re:\n"
    text += f"  Median: {stats['inner_median_frac']:.3f}%\n"
    text += f"  Max |frac|: {stats['inner_max_abs_frac']:.3f}%\n"
    text += f"  Median |frac|: {stats['inner_median_abs_frac']:.3f}%\n\n"

    text += "0.5-4 Re:\n"
    text += f"  Median: {stats['mid_median_frac']:.3f}%\n"
    text += f"  Max |frac|: {stats['mid_max_abs_frac']:.3f}%\n"
    text += f"  Median |frac|: {stats['mid_median_abs_frac']:.3f}%\n\n"

    text += "4-8 Re:\n"
    text += f"  Median: {stats['outer_median_frac']:.3f}%\n"
    text += f"  Max |frac|: {stats['outer_max_abs_frac']:.3f}%\n"
    text += f"  Median |frac|: {stats['outer_median_abs_frac']:.3f}%\n"

    ax6.text(0.1, 0.9, text, transform=ax6.transAxes, verticalalignment="top", fontfamily="monospace", fontsize=10)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved residual analysis to {output_path}")
    plt.close()

    return stats


def test_model_building():
    """Test build_isoster_model() with noiseless Sersic profile."""
    print("=" * 60)
    print("Testing build_isoster_model() with noiseless Sersic n=4")
    print("=" * 60)

    # Test parameters (same as integration test)
    R_e = 20.0
    n = 4.0
    I_e = 1000.0
    eps = 0.4
    pa = np.pi / 4
    oversample = 10

    # Create model
    print(f"\nCreating Sersic model: n={n}, Re={R_e}, eps={eps:.2f}, pa={pa:.2f}")
    data, _, params = _create_sersic_model(R_e, n, I_e, eps, pa, oversample=oversample)
    x0, y0, shape = params["x0"], params["y0"], params["shape"]
    print(f"Image shape: {shape}, center: ({x0}, {y0})")

    # Fit with isoster
    print("\nFitting with isoster...")
    config = IsosterConfig(
        x0=x0,
        y0=y0,
        eps=eps,
        pa=pa,
        sma0=10.0,
        minsma=3.0,
        maxsma=8 * R_e,
        astep=0.1,
        minit=10,
        maxit=50,
        conver=0.05,
        use_eccentric_anomaly=True,
    )

    results = fit_image(data, None, config)
    isophotes = results["isophotes"]

    converged = [iso for iso in isophotes if iso["stop_code"] == 0]
    print(f"Fitted {len(isophotes)} isophotes, {len(converged)} converged")

    # Build model
    print("\nBuilding model...")
    model = build_isoster_model(shape, isophotes)

    # Compute statistics
    print("\nComputing residual statistics...")
    stats, _ = compute_residual_statistics(data, model, R_e, x0, y0)

    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    for key, val in stats.items():
        print(f"{key:25s}: {val:8.3f}%")

    # Check against CLAUDE.md criteria for noiseless mock (0.5-4 Re range)
    mid_max = stats["mid_max_abs_frac"]
    mid_median = stats["mid_median_abs_frac"]

    print(f"\n0.5-4 Re: max |frac| = {mid_max:.3f}%, median |frac| = {mid_median:.3f}%")

    # Create diagnostic plot
    output_dir = resolve_output_directory("tests_validation", "model_residuals")
    plot_residual_analysis(data, model, R_e, x0, y0, stats, output_dir / "model_residuals_current.png")

    # Assertions: noiseless Sersic n=4 model should reconstruct well in 0.5-4 Re
    assert np.isfinite(mid_max), "mid_max_abs_frac is not finite"
    assert np.isfinite(mid_median), "mid_median_abs_frac is not finite"
    assert mid_max < 5.0, f"Max |fractional residual| in 0.5-4 Re = {mid_max:.3f}% exceeds 5%"
    assert mid_median < 2.0, f"Median |fractional residual| in 0.5-4 Re = {mid_median:.3f}% exceeds 2%"


if __name__ == "__main__":
    stats, max_err, med_err = test_model_building()
