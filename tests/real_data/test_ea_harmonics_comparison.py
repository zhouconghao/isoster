"""
Comprehensive EA harmonics comparison on real galaxy data.

This test module compares different isophote fitting methods on real elliptical
galaxies (ESO 243-49 and NGC 3610) to evaluate the benefits of:
1. Eccentric anomaly (EA) based sampling
2. Simultaneous higher-order harmonics fitting
3. Extended harmonic orders (up to n=10)

Tests:
1. photutils.isophote fitting (reference)
2. isoster default (PA mode)
3. isoster EA mode with harmonics [3, 4]
4. isoster EA mode with extended harmonics [3..10]

Outputs:
- Timing benchmarks
- 1-D profile comparison figures
- 2-D residual map figures
- JSON results for further analysis

Run with: pytest tests/real_data/test_ea_harmonics_comparison.py -v -m real_data -s
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

# Import isoster
from isoster import fit_image
from isoster.config import IsosterConfig
from isoster.model import build_isoster_model
from isoster.output_paths import resolve_output_directory

# Import photutils
try:
    from photutils.isophote import Ellipse, EllipseGeometry

    HAS_PHOTUTILS = True
except ImportError:
    HAS_PHOTUTILS = False


# =============================================================================
# Data paths
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data"
ESO243_PATH = DATA_DIR / "eso243-49.fits"
NGC3610_PATH = DATA_DIR / "ngc3610.fits"

# Output directory
OUTPUT_DIR = resolve_output_directory("tests_real_data", "ea_harmonics_comparison")


# =============================================================================
# Helper Functions
# =============================================================================


def load_galaxy_image(fits_path, band_index=1):
    """
    Load a single band from a multi-band FITS file.

    Args:
        fits_path: Path to FITS file
        band_index: Band index (0=g, 1=r/r, 2=z/i depending on survey)

    Returns:
        2D numpy array of the image
    """
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
        if data.ndim == 3:
            return data[band_index].astype(np.float64)
        return data.astype(np.float64)


def find_galaxy_center(image, initial_guess=None):
    """
    Find galaxy center using isophote fitting.

    Strategy:
    1. If no guess, use brightest pixel
    2. Run quick isoster fit with fix_center=False
    3. Return median (x0, y0) from inner isophotes

    Args:
        image: 2D image array
        initial_guess: (x0, y0) initial guess or None

    Returns:
        (x0, y0) center coordinates
    """
    if initial_guess is None:
        # Find brightest pixel as initial guess
        y_peak, x_peak = np.unravel_index(np.argmax(image), image.shape)
        initial_guess = (float(x_peak), float(y_peak))

    # Quick fit to refine center
    config = IsosterConfig(
        x0=initial_guess[0],
        y0=initial_guess[1],
        sma0=5.0,
        minsma=3.0,
        maxsma=30.0,
        astep=0.2,
        eps=0.3,
        pa=0.0,
        fix_center=False,
        maxit=30,
        compute_deviations=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = fit_image(image, mask=None, config=config)

    # Use median center from inner converged isophotes
    inner = [iso for iso in results["isophotes"] if iso["sma"] < 20 and iso["stop_code"] == 0]

    if len(inner) < 3:
        # Fall back to initial guess
        return initial_guess

    x0 = np.median([iso["x0"] for iso in inner])
    y0 = np.median([iso["y0"] for iso in inner])

    return x0, y0


def benchmark_fit(fit_func, n_runs=3, **kwargs):
    """
    Run a fit function multiple times and return timing statistics.

    Args:
        fit_func: Callable that returns results dict
        n_runs: Number of timing runs
        **kwargs: Arguments to pass to fit_func

    Returns:
        dict with 'results', 'mean_time', 'std_time', 'times'
    """
    times = []
    results = None

    for _ in range(n_runs):
        start = time.perf_counter()
        results = fit_func(**kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "results": results,
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "times": times,
    }


def run_photutils(image, x0, y0, sma0=10.0, maxsma=100.0, minsma=3.0, step=0.1, eps0=0.3, pa0=0.0):
    """
    Run photutils.isophote fitting.

    Args:
        image: 2D image array
        x0, y0: Center coordinates
        sma0: Starting semi-major axis
        maxsma: Maximum SMA to fit
        minsma: Minimum SMA to include in output (filter after fitting)
        step: SMA step size
        eps0: Initial ellipticity
        pa0: Initial position angle

    Returns:
        List of isophote dicts in isoster format
    """
    if not HAS_PHOTUTILS:
        return []

    geometry = EllipseGeometry(x0, y0, sma=sma0, eps=eps0, pa=pa0)
    ellipse = Ellipse(image, geometry)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        isolist = ellipse.fit_image(maxsma=maxsma, step=step, linear=False)

    # Convert to isoster dict format
    isophotes = []
    for iso in isolist:
        if iso.sma >= minsma:  # Skip very inner isophotes
            isophotes.append(
                {
                    "sma": iso.sma,
                    "intens": iso.intens,
                    "intens_err": iso.int_err if hasattr(iso, "int_err") else iso.intens * 0.01,
                    "eps": iso.eps,
                    "eps_err": iso.ellip_err if hasattr(iso, "ellip_err") else 0.01,
                    "pa": iso.pa,
                    "pa_err": iso.pa_err if hasattr(iso, "pa_err") else 0.01,
                    "x0": iso.x0,
                    "y0": iso.y0,
                    "a3": iso.a3 if hasattr(iso, "a3") else 0.0,
                    "b3": iso.b3 if hasattr(iso, "b3") else 0.0,
                    "a4": iso.a4 if hasattr(iso, "a4") else 0.0,
                    "b4": iso.b4 if hasattr(iso, "b4") else 0.0,
                    "stop_code": 0 if iso.stop_code == 0 else -1,
                    "niter": iso.niter if hasattr(iso, "niter") else 0,
                }
            )

    return isophotes


def run_isoster(
    image,
    x0,
    y0,
    sma0=10.0,
    maxsma=100.0,
    minsma=3.0,
    step=0.1,
    eps0=0.3,
    pa0=0.0,
    use_ea=False,
    simultaneous=False,
    harmonic_orders=None,
):
    """
    Run isoster fitting with specified configuration.

    Returns:
        List of isophote dicts
    """
    if harmonic_orders is None:
        harmonic_orders = [3, 4]

    config = IsosterConfig(
        x0=x0,
        y0=y0,
        sma0=sma0,
        minsma=minsma,
        maxsma=maxsma,
        astep=step,
        eps=eps0,
        pa=pa0,
        use_eccentric_anomaly=use_ea,
        simultaneous_harmonics=simultaneous,
        harmonic_orders=harmonic_orders,
        compute_deviations=True,
        compute_errors=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = fit_image(image, mask=None, config=config)

    return results["isophotes"]


def compute_comparison_metrics(isophotes_test, isophotes_ref, sma_range=(10, 80)):
    """
    Compute comparison metrics between test and reference isophotes.

    Args:
        isophotes_test: Test isophotes list
        isophotes_ref: Reference (photutils) isophotes list
        sma_range: (min_sma, max_sma) for comparison

    Returns:
        dict of metrics
    """
    # Filter to converged isophotes in range
    test_conv = [iso for iso in isophotes_test if iso["stop_code"] == 0 and sma_range[0] <= iso["sma"] <= sma_range[1]]
    ref_conv = [iso for iso in isophotes_ref if iso["stop_code"] == 0 and sma_range[0] <= iso["sma"] <= sma_range[1]]

    if len(test_conv) < 3 or len(ref_conv) < 3:
        return {"error": "Insufficient converged isophotes"}

    # Extract arrays
    sma_test = np.array([iso["sma"] for iso in test_conv])
    intens_test = np.array([iso["intens"] for iso in test_conv])
    eps_test = np.array([iso["eps"] for iso in test_conv])
    pa_test = np.array([iso["pa"] for iso in test_conv])

    sma_ref = np.array([iso["sma"] for iso in ref_conv])
    intens_ref = np.array([iso["intens"] for iso in ref_conv])
    eps_ref = np.array([iso["eps"] for iso in ref_conv])
    pa_ref = np.array([iso["pa"] for iso in ref_conv])

    # Interpolate reference to test SMA grid
    intens_ref_interp = np.interp(sma_test, sma_ref, intens_ref)
    eps_ref_interp = np.interp(sma_test, sma_ref, eps_ref)
    pa_ref_interp = np.interp(sma_test, sma_ref, pa_ref)

    # Intensity fractional difference
    intens_frac_diff = np.abs(intens_test - intens_ref_interp) / intens_ref_interp * 100

    # Ellipticity absolute difference
    eps_diff = np.abs(eps_test - eps_ref_interp)

    # PA difference (handle wrapping)
    pa_diff = np.abs(pa_test - pa_ref_interp)
    pa_diff = np.minimum(pa_diff, np.pi - pa_diff)
    pa_diff_deg = np.degrees(pa_diff)

    return {
        "n_test": len(test_conv),
        "n_ref": len(ref_conv),
        "intens_diff_median": float(np.median(intens_frac_diff)),
        "intens_diff_max": float(np.max(intens_frac_diff)),
        "eps_diff_median": float(np.median(eps_diff)),
        "eps_diff_max": float(np.max(eps_diff)),
        "pa_diff_median": float(np.median(pa_diff_deg)),
        "pa_diff_max": float(np.max(pa_diff_deg)),
    }


def compute_residual_stats(image, model, sma_ranges=None):
    """
    Compute 2D residual statistics within annular regions.

    Args:
        image: Original image
        model: Model image
        sma_ranges: List of (inner, outer) SMA ranges

    Returns:
        dict of residual statistics per region
    """
    if sma_ranges is None:
        sma_ranges = [(0, 20), (20, 50), (50, 100)]

    h, w = image.shape
    y0, x0 = h / 2, w / 2  # Approximate center
    y, x = np.mgrid[:h, :w]
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

    # Fractional residual
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_resid = (image - model) / image * 100
        frac_resid = np.where(np.isfinite(frac_resid) & (image > 0), frac_resid, np.nan)

    stats = {}
    for inner, outer in sma_ranges:
        mask = (r >= inner) & (r < outer) & np.isfinite(frac_resid)
        if np.sum(mask) > 10:
            resid_region = frac_resid[mask]
            stats[f"{inner}-{outer}"] = {
                "median": float(np.nanmedian(resid_region)),
                "rms": float(np.nanstd(resid_region)),
                "n_pixels": int(np.sum(mask)),
            }
        else:
            stats[f"{inner}-{outer}"] = {"error": "Too few pixels"}

    return stats


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def galaxy_data():
    """
    Load both galaxies and find centers.

    Returns dict with galaxy data including images and centers.
    """
    data = {}

    # ESO 243-49 (use r-band, index=1)
    if ESO243_PATH.exists():
        image_eso = load_galaxy_image(ESO243_PATH, band_index=1)
        x0_eso, y0_eso = find_galaxy_center(image_eso)
        data["eso243-49"] = {
            "image": image_eso,
            "band": "r",
            "center": (x0_eso, y0_eso),
            "path": str(ESO243_PATH),
        }
        print(f"\nESO 243-49 loaded: shape={image_eso.shape}, center=({x0_eso:.1f}, {y0_eso:.1f})")

    # NGC 3610 (use i-band, index=2)
    if NGC3610_PATH.exists():
        image_ngc = load_galaxy_image(NGC3610_PATH, band_index=2)
        x0_ngc, y0_ngc = find_galaxy_center(image_ngc)
        data["ngc3610"] = {
            "image": image_ngc,
            "band": "i",
            "center": (x0_ngc, y0_ngc),
            "path": str(NGC3610_PATH),
        }
        print(f"NGC 3610 loaded: shape={image_ngc.shape}, center=({x0_ngc:.1f}, {y0_ngc:.1f})")

    if not data:
        pytest.skip("No galaxy data files found")

    return data


# =============================================================================
# Test Class
# =============================================================================


@pytest.mark.real_data
class TestEAHarmonicsComparison:
    """
    Compare EA harmonics fitting methods on real galaxy data.
    """

    def test_data_loading(self, galaxy_data):
        """Verify data files are loaded correctly."""
        assert len(galaxy_data) > 0, "No galaxy data loaded"

        for name, gal in galaxy_data.items():
            assert "image" in gal, f"{name}: missing image"
            assert gal["image"].shape == (256, 256), f"{name}: unexpected shape"
            assert "center" in gal, f"{name}: missing center"
            x0, y0 = gal["center"]
            # Center should be within image
            assert 0 < x0 < 256, f"{name}: x0 out of bounds"
            assert 0 < y0 < 256, f"{name}: y0 out of bounds"

        print("\n--- Data Loading Test Passed ---")

    @pytest.mark.skipif(not HAS_PHOTUTILS, reason="photutils not installed")
    def test_all_methods_comparison(self, galaxy_data):
        """
        Run all fitting methods on both galaxies and compare results.

        This is the main comprehensive test that:
        1. Runs photutils as reference
        2. Runs isoster PA mode
        3. Runs isoster EA mode with [3,4] harmonics
        4. Runs isoster EA mode with [3..10] harmonics
        5. Compares all methods
        6. Generates QA figures
        """
        all_results = {}
        benchmark_results = {}

        for galaxy_name, gal in galaxy_data.items():
            print(f"\n{'=' * 60}")
            print(f"Processing {galaxy_name}")
            print(f"{'=' * 60}")

            image = gal["image"]
            x0, y0 = gal["center"]

            # Common parameters
            fit_params = dict(
                image=image,
                x0=x0,
                y0=y0,
                sma0=10.0,
                maxsma=100.0,
                minsma=3.0,
                step=0.1,
                eps0=0.3,
                pa0=0.0,
            )

            results = {}
            timing = {}

            # 1. photutils reference
            print("\n1. Running photutils.isophote (reference)...")
            bench = benchmark_fit(lambda **kw: run_photutils(**kw), n_runs=3, **fit_params)
            results["photutils"] = bench["results"]
            timing["photutils"] = {"mean": bench["mean_time"], "std": bench["std_time"]}
            n_iso = len(results["photutils"])
            n_conv = sum(1 for iso in results["photutils"] if iso["stop_code"] == 0)
            print(f"   Isophotes: {n_iso}, Converged: {n_conv}, Time: {bench['mean_time']:.3f}s")

            # 2. isoster PA mode (default)
            print("\n2. Running isoster (PA mode, default)...")
            bench = benchmark_fit(
                lambda **kw: run_isoster(**kw, use_ea=False, simultaneous=False), n_runs=3, **fit_params
            )
            results["isoster_pa"] = bench["results"]
            timing["isoster_pa"] = {"mean": bench["mean_time"], "std": bench["std_time"]}
            n_iso = len(results["isoster_pa"])
            n_conv = sum(1 for iso in results["isoster_pa"] if iso["stop_code"] == 0)
            print(f"   Isophotes: {n_iso}, Converged: {n_conv}, Time: {bench['mean_time']:.3f}s")

            # 3. isoster EA mode with [3, 4]
            print("\n3. Running isoster (EA mode, harmonics [3,4])...")
            bench = benchmark_fit(
                lambda **kw: run_isoster(**kw, use_ea=True, simultaneous=True, harmonic_orders=[3, 4]),
                n_runs=3,
                **fit_params,
            )
            results["isoster_ea_34"] = bench["results"]
            timing["isoster_ea_34"] = {"mean": bench["mean_time"], "std": bench["std_time"]}
            n_iso = len(results["isoster_ea_34"])
            n_conv = sum(1 for iso in results["isoster_ea_34"] if iso["stop_code"] == 0)
            print(f"   Isophotes: {n_iso}, Converged: {n_conv}, Time: {bench['mean_time']:.3f}s")

            # 4. isoster EA mode with extended harmonics [3..10]
            print("\n4. Running isoster (EA mode, harmonics [3..10])...")
            bench = benchmark_fit(
                lambda **kw: run_isoster(**kw, use_ea=True, simultaneous=True, harmonic_orders=list(range(3, 11))),
                n_runs=3,
                **fit_params,
            )
            results["isoster_ea_310"] = bench["results"]
            timing["isoster_ea_310"] = {"mean": bench["mean_time"], "std": bench["std_time"]}
            n_iso = len(results["isoster_ea_310"])
            n_conv = sum(1 for iso in results["isoster_ea_310"] if iso["stop_code"] == 0)
            print(f"   Isophotes: {n_iso}, Converged: {n_conv}, Time: {bench['mean_time']:.3f}s")

            # Compare all methods vs photutils
            print("\n--- Comparison vs photutils ---")
            comparison = {}
            for method in ["isoster_pa", "isoster_ea_34", "isoster_ea_310"]:
                metrics = compute_comparison_metrics(results[method], results["photutils"])
                comparison[method] = metrics
                if "error" not in metrics:
                    print(f"   {method}:")
                    print(
                        f"      Intensity diff: {metrics['intens_diff_median']:.2f}% median, {metrics['intens_diff_max']:.2f}% max"
                    )
                    print(f"      Eps diff: {metrics['eps_diff_median']:.4f} median")
                    print(f"      PA diff: {metrics['pa_diff_median']:.2f} deg median")

            # Build models and compute 2D residuals
            print("\n--- 2D Residual Statistics ---")
            model_results = {}
            for method, isos in results.items():
                if len(isos) > 5:
                    model = build_isoster_model(image.shape, isos)
                    resid_stats = compute_residual_stats(image, model)
                    model_results[method] = {"model": model, "residual_stats": resid_stats}
                    print(
                        f"   {method}: 20-50 pix region: median={resid_stats.get('20-50', {}).get('median', 'N/A'):.2f}%"
                    )

            all_results[galaxy_name] = {
                "isophotes": {k: v for k, v in results.items()},
                "timing": timing,
                "comparison": comparison,
                "models": {k: v["residual_stats"] for k, v in model_results.items()},
                "center": (x0, y0),
            }
            benchmark_results[galaxy_name] = timing

        # Generate figures
        self._generate_figures(galaxy_data, all_results)

        # Save results to JSON
        self._save_results(all_results)

        # Assertions
        for galaxy_name in galaxy_data.keys():
            if galaxy_name in all_results:
                res = all_results[galaxy_name]
                # Check that all methods produced isophotes
                for method in ["photutils", "isoster_pa", "isoster_ea_34", "isoster_ea_310"]:
                    assert method in res["isophotes"], f"{galaxy_name}: {method} missing"
                    assert len(res["isophotes"][method]) > 10, f"{galaxy_name}: {method} too few isophotes"

        print("\n--- All Methods Comparison Test Passed ---")

    def _generate_figures(self, galaxy_data, all_results):
        """Generate QA figures for all galaxies and methods."""

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        for galaxy_name, gal in galaxy_data.items():
            if galaxy_name not in all_results:
                continue

            res = all_results[galaxy_name]
            image = gal["image"]

            # Figure 1: 1D Profile Comparison
            self._plot_profile_comparison(galaxy_name, res, OUTPUT_DIR)

            # Figure 2: 2D Residual Maps
            self._plot_residual_maps(galaxy_name, image, res, OUTPUT_DIR)

        # Figure 3: Benchmark Timing
        self._plot_timing_comparison(all_results, OUTPUT_DIR)

        print(f"\nFigures saved to {OUTPUT_DIR}")

    def _plot_profile_comparison(self, galaxy_name, res, output_dir):
        """Plot 1D profile comparison figure with both valid and invalid points."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(6, 1, figsize=(10, 14), sharex=True)
        fig.suptitle(f"{galaxy_name} - Method Comparison", fontsize=14)

        # Colors and markers for each method
        styles = {
            "photutils": {"color": "black", "marker": "s", "label": "photutils"},
            "isoster_pa": {"color": "blue", "marker": "o", "label": "isoster PA"},
            "isoster_ea_34": {"color": "green", "marker": "^", "label": "isoster EA [3,4]"},
            "isoster_ea_310": {"color": "red", "marker": "D", "label": "isoster EA [3..10]"},
        }

        for method, isos in res["isophotes"].items():
            if method not in styles:
                continue

            style = styles[method]
            if len(isos) < 3:
                continue

            # Extract ALL isophotes (both converged and failed)
            sma_all = np.array([iso["sma"] for iso in isos])
            stop_codes = np.array([iso["stop_code"] for iso in isos])
            mask_conv = stop_codes == 0
            n_conv = mask_conv.sum()
            n_total = len(isos)

            sma_quarter_all = sma_all**0.25

            # Panel 1: Intensity (log scale)
            intens_all = np.array([iso["intens"] for iso in isos])
            # Plot failed points with open markers (faded)
            if (~mask_conv).any():
                axes[0].scatter(
                    sma_quarter_all[~mask_conv],
                    intens_all[~mask_conv],
                    facecolors="none",
                    edgecolors=style["color"],
                    marker=style["marker"],
                    s=15,
                    alpha=0.3,
                    linewidths=0.5,
                )
            # Plot converged points with filled markers (solid)
            if mask_conv.any():
                axes[0].scatter(
                    sma_quarter_all[mask_conv],
                    intens_all[mask_conv],
                    c=style["color"],
                    marker=style["marker"],
                    s=15,
                    alpha=0.7,
                    label=f"{style['label']} ({n_conv}/{n_total})",
                )

            # Panel 3: Ellipticity
            eps_all = np.array([iso["eps"] for iso in isos])
            if (~mask_conv).any():
                axes[2].scatter(
                    sma_quarter_all[~mask_conv],
                    eps_all[~mask_conv],
                    facecolors="none",
                    edgecolors=style["color"],
                    marker=style["marker"],
                    s=15,
                    alpha=0.3,
                    linewidths=0.5,
                )
            if mask_conv.any():
                axes[2].scatter(
                    sma_quarter_all[mask_conv],
                    eps_all[mask_conv],
                    c=style["color"],
                    marker=style["marker"],
                    s=15,
                    alpha=0.7,
                )

            # Panel 4: Position Angle
            pa_all = np.array([iso["pa"] for iso in isos])
            pa_deg_all = np.degrees(pa_all) % 180
            if (~mask_conv).any():
                axes[3].scatter(
                    sma_quarter_all[~mask_conv],
                    pa_deg_all[~mask_conv],
                    facecolors="none",
                    edgecolors=style["color"],
                    marker=style["marker"],
                    s=15,
                    alpha=0.3,
                    linewidths=0.5,
                )
            if mask_conv.any():
                axes[3].scatter(
                    sma_quarter_all[mask_conv],
                    pa_deg_all[mask_conv],
                    c=style["color"],
                    marker=style["marker"],
                    s=15,
                    alpha=0.7,
                )

            # Panel 5: a4 (boxiness/diskiness)
            a4_all = np.array([iso.get("a4", 0.0) for iso in isos])
            if (~mask_conv).any():
                axes[4].scatter(
                    sma_quarter_all[~mask_conv],
                    a4_all[~mask_conv],
                    facecolors="none",
                    edgecolors=style["color"],
                    marker=style["marker"],
                    s=15,
                    alpha=0.3,
                    linewidths=0.5,
                )
            if mask_conv.any():
                axes[4].scatter(
                    sma_quarter_all[mask_conv],
                    a4_all[mask_conv],
                    c=style["color"],
                    marker=style["marker"],
                    s=15,
                    alpha=0.7,
                )

            # Panel 6: b4
            b4_all = np.array([iso.get("b4", 0.0) for iso in isos])
            if (~mask_conv).any():
                axes[5].scatter(
                    sma_quarter_all[~mask_conv],
                    b4_all[~mask_conv],
                    facecolors="none",
                    edgecolors=style["color"],
                    marker=style["marker"],
                    s=15,
                    alpha=0.3,
                    linewidths=0.5,
                )
            if mask_conv.any():
                axes[5].scatter(
                    sma_quarter_all[mask_conv],
                    b4_all[mask_conv],
                    c=style["color"],
                    marker=style["marker"],
                    s=15,
                    alpha=0.7,
                )

        # Panel 2: Fractional residual vs photutils (converged only for comparison)
        ref_isos = res["isophotes"].get("photutils", [])
        ref_conv = [iso for iso in ref_isos if iso["stop_code"] == 0]
        if len(ref_conv) > 3:
            sma_ref = np.array([iso["sma"] for iso in ref_conv])
            intens_ref = np.array([iso["intens"] for iso in ref_conv])

            for method in ["isoster_pa", "isoster_ea_34", "isoster_ea_310"]:
                if method not in res["isophotes"] or method not in styles:
                    continue

                isos = res["isophotes"][method]
                # Use ALL isophotes for residual comparison
                sma_all = np.array([iso["sma"] for iso in isos])
                intens_all = np.array([iso["intens"] for iso in isos])
                stop_codes = np.array([iso["stop_code"] for iso in isos])
                mask_conv = stop_codes == 0

                intens_ref_interp = np.interp(sma_all, sma_ref, intens_ref)
                frac_diff = (intens_all - intens_ref_interp) / intens_ref_interp * 100
                sma_quarter_all = sma_all**0.25

                style = styles[method]
                # Plot failed points with open markers
                if (~mask_conv).any():
                    axes[1].scatter(
                        sma_quarter_all[~mask_conv],
                        frac_diff[~mask_conv],
                        facecolors="none",
                        edgecolors=style["color"],
                        marker=style["marker"],
                        s=15,
                        alpha=0.3,
                        linewidths=0.5,
                    )
                # Plot converged points with filled markers
                if mask_conv.any():
                    axes[1].scatter(
                        sma_quarter_all[mask_conv],
                        frac_diff[mask_conv],
                        c=style["color"],
                        marker=style["marker"],
                        s=15,
                        alpha=0.7,
                    )

        # Formatting
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Intensity")
        axes[0].legend(loc="upper right", fontsize=8)

        axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.5)
        axes[1].axhline(1, color="gray", linestyle=":", linewidth=0.5)
        axes[1].axhline(-1, color="gray", linestyle=":", linewidth=0.5)
        axes[1].set_ylabel("Residual (%)")
        axes[1].set_ylim(-5, 5)

        axes[2].set_ylabel("Ellipticity")
        axes[2].set_ylim(0, 0.8)

        axes[3].set_ylabel("PA (deg)")

        axes[4].set_ylabel("a4")
        axes[4].axhline(0, color="gray", linestyle="--", linewidth=0.5)

        axes[5].set_ylabel("b4")
        axes[5].axhline(0, color="gray", linestyle="--", linewidth=0.5)
        axes[5].set_xlabel("SMA^0.25 (pixels)")

        plt.tight_layout()
        plt.savefig(output_dir / f"{galaxy_name}_profile_comparison.png", dpi=150)
        plt.close()

    def _plot_residual_maps(self, galaxy_name, image, res, output_dir):
        """Plot 2D residual maps figure."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f"{galaxy_name} - 2D Model and Residuals", fontsize=14)

        methods = ["photutils", "isoster_pa", "isoster_ea_34", "isoster_ea_310"]
        titles = ["photutils", "isoster PA", "isoster EA [3,4]", "isoster EA [3..10]"]

        # Log scale for image display
        vmin, vmax = np.nanpercentile(image[image > 0], [1, 99])

        for i, (method, title) in enumerate(zip(methods, titles)):
            isos = res["isophotes"].get(method, [])
            if len(isos) < 5:
                axes[0, i].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[0, i].transAxes)
                axes[1, i].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[1, i].transAxes)
                continue

            # Build model
            model = build_isoster_model(image.shape, isos)

            # Row 1: Model
            im0 = axes[0, i].imshow(
                np.log10(np.clip(model, vmin, None)),
                cmap="viridis",
                origin="lower",
                vmin=np.log10(vmin),
                vmax=np.log10(vmax),
            )
            axes[0, i].set_title(f"{title}\nModel")
            axes[0, i].axis("off")

            # Row 2: Fractional Residual
            with np.errstate(divide="ignore", invalid="ignore"):
                frac_resid = (image - model) / image * 100
                frac_resid = np.where(np.isfinite(frac_resid) & (image > vmin), frac_resid, np.nan)

            norm = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5)
            im1 = axes[1, i].imshow(frac_resid, cmap="RdBu_r", origin="lower", norm=norm)
            axes[1, i].set_title("Residual (%)")
            axes[1, i].axis("off")

        # Colorbars
        fig.colorbar(im0, ax=axes[0, :], shrink=0.6, label="log10(Intensity)")
        fig.colorbar(im1, ax=axes[1, :], shrink=0.6, label="(Data-Model)/Data (%)")

        plt.tight_layout()
        plt.savefig(output_dir / f"{galaxy_name}_residual_maps.png", dpi=150)
        plt.close()

    def _plot_timing_comparison(self, all_results, output_dir):
        """Plot timing benchmark comparison."""
        import matplotlib.pyplot as plt

        methods = ["photutils", "isoster_pa", "isoster_ea_34", "isoster_ea_310"]
        labels = ["photutils", "isoster PA", "isoster EA [3,4]", "isoster EA [3..10]"]
        colors = ["gray", "blue", "green", "red"]

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(methods))
        width = 0.35
        offset = 0

        for galaxy_name, res in all_results.items():
            timing = res.get("timing", {})
            means = [timing.get(m, {}).get("mean", 0) for m in methods]
            stds = [timing.get(m, {}).get("std", 0) for m in methods]

            bars = ax.bar(x + offset, means, width, yerr=stds, label=galaxy_name, alpha=0.8, capsize=3)
            offset += width

        ax.set_ylabel("Time (seconds)")
        ax.set_title("Fitting Time Comparison")
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "ea_harmonics_benchmark.png", dpi=150)
        plt.close()

    def _save_results(self, all_results):
        """Save results to JSON file."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        # Remove model arrays (too large for JSON)
        json_results = {}
        for galaxy_name, res in all_results.items():
            json_results[galaxy_name] = {
                "timing": res.get("timing", {}),
                "comparison": res.get("comparison", {}),
                "residual_stats": res.get("models", {}),
                "center": res.get("center", (0, 0)),
                "n_isophotes": {method: len(isos) for method, isos in res.get("isophotes", {}).items()},
            }

        output_file = OUTPUT_DIR / "ea_harmonics_results.json"
        with open(output_file, "w") as f:
            json.dump(convert(json_results), f, indent=2)

        print(f"\nResults saved to {output_file}")


# =============================================================================
# Standalone Runner
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "real_data", "-s"])
