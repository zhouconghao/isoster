"""
Profiling script to identify hot paths in isoster for Numba optimization.

This script uses cProfile to identify the most time-consuming functions.
"""

import cProfile
import pstats
from io import StringIO
import numpy as np
from isoster import fit_image
from isoster.config import IsosterConfig


def create_test_image(n=4.0, R_e=30.0, I_e=1000.0, eps=0.4, pa=np.pi/4):
    """Create a test Sersic image for profiling."""
    half_size = int(15 * R_e)
    shape = (2 * half_size, 2 * half_size)
    x0, y0 = half_size, half_size

    b_n = 1.9992 * n - 0.3271

    y, x = np.mgrid[:shape[0], :shape[1]].astype(np.float64)
    dx = x - x0
    dy = y - y0

    cos_pa, sin_pa = np.cos(pa), np.sin(pa)
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    r_ell = np.sqrt(x_rot**2 + (y_rot / (1 - eps))**2)
    r_ell = np.maximum(r_ell, 0.1)

    image = I_e * np.exp(-b_n * ((r_ell / R_e)**(1/n) - 1))

    return image, (x0, y0), (R_e, eps, pa)


def run_profile():
    """Run profiling on isoster fit_image."""
    print("Creating test image...")
    image, (x0, y0), (R_e, eps, pa) = create_test_image()
    print(f"Image shape: {image.shape}")

    config = IsosterConfig(
        x0=x0, y0=y0,
        eps=eps, pa=pa,
        sma0=10.0, minsma=3.0, maxsma=8 * R_e,
        astep=0.15,
        minit=10, maxit=50,
        conver=0.05,
        use_eccentric_anomaly=True,
    )

    print("\nProfiling fit_image (single run)...")

    # Profile
    profiler = cProfile.Profile()
    profiler.enable()

    results = fit_image(image, None, config)

    profiler.disable()

    # Print results
    print(f"\nResults: {len(results['isophotes'])} isophotes fitted")
    n_converged = sum(1 for iso in results['isophotes'] if iso['stop_code'] == 0)
    print(f"Converged: {n_converged}/{len(results['isophotes'])}")

    print("\n" + "="*80)
    print("PROFILING RESULTS (sorted by cumulative time)")
    print("="*80)

    s = StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())

    print("\n" + "="*80)
    print("PROFILING RESULTS (sorted by total time - internal only)")
    print("="*80)

    s = StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('tottime')
    ps.print_stats(30)
    print(s.getvalue())

    # Filter for isoster functions only
    print("\n" + "="*80)
    print("ISOSTER FUNCTIONS ONLY (by total time)")
    print("="*80)

    s = StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('tottime')
    ps.print_stats('isoster', 20)
    print(s.getvalue())

    return results, profiler


def run_multiple_profiles(n_runs=5):
    """Run multiple profiles to get stable timing."""
    print(f"Running {n_runs} profiles for stable timing...")

    image, (x0, y0), (R_e, eps, pa) = create_test_image()

    config = IsosterConfig(
        x0=x0, y0=y0,
        eps=eps, pa=pa,
        sma0=10.0, minsma=3.0, maxsma=8 * R_e,
        astep=0.15,
        minit=10, maxit=50,
        conver=0.05,
        use_eccentric_anomaly=True,
    )

    # Aggregate profiler
    profiler = cProfile.Profile()

    for i in range(n_runs):
        profiler.enable()
        results = fit_image(image, None, config)
        profiler.disable()
        print(f"  Run {i+1}/{n_runs}: {len(results['isophotes'])} isophotes")

    print("\n" + "="*80)
    print(f"AGGREGATE PROFILING RESULTS ({n_runs} runs)")
    print("="*80)

    s = StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('tottime')
    ps.print_stats('isoster', 30)
    print(s.getvalue())

    return profiler


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--multi':
        run_multiple_profiles(5)
    else:
        run_profile()
