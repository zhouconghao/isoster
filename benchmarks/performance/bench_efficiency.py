"""
Efficiency benchmark for isoster optimizations.

This script measures the performance of isoster before and after efficiency
optimizations, ensuring no degradation in 1-D profile quality.
"""

import time
import numpy as np
import json
from pathlib import Path
from isoster import fit_image
from isoster.config import IsosterConfig


def create_sersic_model(R_e, n, I_e, eps, pa, noise_level=None, oversample=1):
    """Create a centered 2D Sersic profile with optional noise.

    Per CLAUDE.md: Image half-size should be >= 10 * R_e (15x for better coverage)
    """
    half_size = max(int(15 * R_e), 150)
    shape = (2 * half_size, 2 * half_size)
    x0, y0 = half_size, half_size

    b_n = 1.9992 * n - 0.3271

    if oversample > 1:
        oversamp_shape = (shape[0] * oversample, shape[1] * oversample)
        y = np.arange(oversamp_shape[0]) / oversample
        x = np.arange(oversamp_shape[1]) / oversample
        yy, xx = np.meshgrid(y, x, indexing='ij')

        dx = xx - x0
        dy = yy - y0
        x_rot = dx * np.cos(pa) + dy * np.sin(pa)
        y_rot = -dx * np.sin(pa) + dy * np.cos(pa)
        r_ell = np.sqrt(x_rot**2 + (y_rot / (1 - eps))**2)

        image_oversamp = I_e * np.exp(-b_n * ((r_ell / R_e)**(1/n) - 1))

        image = np.zeros(shape)
        for i in range(oversample):
            for j in range(oversample):
                image += image_oversamp[i::oversample, j::oversample]
        image /= oversample**2
    else:
        y = np.arange(shape[0])
        x = np.arange(shape[1])
        yy, xx = np.meshgrid(y, x, indexing='ij')

        dx = xx - x0
        dy = yy - y0
        x_rot = dx * np.cos(pa) + dy * np.sin(pa)
        y_rot = -dx * np.sin(pa) + dy * np.cos(pa)
        r_ell = np.sqrt(x_rot**2 + (y_rot / (1 - eps))**2)

        image = I_e * np.exp(-b_n * ((r_ell / R_e)**(1/n) - 1))

    # Add noise if requested
    if noise_level is not None and noise_level > 0:
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        image += rng.normal(0, noise_level, image.shape)

    return image, (x0, y0), shape


def benchmark_case(name, image, config, n_runs=3):
    """Benchmark a single test case with multiple runs."""
    times = []

    for i in range(n_runs):
        start = time.perf_counter()
        results = fit_image(image, None, config)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    # Get results from last run
    isophotes = results['isophotes']
    n_isophotes = len(isophotes)
    n_converged = len([iso for iso in isophotes if iso['stop_code'] == 0])

    return {
        'name': name,
        'times': times,
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'n_isophotes': n_isophotes,
        'n_converged': n_converged,
        'convergence_rate': n_converged / n_isophotes if n_isophotes > 0 else 0.0
    }


def run_benchmark_suite():
    """Run comprehensive benchmark suite."""
    print("="*70)
    print("ISOSTER EFFICIENCY BENCHMARK")
    print("="*70)
    print()

    results = []

    # Test cases: (name, n, R_e, eps, pa, snr, oversample)
    test_cases = [
        # Fast cases (small, circular)
        ("n1_small_circular", 1.0, 10.0, 0.0, 0.0, None, 5),
        ("n4_small_circular", 4.0, 10.0, 0.0, 0.0, None, 10),

        # Medium cases
        ("n1_medium_circular", 1.0, 20.0, 0.0, 0.0, None, 5),
        ("n1_medium_eps04", 1.0, 20.0, 0.4, np.pi/4, None, 5),
        ("n4_medium_eps04", 4.0, 20.0, 0.4, np.pi/4, None, 10),

        # High ellipticity cases
        ("n1_medium_eps07", 1.0, 20.0, 0.7, np.pi/3, None, 5),
        ("n4_medium_eps06", 4.0, 20.0, 0.6, np.pi/4, None, 15),

        # Noisy cases
        ("n1_medium_snr100", 1.0, 20.0, 0.4, np.pi/4, 100, 5),
        ("n4_medium_snr100", 4.0, 20.0, 0.4, np.pi/4, 100, 10),
    ]

    for test_name, n, R_e, eps, pa, snr, oversample in test_cases:
        print(f"Running: {test_name}")
        print(f"  Parameters: n={n}, Re={R_e}, eps={eps:.1f}, PA={pa:.2f}, SNR={snr}, oversample={oversample}")

        # Create image
        I_e = 1000.0
        noise_level = I_e / snr if snr is not None else None
        image, (x0, y0), shape = create_sersic_model(R_e, n, I_e, eps, pa, noise_level, oversample)

        print(f"  Image shape: {shape}")

        # Configure isoster
        config = IsosterConfig(
            x0=x0, y0=y0,
            eps=eps, pa=pa,
            sma0=10.0, minsma=3.0, maxsma=8 * R_e,
            astep=0.15,
            minit=10, maxit=50,
            conver=0.05,
            maxgerr=1.0 if eps > 0.6 else 0.5,  # Relaxed for high ellipticity
            use_eccentric_anomaly=(eps > 0.3),
        )

        # Benchmark
        result = benchmark_case(test_name, image, config, n_runs=3)
        results.append(result)

        print(f"  Time: {result['mean_time']:.3f} ± {result['std_time']:.3f} s")
        print(f"  Isophotes: {result['n_converged']}/{result['n_isophotes']} converged ({result['convergence_rate']*100:.1f}%)")
        print()

    # Summary
    print("="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print()
    print(f"{'Test Case':<30} {'Mean Time (s)':<15} {'Isophotes':<12} {'Conv Rate':<10}")
    print("-"*70)

    total_time = 0
    for result in results:
        print(f"{result['name']:<30} {result['mean_time']:>8.3f} ± {result['std_time']:>5.3f}  "
              f"{result['n_converged']:>4}/{result['n_isophotes']:<4}  "
              f"{result['convergence_rate']*100:>6.1f}%")
        total_time += result['mean_time']

    print("-"*70)
    print(f"{'TOTAL':<30} {total_time:>8.3f} s")
    print()

    return results


def save_results(results, filename="benchmarks/efficiency_baseline.json"):
    """Save benchmark results to JSON file."""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    results_json = []
    for result in results:
        result_copy = result.copy()
        result_copy['times'] = [float(t) for t in result_copy['times']]
        result_copy['mean_time'] = float(result_copy['mean_time'])
        result_copy['std_time'] = float(result_copy['std_time'])
        result_copy['min_time'] = float(result_copy['min_time'])
        result_copy['convergence_rate'] = float(result_copy['convergence_rate'])
        results_json.append(result_copy)

    with open(filename, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"Results saved to: {filename}")


if __name__ == '__main__':
    results = run_benchmark_suite()
    save_results(results)
