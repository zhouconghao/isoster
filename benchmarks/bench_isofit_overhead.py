"""ISOFIT overhead benchmark.

Measures wall-clock overhead of simultaneous_harmonics=True vs False
on a synthetic Sersic mock to verify:
1. Default path (simultaneous_harmonics=False) has zero overhead.
2. ISOFIT overhead is documented and acceptable.

Usage:
    uv run python benchmarks/bench_isofit_overhead.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isoster import fit_image
from isoster.config import IsosterConfig
from tests.fixtures import create_sersic_model


def run_benchmark(n_repeats: int = 3):
    """Run ISOFIT overhead benchmark."""
    # Create a moderate-sized Sersic mock
    r_eff = 25.0
    image, _, params = create_sersic_model(
        R_e=r_eff, n=2.0, I_e=1000.0, eps=0.3, pa=0.5,
        oversample=3, size_factor=10.0, min_half_size=100,
    )
    cx, cy = params['x0'], params['y0']

    base_config = dict(
        x0=cx, y0=cy, sma0=6.0, minsma=3.0, maxsma=100.0,
        astep=0.15, eps=0.3, pa=0.5,
        minit=10, maxit=50, conver=0.05,
        fix_center=True,
    )

    configs = {
        'default (no harmonics)': IsosterConfig(
            **base_config, simultaneous_harmonics=False, harmonic_orders=[3, 4],
        ),
        'ISOFIT [3,4]': IsosterConfig(
            **base_config, simultaneous_harmonics=True, harmonic_orders=[3, 4],
        ),
        'ISOFIT [3,4,5,6]': IsosterConfig(
            **base_config, simultaneous_harmonics=True, harmonic_orders=[3, 4, 5, 6],
        ),
        'ISOFIT [3..7]': IsosterConfig(
            **base_config, simultaneous_harmonics=True, harmonic_orders=[3, 4, 5, 6, 7],
        ),
    }

    print(f"Image shape: {image.shape}")
    print(f"Sersic n=2.0, R_e={r_eff}, eps=0.3")
    print(f"Repeats: {n_repeats}")
    print(f"{'='*60}")

    results = {}
    for name, cfg in configs.items():
        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            res = fit_image(image, mask=None, config=cfg)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        n_iso = len(res['isophotes'])
        sc = [iso['stop_code'] for iso in res['isophotes']]
        converged = sum(1 for s in sc if s == 0)

        median_time = np.median(times)
        results[name] = median_time

        print(f"\n{name}:")
        print(f"  Median time: {median_time:.3f}s")
        print(f"  Isophotes: {n_iso}, converged: {converged}")

    # Summary
    baseline = results['default (no harmonics)']
    print(f"\n{'='*60}")
    print("Overhead relative to default:")
    for name, t in results.items():
        overhead = (t / baseline - 1.0) * 100
        print(f"  {name:30s}: {overhead:+.1f}%")


if __name__ == '__main__':
    run_benchmark()
