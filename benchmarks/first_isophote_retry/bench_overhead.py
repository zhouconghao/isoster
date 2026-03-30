"""
Benchmark: measure overhead of first-isophote retry feature on normal runs.

Compares fit_image() timing with default config (retry=0) and retry-enabled
config (retry=5) on real galaxy images. The first isophote should succeed on
the first try for these images, so retry-enabled should show zero overhead.

Usage:
    uv run python benchmarks/first_isophote_retry/bench_overhead.py
"""

import time
from pathlib import Path

import numpy as np
from astropy.io import fits

from isoster.config import IsosterConfig
from isoster.driver import fit_image

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
N_REPEATS = 20


def load_image(name):
    """Load a FITS image by name from data/."""
    path = DATA_DIR / f"{name}.fits"
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float64)
    # If 3D (multiband), use first band
    if data.ndim == 3:
        data = data[0]
    return data


def benchmark_config(image, config, label, n_repeats=N_REPEATS):
    """Run fit_image n_repeats times and return timing statistics."""
    times = []
    for i in range(n_repeats):
        start = time.perf_counter()
        result = fit_image(image, mask=None, config=config)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        if i == 0:
            n_iso = len(result["isophotes"])
            has_retry_log = "first_isophote_retry_log" in result
            has_failure = result.get("first_isophote_failure", False)

    times = np.array(times)
    print(f"  {label}:")
    print(f"    n_isophotes={n_iso}, retry_log={has_retry_log}, failure={has_failure}")
    print(f"    mean={times.mean():.4f}s  std={times.std():.4f}s  "
          f"min={times.min():.4f}s  max={times.max():.4f}s")
    return times.mean(), times.std()


def main():
    images = {
        "IC3370_mock2": {"sma0": 10.0, "maxsma": 100.0},
        "ngc3610": {"sma0": 5.0, "maxsma": 80.0},
    }

    print(f"First Isophote Retry — Performance Benchmark (n={N_REPEATS})")
    print("=" * 60)

    for name, params in images.items():
        try:
            image = load_image(name)
        except FileNotFoundError as e:
            print(f"\nSkipping {name}: {e}")
            continue

        print(f"\n{name} ({image.shape[0]}x{image.shape[1]}):")

        h, w = image.shape
        config_default = IsosterConfig(
            x0=w / 2.0, y0=h / 2.0,
            sma0=params["sma0"], maxsma=params["maxsma"],
        )
        config_retry = IsosterConfig(
            x0=w / 2.0, y0=h / 2.0,
            sma0=params["sma0"], maxsma=params["maxsma"],
            max_retry_first_isophote=5,
        )

        mean_default, std_default = benchmark_config(image, config_default, "retry=0 (default)")
        mean_retry, std_retry = benchmark_config(image, config_retry, "retry=5 (enabled)")

        overhead_pct = (mean_retry - mean_default) / mean_default * 100
        print(f"  overhead: {overhead_pct:+.2f}%")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
