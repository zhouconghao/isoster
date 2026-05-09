from pathlib import Path

import numpy as np
from astropy.io import fits

import isoster
from isoster.config import IsosterConfig


def debug_drift(galaxy_name, noise_label):
    fits_path = Path(f"outputs/benchmark_noise_floor/drift_mocks{'_noisy' if noise_label=='noisy' else ''}/{galaxy_name}_hsc_{noise_label}.fits")
    if not fits_path.exists():
        print(f"File not found: {fits_path}")
        return

    with fits.open(fits_path) as hdul:
        image = hdul[0].data.astype(np.float64)

    ny, nx = image.shape
    true_center = (nx/2.0, ny/2.0)

    # We'll run a fit and print info for outer isophotes
    config = IsosterConfig(
        x0=true_center[0], y0=true_center[1], eps=0.2, pa=0.0,
        sma0=10.0, minsma=1.0,
        compute_errors=True,
        debug=True # Enable debug to get grad and grad_error
    )

    results = isoster.fit_image(image, None, config)
    isos = results['isophotes']

    print(f"Study: {galaxy_name} ({noise_label})")
    print(f"{'SMA':>8} {'Intens':>10} {'Grad':>10} {'GradErr':>10} {'dR':>8} {'Stop':>4}")

    for iso in isos:
        if iso['sma'] > 50: # focus on outskirts
            dx = iso['x0'] - true_center[0]
            dy = iso['y0'] - true_center[1]
            dr = np.sqrt(dx**2 + dy**2)
            print(f"{iso['sma']:8.1f} {iso['intens']:10.4f} {iso['grad']:10.4f} {iso['grad_error']:10.4f} {dr:8.2f} {iso['stop_code']:4d}")

if __name__ == "__main__":
    debug_drift("NGC1453", "wide")
