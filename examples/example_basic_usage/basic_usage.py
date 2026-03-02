"""
Basic usage example for ISOSTER.

This script demonstrates the fundamental workflow for isophote fitting:
1. Load or create an image
2. Configure fitting parameters
3. Run the fit
4. Examine results
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaincinv

from isoster import fit_image
from isoster.config import IsosterConfig
from isoster.model import build_isoster_model
from isoster.output_paths import resolve_output_directory


def create_simple_galaxy():
    """Create a simple synthetic galaxy image for demonstration."""
    # Image parameters
    size = 201
    center = size // 2
    R_e = 30.0  # Effective radius
    I_e = 1000.0  # Intensity at R_e
    eps = 0.3  # Ellipticity
    pa = np.pi / 4  # Position angle (45 degrees)
    n = 4.0  # Sersic index (de Vaucouleurs)

    # Create coordinate grid
    y, x = np.mgrid[:size, :size].astype(float)
    dx = x - center
    dy = y - center

    # Rotate coordinates
    x_rot = dx * np.cos(pa) + dy * np.sin(pa)
    y_rot = -dx * np.sin(pa) + dy * np.cos(pa)

    # Elliptical radius
    r = np.sqrt(x_rot**2 + (y_rot / (1 - eps))**2)

    # Sersic profile (accurate b_n via inverse incomplete gamma function)
    b_n = gammaincinv(2 * n, 0.5)
    image = I_e * np.exp(-b_n * ((r / R_e)**(1/n) - 1))

    return image, center, center, eps, pa


def main():
    # Create a synthetic galaxy image
    print("Creating synthetic galaxy image...")
    image, x0, y0, true_eps, true_pa = create_simple_galaxy()

    # Configure isophote fitting
    config = IsosterConfig(
        x0=x0,
        y0=y0,
        sma0=10.0,      # Starting semi-major axis
        minsma=3.0,     # Minimum SMA
        maxsma=80.0,    # Maximum SMA
        astep=0.1,      # Logarithmic step size
        eps=true_eps,   # Initial ellipticity guess
        pa=true_pa,     # Initial position angle guess
        minit=10,       # Minimum iterations
        maxit=50,       # Maximum iterations
        conver=0.05,    # Convergence criterion
    )

    # Run isophote fitting
    print("Fitting isophotes...")
    results = fit_image(image, mask=None, config=config)
    isophotes = results['isophotes']

    print(f"Extracted {len(isophotes)} isophotes")

    # Build 2D model from isophotes
    print("Building 2D model...")
    model = build_isoster_model(image.shape, isophotes)

    # Extract profile data
    sma = np.array([iso['sma'] for iso in isophotes])
    intens = np.array([iso['intens'] for iso in isophotes])
    eps_fit = np.array([iso['eps'] for iso in isophotes])
    pa_fit = np.array([iso['pa'] for iso in isophotes])
    stop_codes = np.array([iso['stop_code'] for iso in isophotes])

    # Summary statistics
    converged = (stop_codes == 0).sum()
    print(f"Converged isophotes: {converged}/{len(isophotes)}")

    # Simple visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    vmin, vmax = np.percentile(image[image > 0], [5, 99])
    axes[0].imshow(np.log10(np.clip(image, vmin, None)),
                   cmap='viridis', origin='lower')
    axes[0].set_title('Original Image')
    axes[0].set_xlabel('X (pixels)')
    axes[0].set_ylabel('Y (pixels)')

    # Model
    axes[1].imshow(np.log10(np.clip(model, vmin, None)),
                   cmap='viridis', origin='lower')
    axes[1].set_title('Isophote Model')
    axes[1].set_xlabel('X (pixels)')

    # Surface brightness profile
    good = stop_codes == 0
    axes[2].semilogy(sma[good], intens[good], 'o-', markersize=4)
    axes[2].set_xlabel('Semi-major axis (pixels)')
    axes[2].set_ylabel('Intensity')
    axes[2].set_title('Surface Brightness Profile')
    axes[2].grid(True, alpha=0.3)

    output_dir = resolve_output_directory("examples_basic_usage")
    output_path = output_dir / "basic_usage_example.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved figure to {output_path}")
    plt.close()

    print("\nDone!")


if __name__ == '__main__':
    main()
