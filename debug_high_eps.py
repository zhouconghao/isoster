"""Debug script to investigate high-ellipticity failed datapoints."""

import numpy as np
from isoster.driver import fit_image
from isoster.config import IsosterConfig

def create_sersic_model(shape, x0, y0, I_e, R_e, n, eps, pa):
    h, w = shape
    b_n = 1.9992 * n - 0.3271

    y = np.arange(h)
    x = np.arange(w)
    yy, xx = np.meshgrid(y, x, indexing='ij')

    dx = xx - x0
    dy = yy - y0
    x_rot = dx * np.cos(pa) + dy * np.sin(pa)
    y_rot = -dx * np.sin(pa) + dy * np.cos(pa)

    r = np.sqrt(x_rot**2 + (y_rot / (1 - eps))**2)
    image = I_e * np.exp(-b_n * ((r / R_e)**(1/n) - 1))

    return image

# Test with high ellipticity
print("Testing high ellipticity (eps=0.7) with EA mode...")
image = create_sersic_model(
    shape=(140, 140),
    x0=70.0, y0=70.0,  # Center of image
    I_e=1500.0, R_e=25.0, n=1.0,
    eps=0.7, pa=np.pi/3
)

rng = np.random.RandomState(123)
snr_at_re = 100
noise_level = 1500.0 / snr_at_re
image += rng.normal(0, noise_level, image.shape)

print("\n=== Test 1: Default maxgerr=0.5 ===")
cfg = IsosterConfig(
    x0=70.0, y0=70.0,
    sma0=8.0, minsma=3.0, maxsma=80.0,
    astep=0.15,
    eps=0.7, pa=np.pi/3,
    minit=10, maxit=50,
    conver=0.05,
    maxgerr=0.5,  # Default
    use_eccentric_anomaly=True,
)

results = fit_image(image, mask=None, config=cfg)
isophotes = results['isophotes']

# Analyze stop codes
stop_codes = [iso['stop_code'] for iso in isophotes]
sma_values = [iso['sma'] for iso in isophotes]

print(f"\nTotal isophotes: {len(isophotes)}")
print("\nStop Code Distribution:")
for code in sorted(set(stop_codes)):
    count = stop_codes.count(code)
    print(f"  Code {code:2d}: {count:3d} isophotes ({100*count/len(stop_codes):5.1f}%)")

print("\nFailed isophotes details (stop_code < 0):")
failed = [iso for iso in isophotes if iso['stop_code'] < 0]
if failed:
    print(f"  First failed: SMA={failed[0]['sma']:.2f}")
    print(f"  Last failed: SMA={failed[-1]['sma']:.2f}")
    for iso in failed[:5]:  # Show first 5
        print(f"    SMA={iso['sma']:6.2f}, stop_code={iso['stop_code']:2d}, "
              f"niter={iso['niter']:2d}, intens={iso['intens']:.1f}")

# Check range where fitting worked
converged = [iso for iso in isophotes if iso['stop_code'] == 0]
if converged:
    sma_converged = [iso['sma'] for iso in converged]
    print(f"\nConverged isophotes: {len(converged)}")
    print(f"  SMA range: {min(sma_converged):.1f} - {max(sma_converged):.1f}")

# Test with relaxed maxgerr
print("\n" + "="*60)
print("=== Test 2: Relaxed maxgerr=1.0 ===")
cfg2 = IsosterConfig(
    x0=70.0, y0=70.0,
    sma0=8.0, minsma=3.0, maxsma=80.0,
    astep=0.15,
    eps=0.7, pa=np.pi/3,
    minit=10, maxit=50,
    conver=0.05,
    maxgerr=1.0,  # Relaxed for high ellipticity
    use_eccentric_anomaly=True,
)

results2 = fit_image(image, mask=None, config=cfg2)
isophotes2 = results2['isophotes']

stop_codes2 = [iso['stop_code'] for iso in isophotes2]
print(f"\nTotal isophotes: {len(isophotes2)}")
print("\nStop Code Distribution:")
for code in sorted(set(stop_codes2)):
    count = stop_codes2.count(code)
    print(f"  Code {code:2d}: {count:3d} isophotes ({100*count/len(stop_codes2):5.1f}%)")

converged2 = [iso for iso in isophotes2 if iso['stop_code'] == 0]
if converged2:
    sma_converged2 = [iso['sma'] for iso in converged2]
    print(f"\nConverged isophotes: {len(converged2)}")
    print(f"  SMA range: {min(sma_converged2):.1f} - {max(sma_converged2):.1f}")

print("\n" + "="*60)
print("CONCLUSION:")
print(f"  Default (maxgerr=0.5): {len(converged)}/{len(isophotes)} converged ({100*len(converged)/len(isophotes):.1f}%)")
print(f"  Relaxed (maxgerr=1.0): {len(converged2)}/{len(isophotes2)} converged ({100*len(converged2)/len(isophotes2):.1f}%)")
print(f"\nRecommendation: For high ellipticity (eps>0.6), use maxgerr=1.0 or higher")
