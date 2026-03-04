import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.utils.autoprof_adapter import parse_autoprof_profile

prof_path = Path("outputs/benchmark_noise_floor/drift_investigation_v2/autoprof/NGC5061_hsc_wide/NGC5061_hsc_wide.prof")
profile = parse_autoprof_profile(prof_path, 0.168, zeropoint=27.0)
print(f"Profile: {profile is not None}")
if profile:
    print(f"Keys: {profile.keys()}")
    print(f"SMA: {profile['sma'][:5]}")
