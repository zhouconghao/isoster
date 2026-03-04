import json
from pathlib import Path
import numpy as np

def analyze_drifts():
    output_dir = Path("outputs/benchmark_noise_floor/drift_investigation")
    # I didn't save json results in investigate_drift.py, I should rerun or modify it.
    # Actually, I can just modify investigate_drift.py to save a summary.json.
    # But for now, I'll just check if the plots exist.
    plots = list(output_dir.glob("*.png"))
    print(f"Generated {len(plots)} plots.")
    for p in plots:
        print(f" - {p.name}")

if __name__ == "__main__":
    analyze_drifts()
