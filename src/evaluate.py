from __future__ import annotations
import json
import numpy as np
from pathlib import Path

def save_metrics(metrics, path="models/metrics.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    def convert(v):
        if isinstance(v, (np.floating, float)):
            return float(v)
        if isinstance(v, (np.integer, int)):
            return int(v)
        if isinstance(v, dict):
            return {k: convert(v) for k, v in v.items()}
        if isinstance(v, list):
            return [convert(x) for x in v]
        return v
    with open(path, "w") as f:
        json.dump(convert(metrics), f, indent=2)

def print_report(metrics):
    print(f"{'='*50}")
    print(f"  Segmentation Results")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k:20s}: {v:.4f}")
        elif isinstance(v, list):
            print(f"    {k:20s}: {[round(x,4) for x in v]}")
        else:
            print(f"    {k:20s}: {v}")
