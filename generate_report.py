# generate_report.py - Lee directamente de data/results/ablation_summary.json y hpo_results.json (sin hardcodear SDs)
import json
from pathlib import Path
import numpy as np

def load_summary(root=Path(".")):
    raw = json.loads((root / "data/results/ablation_summary.json").read_text(encoding="utf-8"))
    means = {k: float(np.mean(v["lg_per_seed"])) for k, v in raw.items()}
    stds = {k: float(np.std(v["lg_per_seed"], ddof=1)) for k, v in raw.items()}
    return means, stds

if __name__ == "__main__":
    m, s = load_summary()
    for k in m:
        print(f"{k}: media={m[k]:.4f} +/- sd={s[k]:.5f}")
