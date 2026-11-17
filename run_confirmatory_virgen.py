"""
run_confirmatory_virgen.py
Ejecuta las 30 corridas pareadas confirmatorias (fold_seed = 2027, 2028, 2029 x 5 folds x 2 semillas)
sobre particiones virgenes usando exactamente las variantes congeladas y CFG de src/experiment.py.
"""
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.experiment import CFG, Runner
from src.simulator import ItemBank

VARIANTS_CONGELADAS = ["PPOlin_FS0_raw13", "PPOlin_FS1_zpd4", "PPOlin_D2_clean5_zpd2"]
VIRGIN_FOLD_SEEDS = [2027, 2028, 2029]


def _run_one(task):
    variant, fold_seed, fold, seed = task
    cfg = dict(CFG)
    cfg["fold_seed"] = fold_seed
    bank = ItemBank(cfg["bank_seed"])
    runner = Runner(variant, fold, seed, cfg=cfg, theta_mode="oracle", bank=bank)
    res = runner.train(verbose=False)
    res["fold_seed"] = fold_seed
    # Strip heavy per-episode arrays to keep runs_virgen.json compact and fast
    res.pop("train_lg", None)
    res.pop("train_ret", None)
    return res


def main():
    t0 = time.perf_counter()
    tasks = [
        (v, fs, f, s)
        for fs in VIRGIN_FOLD_SEEDS
        for f in range(CFG["k_folds"])
        for s in CFG["seeds"]
        for v in VARIANTS_CONGELADAS
    ]
    print(f"Lanzando {len(tasks)} corridas ({len(VIRGIN_FOLD_SEEDS)*CFG['k_folds']*len(CFG['seeds'])} pareadas x {len(VARIANTS_CONGELADAS)} variantes)...")
    results = []
    workers = min(12, os.cpu_count() or 4)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for idx, res in enumerate(ex.map(_run_one, tasks), 1):
            results.append(res)
            if idx % 15 == 0 or idx == len(tasks):
                elapsed = time.perf_counter() - t0
                print(f"  Progreso: {idx}/{len(tasks)} ({elapsed:.1f}s)", flush=True)

    out_path = ROOT / "data" / "results" / "runs_virgen.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"OK guardado: {out_path} ({len(results)} registros, {time.perf_counter()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
