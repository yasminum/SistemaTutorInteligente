"""
run_ab.py - CLI reproducible (Slide 13: "si no se puede reproducir, no cuenta").

  python -m src.run_ab --variant PPOlin_FS0_raw13          # Baseline
  python -m src.run_ab --variant PPOlin_FS1_zpd4           # Var1
  python -m src.run_ab --variant PPOlin_FS2_hist6          # Var2
  python -m src.run_ab --all --robustness --workers 4      # todo + prueba de estres theta_hat (Elo)

Salidas: logs/metrics_experimentos.csv, data/results/runs_<modo>.json, data/results/references.json,
         data/item_bank.csv + data/SNAPSHOT.json (hash/tamano/fecha/versiones).
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

from .experiment import (CFG, VARIANTS, Runner, environment_fingerprint, evaluate_reference, file_sha256,
                         log_row, make_log_row)
from .simulator import ItemBank

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _job(args):
    variant, fold, seed, theta_mode = args
    return Runner(variant, fold, seed, CFG, theta_mode=theta_mode).train()


def write_snapshot():
    bank = ItemBank(CFG["bank_seed"])
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    csv_path = os.path.join(ROOT, "data", "item_bank.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(bank.to_csv())
    snap = {"file": "data/item_bank.csv", "sha256": bank.sha256(), "bytes": os.path.getsize(csv_path),
            "rows": 300, "generator": f"ItemBank(seed={CFG['bank_seed']})",
            "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "env": environment_fingerprint()}
    with open(os.path.join(ROOT, "data", "SNAPSHOT.json"), "w", encoding="utf-8") as fh:
        json.dump(snap, fh, indent=2)
    return snap


CORE_MODULES = ("simulator.py", "feature_engineering.py", "agent_ppo_linear.py", "experiment.py")


def code_hash():
    """SHA-256 de los modulos que determinan los resultados (no incluye reportes ni pruebas)."""
    return file_sha256([os.path.join(ROOT, "src", m) for m in CORE_MODULES])


def run(jobs, workers, out_name):
    snap = write_snapshot()
    chash = code_hash()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_job, jobs))
    else:
        results = [_job(j) for j in jobs]
    log_path = os.path.join(ROOT, "logs", "metrics_experimentos.csv")
    for res in results:
        log_row(log_path, make_log_row(res, CFG, chash, snap["sha256"]))
        print(f"{res['variant']:>18} {res['theta_mode']:>6} f{res['fold']} s{res['seed']}  "
              f"LG={res['final']['lg']:+.4f} acc={res['final']['acc']:.3f}")
    os.makedirs(os.path.join(ROOT, "data", "results"), exist_ok=True)
    path = os.path.join(ROOT, "data", "results", out_name)
    existing = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            existing = json.load(fh)["runs"]
    keys = {(r["variant"], r["fold"], r["seed"], r["theta_mode"]) for r in results}
    merged = [r for r in existing if (r["variant"], r["fold"], r["seed"], r["theta_mode"]) not in keys] + results
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"cfg": CFG, "code_sha256": chash, "data_sha256": snap["sha256"], "runs": merged}, fh)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--robustness", action="store_true", help="theta_hat Elo en baseline y Var1 (seed 0)")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    variants = list(VARIANTS) if a.all else ([a.variant] if a.variant else [])
    if variants:
        jobs = [(v, f, s, "oracle") for v in variants for f in range(CFG["k_folds"]) for s in CFG["seeds"]]
        run(jobs, a.workers, "runs_oracle.json")
        refs = {p: [evaluate_reference(p, f) for f in range(CFG["k_folds"])] for p in ("random", "fixed")}
        with open(os.path.join(ROOT, "data", "results", "references.json"), "w", encoding="utf-8") as fh:
            json.dump(refs, fh, indent=2)
    if a.robustness:
        jobs = [(v, f, 0, "elo") for v in ("PPOlin_FS0_raw13", "PPOlin_FS1_zpd4") for f in range(CFG["k_folds"])]
        run(jobs, a.workers, "runs_elo.json")


if __name__ == "__main__":
    main()
