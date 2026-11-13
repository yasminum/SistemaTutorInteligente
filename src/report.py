"""
report.py - Tablas, estadistica, figuras y resumen a partir de data/results (usado por el notebook y el README).
"""
import json
import os
import time

import numpy as np
import pandas as pd

from .experiment import CFG, NOTES, VARIANTS, Runner, _episode_seed, episodes_to_target, make_folds, paired_stats
from .feature_engineering import FeatureBuilder
from .simulator import HORIZON, ItemBank, TutorEnv


def ceiling_references(fold=0, reps=3):
    """Escala de lectura del LG: heuristica ZPD con oraculo y cota 'acierto en ZPD en todos los pasos'."""
    bank = ItemBank(CFG["bank_seed"])
    fb = FeatureBuilder("FS1", bank)
    env = TutorEnv(bank)
    _, test_ids = make_folds(CFG["n_students"], CFG["k_folds"], CFG["fold_seed"])[fold]
    heur, bound = [], []
    for sid in test_ids:
        for r in range(reps):
            seed = _episode_seed(int(sid), 2, r)
            env.reset(int(sid), seed)
            done = False
            while not done:
                phi = fb.action_features(env.student.theta)
                _, done, _ = env.step(int(np.argmax(phi[:, 3] + 1e-3 * phi[:, 0])))
            heur.append(env.learning_gain())
            rng = np.random.default_rng(seed)
            th0 = np.random.default_rng(10_000 + int(sid)).beta(2, 2, 3)
            th = th0.copy()
            for _ in range(HORIZON):
                k = int(np.argmax(1 - th))
                th[k] += rng.uniform(0.02, 0.05) * (1 - th[k])
                th = np.clip(th * (1 - rng.uniform(0.001, 0.005)), 0, 1)
            bound.append((th.mean() - th0.mean()) / (1 - th0.mean()))
    return pd.DataFrame([
        {"Referencia de escala (fold 0, test)": "Heurística ZPD con oráculo θ* (argmax zpd_hit)", "LG_sim": float(np.mean(heur))},
        {"Referencia de escala (fold 0, test)": "Cota: acierto dentro de ZPD en los 50 pasos", "LG_sim": float(np.mean(bound))},
    ])

BASE = "PPOlin_FS0_raw13"
METRICS = [("lg", "LG_sim holdout"), ("acc", "Acierto"), ("learn", "Eventos de aprendizaje/paso"),
           ("shield", "Violaciones (shield)/paso"), ("cost", "Costo CMDP/paso"), ("ret", "Retorno")]


def load_runs(root, mode="oracle"):
    with open(os.path.join(root, "data", "results", f"runs_{mode}.json"), encoding="utf-8") as fh:
        blob = json.load(fh)
    return blob, blob["runs"]


def _by_variant(runs):
    out = {}
    for r in runs:
        out.setdefault(r["variant"], {})[(r["fold"], r["seed"])] = r
    return out


def _smooth(y, w=3):
    y = np.asarray(y, float)
    return np.array([y[max(0, i - w + 1):i + 1].mean() for i in range(len(y))])


def convergence(runs):
    """Episodios hasta que la curva holdout (media movil 3) alcanza el LG final del baseline del MISMO par."""
    bv = _by_variant(runs)
    rows = []
    for v, d in bv.items():
        for key, r in d.items():
            target = bv[BASE][key]["final"]["lg"]
            curve = [{"episode": c["episode"], "lg": s} for c, s in zip(r["curve"], _smooth([c["lg"] for c in r["curve"]]))]
            auc = float(np.mean([c["lg"] for c in r["curve"]]))
            rows.append({"variant": v, "fold": key[0], "seed": key[1], "ep_to_target": episodes_to_target(curve, target),
                         "curve_mean_lg": auc})
    return pd.DataFrame(rows)


def results_table(runs, log_path="logs/metrics_experimentos.csv"):
    bv = _by_variant(runs)
    conv = convergence(runs)
    rows = []
    for v in VARIANTS:
        d = list(bv[v].values())
        row = {"ID experimento": f"FE5-{v}-oracle-f*-s*", "Variante": v}
        for m, label in METRICS:
            vals = np.array([r["final"][m] for r in d])
            row[label] = f"{vals.mean():+.4f} ± {vals.std(ddof=1):.4f}" if m == "lg" else f"{vals.mean():.3f}"
        c = conv[conv.variant == v]
        row["Episodios a LG_base (mediana)"] = (f"{np.nanmedian(c.ep_to_target):.0f}"
                                                 if c.ep_to_target.notna().any() else "no alcanza")
        row["Pares que alcanzan"] = f"{int(c.ep_to_target.notna().sum())}/{len(c)}"
        row["Ruta log"] = log_path
        row["Nota (qué cambió)"] = NOTES[v]
        rows.append(row)
    return pd.DataFrame(rows)


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def stats_table(runs, metric="lg"):
    bv = _by_variant(runs)
    keys = sorted(bv[BASE])
    rows = []
    for v in VARIANTS:
        if v == BASE:
            continue
        a = [bv[BASE][k]["final"][metric] for k in keys]
        b = [bv[v][k]["final"][metric] for k in keys]
        s = paired_stats(a, b)
        rows.append({"Comparación": f"{v} vs {BASE}", **s})
    df = pd.DataFrame(rows)
    df["p_Holm"] = holm(df["p_perm"].values)
    return df


def per_pair_table(runs):
    bv = _by_variant(runs)
    rows = []
    for k in sorted(bv[BASE]):
        row = {"fold": k[0], "seed": k[1]}
        for v in VARIANTS:
            row[v] = bv[v][k]["final"]["lg"]
        rows.append(row)
    return pd.DataFrame(rows)


def latency_benchmark(n_decisions=3000, fold=0):
    """Micro-benchmark SECUENCIAL (misma maquina, mismas condiciones): features + escalado + politica."""
    rows = []
    for v in VARIANTS:
        r = Runner(v, fold, 0, CFG)
        r.fit_scaler()
        times = []
        sid_cycle = list(r.test_ids)
        i = 0
        while len(times) < n_decisions:
            r._reset(int(sid_cycle[i % len(sid_cycle)]), 555 + i)
            i += 1
            done = False
            while not done and len(times) < n_decisions:
                t0 = time.perf_counter()
                xs, phi = r._observe()
                a, _, _ = r.agent.act(r.scaler.tx_state(xs), r.scaler.tx_action(phi), greedy=True)
                times.append(time.perf_counter() - t0)
                _, done, info = r.env.step(a)
                r._post_step(info)
        t = np.array(times) * 1e3
        rows.append({"Variante": v, "n_rasgos_estado": r.fb.d_state, "n_rasgos_accion": r.fb.d_action,
                     "n_parametros_politica": r.agent.W.size + (r.agent.wa.size if r.fb.d_action else 0),
                     "latencia_mediana_ms": float(np.median(t)), "latencia_p95_ms": float(np.percentile(t, 95))})
    df = pd.DataFrame(rows)
    df["overhead_vs_base_ms"] = df["latencia_mediana_ms"] - df.loc[df.Variante == BASE, "latencia_mediana_ms"].values[0]
    return df


def robustness_table(runs_oracle, runs_elo):
    bo, be = _by_variant(runs_oracle), _by_variant(runs_elo)
    rows = []
    for v in ("PPOlin_FS0_raw13", "PPOlin_FS1_zpd4", "PPOlin_D2_clean5_zpd2"):
        keys = sorted(be[v])
        o = np.array([bo[v][k]["final"]["lg"] for k in keys])
        e = np.array([be[v][k]["final"]["lg"] for k in keys])
        rows.append({"Variante": v, "LG oráculo θ* (seed 0)": o.mean(), "LG observable θ̂ Elo (seed 0)": e.mean(),
                     "Δ (θ̂ − θ*)": e.mean() - o.mean()})
    keys = sorted(be[BASE])
    s = paired_stats([be[BASE][k]["final"]["lg"] for k in keys], [be["PPOlin_FS1_zpd4"][k]["final"]["lg"] for k in keys])
    return pd.DataFrame(rows), s


def curves_figure(runs, path):
    import matplotlib.pyplot as plt
    bv = _by_variant(runs)
    colors = {"PPOlin_FS0_raw13": "#7f8c8d", "PPOlin_FS1_zpd4": "#2471a3", "PPOlin_D1_zpd2": "#5b2c6f", "PPOlin_D2_clean5_zpd2": "#1e8449", "PPOlin_FS2_hist6": "#d35400"}
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    for v in VARIANTS:
        runs_v = list(bv[v].values())
        ep = np.array([c["episode"] for c in runs_v[0]["curve"]])
        for j, (m, title) in enumerate([("lg", "LG_sim en estudiantes NO vistos (holdout)"),
                                        ("shield", "Violaciones: activaciones del shield por paso")]):
            Y = np.array([_smooth([c[m] for c in r["curve"]]) for r in runs_v])
            mu, se = Y.mean(0), Y.std(0, ddof=1) / np.sqrt(len(Y))
            ax[j].plot(ep, mu, color=colors[v], lw=2, label=v)
            ax[j].fill_between(ep, mu - se, mu + se, color=colors[v], alpha=0.18)
            ax[j].set_title(title, fontweight="bold")
            ax[j].set_xlabel("Episodios de entrenamiento (evaluaciones cada 50)")
    ax[0].axhline(0, color="k", lw=0.8, ls=":")
    ax[0].set_ylabel("LG_sim (media 10 pares fold×seed, ±1 EE)")
    ax[1].set_ylabel("tasa de shield")
    ax[0].legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def diffs_figure(runs, path):
    import matplotlib.pyplot as plt
    pp = per_pair_table(runs)
    st = stats_table(runs)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for i, v in enumerate([v for v in VARIANTS if v != BASE]):
        d = pp[v] - pp[BASE]
        ax.scatter(np.full(len(d), i) + np.linspace(-0.12, 0.12, len(d)), d, s=36, alpha=0.8)
        row = st.iloc[i]
        ax.errorbar(i + 0.3, row.mean_diff, yerr=[[row.mean_diff - row.ci_low], [row.ci_high - row.mean_diff]],
                    fmt="D", color="k", capsize=6)
        ax.text(i + 0.36, row.mean_diff, f"Δ={row.mean_diff:+.4f}\nIC95 [{row.ci_low:+.4f}, {row.ci_high:+.4f}]\n"
                f"p_Holm={row.p_Holm:.3f}  ({row.wins}/{row.n} pares)", va="center", fontsize=8.5)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    n_v = len(VARIANTS) - 1
    ax.set_xticks(range(n_v))
    ax.set_xticklabels([f"{v.replace('PPOlin_','')}\n− Baseline" for v in VARIANTS if v != BASE], fontsize=8)
    ax.set_xlim(-0.4, n_v - 0.1)
    ax.set_ylabel("Δ LG_sim pareado (mismo fold, misma semilla)")
    ax.set_title("Efecto causal de cada cambio único (10 pares)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig
