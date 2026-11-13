"""
diagnostics.py - Diagnostico univariado de features (solo con estudiantes de TRAIN del fold 0).

Pregunta que responde: que representacion de la distancia habilidad-dificultad explica mejor el evento
de aprendizaje del simulador? Se compara el escalado sigmoidal directo (theta - sigmoid(b)) con la distancia
en la escala nativa del modelo generativo (theta - b_adj). Metrica: AUC (Mann-Whitney) para predecir `learned` y `correct`.
"""
import numpy as np

from .experiment import CFG, _episode_seed, make_folds
from .feature_engineering import FS0_NAMES, CausalHistory, FeatureBuilder
from .simulator import N_ACTIONS, ItemBank, TutorEnv


def auc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, int)
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):                      # rangos promedio con empates
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def collect_random_interactions(n_episodes=200, fold=0, seed=0):
    bank = ItemBank(CFG["bank_seed"])
    env = TutorEnv(bank)
    train_ids, _ = make_folds(CFG["n_students"], CFG["k_folds"], CFG["fold_seed"])[fold]
    rng = np.random.default_rng(seed)
    fb = FeatureBuilder("FS0", bank, noise_seed=seed)
    rows, X0 = [], []
    for e in range(n_episodes):
        sid = int(rng.choice(train_ids))
        env.reset(sid, _episode_seed(sid, 4, e))
        hist = CausalHistory()
        done = False
        while not done:
            X0.append(fb.state_features(env.student.theta, hist))
            theta_before = env.student.theta.copy()
            _, done, info = env.step(int(rng.integers(N_ACTIONS)))
            hist.update(info["correct"], info["level_applied"], info["b_raw"])
            th = theta_before[info["k"]]
            rows.append({"theta_k": th, "b_adj": info["b_adj"], "a": info["a"],
                         "gap_raw": th - info["b_adj"], "gap_sigmoid": th - 1 / (1 + np.exp(-info["b_adj"])),
                         "correct": info["correct"], "learned": info["learned"], "student": sid})
    return rows, np.array(X0), train_ids


def feature_diagnostics():
    rows, X0, train_ids = collect_random_interactions()
    y_learn = np.array([r["learned"] for r in rows])
    y_corr = np.array([r["correct"] for r in rows])
    g_raw = np.array([r["gap_raw"] for r in rows])
    g_sig = np.array([r["gap_sigmoid"] for r in rows])
    table = [
        {"feature": "-|theta - b_adj|  (FS1, escala nativa del modelo)", "AUC_learned": auc(-np.abs(g_raw), y_learn),
         "AUC_correct": auc(g_raw, y_corr)},
        {"feature": "-|theta - sigmoid(b)|  (Escalado sigmoidal directo)", "AUC_learned": auc(-np.abs(g_sig), y_learn),
         "AUC_correct": auc(g_sig, y_corr)},
        {"feature": "zpd_hit = 1[|theta - b_adj| < 0.5]", "AUC_learned": auc((np.abs(g_raw) < 0.5).astype(float), y_learn),
         "AUC_correct": auc((np.abs(g_raw) < 0.5).astype(float), y_corr)},
    ]
    # Diagnostico de varianza del vector x_t base con respecto al evento de aprendizaje
    audit = []
    for j, name in enumerate(FS0_NAMES):
        col = X0[:, j]
        sd = float(col.std())
        a = auc(col, y_learn) if sd > 1e-12 else 0.5
        kind = "constante" if sd < 1e-12 else ("ruido U(0.05,0.15)" if name.startswith("sigma") else "variable")
        audit.append({"rasgo_x_t": name, "desv_std": sd, "AUC_learned": a, "diagnostico": kind})
    meta = {"n_interacciones": len(rows), "n_estudiantes_train": len(set(r["student"] for r in rows)),
            "tasa_aprendizaje_aleatoria": float(y_learn.mean()), "tasa_acierto_aleatoria": float(y_corr.mean())}
    return table, audit, meta, rows
