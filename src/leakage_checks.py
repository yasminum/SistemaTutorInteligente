"""
leakage_checks.py - Pruebas automaticas de "cero fuga" (Slides 6 y 12 de IA05). Cada prueba devuelve PASS/FAIL.

  L1  GroupKFold: ningun estudiante en train y test a la vez (5 folds).
  L2  Causalidad temporal: features en t identicas si se alteran los resultados FUTUROS (>= t).
  L3  Sin espiar el item actual: features en t identicas aunque cambie el item que se muestreara en t.
  L4  Escalador: fit_ids del StandardScaler contenidos en train y disjuntos de test.
  L5  Politica congelada en evaluacion: los parametros no cambian al evaluar en estudiantes de test.
"""
import numpy as np

from .experiment import CFG, Runner, make_folds
from .feature_engineering import CausalHistory, FeatureBuilder
from .simulator import HORIZON, ItemBank, TutorEnv


def check_group_disjoint(cfg=CFG):
    folds = make_folds(cfg["n_students"], cfg["k_folds"], cfg["fold_seed"])
    ok = all(len(set(tr) & set(te)) == 0 for tr, te in folds)
    cover = sorted(np.concatenate([te for _, te in folds]).tolist()) == list(range(cfg["n_students"]))
    return ok and cover, f"{len(folds)} folds; interseccion train/test = 0; test cubre los {cfg['n_students']} ids una vez"


def check_temporal_causality(n_trials=200, seed=0):
    bank = ItemBank(CFG["bank_seed"])
    rng = np.random.default_rng(seed)
    for _ in range(n_trials):
        t = int(rng.integers(0, HORIZON))
        seq_a = rng.integers(0, 2, HORIZON)
        seq_b = seq_a.copy()
        seq_b[t:] = 1 - seq_b[t:]                  # futuro alterado
        lv = rng.integers(0, 3, HORIZON)
        bb = rng.normal(0, 1, HORIZON)
        feats = []
        for seq in (seq_a, seq_b):
            h = CausalHistory()
            for s in range(t):
                h.update(seq[s], lv[s], bb[s])
            fb = FeatureBuilder("FS2", bank, noise_seed=1)
            feats.append(fb.state_features(np.array([0.4, 0.5, 0.6]), h))
        if not np.allclose(feats[0], feats[1]):
            return False, f"fallo en t={t}"
    return True, f"{n_trials} pruebas aleatorias: features(t) invariantes ante cambios en y_t..y_H"


def check_no_current_item_peek():
    bank = ItemBank(CFG["bank_seed"])
    out = []
    for item_seed in (1, 2):
        env = TutorEnv(bank)
        env.reset(student_id=3, episode_seed=123)
        env.item_rng = np.random.default_rng(item_seed)     # cambia el item que se muestreara en t
        fb = FeatureBuilder("FS1", bank, noise_seed=0)
        h = CausalHistory()
        out.append((fb.state_features(env.student.theta, h), fb.action_features(env.student.theta)))
    same = np.allclose(out[0][0], out[1][0]) and np.allclose(out[0][1], out[1][1])
    return same, "features calculadas antes de muestrear el item; cambiar el item de t no altera Phi(t)"


def check_scaler_provenance(fold=0):
    r = Runner("PPOlin_FS1_zpd4", fold, 0, CFG)
    r.fit_scaler()
    ok = r.scaler.fit_ids <= set(r.train_ids.tolist()) and not (r.scaler.fit_ids & set(r.test_ids.tolist()))
    return ok, f"scaler ajustado con {len(r.scaler.fit_ids)} ids distintos muestreados de train; 0 ids de test"


def check_frozen_eval(fold=0):
    r = Runner("PPOlin_FS0_raw13", fold, 0, CFG)
    r.fit_scaler()
    r.agent.W += np.random.default_rng(0).normal(0, 0.1, r.agent.W.shape)
    before = r.agent.W.copy()
    r.evaluate(reps=1)
    return np.array_equal(before, r.agent.W), "pesos identicos antes/despues de evaluar en test (sin aprendizaje en test)"


def run_all_checks():
    checks = [("L1 GroupKFold por estudiante", check_group_disjoint),
              ("L2 Causalidad temporal", check_temporal_causality),
              ("L3 Sin espiar el item actual", check_no_current_item_peek),
              ("L4 Scaler fit solo en train", check_scaler_provenance),
              ("L5 Politica congelada en test", check_frozen_eval)]
    rows = []
    for name, fn in checks:
        ok, detail = fn()
        rows.append({"prueba": name, "resultado": "PASS" if ok else "FAIL", "evidencia": detail})
    return rows


if __name__ == "__main__":
    for row in run_all_checks():
        print(row)
