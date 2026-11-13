"""
experiment.py - Protocolo A/B (IA05): GroupKFold por estudiante, semillas fijas, logging y estadistica pareada.
"""
import csv
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone

import numpy as np

from .agent_ppo_linear import LinearPPO
from .feature_engineering import CausalHistory, EloTheta, FeatureBuilder, Scaler
from .simulator import HORIZON, N_ACTIONS, N_LEVELS, ItemBank, TutorEnv

VARIANTS = {
    "PPOlin_FS0_raw13": "FS0",                # Baseline: vector x_t crudo (13 rasgos)
    "PPOlin_FS1_zpd4": "FS1",                 # Var1: +4 rasgos ZPD por accion (con colinealidad VIF=3.96)
    "PPOlin_D1_zpd2": "D1_zpd2",              # Ablacion parsimoniosa: +2 rasgos ZPD (abs_gap, zpd_hit; VIF=2.12)
    "PPOlin_D2_clean5_zpd2": "D2_clean5_zpd2",# Var2 Adoptada: Estado depurado (5 vars, sin 5 const ni 3 ruidos) + 2 vars ZPD
    "PPOlin_FS2_hist6": "FS2",                # Variante temporal: +6 rasgos de historial causal (descartada)
}
NOTES = {
    "PPOlin_FS0_raw13": "Baseline: 13 rasgos crudos x_t de environment.py (incluye 5 constantes SD=0 y 3 ruidos U(0.05,0.15))",
    "PPOlin_FS1_zpd4": "Var1: Estado crudo (13) + 4 rasgos ZPD por accion (p_hat, abs_gap, zpd_prox, zpd_hit; VIF=3.96)",
    "PPOlin_D1_zpd2": "Ablacion ZPD-2: Estado crudo (13) + solo 2 rasgos ZPD seleccionados por SHAP/Perm (abs_gap, zpd_hit)",
    "PPOlin_D2_clean5_zpd2": "Var2 (Adoptada): Poda de 8 columnas muertas/ruido del estado (13->5) + 2 rasgos ZPD (abs_gap, zpd_hit)",
    "PPOlin_FS2_hist6": "Var Historial: Estado crudo (13) + 6 rasgos temporales causales (acc_w5, racha, EWMA, nivel t-1)",
}

CFG = {
    "n_students": 100, "k_folds": 5, "fold_seed": 2026, "bank_seed": 2026, "seeds": [0, 1],
    "train_episodes": 1500, "episodes_per_update": 8, "eval_every": 50,
    "eval_reps_curve": 1, "eval_reps_final": 3, "scaler_episodes": 20,
    "lr_pi": 0.02, "lr_v": 0.01, "epochs": 4, "minibatch": 128, "clip": 0.2, "ent_coef": 0.01,
    "gamma": 0.99, "lam": 0.95, "elo_k": 0.1,
}


def make_folds(n_students, k, seed):
    """GroupKFold: la entidad es el estudiante; ningun id aparece en train y test del mismo fold."""
    perm = np.random.default_rng(seed).permutation(n_students)
    groups = np.array_split(perm, k)
    return [(np.sort(np.concatenate([g for j, g in enumerate(groups) if j != f])), np.sort(groups[f]))
            for f in range(k)]


def file_sha256(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        with open(p, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def _episode_seed(student_id, tag, rep):
    return int(1_000_003 * (student_id + 1) + 7_919 * tag + rep)


class Runner:
    """Ejecuta una variante en un (fold, seed). theta_mode: 'oracle' (como la tesis) o 'elo' (observable)."""

    def __init__(self, variant, fold, seed, cfg=CFG, theta_mode="oracle", bank=None):
        self.variant, self.fold, self.seed, self.cfg = variant, fold, seed, cfg
        self.fs = VARIANTS[variant]
        self.theta_mode = theta_mode
        self.bank = bank or ItemBank(cfg["bank_seed"])
        self.env = TutorEnv(self.bank)
        self.fb = FeatureBuilder(self.fs, self.bank, noise_seed=seed)
        self.hist = CausalHistory()
        self.elo = EloTheta(cfg["elo_k"])
        folds = make_folds(cfg["n_students"], cfg["k_folds"], cfg["fold_seed"])
        self.train_ids, self.test_ids = folds[fold]
        self.rng = np.random.default_rng(10 * seed + fold)
        self.agent = LinearPPO(self.fb.d_state, self.fb.d_action, N_ACTIONS, cfg, seed=seed)
        self.scaler = Scaler()

    # ---------- observacion causal ----------
    def _theta_obs(self):
        return self.env.student.theta if self.theta_mode == "oracle" else self.elo.theta

    def _observe(self):
        th = self._theta_obs()
        xs = self.fb.state_features(th, self.hist)
        phi = self.fb.action_features(th)
        return xs, phi

    def _post_step(self, info):
        self.hist.update(info["correct"], info["level_applied"], info["b_raw"])
        if self.theta_mode == "elo":
            self.elo.update(info["correct"], info["a"], info["b_adj"], info["k"])

    def _reset(self, student_id, ep_seed):
        self.env.reset(student_id, ep_seed)
        self.hist.reset()
        self.elo.reset()

    # ---------- escalado: fit SOLO con estudiantes de train ----------
    def fit_scaler(self):
        Xs, Ph, used = [], [], []
        rng = np.random.default_rng(999 + self.fold)
        for e in range(self.cfg["scaler_episodes"]):
            sid = int(rng.choice(self.train_ids))
            used.append(sid)
            self._reset(sid, _episode_seed(sid, 1, e))
            done = False
            while not done:
                xs, phi = self._observe()
                Xs.append(xs)
                if phi is not None:
                    Ph.append(phi)
                _, done, info = self.env.step(int(rng.integers(N_ACTIONS)))
                self._post_step(info)
        Ph = np.concatenate(Ph) if Ph else None
        self.scaler.fit(np.array(Xs), Ph, fit_ids=used)

    # ---------- episodio ----------
    def run_episode(self, student_id, ep_seed, greedy, collect=False, timer=None):
        self._reset(student_id, ep_seed)
        traj = {"X1": [], "PHI": [], "A": [], "LOGP": [], "R": [], "VF": []}
        done, ret = False, 0.0
        while not done:
            t0 = time.perf_counter()
            xs, phi = self._observe()
            xs_n, phi_n = self.scaler.tx_state(xs), self.scaler.tx_action(phi)
            a, logp, xs1 = self.agent.act(xs_n, phi_n, greedy=greedy)
            if timer is not None:
                timer.append(time.perf_counter() - t0)
            r, done, info = self.env.step(a)
            self._post_step(info)
            ret += r
            if collect:
                traj["X1"].append(xs1); traj["PHI"].append(phi_n); traj["A"].append(a)
                traj["LOGP"].append(logp); traj["R"].append(r)
                traj["VF"].append(self.agent.value_features(xs1, phi_n))
        out = {"lg": self.env.learning_gain(), "acc": self.env.n_correct / HORIZON, "ret": ret,
               "shield": self.env.n_shield / HORIZON, "cost": self.env.n_cost / HORIZON,
               "learn": self.env.n_learn / HORIZON}
        return out, traj

    def evaluate(self, reps, timer=None):
        fp = self.agent.param_fingerprint()
        rows = [self.run_episode(sid, _episode_seed(sid, 2, r), greedy=True, timer=timer)[0]
                for sid in self.test_ids for r in range(reps)]
        assert self.agent.param_fingerprint() == fp, "La politica cambio durante la evaluacion"
        return {k: float(np.mean([x[k] for x in rows])) for k in rows[0]}

    def train(self, verbose=False):
        cfg = self.cfg
        t_start = time.perf_counter()
        self.fit_scaler()
        curve = []
        buf = []
        train_lg, train_ret = [], []
        for ep in range(1, cfg["train_episodes"] + 1):
            sid = int(self.rng.choice(self.train_ids))
            out, traj = self.run_episode(sid, _episode_seed(sid, 0, ep + 100_000 * self.seed),
                                         greedy=False, collect=True)
            train_lg.append(out["lg"]); train_ret.append(out["ret"])
            buf.append(traj)
            if len(buf) == cfg["episodes_per_update"]:
                self._update(buf)
                buf = []
            if ep % cfg["eval_every"] == 0:
                ev = self.evaluate(cfg["eval_reps_curve"])
                curve.append({"episode": ep, **ev})
                if verbose:
                    print(f"  {self.variant} f{self.fold} s{self.seed} ep{ep}: LG={ev['lg']:.4f}")
        train_time = time.perf_counter() - t_start
        timer = []
        final = self.evaluate(cfg["eval_reps_final"], timer=timer)
        lat = np.array(timer) * 1e3
        return {"variant": self.variant, "fold": self.fold, "seed": self.seed, "theta_mode": self.theta_mode,
                "final": final, "curve": curve, "train_lg": train_lg, "train_ret": train_ret,
                "lat_ms_median": float(np.median(lat)), "lat_ms_p95": float(np.percentile(lat, 95)),
                "train_time_s": train_time, "train_ids": self.train_ids.tolist(),
                "test_ids": self.test_ids.tolist(), "scaler_fit_ids": sorted(self.scaler.fit_ids)}

    def _update(self, buf):
        cfg = self.cfg
        X1, PHI, A, LOGP, ADV, RET, VF = [], [], [], [], [], [], []
        for tr in buf:
            vf = np.array(tr["VF"])
            values = vf @ self.agent.v
            adv, ret = self.agent.gae(np.array(tr["R"]), values, cfg["gamma"], cfg["lam"])
            X1 += tr["X1"]; A += tr["A"]; LOGP += tr["LOGP"]; VF.append(vf)
            ADV.append(adv); RET.append(ret)
            if self.agent.d_action:
                PHI += tr["PHI"]
        self.agent.update({"X1": np.array(X1), "PHI": np.array(PHI) if PHI else None,
                           "A": np.array(A), "LOGP": np.array(LOGP), "ADV": np.concatenate(ADV),
                           "RET": np.concatenate(RET), "VF": np.concatenate(VF)})


# ---------- politicas de referencia (no son variantes A/B; solo contexto) ----------
def evaluate_reference(policy, fold, cfg=CFG, bank=None):
    bank = bank or ItemBank(cfg["bank_seed"])
    env = TutorEnv(bank)
    _, test_ids = make_folds(cfg["n_students"], cfg["k_folds"], cfg["fold_seed"])[fold]
    rows = []
    for sid in test_ids:
        for r in range(cfg["eval_reps_final"]):
            env.reset(int(sid), _episode_seed(int(sid), 2, r))
            rng = np.random.default_rng(_episode_seed(int(sid), 3, r))
            done, ret = False, 0.0
            while not done:
                if policy == "random":
                    a = int(rng.integers(N_ACTIONS))
                else:                                   # secuencia fija: textos en orden, nivel medio
                    a = (env.t % 30) * N_LEVELS + 1
                rwd, done, _ = env.step(a)
                ret += rwd
            rows.append({"lg": env.learning_gain(), "acc": env.n_correct / HORIZON, "ret": ret,
                         "shield": env.n_shield / HORIZON, "cost": env.n_cost / HORIZON,
                         "learn": env.n_learn / HORIZON})
    return {k: float(np.mean([x[k] for x in rows])) for k in rows[0]}


# ---------- estadistica pareada (sin scipy: corre en Pyodide) ----------
def paired_stats(a, b, n_boot=10_000, seed=0):
    """Diferencia pareada d = b - a: media, IC95% bootstrap, p exacto por permutacion de signos, d_z de Cohen."""
    d = np.asarray(b) - np.asarray(a)
    n = len(d)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, n, (n_boot, n))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    if n <= 16:
        signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    else:
        signs = rng.choice([-1, 1], size=(20_000, n))
    null = np.abs((signs * d).mean(1))
    p = float((null >= abs(d.mean()) - 1e-12).mean())
    dz = float(d.mean() / (d.std(ddof=1) + 1e-12))
    return {"mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "p_perm": p, "d_z": dz,
            "wins": int((d > 0).sum()), "n": n}


def episodes_to_target(curve, target):
    for row in curve:
        if row["lg"] >= target:
            return row["episode"]
    return np.nan


def environment_fingerprint():
    import numpy, pandas, matplotlib  # noqa
    return {"python": platform.python_version(), "numpy": numpy.__version__,
            "pandas": pandas.__version__, "matplotlib": matplotlib.__version__}


def log_row(path, row):
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def make_log_row(res, cfg, code_hash, data_hash):
    f = res["final"]
    return {
        "exp_id": f"FE5-{res['variant']}-{res['theta_mode']}-f{res['fold']}-s{res['seed']}",
        "variant": res["variant"], "theta_mode": res["theta_mode"], "fold": res["fold"], "seed": res["seed"],
        "LG_holdout": round(f["lg"], 6), "acc": round(f["acc"], 6), "return": round(f["ret"], 6),
        "shield_rate": round(f["shield"], 6), "cost_rate": round(f["cost"], 6),
        "learn_event_rate": round(f["learn"], 6),
        "lat_ms_median": round(res["lat_ms_median"], 5), "lat_ms_p95": round(res["lat_ms_p95"], 5),
        "train_time_s": round(res["train_time_s"], 3), "train_episodes": cfg["train_episodes"],
        "n_train_students": len(res["train_ids"]), "n_test_students": len(res["test_ids"]),
        "code_sha256": code_hash[:16], "data_sha256": data_hash[:16],
        "config": json.dumps({k: cfg[k] for k in sorted(cfg) if k != "seeds"}),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
