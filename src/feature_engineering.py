"""
feature_engineering.py - Constructores de estado y accion (FS0, FS1_zpd4, FS2_hist6, D1_zpd2, D2_clean5_zpd2),
historial causal, estimador Elo online y StandardScaler con registro de student_id de entrenamiento.

Regla metodologica (divide primero, transforma despues):
  * Toda variable del paso t se evalua ANTES de ejecutar a_t, empleando unicamente:
      - habilidad observada (theta* en modo oracle o theta_hat Elo en prueba de estres),
      - historial causal de pasos 1..t-1 (CausalHistory),
      - parametros publicos del banco (a, b, k) promediados por texto (sin observar el item muestreado en t).
  * El StandardScaler se ajusta exclusivamente sobre trayectorias de train_ids del fold activo.

Conjuntos de datos / features evaluados:
  FS0  raw13         : Vector x_t original (13 rasgos de environment.py: incluye 5 constantes y 3 ruidos sigma).
  FS1  zpd4          : FS0 + 4 variables ZPD por accion (p_hat, abs_gap, zpd_prox, zpd_hit; VIF_max = 3.96).
  FS2  hist6         : FS0 + 6 variables temporales causales (acc_w5, err_streak, ewma_err, one-hot nivel t-1).
  D1   zpd2          : FS0 + 2 variables ZPD seleccionadas tras poda de colinealidad (abs_gap, zpd_hit; VIF = 2.12).
  D2   clean5_zpd2   : Estado depurado (5 variables con senal real: theta_1..3, u_progress, b_prev_raw; se eliminan
                       5 constantes SD=0 y 3 ruidos U(0.05,0.15)) + las 2 variables ZPD (abs_gap, zpd_hit).
"""
import numpy as np

from .simulator import (C_GUESS, HORIZON, LEVEL_SHIFT, N_ACTIONS, N_LEVELS, N_SKILLS,
                        SIGMA_F, ZPD_HALF_WIDTH, p3pl)

FS0_NAMES = ["theta_1", "theta_2", "theta_3", "sigma_1", "sigma_2", "sigma_3",
             "u_acc_const", "u_streak_const", "u_progress", "tau_rt_const", "tau_lag_const",
             "E_const", "b_prev_raw"]
CLEAN5_IDX = [0, 1, 2, 8, 12]
CLEAN5_NAMES = [FS0_NAMES[i] for i in CLEAN5_IDX]
FS1_ACTION_NAMES = ["p_hat", "abs_gap", "zpd_prox", "zpd_hit"]
ZPD2_IDX = [1, 3]
ZPD2_NAMES = ["abs_gap", "zpd_hit"]
FS2_NAMES = ["roll_acc_w5", "err_streak", "ewma_err", "last_lvl_0", "last_lvl_1", "last_lvl_2"]


class CausalHistory:
    """Almacena estadisticos causales hasta t-1. `update` se invoca tras observar la respuesta del paso t."""

    def __init__(self, window=5, alpha=0.2):
        self.window, self.alpha = window, alpha
        self.reset()

    def reset(self):
        self.outcomes = []
        self.t = 0
        self.b_prev = 0.0
        self.last_level = 1
        self.err_streak = 0
        self.ewma_err = 0.0

    def update(self, correct, level_applied, b_raw):
        self.outcomes.append(int(correct))
        self.t += 1
        self.b_prev = float(b_raw)
        self.last_level = int(level_applied)
        self.err_streak = 0 if correct else self.err_streak + 1
        self.ewma_err = (1 - self.alpha) * self.ewma_err + self.alpha * (1 - int(correct))

    def rolling_acc(self):
        w = self.outcomes[-self.window:]
        return (sum(w) + 1.0) / (len(w) + 2.0)


class EloTheta:
    """Estimador online observable de habilidad (Pelanek, 2016) para evaluar sensibilidad sin oraculo theta*."""

    def __init__(self, k_factor=0.1, init=0.5):
        self.k_factor, self.init = k_factor, init
        self.reset()

    def reset(self):
        self.theta = np.full(N_SKILLS, self.init)

    def update(self, correct, a, b_adj, k):
        p_hat = p3pl(self.theta[k], a, b_adj)
        self.theta[k] = float(np.clip(self.theta[k] + self.k_factor * (correct - p_hat), 0.0, 1.0))


class FeatureBuilder:
    """Construye x_s (estado) y Phi (accion) para FS0, FS1, FS2, D1_zpd2 o D2_clean5_zpd2."""

    def __init__(self, feature_set, bank, noise_seed=0):
        assert feature_set in ("FS0", "FS1", "FS2", "D1_zpd2", "D2_clean5_zpd2")
        self.fs = feature_set
        self.bank = bank
        self.noise_rng = np.random.default_rng(noise_seed)
        if feature_set == "D2_clean5_zpd2":
            self.d_state = 5
            self.state_names = list(CLEAN5_NAMES)
        elif feature_set == "FS2":
            self.d_state = 19
            self.state_names = FS0_NAMES + FS2_NAMES
        else:
            self.d_state = 13
            self.state_names = list(FS0_NAMES)

        if feature_set == "FS1":
            self.d_action = 4
            self.action_names = list(FS1_ACTION_NAMES)
        elif feature_set in ("D1_zpd2", "D2_clean5_zpd2"):
            self.d_action = 2
            self.action_names = list(ZPD2_NAMES)
        else:
            self.d_action = 0
            self.action_names = []

        self.b_adj = bank.b[None, :, :] + LEVEL_SHIFT[:, None, None]
        self.a3 = np.broadcast_to(bank.a[None, :, :], self.b_adj.shape)

    def state_features(self, theta_obs, hist):
        sigma = self.noise_rng.uniform(0.05, 0.15, size=N_SKILLS)
        if self.fs == "D2_clean5_zpd2":
            return np.array([theta_obs[0], theta_obs[1], theta_obs[2], hist.t / HORIZON, hist.b_prev], dtype=np.float64)
        x = [theta_obs[0], theta_obs[1], theta_obs[2], sigma[0], sigma[1], sigma[2],
             0.5, 0.5, hist.t / HORIZON, 0.5, 0.5, 0.7, hist.b_prev]
        if self.fs == "FS2":
            lvl = [0.0, 0.0, 0.0]
            lvl[hist.last_level] = 1.0
            x += [hist.rolling_acc(), min(hist.err_streak, 5) / 5.0, hist.ewma_err] + lvl
        return np.asarray(x, dtype=np.float64)

    def action_features(self, theta_obs):
        """Phi[a] para a = texto*3 + nivel. Promedia sobre los 10 items del texto sin observar el item de t."""
        if self.d_action == 0:
            return None
        th = np.asarray(theta_obs)[self.bank.k][None, :, :]
        gap = th - self.b_adj
        p_hat = p3pl(th, self.a3, self.b_adj).mean(axis=2)
        abs_gap = np.abs(gap).mean(axis=2)
        zpd_prox = np.exp(-(gap ** 2) / (2 * SIGMA_F ** 2)).mean(axis=2)
        zpd_hit = (np.abs(gap) < ZPD_HALF_WIDTH).mean(axis=2)
        phi4 = np.stack([p_hat, abs_gap, zpd_prox, zpd_hit], axis=-1).transpose(1, 0, 2).reshape(N_ACTIONS, 4)
        if self.d_action == 2:
            return phi4[:, ZPD2_IDX]
        return phi4


class Scaler:
    """StandardScaler con trazabilidad de student_id empleados en el ajuste."""

    def __init__(self):
        self.fitted = False

    def fit(self, X_state, Phi_rows, fit_ids):
        self.mu_s, self.sd_s = X_state.mean(0), X_state.std(0)
        self.sd_s[self.sd_s < 1e-8] = 1.0
        if Phi_rows is not None and len(Phi_rows):
            self.mu_a, self.sd_a = Phi_rows.mean(0), Phi_rows.std(0)
            self.sd_a[self.sd_a < 1e-8] = 1.0
        else:
            self.mu_a = self.sd_a = None
        self.fit_ids = set(int(i) for i in fit_ids)
        self.fitted = True
        return self

    def tx_state(self, x):
        return (x - self.mu_s) / self.sd_s

    def tx_action(self, phi):
        return None if phi is None else (phi - self.mu_a) / self.sd_a


def sigmoid_gap(theta_k, b):
    """Escalado sigmoidal directo: theta - sigma(b)."""
    return theta_k - 1.0 / (1.0 + np.exp(-b))
