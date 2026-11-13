"""
simulator.py - Port numpy (sin PyTorch) de la dinamica de `tesis-drl-its/src/environment.py`.

Fidelidad con environment.py (misma regla, linea a linea):
  respond():  P = c + (1-c) / (1 + exp(-1.7 a (theta_k - b_adj)))   ; acierto si U < P * (1 - 0.1*fatiga)
  learn():    si |theta_k - b_adj| < 0.5 y acierto:  theta_k += U(0.02, 0.05) * (1 - theta_k)
              theta <- clip(theta * (1 - U(0.001, 0.005)), 0, 1)      (olvido multiplicativo)
              racha  <- racha + 1 si error, si no max(0, racha - 1)
              fatiga <- min(1, f + 0.05) si error, si no max(0, f - 0.02)
  step():     shield CMDP: si racha >= 4 -> nivel = 0 ;  b_adj = b + (nivel - 1) * 0.5
              R = clip(0.5*dKS + 0.25*E + 0.25*F - 0.5*costo, -1, 1)

Diferencias DECLARADAS respecto de environment.py:
  (1) Se omite el LSTM congelado con pesos aleatorios (h_t en R^256): no se entrena, es una
      proyeccion aleatoria de x_t; no aporta informacion nueva y se excluye por igual en todas las variantes.
  (2) Se omite la dimension "andamiaje" de la accion: environment.py la recibe pero nunca la usa (no-op).
      Espacio de accion efectivo: texto (30) x nivel de dificultad (3) = 90 acciones.
  (3) Banco de items FIJO y versionado (snapshot con hash SHA-256) en lugar de un banco distinto por semilla.
  (4) Cada episodio es una sesion de un estudiante identificado por `student_id` (perfil theta0 fijo por id),
      lo que permite particionar por estudiante (GroupKFold) sin fuga entre train y test.
"""
import hashlib
import io

import numpy as np

N_TEXTS, ITEMS_PER_TEXT, N_SKILLS, N_LEVELS = 30, 10, 3, 3
HORIZON = 50
C_GUESS = 0.25
ZPD_HALF_WIDTH = 0.5      # ventana de aprendizaje del simulador |theta - b_adj| < 0.5
SIGMA_F = 0.25            # ancho del termino de flujo F de la recompensa
N_ACTIONS = N_TEXTS * N_LEVELS
LEVEL_SHIFT = np.array([-0.5, 0.0, 0.5])   # (nivel - 1) * 0.5


def decode_action(action):
    """Accion entera -> (texto, nivel)."""
    return int(action) // N_LEVELS, int(action) % N_LEVELS


def p3pl(theta_k, a, b):
    """Probabilidad de acierto IRT-3PL (Birnbaum, 1968) con constante D = 1.7."""
    return C_GUESS + (1.0 - C_GUESS) / (1.0 + np.exp(-1.7 * a * (theta_k - b)))


class ItemBank:
    """Banco de 30 textos x 10 items: a ~ U(0.8, 2.0), b ~ N(0, 1), c = 0.25, competencia k ~ U{0,1,2}."""

    def __init__(self, seed=2026):
        rng = np.random.default_rng(seed)
        self.seed = seed
        self.a = rng.uniform(0.8, 2.0, (N_TEXTS, ITEMS_PER_TEXT))
        self.b = rng.normal(0.0, 1.0, (N_TEXTS, ITEMS_PER_TEXT))
        self.k = rng.integers(0, N_SKILLS, (N_TEXTS, ITEMS_PER_TEXT))

    def to_csv(self):
        buf = io.StringIO()
        buf.write("text_id,item_id,a,b,c,skill_k\n")
        for t in range(N_TEXTS):
            for i in range(ITEMS_PER_TEXT):
                buf.write(f"{t},{i},{self.a[t, i]:.6f},{self.b[t, i]:.6f},{C_GUESS:.2f},{self.k[t, i]}\n")
        return buf.getvalue()

    def sha256(self):
        return hashlib.sha256(self.to_csv().encode("utf-8")).hexdigest()


class Student:
    """Estudiante sintetico IRT-3PL. theta0 depende SOLO de student_id (perfil fijo por entidad)."""

    def __init__(self, student_id, episode_seed):
        self.student_id = int(student_id)
        self.theta0 = np.random.default_rng(10_000 + self.student_id).beta(2, 2, N_SKILLS)
        self.rng = np.random.default_rng(episode_seed)
        self.theta = self.theta0.copy()
        self.fatigue = 0.0
        self.streak_errors = 0

    def respond(self, a, b_adj, k):
        p_base = p3pl(self.theta[k], a, b_adj)
        return 1 if self.rng.random() < p_base * (1.0 - 0.1 * self.fatigue) else 0

    def learn(self, b_adj, k, correct):
        if abs(self.theta[k] - b_adj) < ZPD_HALF_WIDTH and correct:
            self.theta[k] += self.rng.uniform(0.02, 0.05) * (1.0 - self.theta[k])
        self.theta = np.clip(self.theta * (1.0 - self.rng.uniform(0.001, 0.005)), 0.0, 1.0)
        self.streak_errors = self.streak_errors + 1 if not correct else max(0, self.streak_errors - 1)
        self.fatigue = min(1.0, self.fatigue + 0.05) if not correct else max(0.0, self.fatigue - 0.02)


class TutorEnv:
    """Entorno de tutoria (MDP, H = 50). La recompensa replica environment.py sin cambios."""

    def __init__(self, bank):
        self.bank = bank

    def reset(self, student_id, episode_seed):
        self.student = Student(student_id, episode_seed)
        self.item_rng = np.random.default_rng(episode_seed + 7919)
        self.t = 0
        self.last_item = None
        self.last_level = 0
        self.pre_theta = self.student.theta.copy()
        self.n_shield = 0
        self.n_cost = 0
        self.n_correct = 0
        self.n_learn = 0

    def step(self, action):
        text, level = decode_action(action)
        shield = self.student.streak_errors >= 4
        if shield:                          # escudo CMDP (environment.py L88-89)
            level = 0
            self.n_shield += 1
        i = int(self.item_rng.integers(ITEMS_PER_TEXT))   # el item concreto se muestrea DESPUES de decidir
        a, b, k = self.bank.a[text, i], self.bank.b[text, i], int(self.bank.k[text, i])
        b_adj = b + (level - 1) * 0.5
        theta_before = self.student.theta[k]
        correct = self.student.respond(a, b_adj, k)
        self.student.learn(b_adj, k, correct)
        learned = int(abs(theta_before - b_adj) < ZPD_HALF_WIDTH and correct)

        delta_ks = 0.01 * correct           # replica literal de environment.py L97
        e_hat = 0.7 + 0.3 * correct
        f_hat = float(np.exp(-((b_adj - self.student.theta[k]) ** 2) / (2 * SIGMA_F ** 2)))
        cost = 0.0
        if self.last_item == (text, i):
            cost += 0.5
        if abs(level - self.last_level) > 1:
            cost += 0.5
        reward = float(np.clip(0.5 * delta_ks + 0.25 * e_hat + 0.25 * f_hat - 0.5 * cost, -1.0, 1.0))

        self.last_item = (text, i)
        self.last_level = level
        self.t += 1
        self.n_correct += correct
        self.n_cost += int(cost > 0)
        self.n_learn += learned
        done = self.t >= HORIZON
        info = {"correct": correct, "k": k, "a": a, "b_raw": b, "b_adj": b_adj,
                "level_applied": level, "shield": shield, "learned": learned}
        return reward, done, info

    def learning_gain(self):
        """Ganancia normalizada de Hake sobre la habilidad VERDADERA theta* (metrica externa)."""
        pre, post = float(np.mean(self.pre_theta)), float(np.mean(self.student.theta))
        return (post - pre) / (1.0 - pre) if (1.0 - pre) != 0 else 0.0
