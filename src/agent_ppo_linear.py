"""
agent_ppo_linear.py - PPO-clip (Schulman et al., 2017) con politica softmax LINEAL y critico lineal, en numpy.

logits(s, a) = W[a] . [x_s, 1]  +  w_a . Phi(s, a)
  - W   : pesos por accion sobre los rasgos de estado (equivale a one-hot(a) (x) x_s).
  - w_a : pesos compartidos sobre los rasgos condicionados a la accion (solo FS1).
V(s)    = v . [x_s, 1, mean_a Phi, max_a Phi]

Por que lineal: aisla el efecto de la REPRESENTACION (el unico factor que cambia entre variantes) y corre en
CPU / Colab / JupyterLite sin PyTorch. Limitacion declarada: el tamano del efecto es especifico de esta clase
de politica; la replica con el MLP de la tesis es el siguiente sprint.
"""
import numpy as np


class Adam:
    def __init__(self, shape, lr):
        self.lr, self.m, self.v, self.t = lr, np.zeros(shape), np.zeros(shape), 0

    def step(self, param, grad_ascent):
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * grad_ascent
        self.v = 0.999 * self.v + 0.001 * grad_ascent ** 2
        m_hat = self.m / (1 - 0.9 ** self.t)
        v_hat = self.v / (1 - 0.999 ** self.t)
        param += self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class LinearPPO:
    def __init__(self, d_state, d_action, n_actions, cfg, seed):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.n_actions, self.d_action = n_actions, d_action
        self.W = np.zeros((n_actions, d_state + 1))
        self.wa = np.zeros(max(d_action, 1))
        self.d_value = d_state + 1 + 2 * d_action
        self.v = np.zeros(self.d_value)
        self.opt_W = Adam(self.W.shape, cfg["lr_pi"])
        self.opt_wa = Adam(self.wa.shape, cfg["lr_pi"])
        self.opt_v = Adam(self.v.shape, cfg["lr_v"])

    # ---------- inferencia ----------
    def _logits(self, xs1, phi):
        z = self.W @ xs1
        if self.d_action:
            z = z + phi @ self.wa
        return z

    def value_features(self, xs1, phi):
        if not self.d_action:
            return xs1
        return np.concatenate([xs1, phi.mean(0), phi.max(0)])

    def act(self, xs, phi, greedy=False):
        xs1 = np.append(xs, 1.0)
        pi = _softmax(self._logits(xs1, phi))
        a = int(np.argmax(pi)) if greedy else int(self.rng.choice(self.n_actions, p=pi))
        return a, float(np.log(pi[a] + 1e-12)), xs1

    def param_fingerprint(self):
        return float(np.sum(self.W) + 3.0 * np.sum(self.wa) + 7.0 * np.sum(self.v))

    # ---------- aprendizaje ----------
    def update(self, batch):
        """batch: dict con X1 (N,d+1), PHI (N,A,da) o None, A (N,), LOGP (N,), ADV (N,), RET (N,), VF (N,dv)."""
        cfg = self.cfg
        X1, PHI, acts, logp_old = batch["X1"], batch["PHI"], batch["A"], batch["LOGP"]
        adv, ret, VF = batch["ADV"], batch["RET"], batch["VF"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(acts)
        for _ in range(cfg["epochs"]):
            idx = self.rng.permutation(n)
            for s in range(0, n, cfg["minibatch"]):
                b = idx[s:s + cfg["minibatch"]]
                x, a, lo, ad = X1[b], acts[b], logp_old[b], adv[b]
                z = x @ self.W.T
                if self.d_action:
                    ph = PHI[b]
                    z = z + np.einsum("nad,d->na", ph, self.wa)
                pi = _softmax(z)
                logp = np.log(pi[np.arange(len(b)), a] + 1e-12)
                ratio = np.exp(logp - lo)
                eps = cfg["clip"]
                active = ((ad >= 0) & (ratio < 1 + eps)) | ((ad < 0) & (ratio > 1 - eps))
                coef = np.where(active, ratio * ad, 0.0)
                onehot = np.zeros_like(pi)
                onehot[np.arange(len(b)), a] = 1.0
                g = coef[:, None] * (onehot - pi)                        # d L_clip / d logits
                logpi = np.log(pi + 1e-12)
                ent = -(pi * logpi).sum(1, keepdims=True)
                g += cfg["ent_coef"] * (-pi * (logpi + ent))             # d H / d logits
                self.opt_W.step(self.W, g.T @ x / len(b))
                if self.d_action:
                    self.opt_wa.step(self.wa, np.einsum("na,nad->d", g, ph) / len(b))
                # critico: descenso en MSE  <=>  ascenso en -MSE
                vf = VF[b]
                err = ret[b] - vf @ self.v
                self.opt_v.step(self.v, vf.T @ err / len(b))

    def gae(self, rewards, values, gamma, lam):
        """Episodio terminal en H (sin bootstrap)."""
        T = len(rewards)
        adv = np.zeros(T)
        last = 0.0
        for t in reversed(range(T)):
            nv = values[t + 1] if t + 1 < T else 0.0
            delta = rewards[t] + gamma * nv - values[t]
            last = delta + gamma * lam * last
            adv[t] = last
        return adv, adv + np.asarray(values)
