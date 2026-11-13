import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from collections import deque

class IRTStudent:
    def __init__(self, skill_dim=3, seed=None):
        self.rng = np.random.default_rng(seed)
        self.theta = self.rng.beta(2, 2, size=skill_dim)
        self.skill_dim = skill_dim
        self.fatigue = 0.0
        self.streak_errors = 0

    def respond(self, item_params):
        a, b, c, k = item_params
        p_base = c + (1 - c) / (1 + np.exp(-1.7 * a * (self.theta[k] - b)))
        return 1 if self.rng.random() < p_base * (1 - 0.1 * self.fatigue) else 0

    def learn(self, item_params, correct):
        a, b, c, k = item_params
        if abs(self.theta[k] - b) < 0.5 and correct:
            self.theta[k] += self.rng.uniform(0.02, 0.05) * (1 - self.theta[k])
        self.theta = np.clip(self.theta * (1 - self.rng.uniform(0.001, 0.005)), 0.0, 1.0)
        self.streak_errors = self.streak_errors + 1 if not correct else max(0, self.streak_errors - 1)
        self.fatigue = min(1.0, self.fatigue + 0.05) if not correct else max(0.0, self.fatigue - 0.02)
        return self.theta.copy()

class ReadingTutorEnv(gym.Env):
    def __init__(self, config, seed=None):
        super().__init__()
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.num_texts = config["num_texts"]
        self.horizon = config["horizon"]
        
        self.item_bank = []
        for t in range(self.num_texts):
            for _ in range(config["num_items_per_text"]):
                self.item_bank.append((self.rng.uniform(0.8, 2.0), self.rng.normal(0, 1), 0.25, self.rng.integers(0, 3), t))
                
        self.action_space = spaces.MultiDiscrete([self.num_texts, 3, 4])
        self.obs_dim = 269
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.obs_dim,), dtype=np.float32)
        self.history = deque(maxlen=20)
        self.lstm = nn.LSTM(input_size=13, hidden_size=256, batch_first=True).eval()
        for p in self.lstm.parameters():
            p.requires_grad = False
        self._init_lstm_weights()

    def _init_lstm_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.student = IRTStudent(3, seed=seed)
        self.history.clear()
        self.step_count = 0
        self.pre_test_theta = self.student.theta.copy()
        self.last_item_idx = -1
        self.last_difficulty = 0
        return self._get_obs(), {}

    def _get_obs(self):
        theta = self.student.theta
        sigma = np.random.uniform(0.05, 0.15, size=3)
        hist = [0.5, 0.5, self.step_count / self.horizon]
        tau = [0.5, 0.5]
        E = 0.7
        b_prev = 0.0 if self.last_item_idx == -1 else self.item_bank[self.last_item_idx][1]
        x_t = np.concatenate([theta, sigma, hist, tau, [E], [b_prev]]).astype(np.float32)
        with torch.no_grad():
            self.history.append(x_t)
            if len(self.history) < 2:
                h_t = np.zeros(256, dtype=np.float32)
            else:
                seq = torch.tensor(np.array(list(self.history)[-10:]).reshape(1, -1, 13), dtype=torch.float32)
                _, (h_n, _) = self.lstm(seq)
                h_t = h_n.view(-1).numpy()
        return np.concatenate([h_t, x_t])

    def step(self, action):
        text_idx, diff_level, scaffold = action
        self.step_count += 1
        if self.student.streak_errors >= 4:
            diff_level = 0 # CMDP Shield
        item_pool = [i for i, item in enumerate(self.item_bank) if item[4] == text_idx]
        item_idx = self.rng.choice(item_pool) if item_pool else self.rng.integers(0, len(self.item_bank))
        a, b, c, k, _ = self.item_bank[item_idx]
        b_adj = b + (diff_level - 1) * 0.5
        correct = self.student.respond((a, b_adj, c, k))
        self.student.learn((a, b_adj, c, k), correct)
        
        delta_ks = max(0, self.student.theta[k] - (self.student.theta[k] - 0.01 * (1 if correct else 0)))
        E_hat = 0.7 + 0.3 * correct
        F_hat = np.exp(-((b_adj - self.student.theta[k])**2) / (2 * 0.25**2))
        cost = 0.0
        
        if self.last_item_idx == item_idx:
            cost += 0.5
        if abs(diff_level - self.last_difficulty) > 1:
            cost += 0.5
            
        self.last_item_idx = item_idx
        self.last_difficulty = diff_level
        R = np.clip(0.5 * delta_ks + 0.25 * E_hat + 0.25 * F_hat - 0.5 * cost, -1, 1)
        done = self.step_count >= self.horizon
        
        return self._get_obs(), R, done, done, {"correct": correct, "theta": self.student.theta.copy()}

    def get_lg(self):
        pre, post = np.mean(self.pre_test_theta), np.mean(self.student.theta)
        return (post - pre) / (1.0 - pre) if (1.0 - pre) != 0 else 0.0
