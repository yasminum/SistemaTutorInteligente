# config.py - Configuración ganadora tras Optimización de Hiperparámetros (Semana 8)
import os
import random
import numpy as np

MASTER_SEED = 42
EVAL_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 10 semillas pareadas (seed_0 .. seed_9)


def set_global_seed(seed: int = MASTER_SEED) -> None:
    """Fija las semillas globales (Python, NumPy, Hash y PyTorch/CUDA) para reproducibilidad."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


CONFIG = {
    "num_texts": 30,
    "num_items_per_text": 10,
    "num_skills": 3,
    "horizon": 50,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.20,
    "entropy_coef": 0.01,
    "vf_coef": 0.5,
    "ppo_epochs": 10,
    "batch_size": 64,
    "lr": 1e-4,                 # Optimizado en HPO (antes 3e-4)
    "total_episodes": 5000,
    "seed": MASTER_SEED,        # Semilla maestra (split 5-Fold y TPESampler)
    "seeds": EVAL_SEEDS,        # 10 semillas experimentales pareadas [0..9]
    "dyna_rollouts": 5,
    "dyna_horizon": 5,          # H = 5 confirmado en HPO
    "wm_mse_threshold": 0.05,   # eps_wm = 0.05 optimizado en HPO (antes 0.10)
    "device": "cpu",
}
