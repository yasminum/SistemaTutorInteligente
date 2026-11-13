import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh()
        )
        self.actor_text = nn.Linear(256, 30)
        self.actor_diff = nn.Linear(256, 3)
        self.actor_scaffold = nn.Linear(256, 4)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        f = self.shared(x)
        return self.actor_text(f), self.actor_diff(f), self.actor_scaffold(f), self.critic(f)

class WorldModel(nn.Module):
    def __init__(self, obs_dim, action_dim=37):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, obs_dim + 1)
        )

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))
