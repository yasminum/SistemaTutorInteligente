import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from .models import ActorCritic, WorldModel

class PPOAgent:
    def __init__(self, obs_dim, config):
        self.config = config
        self.device = torch.device(config["device"])
        self.ac = ActorCritic(obs_dim).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=config["lr"])
        self.memory = []

    def get_action(self, obs, deterministic=False):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        logits_t, logits_d, logits_s, _ = self.ac(obs_t)
        dist_t = torch.distributions.Categorical(logits=logits_t)
        dist_d = torch.distributions.Categorical(logits=logits_d)
        dist_s = torch.distributions.Categorical(logits=logits_s)
        
        a_t = torch.argmax(logits_t) if deterministic else dist_t.sample()
        a_d = torch.argmax(logits_d) if deterministic else dist_d.sample()
        a_s = torch.argmax(logits_s) if deterministic else dist_s.sample()
        
        log_prob = dist_t.log_prob(a_t) + dist_d.log_prob(a_d) + dist_s.log_prob(a_s)
        return torch.tensor([a_t.item(), a_d.item(), a_s.item()]).numpy(), log_prob.item()

    def store_transition(self, obs, action, log_prob, reward, next_obs, done):
        self.memory.append((obs, action, log_prob, reward, next_obs, done))

    def update(self):
        if len(self.memory) < self.config["batch_size"]:
            return 0.0
            
        obs, actions, old_logp, rewards, next_obs, dones = zip(*self.memory)
        obs_t = torch.FloatTensor(np.array(obs)).to(self.device)
        act_t = torch.LongTensor(np.array(actions)).to(self.device)
        old_logp_t = torch.FloatTensor(old_logp).to(self.device)
        rew_t = torch.FloatTensor(rewards).to(self.device)
        next_obs_t = torch.FloatTensor(np.array(next_obs)).to(self.device)
        done_t = torch.FloatTensor(dones).to(self.device)
        
        with torch.no_grad():
            _, _, _, values = self.ac(obs_t)
            _, _, _, next_values = self.ac(next_obs_t)
            deltas = rew_t + self.config["gamma"] * next_values.squeeze() * (1 - done_t) - values.squeeze()
            advantages = torch.zeros_like(rew_t)
            adv = 0
            for t in reversed(range(len(rew_t))):
                adv = deltas[t] + self.config["gamma"] * self.config["gae_lambda"] * adv * (1 - done_t[t])
                advantages[t] = adv
            returns = advantages + values.squeeze()
            
        for _ in range(self.config["ppo_epochs"]):
            idx = np.random.permutation(len(self.memory))
            for start in range(0, len(idx), self.config["batch_size"]):
                b_idx = idx[start:start+self.config["batch_size"]]
                logits_t, logits_d, logits_s, val = self.ac(obs_t[b_idx])
                
                dist_t = torch.distributions.Categorical(logits=logits_t)
                dist_d = torch.distributions.Categorical(logits=logits_d)
                dist_s = torch.distributions.Categorical(logits=logits_s)
                
                new_logp = dist_t.log_prob(act_t[b_idx][:,0]) + dist_d.log_prob(act_t[b_idx][:,1]) + dist_s.log_prob(act_t[b_idx][:,2])
                ratio = torch.exp(new_logp - old_logp_t[b_idx])
                clip_adv = torch.clamp(ratio, 1-self.config["clip_epsilon"], 1+self.config["clip_epsilon"]) * advantages[b_idx]
                loss_actor = -torch.min(ratio * advantages[b_idx], clip_adv).mean()
                loss_critic = nn.MSELoss()(val.squeeze(), returns[b_idx])
                entropy = (dist_t.entropy() + dist_d.entropy() + dist_s.entropy()).mean()
                loss = loss_actor + self.config["vf_coef"] * loss_critic - self.config["entropy_coef"] * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.optimizer.step()
                
        self.memory.clear()
        return loss.item()

class DynaPPOAgent(PPOAgent):
    def __init__(self, obs_dim, config):
        super().__init__(obs_dim, config)
        self.world_model = WorldModel(obs_dim).to(self.device)
        self.wm_optimizer = optim.Adam(self.world_model.parameters(), lr=1e-3)
        self.wm_memory = []
        self.current_mse = 1.0
        self.wm_warmup_steps = 5000
        self.wm_mse_threshold = config.get("wm_mse_threshold", 0.05)

    def _action_to_onehot(self, action):
        action_oh = torch.zeros(37, device=self.device)
        action_oh[action[0]] = 1
        action_oh[30 + action[1]] = 1
        action_oh[33 + action[2]] = 1
        return action_oh

    def store_transition(self, obs, action, log_prob, reward, next_obs, done):
        super().store_transition(obs, action, log_prob, reward, next_obs, done)
        self.wm_memory.append((obs, self._action_to_onehot(torch.tensor(action, device=self.device)).cpu().numpy(), reward, next_obs))

    def update_world_model(self):
        if len(self.wm_memory) < 64:
            return 1.0
        batch = random.sample(self.wm_memory, 64)
        o, a, r, no = zip(*batch)
        o_t = torch.FloatTensor(np.array(o)).to(self.device)
        a_t = torch.FloatTensor(np.array(a)).to(self.device)
        r_t = torch.FloatTensor(r).to(self.device)
        no_t = torch.FloatTensor(np.array(no)).to(self.device)
        
        pred = self.world_model(o_t, a_t)
        loss = nn.MSELoss()(pred[:, :-1], no_t) + nn.MSELoss()(pred[:, -1], r_t)
        
        self.wm_optimizer.zero_grad()
        loss.backward()
        self.wm_optimizer.step()
        self.current_mse = loss.item()
        return self.current_mse

    def can_use_dyna(self, current_total_steps):
        return current_total_steps >= self.wm_warmup_steps and self.current_mse <= self.wm_mse_threshold

    def get_dyna_rollouts(self, obs):
        rollouts = []
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            for _ in range(self.config["dyna_rollouts"]):
                traj = []
                current_obs = obs_t
                for _ in range(self.config["dyna_horizon"]):
                    logits_t, logits_d, logits_s, _ = self.ac(current_obs)
                    
                    dist_t = torch.distributions.Categorical(logits=logits_t)
                    dist_d = torch.distributions.Categorical(logits=logits_d)
                    dist_s = torch.distributions.Categorical(logits=logits_s)
                    
                    a_t = dist_t.sample()
                    a_d = dist_d.sample()
                    a_s = dist_s.sample()
                    action = torch.tensor([a_t.item(), a_d.item(), a_s.item()], device=self.device)
                    log_prob = dist_t.log_prob(a_t) + dist_d.log_prob(a_d) + dist_s.log_prob(a_s)
                    
                    pred = self.world_model(current_obs, self._action_to_onehot(action).unsqueeze(0))
                    traj.append((current_obs.squeeze(0).cpu().numpy(), action.cpu().numpy(), log_prob.item(), pred[:, -1].item(), pred[:, :-1].squeeze(0).cpu().numpy()))
                    current_obs = pred[:, :-1]
                rollouts.append(traj)
        return rollouts
