"""PPO Agent for FlyControl.

Uses Actor-Critic with GAE advantage estimation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import os


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_steps: int = 512
    batch_size: int = 64
    n_epochs: int = 10
    hidden_size: int = 256


class ActorCritic(nn.Module):
    # Target hover throttle for a 0.6 kg quad with 28 N max thrust:
    #   T/W ≈ 4.75, so per-motor hover fraction ≈ 0.21.
    # Bias the final actor layer so zero-weighted output lands at hover,
    # giving PPO a sane starting distribution — otherwise the drone
    # flails and crashes before collecting useful experience.
    HOVER_THROTTLE = 0.21  # sigmoid(-1.33) ≈ 0.21

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(),
            nn.Linear(hidden // 2, act_dim), nn.Sigmoid(),
        )
        # Hover-bias the final linear layer: small weights + logit(hover) bias.
        final = self.actor_mean[-2]  # the Linear before Sigmoid
        nn.init.orthogonal_(final.weight, gain=0.01)
        logit_hover = float(np.log(self.HOVER_THROTTLE / (1.0 - self.HOVER_THROTTLE)))
        nn.init.constant_(final.bias, logit_hover)
        # Lower initial exploration std — -0.5 gives σ≈0.61 (huge on [0,1]),
        # -1.0 gives σ≈0.37 which is still plenty of exploration without
        # making every motor command random noise in the first episodes.
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim) - 1.0)
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        h = self.shared(obs)
        mean = self.actor_mean(h)
        std = self.actor_log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        value = self.critic(h).squeeze(-1)
        return dist, value

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        dist, value = self(obs)
        action = dist.mean if deterministic else dist.sample()
        action = torch.clamp(action, 0.0, 1.0)
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value


class RolloutBuffer:
    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.dones: List[bool] = []

    def add(self, obs, action, reward, value, log_prob, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)

    def compute_returns(self, last_value: float, gamma: float, gae_lambda: float):
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            next_v = last_value if t == n - 1 else self.values[t + 1]
            next_done = 0.0 if t == n - 1 else float(self.dones[t + 1])
            delta = self.rewards[t] + gamma * next_v * (1 - next_done) - self.values[t]
            last_gae = delta + gamma * gae_lambda * (1 - next_done) * last_gae
            advantages[t] = last_gae
        returns = advantages + np.array(self.values, dtype=np.float32)
        return advantages, returns


class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int = 4,
        config: Optional[PPOConfig] = None,
        device: str = "auto",
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.config = config or PPOConfig()

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.policy = ActorCritic(obs_dim, act_dim, self.config.hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.lr)
        self.buffer = RolloutBuffer()
        self.total_steps = 0
        self._last_obs: Optional[np.ndarray] = None

    def select_action(self, obs: np.ndarray, deterministic: bool = False):
        t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.policy.act(t, deterministic)
        return (
            action.squeeze(0).cpu().numpy(),
            {"log_prob": log_prob.item(), "value": value.item()},
        )

    def store(self, obs, action, reward, value, log_prob, done):
        self.buffer.add(obs, action, reward, value, log_prob, done)
        self.total_steps += 1

    def update(self, last_obs: np.ndarray) -> Dict[str, float]:
        if len(self.buffer) < 2:
            return {}

        t = torch.FloatTensor(last_obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, last_val = self.policy.act(t)
        last_value = last_val.item()

        c = self.config
        advantages, returns = self.buffer.compute_returns(last_value, c.gamma, c.gae_lambda)

        obs_t    = torch.FloatTensor(np.array(self.buffer.obs)).to(self.device)
        acts_t   = torch.FloatTensor(np.array(self.buffer.actions)).to(self.device)
        lps_t    = torch.FloatTensor(np.array(self.buffer.log_probs)).to(self.device)
        advs_t   = torch.FloatTensor(advantages).to(self.device)
        rets_t   = torch.FloatTensor(returns).to(self.device)
        advs_t   = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)

        losses = []
        n = len(self.buffer)
        for _ in range(c.n_epochs):
            idx = torch.randperm(n)
            for start in range(0, n, c.batch_size):
                b = idx[start:start + c.batch_size]
                dist, values = self.policy(obs_t[b])
                new_lp = dist.log_prob(acts_t[b]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                ratio = (new_lp - lps_t[b]).exp()
                surr1 = ratio * advs_t[b]
                surr2 = ratio.clamp(1 - c.clip_eps, 1 + c.clip_eps) * advs_t[b]
                actor_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, rets_t[b])
                loss = actor_loss + c.value_coef * value_loss - c.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), c.max_grad_norm)
                self.optimizer.step()
                losses.append(loss.item())

        self.buffer.clear()
        return {"loss": float(np.mean(losses))}

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "total_steps": self.total_steps,
            "config": self.config,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt.get("total_steps", 0)

    @classmethod
    def from_file(cls, path: str, device: str = "auto") -> "PPOAgent":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        agent = cls(
            obs_dim=ckpt["obs_dim"],
            act_dim=ckpt["act_dim"],
            config=ckpt.get("config"),
            device=device,
        )
        agent.policy.load_state_dict(ckpt["policy"])
        agent.total_steps = ckpt.get("total_steps", 0)
        return agent

    def clone(self) -> "PPOAgent":
        import copy
        new = PPOAgent(self.obs_dim, self.act_dim, self.config, str(self.device))
        new.policy.load_state_dict(copy.deepcopy(self.policy.state_dict()))
        new.total_steps = self.total_steps
        return new

    def mutate(self, noise_std: float = 0.05, prob: float = 0.1) -> "PPOAgent":
        child = self.clone()
        with torch.no_grad():
            for p in child.policy.parameters():
                mask = torch.rand_like(p) < prob
                p.add_(mask.float() * torch.randn_like(p) * noise_std)
        return child
