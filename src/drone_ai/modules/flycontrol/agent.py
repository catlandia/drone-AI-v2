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
        # Exploration std. A 5" FPV's motor commands are extremely
        # leverage-sensitive — a ±0.3 jitter per motor (σ≈0.37,
        # log_std=-1.0) inverts the drone within a second and makes
        # every stochastic rollout a crash. log_std=-1.8 gives
        # σ≈0.17, which lets PPO perturb the hover-biased actor
        # enough to learn but not so much the drone flips on init.
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim) - 1.8)
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

    def bc_warmup(
        self,
        observations: np.ndarray,
        expert_actions: np.ndarray,
        n_epochs: int = 60,
        batch_size: int = 128,
        lr: float = 1e-3,
        progress_cb=None,
        post_log_std: Optional[float] = -2.2,
        rewards: Optional[np.ndarray] = None,
        dones: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Supervised pre-training of the actor on (obs, expert_action)
        pairs — Behavior Cloning from a PD controller — and, when
        `rewards` + `dones` are provided, the critic on Monte Carlo
        returns from the same rollouts.

        PLAN.md §8 / §18: fresh PPO tumbles the drone in a handful of
        steps. BC warm-up hands the actor a decent starting policy so
        PPO has something to refine instead of something to survive.

        Critic warm-up matters in shallow per-drone budgets (population
        mode): without it the value function is random for the first
        ~10 PPO updates, advantage estimates are noise, and the policy
        drifts before the critic catches up. Training V(s) → Σ γ^k r_k
        on the same PD rollouts gives PPO a calibrated baseline from
        update 1.

        After BC, `log_std` is clamped down to `post_log_std`
        (default −2.2, so σ ≈ 0.11) so the stochastic policy doesn't
        immediately destroy the learned mean with aggressive noise.
        PPO's own updates to log_std will relax it back up during
        real exploration if the advantages support it.

        `progress_cb(epoch, avg_loss)` is called once per epoch so UIs
        can show a progress bar.
        """
        import torch.nn.functional as F

        if len(observations) == 0:
            return {"loss": float("nan"), "epochs": 0, "samples": 0}

        obs_t = torch.from_numpy(observations).float().to(self.device)
        act_t = torch.from_numpy(expert_actions).float().to(self.device)
        n = obs_t.shape[0]

        # Train shared + actor_mean only. Everything else frozen.
        actor_params = list(self.policy.shared.parameters()) + \
                       list(self.policy.actor_mean.parameters())
        opt = optim.Adam(actor_params, lr=lr)

        last_loss = float("nan")
        for epoch in range(n_epochs):
            idx = torch.randperm(n, device=self.device)
            losses = []
            for start in range(0, n, batch_size):
                b = idx[start:start + batch_size]
                h = self.policy.shared(obs_t[b])
                pred = self.policy.actor_mean(h)
                loss = F.mse_loss(pred, act_t[b])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor_params, self.config.max_grad_norm)
                opt.step()
                losses.append(loss.item())
            last_loss = float(np.mean(losses)) if losses else float("nan")
            if progress_cb is not None:
                try:
                    progress_cb(epoch + 1, last_loss, phase="actor")
                except TypeError:
                    # Older 2-arg cb signatures still work.
                    progress_cb(epoch + 1, last_loss)
                except Exception:
                    pass

        # --- Critic warm-up ----------------------------------------------
        # Compute Monte Carlo discounted returns from the PD rollout's
        # reward stream. The returns are reset at episode boundaries
        # (where dones == True) so cross-episode bootstrapping doesn't
        # leak into the targets.
        critic_loss = float("nan")
        if rewards is not None and dones is not None and len(rewards) == n:
            rew_np = np.asarray(rewards, dtype=np.float32)
            done_np = np.asarray(dones, dtype=np.bool_)
            returns = np.zeros(n, dtype=np.float32)
            running = 0.0
            for t in range(n - 1, -1, -1):
                if done_np[t]:
                    running = 0.0
                running = float(rew_np[t]) + self.config.gamma * running
                returns[t] = running
            ret_t = torch.from_numpy(returns).float().to(self.device)

            critic_params = list(self.policy.shared.parameters()) + \
                            list(self.policy.critic.parameters())
            crit_opt = optim.Adam(critic_params, lr=lr)
            # Halve the critic epochs vs the actor's: the value function
            # converges fast on a small PD-rollout dataset and a longer
            # critic phase mostly added UI dead-time without measurable
            # benefit on shallow per-drone budgets.
            n_critic_epochs = max(1, n_epochs // 2)
            for epoch in range(n_critic_epochs):
                idx = torch.randperm(n, device=self.device)
                losses = []
                for start in range(0, n, batch_size):
                    b = idx[start:start + batch_size]
                    h = self.policy.shared(obs_t[b])
                    v = self.policy.critic(h).squeeze(-1)
                    loss = F.mse_loss(v, ret_t[b])
                    crit_opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        critic_params, self.config.max_grad_norm
                    )
                    crit_opt.step()
                    losses.append(loss.item())
                critic_loss = float(np.mean(losses)) if losses else float("nan")
                if progress_cb is not None:
                    try:
                        progress_cb(epoch + 1, critic_loss, phase="critic")
                    except TypeError:
                        progress_cb(epoch + 1, critic_loss)
                    except Exception:
                        pass

        # Tighten exploration noise so the BC-learned policy isn't
        # drowned by sigma on the first stochastic rollouts.
        if post_log_std is not None:
            with torch.no_grad():
                self.policy.actor_log_std.fill_(float(post_log_std))

        return {
            "loss": last_loss,
            "critic_loss": critic_loss,
            "epochs": n_epochs,
            "samples": int(n),
        }

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
