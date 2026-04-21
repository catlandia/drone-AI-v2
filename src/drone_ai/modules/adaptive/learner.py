"""AdaptiveLearner — online fine-tuner for a FlyControl PPO policy.

This is the 5th module in the stack. It sits on top of a trained
FlyControl agent and keeps it improving from the drone's own in-field
experience. Fully offline: all updates happen on whatever CPU/GPU the
drone carries. No remote training server, no satellite connection.

Deployment:
- `AdaptiveLearner(agent, enabled=False)` — observe only, no grad updates.
  Equivalent to deploying WITHOUT adaptive.
- `AdaptiveLearner(agent, enabled=True)`  — collects rollouts during
  flight and runs small PPO updates when the buffer is full. This is the
  "learn in active" mode the user described.

The module writes *into* the wrapped agent's parameters — so saving the
agent after a mission captures what the drone learned in the field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from drone_ai.modules.flycontrol.agent import PPOAgent, PPOConfig


@dataclass
class AdaptiveConfig:
    """Conservative online-learning hyperparameters.

    The defaults are deliberately gentler than offline PPO — the field
    buffer is small and we can't afford to crater a working policy.
    """
    enabled: bool = True
    # How often (in rollout steps) to attempt a fine-tune update.
    steps_per_update: int = 256
    # How many PPO epochs per update — smaller than offline PPO.
    n_epochs: int = 3
    # Finetune LR is smaller than the original PPO LR (3e-4) so online
    # updates don't overwrite what was already learned well.
    lr_scale: float = 0.25
    # Ratio-clip kept tight for safety.
    clip_eps: float = 0.15
    # Cap the number of in-flight updates so a very long mission doesn't
    # drift the policy arbitrarily far. 0 = unlimited.
    max_updates: int = 0


@dataclass
class AdaptiveMetrics:
    """Graded: how much did online adaptation help?"""
    baseline_score: float            # avg episode reward WITHOUT adaptation
    adapted_score: float             # avg episode reward WITH adaptation
    recovery_rate: float             # (adapted - baseline) / max(|baseline|, 1)
    updates_performed: int


class AdaptiveLearner:
    """Wraps a PPOAgent, records rollouts in-flight, runs mini updates."""

    def __init__(self, agent: PPOAgent, config: Optional[AdaptiveConfig] = None):
        self.agent = agent
        self.config = config or AdaptiveConfig()
        self._steps_since_update = 0
        self._updates_performed = 0
        self._last_obs: Optional[np.ndarray] = None
        # Rescale the wrapped agent's optimizer LR for the duration of
        # adaptive operation. Keep the original so we can restore it.
        self._original_lr = self.agent.config.lr
        if self.config.enabled:
            for g in self.agent.optimizer.param_groups:
                g["lr"] = self._original_lr * self.config.lr_scale
            self.agent.config.clip_eps = self.config.clip_eps

    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, deterministic: bool = False):
        """Drop-in replacement for PPOAgent.select_action.

        When enabled, also stashes the obs so we know what 'last_obs'
        to use when closing the rollout.
        """
        action, info = self.agent.select_action(obs, deterministic=deterministic)
        self._last_obs = obs
        return action, info

    def observe(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        info: dict,
        done: bool,
    ) -> None:
        """Called once per environment step after the action is applied."""
        if not self.config.enabled:
            return
        if self._at_update_limit():
            return
        self.agent.store(obs, action, float(reward), info["value"], info["log_prob"], done)
        self._steps_since_update += 1
        if self._steps_since_update >= self.config.steps_per_update:
            self._finetune()

    def end_episode(self, final_obs: np.ndarray) -> None:
        """Optional hint — lets the learner flush a partial rollout at
        the end of an episode if it's collected enough data to be worth
        updating on."""
        if not self.config.enabled:
            return
        if self._steps_since_update >= self.config.steps_per_update // 2:
            self._last_obs = final_obs
            self._finetune()

    # ------------------------------------------------------------------

    def _at_update_limit(self) -> bool:
        return (self.config.max_updates > 0
                and self._updates_performed >= self.config.max_updates)

    def _finetune(self) -> None:
        if self._last_obs is None:
            return
        # Temporarily override n_epochs to the gentler online value.
        original_n_epochs = self.agent.config.n_epochs
        self.agent.config.n_epochs = self.config.n_epochs
        try:
            self.agent.update(self._last_obs)
        except Exception as e:
            print(f"[adaptive] finetune skipped: {e}")
        finally:
            self.agent.config.n_epochs = original_n_epochs
        self._steps_since_update = 0
        self._updates_performed += 1

    # ------------------------------------------------------------------

    @property
    def updates_performed(self) -> int:
        return self._updates_performed

    def disable(self) -> None:
        """Freeze the learner — restore original LR and stop collecting."""
        self.config.enabled = False
        for g in self.agent.optimizer.param_groups:
            g["lr"] = self._original_lr
