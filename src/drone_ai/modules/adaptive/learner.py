"""AdaptiveLearner — online fine-tuner for a FlyControl PPO policy.

This is the 5th layer. It sits on top of a trained FlyControl agent
and keeps it improving from the drone's own in-field experience.

Guarded by three non-negotiables (PLAN.md §15, docs/modules/adaptive.md):

  1. **Warden.** A frozen copy of the env reward runs N sim episodes
     on both the pre-update and post-update policy. If the proposed
     policy scores worse than baseline → update rejected.
  2. **Rollback.** A rolling window of the last 20 field-episode
     rewards tracks real-world performance. If the rolling average
     drops below the previous best, the learner rolls back to the
     last accepted checkpoint.
  3. **Soft-bound registry.** Layer 5 may NOT push past a soft limit
     (tilt bound, hover throttle, action clipping, battery threshold)
     until it has N ≈ 50 successful simulated recoveries past that
     bound. Hard limits (ground-impact, reward structure, goals,
     module toggles) are never touched.

Deployment:
- `AdaptiveLearner(agent, enabled=False)` — observe only. Equivalent
  to the pre-trained deploy mode.
- `AdaptiveLearner(agent, enabled=True)`  — collects rollouts during
  flight and proposes gentle PPO updates. Every proposed update must
  pass the warden before it is applied to the wrapped agent.

The module writes *into* the wrapped agent's parameters — so saving
the agent after a mission captures what the drone learned in the field.

Update timing (hybrid):
- Mid-flight: FlyControl ONLY. Fast reflex layer — a bad update is
  recoverable within milliseconds.
- Landed + idle: Manager, Pathfinder, Perception, and Layer 5 itself.
  A bad high-level update ruins an entire mission.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from drone_ai.modules.flycontrol.agent import PPOAgent, PPOConfig
from drone_ai.modules.adaptive.warden import Warden, WardenVerdict
from drone_ai.modules.adaptive.rollback import RollbackMonitor, RollbackDecision
from drone_ai.modules.adaptive.soft_bounds import SoftBoundRegistry


@dataclass
class AdaptiveConfig:
    """Conservative online-learning hyperparameters.

    The defaults are deliberately gentler than offline PPO — the field
    buffer is small and we can't afford to crater a working policy.
    """
    enabled: bool = True
    steps_per_update: int = 256
    n_epochs: int = 3
    lr_scale: float = 0.25
    clip_eps: float = 0.15
    max_updates: int = 0            # 0 = unlimited
    # Warden config — tiny by default because it runs inline on a drone.
    warden_episodes: int = 3
    warden_max_steps: int = 300
    # Rollback window.
    rollback_window: int = 20
    # Soft-bound recovery count required before Layer 5 may push past.
    soft_bound_N: int = 50
    # Only accept FlyControl updates mid-flight (PLAN.md §15).
    # Layer 5 on other layers waits for landed+idle. Kept as a flag so
    # integration code can opt in explicitly where appropriate.
    midflight_flycontrol_only: bool = True


@dataclass
class AdaptiveMetrics:
    """How much did online adaptation help?"""
    baseline_score: float
    adapted_score: float
    recovery_rate: float
    updates_performed: int
    updates_rejected: int = 0
    rollbacks: int = 0


class AdaptiveLearner:
    """Wraps a PPOAgent. Records rollouts in-flight, proposes updates,
    runs them through the warden, rolls back if the field performance
    regresses.

    Optional hooks for the rest of the stack:
      - `storage` — Layer 6 Storage; every accept/reject/rollback is
        logged for Layer 7 personality selection.
      - `mission_id` — tagged on every storage row.
      - `layer`      — which layer this learner guards. Defaults to
        "flycontrol", the only layer allowed mid-flight updates.
    """

    def __init__(
        self,
        agent: PPOAgent,
        config: Optional[AdaptiveConfig] = None,
        warden: Optional[Warden] = None,
        storage: Optional[Any] = None,   # typed Any to avoid circular import
        mission_id: str = "",
        layer: str = "flycontrol",
    ):
        self.agent = agent
        self.config = config or AdaptiveConfig()
        self.layer = layer
        self.mission_id = mission_id
        self._storage = storage

        self.warden = warden or Warden(
            n_episodes=self.config.warden_episodes,
            max_steps=self.config.warden_max_steps,
        )
        self.rollback = RollbackMonitor(window=self.config.rollback_window)
        self.soft_bounds = SoftBoundRegistry(required=self.config.soft_bound_N)

        self._steps_since_update = 0
        self._updates_performed = 0
        self._updates_rejected = 0
        self._rollbacks = 0
        self._last_obs: Optional[np.ndarray] = None

        # Pre-update checkpoint blob — state_dict of the policy before
        # each proposed update, so rollback can restore it.
        self._last_accepted_state = copy.deepcopy(self.agent.policy.state_dict())

        # Rescale the wrapped agent's optimizer LR. Keep the original
        # so we can restore it on disable().
        self._original_lr = self.agent.config.lr
        self._original_clip = self.agent.config.clip_eps
        if self.config.enabled:
            for g in self.agent.optimizer.param_groups:
                g["lr"] = self._original_lr * self.config.lr_scale
            self.agent.config.clip_eps = self.config.clip_eps

    # ------------------------------------------------------------------
    # Mid-flight API (FlyControl only)
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, deterministic: bool = False):
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
        """Called once per env step. Stores rollout; triggers a proposed
        update (warden-guarded) when the buffer fills."""
        if not self.config.enabled:
            return
        if self._at_update_limit():
            return
        # PLAN.md §15: mid-flight updates are FlyControl only. If this
        # learner is guarding another layer, observations pile up but
        # actual updates wait for a landed+idle propose_landed_update().
        self.agent.store(obs, action, float(reward), info["value"], info["log_prob"], done)
        self._steps_since_update += 1

        if self.config.midflight_flycontrol_only and self.layer != "flycontrol":
            return

        if self._steps_since_update >= self.config.steps_per_update:
            self._propose_update()

    def end_episode(self, final_obs: np.ndarray, total_reward: float) -> None:
        """Call at episode end. Flushes a partial rollout if it has
        enough data and feeds the rolling-average rollback monitor."""
        if not self.config.enabled:
            return
        self._last_obs = final_obs
        self.rollback.record_episode(total_reward)

        if self._steps_since_update >= self.config.steps_per_update // 2 and (
            not self.config.midflight_flycontrol_only or self.layer == "flycontrol"
        ):
            self._propose_update()

        # Rollback check — only if the rolling average has slid below
        # the previous best.
        decision = self.rollback.should_rollback()
        if decision.rollback:
            self._rollback_to_last_accepted(decision)

    # ------------------------------------------------------------------
    # Landed + idle API (Manager / Pathfinder / Perception / Adaptive)
    # ------------------------------------------------------------------

    def propose_landed_update(self) -> bool:
        """Call when the drone is landed and idle. Safe for any layer.

        Returns True if an update was proposed and accepted.
        """
        if not self.config.enabled:
            return False
        if self._at_update_limit():
            return False
        if self._steps_since_update < max(8, self.config.steps_per_update // 4):
            # Not enough field data to propose anything meaningful.
            return False
        return self._propose_update()

    # ------------------------------------------------------------------
    # Soft-bound gate
    # ------------------------------------------------------------------

    def record_recovery(self, bound: str) -> None:
        """Register one sim recovery past a soft bound. See
        docs/modules/adaptive.md and PLAN.md §15."""
        self.soft_bounds.record_recovery(bound)

    def can_push_soft_bound(self, bound: str) -> bool:
        return self.soft_bounds.can_push(bound)

    # ------------------------------------------------------------------

    def _at_update_limit(self) -> bool:
        return (self.config.max_updates > 0
                and self._updates_performed >= self.config.max_updates)

    def _propose_update(self) -> bool:
        """Run a PPO update on a clone of the agent, score it with the
        warden, and either commit it or discard it.
        """
        if self._last_obs is None:
            return False
        if len(self.agent.buffer) < 2:
            return False

        # Snapshot the current policy so we can restore it if the
        # warden rejects the proposal or rollback fires later.
        pre_update_state = copy.deepcopy(self.agent.policy.state_dict())
        pre_lr_epochs = self.agent.config.n_epochs

        # Baseline agent for the warden: cheap clone that shares config
        # but starts from the pre-update state.
        baseline_agent = self._clone_with_state(pre_update_state)

        # Temporarily switch epochs down for the online path.
        self.agent.config.n_epochs = self.config.n_epochs
        try:
            self.agent.update(self._last_obs)
        except Exception as e:
            # PPO update blew up — restore, skip.
            self.agent.policy.load_state_dict(pre_update_state)
            self.agent.config.n_epochs = pre_lr_epochs
            self._log_update(
                accepted=False,
                reason=f"ppo_update_error:{type(e).__name__}",
                verdict=None,
            )
            self._steps_since_update = 0
            return False
        self.agent.config.n_epochs = pre_lr_epochs

        # Warden scores both policies on frozen reward.
        try:
            verdict = self.warden.evaluate(baseline_agent, self.agent)
        except Exception as e:
            # If the warden itself fails, be conservative: reject.
            self.agent.policy.load_state_dict(pre_update_state)
            self._log_update(
                accepted=False,
                reason=f"warden_error:{type(e).__name__}",
                verdict=None,
            )
            self._steps_since_update = 0
            return False

        if not verdict.accepted:
            # Reject — restore pre-update weights.
            self.agent.policy.load_state_dict(pre_update_state)
            self._updates_rejected += 1
            self._log_update(False, verdict.reason, verdict)
            self._steps_since_update = 0
            return False

        # Accept — the new weights stay in the agent. Mark the rolling
        # average so rollback is measured against the post-accept bar.
        self._last_accepted_state = copy.deepcopy(self.agent.policy.state_dict())
        self.rollback.checkpoint()
        self._updates_performed += 1
        self._log_update(True, verdict.reason, verdict)
        self._steps_since_update = 0
        return True

    def _clone_with_state(self, state_dict) -> PPOAgent:
        """Build a baseline PPOAgent that shares architecture/config
        with the wrapped agent but uses a copy of `state_dict`."""
        clone = PPOAgent(
            obs_dim=self.agent.obs_dim,
            act_dim=self.agent.act_dim,
            config=self.agent.config,
            device=str(self.agent.device),
        )
        clone.policy.load_state_dict(copy.deepcopy(state_dict))
        return clone

    def _rollback_to_last_accepted(self, decision: RollbackDecision) -> None:
        self.agent.policy.load_state_dict(copy.deepcopy(self._last_accepted_state))
        self._rollbacks += 1
        self.rollback.reset()
        self._log_update(
            accepted=False,
            reason=f"rollback:{decision.reason}",
            verdict=None,
            rollback_triggered=True,
        )

    def _log_update(
        self,
        accepted: bool,
        reason: str,
        verdict: Optional[WardenVerdict],
        rollback_triggered: bool = False,
    ) -> None:
        if self._storage is None:
            return
        try:
            # Late import so storage/learner can sit in either dep order.
            from drone_ai.modules.storage import UpdateRecord
            rec = UpdateRecord(
                mission_id=self.mission_id,
                layer=self.layer,
                accepted=accepted,
                rejected_reason=None if accepted else reason,
                warden_score_pre=verdict.baseline_score if verdict else None,
                warden_score_post=verdict.proposed_score if verdict else None,
                rollback_triggered=rollback_triggered,
            )
            self._storage.record_update(rec)
        except Exception as e:
            # Never let Storage writes break the hot path.
            print(f"[adaptive] storage log failed: {e}")

    # ------------------------------------------------------------------

    @property
    def updates_performed(self) -> int:
        return self._updates_performed

    @property
    def updates_rejected(self) -> int:
        return self._updates_rejected

    @property
    def rollbacks(self) -> int:
        return self._rollbacks

    def disable(self) -> None:
        """Freeze the learner — restore original LR/clip and stop collecting."""
        self.config.enabled = False
        for g in self.agent.optimizer.param_groups:
            g["lr"] = self._original_lr
        self.agent.config.clip_eps = self._original_clip
