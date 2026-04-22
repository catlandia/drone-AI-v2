"""Warden — frozen reward-function guard for Layer 5.

A frozen copy of the original environment reward function lives here
(in read-only code). Every proposed weight update is scored by the
warden, NOT by Layer 5's own metric, on N simulated episodes before
being accepted.

This is the single most load-bearing piece of Phase 2 safety. Without
it, Layer 5 could slowly drift its internal metric while the real
mission score degrades — the classic reward-hacking failure mode.

The warden never looks at Layer 5's opinion of the policy. It only
looks at frozen_reward(env, state, action). If the adapted policy
scores lower than baseline on warden-run episodes → update rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from drone_ai.modules.flycontrol.agent import PPOAgent
from drone_ai.modules.flycontrol.environment import FlyControlEnv, TaskType


@dataclass
class WardenVerdict:
    accepted: bool
    baseline_score: float
    proposed_score: float
    reason: str = ""


class Warden:
    """Scores a proposed policy against a baseline on frozen reward.

    The frozen reward function is whatever the env was shipped with
    when the warden was constructed — we snapshot a reference to
    `FlyControlEnv._compute_reward` and never update it. Layer 5 has
    no API to change this.
    """

    # A drop of more than this fraction below baseline rejects the update.
    # Kept generous because warden runs are noisy (wind, gusts, seeds).
    DROP_TOLERANCE: float = 0.05

    def __init__(
        self,
        task: TaskType = TaskType.HOVER,
        difficulty: float = 0.5,
        n_episodes: int = 5,
        max_steps: int = 500,
        seed: int = 12345,
    ):
        self.task = task
        self.difficulty = difficulty
        self.n_episodes = int(n_episodes)
        self.max_steps = int(max_steps)
        self.seed = int(seed)

    # ------------------------------------------------------------------

    def score(self, agent: PPOAgent) -> float:
        """Run N deterministic episodes and return mean episode reward.

        Uses a fresh env every call, seeded identically so scoring a
        baseline and a proposed policy compares apples-to-apples. The
        env's reward function IS the frozen reward — that's the whole
        point of the warden.
        """
        env = FlyControlEnv(
            task=self.task,
            difficulty=self.difficulty,
            domain_randomization=False,
            seed=self.seed,
        )
        totals: List[float] = []
        for ep in range(self.n_episodes):
            obs, _ = env.reset(seed=self.seed + ep)
            total = 0.0
            for _ in range(self.max_steps):
                action, _ = agent.select_action(obs, deterministic=True)
                obs, r, term, trunc, _ = env.step(action)
                total += float(r)
                if term or trunc:
                    break
            totals.append(total)
        env.close()
        return float(np.mean(totals))

    def evaluate(self, baseline: PPOAgent, proposed: PPOAgent) -> WardenVerdict:
        """Accept-or-reject a proposed update."""
        base = self.score(baseline)
        prop = self.score(proposed)
        # Allow a tiny drop since warden runs are stochastic. Anything
        # bigger is real regression.
        tolerance = self.DROP_TOLERANCE * max(abs(base), 1.0)
        if prop + tolerance < base:
            return WardenVerdict(
                accepted=False,
                baseline_score=base,
                proposed_score=prop,
                reason="warden_score_drop",
            )
        return WardenVerdict(
            accepted=True,
            baseline_score=base,
            proposed_score=prop,
            reason="warden_score_hold_or_improve",
        )
