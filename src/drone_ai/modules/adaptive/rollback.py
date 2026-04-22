"""Rolling-window rollback monitor for Layer 5.

Rule (PLAN.md §15):

    Rolling average of the LAST 20 episodes. If it drops below the
    previous best, roll back to the pre-update checkpoint.
    Single-episode failures do not trigger rollback (too sensitive
    to wind / noise).

This module tracks the rolling window and tells the caller when to
roll back. The caller holds the pre-update checkpoint blob and is
responsible for actually restoring it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional


@dataclass
class RollbackDecision:
    rollback: bool
    current_avg: float
    previous_best: float
    reason: str


class RollbackMonitor:
    """Tracks the last `window` episode rewards; flags regressions.

    `previous_best` is updated ONLY when the agent improves — a bad
    run can lower the rolling average but does not lower the bar the
    agent is measured against.
    """

    def __init__(self, window: int = 20):
        self.window = int(window)
        self._recent: Deque[float] = deque(maxlen=self.window)
        self._previous_best: Optional[float] = None

    # ------------------------------------------------------------------

    def record_episode(self, total_reward: float) -> None:
        self._recent.append(float(total_reward))

    def avg(self) -> Optional[float]:
        if not self._recent:
            return None
        return sum(self._recent) / len(self._recent)

    def checkpoint(self) -> None:
        """Mark the current rolling average as the new best to beat.

        Called by Layer 5 right before it accepts a warden-approved
        update — the pre-update performance becomes the floor the
        post-update policy must hold.
        """
        cur = self.avg()
        if cur is None:
            return
        if self._previous_best is None or cur > self._previous_best:
            self._previous_best = cur

    def should_rollback(self) -> RollbackDecision:
        cur = self.avg()
        prev = self._previous_best
        if cur is None or prev is None:
            return RollbackDecision(False, cur or 0.0, prev or 0.0, "insufficient_data")
        # Require a full window of samples before deciding. Avoids
        # premature rollbacks after a single bad post-update episode.
        if len(self._recent) < self.window:
            return RollbackDecision(False, cur, prev, "window_not_filled")
        if cur < prev:
            return RollbackDecision(True, cur, prev, "rolling_avg_below_best")
        return RollbackDecision(False, cur, prev, "holding")

    # ------------------------------------------------------------------

    @property
    def previous_best(self) -> Optional[float]:
        return self._previous_best

    def reset(self) -> None:
        self._recent.clear()
        self._previous_best = None
