"""Soft-bound registry — gates Layer 5 from pushing past safe limits.

The plan (PLAN.md §15, docs/modules/adaptive.md):

    Softenable (Layer 5 may push past after N successful simulated
    recoveries past the bound with no crash):
      - tilt bound (90° inversion rule)
      - hover throttle assumption
      - action clipping
      - battery "land now" threshold

    Hard, never touchable:
      - ground-impact crash detection
      - reward function structure   (enforced by warden)
      - goals                       (enforced by goal-structure mask)
      - module toggles

This module implements the soft side: we only let Layer 5 relax a
bound after it has produced **N** successful sim recoveries (no
crash) while operating near that bound.

Default N = 50. Tunable at construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


# Canonical names for the soft bounds Layer 5 may push on.
SOFT_BOUNDS = (
    "tilt_bound",
    "hover_throttle",
    "action_clipping",
    "battery_threshold",
)

# Hard limits Layer 5 must NEVER touch. Listed here so the adaptive
# code can explicitly refuse to register recoveries against them.
HARD_LIMITS = (
    "ground_impact",
    "reward_structure",
    "goals",
    "module_toggles",
)


@dataclass
class BoundStatus:
    name: str
    recoveries: int
    required: int
    promoted: bool          # has Layer 5 earned the right to soften it?

    def progress(self) -> float:
        if self.required <= 0:
            return 1.0
        return min(1.0, self.recoveries / self.required)


class SoftBoundRegistry:
    """Tracks recovery counts per soft bound, refuses hard limits."""

    def __init__(self, required: int = 50):
        self.required = int(required)
        self._recoveries: Dict[str, int] = {name: 0 for name in SOFT_BOUNDS}
        self._promoted: Dict[str, bool] = {name: False for name in SOFT_BOUNDS}

    # ------------------------------------------------------------------

    def record_recovery(self, bound: str) -> BoundStatus:
        """Register one successful simulated recovery past a soft bound.

        Raises ValueError if called with a HARD_LIMITS name — we want
        that to be loud, not silent.
        """
        if bound in HARD_LIMITS:
            raise ValueError(
                f"'{bound}' is a hard limit and cannot accumulate recoveries. "
                f"See PLAN.md §15."
            )
        if bound not in SOFT_BOUNDS:
            raise ValueError(
                f"Unknown soft bound '{bound}'. Known: {SOFT_BOUNDS}"
            )
        self._recoveries[bound] += 1
        if self._recoveries[bound] >= self.required:
            self._promoted[bound] = True
        return self.status(bound)

    def can_push(self, bound: str) -> bool:
        if bound in HARD_LIMITS:
            return False
        return bool(self._promoted.get(bound, False))

    def status(self, bound: str) -> BoundStatus:
        return BoundStatus(
            name=bound,
            recoveries=self._recoveries.get(bound, 0),
            required=self.required,
            promoted=self._promoted.get(bound, False),
        )

    def all_statuses(self) -> List[BoundStatus]:
        return [self.status(b) for b in SOFT_BOUNDS]

    def to_dict(self) -> Dict[str, Dict[str, int]]:
        return {
            b: {
                "recoveries": self._recoveries[b],
                "required": self.required,
                "promoted": int(self._promoted[b]),
            }
            for b in SOFT_BOUNDS
        }
