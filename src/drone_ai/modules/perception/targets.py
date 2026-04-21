"""Perception-Targets — landing zones, package drop markers, recipient markers.

Phase 1 stub. The real Targets model is what the drone uses to:
  - Re-acquire the base landing pad on return (visual pattern + engraved QR).
  - Identify package drop markers at delivery sites.
  - Find an alternate landing zone when the base is unreachable
    (see docs/comms.md — can't-land-at-base fallback ladder).

Grades independently under P→W.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from drone_ai.modules.perception.detector import Detection
from drone_ai.simulation.world import World


TARGET_CLASSES = (
    "base_pad",
    "drop_marker",
    "recipient_marker",
    "alt_landing_zone",
)


class PerceptionTargets:
    """Finds mission-relevant targets. Stub until CNN integration."""

    KIND = "targets"

    def __init__(
        self,
        detection_range: float = 50.0,
        grade: str = "P",
        seed: Optional[int] = None,
    ):
        self.detection_range = float(detection_range)
        self.grade = grade
        self._rng = np.random.default_rng(seed)

    def set_grade(self, grade: str) -> None:
        self.grade = grade

    def detect(self, drone_position: np.ndarray, world: World) -> List[Detection]:
        # Phase 1 stub; awaits target-annotated world + CNN.
        return []
