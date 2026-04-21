"""Perception-Agents — other drones, for Layer 8 visual mutual avoidance.

Phase 1 stub. Phase 2 consumer is the Swarm Cooperation layer
(see docs/modules/swarm.md). The Agents model classifies a visual
contact as drone-vs-not-drone, estimates range + closing speed, and
distinguishes swarm-mate vs unknown drone (via QR / visual ID).

Grades independently under P→W.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from drone_ai.modules.perception.detector import Detection
from drone_ai.simulation.world import World


class PerceptionAgents:
    """Detects other drones in camera view. Stub until Layer 8 lands."""

    KIND = "agents"

    def __init__(
        self,
        detection_range: float = 100.0,
        grade: str = "P",
        seed: Optional[int] = None,
    ):
        self.detection_range = float(detection_range)
        self.grade = grade
        self._rng = np.random.default_rng(seed)

    def set_grade(self, grade: str) -> None:
        self.grade = grade

    def detect(self, drone_position: np.ndarray, world: World) -> List[Detection]:
        # Phase 1 stub; multi-drone simulation not yet implemented.
        return []
