"""Perception-Obstacles — generic obstacle CNN + Kalman tracker.

Phase 1 sub-model. For now this is a thin wrapper around the existing
`PerceptionAI` noise model, preserved so the rest of the stack keeps
working while the architectural split lands. The "real" Obstacles
model will be a CNN once Phase 1.5 hits.

All four perception sub-models share the Detection dataclass and the
grade-parameterized noise so they can be graded under the same
P→W tier system (see docs/tier_system.md).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from drone_ai.modules.perception.detector import PerceptionAI, Detection
from drone_ai.simulation.world import World


class PerceptionObstacles:
    """Detects generic obstacles (boxes, walls, trees) around the drone."""

    KIND = "obstacles"

    def __init__(
        self,
        detection_range: float = 80.0,
        grade: str = "P",
        seed: Optional[int] = None,
    ):
        self._ai = PerceptionAI(
            detection_range=detection_range, grade=grade, seed=seed
        )

    @property
    def grade(self) -> str:
        return self._ai.grade

    def set_grade(self, grade: str) -> None:
        self._ai.set_grade(grade)

    def detect(self, drone_position: np.ndarray, world: World) -> List[Detection]:
        return self._ai.detect(drone_position, world)
