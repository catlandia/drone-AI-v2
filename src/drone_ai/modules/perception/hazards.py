"""Perception-Hazards — specific hazard classes.

Classes:
  - people
  - animals
  - vehicles
  - power_lines
  - water
  - other_danger

Phase 1 stub: the World doesn't yet carry hazard-class labels, so for
now detect() returns an empty list. The interface is stable so that
when the hazard-aware simulator and CNN land, only the body changes.

Grading: graded independently under the same P→W system as every
other perception sub-model. Crucially, under LIFE_CRITICAL missions,
harm-to-bystanders stays hard — see docs/mission_classes.md. The
Hazards model is what enforces that bystander gate in practice.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from drone_ai.modules.perception.detector import Detection
from drone_ai.simulation.world import World


HAZARD_CLASSES = (
    "person",
    "animal",
    "vehicle",
    "power_line",
    "water",
    "other_danger",
)


class PerceptionHazards:
    """Classifies specific hazards in view. Stub until Phase 1.5 CNN."""

    KIND = "hazards"

    def __init__(
        self,
        detection_range: float = 80.0,
        grade: str = "P",
        seed: Optional[int] = None,
    ):
        self.detection_range = float(detection_range)
        self.grade = grade
        self._rng = np.random.default_rng(seed)

    def set_grade(self, grade: str) -> None:
        self.grade = grade

    def detect(self, drone_position: np.ndarray, world: World) -> List[Detection]:
        # Phase 1 stub. World has no hazard-class annotations yet.
        return []
