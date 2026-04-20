"""Perception AI — obstacle detection from drone's simulated sensor data.

In simulation, the drone "sees" obstacles within range but with accuracy
and noise determined by the model's grade. This allows grade-mixing experiments:
a C-grade perception will miss ~35% of obstacles and have high position error.

For real hardware: replace simulate_detections() with CNN inference.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from drone_ai.simulation.world import World, Obstacle


@dataclass
class Detection:
    position: np.ndarray           # Estimated world position
    size: np.ndarray               # Estimated size
    confidence: float              # 0-1
    track_id: int = -1
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))


# Accuracy parameters per grade — (detection_prob, pos_noise_m, false_pos_rate)
_GRADE_PARAMS = {
    "P":  (0.97, 0.3,  0.01),
    "S+": (0.95, 0.4,  0.02),
    "S":  (0.92, 0.5,  0.03),
    "S-": (0.90, 0.6,  0.04),
    "A+": (0.87, 0.8,  0.05),
    "A":  (0.84, 1.0,  0.06),
    "A-": (0.82, 1.2,  0.07),
    "B+": (0.79, 1.4,  0.08),
    "B":  (0.76, 1.6,  0.09),
    "B-": (0.73, 1.8,  0.10),
    "C+": (0.70, 2.2,  0.12),
    "C":  (0.65, 3.0,  0.14),
    "C-": (0.60, 3.5,  0.16),
    "D+": (0.55, 4.0,  0.18),
    "D":  (0.50, 4.5,  0.20),
    "D-": (0.45, 5.0,  0.22),
    "F+": (0.38, 6.0,  0.25),
    "F":  (0.30, 7.0,  0.28),
    "F-": (0.22, 8.0,  0.30),
    "W":  (0.05, 15.0, 0.50),
}


class PerceptionAI:
    """Simulates obstacle perception with grade-dependent accuracy."""

    def __init__(
        self,
        detection_range: float = 80.0,
        grade: str = "P",
        seed: Optional[int] = None,
    ):
        self.detection_range = detection_range
        self.grade = grade
        self._rng = np.random.default_rng(seed)
        self._set_params(grade)

    def _set_params(self, grade: str):
        params = _GRADE_PARAMS.get(grade, _GRADE_PARAMS["W"])
        self.detection_prob = params[0]
        self.pos_noise_m = params[1]
        self.false_pos_rate = params[2]

    def set_grade(self, grade: str):
        self.grade = grade
        self._set_params(grade)

    def detect(
        self,
        drone_position: np.ndarray,
        world: World,
    ) -> List[Detection]:
        """Return list of obstacle detections (with grade-dependent noise)."""
        detections: List[Detection] = []

        nearby = world.obstacles_in_radius(drone_position, self.detection_range)
        for obs in nearby:
            if self._rng.random() > self.detection_prob:
                continue  # missed detection

            noise = self._rng.normal(0, self.pos_noise_m, size=3)
            est_pos = obs.position + noise
            confidence = max(0.0, self.detection_prob - abs(float(np.linalg.norm(noise))) * 0.05)

            detections.append(Detection(
                position=est_pos,
                size=obs.size * self._rng.uniform(0.9, 1.1, size=3),
                confidence=float(confidence),
            ))

        # False positives
        n_false = self._rng.poisson(self.false_pos_rate * len(nearby))
        for _ in range(n_false):
            fp_pos = drone_position + self._rng.normal(0, 20.0, size=3)
            fp_pos[2] = abs(fp_pos[2])
            detections.append(Detection(
                position=fp_pos,
                size=self._rng.uniform(1.0, 5.0, size=3),
                confidence=float(self._rng.uniform(0.1, 0.4)),
            ))

        return detections

    def detections_to_obstacles(self, detections: List[Detection]) -> List[Obstacle]:
        return [
            Obstacle(position=d.position.copy(), size=d.size.copy())
            for d in detections if d.confidence > 0.2
        ]

    def get_nearest_distance(self, detections: List[Detection], drone_pos: np.ndarray) -> float:
        if not detections:
            return float("inf")
        return min(float(np.linalg.norm(d.position - drone_pos)) for d in detections)
