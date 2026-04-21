"""Mission Manager — task queue, scheduling, and battery management.

Grade-dependent behavior:
- P-grade: optimal priority + nearest-neighbor routing + perfect battery budget
- C-grade: random priority sometimes, suboptimal routing, occasional battery waste
- W-grade: random order, ignores battery, sends drone to completed deliveries

The grade_quality (0-1) parameter controls how well the planner performs.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum


class Priority(Enum):
    NORMAL = 1
    URGENT = 2
    CRITICAL = 3


@dataclass
class DeliveryRequest:
    delivery_id: int
    target: np.ndarray
    priority: Priority = Priority.NORMAL
    weight: float = 1.0
    created_at: float = 0.0
    completed: bool = False
    failed: bool = False


@dataclass
class MissionState:
    pending: List[DeliveryRequest] = field(default_factory=list)
    current: Optional[DeliveryRequest] = None
    completed: List[DeliveryRequest] = field(default_factory=list)
    failed: List[DeliveryRequest] = field(default_factory=list)
    base_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    elapsed_time: float = 0.0
    total_distance: float = 0.0
    _next_id: int = 0


# Grade quality factors (0=worst, 1=best)
_GRADE_QUALITY = {
    "P":  1.00, "S+": 0.97, "S":  0.94, "S-": 0.91,
    "A+": 0.87, "A":  0.83, "A-": 0.79,
    "B+": 0.74, "B":  0.69, "B-": 0.64,
    "C+": 0.58, "C":  0.52, "C-": 0.46,
    "D+": 0.38, "D":  0.30, "D-": 0.22,
    "F+": 0.15, "F":  0.08, "F-": 0.03,
    "W":  0.00,
}


class MissionPlanner:
    """Plans delivery missions with grade-dependent quality."""

    BATTERY_CAPACITY_M = 5000.0    # meters of flight on full battery
    HOVER_DRAIN_PER_S = 0.0002     # battery per second hovering
    # Pre-flight deadline margin. Manager rejects a mission if
    # ETA + margin > deadline. Layer 5 will be allowed to tune this
    # per drone in Phase 2; 15% is the documented default.
    DEADLINE_MARGIN = 0.15

    def __init__(
        self,
        base_position: Optional[np.ndarray] = None,
        grade: str = "P",
        seed: Optional[int] = None,
    ):
        self.base_position = base_position if base_position is not None else np.zeros(3)
        self.grade = grade
        self.quality = _GRADE_QUALITY.get(grade, 0.0)
        self._rng = np.random.default_rng(seed)
        self.state = MissionState(base_position=self.base_position.copy())
        self._battery = 1.0
        self._position = self.base_position.copy()
        # Current braking distance estimate (metres). Physics layer
        # pushes this in via update_braking() each tick so the planner
        # can factor inertia into feasibility. See docs/physics_realism.md.
        self._braking_distance = 0.0

    def set_grade(self, grade: str):
        self.grade = grade
        self.quality = _GRADE_QUALITY.get(grade, 0.0)

    def add_delivery(
        self,
        target: np.ndarray,
        priority: Priority = Priority.NORMAL,
        weight: float = 1.0,
    ) -> DeliveryRequest:
        req = DeliveryRequest(
            delivery_id=self.state._next_id,
            target=target.copy(),
            priority=priority,
            weight=weight,
            created_at=self.state.elapsed_time,
        )
        self.state._next_id += 1
        self.state.pending.append(req)
        return req

    def select_next(self, current_position: np.ndarray) -> Optional[DeliveryRequest]:
        """Select next delivery. Quality determines how optimal the choice is."""
        if not self.state.pending:
            return None
        if self.state.current is not None:
            return self.state.current

        # Battery check — return to base if low
        if self._should_return_to_base(current_position):
            return None

        candidates = list(self.state.pending)

        if self.quality > 0.9:
            # Perfect: sort by priority * 1/distance
            chosen = self._optimal_choice(candidates, current_position)
        elif self.quality > 0.5:
            # Good: sometimes random, usually optimal
            if self._rng.random() < self.quality:
                chosen = self._optimal_choice(candidates, current_position)
            else:
                chosen = self._rng.choice(candidates)
        else:
            # Poor: mostly random
            chosen = self._rng.choice(candidates) if self._rng.random() > self.quality else \
                     self._optimal_choice(candidates, current_position)

        self.state.pending.remove(chosen)
        self.state.current = chosen
        return chosen

    def _optimal_choice(self, candidates, position):
        def score(req):
            dist = float(np.linalg.norm(req.target - position))
            return req.priority.value * 100.0 / max(dist, 1.0)
        return max(candidates, key=score)

    def _should_return_to_base(self, position: np.ndarray) -> bool:
        if self.quality < 0.3:
            return False  # Low grade ignores battery
        dist_to_nearest = min(
            (np.linalg.norm(r.target - position) for r in self.state.pending),
            default=0.0,
        )
        dist_to_base = np.linalg.norm(self.base_position - position)
        needed = (dist_to_nearest + dist_to_base) / self.BATTERY_CAPACITY_M
        # Add quality-dependent noise to battery estimation
        noise = self._rng.uniform(-0.1, 0.1) * (1.0 - self.quality)
        return (self._battery - noise) < needed + 0.1

    def complete_current(self, success: bool = True):
        if self.state.current is None:
            return
        req = self.state.current
        req.completed = success
        req.failed = not success
        if success:
            self.state.completed.append(req)
        else:
            self.state.failed.append(req)
        self.state.current = None

    def update(self, dt: float, position: np.ndarray, battery: float,
               braking_distance: float = 0.0):
        self.state.elapsed_time += dt
        dist = float(np.linalg.norm(position - self._position))
        self.state.total_distance += dist
        self._position = position.copy()
        self._battery = battery
        self._braking_distance = float(braking_distance)

    # ------------------------------------------------------------------
    # Pre-flight feasibility. Prevention principle: we do not resolve
    # conflicts at runtime — we reject at the gate.
    # See docs/prevention.md, docs/deadlines.md.
    # ------------------------------------------------------------------

    def estimate_eta(self, target: np.ndarray, cruise_speed: float = 12.0) -> float:
        """Straight-line ETA from base to target at a typical cruise.

        Cruise of 12 m/s is conservative for a 5" FPV. Layer 5 can tune
        this per drone in Phase 2 based on observed flight data.
        """
        dist = float(np.linalg.norm(target - self.base_position))
        return dist / max(cruise_speed, 1e-3)

    def feasible(
        self,
        target: np.ndarray,
        deadline_s: Optional[float] = None,
        cruise_speed: float = 12.0,
    ) -> bool:
        """Pre-flight gate. True if the mission fits within deadline + margin
        and battery budget (round-trip from base)."""
        # Battery: round trip must fit with 10% reserve.
        round_trip = 2.0 * float(np.linalg.norm(target - self.base_position))
        if round_trip / self.BATTERY_CAPACITY_M > (self._battery - 0.10):
            return False
        if deadline_s is not None:
            eta = self.estimate_eta(target, cruise_speed=cruise_speed)
            if eta * (1.0 + self.DEADLINE_MARGIN) > deadline_s:
                return False
        return True

    def reset(self):
        self.state = MissionState(base_position=self.base_position.copy())
        self._battery = 1.0
        self._position = self.base_position.copy()

    def is_complete(self) -> bool:
        return len(self.state.pending) == 0 and self.state.current is None

    def get_summary(self) -> Dict:
        total = len(self.state.completed) + len(self.state.failed) + len(self.state.pending)
        if self.state.current:
            total += 1
        return {
            "completed": len(self.state.completed),
            "failed": len(self.state.failed),
            "pending": len(self.state.pending),
            "total": total,
            "completion_rate": len(self.state.completed) / max(total, 1),
            "total_distance_m": self.state.total_distance,
            "elapsed_s": self.state.elapsed_time,
        }
