"""SwarmCoordinator — per-drone in-flight logic for Layer 8.

Zero radio comms after takeoff. The coordinator runs on a single
drone, given:

  - This drone's `SwarmPlan` assignment + the full plan (for peers).
  - A stream of `VisualContact`s from Perception-Agents.
  - Its own current position + mission state.

It returns an `AvoidanceAction` telling the mission loop how to
deviate, or `None` to continue on-route.

Contingency handling:
  - Swarm-mate-failed: nearest surviving drone diverts to mark the
    failure point UNLESS it itself is on LIFE_CRITICAL. Others
    continue their missions unchanged.
  - Can't-land-at-base: ladder lives in docs/comms.md, run
    independently per drone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from drone_ai.modules.swarm.plan import (
    SwarmPlan,
    DroneAssignment,
    ContingencyKind,
)


class AvoidanceKind(str, Enum):
    CONTINUE = "continue"
    YIELD = "yield"                  # brake / descend to pass below
    CLIMB = "climb"                  # go up to pass over
    BANK_RIGHT = "bank_right"        # right-of-way convention
    DIVERT_TO_MARK = "divert_to_mark"  # swarm-mate-failed response


@dataclass
class VisualContact:
    """A detection from Perception-Agents.

    `agent_id` is either a known peer drone_id (if the QR / visual ID
    matched a peer in the plan) or None for an unknown drone.
    """
    agent_id: Optional[str]
    is_peer: bool
    position: np.ndarray                        # estimated VIO-frame position
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    range_m: float = 0.0
    closing_speed: float = 0.0                  # m/s, positive = approaching
    confidence: float = 1.0


@dataclass
class AvoidanceAction:
    kind: AvoidanceKind
    target: Optional[np.ndarray] = None         # e.g. beacon / divert point
    reason: str = ""


class SwarmCoordinator:
    """Single-drone in-flight logic. Never radios anything.

    Stateful across frames: tracks whether this drone is the closest
    peer to a failed mate (and thus owes a divert), and whether it's
    already servicing that divert.
    """

    # Distance (m) below which we must actively avoid a peer.
    AVOID_RADIUS: float = 8.0
    # Relative-bearing threshold for "head-on". Above this we treat as
    # side-pass (no action).
    HEADON_DOT: float = 0.85

    def __init__(self, plan: SwarmPlan, self_drone_id: str):
        self.plan = plan
        self.drone_id = self_drone_id
        self._self: DroneAssignment = plan.assignments[self_drone_id]
        self._servicing_divert_for: Optional[str] = None
        self._divert_target: Optional[np.ndarray] = None
        self._failed_peers: Dict[str, np.ndarray] = {}  # peer_id → last-known pos

    # ------------------------------------------------------------------

    def step(
        self,
        self_position: np.ndarray,
        contacts: List[VisualContact],
    ) -> Optional[AvoidanceAction]:
        """One coordination tick. Returns an action or None.

        Callers should prefer the coordinator's action over their own
        pathfinder when `None` is not returned.
        """
        # 1. Servicing a divert already? Stay on it until we've arrived.
        if self._divert_target is not None:
            if np.linalg.norm(self_position - self._divert_target) < 4.0:
                self._divert_target = None
                self._servicing_divert_for = None
                # Divert finished — resume mission next frame.
            else:
                return AvoidanceAction(
                    kind=AvoidanceKind.DIVERT_TO_MARK,
                    target=self._divert_target,
                    reason="divert_in_progress",
                )

        # 2. Visual mutual avoidance — check closest contact first.
        contact = self._closest_threat(self_position, contacts)
        if contact is not None:
            action = self._avoid(self_position, contact)
            if action is not None:
                return action

        # 3. Nothing to do — continue on the pre-planned route.
        return None

    # ------------------------------------------------------------------

    def mark_peer_failed(self, peer_id: str, last_position: np.ndarray) -> None:
        """Call when Perception or mission state indicates a peer has
        failed (crashed, disappeared, mission-aborted). Decides whether
        THIS drone is the nearest surviving peer and should divert.

        Implements the swarm-mate-failed contingency per
        docs/modules/swarm.md. LIFE_CRITICAL drones do NOT divert —
        their own mission outranks marking a failure point.
        """
        if peer_id not in self.plan.assignments:
            return
        self._failed_peers[peer_id] = last_position.copy()

        if self._self.mission_class == "LIFE_CRITICAL":
            # Exception: a life-critical drone does not divert.
            return

        # Figure out who's nearest among surviving, non-LIFE_CRITICAL peers.
        survivors: List[Tuple[float, str]] = []
        for other_id, other in self.plan.assignments.items():
            if other_id == peer_id:
                continue
            if other_id in self._failed_peers:
                continue
            if other.mission_class == "LIFE_CRITICAL":
                continue
            # We don't know each peer's live position (no radio), but
            # we DO know their route start and can use the assignment
            # route's initial waypoint as a best-effort proxy.
            ref_pos = other.route[0] if other.route else np.zeros(3)
            survivors.append((float(np.linalg.norm(ref_pos - last_position)), other_id))

        if not survivors:
            return
        survivors.sort()
        nearest_id = survivors[0][1]
        if nearest_id == self.drone_id and self._servicing_divert_for is None:
            # Take the divert. Next `step` call returns DIVERT_TO_MARK.
            cont = self.plan.contingency(ContingencyKind.SWARM_MATE_FAILED)
            alt = 10.0
            if cont is not None:
                alt = float(cont.params.get("beacon_altitude", 10.0))
            self._servicing_divert_for = peer_id
            self._divert_target = last_position.copy()
            self._divert_target[2] = alt

    # ------------------------------------------------------------------

    def _closest_threat(
        self,
        self_pos: np.ndarray,
        contacts: List[VisualContact],
    ) -> Optional[VisualContact]:
        threats = [c for c in contacts if c.range_m < self.AVOID_RADIUS * 2.0]
        if not threats:
            return None
        return min(threats, key=lambda c: c.range_m)

    def _avoid(
        self,
        self_pos: np.ndarray,
        contact: VisualContact,
    ) -> Optional[AvoidanceAction]:
        if contact.range_m > self.AVOID_RADIUS:
            # In sight but not inside the action radius — keep watching.
            return None

        # Head-on vs side-pass classification.
        rel = contact.position - self_pos
        rel_norm = float(np.linalg.norm(rel))
        if rel_norm < 1e-6:
            # Co-located contact is either a false positive or we've
            # already lost. Bank right to clear.
            return AvoidanceAction(AvoidanceKind.BANK_RIGHT, reason="co_located")
        rel_dir = rel / rel_norm

        # Closing-speed dot: approx head-on if closing_speed is high
        # AND the contact sits roughly along our heading.
        if contact.closing_speed > 3.0 and abs(rel_dir[2]) < 0.3:
            # Head-on in horizontal plane. Right-of-way: bank right.
            # Also climb slightly if we're the higher-priority drone
            # (heuristic: PRIMARY / LIFE_CRITICAL climbs; others yield).
            assignment = self.plan.assignment_for(self.drone_id)
            higher_priority = (
                assignment is not None
                and (assignment.mission_class == "LIFE_CRITICAL"
                     or str(assignment.role).endswith("primary"))
            )
            if higher_priority:
                return AvoidanceAction(AvoidanceKind.CLIMB, reason="headon_higher_priority")
            return AvoidanceAction(AvoidanceKind.YIELD, reason="headon_yield")

        # Side-pass with small vertical offset — stack vertically.
        if rel_dir[2] > 0.3:
            # Contact above us; descend to pass below.
            return AvoidanceAction(AvoidanceKind.YIELD, reason="vertical_stack_below")
        if rel_dir[2] < -0.3:
            return AvoidanceAction(AvoidanceKind.CLIMB, reason="vertical_stack_above")

        # Close lateral pass: bank right.
        return AvoidanceAction(AvoidanceKind.BANK_RIGHT, reason="lateral_pass")
