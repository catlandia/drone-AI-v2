"""Swarm pre-mission plan — authored at base, read-only in flight.

The plan is the ONLY thing drones share about each other's intent.
After takeoff there is no radio negotiation, so every contingency has
to be pre-committed here. See docs/modules/swarm.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class DroneRole(str, Enum):
    PRIMARY = "primary"              # leads multi-drone deliveries
    SECONDARY = "secondary"          # supports the primary
    SPOTTER = "spotter"              # advance perception
    PAYLOAD = "payload"              # carries part of a split payload
    RELAY = "relay"                  # visual relay between drones
    SOLO = "solo"                    # independent mission


@dataclass
class AirspaceSegment:
    """A time-slotted corridor that minimises visual-avoidance load.

    A drone must stay inside its segment for the segment's time window.
    `altitude_band` is the (min, max) z allowed inside the corridor.
    """
    drone_id: str
    start_s: float
    end_s: float
    # Polyline of (x, y) in the base-relative VIO frame defining the
    # centerline of the corridor. The corridor is inflated by
    # `half_width` meters laterally.
    centerline: List[Tuple[float, float]] = field(default_factory=list)
    half_width: float = 5.0
    altitude_band: Tuple[float, float] = (3.0, 25.0)


class ContingencyKind(str, Enum):
    SWARM_MATE_FAILED = "swarm_mate_failed"
    CANT_LAND_AT_BASE = "cant_land_at_base"


@dataclass
class Contingency:
    kind: ContingencyKind
    # Free-form parameters. For SWARM_MATE_FAILED, "beacon_altitude"
    # sets how high to circle the failure marker; for CANT_LAND, the
    # ladder itself is fixed (comms.md) and params here only tune it.
    params: Dict[str, float] = field(default_factory=dict)


@dataclass
class DroneAssignment:
    """One drone's slice of a swarm mission."""
    drone_id: str
    role: DroneRole = DroneRole.SOLO
    route: List[np.ndarray] = field(default_factory=list)   # VIO-frame waypoints
    mission_class: str = "STANDARD"                         # STANDARD / LIFE_CRITICAL
    deadline_s: float = 0.0
    deadline_type: str = "SOFT"                             # HARD / SOFT / CRITICAL_WINDOW
    preflight_margin: float = 0.15


@dataclass
class SwarmPlan:
    """Whole-fleet plan, authored at base pre-takeoff.

    Drones load this plan into a read-only memory page before takeoff,
    along with the signed/hashed mission file. See docs/comms.md for
    the pre-flight handshake that authenticates both.
    """
    mission_id: str
    assignments: Dict[str, DroneAssignment] = field(default_factory=dict)
    airspace: List[AirspaceSegment] = field(default_factory=list)
    contingencies: List[Contingency] = field(default_factory=list)

    # ------------------------------------------------------------------

    def drone_ids(self) -> List[str]:
        return list(self.assignments.keys())

    def assignment_for(self, drone_id: str) -> Optional[DroneAssignment]:
        return self.assignments.get(drone_id)

    def peers_of(self, drone_id: str) -> List[str]:
        return [d for d in self.assignments if d != drone_id]

    def contingency(self, kind: ContingencyKind) -> Optional[Contingency]:
        for c in self.contingencies:
            if c.kind == kind:
                return c
        return None


def build_swarm_plan(
    mission_id: str,
    assignments: List[DroneAssignment],
    airspace: Optional[List[AirspaceSegment]] = None,
    contingencies: Optional[List[Contingency]] = None,
) -> SwarmPlan:
    """Convenience constructor — keyed-by-id for O(1) peer lookup."""
    return SwarmPlan(
        mission_id=mission_id,
        assignments={a.drone_id: a for a in assignments},
        airspace=list(airspace or []),
        contingencies=list(contingencies or [
            # Sensible defaults — can be overridden by the mission file.
            Contingency(ContingencyKind.SWARM_MATE_FAILED,
                        params={"beacon_altitude": 10.0, "circle_seconds": 30.0}),
            Contingency(ContingencyKind.CANT_LAND_AT_BASE),
        ]),
    )
