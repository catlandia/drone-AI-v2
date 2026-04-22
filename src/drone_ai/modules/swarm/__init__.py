"""Swarm Cooperation — Layer 8.

Multi-drone coordination with ZERO radio comms after takeoff. All
coordination happens at base, pre-mission. In-flight coordination is
visual-only via Perception-Agents.

See docs/modules/swarm.md, docs/comms.md.
"""

from drone_ai.modules.swarm.plan import (
    SwarmPlan,
    DroneRole,
    DroneAssignment,
    AirspaceSegment,
    Contingency,
    build_swarm_plan,
)
from drone_ai.modules.swarm.coordinator import (
    SwarmCoordinator,
    AvoidanceAction,
    VisualContact,
)

__all__ = [
    "SwarmPlan",
    "DroneRole",
    "DroneAssignment",
    "AirspaceSegment",
    "Contingency",
    "build_swarm_plan",
    "SwarmCoordinator",
    "AvoidanceAction",
    "VisualContact",
]
