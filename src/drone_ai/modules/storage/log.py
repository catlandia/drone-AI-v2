"""Storage of Learnings — append-only per-drone JSONL log.

Two record kinds share one file:

- `UpdateRecord`   — one row per Layer 5 proposed-update action
                     (accepted, rejected, rolled-back, soft-bound push).
- `MissionRecord`  — one row per mission outcome (delivered / late /
                     aborted / crashed), tagged with deadline class,
                     mission class, pre-flight margin, upstream cause.

The pair gives Layer 7 everything it needs to rank drones, and the
Phase 2 A/B subset test everything it needs to compare cohorts.

`runs.csv` is the SIMULATION log. This file is the FIELD log — they
live apart on purpose: one captures training, the other captures
deployed-drone behavior. Both share the tier system and failure tags.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional


class MissionOutcome(str, Enum):
    DELIVERED = "delivered"
    DELIVERED_LATE = "delivered_late"
    ABORTED = "aborted"
    CRASHED = "crashed"
    RTB = "rtb"


class UpstreamCause(str, Enum):
    """Why a mission failed — populated by Layer 5 post-mortem so the
    next pre-flight gate can prevent the same class of failure."""
    MARGIN_TOO_THIN = "margin_too_thin"
    BATTERY_CURVE_STALE = "battery_curve_stale"
    PERCEPTION_MISSED = "perception_missed"
    PATHFINDER_TRAP = "pathfinder_trap"
    FLYCONTROL_DEVIATION = "flycontrol_deviation"
    WIND_EXCEEDED = "wind_exceeded"
    HARDWARE_FAILURE = "hardware_failure"
    UNKNOWN = "unknown"


@dataclass
class UpdateRecord:
    """A single Layer 5 action on a single layer."""
    mission_id: str
    layer: str                          # flycontrol / manager / pathfinder / perception / adaptive
    accepted: bool                      # did the warden allow it?
    rejected_reason: Optional[str] = None  # "warden_score_drop", "rollback_triggered", ...
    warden_score_pre: Optional[float] = None
    warden_score_post: Optional[float] = None
    rollback_triggered: bool = False
    soft_bound: Optional[str] = None    # e.g. "tilt_bound", "battery_threshold"
    soft_bound_recoveries: int = 0      # how many successful sim recoveries so far
    hparam_diff: Dict[str, Any] = field(default_factory=dict)
    weight_norm_delta: Optional[float] = None  # ||Δw||₂, summarized
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    kind: str = "update"

    def to_row(self) -> Dict[str, Any]:
        r = asdict(self)
        return r


@dataclass
class MissionRecord:
    """One mission outcome on the deployed drone."""
    mission_id: str
    outcome: MissionOutcome
    deadline_type: str                  # HARD / SOFT / CRITICAL_WINDOW
    mission_class: str                  # STANDARD / LIFE_CRITICAL
    preflight_margin: float = 0.15
    deliveries_done: int = 0
    deliveries_total: int = 0
    upstream_cause: Optional[UpstreamCause] = None
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    kind: str = "mission"

    def to_row(self) -> Dict[str, Any]:
        r = asdict(self)
        r["outcome"] = self.outcome.value if isinstance(self.outcome, MissionOutcome) else str(self.outcome)
        if self.upstream_cause is not None:
            r["upstream_cause"] = (
                self.upstream_cause.value
                if isinstance(self.upstream_cause, UpstreamCause)
                else str(self.upstream_cause)
            )
        return r


class Storage:
    """Per-drone append-only log.

    Path: `logs/storage/drone_{drone_id}.jsonl`. One JSON object per
    line, `kind` field disambiguates `update` vs `mission`. Appending
    is single-process safe on disk — if we ever move to multi-process
    writers we'll swap in a lock file.
    """

    def __init__(self, drone_id: str, root: str = "logs/storage"):
        safe_id = str(drone_id).replace(os.sep, "_").replace("/", "_")
        self.drone_id = safe_id
        self.root = root
        self.path = os.path.join(root, f"drone_{safe_id}.jsonl")

    # ------------------------------------------------------------------

    def record_update(self, rec: UpdateRecord) -> None:
        self._append(rec.to_row())

    def record_mission(self, rec: MissionRecord) -> None:
        self._append(rec.to_row())

    def _append(self, row: Dict[str, Any]) -> None:
        os.makedirs(self.root, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    # ------------------------------------------------------------------

    def iter_rows(self) -> Iterator[Dict[str, Any]]:
        if not os.path.isfile(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # A half-written row at crash time. Skip rather than
                    # fail the whole read — Layer 7 selection is the
                    # consumer and it prefers partial data to none.
                    continue

    def updates(self) -> List[Dict[str, Any]]:
        return [r for r in self.iter_rows() if r.get("kind") == "update"]

    def missions(self) -> List[Dict[str, Any]]:
        return [r for r in self.iter_rows() if r.get("kind") == "mission"]

    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Aggregate used by Layer 7 personality selection and by the
        A/B cohort comparator. Kept small so a base station can fold
        it over many drones cheaply."""
        missions = self.missions()
        updates = self.updates()

        outcomes = {o.value: 0 for o in MissionOutcome}
        for m in missions:
            outcomes[m.get("outcome", "unknown")] = outcomes.get(m.get("outcome", "unknown"), 0) + 1

        accepted = sum(1 for u in updates if u.get("accepted"))
        rejected = sum(1 for u in updates if not u.get("accepted"))
        rolled_back = sum(1 for u in updates if u.get("rollback_triggered"))
        soft_pushes = sum(1 for u in updates if u.get("soft_bound"))

        delivered = outcomes.get("delivered", 0) + outcomes.get("delivered_late", 0)
        crashed = outcomes.get("crashed", 0)
        total = max(len(missions), 1)

        return {
            "drone_id": self.drone_id,
            "missions_total": len(missions),
            "missions_by_outcome": outcomes,
            "delivery_rate": delivered / total,
            "crash_rate": crashed / total,
            "updates_total": len(updates),
            "updates_accepted": accepted,
            "updates_rejected": rejected,
            "rollbacks": rolled_back,
            "soft_bound_pushes": soft_pushes,
        }
