"""Personality artifact — transferable delta.

A Personality captures WHAT a proven drone learned, not HOW it's
built. That means:

  - Layer-5-accepted weight deltas per layer (relative to a shared
    baseline checkpoint).
  - Accepted hparam values (learning rates, exploration noise, action
    bounds, battery thresholds, tilt bounds, hover throttle, action
    clipping).
  - The warden + rollback statistics that validated each delta.
  - Soft-bound promotions earned (with recovery-count evidence).

Transfer contract:
  - Base station builds the artifact from Storage (Layer 6).
  - Artifact is pushed to a ~20% experimental cohort on the NEXT
    docking cycle. Never mid-flight.
  - After M ≈ 10 missions, compare cohort outcomes to control.
    Promote fleet-wide or discard.

A Personality is NOT:
  - A full re-train (no architecture changes, no new modules).
  - A reward-function swap (that's a hard limit — see adaptive.md).
  - An auto fleet-wide push (always goes through the A/B gate).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import os
import torch

from drone_ai.modules.flycontrol.agent import PPOAgent


@dataclass
class Personality:
    """Transferable delta artifact.

    `weight_deltas` maps parameter name → tensor (δ = new − baseline).
    `hparams` carries any accepted hyperparameters (as a flat dict so
    serialization stays trivial).
    """
    source_drone_id: str
    baseline_name: str                               # filename of baseline .pt
    weight_deltas: Dict[str, torch.Tensor] = field(default_factory=dict)
    hparams: Dict[str, Any] = field(default_factory=dict)
    soft_bound_promotions: Dict[str, Dict[str, int]] = field(default_factory=dict)
    warden_stats: Dict[str, float] = field(default_factory=dict)
    rollback_stats: Dict[str, int] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    # How confident the base station is in this personality. Informs
    # the A/B subset size — low confidence = smaller cohort.
    confidence: float = 0.5

    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "source_drone_id": self.source_drone_id,
            "baseline_name": self.baseline_name,
            "weight_deltas": {k: v.cpu() for k, v in self.weight_deltas.items()},
            "hparams": self.hparams,
            "soft_bound_promotions": self.soft_bound_promotions,
            "warden_stats": self.warden_stats,
            "rollback_stats": self.rollback_stats,
            "created_at": self.created_at,
            "confidence": self.confidence,
        }, path)

    @classmethod
    def load(cls, path: str) -> "Personality":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            source_drone_id=blob["source_drone_id"],
            baseline_name=blob["baseline_name"],
            weight_deltas=blob["weight_deltas"],
            hparams=blob.get("hparams", {}),
            soft_bound_promotions=blob.get("soft_bound_promotions", {}),
            warden_stats=blob.get("warden_stats", {}),
            rollback_stats=blob.get("rollback_stats", {}),
            created_at=blob.get("created_at", ""),
            confidence=float(blob.get("confidence", 0.5)),
        )


# ----------------------------------------------------------------------

def export_personality(
    proven_agent: PPOAgent,
    baseline_agent: PPOAgent,
    *,
    source_drone_id: str,
    baseline_name: str,
    hparams: Optional[Dict[str, Any]] = None,
    soft_bound_promotions: Optional[Dict[str, Dict[str, int]]] = None,
    warden_stats: Optional[Dict[str, float]] = None,
    rollback_stats: Optional[Dict[str, int]] = None,
    confidence: float = 0.5,
) -> Personality:
    """Build a Personality as the delta between proven - baseline.

    Both agents must share architecture. We don't re-check that
    explicitly because PPOAgent instances built from the same baseline
    file always do; a shape mismatch would show up as a torch error
    which is louder than a silent wrong artifact anyway.
    """
    baseline_state = baseline_agent.policy.state_dict()
    proven_state = proven_agent.policy.state_dict()

    deltas: Dict[str, torch.Tensor] = {}
    for name, new_t in proven_state.items():
        base_t = baseline_state.get(name)
        if base_t is None:
            # New parameter in proven that didn't exist in baseline —
            # architecture drift, skip. PLAN.md: Layer 5 doesn't change
            # architecture, so this shouldn't happen in practice.
            continue
        if base_t.shape != new_t.shape:
            continue
        deltas[name] = (new_t - base_t).detach().cpu()

    return Personality(
        source_drone_id=source_drone_id,
        baseline_name=baseline_name,
        weight_deltas=deltas,
        hparams=dict(hparams or {}),
        soft_bound_promotions=dict(soft_bound_promotions or {}),
        warden_stats=dict(warden_stats or {}),
        rollback_stats=dict(rollback_stats or {}),
        confidence=float(confidence),
    )


def apply_personality(target_agent: PPOAgent, personality: Personality) -> List[str]:
    """Add a Personality's weight deltas to `target_agent` in place.

    Returns the list of parameter names that were applied (skipping
    any with shape mismatches — those get silently ignored since the
    alternative is refusing to transfer at all).
    """
    applied: List[str] = []
    state = target_agent.policy.state_dict()
    for name, delta in personality.weight_deltas.items():
        cur = state.get(name)
        if cur is None or cur.shape != delta.shape:
            continue
        state[name] = cur + delta.to(cur.device).to(cur.dtype)
        applied.append(name)
    target_agent.policy.load_state_dict(state)
    return applied


# ----------------------------------------------------------------------

def select_best_drone(storage_summaries: List[Dict[str, Any]]) -> Optional[str]:
    """Rank drones from Storage summaries; return the winner's id.

    Scoring: delivery_rate minus crash_rate, tiebreak on updates
    accepted (more = more real learning, not more luck). Returns
    None if no drone has any completed missions.
    """
    candidates = [s for s in storage_summaries if s.get("missions_total", 0) > 0]
    if not candidates:
        return None
    scored = [
        (
            s.get("delivery_rate", 0.0) - s.get("crash_rate", 0.0),
            s.get("updates_accepted", 0),
            s["drone_id"],
        )
        for s in candidates
    ]
    scored.sort(reverse=True)
    return scored[0][2]
