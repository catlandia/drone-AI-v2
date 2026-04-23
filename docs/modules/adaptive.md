# Adaptive — Layer 5

**Phase 2. In code:** `src/drone_ai/modules/adaptive/`
(`learner.py`, `warden.py`, `rollback.py`, `soft_bounds.py`).
Benchmarked from the launcher's "Adaptive" card.

**Visual inspector:** `viz/inspector_adaptive.py` runs the same
baseline-vs-adapted comparison with side-by-side per-episode reward
bars so the user can see the online learner close (or not close)
the gap on a perturbed environment. See
[inspector_ui.md](../inspector_ui.md).

Online learning of every other layer, and of itself. Every update
is guarded.

## Scope

**CAN modify:**
- Neural-network weights of all layers, including itself.
- Hyperparameters: learning rates, exploration noise, action bounds,
  battery thresholds, tilt bounds, hover throttle, action clipping.

**CANNOT modify:**
- Network architecture, source code.
- Reward function structure (see Warden below).
- Goals or goal parsing (goal-structure mask on updates).
- Module on/off switches.

## Soft vs hard limits

**Softenable** — Layer 5 may push past after **N** successful
simulated recoveries past the bound with no crash:
- Tilt bound (90° inversion rule — e.g. aggressive dive recovery).
- Hover throttle assumption.
- Action clipping.
- Battery "land now" threshold (small adjustments — this is literally
  why it's the *Adaptation* layer).

**Hard — never touchable:**
- Ground-impact crash detection.
- Reward function structure.
- Goals.
- Module toggles.

Default N = **50** (Phase 2 tunable).

## Update timing (hybrid)

- **Mid-flight:** FlyControl only. Fast reflex layer — a bad update
  is recoverable within milliseconds.
- **Landed + idle only:** Manager, Pathfinder, Perception, and
  Layer 5 itself. A bad high-level update ruins an entire mission,
  so it waits for a safe boundary.

## Warden (non-negotiable)

A frozen copy of the original reward function lives in read-only
code. Every proposed weight update is scored by the warden (not by
Layer 5's own metric) on N simulated episodes before being accepted.

- Warden score drops vs baseline → **update rejected**.
- Warden score holds or improves → update accepted, goes to
  rollback monitor.

The warden is what prevents Layer 5 from slowly drifting its internal
metric while the real job gets worse.

## Rollback

Rolling average of the **last 20 episodes**. If it drops below the
previous best, roll back to the pre-update checkpoint. Single-episode
failures do not trigger rollback (too sensitive to wind / noise).

## Fleet propagation (A/B subset)

1. Layer 5 discovers an improvement on a single drone.
2. Layer 6 captures it; Layer 7 encapsulates it into a transferable
   personality artifact.
3. On docking, the improvement is pushed to a **~20% subset** of the
   fleet (experimental cohort), not all drones.
4. After **M ≈ 10** missions, compare subset performance against
   control cohort.
5. If subset outperforms → promote fleet-wide.
6. If not → discard. *"It was luck, not skill."*

Phase 2 tunables: subset fraction, M, warden episode count.

## Deployment modes

- **Train + Deploy:** `DroneAI(adaptive=True, ...)`. Drone learns in
  field.
- **Pre-trained Deploy:** default. Frozen weights, predictable,
  smaller compute.

## Interactions with other layers

- **FlyControl:** weights may update mid-flight under the warden.
- **Manager:** only landed updates; tunes pre-flight deadline margin,
  CRITICAL_WINDOW aggression.
- **Pathfinder:** only landed updates; tunes replan budget heuristics.
- **Perception (all sub-models):** only landed updates.
- **Layer 6 (Storage):** receives every Layer 5 action as a log row.
- **Layer 7 (Personality):** packages accepted updates for fleet
  transfer.

## What Layer 5 does NOT do

- Arbitrate runtime conflicts — see [`../prevention.md`](../prevention.md).
- Touch goals, reward structure, or module toggles — see
  [`../goals.md`](../goals.md).
- Phone home — see [`../comms.md`](../comms.md).
