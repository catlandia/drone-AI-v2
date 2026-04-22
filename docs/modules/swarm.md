# Swarm Cooperation — Layer 8

**Phase 2. In code:** `src/drone_ai/modules/swarm/`
(`plan.py`, `coordinator.py`). Benchmarked from the launcher's
"Swarm" card against a hand-written right-of-way spec.

Multi-drone coordination with **zero radio comms after takeoff**.

## Core rule

All coordination happens at base, before takeoff. After takeoff, the
only "comms" between drones is visual — through Perception-Agents.

## Pre-mission plan

Distributed to all drones at the base before any of them takes off:

- **Routes.** Each drone knows its own route and the planned routes
  of every other drone in the mission.
- **Roles.** Primary, secondary, spotter, payload-carrier, etc.
- **Airspace segments.** Time-slotted corridors that minimize
  visual-avoidance burden.
- **Pre-committed contingencies.** See below.

## In-flight: visual mutual avoidance

Perception-Agents (a Perception sub-model) detects other drones in
camera view. Standard right-of-way + vertical-stack rules kick in.
No radio handshake — the drones were briefed on the handshake at
base.

Example: a 10-drone payload delivery (one heavy package split across
10 drones) coordinates entirely via the pre-plan + Perception-Agents
once in the air.

## Pre-committed contingencies

Written into the mission file; no runtime negotiation required.

### Swarm-mate failed
- **Nearest surviving drone** diverts to visually mark the failure
  point (drops a beacon or logs the location and circles briefly).
- **Exception:** if the nearest drone is itself on `LIFE_CRITICAL`,
  it does not divert — its mission outranks marking a failure point.
- **All other drones** continue their missions unchanged.
- Marker / location is logged locally; ground crew retrieves after
  the fleet returns.

### Can't-land-at-base
Same ladder as a single-drone mission — see [`../comms.md`](../comms.md).
Each drone runs it independently.

## Why no radio

Two reasons:
- **Jamming resistance.** A fleet-wide radio protocol is a fleet-wide
  attack surface. Zero radio = nothing to jam.
- **No operator trust model needed mid-flight.** Everything the fleet
  needs to agree on was agreed at the base, where the physical
  pre-flight handshake authenticated each drone.

See [`../comms.md`](../comms.md) for the full comms design and why
we don't ship an "emergency" channel.

## What Perception-Agents must do well

- Classify "drone" vs other moving objects.
- Estimate range and closing speed.
- Classify swarm-mate vs unknown drone (via QR / visual ID —
  detail TBD).

Graded independently like all Perception sub-models.
