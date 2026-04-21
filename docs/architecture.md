# Architecture

Drone AI is built from **eight layers**. Layers 1-4 own the immediate
job of flying a delivery. Layers 5-8 own everything that makes the
fleet improve over time.

Authoritative master design: [`../PLAN.md`](../PLAN.md).

## Layer table

| # | Layer | Purpose | Phase |
|---|-------|---------|-------|
| 1 | Manager | Mission planning, priority queue, routing, pre-flight feasibility | 1 |
| 2 | Pathfinder | A*/RRT path planning, replan on new obstacles | 1 |
| 3 | Perception | CNN + Kalman; split into Obstacles / Hazards / Targets / Agents sub-models | 1 |
| 4 | FlyControl | PPO motor control for 5" FPV quadrotor | 1 |
| 5 | Adaptive | Online learning of all layers including itself; warden + rollback + A/B | 2 |
| 6 | Storage of Learnings | Per-drone persistent log of what Layer 5 learned | 2 |
| 7 | Personality | Transferable artifact — best drone's personality copied to others | 2 |
| 8 | Swarm Cooperation | Pre-mission plan + visual mutual avoidance; zero comms after takeoff | 2 |

All layers grade under the same P→W tier system (see
[`tier_system.md`](tier_system.md)). Models follow the naming
convention `{Grade} {DD-MM-YYYY} {module} v{N}.pt`.

## Layer diagram (Phase 1 active path)

```
       ┌────────────────────────────────────────────┐
       │              DroneAI (orchestrator)        │
       │                                            │
       │   ┌──────────────┐    ┌──────────────┐     │
       │   │   Manager    │──▶│   Pathfinder  │     │
       │   │  (Layer 1)   │    │   (Layer 2)  │     │
       │   └──────────────┘    └──────┬───────┘     │
       │         ▲                    │             │
       │         │                    ▼             │
       │   ┌──────────────┐    ┌──────────────┐     │
       │   │  Perception  │◀──│  FlyControl   │     │
       │   │  (Layer 3)   │    │   (Layer 4)  │     │
       │   └──────────────┘    └──────────────┘     │
       └────────────────────────────────────────────┘
                      │
                      ▼
        ┌───────────────────────────┐
        │   QuadrotorPhysics (sim)  │
        └───────────────────────────┘
```

Layers 5-8 in Phase 2 sit **beside** this stack, not inside it. Layer
5 reads all layers, writes weights + hyperparameters under strict
rails. Layer 6 is a write-only log. Layer 7 is an artifact exchanged
at base. Layer 8 runs before takeoff and via Perception-Agents
during flight.

## Data flow per tick (Phase 1)

1. **Perception** observes the world, returns noisy detections.
2. **Pathfinder** updates its internal world model from confirmed
   detections.
3. **Manager** picks the next delivery (or decides to RTB).
4. **Pathfinder** plans a path to the manager's chosen destination.
5. **FlyControl** converts the next waypoint into motor commands via
   the PPO policy.
6. **QuadrotorPhysics** integrates motor commands forward by `DT`.

Layer 5 (when enabled) observes this loop and proposes weight updates
through the warden. Layer 8 constrains Pathfinder at step 4 via
pre-agreed airspace segments; Perception-Agents adds other-drone
avoidance at step 1.

## Why separate AIs

Each layer has different ML requirements:

| Layer | Approach | Why |
|-------|----------|-----|
| Manager | heuristic / RL | discrete scheduling |
| Pathfinder | A* / RRT | deterministic, optimal on grid |
| Perception | CNN + Kalman | pattern recognition + tracking |
| FlyControl | PPO RL | continuous control, sparse rewards |
| Adaptive | meta-learning | guarded gradient edits |
| Storage | append-only log | audit trail, not inference |
| Personality | weight-subset export | transfer artifact |
| Swarm | pre-plan + vision | avoids radio |

Training them separately means each uses the right algorithm, has its
own curriculum, and is graded independently.

## Physics simulation

`QuadrotorPhysics` is a self-contained quadrotor simulator (no
external deps):

- **State:** `position(3), velocity(3), orientation(3), angular_velocity(3)` = 12D.
- **Action:** 4 motor commands in `[0, 1]`.
- **X-configuration** motor layout.
- **Integration:** 50 Hz (`DT = 0.02s`).
- **Crash conditions:** hard ground impact, inversion (see
  [`crash_model.md`](crash_model.md)).

Planned upgrades in [`physics_realism.md`](physics_realism.md):
inertia tensor, gyroscopic coupling, anisotropic drag, wind,
ground effect, prop wear, battery temperature.

## Observation space (FlyControl)

Every slot is annotated with its source sensor — the drone must be
able to produce this data **without GPS, without cloud, without
absolute world coordinates**. See [`sensors.md`](sensors.md).

Current 25-dim vector (normalized to `[-1, 1]`):

```
[0:3]    IMU+VIO     displacement from takeoff   / 50 m
[3:6]    IMU         velocity                     / 15 m/s
[6:9]    IMU         linear acceleration          / 20 m/s²
[9:12]   IMU+mag     orientation (roll,pitch,yaw)/ π
[12:15]  IMU         angular velocity             / 6 rad/s
[15:18]  internal    target − displacement        / 50 m
[18]     internal    distance to target           / 100 m
[19]     battery     state-of-charge ∈ [0, 1]
[20]     battery     temperature (normalized ±30 °C around 25 °C ref)
[21]     barometer   altitude above takeoff       / 50 m
[22]     physics     braking distance estimate    / 30 m
[23]     camera/ToF  nearest-obstacle distance    / 20 m
[24]     internal    carrying_package ∈ {0, 1}
```

Next planned: sensor noise injection (IMU bias, VIO drift, baro
drift) so training matches sim-to-real expectations more tightly.

## Grading and run log

Every training/eval run appends a row to `models/runs.csv`. Training
launches from the UI auto-save to `models/{Grade} {DD-MM-YYYY} flycontrol v{N}.pt`.

## File layout

```
src/drone_ai/
├── grading.py               # Unified tier system + run log
├── simulation/
│   ├── physics.py
│   └── world.py
├── modules/
│   ├── flycontrol/          # PPO motor control (Layer 4)
│   ├── pathfinder/          # A*/RRT planning (Layer 2)
│   ├── perception/          # Detection + tracking (Layer 3)
│   │   ├── obstacles.py     # generic obstacles (landed)
│   │   ├── hazards.py       # people/animals/vehicles/powerlines/water
│   │   ├── targets.py       # landing pads, drop markers
│   │   ├── agents.py        # other drones (for Layer 8)
│   │   ├── detector.py      # shared noise model + Detection dataclass
│   │   └── tracker.py       # Kalman tracker
│   ├── manager/             # Mission planning (Layer 1)
│   └── adaptive/            # Layer 5 — Phase 2 design locked
├── viz/
│   ├── launcher.py          # Stage launcher
│   ├── trainer_ui.py        # Live training window + auto-save
│   └── renderer3d.py        # Shared 3D renderer
├── drone.py                 # DroneAI orchestrator
├── curriculum.py            # Full training pipeline
└── cli.py
```
