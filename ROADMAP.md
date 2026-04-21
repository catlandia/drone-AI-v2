# Drone AI — Roadmap

This file captures the design intent the project is being built against.
It isn't a to-do list — it's the constraints and decisions that every
future change should be measured against.

## The target system

A **fully autonomous, offline delivery drone** — not a teleop assistant,
not a cloud-connected flyer, not something a human pilots. The operator
may be in an area where internet is jammed or deliberately cut off, so
the drone has to decide on its own whether to continue, return, abort,
or land.

Everything in the architecture flows from that single goal.

## Constraints

- **Offline at inference.** All models load from disk and run locally
  on the drone. No cloud APIs, no hosted inference, no remote telemetry.
- **No human in the loop.** The AI owns the mission once it takes off.
- **No satellites.** GPS is not considered available. The drone has to
  localise itself from onboard sensors alone.
- **Zero comms after takeoff.** Connection exists only at the base.
  Mid-mission link failure isn't a scenario — it's the default assumption.
  Jamming-resistant by construction.

## 8-layer architecture

| # | Layer | Purpose | Phase |
|---|-------|---------|-------|
| 1 | Manager | Mission planning, priority queue, routing, pre-flight feasibility | 1 |
| 2 | Pathfinder | A* / RRT path planning, replan on new obstacles | 1 |
| 3 | Perception | CNN + Kalman — split into Obstacles / Hazards / Targets / Agents sub-models | 1 |
| 4 | FlyControl | PPO motor control for 5" FPV quadrotor | 1 |
| 5 | Adaptive | Online learning of all layers including itself, with warden + rollback + A/B | 2 |
| 6 | Storage of Learnings | Per-drone persistent log of what Layer 5 learned | 2 |
| 7 | Personality | Transferable artifact — best drone's personality copied to others | 2 |
| 8 | Swarm Cooperation | Pre-mission plan + visual mutual avoidance, no comms after takeoff | 2 |

Every layer is gradable under the same **P → W** tier-list system
(`grading.py`, 20 tiers). Models are named `{Grade} {DD-MM-YYYY} {module} v{N}.pt`.

## Phasing

- **Phase 1 (current):** train Layers 1-4 to their best on realistic
  physics. Unblock by fixing the pygame exit bug; add inertia/gyro/drag,
  wind, ground effect, prop wear, battery-temp to the sim.
- **Phase 1.5:** real-life test of trained Layers 1-4.
- **Phase 2:** build Layers 5-8. Designs are locked in memory/; no code
  until Phase 1.5 is done.

## Run log

Every training run and every eval appends a row to `models/runs.csv`
with: timestamp, date, minutes, module, stage, best/avg score, letter
grade, updates, episodes, run tag. Deadline outcomes and failure tags
are included so Layer 5 has something to study later.

## Sensor realism (observation space)

The FlyControl observation only contains data a real offline drone
could sense. Each slot is annotated with its source sensor:

- IMU: orientation, angular velocity, (derived) velocity, acceleration
- Barometer: altitude above takeoff
- VIO (camera + IMU): displacement from takeoff point (drift-accumulating)
- Camera / ToF: nearest-obstacle distance
- Battery monitor: state-of-charge, temperature
- Internal state: carrying flag

Not used anywhere:
- GPS (satellite — explicitly ruled out)
- Cloud maps, weather APIs, any remote data source
- Ground-truth absolute world position
- Radio comms mid-mission (design forbids it)

## Physics upgrades (Phase 1)

- Moment of inertia tensor, gyroscopic coupling, anisotropic drag,
  tighter motor torque reaction.
- Wind + turbulence, ground effect, prop damage / motor wear,
  battery temperature model.
- Braking-distance estimate exposed to Pathfinder + Manager.

Realism > convenience. Training runs on unrealistic physics don't
count toward Phase 1 completion.

## Crash model

Real FPV drones are expensive and slow to rebuild, so crashing is a
real event:

- Inversion (roll or pitch ≥ 90°) → dead. Propellers pointing down
  means the thrust vector can't recover the drone in time.
- Hard ground impact (|vz| > 8 m/s) → dead.
- Soft touchdown → bounce and settle, not a crash.
- Crash reward is **-50**. Do not soften this to "help" training; fix
  exploration through hover-biased actor init, lowered log_std, and
  BC warm-up instead.

Layer 5 may soften the inversion bound **only after** N successful
simulated recoveries past it — never by default.

## Mission classes

Orthogonal to priority and deadline type.

- **LIFE_CRITICAL** — life-or-death delivery (adrenaline, insulin,
  blood, AED). Overrides drone safety: a crashed drone that delivered
  in time is a success; a safely-returned drone whose patient died
  is a failure. Soft limits loosen; bystander safety stays hard.
- **STANDARD** — everything else. Normal safety absolutes apply.

## Deadlines

Each delivery declares `deadline` + `deadline_type`.

- **HARD** — worthless after deadline → abort on infeasible, save battery.
- **SOFT** — late still useful → continue and log late.
- **CRITICAL_WINDOW** — push the envelope; Layer 5 may use softened limits.

Manager rejects missions pre-flight if `ETA + 15% margin > deadline`.
Mid-flight ETA is re-estimated continuously; behavior on infeasibility
depends on the type.

## Goals (read-only)

Goals are parameterized (task + parameters + deadline + type + priority
+ mission class), authored in a mission file before takeoff, stored
in a read-only memory page. Layer 5's gradient updates have a
goal-structure mask zeroing out weights that touch goal parsing.

Why: prevents the AI from "giving itself infinite points for doing nothing."

## Prevention over runtime arbitration

Runtime "conflicts" (battery vs delivery, priority vs efficiency,
replan vs deadline) are planning failures, not runtime problems.

- Manager pre-flight rejects missions that can't fit with margin.
- Mission files pre-commit rules for known-unknown events (swarm-mate
  failed, can't-land-at-base).
- Continuous re-estimation triggers RTB early — never at crisis.
- When a conflict fires anyway → `runs.csv` gets a failure tag, and
  Layer 5 studies the upstream cause.

We do **not** build a general runtime conflict resolver.

## Swarm rules (Layer 8)

- All comms happen only at base. No radio post-takeoff.
- Swarm-mate-failed policy: nearest surviving drone diverts to mark
  the failure point (unless itself on LIFE_CRITICAL); others continue.
- Can't-land-at-base ladder: no loiter → search alternate zone →
  keep searching while battery allows → slow vertical descent at
  critical battery.

## Adaptive module — Layer 5 safety rails

- **Can modify:** weights + hyperparameters of all layers + itself.
- **Cannot modify:** architecture, source code, reward function structure,
  goals, module on/off switches.
- **Soft limits** (can push past after N sim recoveries): tilt bound,
  hover throttle, action clipping, battery RTB threshold.
- **Hard limits** (never): ground-impact detection, reward structure,
  goals, module toggles.
- **Mid-flight updates:** FlyControl only. Landed-only for Manager /
  Pathfinder / Perception / Layer 5 itself.
- **Warden:** frozen reward copy scores every proposed update on N sim
  episodes. Rejects any update that reduces warden score. Non-negotiable.
- **Rollback:** rolling average of last 20 episodes below previous best
  → roll back to pre-update checkpoint.
- **Fleet propagation:** improvement pushes to a ~20% A/B subset first.
  Only promotes to full fleet after ~10-mission trial outperforms control.

### Deployment modes

- **Train + Deploy:** Adaptive on, drone learns in field.
- **Pre-trained Deploy:** Adaptive disabled, frozen weights, predictable,
  smaller compute.

`DroneAI(adaptive=True, ...)` opts in; default off.

## Known open work

- Fix pygame exit / file-reload bug (Phase 1 unblocker).
- Phase 1 physics upgrades (inertia tensor, gyro, drag, wind, ground
  effect, prop wear, battery temp).
- Perception split into sub-models.
- BC warm-up — use the onboard PD controller as an imitation signal for
  the PPO actor before PPO takes over.
- Sensor noise injection — IMU bias, VIO drift, baro drift.
- Full curriculum run across Layers 1-4, producing graded checkpoints.

See the auto-memory index at
`~/.claude/projects/E--Projects-Drone-AI/memory/MEMORY.md` for the
full list of design decisions captured so far.
