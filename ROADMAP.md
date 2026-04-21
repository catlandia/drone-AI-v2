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
- **Robust to comms loss.** Mid-mission link failure is a normal
  scenario, not an exception.

## 5 AI modules

| # | Module      | Purpose                                                      |
|---|-------------|--------------------------------------------------------------|
| 1 | Manager     | Mission planning — priority queue + routing                  |
| 2 | Pathfinder  | A* / RRT path planning                                       |
| 3 | Perception  | CNN obstacle detection + Kalman tracking                     |
| 4 | FlyControl  | PPO motor control for FPV quadrotor                          |
| 5 | Adaptive    | Optional online learner — fine-tunes in the field, swappable |

Every module is gradable under the same **P → W** tier-list system
(`grading.py`). Models are named `{Grade} {DD-MM-YYYY} {module} v{N}.pt`.

## Run log

Every training run and every eval appends a row to `models/runs.csv`
with: timestamp, date, minutes, module, stage, best/avg score, letter
grade, updates, episodes, run tag. This is the single file the user
uses to compare how much a change actually moved the needle.

## Sensor realism (observation space)

The FlyControl observation only contains data a real offline drone
could sense. Each slot is annotated with its source sensor:

- IMU: orientation, angular velocity, (derived) velocity
- Barometer: altitude above takeoff
- VIO (camera + IMU): displacement from takeoff point (drift-accumulating)
- Camera / ToF: nearest-obstacle distance
- Battery monitor: state-of-charge
- Internal state: carrying flag

Not used anywhere:
- GPS (satellite — explicitly ruled out)
- Cloud maps, weather APIs, any remote data source
- Ground-truth absolute world position

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

## Adaptive module — modes

- **Train + Deploy**: Adaptive layer present; drone keeps fine-tuning
  the FlyControl policy on its own in-field experience.
- **Pre-trained Deploy**: Adaptive layer disabled; frozen weights only,
  predictable behavior, smaller compute footprint.

Both modes are first-class. `DroneAI(adaptive=True, ...)` opts in; the
default is off. The `adaptive.train` benchmark grades the module by
measuring the recovery delta on a perturbed (OOD) environment.

## Known open work

- BC warm-up — use the onboard PD controller as an imitation signal for
  the PPO actor before PPO takes over. Reduces early-episode chaos.
- Sensor noise injection — today the sim provides clean IMU/VIO. Add
  an optional noise model (IMU bias drift, VIO drift, baro drift) so
  trained policies don't overfit to noise-free data.
- Comms-loss reasoning in Manager — explicit rules for "should I
  continue, return, or emergency-land" when the operator is silent.
- Full curriculum run across all 5 modules, producing graded checkpoints.

See the auto-memory index at
`~/.claude/projects/E--Projects-Drone-AI/memory/MEMORY.md` for the
full list of design decisions captured so far.
