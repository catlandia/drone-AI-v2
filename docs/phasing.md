# Phasing

The project is explicitly sequenced. Phase 2 does not begin until
Phase 1.5 passes.

## Phase 1 — train Layers 1-4 to their best

Constraint: training on unrealistic physics **does not count**. The
simulator must have the upgrades in [`physics_realism.md`](physics_realism.md)
before a run is considered a Phase 1 baseline.

### Immediate unblocker (done)
- Pygame exit / file-reload bug: trainer UI now auto-saves every run
  as `models/{Grade} {DD-MM-YYYY} flycontrol v{N}.pt`. Launcher
  `--stage` returns exit code 0 on success.

### Physics realism
- Moment of inertia tensor.
- Gyroscopic coupling.
- Anisotropic drag.
- Wind + turbulence.
- Ground effect.
- Prop damage / motor wear.
- Battery temperature model.
- Finite-differenced acceleration in FlyControl obs.
- Braking-distance estimate exposed to Pathfinder + Manager.

### Perception split
Into four sub-models, each graded and versioned independently:
- Perception-Obstacles
- Perception-Hazards
- Perception-Targets
- Perception-Agents

### Training
- BC warm-up using the onboard PD controller as an imitation signal
  for the PPO actor.
- Sensor noise injection: IMU bias, VIO drift, baro drift.
- Full curriculum run across all four stages; every run logged to
  `models/runs.csv`.

### Exit criteria
All four Phase 1 modules reach a **B+ or better** grade on the
deployment stage, and at least one pass shows stable LIFE_CRITICAL-class
performance under the crash model. Then move to Phase 1.5.

## Phase 1.5 — real-life test

Real hardware. Same tier system. Same run log (tagged with a
hardware run tag). Purpose: catch anything the sim still gets wrong
before building the meta-learner on top.

### Exit criteria
A real 5" FPV drone flies a scripted delivery mission with the
trained Layers 1-4 stack, meeting the same B+ bar on real-world
scoring, with the crash model behaviors confirmed
(inversion bail-out, soft touchdown bounce, hard-impact crash).

## Phase 2 — build Layers 5-8

Only after 1.5 passes. Designs are already locked — see:
- [`modules/adaptive.md`](modules/adaptive.md) — Layer 5
- [`modules/storage.md`](modules/storage.md) — Layer 6
- [`modules/personality.md`](modules/personality.md) — Layer 7
- [`modules/swarm.md`](modules/swarm.md) — Layer 8

### Tunables deferred to Phase 2
- **N** — sim recoveries required before Layer 5 can soften a soft
  bound. Default proposal: **50**.
- **Subset size** for fleet A/B test. Default proposal: **~20%**.
- **M** missions per A/B trial. Default proposal: **10**.
- **Pre-flight deadline margin.** Default: **15%** (Layer 5-tunable
  per drone).
