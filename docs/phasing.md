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

### Physics realism (done)
- Moment of inertia tensor.
- Gyroscopic coupling.
- Anisotropic drag.
- Wind + turbulence.
- Ground effect.
- Prop damage / motor wear.
- Battery temperature model.
- Finite-differenced acceleration in FlyControl obs.
- Braking-distance estimate exposed to Pathfinder + Manager.

### Perception split (done)
Four sub-models ship in `modules/perception/`, each graded and
versioned independently. Obstacles uses the real noise model;
Hazards / Targets / Agents are Phase-1 stubs awaiting the CNN +
world-annotation upgrade:
- Perception-Obstacles
- Perception-Hazards
- Perception-Targets
- Perception-Agents

### Training

- **BC warm-up (done).** Fresh FlyControl launches pre-train the PPO
  actor on the onboard PD controller's actions, then clamp `log_std`
  down to −2.2 so the first stochastic rollouts don't wipe the
  learned policy. See `modules/flycontrol/pd_controller.py` and
  `PPOAgent.bc_warmup`.
- **Hover reward gradient (done).** The old
  `max(0, 2 − dist) * 0.5` was zero beyond 2 m, leaving PPO with no
  signal during the approach phase. A faint 20 m linear gradient
  now rides on top of the steep on-target bonus.
- **Sensor noise injection (open).** IMU bias, VIO drift, baro drift.
- **Full curriculum run (open).** Across all four stages; every run
  logged to `models/runs.csv`.

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

Phase 2 was originally gated on Phase 1.5 passing, but all four
layers now have code + a launcher-driven benchmark so the UI covers
every layer even while Phase 1 is still in flight. Real field
deployment still waits on Phase 1.5.

- [`modules/adaptive.md`](modules/adaptive.md) — **Layer 5: in code.**
  `AdaptiveLearner` with frozen `Warden`, 20-episode `RollbackMonitor`,
  and `SoftBoundRegistry` requiring N sim recoveries before Layer 5
  may push past a soft limit.
- [`modules/storage.md`](modules/storage.md) — **Layer 6: in code.**
  Append-only JSONL per drone with `UpdateRecord` and `MissionRecord`
  rows. Wired into `DroneAI` so missions auto-log outcomes.
- [`modules/personality.md`](modules/personality.md) — **Layer 7: in code.**
  Delta artifact of weights + hparams + soft-bound promotions;
  `export_personality` / `apply_personality`; manual
  `select_best_drone` ranker.
- [`modules/swarm.md`](modules/swarm.md) — **Layer 8: in code.**
  `SwarmPlan` + `SwarmCoordinator` with visual mutual avoidance,
  LIFE_CRITICAL-exempt swarm-mate-failed contingency, and zero
  post-takeoff radio by construction.

### Tunables deferred to Phase 2
- **N** — sim recoveries required before Layer 5 can soften a soft
  bound. Default: **50**.
- **Subset size** for fleet A/B test. Default: **~20%**.
- **M** missions per A/B trial. Default: **10**.
- **Pre-flight deadline margin.** Default: **15%** (Layer 5-tunable
  per drone).
