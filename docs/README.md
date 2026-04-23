# Documentation

Full docs for Drone AI — a fully autonomous, offline, jamming-resistant
delivery drone. Built as an 8-layer architecture, phased over three
stages.

For the single authoritative design document, see the top-level
[`PLAN.md`](../PLAN.md). Everything here deep-dives into one slice of
that plan.

## Start here

- [`phasing.md`](phasing.md) — Phase 1 / 1.5 / 2 sequencing and what
  each phase must deliver.
- [`architecture.md`](architecture.md) — the 8 layers, how they fit.
- [`tier_system.md`](tier_system.md) — the P→W grading scale (all
  layers share it).
- [`training.md`](training.md) — training pipeline and curriculum.
- [`quickstart.md`](quickstart.md) — get it running.
- [`lessons_learned.md`](lessons_learned.md) — what broke before.

## Cross-cutting design

- [`prevention.md`](prevention.md) — why we don't build a runtime
  conflict resolver.
- [`mission_classes.md`](mission_classes.md) — LIFE_CRITICAL vs
  STANDARD; when the drone is allowed to trade its own safety for
  delivery success.
- [`deadlines.md`](deadlines.md) — HARD / SOFT / CRITICAL_WINDOW.
- [`goals.md`](goals.md) — parameterized, read-only mission files;
  goal-structure mask on Layer 5 updates.
- [`comms.md`](comms.md) — zero post-takeoff comms, pre-flight
  handshake, base reauth, can't-land-at-base ladder.
- [`sensors.md`](sensors.md) — IMU, baro, VIO, camera/ToF, battery;
  explicitly no GPS / no cloud / no absolute coords.
- [`physics_realism.md`](physics_realism.md) — inertia tensor, gyro,
  drag, wind, ground effect, prop wear, battery temp.
- [`crash_model.md`](crash_model.md) — inversion, hard-impact,
  crash reward = -50.

## Per-module

### Phase 1 (implemented / being trained)
- [`modules/flycontrol.md`](modules/flycontrol.md) — Layer 4: PPO motor control.
- [`modules/pathfinder.md`](modules/pathfinder.md) — Layer 2: A*/RRT.
- [`modules/perception.md`](modules/perception.md) — Layer 3: CNN + Kalman, split sub-models.
- [`modules/manager.md`](modules/manager.md) — Layer 1: mission scheduling + pre-flight gates.

### Phase 2 (design locked; not implemented)
- [`modules/adaptive.md`](modules/adaptive.md) — Layer 5: online learning with warden + rollback + A/B propagation.
- [`modules/storage.md`](modules/storage.md) — Layer 6: per-drone learning log.
- [`modules/personality.md`](modules/personality.md) — Layer 7: transferable personality artifact.
- [`modules/swarm.md`](modules/swarm.md) — Layer 8: pre-flight plan + visual mutual avoidance.

## Experiments + history

- [`experiments.md`](experiments.md) — grade-mixing experiments.

## Future work (not implemented)

- [`future_image_training.md`](future_image_training.md) — plan for
  self-building a perception dataset by scraping the web,
  filename-labelling, and CNN-training offline. Replaces the Phase 1
  noise-simulated perception when the time comes.
- [`inspector_ui.md`](inspector_ui.md) — how the visual inspectors
  work, what each pane shows, and the keyboard shortcuts.
