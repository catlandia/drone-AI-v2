# Physics realism

> "The more realistic in the way of physics and situations, the more
> the drones are prepared." — design rule

A Phase 1 completion gate: unrealistic-physics runs do not count.

## Core quadrotor dynamics

Currently scalar. Upgrades:

- **Moment of inertia tensor.** Pitch/roll/yaw have different inertias
  on a 5" FPV frame; a scalar glosses over that and lets the policy
  learn physically-impossible recoveries.
- **Gyroscopic coupling.** Spinning props add cross-axis torque when
  the airframe rotates. Matters most during aggressive yaw changes
  and inversion-recovery maneuvers.
- **Anisotropic drag.** Forward drag ≠ sideways drag ≠ vertical drag.
  A 5" frame has very different frontal vs planform areas.
- **Tighter motor torque reaction.** When collective thrust changes,
  the reactive yaw torque kicks the airframe. Currently modeled too
  loosely.

## Environment

- **Wind + turbulence.** Steady field component + random gusts
  sampled from a von Kármán-like profile. Training must be robust to
  ~10 m/s gusts.
- **Ground effect.** Extra lift when hovering close to the ground.
  Critical for landing and low-altitude maneuvers.
- **Prop damage / motor wear.** Gradual efficiency loss or sudden
  failure — feeds into Layer 5's "every possible failure" scope
  later.
- **Battery temperature model.** Voltage sag depends on temperature
  (and cell age). Cold battery in winter ops means shorter mission
  envelopes; Layer 5 per-drone data.

## Inertia as decision context

Inertia is useful beyond FlyControl — the *planners* need it too.

- **Braking distance at current velocity** is exposed as a new
  observation to Pathfinder and Manager. They use it when deciding
  whether a sharp corner fits, or whether ETA is realistic.
- **Finite-differenced acceleration** is added to FlyControl's
  observation so the policy can feel "still decelerating from the
  last command — don't command full stop yet."

## Impact on grading

Existing trained checkpoints from scalar-inertia runs are not
Phase 1-valid. After the upgrades land, re-run the curriculum,
write new `{Grade} {DD-MM-YYYY} flycontrol v{N}.pt` artifacts, and
tag runs with a physics version in `runs.csv:run_tag`.
