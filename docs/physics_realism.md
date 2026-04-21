# Physics realism

> "The more realistic in the way of physics and situations, the more
> the drones are prepared." — design rule

A Phase 1 completion gate: unrealistic-physics runs do not count.

**Status:** landed in `src/drone_ai/simulation/physics.py` on
2026-04-22. The constants and formulas below match the implementation
1:1; update both when tuning.

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

## Implementation details

### Inertia tensor
Exposed via `QuadrotorPhysics.I_tensor` (3×3, diagonal) with inverse
cached. Values on a 5" FPV frame: `IXX = IYY = 0.0048 kg·m²`,
`IZZ = 0.0090 kg·m²`. Off-diagonal products of inertia are zero
for an X-quad; the tensor form is there so later asymmetric payloads
(e.g., side-mounted package) fit without changing the integrator.

### Propeller gyroscopic coupling
Rotor inertia `ROTOR_INERTIA = 6e-6 kg·m²`, full-throttle
`ROTOR_MAX_OMEGA = 3500 rad/s`. Net rotor angular momentum along
body +z is `H_prop = J · (ω_CCW − ω_CW)`. Gyroscopic torque is
`H_prop × ω_body`, added alongside the airframe gyroscopic term.

### Anisotropic drag
`DRAG_COEFF_BODY = [0.20, 0.20, 0.40]` (body-frame x/y/z).
Quadratic drag (`F = −c · |v| · v`) applied in body frame, then
rotated back to world. Forward and sideways are equal (the 5" frame
is roughly symmetric in the plane); vertical is roughly 2× higher
because the planform is the largest face.

### Reactive yaw torque (collective thrust ramp)
When the sum of commanded thrust changes, CCW and CW rotor pairs
don't accelerate identically on a real drone, producing the
"punchout yaw kick." Modeled as
`τ_reactive = −J · (Δω_CCW − Δω_CW) / Δt`, added to the
differential yaw torque.

### Wind + turbulence
Ornstein-Uhlenbeck-ish gust on top of a steady field. `set_wind(
mean, gust_std, gust_tau)` at reset. `DEPLOYMENT` stage picks
`mean ∈ [-4, 4] m/s` horizontal, `[-1, 1] m/s` vertical, and
`gust_std ∈ [0, 1.5]`. Correlation time `τ = 1.5 s`. Drag is
computed on airspeed (`v − wind`), not inertial velocity.

### Ground effect
`T_eff = T · (1 + GE_GAIN · max(0, 1 − z/GE_HEIGHT)²)` with
`GE_GAIN = 0.15`, `GE_HEIGHT = 0.5 m`. Effect vanishes above
~0.5 m.

### Prop wear + motor failure
Off by default; `set_wear(True, rate, failure_prob)` enables it.
Per-motor `motor_health ∈ [0, 1]`. Wear drifts health down
proportional to throttle; independent failure probability per
motor per step can instantly seize a rotor. Thrust per motor is
scaled by its health.

### Battery temperature model
`BAT_TEMP_REF = 25 °C`, `BAT_TEMP_SAG = 0.30` (max 30% extra sag
at 0 °C). Temperature warms by `0.05 · mean_throttle` per step and
cools toward ambient at `0.002` per step. Exposed to FlyControl
observation (`battery_temp` slot, normalized around ref).

## Inertia as decision context

Inertia is useful beyond FlyControl — the *planners* need it too.

- **`QuadrotorPhysics.braking_distance(v)`** — returns a straight-line
  braking estimate assuming `0.8g` deceleration (leaves margin). Used
  as a new FlyControl observation and pushed to:
  - **Pathfinder:** `PathPlanner.set_braking_distance(m)`. The
    planner stores it; downstream corners that demand a tighter
    stop than current inertia allows can inflate margins.
  - **Manager:** `MissionPlanner.feasible(target, deadline_s)` uses
    the ETA estimate together with the battery budget.
- **Finite-differenced acceleration** is recorded on the drone state
  each step and added to FlyControl's observation vector.

## Observation updates (FlyControl, 19D → 25D)

New slots:
- `acceleration` (3) — world-frame linear acceleration.
- `battery_temp` (1) — normalized around 25 °C ±30 °C.
- `barometer_alt` (1) — altitude above takeoff (separate axis from VIO z).
- `braking_dist` (1) — current braking distance, normalized by 30 m.

Obs layout is documented at the top of
`src/drone_ai/modules/flycontrol/environment.py`.

## Impact on grading

Existing trained checkpoints from scalar-inertia / 19-dim-obs runs
are **not** Phase 1-valid. After the upgrades land, re-run the
curriculum, write new `{Grade} {DD-MM-YYYY} flycontrol v{N}.pt`
artifacts, and tag runs with a physics version in
`runs.csv:run_tag`.
