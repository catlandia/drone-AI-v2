# Crash model

Real FPV drones are expensive and slow to rebuild, so the crash model
takes that seriously. Crashing is a real event, not a training reset.

## Conditions

- **Inversion.** |roll| ≥ 90° or |pitch| ≥ 90° → **dead**. Propellers
  pointing down means the thrust vector can't recover the drone in
  time.
- **Hard ground impact.** |vz| at ground contact > 8 m/s → **dead**.
- **Soft touchdown.** |vz| ≤ 8 m/s at ground contact → bounce and
  settle, **not a crash**. The drone treats this as a completed
  landing or a re-seat.

## Crash reward

**-50.** Do not soften this value. Past attempts to help exploration
by reducing the penalty caused the policy to accept crashes as a
routine cost.

Fix exploration through the policy-side levers instead:
- **Hover-biased actor init** — start near a reasonable hover throttle.
- **Lowered log_std** — cleaner early signal, reduced noise.
- **BC warm-up** — use the onboard PD controller as an imitation
  signal for the PPO actor before PPO takes over.

## Layer 5 and soft limits

The 90° inversion bound is a **soft limit** — Layer 5 may push past
it for aggressive recoveries (e.g., dive recovery), but only after
**N successful simulated recoveries past the bound with no crash**.
Default N = 50, tunable in Phase 2.

Hard limits never move:
- Ground-impact crash detection (hitting the ground always ends the
  episode).
- Crash reward structure.
- Harm-to-bystanders gate from Perception-Hazards.

See [`modules/adaptive.md`](modules/adaptive.md) for full Layer 5
rules.

## Mission-class interaction

Under **LIFE_CRITICAL**, crash *avoidance* stops being a hard absolute —
a crashed drone that delivered the adrenaline on time is a success
(see [`mission_classes.md`](mission_classes.md)). The crash *model*
itself (inversion = dead, hard impact = dead) does not change; only
the policy's authorization to accept crash risk does.
