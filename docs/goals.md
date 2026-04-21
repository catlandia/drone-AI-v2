# Goals

Goals are parameterized, authored before takeoff, and stored in a
read-only memory page that Layer 5 cannot touch.

## Format

Every goal has:

```
task             enum (pickup, dropoff, inspect, search, land, ...)
parameters       task-specific dict (coordinates, payload_id, etc.)
deadline         ISO-like timestamp
deadline_type    HARD | SOFT | CRITICAL_WINDOW
priority         low / normal / urgent / critical
mission_class    STANDARD | LIFE_CRITICAL
```

Goals are **not** a flat enum — the parameter block lets us extend
task types without changing the goal-parsing code every time.

## Authoring

The operator writes a mission file before takeoff. The mission file
is:
- **Signed / hashed** (exact method TBD — part of the pre-flight
  handshake in [`comms.md`](comms.md)).
- Loaded into a read-only memory page at boot.
- Never modified after takeoff by any layer, including Layer 5.

## Why read-only

> "giving the AI ability to self change the points is bad because it
> could just give itself inf points for doing nothing." — design rule

If Layer 5 could edit the goal structure, it could collapse the
reward landscape to "0 effort = max reward" and the drone stops doing
its job. So:

- **Goal structure** lives in read-only memory.
- **Layer 5's gradient updates** have a goal-structure mask that
  zeros out any weight that touches goal parsing. Hard isolation.
- **Reward function structure** is equally untouchable — the warden
  is a frozen copy of the original reward, and no proposed update is
  accepted if it reduces the warden's score (see
  [`modules/adaptive.md`](modules/adaptive.md)).

## What Layer 5 *can* do with goals

Nothing structural. Layer 5 may only tune:
- Pre-flight deadline margin (per drone).
- CRITICAL_WINDOW aggression.

That is it.
