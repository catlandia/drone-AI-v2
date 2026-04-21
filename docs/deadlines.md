# Deadlines

Each delivery in the mission file declares `deadline` +
`deadline_type`. Type controls what the drone does when the ETA no
longer fits.

## Types

### HARD
Package is worthless after the deadline. Blood for surgery, hot food,
time-locked drop.

Mid-flight infeasibility → **abort delivery, RTB, save battery**. Log
`deadline_infeasible_aborted`.

### SOFT
Late delivery is still valuable. Standard parcel, non-urgent
medicine.

Mid-flight infeasibility → **continue, deliver late**. Log
`delivered_late`.

### CRITICAL_WINDOW
Time-sensitive life/safety. Usually paired with `LIFE_CRITICAL`
mission class — AED, life-support supplies.

Mid-flight infeasibility → **continue, push the flight envelope
harder**. Layer 5 may use softened limits. Log the outcome either
way.

## Pre-flight gate

Manager rejects a mission before takeoff if:

```
ETA + margin > deadline
```

- Margin default: **15%**.
- Margin is Layer 5-tunable per drone in Phase 2.

Missions that fail the gate never take off — no runtime "fuel
emergency" to resolve. See [`prevention.md`](prevention.md).

## Mid-flight

Continuous ETA recomputation every N seconds. When the re-estimate
pushes past the deadline, behavior depends on `deadline_type` above.

## What Layer 5 can tune

- Pre-flight margin (per drone).
- CRITICAL_WINDOW aggression level.

## What Layer 5 cannot touch

- Whether deadlines exist.
- Deadline values.
- Deadline type assigned to a given delivery.
- The `HARD` + infeasible = abort rule.

## Logging

Every deadline outcome is tagged in `runs.csv`: on-time / late /
aborted. Layer 5 studies **upstream causes** (margin too thin,
battery curve stale, VIO drift larger than expected) — it does not
reach in at runtime.
