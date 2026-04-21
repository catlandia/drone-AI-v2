# Mission classes

Each mission file declares a `mission_class`. Orthogonal to priority
and deadline type — the three compose cleanly.

## LIFE_CRITICAL

Life-or-death delivery. Adrenaline, insulin, blood, AED, life-support
supplies.

**Overrides drone safety.** A crashed drone that delivered the
adrenaline on time is a **success**. A safely-returned drone whose
patient died is a **failure**.

What softens under LIFE_CRITICAL:
- Drone may fly in weather it would normally refuse.
- Drone may push soft limits without prior sim-recovery gating.
- Drone may accept crash-risk flight paths.
- Layer 5 is authorized to pick aggressive soft-bound promotions
  for this mission class.
- Battery RTB threshold can drop to 0% (no reserve held back).

What stays hard even under LIFE_CRITICAL:
- **Goals / reward function structure.** No reward hacking.
- **Module on/off switches.**
- **Mission-file authenticity.** LIFE_CRITICAL class can't be spoofed
  to unlock unsafe behavior — pre-flight handshake verifies the
  signature.
- **Harm to bystanders.** Risking *the drone* is authorized; risking
  people, animals, or property on the ground is not.
  Perception-Hazards still vetoes paths that endanger third parties.

## STANDARD

Everything else. Normal safety absolutes apply.

- Crash avoidance is hard.
- Hard-impact prevention is hard.
- Inversion base rule applies (Layer 5 may soften after N sim
  recoveries).
- Battery < 10% RTB is the default (Layer 5-tunable).

## Composition with priority and deadline

- **Priority** orders deliveries in the route plan.
- **Deadline + deadline_type** control what to do when ETA slips
  (see [`deadlines.md`](deadlines.md)).
- **Mission class** controls what the drone is *authorized to risk*
  to meet the other two.

Example: a `CRITICAL_WINDOW` + `LIFE_CRITICAL` delivery is both
time-pressured and life-pressured. Layer 5 is cleared to push
the envelope and not slow down, as long as bystander safety stays
intact.

## Logging

Mission class is written to `runs.csv:run_tag`. Every outcome
(delivered, delivered-late, aborted, crashed) gets a failure tag
indexed by class so Phase 2 Layer 5 can study upstream causes per
class.
