# Storage of Learnings — Layer 6

**Phase 2. In code:** `src/drone_ai/modules/storage/log.py`.
Benchmarked from the launcher's "Storage" card (writes synthetic
rows, stress-tests tail-truncation + malformed-JSON tolerance).

Per-drone persistent log of everything Layer 5 learned during flight.

## Purpose

- **Audit trail.** What did Layer 5 propose, accept, reject, roll
  back? Which warden runs passed? Which soft bounds did it push
  past and after how many sim recoveries?
- **Basis for Layer 7 personality selection.** The "best drone" is
  selected from Storage metrics.
- **Basis for the fleet A/B propagation test.** Subset cohort's
  outcomes are compared against control cohort using Storage
  aggregates.
- **Root-cause fodder** for Phase 1.5 failures — what was the last
  Layer 5 change before the failure?

## What gets written

Per-update rows:
- Timestamp, mission id.
- Which layer, which weight subset + hyperparameter diff.
- Warden-score pre/post.
- Rollback triggered? Y/N.
- Soft-bound push? Which, recovery-count evidence.

Per-mission rows:
- Outcome tag (delivered / late / aborted / crashed).
- Deadline class + result.
- Mission class.
- Manager pre-flight margin used.
- Upstream-cause tag if failed.

## Where it lives

- On-drone storage (write-only during flight).
- Uploaded to base station on docking via the secure pre-flight
  handshake path — same channel used for mission-file authentication,
  just in the reverse direction.
- No radio uploads mid-flight (see [`../comms.md`](../comms.md)).

## Relationship to `runs.csv`

`models/runs.csv` is the **training** log — one row per simulator
run. Storage of Learnings is the **field** log — one row per real
Layer 5 action on a deployed drone. They share the tier system and
failure tags, but live in different files.
