# Personality — Layer 7

**Phase 2. In code:** `src/drone_ai/modules/personality/artifact.py`.
Benchmarked from the launcher's "Personality" card (applies the
exported delta to noisy sibling baselines and measures recovery
residual).

Transferable artifact encapsulating a drone's learned personality —
the weight + hyperparameter subset that made it good.

## Purpose

- Copy proven improvements across the fleet without re-training each
  drone from scratch.
- Run controlled A/B trials: the experimental cohort gets the new
  personality; the control cohort doesn't.
- Preserve the drone's identity across hardware swaps: when the
  physical frame is retired, its personality can be loaded into a
  new body.

## Contents

A personality is **not** a full model — it's a delta. Specifically:

- The Layer-5-accepted weight deltas per layer (relative to the
  baseline checkpoint).
- The accepted hyperparameter values (learning rates, exploration
  noise, action bounds, battery thresholds, tilt bounds, hover
  throttle assumption, action clipping).
- The warden + rollback statistics that validated each delta.
- The soft-bound promotions earned (with recovery-count evidence).

## Selection

Phase-2-early: manual. The operator looks at Storage of Learnings
metrics and designates the best drone. Its personality is copied.

Phase-2-later: automatic at base station. The station reads Storage
via QR-identified drones (QR codes engraved into the drone frame,
read via memory-drive extraction — not a camera job). It ranks
drones, exports the top personality as a transferable artifact,
and pushes per the A/B-subset rule in
[`adaptive.md`](adaptive.md).

## Transfer

1. Base station builds the personality artifact from the source
   drone's Storage.
2. Pushes it to the experimental cohort (~20% of fleet) during the
   next docking cycle.
3. `M ≈ 10` missions later, compare cohort outcomes against control.
4. Promote to full fleet or discard.

No mid-flight transfers. Ever.

## What a personality is not

- Not a full re-train — no architecture changes, no new modules.
- Not a reward-function swap — that's a hard limit.
- Not automatic fleet-wide deployment — always through the A/B gate.
