# Prevention principle

> "the solution to the problem of batteries is actually just to prevent
> that from happening entirely. if somehow it still happened, then it
> means we have failed at some point." — user

Runtime "conflicts" (battery vs delivery, priority vs efficiency,
replan vs deadline, comms-loss vs deadline) are **planning failures**,
not runtime problems. We do not build a general runtime conflict
resolver.

## Why no runtime arbitrator

A learned conflict resolver, even as a "safety net," is a
reward-hacking door. It lives adjacent to the goal structure; it
naturally wants to tune its own objective to make conflicts go
away. That violates the read-only goals rule (see
[`goals.md`](goals.md)).

## How we prevent instead

### Manager pre-flight gates
- **Battery vs delivery.** Manager rejects missions whose energy
  budget doesn't fit with margin. No in-flight fuel crisis to
  resolve.
- **Priority vs efficiency.** Manager pre-computes the committed
  route before takeoff. No runtime re-ordering.
- **Deadline slack.** `ETA + 15% margin > deadline` → rejected at
  the gate.
- **Replan slack.** Pathfinder reserves replan budget; if a detour
  doesn't fit, abort cleanly (per deadline type).

### Mission-file pre-committed rules
- **Swarm-mate-failed policy** — nearest surviving drone diverts
  to mark the failure point (unless it's itself on
  `LIFE_CRITICAL`). Others continue.
- **Can't-land-at-base fallback ladder** — see [`comms.md`](comms.md).

### Continuous re-estimation (not crisis-mode)
- Mid-flight ETA + battery recomputation every N seconds.
- RTB trigger fires **immediately** when margin shrinks — not at a
  crisis threshold.

### When a conflict fires anyway
- `runs.csv` gets a **hard failure tag**.
- Layer 5's job is to study the upstream cause (was the margin too
  thin, battery curve stale, wind estimate wrong) and tighten
  margins for the next mission — not to invent runtime arbitration.

## What this forbids

- A "learned tradeoff module" at any layer. User explicitly killed
  this direction.
- "Emergency override" channels that let the drone route around its
  own rules at runtime.
- Any Layer 5 edit that expands its authority beyond the scoped
  weights + hyperparameters list in [`modules/adaptive.md`](modules/adaptive.md).
