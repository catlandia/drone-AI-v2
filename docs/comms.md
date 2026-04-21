# Comms

**Zero radio comms after takeoff. Ever.** Connection exists only at
the base.

> "connection is back only at the base. it will help them survive
> signal blockage if some jackass thinks he is a cool hacker." — design rule

Any code path that would "just ping the operator" or "check in with
base" mid-flight is wrong by construction. If the drone has a question
mid-mission, the question was a pre-flight planning bug — solve it
there, not over the air.

## Where trust lives

### 1. Pre-takeoff handshake
- Mission file authenticated (signed / hashed — exact method TBD).
- Base station identity verified by drone.
- Drone runs self-integrity check.
- Any failure → refuse takeoff.

### 2. Base reauthentication on return
- Visual landing-pad pattern + engraved QR must match the drone's
  expected base.
- Decoy base → do not land.

### 3. Can't-land-at-base fallback ladder
1. **No loiter** at base (save battery — hovering eats charge for no
   gain).
2. **Search for alternate landing zone** via Perception-Targets.
3. **Keep searching** as long as battery allows — no early give-up.
4. **Battery critical** → slow controlled vertical descent wherever
   the drone currently is.

### 4. Deadlines are absolute
No operator to extend them. Whatever the mission file said at takeoff
is final. Miss = logged failure. Layer 5 studies the upstream cause.

## Why not even an emergency channel

An emergency abort channel is an attack surface. A jammer that knows
the abort frequency can force-land the entire fleet on command. The
only way to be jamming-resistant by construction is to have zero
runtime radio at all — the drone decides everything on its own from
the mission file it was handed at takeoff.

## What the drone does instead of asking

| Situation | What a cloud drone would do | What this drone does |
|-----------|-----------------------------|----------------------|
| Lost uplink | "Please confirm continue" | Already decided pre-flight |
| Base not visible on return | "Please send new LZ" | Fallback ladder above |
| Mid-flight obstacle storm | "Request rerouting" | Pathfinder replan + margin budget |
| Swarm-mate failure | "Request new role" | Pre-committed rule in mission file |
| Deadline will slip | "Extend?" | Deadline type decides (HARD/SOFT/CRITICAL_WINDOW) |

See [`prevention.md`](prevention.md) for the underlying principle.
