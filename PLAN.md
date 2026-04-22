# Drone AI — Master Plan

This is the single source of truth for the project's design. Every
architectural decision we've locked in lives here. If something about
the design is ambiguous, this file decides. If this file is ambiguous,
it's a bug — fix the file before writing code.

Reference only:
- `ROADMAP.md` — condensed public-facing version of this plan.
- `docs/` — per-topic deep dives (architecture, training, tiers, etc.).
- `~/.claude/projects/E--Projects-Drone-AI/memory/` — per-decision memory notes.

---

## 1. The target system

A **fully autonomous, offline delivery drone**. Not a teleop assistant,
not a cloud-connected flyer, not something a human pilots mid-mission.

The operator may be in an area where internet is jammed or deliberately
cut off. The drone has to decide on its own whether to continue, return,
abort, or land. Everything else in the architecture flows from that.

Class: 5" FPV quadrotor. Custom drones are expensive and slow to
rebuild — crashing is a real event, and the design takes that seriously.

---

## 2. Hard constraints (never compromise)

- **Offline at inference.** All models load from disk and run locally on
  the drone. No cloud APIs, no hosted inference, no remote telemetry.
- **No human in the loop.** The AI owns the mission once it takes off.
- **No satellites.** GPS is not available. The drone localises itself
  from onboard sensors alone (IMU + baro + VIO + camera/ToF).
- **Zero comms after takeoff.** Connection exists only at the base.
  Mid-mission link failure isn't a scenario — it's the default
  assumption. Jamming-resistant by construction.
- **No runtime reward editing.** Layer 5 cannot change reward structure,
  goals, or module on/off switches — prevents the AI from giving itself
  infinite points for doing nothing.

---

## 3. 8-layer architecture

| # | Layer | Purpose | Phase |
|---|-------|---------|-------|
| 1 | **Manager** | Mission planning, priority queue, routing, pre-flight feasibility gates | 1 |
| 2 | **Pathfinder** | A*/RRT path planning, replan on new obstacles | 1 |
| 3 | **Perception** | CNN + Kalman, split into sub-models | 1 |
| 4 | **FlyControl** | PPO motor control for 5" FPV quadrotor | 1 |
| 5 | **Adaptive** | Online learning of all layers including itself | 2 |
| 6 | **Storage of Learnings** | Per-drone persistent log of what Layer 5 learned | 2 |
| 7 | **Personality** | Transferable artifact — best drone's personality copied to others | 2 |
| 8 | **Swarm Cooperation** | Pre-mission plan + visual mutual avoidance, no comms after takeoff | 2 |

Every layer is gradable under the same P → W tier system.
Models are named `{Grade} {DD-MM-YYYY} {module} v{N}.pt`.

### Perception sub-models (Layer 3)

1. **Perception-Obstacles** — generic obstacle CNN + Kalman tracker.
2. **Perception-Hazards** — specific hazard classes (people, animals,
   vehicles, power lines, water).
3. **Perception-Targets** — landing zones, package drop markers,
   recipient markers.
4. **Perception-Agents** — other drones (for visual mutual avoidance
   in Layer 8).

Each sub-model grades independently and produces its own `.pt`.

**Not in scope:** camera-based best-drone recognition at the base
station. That's station-side later, via QR codes engraved into the
drone's frame and read via memory-drive extraction.

---

## 4. Phasing

- **Phase 1 (current).** Train Layers 1-4 to their best on realistic
  physics. Immediate unblocker: the pygame exit / file-reload bug.
  Then ship inertia/gyro/drag, wind, ground effect, prop wear,
  battery-temp upgrades. Perception split into sub-models.
- **Phase 1.5.** Real-life test of trained Layers 1-4.
- **Phase 2.** Build Layers 5-8. Designs are locked here; no code
  until Phase 1.5 completes successfully.

Realism over convenience: training runs on unrealistic physics do
**not** count toward Phase 1 completion.

---

## 5. Tier list (20 tiers — names locked)

P (PERFECT) → S+/S/S- (SUPREME) → A+/A/A- (ALPHA) → B+/B/B- (BETTER) →
C+/C/C- (COOL) → D+/D/D- (DELUSIONAL) → F+/F/F- (FAILURE) → W (WORST).

Defined in `src/drone_ai/grading.py` (`GRADE_ORDER`, `GRADE_NAMES`,
`GRADE_DESCRIPTIONS`). No renames.

### Run log

Every training run and every eval appends a row to `models/runs.csv`:
- `timestamp_iso`, `date` (DD-MM-YYYY), `minutes`
- `module`, `stage`, `best_score`, `avg_score`, `grade`
- `updates`, `episodes`, `run_tag`

Deadline outcomes and failure tags are included so Layer 5 has
something to study later.

### Auto-save

Training runs launched from the UI auto-save with the tier-list
filename convention: `{Grade} {DD-MM-YYYY} flycontrol v{N}.pt` in
`models/`. Version auto-increments per module. This fixes the
"files not loading after completion" bug.

---

## 6. Sensors (observation space)

The FlyControl observation only contains data a real offline drone
could sense. Each slot is annotated with its source sensor.

**Used:**
- **IMU** — orientation, angular velocity, (derived) velocity,
  acceleration.
- **Barometer** — altitude above takeoff.
- **VIO (camera + IMU)** — displacement from takeoff (drift-
  accumulating, relative only).
- **Camera / ToF** — nearest-obstacle distance, plus Perception
  inputs.
- **Battery monitor** — state-of-charge, temperature.
- **Internal state** — carrying flag.

**Not used (by design):**
- GPS (satellite — explicitly ruled out).
- Cloud maps, weather APIs, any remote data source.
- Ground-truth absolute world position.
- Radio comms mid-mission.

Planned sensor noise injection: IMU bias, VIO drift, baro drift.

---

## 7. Physics realism (Phase 1)

User rule: "the more realistic in the way of physics and situations,
the more the drones are prepared."

### Core upgrades
- Moment of inertia tensor (replace scalar).
- Gyroscopic coupling from spinning props.
- Anisotropic drag (forward ≠ sideways).
- Tighter motor torque reaction (yaw kick when collective thrust
  changes).

### Environment upgrades
- Wind + turbulence (steady field + random gusts).
- Ground effect (extra lift near the ground).
- Prop damage / motor wear (gradual efficiency loss or sudden failure).
- Battery temperature model (sag varies with cold/heat).

### Inertia as decision context
- Expose **braking distance at current velocity** to Pathfinder and
  Manager — they need it for feasibility and replan.
- Add **acceleration** (finite-differenced) to FlyControl's obs so the
  policy can feel "still decelerating, don't command full stop yet."

---

## 8. Crash model

Real FPV drones are expensive and slow to rebuild.

- **Inversion** (|roll| or |pitch| ≥ 90°) → dead. Propellers pointing
  down means the thrust vector can't recover the drone in time.
- **Hard ground impact** (|vz| > 8 m/s) → dead.
- **Soft touchdown** → bounce and settle, not a crash.
- **Crash reward = -50.** Do not soften this to "help" training. Fix
  exploration through hover-biased actor init, lowered log_std, and
  BC warm-up instead.

Layer 5 may soften the inversion bound **only after** N successful
simulated recoveries past it — never by default.

---

## 9. Goals (read-only)

- **Format:** parameterized. Each goal = task + parameters + deadline +
  deadline_type + priority + mission class.
- **Authoring:** operator writes a mission file before takeoff.
- **Lock:** stored in a read-only memory page. Layer 5's gradient
  updates have a goal-structure mask zeroing out any weight that
  touches goal parsing. Hard isolation.

Why this is load-bearing: "giving the AI ability to self change the
points is bad because it could just give itself inf points for doing
nothing." (user)

---

## 10. Mission classes

Orthogonal to priority and deadline type.

### LIFE_CRITICAL
Life-or-death delivery — adrenaline, insulin, blood, AED.

**Overrides drone safety.** A crashed drone that delivered the
adrenaline on time is a success; a safely-returned drone whose
patient died is a failure.

- Drone may fly in weather it would normally refuse.
- Drone may push soft limits without prior sim-recovery gating.
- Drone may accept crash-risk flight paths.
- Layer 5 is authorized to pick aggressive soft-bound promotions
  for this mission class.

**Still hard under LIFE_CRITICAL:**
- Goals / reward-function structure (no reward hacking).
- Module on/off switches.
- Mission-file authenticity (LIFE_CRITICAL can't be spoofed).
- Harm to bystanders — risking **the drone** is authorized, risking
  people / animals / property on the ground is not. Perception-Hazards
  still vetoes paths that endanger third parties.

### STANDARD
Everything else. Normal safety absolutes apply.

- Crash avoidance stays hard.
- Hard-impact prevention stays hard.
- Inversion base rule (Layer 5 can soften after N sim recoveries).
- Battery < 10% RTB (can be relaxed by Layer 5 tuning; lowered to 0%
  under LIFE_CRITICAL).

---

## 11. Deadlines

Each delivery declares `deadline` + `deadline_type` in the mission file.

- **HARD** — package worthless after deadline. Mid-flight infeasibility
  → abort, RTB, save battery. Log `deadline_infeasible_aborted`.
- **SOFT** — late is still valuable. Mid-flight infeasibility →
  continue, deliver late. Log `delivered_late`.
- **CRITICAL_WINDOW** — time-sensitive life/safety (often paired with
  LIFE_CRITICAL). Mid-flight infeasibility → continue, push the flight
  envelope harder. Layer 5 may use softened limits. Log outcome.

**Pre-flight gate:** Manager rejects if `ETA + margin > deadline`.
Margin default 15%, tunable by Layer 5 per drone.

**Mid-flight:** continuous ETA recomputation every N seconds.

**Logging:** every deadline outcome tagged in `runs.csv` (on-time /
late / aborted). Layer 5 studies upstream causes (margin too thin,
battery curve stale, etc.) — not the runtime tradeoff.

**What Layer 5 CAN tune:** pre-flight margin, CRITICAL_WINDOW
aggression level.
**What Layer 5 CANNOT touch:** whether deadlines exist, deadline
values, deadline types assigned per delivery, the HARD-missed =
abort rule.

---

## 12. Prevention principle

Runtime "conflicts" (battery vs delivery, priority vs efficiency,
replan vs deadline) are **planning failures**, not runtime problems.

User insight: "the solution to the problem of batteries is actually
just to prevent that from happening entirely. if somehow it still
happened, then it means we have failed at some point."

- Manager pre-flight rejects missions that can't fit with margin.
- Mission files pre-commit rules for known-unknown events
  (swarm-mate-failed, can't-land-at-base).
- Continuous re-estimation triggers RTB **early**, never at crisis.
- When a conflict fires anyway → `runs.csv` gets a **failure tag**,
  Layer 5 studies the upstream cause.

We do **not** build a general runtime conflict resolver. That's too
close to "letting Layer 5 change goals" and opens a reward-hacking
door.

---

## 13. Comms rules

**Zero radio comms after takeoff.** Connection exists only at base.

User: "conection is back only at the base. it will help them survive
signal blockage if some jackass thinks he is a cool hacker."

Trust lives in:

**Pre-takeoff handshake**
- Mission file authenticated (signed / hashed — method TBD).
- Base station identity verified by drone.
- Drone runs self-integrity check.
- Any failure → refuse takeoff.

**Base reauthentication on return**
- Visual landing-pad pattern + engraved QR must match the drone's
  expected base.
- Decoy base → don't land.

**Can't-land-at-base fallback ladder**
1. No loiter at base (save battery).
2. Search for alternate landing zone via Perception-Targets.
3. Keep searching as long as battery allows — no early give-up.
4. Battery critical → slow controlled vertical descent wherever the
   drone currently is.

Any code that would "just ping the operator" or "check in with base"
mid-flight is wrong by construction. Work must be done in pre-flight
gates or post-landing analysis.

---

## 14. Swarm rules (Layer 8)

- All comms happen at base only. No radio post-takeoff.
- Pre-mission plan distributed to all drones before takeoff — routes,
  roles, mutual-avoidance handshakes.
- After takeoff: visual mutual avoidance via Perception-Agents.
- A multi-drone delivery (e.g., 10-drone payload) coordinates entirely
  via the pre-plan + vision once in the air.

**Swarm-mate-failed policy:**
Nearest surviving drone diverts to visually mark the failure point
(unless it's itself on LIFE_CRITICAL). All others continue their
missions. Marker location is logged locally; ground crew retrieves
after fleet returns.

---

## 15. Layer 5 — Adaptive module (full rules)

### Scope of change
**CAN modify:**
- Neural-network weights of all layers, including itself.
- Hyperparameters: learning rates, exploration noise, action bounds,
  battery thresholds, tilt bounds, hover throttle, action clipping.

**CANNOT modify:**
- Network architecture, source code.
- Reward function structure.
- Goals or goal parsing (goal-structure mask zeros the gradients).
- Module on/off switches.

### Soft vs hard limits

**Softenable** (Layer 5 can push past after N successful sim
recoveries with no crash):
- Tilt bound (90° inversion rule — e.g., aggressive dive recovery).
- Hover throttle assumption.
- Action clipping.
- Battery "land now" threshold (small adjustments — this is literally
  why it's the *Adaptation* layer).

**Hard, never touchable:**
- Ground-impact crash detection (ground hit always ends the episode).
- Reward function structure.
- Goals.
- Module toggles.

### Update timing (hybrid)

- **Mid-flight:** FlyControl only. Fast reflex layer, a bad update
  is recoverable within milliseconds.
- **Landed + idle only:** Manager, Pathfinder, Perception, and
  Layer 5 itself. A bad high-level update ruins an entire mission,
  so it waits for a safe boundary.

### Warden (non-negotiable)

A frozen copy of the original reward function lives in read-only
code. Every proposed weight update is scored by the warden (not by
Layer 5's own metric) on N simulated episodes before being accepted.
If warden score drops, update is rejected. Prevents Layer 5 from
slowly drifting its internal metric while the real job gets worse.

### Rollback

Rolling average of the **last 20 episodes**. If it drops below the
previous best, roll back to the pre-update checkpoint. Single-episode
failures do not trigger rollback (too sensitive to wind / noise).

### Fleet propagation (A/B subset)

1. Layer 5 discovers an improvement on a single drone.
2. Layer 6 captures it; Layer 7 encapsulates it into a transferable
   personality artifact.
3. On docking, the improvement is pushed to a **~20% subset** of the
   fleet (experimental cohort), not all drones.
4. After **M ≈ 10 missions**, compare subset performance against
   control cohort.
5. If the subset outperforms → push fleet-wide.
6. If not → discard. "It was luck, not skill."

### Deployment modes
- **Train + Deploy:** Adaptive on, drone learns in field.
- **Pre-trained Deploy:** Adaptive disabled, frozen weights,
  predictable, smaller compute.

`DroneAI(adaptive=True, ...)` opts in; default off.

---

## 16. Layer 6 — Storage of Learnings

- Per-drone persistent log of everything Layer 5 learned during flight.
- Uploaded to base station on docking.
- Basis for (a) Layer 7 personality selection and (b) the fleet A/B
  propagation test.

---

## 17. Layer 7 — Personality

- Transferable artifact encapsulating a drone's learned "personality"
  (tuned weights + hyperparameters subset).
- **Phase-2-early:** manual copy (operator designates best drone).
- **Phase-2-later:** base station auto-selects via Storage metrics +
  QR identification, pushes per the A/B-subset rule.

---

## 18. Open work (Phase 1)

- ~~Fix pygame exit / file-reload bug~~ **DONE** (auto-save added,
  sys.exit code fixed).
- ~~Phase 1 physics upgrades~~ **DONE** (inertia tensor, gyro, drag,
  wind, ground effect, prop wear, battery temp, braking distance,
  linear acceleration in obs).
- ~~Perception split into sub-models~~ **DONE** (Obstacles/Hazards/
  Targets/Agents — Obstacles uses the noise model, the other three
  are Phase-1 stubs awaiting the CNN + world-annotation upgrade).
- ~~**BC warm-up**~~ **DONE** (`modules/flycontrol/pd_controller.py`
  + `PPOAgent.bc_warmup`; TrainerUI auto-enables on fresh launches;
  post-BC `log_std` clamped to −2.2).
- ~~Wider hover reward gradient~~ **DONE** (faint 20 m linear on top
  of the steep on-target bonus so PPO gets signal during approach).
- Sensor noise injection — IMU bias, VIO drift, baro drift.
- Full curriculum run across Layers 1-4, producing graded checkpoints.

## 18a. Phase 2 implementation status

Originally gated on Phase 1.5, but all four Phase 2 layers now have
code + a launcher benchmark so the UI covers every layer.

- **Layer 5 — Adaptive.** `AdaptiveLearner` with frozen `Warden`,
  20-episode `RollbackMonitor`, and `SoftBoundRegistry` (N = 50
  recoveries required before a soft bound may be softened). Mid-
  flight updates restricted to FlyControl; landed-only for the rest.
- **Layer 6 — Storage of Learnings.** Append-only per-drone JSONL at
  `logs/storage/drone_{id}.jsonl` with `UpdateRecord` + `MissionRecord`
  rows. Wired into `DroneAI` so missions auto-log outcomes.
- **Layer 7 — Personality.** Transferable delta artifact;
  `export_personality` / `apply_personality`; manual
  `select_best_drone` ranker.
- **Layer 8 — Swarm Cooperation.** `SwarmPlan` + `SwarmCoordinator`
  with visual mutual avoidance via Perception-Agents, LIFE_CRITICAL-
  exempt swarm-mate-failed contingency, zero post-takeoff radio.

Real field deployment still waits on Phase 1.5.

## 19. Open tunables (Phase 2)

Proposed defaults; Layer 5 can tune later.

- **N** for soft-limit promotion (sim recoveries required before
  Layer 5 can push past a soft bound). Default: **50**.
- **Subset size** for fleet A/B test. Default: **~20%** of fleet.
- **M** missions per A/B trial. Default: **10**.
- **Pre-flight deadline margin.** Default: **15%**.

---

## 20. Verification checklist

- **Bug fix:** Launch via `py -m drone_ai.viz.launcher`, start a
  training stage, close child window via X. Launcher stays alive and
  reusable. `models/{Grade} {DD-MM-YYYY} flycontrol v{N}.pt` is on
  disk. `models/runs.csv` has a matching row. Close launcher via X.
  No traceback.
- **Physics upgrades:** Short FlyControl training run appears in
  `runs.csv` with an updated-physics tag. Compared against pre-upgrade
  run of same seed/length, grading still discriminates.
- **Perception split:** Each sub-module trains and grades
  independently; each produces a tier-named `.pt`.

---

## 21. Out of scope (for now)

- Cross-drone runtime sync protocol details.
- Station-side best-drone identification implementation (QR +
  memory-drive read — deferred).
- CNN-backed Perception sub-models (Phase 1.5 hardware work).
- Mid-flight Layer-5 updates on non-FlyControl layers (still
  landed-only by design).
