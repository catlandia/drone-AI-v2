# Drone AI v2.0

Autonomous drone delivery system with an **8-layer AI architecture** and a **tier-based grading system** for every module.

Previously split across five repos (`drone-AI`, `drone-AI-full`, `drone-AI-pathfinder`, `drone-AI-manager`, `drone-AI-perception`). This is the unified rebuild.

---

## Architecture

Phase 1 layers (learn + act) and Phase 2 layers (self-improve + coordinate) ship
in the same codebase. All eight grade independently under the same P→W tier list
and get saved with a standardized filename: `{Grade} {DD-MM-YYYY} {module} v{N}.pt`.

| # | Layer | Role | Phase |
|---|---|---|---|
| 1 | **Manager** | Mission planning — task queue, priority scheduling, routing, pre-flight feasibility gates | 1 |
| 2 | **Pathfinder** | Path planning — A\* + RRT with smoothing, obstacle avoidance | 1 |
| 3 | **Perception** | Obstacle detection — split into Obstacles / Hazards / Targets / Agents sub-models | 1 |
| 4 | **FlyControl** | Flight control — PPO with BC warm-up from a PD teacher | 1 |
| 5 | **Adaptive** | Online learning of all layers + itself, guarded by a frozen warden, 20-episode rollback, and a soft-bound registry | 2 |
| 6 | **Storage of Learnings** | Per-drone field log of every Layer-5 action (kept separate from `runs.csv`) | 2 |
| 7 | **Personality** | Transferable delta artifact — weight + hparam deltas pushed through an A/B subset cohort | 2 |
| 8 | **Swarm Cooperation** | Pre-mission plan + visual mutual avoidance (Perception-Agents); zero radio after takeoff | 2 |

See [`PLAN.md`](PLAN.md) for the full design intent and [`docs/`](docs/) for per-topic deep dives.

---

## Run the app

**Windows:** double-click `run.bat` (first run installs deps, then it just launches).
**macOS / Linux:** `./run.sh`

The launcher opens with cards grouped by section:

- **FlyControl Curriculum (Layer 4)** — hover → waypoint → delivery →
  delivery_route → deployment. Each stage opens a live 3D training window
  and warm-starts from the previous stage's latest checkpoint.
- **Module Benchmarks (Layers 1, 2, 3, 5)** — each opens its own
  **visual inspector window** (top-down world views, live metric
  sidebars, step/auto-play controls) so you can actually see what
  the module does per trial. Writes a tier-named `.pt` under
  `models/<layer>/` and a row to `models/runs.csv`.
- **Phase 2 Ops (Layers 6, 7, 8) + Demo** — Storage field-log stress
  test, Personality export + sibling-transfer test, Swarm coordinator
  correctness benchmark, and the PD-controller free-fly demo. Each
  has a visual inspector; structure + thinking diagrams for the
  modules without a natural world view.

Full inspector reference: [`docs/inspector_ui.md`](docs/inspector_ui.md).

Click a card to open the **pre-launch picker**, pick a warm-start
base (`(fresh)`, `(auto — newest)`, or any specific checkpoint), then
hit Enter. Fresh launches are labeled **TEST TRAINING** in the HUD and
tagged `test` in `runs.csv` so real curriculum steps stay comparable.

### 3D training window

| Key | Action |
|---|---|
| `1` / `2` / `3` / `4` | Camera mode: Follow / Free / FPV / Top-down |
| Arrow keys | Orbit camera |
| `+` / `−` | Zoom in / out |
| `[` / `]` | Sim speed (1× → 20×) |
| `Space` | Pause |
| `T` / `H` | Toggle trail / HUD |
| `Q` / `Esc` | Close window (returns to launcher) |

At end of training the results modal waits for input — nothing auto-exits.

---

## Command-line usage

```bash
pip install -e .
drone-ai launch                       # visual launcher (cards → inspectors)
drone-ai-app                          # same thing via the GUI wrapper
drone-ai demo                         # headless demo mission

# Phase 1 — Layers 1-4
drone-ai train flycontrol --population 6 --ages 10 --steps 10000
drone-ai train pathfinder
drone-ai train perception --grade P
drone-ai train manager --grade P
drone-ai curriculum                   # everything in sequence

# Phase 2 — Layers 5-8
python -m drone_ai.modules.adaptive.train       # auto-finds latest fc ckpt
python -m drone_ai.modules.storage.train        # field-log stress test
python -m drone_ai.modules.personality.train    # export artifact from newest fc
python -m drone_ai.modules.swarm.train          # coordinator benchmark

# CLI helpers
drone-ai storage <drone-id>                         # dump a field log summary
drone-ai personality export --baseline … --proven …  # build a transferable delta
drone-ai personality inspect <path>                 # print a personality summary

# Grade-mixing experiments
drone-ai experiment all-P          # perfect drone
drone-ai experiment all-F          # disaster
drone-ai experiment blind-ace      # P-flight + F-perception
drone-ai experiment list
drone-ai sweep perception --output experiments/perception_sweep.json
```

---

## The tier system

20 grades from **P** (perfect) down to **W** (worst). Each module gets graded independently after training. See [`docs/tier_system.md`](docs/tier_system.md) for the full list and thresholds.

Grade combinations let you run experiments like:
- What does an all-P drone do?
- What happens when only perception is C-tier?
- How does the drone behave with everything at F-tier?

---

## Training quirks that matter

- **FlyControl BC warm-up.** Fresh launches pre-train the PPO actor on
  the onboard PD controller's actions before PPO takes over, then
  clamp `log_std` to −2.2 so the first stochastic rollouts don't wipe
  the learned policy. Loading a real checkpoint skips BC.
- **Hover reward has a faint 20 m gradient** on top of the steep on-target
  bonus, so PPO has a learning signal during the approach phase.
- **`runs.csv` is gitignored** — it's per-machine training history.
- **Models are gitignored too** (`models/**/*.pt`, `*.json`) — ship code,
  not checkpoints.

---

## Documentation

- [`PLAN.md`](PLAN.md) — single source of truth for every locked design decision
- [`ROADMAP.md`](ROADMAP.md) — condensed public-facing version
- [`docs/architecture.md`](docs/architecture.md) — full system design
- [`docs/phasing.md`](docs/phasing.md) — Phase 1 / 1.5 / 2 sequencing and exit criteria
- [`docs/tier_system.md`](docs/tier_system.md) — grading criteria per module
- [`docs/training.md`](docs/training.md) — how training works
- [`docs/experiments.md`](docs/experiments.md) — grade-mixing experiments
- [`docs/modules/`](docs/modules/) — per-module details (one file per layer)
- [`docs/lessons_learned.md`](docs/lessons_learned.md) — what went wrong last time

---

## What changed from v1

- Single repo, single installable package (`pip install -e .`)
- 4-layer → 8-layer architecture (Phase 2 Layers 5-8 now in code)
- Clean module boundaries: shared `simulation/` and `grading.py`
- Grade-mixing experiments are a **first-class feature**, not an afterthought
- Realistic 5" FPV physics (inertia tensor, gyro, anisotropic drag, wind, ground effect, prop wear, battery temp)
- BC warm-up via the PD controller — fresh PPO no longer tumbles the drone in a handful of steps
- Launcher state machine: MENU → PICKER → RUNNING → RESULTS (nothing auto-exits)

---

## License

MIT
