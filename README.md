# Drone AI v2.0

Autonomous drone delivery system with a **4-layer AI architecture** and a **tier-based grading system** for every module.

Previously split across five repos (`drone-AI`, `drone-AI-full`, `drone-AI-pathfinder`, `drone-AI-manager`, `drone-AI-perception`). This is the unified rebuild.

---

## Architecture

| Layer | Module | Role |
|-------|--------|------|
| 1 | **manager** | Mission planning — task queue, priority scheduling, routing, battery budget |
| 2 | **pathfinder** | Path planning — A\* + RRT with smoothing, obstacle avoidance |
| 3 | **perception** | Obstacle detection — CNN (real) / noise-based (sim), Kalman tracking |
| 4 | **flycontrol** | Flight control — PPO reinforcement learning for motor commands |

Each module trains independently, is graded on the same **P→W** tier list, and is saved with a standardized filename: `{Grade} {DD-MM-YYYY} {module} v{N}.pt`.

---

## Run the app

The easiest way to try it — a visual top-down simulator with a side panel of stats, click-to-add deliveries, and speed controls.

**Windows:** double-click `run.bat` (first run installs deps, then it just launches).
**macOS / Linux:** `./run.sh`

Controls inside the app:

| Key / Mouse | Action |
|---|---|
| Left click | Add NORMAL delivery at cursor |
| Right click | Add URGENT delivery |
| Middle click | Add CRITICAL delivery |
| Space | Pause / resume |
| R | Reset drone (keeps deliveries) |
| N | New scenario (fresh obstacles) |
| C | Clear deliveries |
| O | +5 random obstacles |
| 1 – 5 | Sim speed (1×, 2×, 5×, 10×, 20×) |
| + / – | Zoom |
| Arrows | Pan |
| Q / Esc | Quit |

---

## Command-line usage

```bash
pip install -e .
drone-ai-app                          # launch the visual app
drone-ai demo                         # headless demo mission

# Train FlyControl (PPO with evolutionary population)
drone-ai train flycontrol --population 6 --ages 10 --steps 10000

# Benchmark all other modules
drone-ai train pathfinder
drone-ai train perception --grade P
drone-ai train manager --grade P

# Or train everything in sequence
drone-ai curriculum

# Run grade-mixing experiments
drone-ai experiment all-P          # perfect drone
drone-ai experiment all-F          # disaster
drone-ai experiment blind-ace      # P-flight + F-perception
drone-ai experiment list

# Sweep one module across all grades
drone-ai sweep perception --output experiments/perception_sweep.json

# Custom grade combination
python -m drone_ai.experiment custom \
    --flycontrol P --pathfinder P --perception C --manager P
```

---

## The tier system

20 grades from **P** (perfect) down to **W** (worst). Each module gets graded independently after training. See [`docs/tier_system.md`](docs/tier_system.md) for the full list and thresholds.

Grade combinations let you run experiments like:
- What does an all-P drone do?
- What happens when only perception is C-tier?
- How does the drone behave with everything at F-tier?

---

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — full system design
- [`docs/tier_system.md`](docs/tier_system.md) — grading criteria per module
- [`docs/training.md`](docs/training.md) — how training works
- [`docs/experiments.md`](docs/experiments.md) — grade-mixing experiments
- [`docs/modules/`](docs/modules/) — per-module details
- [`docs/lessons_learned.md`](docs/lessons_learned.md) — what went wrong last time

---

## What changed from v1

- Single repo, single installable package (`pip install -e .`)
- Clean module boundaries: shared `simulation/` and `grading.py`
- Grade-mixing experiments are a **first-class feature**, not an afterthought
- Realistic but simple quadrotor physics (no external sim dependency)
- Evolutionary PPO training preserved, cleaned up
- All modules use the same tier list (originally only FlyControl/Perception did)

---

## License

MIT
