# Quickstart

## 1. Install

```bash
cd "E:\Projects\Drone AI"
pip install -e .
```

Optional extras:
```bash
pip install -e ".[viz]"     # pygame + matplotlib for rendering
pip install -e ".[train]"   # tensorboard + stable-baselines3
pip install -e ".[all]"     # everything
```

## 2. Open the launcher (recommended first step)

Every module has a visual inspector now — no terminal staring.
Launch the menu:

```bash
drone-ai launch     # or: python -m drone_ai.viz.launcher
```

Pick any card (Pathfinder, Perception, Manager, Swarm, Storage,
Personality, Adaptive, or a FlyControl stage). The inspector opens
in its own window with top-down views, pipeline diagrams, live
metrics, and pass/fail indicators. Keyboard: **Space** play/pause,
**→** step, **+/−** speed, **Esc** close. Details in
[inspector_ui.md](inspector_ui.md).

For FlyControl cards specifically, the launcher menu also accepts:

- **`+ / −`** — adjust the update budget for the run
- **`P`** — cycle the population (1 → 2 → 4 → 6 → 8 → 12). With
  `population > 1`, that many drones train in parallel inside one
  window; the best policy is saved at the end. See
  [modules/flycontrol.md](modules/flycontrol.md#live-population-mode-in-the-launcher)
  for details.

## 3. Run the demo

No training required — this uses an untrained agent + PD controller fallback:

```bash
drone-ai demo --verbose
```

## 3. Train FlyControl

Short training run to produce a first graded model:

```bash
drone-ai train flycontrol --population 4 --ages 5 --steps 3000
```

This takes ~5 minutes on CPU, under 1 minute on GPU. Result saved to
`models/flycontrol/{Grade} {DATE} flycontrol v1.pt`.

For a serious run:
```bash
drone-ai train flycontrol --population 6 --ages 15 --steps 15000
```

## 4. Benchmark other modules

```bash
drone-ai train pathfinder
drone-ai train perception --grade P
drone-ai train manager   --grade P
```

Each produces a `.pt` + `.json` under `models/{module}/`.

## 5. Run your first experiment

```bash
drone-ai experiment all-P --model models/flycontrol/*.pt
drone-ai experiment all-F --model models/flycontrol/*.pt
drone-ai experiment blind-ace --model models/flycontrol/*.pt
```

## 6. Sweep perception across all grades

```bash
drone-ai sweep perception --trials 5 --output experiments/sweep.json
```

Inspect `experiments/sweep.json` to see how completion rate and crash rate degrade as perception quality drops.

## Common flags

| Flag | Meaning |
|------|---------|
| `--seed` | RNG seed (default 42) |
| `--quiet` | suppress progress output |
| `--output` | save results to a JSON file |
| `--trials` | number of trials for experiments |
| `--deliveries` | deliveries per trial |

## Python API

```python
from drone_ai import DroneAI
from drone_ai.drone import GradeConfig
from drone_ai.modules.manager.planner import Priority

drone = DroneAI(
    grades=GradeConfig(flycontrol="P", pathfinder="P",
                       perception="C", manager="P"),
    flycontrol_model="models/flycontrol/P 20-04-2026 flycontrol v1.pt",
)
drone.reset()
drone.add_delivery([30, 20, 0], Priority.URGENT)
summary = drone.run(max_steps=10000, verbose=True)
print(summary)
```
