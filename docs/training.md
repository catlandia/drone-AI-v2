# Training

Each module has its own training pipeline. They can be run individually or all at once via `curriculum`.

## FlyControl — evolutionary PPO

FlyControl uses a **population of 6 PPO agents** trained in parallel with evolutionary selection. This is inherited from v1 and survived into v2 because it works well.

### Algorithm

1. Spawn a population of `N` agents (default 6) with random weights
2. For each curriculum stage (Hover → Delivery → Route → Deployment):
   1. For each *age* (default 10):
      - All agents train in parallel for `steps_per_age` steps
      - After each age, top 2 agents survive; rest are mutated copies
3. After round 1 (all stages), keep top 2 agents, re-expand to population size
4. Run round 2 (all stages again) with survivors + fresh mutations
5. Select best final agent, evaluate on all stages
6. Compute weighted final score → grade → save

### Curriculum

| Stage | Task | Difficulty | Domain randomization |
|-------|------|-----------:|:--------------------:|
| Hover     | maintain position | 0.3 | off |
| Delivery  | pickup + drop | 0.5 | off |
| Route     | multi-waypoint | 0.6 | off |
| Deployment| full mission | 1.0 | **on** |

Domain randomization (mass, gravity variance) kicks in at the last stage to improve real-world transfer.

### Command

```bash
drone-ai train flycontrol --population 6 --ages 10 --steps 10000
```

Per-age budget: `population_size × steps_per_age` = 60 000 env steps.
Total budget for defaults: `6 × 10 × 4 × 2 × 10 000 ≈ 4.8M` steps.

## Pathfinder — algorithmic benchmark

Pathfinder uses A\* on a voxel grid (fast, optimal) with RRT fallback (continuous space). There is no neural network to train — "training" means **benchmarking** across randomly-generated worlds and assigning a grade.

### Metrics collected

- Path optimality: `actual_length / straight_line_distance`
- Collision-free rate
- Planning time (ms)

```bash
drone-ai train pathfinder --trials 50
```

## Perception — noise-calibrated simulation

In simulation, Perception uses **grade-parameterized noise** on top of the ground-truth obstacle list:

- Higher grade → higher detection probability, lower position noise, fewer false positives
- Lower grade → more missed detections, more drift, more false positives

This is the mechanism that lets you simulate "C-grade perception" without training a CNN. When you eventually plug in a real CNN, replace the noise model with the CNN's actual error distribution — the rest of the system is unchanged.

```bash
drone-ai train perception --grade P --trials 100
```

## Manager — quality-parameterized heuristic

Same approach as Perception: grade maps to a quality factor `q ∈ [0, 1]` that controls how optimal the scheduling decisions are. An RL variant can replace this later.

```bash
drone-ai train manager --grade P --trials 20
```

## Full curriculum

Runs all four in sequence with sensible defaults:

```bash
drone-ai curriculum --population 6 --ages 10 --steps 10000
```

## Seeds and reproducibility

Every training command accepts `--seed`. Defaults to 42.
