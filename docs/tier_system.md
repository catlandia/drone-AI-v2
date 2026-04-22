# Tier System

Every AI module is graded on the same **P → W** scale. The grade is computed from module-specific metrics and encoded into the model filename.

## The 20 grades

| Grade | Name | Description |
|-------|------|-------------|
| **P**   | PERFECT      | Flawless — production ready |
| **S+**  | SUPREME+     | Near perfect performance |
| **S**   | SUPREME      | Outstanding results |
| **S-**  | SUPREME-     | Excellent, almost supreme |
| **A+**  | ALPHA+       | Top tier performer |
| **A**   | ALPHA        | Dominant performance |
| **A-**  | ALPHA-       | Strong performance |
| **B+**  | BETTER+      | Impressive, above average |
| **B**   | BETTER       | Solid performance |
| **B-**  | BETTER-      | Getting there |
| **C+**  | COOL+        | Functional with issues |
| **C**   | COOL         | Minimal viable |
| **C-**  | COOL-        | Below expectations |
| **D+**  | DELUSIONAL+  | Poor but trying |
| **D**   | DELUSIONAL   | Very poor |
| **D-**  | DELUSIONAL-  | Barely functional |
| **F+**  | FAILURE+     | Failed but tried |
| **F**   | FAILURE      | Complete failure |
| **F-**  | FAILURE-     | Spectacular failure |
| **W**   | WORST        | Does not work at all |

## Model filename format

```
{Grade} {DD-MM-YYYY} {module} v{N}.pt
```

Examples:
- `P 20-04-2026 flycontrol v1.pt`
- `A+ 20-04-2026 perception v3.pt`
- `C- 21-04-2026 manager v2.pt`

Each `.pt` file is paired with a `.json` that stores the full metrics.

FlyControl checkpoints are stored per curriculum stage:
`models/flycontrol/{stage}/{filename}`. See
[training.md](training.md#ui-curriculum-chain-warm-start).

## Run log schema (`models/runs.csv`)

Every training or evaluation run appends one row. Columns (in order):

| Column | Type | Notes |
|--------|------|-------|
| `timestamp_iso` | ISO 8601 | seconds precision, wall clock |
| `date`          | `DD-MM-YYYY` | matches the model filename convention |
| `minutes`       | float  | wall-clock training duration, 1 decimal |
| `module`        | string | `flycontrol` / `pathfinder` / `perception` / `manager` / `adaptive` |
| `stage`         | string | `hover` / `waypoint` / `delivery` / `delivery_route` / `deployment` / `eval` |
| `best_score`    | float  | best numeric score observed in the run |
| `avg_score`     | float  | mean score across the run |
| `grade`         | string | letter tier (P..W) derived from `best_score` |
| `updates`       | int    | PPO / gradient updates (0 for pure eval) |
| `episodes`      | int    | episodes completed |
| `run_tag`       | string | free-form tag for seed / config distinction |

Appended by `drone_ai.grading.RunLogger`; consumed by the launcher's
recent-runs strip. Schema constant: `RUN_LOG_FIELDS` in `grading.py`.

## Per-module criteria

### FlyControl

Weighted sum of evaluation scores across 4 curriculum tasks.

| Task | Weight |
|------|--------|
| Hover | 15% |
| Delivery | 25% |
| Delivery Route | 30% |
| Deployment Ready | 30% |

Score → grade thresholds: P ≥ 800, S+ ≥ 700, S ≥ 600, ... F ≥ 0, W < -50.

### Pathfinder

| Metric | Weight | Notes |
|--------|--------|-------|
| Path optimality | 300 pts | `actual / optimal`; 1.0 = perfect |
| Collision-free rate | 400 pts | fraction of paths with zero collisions |
| Planning speed | 100 pts | sub-500ms is ideal |

### Perception

| Metric | Effect |
|--------|--------|
| Detection accuracy | +4.0 per % |
| False positive rate | −2.0 per % |
| Position error | −20.0 per m (capped) |
| FPS | +5.0 per frame (capped at 100) |

### Manager

| Metric | Weight |
|--------|--------|
| Completion rate | 400 pts |
| Distance efficiency | 200 pts |
| Priority adherence | 150 pts |
| Battery conservation | 50 pts |

## Grade comparison

`ModelGrader.compare(g1, g2)` returns:
- `1` if `g1` is better
- `-1` if `g1` is worse
- `0` if equal

Lower index in `GRADE_ORDER` = better grade.
