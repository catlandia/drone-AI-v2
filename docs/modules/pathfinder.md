# Pathfinder

**Layer 2** — path planning with obstacle avoidance.

## Files

- `modules/pathfinder/algorithms.py` — `AStarPlanner`, `RRTPlanner`, `PathPlanner`
- `modules/pathfinder/train.py` — benchmark-based grading (terminal)
- `viz/inspector_pathfinder.py` — visual inspector (top-down grid +
  start/goal/path per trial). See [inspector_ui.md](../inspector_ui.md).

## Algorithms

### A\* (primary)

Runs on a 3D voxel grid (default 2m resolution) with 26-connected neighbors.
- Fast and optimal when the graph is reasonably small
- Respects obstacle margins (0.5m default)
- Bounded to `max_iter = 5000` expansions

### RRT (fallback)

Rapidly-exploring Random Tree in continuous space.
- Used when A\* fails (too dense, too large)
- Goal-biased sampling (10% chance to sample goal directly)
- Step size 3m, max 3000 iterations

### PathPlanner (high-level)

Combines both:
1. Try A\* first
2. Fall back to RRT if A\* returns None
3. Smooth the result with 3 iterations of weighted averaging

## Why no ML?

Path planning is one of those problems where classical algorithms dominate — A\* is optimal, RRT is probabilistically complete, and there's no training data advantage. An RL-based planner might win on speed in specific distributions, but doesn't generalize as well.

The pathfinder is still **graded** because performance varies with world complexity.

## Grading

Benchmarks on 50 randomly-generated worlds with 5-25 obstacles each.

```python
PathfinderMetrics(
    path_optimality = actual_length / straight_line_distance,  # lower is better
    avoidance_rate  = fraction_without_collisions,
    planning_ms     = average_plan_time,
)
```

Scoring:
- **Optimality score:** `(2.0 - ratio) × 300` (so ratio=1.0 → 300 pts, ratio=2.0 → 0)
- **Avoidance score:** `rate × 400`
- **Speed score:** `max(0, (500 - ms)/500) × 100`

Max possible: 800 pts.

## Usage

```python
from drone_ai.simulation.world import World
from drone_ai.modules.pathfinder import PathPlanner

world = World()
world.generate_random_obstacles(10, np.random.default_rng(42))
planner = PathPlanner(world)

path = planner.plan(start=np.array([0, 0, 5]), goal=np.array([50, 50, 10]))
```

`path` is a list of `np.ndarray` waypoints.

## Future work

- Lazy A\* for faster replanning when obstacles update incrementally
- Dynamic windowing to avoid moving obstacles
- RL-enhanced heuristic (learned A\* guidance)
