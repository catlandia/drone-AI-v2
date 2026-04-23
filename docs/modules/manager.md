# Manager

**Layer 1** — mission planning, task queue, and battery management.

## Files

- `modules/manager/planner.py` — `MissionPlanner`, `DeliveryRequest`
- `modules/manager/train.py` — benchmarking + grading (terminal)
- `viz/inspector_manager.py` — visual inspector showing the mission
  queue, priority colours, and the chosen visit order per trial.
  See [inspector_ui.md](../inspector_ui.md).

## Responsibilities

1. Accept delivery requests (target, priority, weight)
2. Decide which delivery to execute next
3. Decide when to return to base (battery budget)
4. Track completed/failed deliveries
5. Expose mission summary stats

## Delivery selection

`select_next(current_position)` returns the next `DeliveryRequest` to execute.

### Optimal policy (P-grade)

Pick the delivery that maximizes `priority_value × 100 / distance`:
- High priority wins ties
- Closer targets preferred all else equal
- Equivalent to greedy nearest-neighbor with priority weighting

### Grade-quality mechanism

Each grade maps to a **quality factor** `q ∈ [0, 1]`:

| Grade | Quality |
|-------|--------:|
| P   | 1.00 |
| A   | 0.83 |
| B   | 0.69 |
| C   | 0.52 |
| D   | 0.30 |
| F   | 0.08 |
| W   | 0.00 |

Behavior is interpolated:
- `q > 0.9`: always optimal
- `q > 0.5`: optimal with probability `q`, random otherwise
- `q ≤ 0.5`: mostly random, occasionally optimal

## Battery management

`_should_return_to_base()` estimates `remaining_distance / battery_capacity` and returns true if the drone can't make the next delivery + return. Quality affects this:
- High quality: accurate estimation, reliable RTB
- Low quality: noisy estimation, may return too early or run out mid-flight
- `q < 0.3`: ignores battery entirely (drone will die mid-mission)

## Grading

`ManagerMetrics`:

```python
ManagerMetrics(
    completion_rate     = fraction_delivered,
    distance_efficiency = optimal_tsp / actual_distance,
    priority_score      = fraction_respecting_priority,
    battery_waste       = 1 - distance_efficiency,
)
```

Scoring:
- Completion: 400 pts max
- Efficiency: 200 pts max
- Priority adherence: 150 pts max
- Battery conservation: 50 pts max

## Future work

- Replace quality-factor heuristic with an actual RL scheduler
- Multi-drone fleet coordination
- Deadline-aware scheduling (priorities with hard time limits)
- Weight-aware routing (heavy packages cost more battery)

## Usage

```python
from drone_ai.modules.manager import MissionPlanner
from drone_ai.modules.manager.planner import Priority

mgr = MissionPlanner(base_position=np.zeros(3), grade="P")
mgr.add_delivery(np.array([50, 30, 0]), Priority.URGENT)
mgr.add_delivery(np.array([-20, 40, 0]), Priority.CRITICAL)

chosen = mgr.select_next(current_drone_position)
# ... fly there ...
mgr.complete_current(success=True)
```
