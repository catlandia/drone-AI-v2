# Perception

**Layer 3** — obstacle detection and tracking.

## Sub-models

Perception is split into four sub-models, each graded independently
under the P→W tier system:

| Sub-model | File | Status | Purpose |
|-----------|------|--------|---------|
| Obstacles | `obstacles.py` | **landed** (wraps the shared noise model) | Generic obstacle detection + Kalman tracking |
| Hazards   | `hazards.py`   | Phase 1 stub | Person / animal / vehicle / power line / water / other danger |
| Targets   | `targets.py`   | Phase 1 stub | Base pad / drop marker / recipient marker / alt landing zone |
| Agents    | `agents.py`    | Phase 1 stub | Other drones — Layer 8 visual mutual avoidance |

Naming convention for saved checkpoints:
`{Grade} {DD-MM-YYYY} perception-{kind} v{N}.pt`. Each sub-model has
its own training curriculum and its own grade in `models/runs.csv`.

The stubs return empty detection lists on Phase 1 — they exist so the
orchestrator can import and call them today, and so their grading
hooks are wired before Layer 8 lands.

## Files

- `modules/perception/detector.py` — shared noise model, `Detection` dataclass
- `modules/perception/tracker.py` — `ObjectTracker` (Kalman filter)
- `modules/perception/obstacles.py` — Perception-Obstacles (active)
- `modules/perception/hazards.py` — Perception-Hazards (stub)
- `modules/perception/targets.py` — Perception-Targets (stub)
- `modules/perception/agents.py` — Perception-Agents (stub, for Layer 8)
- `viz/inspector_perception.py` — visual inspector that shows
  detections vs ground-truth, false positives, and misses per frame.
  See [inspector_ui.md](../inspector_ui.md). The future CNN swap-in
  is sketched in [future_image_training.md](../future_image_training.md).

## Simulation vs real

### Simulation mode (current)

`PerceptionAI` uses the ground-truth obstacle list and adds **grade-parameterized noise**:

| Grade | Detection prob | Pos noise (m) | FP rate |
|-------|---------------:|--------------:|--------:|
| P   | 0.97 | 0.3 | 0.01 |
| S   | 0.92 | 0.5 | 0.03 |
| A   | 0.84 | 1.0 | 0.06 |
| B   | 0.76 | 1.6 | 0.09 |
| C   | 0.65 | 3.0 | 0.14 |
| D   | 0.50 | 4.5 | 0.20 |
| F   | 0.30 | 7.0 | 0.28 |
| W   | 0.05 | 15.0 | 0.50 |

Full table in `detector.py`'s `_GRADE_PARAMS`.

**Why this approach?** Training an actual CNN on synthetic drone imagery is a multi-day job by itself. The noise model captures what the CNN *would* produce — miss rate, localization error, false positive rate — without needing the model. The rest of the pipeline is agnostic.

### Real hardware (future)

Replace `PerceptionAI.detect()` with CNN inference:

```python
def detect(self, drone_position, world=None):
    # Capture camera frame
    frame = self.camera.get_frame()
    # Run CNN inference
    boxes = self.cnn.infer(frame)
    # Depth estimation
    distances = self.depth_model.infer(frame, boxes)
    # Transform to world coordinates
    return self._project_to_world(boxes, distances, drone_position)
```

The `Detection` dataclass and everything downstream stays the same.

## Tracker

`ObjectTracker` associates detections across frames:

- **Constant-velocity Kalman filter** per track (6D state: position + velocity)
- **Nearest-neighbor matching** with a 5m gate
- Tracks confirmed after 2 hits, dropped after 10 misses
- Returns only confirmed tracks

## Grading

`PerceptionMetrics`:

```python
PerceptionMetrics(
    detection_accuracy  = %_true_positives,
    false_positive_rate = %_false_positives,
    position_error      = avg_error_m,
    fps                 = inference_fps,
)
```

Score formula:
```
score = acc*4 - fp*2 - min(err*20, 200) + min(fps*5, 100)
```

Max ≈ 700, which maps to **S+** or better.

## Usage

```python
from drone_ai.modules.perception import PerceptionAI
from drone_ai.simulation.world import World

world = World()  # populate with obstacles
perception = PerceptionAI(grade="A")  # or "P", "F", etc.

detections = perception.detect(drone_position, world)
nearby_obstacles = perception.detections_to_obstacles(detections)
```
