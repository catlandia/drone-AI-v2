# Visual inspectors — one UI per module

Every module used to print to the terminal. Benchmarks were
"run the command, read the numbers." That's useless for
understanding *what* the module is actually doing. Every card in the
launcher now opens a **visual inspector** instead. You watch the
module think, trial by trial, and a sidebar tracks the live metrics.

## Shared controls (every inspector)

| Key | Action |
|-----|--------|
| **Space**    | Toggle auto-play / pause |
| **→** / **N**| One step (pauses auto-play) |
| **+** / **−**| Speed up / slow down auto-play |
| **Esc** / **Q** | Close the inspector and return to the launcher |
| Click / key (on results modal) | Dismiss the results screen |

Every inspector has:
- A title bar with current trial index and auto/step mode.
- A main view pane (top-down world or pipeline diagram).
- A sidebar with live metrics.
- A footer with the keyboard hint.
- A final-results modal with the letter grade, raw metrics, and the
  `.pt` save path.

## Per-module inspectors

### Pathfinder (`viz/inspector_pathfinder.py`)
- **What you see:** top-down view of each random test world.
  Obstacles are grey boxes; **start = green**, **goal = red**, the
  planned path is a **blue polyline** with dots at every waypoint.
  A thin dim-red line means the planner returned nothing (no path).
- **Sidebar:** obstacles this trial, straight-line distance vs
  actual path length, path clearance (collision-free?), planning
  time, running averages of optimality ratio + plan time +
  collision-free rate.
- **What it's actually running:** the same A\* + RRT benchmark as
  `modules/pathfinder/train.py`, one trial per Space-press.

### Perception (`viz/inspector_perception.py`)
- **What you see:** a top-down snapshot at each drone position.
  The **green dot** is the drone; the **blue translucent disc** is
  its detection range. Real obstacles within range appear as
  **green circles**; obstacles the model *missed* turn **orange**.
  Every detection returns a **blue filled dot**; a yellow line
  connects a detection to the ground-truth it matched (visual
  position error). **Red X** = false positive.
- **Sidebar:** per-frame TP/FP/miss counts, cumulative detection
  accuracy %, FP rate %, mean position error, estimated FPS.
- **Grade story:** set the card to grade C- and you'll see the
  misses + drift grow visibly.

### Manager (`viz/inspector_manager.py`)
- **What you see:** top-down with the **cyan BASE** square and six
  delivery requests placed randomly, colored by priority (N=Normal
  blue, U=Urgent amber, C=Critical red). Once the scenario runs,
  the chosen visit order is drawn as a **polyline** through the
  pickups, with step numbers labeling the order.
- **Sidebar:** completed/total this trial, actual vs optimal
  distance, trip efficiency, priority adherence %, plus cumulative
  averages.
- **What it's catching:** a broken manager that ignores priority
  or takes wildly circuitous routes is immediately visible.

### Swarm (`viz/inspector_swarm.py`)
- **What you see:** a top-down scene with N drones (**green** =
  standard, **red** = LIFE_CRITICAL). The "self" drone (the one the
  coordinator decides for) has a white ring around it. A **blue
  translucent disc** shows the avoidance radius (8 m). An
  **orange dot** is the incoming contact; the line from self →
  contact shows the geometry. A colored ring around self encodes
  the coordinator's chosen action:
  - grey = CONTINUE
  - sky blue = BANK_RIGHT
  - green = CLIMB
  - amber = YIELD
  - magenta = DIVERT_TO_MARK
- **Top-left overlay** prints `action: X   expected: Y` in green
  (match) or red (mismatch) so you can see the coordinator fail in
  real time.

### Storage (`viz/inspector_storage.py`)
- **What you see:** the diagram pane shows the 5-step stress
  pipeline (write → round-trip → inject malformed → truncate →
  isolation). The currently-active box is highlighted; passed
  stages turn green, failures turn red.
- **Right pane:** "Current thinking" (stage name, missions
  written, bytes trimmed, running check count) and an event log
  of every check as it runs.
- **Why it's interesting:** a reader regression shows up as a
  specific check turning red — not a stdout blob you have to diff.

### Personality (`viz/inspector_personality.py`)
- **Diagram:** 5-step delta-transfer pipeline. The bottom pane is
  a **per-sibling bar chart of residuals** — green < 0.5 (good
  transfer), amber < 1.0, red ≥ 1.0 (delta didn't survive the
  sibling noise). You can literally watch the bars appear as each
  sibling is evaluated.
- **Sidebar:** trivial residual (sanity), mean/max sibling
  residual, estimated score live.

### Adaptive (`viz/inspector_adaptive.py`)
- **Diagram:** 5-step pipeline — load baseline → perturb env →
  baseline episodes → adapted episodes → score delta.
- **Bottom pane:** side-by-side **bar charts** of per-episode
  rewards: baseline (frozen, grey) vs adapted (online-tuned,
  green). Watch the adapted bars climb — or fail to — against the
  baseline.
- **Sidebar:** episode counters, means, delta.

## Common architecture

All inspectors share:

- `viz/inspector_common.py` — `InspectorBase` (pygame window,
  controls, sidebar, final modal), `TopDownProjector` (world↔screen
  projection), `RunningStats` (rolling aggregates), `draw_grid()`.
- `viz/inspector_structure.py` — `StructureInspector` subclass that
  adds a pipeline-diagram pane + current-thinking panel + event
  log. Used by Storage / Personality / Adaptive.

Each per-module file is a thin subclass that implements `setup()`
→ `step()` → `render()` + `sidebar_lines()` + `final_summary()`.
The underlying benchmark logic is unchanged — the same `.pt` files,
the same `runs.csv` rows, the same graders.

## Launcher wiring

`viz/launcher.py::_run_card()` dispatches each card (except
FlyControl, which has its own full 3D trainer) to the matching
`run_*_inspector()` helper. The launcher tears down its own pygame
window before the inspector runs, then re-creates it when the
inspector closes — this is the same pattern the FlyControl trainer
has always used.

The terminal entry points (`modules/*/train.py::main`) are
unchanged — they still print to stdout for batch / CI use. The
interactive launcher just no longer goes through them.
