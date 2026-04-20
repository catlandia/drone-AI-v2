# Lessons learned from v1

The previous attempt was split across 5 repos:
- `drone-AI` (flycontrol)
- `drone-AI-manager`
- `drone-AI-pathfinder`
- `drone-AI-perception`
- `drone-AI-full` (integration)

Root cause of failure is unclear, but several structural issues contributed. This doc records them so we don't repeat them.

## What went wrong

### 1. Fragmented repos made integration painful
Every module had its own `src/drone_ai/` package, its own `pyproject.toml`, its own `environment.py`. When `drone-AI-full` tried to import from all four, you had to pip-install each separately and pray that the versions lined up. They often didn't.

**Fix (v2):** One repo, one installable package, one `simulation/` module shared by everyone.

### 2. Duplicated logic drifted apart
Each module had its own copy of `environment.py`, `learning_sequence.py`, observation vectors — some 19D, some 31D, some inconsistent with each other. When one changed, the others silently broke.

**Fix (v2):** Shared `simulation/physics.py` and `simulation/world.py`. The observation vector is defined once in `modules/flycontrol/environment.py` and re-used by the integration layer.

### 3. No unified grading across modules
The original tier system (P → W) existed in FlyControl's `learning_sequence.py` as a score-to-grade table, and in Perception's `grading.py` as threshold-based grading. Pathfinder and Manager had no grading at all.

**Fix (v2):** `grading.py` at the top level defines the grade order, names, file naming, and per-module scoring functions. Every module uses the same `ModelGrader`.

### 4. Integration was an afterthought
`drone-AI-full` existed but was clearly bolted on. Module APIs weren't designed for orchestration — `DroneAI` had to wrap every call in try/except and re-normalize observations.

**Fix (v2):** `DroneAI` (in `drone.py`) is a first-class citizen. It was designed alongside the modules, not after.

### 5. No experiment framework
The tier lists were produced, but there was no way to say "what if I plug in this P-flycontrol with this C-perception and see what happens." Every experiment was manual.

**Fix (v2):** `experiment.py` is the whole point. Presets, custom combos, and full sweeps are one command each.

### 6. Training scripts produced models, but deployment was separate
FlyControl's `evaluate.py` had hardware interfaces (Crazyflie, Tello). But running a trained model through the full integrated system was a different code path entirely.

**Fix (v2):** Same code runs in training, eval, and integrated mode. `DroneAI.run()` is how you test a mission — load any model, any grade, same interface.

## What's still undone

- **Real CNN for Perception.** v2 uses noise-parameterized perception (adequate for experiments, not for real-world). Add a proper CNN + depth estimator when you need real-camera input.
- **Real hardware bridges.** v1 had Crazyflie and Tello interfaces. v2 dropped them for clarity. Re-add them as `drone_ai/hardware/` when ready.
- **Visualization.** v1 had a pygame renderer. v2 has a stub but no implementation — add `drone_ai/viz.py` with pygame when needed.
- **More sophisticated Manager.** Current manager is a greedy nearest-neighbor + priority heuristic. Could be replaced with an RL scheduler later.

## Guidance for future changes

- **Don't split into separate packages again.** If a module gets too big, use subpackages (`modules/flycontrol/agent/`, `modules/flycontrol/env/`) — not separate repos.
- **Keep the grading system consistent.** Any new module must use `ModelGrader` and the same file naming convention.
- **Experiment first, optimize later.** If you can't run the combo in `experiment.py`, the abstraction is wrong.
