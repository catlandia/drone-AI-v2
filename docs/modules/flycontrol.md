# FlyControl

**Layer 4** — low-level motor control via PPO reinforcement learning.

## Files

- `modules/flycontrol/environment.py` — Gym environment (`FlyControlEnv`)
- `modules/flycontrol/agent.py` — PPO actor-critic + rollout buffer
- `modules/flycontrol/train.py` — evolutionary training pipeline

## Environment

`FlyControlEnv` implements `gymnasium.Env`:

- **Observation:** 25D vector (see [architecture.md](../architecture.md))
- **Action:** 4D `[0, 1]` — motor commands in X-configuration
- **Reward:** task-specific (see `_compute_reward`)

Tasks (via `TaskType`):
- `HOVER` — stay at target position
- `DELIVERY` — pick up at A, drop at B
- `DELIVERY_ROUTE` — visit N waypoints, avoiding obstacles
- `DEPLOYMENT` — same as route + domain randomization (wind, prop wear, battery temp)

## Agent

`PPOAgent` is a clean PPO implementation:
- **Actor:** tanh MLP → sigmoid mean + learned log_std (per-dim Gaussian)
- **Critic:** shared trunk → scalar value
- **GAE** advantage estimation (λ = 0.95)
- **Clipped surrogate loss** (ε = 0.2)
- **Value loss** coef 0.5, entropy coef 0.01

`PPOConfig` exposes every hyperparameter. Change `lr`, `n_steps`, `batch_size`, etc.

### Evolutionary methods

`PPOAgent.clone()` deep-copies weights. `PPOAgent.mutate()` clones and perturbs ~10% of parameters with Gaussian noise (σ=0.05). Used by `train.py`'s `_evolve()` between ages.

## Training

See [training.md](../training.md#flycontrol) for the curriculum.

```bash
drone-ai train flycontrol --population 6 --ages 10 --steps 10000
```

### Live population mode in the launcher

The launcher's per-stage cards run a single drone by default but can
also train a parallel population inside the same window. From the menu,
press **`P`** to cycle the population through `1 → 2 → 4 → 6 → 8 → 12`
before opening a card. With `population > 1`:

- N envs and N PPO agents are created. BC warm-up runs once on drone 0;
  the rest are cloned from the warmed-up drone and mutated (σ=0.05,
  10% of params) so they start with the same hover competence but
  diverge.
- Every drone steps once per tick. Each drone's PPO buffer fills and
  updates independently. `total_updates` is the *sum* across the
  population — a 6-drone run in the same budget trades per-drone depth
  for parallel diversity.
- The camera follows the current leader (highest recent-mean reward).
  Peer drones render as small colored crosses at their world positions
  so you can watch the population diverge / converge live.
- At the end, the drone with the highest consistency-weighted score
  (`0.9*avg + 0.1*best − 0.5*std`) is saved. Other policies are
  discarded. The runs.csv row is tagged `popN`.

CLI equivalent:

```bash
python -m drone_ai.viz.launcher --stage hover --updates 400 --population 6
```

## Grading

`FlyControlMetrics` captures eval rewards on each of the 4 stages. Weighted sum → grade via `ModelGrader.grade_flycontrol()`:

| Grade | Score |
|-------|------:|
| P  | ≥ 800 |
| S+ | ≥ 700 |
| S  | ≥ 600 |
| … | … |
| F  | ≥ 0 |
| W  | < -50 |

## Loading a trained model

Checkpoints live per-stage under the curriculum chain, e.g.
`models/flycontrol/deployment/P 20-04-2026 flycontrol v1.pt`. See
[training.md](../training.md#ui-curriculum-chain-warm-start) for the
chain and the launcher's warm-start behavior.

```python
from drone_ai.modules.flycontrol import PPOAgent
agent = PPOAgent.from_file("models/flycontrol/deployment/P 20-04-2026 flycontrol v1.pt")
action, _ = agent.select_action(obs, deterministic=True)
```

Or via `DroneAI`:
```python
DroneAI(flycontrol_model="models/flycontrol/deployment/...")
```
