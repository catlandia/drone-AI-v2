# FlyControl

**Layer 4** — low-level motor control via PPO reinforcement learning.

## Files

- `modules/flycontrol/environment.py` — Gym environment (`FlyControlEnv`)
- `modules/flycontrol/agent.py` — PPO actor-critic + rollout buffer
- `modules/flycontrol/train.py` — evolutionary training pipeline

## Environment

`FlyControlEnv` implements `gymnasium.Env`:

- **Observation:** 19D vector (see [architecture.md](../architecture.md))
- **Action:** 4D `[0, 1]` — motor commands in X-configuration
- **Reward:** task-specific (see `_compute_reward`)

Tasks (via `TaskType`):
- `HOVER` — stay at target position
- `DELIVERY` — pick up at A, drop at B
- `DELIVERY_ROUTE` — visit N waypoints, avoiding obstacles
- `DEPLOYMENT` — same as route + domain randomization

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

```python
from drone_ai.modules.flycontrol import PPOAgent
agent = PPOAgent.from_file("models/flycontrol/P 20-04-2026 flycontrol v1.pt")
action, _ = agent.select_action(obs, deterministic=True)
```

Or via `DroneAI`:
```python
DroneAI(flycontrol_model="models/flycontrol/...")
```
