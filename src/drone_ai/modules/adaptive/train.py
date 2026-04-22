"""Adaptive module benchmark.

Measures the delta between running a trained FlyControl policy WITH
online adaptation vs WITHOUT, on a perturbed version of the environment
(heavier mass, weaker battery). The perturbation stands in for the kind
of real-world distribution shift that adaptive learning is meant to
handle: the drone that was trained in the sim weighs less than the
one we actually deploy, or the battery sags harder than expected.

Run:
    py -m drone_ai.modules.adaptive.train --model path/to/flycontrol.pt
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

import numpy as np

from drone_ai.grading import (
    AdaptiveMetricsGrading, ModelGrader, RunLogger, RunRecord,
)
from drone_ai.modules.adaptive.learner import AdaptiveConfig, AdaptiveLearner
from drone_ai.modules.flycontrol.agent import PPOAgent
from drone_ai.modules.flycontrol.environment import (
    FlyControlEnv, TaskType,
)


def _perturb_env(env: FlyControlEnv) -> None:
    """Apply an OOD perturbation — heavier drone, weaker battery."""
    env.physics.MASS *= 1.2
    env.physics.BAT_THRUST_MIN = 0.70  # more voltage sag


def _run_episodes(
    env: FlyControlEnv,
    agent: PPOAgent,
    n: int,
    learner: AdaptiveLearner = None,
) -> List[float]:
    rewards: List[float] = []
    for _ in range(n):
        obs, _ = env.reset()
        total = 0.0
        while True:
            if learner is not None:
                action, info = learner.select_action(obs, deterministic=False)
            else:
                action, info = agent.select_action(obs, deterministic=True)
            next_obs, r, term, trunc, _ = env.step(action)
            done = bool(term or trunc)
            if learner is not None:
                learner.observe(obs, action, float(r), info, done)
            total += float(r)
            obs = next_obs
            if done:
                if learner is not None:
                    learner.end_episode(next_obs, total)
                break
        rewards.append(total)
    return rewards


def benchmark(
    model_path: str,
    task: TaskType = TaskType.HOVER,
    episodes: int = 10,
    seed: int = 42,
) -> Tuple[str, float, AdaptiveMetricsGrading]:
    grader = ModelGrader()

    # Baseline: adaptation OFF, perturbed env.
    env = FlyControlEnv(task=task, difficulty=0.5, seed=seed)
    _perturb_env(env)
    baseline_agent = PPOAgent.from_file(model_path)
    baseline = _run_episodes(env, baseline_agent, episodes)

    # Adapted: adaptation ON, SAME seed + perturbation.
    env = FlyControlEnv(task=task, difficulty=0.5, seed=seed)
    _perturb_env(env)
    agent = PPOAgent.from_file(model_path)
    learner = AdaptiveLearner(agent, AdaptiveConfig(enabled=True))
    adapted = _run_episodes(env, agent, episodes, learner=learner)

    b_mean = float(np.mean(baseline))
    a_mean = float(np.mean(adapted))
    recovery = (a_mean - b_mean) / max(abs(b_mean), 1.0)
    stability = 1.0 - float(np.std(adapted)) / (abs(a_mean) + 1.0)
    stability = max(0.0, min(1.0, stability))

    metrics = AdaptiveMetricsGrading(
        baseline_score=b_mean,
        adapted_score=a_mean,
        recovery_rate=recovery,
        stability=stability,
    )
    grade, score = grader.grade_adaptive(metrics)
    return grade, score, metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to a saved FlyControl .pt")
    p.add_argument("--task", default="hover",
                   choices=["hover", "delivery", "delivery_route", "deployment"])
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-tag", default="")
    args = p.parse_args()

    task = TaskType(args.task)
    grade, score, m = benchmark(args.model, task, args.episodes, args.seed)
    print(f"Baseline (no adapt): {m.baseline_score:.2f}")
    print(f"Adapted  (online):   {m.adapted_score:.2f}")
    print(f"Recovery rate:       {m.recovery_rate:+.2f}")
    print(f"Stability:           {m.stability:.2f}")
    print(f"Score: {score:.1f}  Grade: {grade}")

    RunLogger().append(RunRecord(
        module="adaptive",
        stage=args.task,
        best_score=score,
        avg_score=m.adapted_score,
        grade=grade,
        minutes=0.0,
        run_tag=args.run_tag,
    ))


if __name__ == "__main__":
    main()
