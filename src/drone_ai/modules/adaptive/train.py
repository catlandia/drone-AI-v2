"""Adaptive module benchmark.

Measures the delta between running a trained FlyControl policy WITH
online adaptation vs WITHOUT, on a perturbed version of the environment
(heavier mass, weaker battery). The perturbation stands in for the kind
of real-world distribution shift that adaptive learning is meant to
handle: the drone that was trained in the sim weighs less than the
one we actually deploy, or the battery sags harder than expected.

Run:
    py -m drone_ai.modules.adaptive.train --model path/to/flycontrol.pt
    py -m drone_ai.modules.adaptive.train          # auto-find latest fc ckpt
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from drone_ai.grading import (
    AdaptiveMetricsGrading, ModelGrader, RunLogger, RunRecord,
    generate_model_name, next_version, parse_model_name,
)
from drone_ai.modules.adaptive.learner import AdaptiveConfig, AdaptiveLearner
from drone_ai.modules.flycontrol.agent import PPOAgent, PPOConfig
from drone_ai.modules.flycontrol.environment import (
    FlyControlEnv, TaskType, OBS_DIM, ACT_DIM,
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


def _newest_flycontrol_checkpoint(models_root: str = "models") -> Optional[str]:
    fc_root = os.path.join(models_root, "flycontrol")
    if not os.path.isdir(fc_root):
        return None
    candidates = []
    for stage in os.listdir(fc_root):
        sd = os.path.join(fc_root, stage)
        if not os.path.isdir(sd):
            continue
        for fname in os.listdir(sd):
            parsed = parse_model_name(fname)
            if parsed and parsed["module"] == "flycontrol":
                candidates.append((parsed["date"], parsed["version"],
                                   os.path.join(sd, fname)))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]


def run_training(
    model_path: Optional[str] = None,
    task: TaskType = TaskType.HOVER,
    episodes: int = 5,
    save_dir: str = "models/adaptive",
    seed: int = 42,
    verbose: bool = True,
    log_path: str = "models/runs.csv",
    run_tag: str = "",
) -> Tuple[str, float]:
    """No-args-callable entry. If no model is supplied, auto-discover
    the newest FlyControl checkpoint; if none exists, build a fresh
    PPOAgent on disk so the benchmark still runs end-to-end."""
    if verbose:
        print(f"\n{'='*60}\n  ADAPTIVE LEARNER BENCHMARK\n{'='*60}")

    if model_path is None:
        model_path = _newest_flycontrol_checkpoint()

    if model_path is None:
        # No FlyControl checkpoint anywhere — write a fresh init so the
        # benchmark isn't blocked on training a real one first.
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        seed_path = str(Path(save_dir) / "fresh_baseline.pt")
        agent = PPOAgent(OBS_DIM, ACT_DIM, PPOConfig())
        agent.save(seed_path)
        model_path = seed_path
        if verbose:
            print(f"  No FlyControl checkpoint found — wrote fresh baseline: {seed_path}")
    elif verbose:
        print(f"  Using FlyControl baseline: {model_path}")

    grade, score, metrics = benchmark(model_path, task, episodes, seed)
    if verbose:
        print(f"  Baseline:  {metrics.baseline_score:.2f}")
        print(f"  Adapted:   {metrics.adapted_score:.2f}")
        print(f"  Recovery:  {metrics.recovery_rate:+.2f}")
        print(f"  Stability: {metrics.stability:.2f}")
        print(f"  Score: {score:.1f}  Grade: {grade}")

    # Tier-named placeholder + json metadata.
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    version = next_version(save_dir, "adaptive")
    fname = generate_model_name(grade, "adaptive", version)
    path = Path(save_dir) / fname
    torch.save({
        "grade": grade, "score": score,
        "baseline_path": model_path,
        "metrics": {
            "baseline_score": metrics.baseline_score,
            "adapted_score": metrics.adapted_score,
            "recovery_rate": metrics.recovery_rate,
            "stability": metrics.stability,
        },
    }, str(path))
    with open(path.with_suffix(".json"), "w") as f:
        json.dump({
            "grade": grade, "score": score,
            "baseline_path": model_path,
            "metrics": {
                "baseline_score": metrics.baseline_score,
                "adapted_score": metrics.adapted_score,
                "recovery_rate": metrics.recovery_rate,
                "stability": metrics.stability,
            },
            "timestamp": datetime.now().isoformat(),
            "model_file": fname,
        }, f, indent=2)

    try:
        RunLogger(log_path).append(RunRecord(
            module="adaptive",
            stage=str(task.value if isinstance(task, TaskType) else task),
            best_score=score,
            avg_score=metrics.adapted_score,
            grade=grade,
            minutes=0.0,
            episodes=episodes,
            run_tag=run_tag,
        ))
    except Exception as e:
        print(f"[adaptive] run-log append failed: {e}")

    if verbose:
        print(f"  Saved: {path}")
    return grade, score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None,
                   help="Path to a saved FlyControl .pt (auto-discovered if omitted)")
    p.add_argument("--task", default="hover",
                   choices=["hover", "delivery", "delivery_route", "deployment"])
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", default="models/adaptive")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--run-tag", default="")
    args = p.parse_args()
    run_training(
        model_path=args.model,
        task=TaskType(args.task),
        episodes=args.episodes,
        save_dir=args.save_dir,
        seed=args.seed,
        verbose=not args.quiet,
        run_tag=args.run_tag,
    )


if __name__ == "__main__":
    main()
