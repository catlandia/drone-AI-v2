"""FlyControl training — evolutionary PPO with curriculum.

Population of drones trains together. Top 2 survive, rest mutate.
Runs 2 rounds: full population → top 2 rematch.
Final model is graded and saved as: {Grade} {DD-MM-YYYY} flycontrol v{N}.pt
"""

import json
import copy
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from drone_ai.modules.flycontrol.environment import FlyControlEnv, TaskType, OBS_DIM
from drone_ai.modules.flycontrol.agent import PPOAgent, PPOConfig
from drone_ai.grading import (
    ModelGrader, FlyControlMetrics, generate_model_name, next_version
)

STAGES = [
    {"name": "Hover",       "task": TaskType.HOVER,          "difficulty": 0.3, "rand": False},
    {"name": "Delivery",    "task": TaskType.DELIVERY,        "difficulty": 0.5, "rand": False},
    {"name": "Route",       "task": TaskType.DELIVERY_ROUTE,  "difficulty": 0.6, "rand": False},
    {"name": "Deployment",  "task": TaskType.DEPLOYMENT,      "difficulty": 1.0, "rand": True},
]

EVAL_EPISODES = 5
EVAL_MAX_STEPS = 1000


def evaluate_agent(agent: PPOAgent, task: TaskType, difficulty: float, rand: bool) -> float:
    env = FlyControlEnv(task=task, difficulty=difficulty, domain_randomization=rand)
    total = 0.0
    for ep in range(EVAL_EPISODES):
        obs, _ = env.reset(seed=ep * 7)
        for _ in range(EVAL_MAX_STEPS):
            action, _ = agent.select_action(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            total += r
            if term or trunc:
                break
    env.close()
    return total / EVAL_EPISODES


def train_stage(
    population: List[PPOAgent],
    stage: dict,
    ages: int,
    steps_per_age: int,
    seed: int,
    verbose: bool,
) -> List[float]:
    task = stage["task"]
    diff = stage["difficulty"]
    rand = stage["rand"]
    n = len(population)

    envs = [FlyControlEnv(task=task, difficulty=diff, domain_randomization=rand) for _ in range(n)]
    total_rewards = [0.0] * n

    for age in range(ages):
        base_seed = seed + age * 1000
        obs_list = []
        for i, env in enumerate(envs):
            obs, _ = env.reset(seed=base_seed + i)
            obs_list.append(obs)

        alive = [True] * n
        rewards = [0.0] * n
        steps_since_update = [0] * n

        itr = range(steps_per_age)
        if HAS_TQDM and verbose:
            itr = tqdm(itr, desc=f"  {stage['name']} age {age+1}/{ages}", leave=False)

        for step in itr:
            for i in range(n):
                if not alive[i]:
                    continue
                action, info = population[i].select_action(obs_list[i])
                next_obs, reward, term, trunc, _ = envs[i].step(action)
                done = term or trunc
                population[i].store(obs_list[i], action, reward, info["value"], info["log_prob"], done)
                rewards[i] += reward
                steps_since_update[i] += 1
                if done:
                    alive[i] = False
                else:
                    obs_list[i] = next_obs
                if steps_since_update[i] >= population[i].config.n_steps:
                    population[i].update(obs_list[i])
                    steps_since_update[i] = 0

            if not any(alive):
                for i in range(n):
                    obs_list[i], _ = envs[i].reset(seed=base_seed + step)
                    alive[i] = True
                    rewards[i] = 0.0

        for i in range(n):
            total_rewards[i] += rewards[i]

        # Evolve between ages (not after last)
        if age < ages - 1:
            _evolve(population, rewards)

        if verbose:
            best = max(rewards)
            print(f"    {stage['name']} age {age+1}: best={best:.1f} avg={np.mean(rewards):.1f}")

    for env in envs:
        env.close()

    return total_rewards


def _evolve(population: List[PPOAgent], fitnesses: List[float]):
    order = np.argsort(fitnesses)[::-1]
    top2 = [population[order[0]].clone(), population[order[1]].clone()]
    new_pop = list(top2)
    for i in range(len(population) - 2):
        parent = top2[i % 2]
        new_pop.append(parent.mutate())
    population[:] = new_pop


def run_training(
    population_size: int = 6,
    ages_per_stage: int = 10,
    steps_per_age: int = 10000,
    save_dir: str = "models/flycontrol",
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[str, float]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        print(f"\n{'='*60}")
        print("  FLYCONTROL TRAINING")
        print(f"{'='*60}")
        print(f"  Population: {population_size}  Ages/stage: {ages_per_stage}")
        print(f"  Steps/age: {steps_per_age:,}  Device: {device}")

    population = [
        PPOAgent(OBS_DIM, 4, PPOConfig(), device)
        for _ in range(population_size)
    ]

    # Round 1
    round1_scores = []
    for stage in STAGES:
        scores = train_stage(population, stage, ages_per_stage, steps_per_age, seed, verbose)
        round1_scores.append(scores)

    totals = [sum(round1_scores[s][i] for s in range(len(STAGES))) for i in range(population_size)]
    order = np.argsort(totals)[::-1]

    if verbose:
        print("\n  Round 1 results:")
        for rank, idx in enumerate(order[:3]):
            print(f"    #{rank+1}: Drone {idx+1} score={totals[idx]:.1f}")

    top2 = [population[order[0]].clone(), population[order[1]].clone()]

    # Round 2
    population = [top2[0], top2[1]] + [top2[i % 2].mutate() for i in range(population_size - 2)]
    for stage in STAGES:
        train_stage(population, stage, ages_per_stage, steps_per_age, seed + 99999, verbose)

    # Final evaluation
    totals2 = [sum(evaluate_agent(a, s["task"], s["difficulty"], s["rand"]) for s in STAGES)
               for a in population]
    best_idx = int(np.argmax(totals2))
    best = population[best_idx]

    stage_scores = {s["name"]: evaluate_agent(best, s["task"], s["difficulty"], s["rand"])
                    for s in STAGES}

    metrics = FlyControlMetrics(
        hover_score=stage_scores["Hover"],
        delivery_score=stage_scores["Delivery"],
        route_score=stage_scores["Route"],
        deploy_score=stage_scores["Deployment"],
    )
    grader = ModelGrader()
    grade, score = grader.grade_flycontrol(metrics)

    version = next_version(save_dir, "flycontrol")
    fname = generate_model_name(grade, "flycontrol", version)
    path = Path(save_dir) / fname
    path.parent.mkdir(parents=True, exist_ok=True)
    best.save(str(path))

    results_path = path.with_suffix(".json")
    with open(results_path, "w") as f:
        json.dump({
            "grade": grade, "score": score,
            "stage_scores": stage_scores,
            "timestamp": datetime.now().isoformat(),
            "model_file": fname,
        }, f, indent=2)

    if verbose:
        print(grader.report("flycontrol", grade, score))
        print(f"  Saved: {path}")

    return grade, score


def main():
    parser = argparse.ArgumentParser(description="Train FlyControl AI")
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--ages", type=int, default=10)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--save-dir", default="models/flycontrol")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_training(args.population, args.ages, args.steps, args.save_dir, args.seed, not args.quiet)


if __name__ == "__main__":
    main()
