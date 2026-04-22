"""Pathfinder training — evaluates A*/RRT quality and assigns grade.

The pathfinder uses A* and RRT (no neural net needed for basic operation).
Training here means: run benchmarks across test worlds and assign a grade.
An RL-enhanced variant can be enabled for future work.
"""

import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple

import numpy as np

from drone_ai.simulation.world import World, Obstacle
from drone_ai.modules.pathfinder.algorithms import PathPlanner
from drone_ai.grading import (
    ModelGrader, PathfinderMetrics, generate_model_name, next_version,
    RunLogger, RunRecord,
)


def _optimal_distance(start: np.ndarray, goal: np.ndarray) -> float:
    return float(np.linalg.norm(goal - start))


def _path_length(path) -> float:
    if not path or len(path) < 2:
        return float("inf")
    return float(sum(np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1)))


def _path_collides(path, world: World, margin: float = 0.3) -> bool:
    for p in path:
        if world.in_collision(p, margin):
            return True
    return False


def benchmark(
    n_trials: int = 50,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[str, float]:
    rng = np.random.default_rng(seed)
    world = World()
    planner = PathPlanner(world)

    optimality_ratios = []
    collision_free = 0
    plan_times_ms = []

    for trial in range(n_trials):
        world.clear()
        n_obs = rng.integers(5, 25)
        world.generate_random_obstacles(n_obs, rng)

        start = world.random_free_point(rng)
        goal = world.random_free_point(rng)
        optimal = _optimal_distance(start, goal)
        if optimal < 5.0:
            continue

        t0 = time.perf_counter()
        path = planner.plan(start, goal)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        plan_times_ms.append(elapsed_ms)

        if path and len(path) >= 2:
            length = _path_length(path)
            ratio = length / max(optimal, 1.0)
            optimality_ratios.append(ratio)
            if not _path_collides(path, world):
                collision_free += 1

        if verbose and (trial + 1) % 10 == 0:
            avg_ms = np.mean(plan_times_ms) if plan_times_ms else 0
            print(f"  Trial {trial+1}/{n_trials}  avg_ms={avg_ms:.1f}")

    avoidance = collision_free / max(len(optimality_ratios), 1)
    avg_optimality = float(np.mean(optimality_ratios)) if optimality_ratios else 5.0
    avg_ms = float(np.mean(plan_times_ms)) if plan_times_ms else 999.0

    metrics = PathfinderMetrics(
        path_optimality=avg_optimality,
        avoidance_rate=avoidance,
        planning_ms=avg_ms,
    )
    grader = ModelGrader()
    grade, score = grader.grade_pathfinder(metrics)

    if verbose:
        print(f"\n  Optimality ratio: {avg_optimality:.2f} (1.0 = perfect)")
        print(f"  Collision-free:   {avoidance*100:.1f}%")
        print(f"  Avg plan time:    {avg_ms:.1f} ms")
        print(grader.report("pathfinder", grade, score))

    return grade, score, metrics


def run_training(
    trials: int = 50,
    save_dir: str = "models/pathfinder",
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[str, float]:
    if verbose:
        print(f"\n{'='*60}")
        print("  PATHFINDER BENCHMARK")
        print(f"{'='*60}")

    grade, score, metrics = benchmark(trials, seed, verbose)

    version = next_version(save_dir, "pathfinder")
    fname = generate_model_name(grade, "pathfinder", version)
    path = Path(save_dir) / fname
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save metadata (A*/RRT has no neural weights to save)
    results = {
        "grade": grade, "score": score,
        "metrics": {
            "path_optimality": metrics.path_optimality,
            "avoidance_rate": metrics.avoidance_rate,
            "planning_ms": metrics.planning_ms,
        },
        "timestamp": datetime.now().isoformat(),
        "model_file": fname,
        "algorithm": "A*+RRT",
    }
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save a placeholder .pt so file naming convention is consistent
    import torch
    torch.save({"grade": grade, "algorithm": "A*+RRT", "score": score}, str(path))

    try:
        RunLogger().append(RunRecord(
            module="pathfinder", stage="benchmark",
            best_score=score, avg_score=score, grade=grade,
            minutes=0.0, episodes=trials,
        ))
    except Exception as e:
        print(f"[pathfinder] run-log append failed: {e}")

    if verbose:
        print(f"  Saved: {path}")

    return grade, score


def main():
    parser = argparse.ArgumentParser(description="Benchmark Pathfinder AI")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--save-dir", default="models/pathfinder")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_training(args.trials, args.save_dir, args.seed, not args.quiet)


if __name__ == "__main__":
    main()
