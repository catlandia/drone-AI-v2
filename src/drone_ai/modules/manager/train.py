"""Manager training — benchmark and grade the mission planner."""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple

import numpy as np

from drone_ai.modules.manager.planner import MissionPlanner, DeliveryRequest, Priority
from drone_ai.grading import (
    ModelGrader, ManagerMetrics, generate_model_name, next_version, GRADE_ORDER,
    RunLogger, RunRecord,
)


def _optimal_distance(targets, start) -> float:
    """Greedy TSP lower bound from start through all targets."""
    pos = start.copy()
    remaining = list(range(len(targets)))
    total = 0.0
    while remaining:
        dists = [np.linalg.norm(targets[i] - pos) for i in remaining]
        best = remaining[int(np.argmin(dists))]
        total += np.linalg.norm(targets[best] - pos)
        pos = targets[best]
        remaining.remove(best)
    return total


def benchmark_grade(
    grade: str,
    n_trials: int = 20,
    deliveries_per_trial: int = 6,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[str, float]:
    rng = np.random.default_rng(seed)
    base = np.zeros(3)
    planner = MissionPlanner(base_position=base, grade=grade, seed=seed)

    completion_rates = []
    distance_efficiencies = []
    priority_scores = []

    for trial in range(n_trials):
        planner.reset()
        targets = []
        priorities = []

        for _ in range(deliveries_per_trial):
            t = rng.uniform([-100, -100, 0], [100, 100, 0]).astype(float)
            p = rng.choice([Priority.NORMAL, Priority.URGENT, Priority.CRITICAL])
            planner.add_delivery(t, p)
            targets.append(t)
            priorities.append(p)

        # Simulate mission
        pos = base.copy()
        order = []
        for _ in range(deliveries_per_trial + 2):
            chosen = planner.select_next(pos)
            if chosen is None:
                break
            order.append(chosen)
            pos = chosen.target.copy()
            planner.update(0.1, pos, 0.9)
            planner.complete_current(success=True)

        # Metrics
        comp_rate = len(planner.state.completed) / deliveries_per_trial
        completion_rates.append(comp_rate)

        # Distance efficiency
        actual_dist = float(sum(
            np.linalg.norm(order[i].target - (order[i-1].target if i > 0 else base))
            for i in range(len(order))
        ))
        opt_dist = _optimal_distance(targets, base)
        eff = min(opt_dist / max(actual_dist, 1.0), 1.0)
        distance_efficiencies.append(eff)

        # Priority adherence — high priority should be served before normal
        priority_ok = 0
        for i, req in enumerate(order):
            later_priorities = [order[j].priority.value for j in range(i + 1, len(order))]
            if not later_priorities or req.priority.value >= max(later_priorities, default=0):
                priority_ok += 1
        priority_scores.append(priority_ok / max(len(order), 1))

    metrics = ManagerMetrics(
        completion_rate=float(np.mean(completion_rates)),
        distance_efficiency=float(np.mean(distance_efficiencies)),
        priority_score=float(np.mean(priority_scores)),
        battery_waste=max(0.0, 1.0 - float(np.mean(distance_efficiencies))),
    )
    grader = ModelGrader()
    measured_grade, score = grader.grade_manager(metrics)

    if verbose:
        print(f"\n  Input grade: {grade}  Measured: {measured_grade}")
        print(f"  Completion:  {metrics.completion_rate*100:.1f}%")
        print(f"  Efficiency:  {metrics.distance_efficiency*100:.1f}%")
        print(f"  Priority:    {metrics.priority_score*100:.1f}%")
        print(grader.report("manager", measured_grade, score))

    return measured_grade, score, metrics


def run_training(
    grade: str = "P",
    trials: int = 20,
    save_dir: str = "models/manager",
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[str, float]:
    if verbose:
        print(f"\n{'='*60}")
        print(f"  MANAGER BENCHMARK (grade={grade})")
        print(f"{'='*60}")

    measured_grade, score, metrics = benchmark_grade(grade, trials, 6, seed, verbose)

    version = next_version(save_dir, "manager")
    fname = generate_model_name(measured_grade, "manager", version)
    path = Path(save_dir) / fname
    path.parent.mkdir(parents=True, exist_ok=True)

    import torch
    torch.save({"grade": measured_grade, "input_grade": grade, "score": score}, str(path))

    with open(path.with_suffix(".json"), "w") as f:
        json.dump({
            "grade": measured_grade, "score": score,
            "input_grade": grade,
            "metrics": {
                "completion_rate": metrics.completion_rate,
                "distance_efficiency": metrics.distance_efficiency,
                "priority_score": metrics.priority_score,
                "battery_waste": metrics.battery_waste,
            },
            "timestamp": datetime.now().isoformat(),
            "model_file": fname,
        }, f, indent=2)

    try:
        RunLogger().append(RunRecord(
            module="manager", stage="benchmark",
            best_score=score, avg_score=score, grade=measured_grade,
            minutes=0.0, episodes=trials,
        ))
    except Exception as e:
        print(f"[manager] run-log append failed: {e}")

    if verbose:
        print(f"  Saved: {path}")

    return measured_grade, score


def main():
    parser = argparse.ArgumentParser(description="Benchmark Manager AI")
    parser.add_argument("--grade", default="P", choices=GRADE_ORDER)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--save-dir", default="models/manager")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_training(args.grade, args.trials, args.save_dir, args.seed, not args.quiet)


if __name__ == "__main__":
    main()
