"""Perception training — benchmark and grade the perception module.

Since simulation uses noise-based perception (no CNN), this benchmarks
detection accuracy across many scenarios and assigns a grade.
The grade can then be improved by reducing noise parameters manually
or by connecting a real CNN model.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple

import numpy as np

from drone_ai.simulation.world import World
from drone_ai.modules.perception.detector import PerceptionAI, _GRADE_PARAMS
from drone_ai.grading import (
    ModelGrader, PerceptionMetrics, generate_model_name, next_version, GRADE_ORDER
)


def benchmark_grade(
    grade: str,
    n_trials: int = 100,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[str, float]:
    """Benchmark a specific grade level's perception performance."""
    rng = np.random.default_rng(seed)
    perception = PerceptionAI(grade=grade, seed=seed)
    world = World()

    true_positives = 0
    false_positives = 0
    total_true = 0
    position_errors = []

    import time
    times = []

    for _ in range(n_trials):
        world.clear()
        n_obs = rng.integers(5, 20)
        world.generate_random_obstacles(n_obs, rng)
        drone_pos = rng.uniform([-30, -30, 5], [30, 30, 30])

        nearby = world.obstacles_in_radius(drone_pos, perception.detection_range)
        total_true += len(nearby)

        t0 = time.perf_counter()
        detections = perception.detect(drone_pos, world)
        times.append((time.perf_counter() - t0) * 1000)

        for det in detections:
            dists = [np.linalg.norm(det.position - obs.position) for obs in nearby]
            if dists and min(dists) < 5.0:
                true_positives += 1
                position_errors.append(min(dists))
            else:
                false_positives += 1

    tp_rate = true_positives / max(total_true, 1) * 100
    fp_rate = false_positives / max(true_positives + false_positives, 1) * 100
    avg_err = float(np.mean(position_errors)) if position_errors else 10.0
    avg_fps = 1000.0 / max(float(np.mean(times)), 0.001)

    metrics = PerceptionMetrics(
        detection_accuracy=tp_rate,
        false_positive_rate=fp_rate,
        position_error=avg_err,
        fps=avg_fps,
    )
    grader = ModelGrader()
    measured_grade, score = grader.grade_perception(metrics)

    if verbose:
        print(f"\n  Input grade: {grade}  Measured grade: {measured_grade}")
        print(f"  Detection: {tp_rate:.1f}%  FP rate: {fp_rate:.1f}%")
        print(f"  Pos error: {avg_err:.2f}m  FPS: {avg_fps:.0f}")
        print(grader.report("perception", measured_grade, score))

    return measured_grade, score, metrics


def run_training(
    grade: str = "P",
    trials: int = 100,
    save_dir: str = "models/perception",
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[str, float]:
    if verbose:
        print(f"\n{'='*60}")
        print(f"  PERCEPTION BENCHMARK (simulating grade={grade})")
        print(f"{'='*60}")

    measured_grade, score, metrics = benchmark_grade(grade, trials, seed, verbose)

    version = next_version(save_dir, "perception")
    fname = generate_model_name(measured_grade, "perception", version)
    path = Path(save_dir) / fname
    path.parent.mkdir(parents=True, exist_ok=True)

    import torch
    torch.save({
        "grade": measured_grade,
        "input_grade": grade,
        "score": score,
        "metrics": {
            "detection_accuracy": metrics.detection_accuracy,
            "false_positive_rate": metrics.false_positive_rate,
            "position_error": metrics.position_error,
            "fps": metrics.fps,
        },
    }, str(path))

    results = {
        "grade": measured_grade, "score": score,
        "input_grade": grade,
        "metrics": {
            "detection_accuracy": metrics.detection_accuracy,
            "false_positive_rate": metrics.false_positive_rate,
            "position_error": metrics.position_error,
            "fps": metrics.fps,
        },
        "timestamp": datetime.now().isoformat(),
        "model_file": fname,
    }
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"  Saved: {path}")

    return measured_grade, score


def main():
    parser = argparse.ArgumentParser(description="Benchmark Perception AI")
    parser.add_argument("--grade", default="P", choices=GRADE_ORDER,
                        help="Grade level to simulate")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--save-dir", default="models/perception")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_training(args.grade, args.trials, args.save_dir, args.seed, not args.quiet)


if __name__ == "__main__":
    main()
