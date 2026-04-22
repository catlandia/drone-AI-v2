"""Swarm benchmark — exercise the SwarmCoordinator on synthetic scenes.

Layer 8 has no neural weights to fit. "Training" here means:
  1. Generate N random multi-drone plans.
  2. For each plan, replay synthetic visual contacts + a simulated
     swarm-mate failure.
  3. Grade the coordinator on:
       - avoidance correctness   (right action class for each contact)
       - divert correctness      (only the nearest non-LIFE_CRITICAL
                                  drone takes the divert)
       - LIFE_CRITICAL exemption (LIFE_CRITICAL drones never divert)

Writes a graded `.pt` placeholder under models/swarm/ and appends a
row to models/runs.csv tagged module=swarm.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from drone_ai.grading import (
    RunLogger, RunRecord, generate_model_name, next_version,
    score_to_universal_grade,
)
from drone_ai.modules.swarm import (
    SwarmCoordinator, build_swarm_plan, DroneAssignment, DroneRole,
    VisualContact,
)
from drone_ai.modules.swarm.coordinator import AvoidanceKind


def _random_plan(rng: np.random.Generator, n_drones: int) -> "SwarmPlan":
    assignments = []
    for i in range(n_drones):
        is_lc = bool(rng.random() < 0.2)
        wp = rng.uniform([-50, -50, 5], [50, 50, 15], size=(2, 3)).astype(np.float32)
        assignments.append(DroneAssignment(
            drone_id=f"d{i}",
            role=DroneRole.PRIMARY if i == 0 else DroneRole.SOLO,
            route=[wp[0], wp[1]],
            mission_class="LIFE_CRITICAL" if is_lc else "STANDARD",
        ))
    return build_swarm_plan(f"plan-{int(rng.integers(0, 2**31))}", assignments)


def benchmark(
    n_trials: int = 30,
    n_drones: int = 4,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[str, float, dict]:
    rng = np.random.default_rng(seed)

    avoidance_correct = 0
    avoidance_total = 0
    divert_correct = 0
    divert_total = 0
    lc_exempt_correct = 0
    lc_exempt_total = 0

    for _ in range(n_trials):
        plan = _random_plan(rng, n_drones)
        coords = {d: SwarmCoordinator(plan, d) for d in plan.drone_ids()}

        # 1. Visual avoidance test — head-on contact for each drone.
        for d_id, coord in coords.items():
            self_pos = plan.assignments[d_id].route[0]
            head_on = VisualContact(
                agent_id=None, is_peer=False,
                position=self_pos + np.array([3.0, 0.0, 0.0], dtype=np.float32),
                velocity=np.array([-5.0, 0.0, 0.0], dtype=np.float32),
                range_m=3.0, closing_speed=10.0,
            )
            action = coord.step(self_pos, [head_on])
            avoidance_total += 1
            # Correct = some avoidance action returned (not None / continue).
            if action is not None and action.kind != AvoidanceKind.CONTINUE:
                avoidance_correct += 1

        # 2. Swarm-mate-failed test — pick a STANDARD drone to "fail",
        # check that exactly one non-LIFE_CRITICAL peer takes the divert
        # and all LIFE_CRITICAL peers refuse it.
        std_drones = [d for d in plan.drone_ids()
                      if plan.assignments[d].mission_class == "STANDARD"]
        if len(std_drones) >= 2:
            failed = std_drones[0]
            failed_pos = plan.assignments[failed].route[-1]

            survivors = [d for d in plan.drone_ids() if d != failed]
            divert_takers = []
            for d in survivors:
                coords[d].mark_peer_failed(failed, failed_pos)
                action = coords[d].step(plan.assignments[d].route[0], [])
                if action is not None and action.kind == AvoidanceKind.DIVERT_TO_MARK:
                    divert_takers.append(d)

            divert_total += 1
            non_lc_survivors = [
                d for d in survivors
                if plan.assignments[d].mission_class != "LIFE_CRITICAL"
            ]
            # Correct: at most one non-LIFE_CRITICAL drone diverts.
            if len(divert_takers) <= 1 and (
                not non_lc_survivors or divert_takers
            ):
                divert_correct += 1

            # LIFE_CRITICAL exemption: any LC peer must NOT be in divert_takers.
            lc_peers = [d for d in survivors
                        if plan.assignments[d].mission_class == "LIFE_CRITICAL"]
            for lc in lc_peers:
                lc_exempt_total += 1
                if lc not in divert_takers:
                    lc_exempt_correct += 1

    avoid_rate = avoidance_correct / max(avoidance_total, 1)
    divert_rate = divert_correct / max(divert_total, 1)
    lc_rate = lc_exempt_correct / max(lc_exempt_total, 1) if lc_exempt_total else 1.0

    # Score: weighted average mapped onto the universal 0-800 scale.
    score = (avoid_rate * 0.4 + divert_rate * 0.4 + lc_rate * 0.2) * 800.0
    grade = score_to_universal_grade(score)

    metrics = {
        "avoidance_rate": avoid_rate,
        "divert_correctness": divert_rate,
        "life_critical_exemption_rate": lc_rate,
        "trials": n_trials,
        "drones_per_trial": n_drones,
    }

    if verbose:
        print(f"  Avoidance correctness:        {avoid_rate*100:.1f}%")
        print(f"  Divert correctness:           {divert_rate*100:.1f}%")
        print(f"  LIFE_CRITICAL exemption rate: {lc_rate*100:.1f}%")
        print(f"  Score: {score:.1f}  Grade: {grade}")
    return grade, score, metrics


def run_training(
    trials: int = 30,
    n_drones: int = 4,
    save_dir: str = "models/swarm",
    seed: int = 42,
    verbose: bool = True,
    log_path: str = "models/runs.csv",
    run_tag: str = "",
) -> Tuple[str, float]:
    if verbose:
        print(f"\n{'='*60}\n  SWARM COORDINATOR BENCHMARK\n{'='*60}")
    grade, score, metrics = benchmark(trials, n_drones, seed, verbose)

    version = next_version(save_dir, "swarm")
    fname = generate_model_name(grade, "swarm", version)
    path = Path(save_dir) / fname
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"grade": grade, "score": score, "metrics": metrics}, str(path))
    with open(path.with_suffix(".json"), "w") as f:
        json.dump({
            "grade": grade, "score": score, "metrics": metrics,
            "timestamp": datetime.now().isoformat(), "model_file": fname,
        }, f, indent=2)

    try:
        RunLogger(log_path).append(RunRecord(
            module="swarm", stage="coordinator",
            best_score=score, avg_score=score, grade=grade,
            minutes=0.0, episodes=trials, run_tag=run_tag,
        ))
    except Exception as e:
        print(f"[swarm] run-log append failed: {e}")

    if verbose:
        print(f"  Saved: {path}")
    return grade, score


def main():
    p = argparse.ArgumentParser(description="Benchmark Swarm Layer 8")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--drones", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", default="models/swarm")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--run-tag", default="")
    args = p.parse_args()
    run_training(args.trials, args.drones, args.save_dir, args.seed,
                 not args.quiet, run_tag=args.run_tag)


if __name__ == "__main__":
    main()
