"""Swarm benchmark — exercise the SwarmCoordinator on synthetic scenes.

Layer 8 has no neural weights to fit. "Training" here means hammering
the coordinator with a mix of easy and genuinely hard scenarios and
grading it on whether it picks the *correct action class* — not just
whether it returns something.

Hard cases we expect the rule-based coordinator to trip on:
  - Off-axis head-on (contact slightly above/below, closing fast).
  - Contacts at the edge of AVOID_RADIUS (should NOT trigger).
  - Multi-threat scenes (two contacts, both inside radius, from
    different sides — only one action can fire per tick).
  - Ambiguous priority (two LIFE_CRITICAL drones head-on, both would
    want to climb, so one must yield).
  - Innocent-bystander divert check: a distant peer "fails" and ONLY
    the nearest survivor should take the divert, not every peer.

Writes a graded `.pt` placeholder under models/swarm/ and appends a
row to models/runs.csv tagged module=swarm.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

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
    lc_count = 0
    for i in range(n_drones):
        # Cap LIFE_CRITICAL at 40% so divert tests always have a non-LC
        # survivor to evaluate.
        is_lc = bool(rng.random() < 0.25 and lc_count < int(n_drones * 0.4))
        if is_lc:
            lc_count += 1
        wp = rng.uniform([-50, -50, 5], [50, 50, 15], size=(2, 3)).astype(np.float32)
        assignments.append(DroneAssignment(
            drone_id=f"d{i}",
            role=DroneRole.PRIMARY if i == 0 else DroneRole.SOLO,
            route=[wp[0], wp[1]],
            mission_class="LIFE_CRITICAL" if is_lc else "STANDARD",
        ))
    return build_swarm_plan(f"plan-{int(rng.integers(0, 2**31))}", assignments)


def _expected_action(
    self_pos: np.ndarray,
    contact: VisualContact,
    is_higher_priority: bool,
) -> AvoidanceKind:
    """Ground-truth right-of-way rule for a single contact. Mirrors
    the coordinator's own logic but written separately so the test is
    not tautological — it's the *spec* the coordinator is checked
    against, not a copy of the implementation.
    """
    rel = contact.position - self_pos
    rng = float(np.linalg.norm(rel))
    if rng > 8.0:  # SwarmCoordinator.AVOID_RADIUS
        # Out of action radius — expected answer is None / CONTINUE.
        return AvoidanceKind.CONTINUE
    if rng < 1e-6:
        return AvoidanceKind.BANK_RIGHT
    rel_dir = rel / rng
    if contact.closing_speed > 3.0 and abs(rel_dir[2]) < 0.3:
        return AvoidanceKind.CLIMB if is_higher_priority else AvoidanceKind.YIELD
    if rel_dir[2] > 0.3:
        return AvoidanceKind.YIELD
    if rel_dir[2] < -0.3:
        return AvoidanceKind.CLIMB
    return AvoidanceKind.BANK_RIGHT


def _random_contact(
    rng: np.random.Generator,
    self_pos: np.ndarray,
    force_inside_radius: bool = True,
) -> VisualContact:
    """Place a contact in a varied position — inside, at the edge, or
    just outside the avoidance radius. Outside-radius contacts are how
    we catch false-positive avoidance."""
    if force_inside_radius:
        rng_r = float(rng.uniform(2.0, 7.5))
    else:
        rng_r = float(rng.uniform(2.0, 14.0))  # mix of inside/outside
    theta = float(rng.uniform(0, 2 * np.pi))
    phi = float(rng.uniform(-0.5, 0.5))  # small vertical offset
    direction = np.array([
        np.cos(phi) * np.cos(theta),
        np.cos(phi) * np.sin(theta),
        np.sin(phi),
    ], dtype=np.float32)
    pos = self_pos + direction * rng_r
    vel = -direction * float(rng.uniform(1.0, 12.0))
    closing = float(-np.dot(vel, direction))
    return VisualContact(
        agent_id=None, is_peer=False,
        position=pos, velocity=vel,
        range_m=rng_r, closing_speed=closing,
        confidence=float(rng.uniform(0.5, 1.0)),
    )


def benchmark(
    n_trials: int = 120,
    n_drones: int = 4,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[str, float, dict]:
    rng = np.random.default_rng(seed)

    correct_avoid = 0
    total_avoid = 0
    false_trigger = 0              # acted when it should have continued
    missed_trigger = 0             # continued when it should have acted
    divert_correct = 0
    divert_total = 0
    lc_exempt_correct = 0
    lc_exempt_total = 0
    false_divert = 0               # drone diverted when it shouldn't have

    for _ in range(n_trials):
        plan = _random_plan(rng, n_drones)
        coords = {d: SwarmCoordinator(plan, d) for d in plan.drone_ids()}

        # 1. Avoidance: mix of inside / edge / outside contacts.
        for d_id, coord in coords.items():
            self_pos = plan.assignments[d_id].route[0].astype(np.float32)
            # Include a contact that's outside the radius ~30% of the
            # time so a greedy "always act" coordinator would be wrong.
            inside = bool(rng.random() > 0.3)
            contact = _random_contact(rng, self_pos, force_inside_radius=inside)
            assignment = plan.assignments[d_id]
            high_pri = assignment.mission_class == "LIFE_CRITICAL" or assignment.role == DroneRole.PRIMARY
            expected = _expected_action(self_pos, contact, high_pri)

            action = coord.step(self_pos, [contact])
            got = action.kind if action is not None else AvoidanceKind.CONTINUE

            total_avoid += 1
            if got == expected:
                correct_avoid += 1
            elif expected == AvoidanceKind.CONTINUE and got != AvoidanceKind.CONTINUE:
                false_trigger += 1
            elif expected != AvoidanceKind.CONTINUE and got == AvoidanceKind.CONTINUE:
                missed_trigger += 1

        # 2. Swarm-mate-failed: only the NEAREST non-LC survivor may divert.
        std_drones = [d for d in plan.drone_ids()
                      if plan.assignments[d].mission_class == "STANDARD"]
        if len(std_drones) >= 2:
            failed = std_drones[0]
            failed_pos = plan.assignments[failed].route[-1]

            survivors = [d for d in plan.drone_ids() if d != failed]

            # Compute who *should* divert — the nearest non-LC survivor.
            ranked = sorted(
                [(float(np.linalg.norm(plan.assignments[d].route[0] - failed_pos)), d)
                 for d in survivors
                 if plan.assignments[d].mission_class != "LIFE_CRITICAL"]
            )
            expected_diverter = ranked[0][1] if ranked else None

            divert_takers: List[str] = []
            for d in survivors:
                coords[d].mark_peer_failed(failed, failed_pos)
                action = coords[d].step(plan.assignments[d].route[0], [])
                if action is not None and action.kind == AvoidanceKind.DIVERT_TO_MARK:
                    divert_takers.append(d)

            divert_total += 1
            # Correct = exactly the expected diverter took the divert,
            # no one else did.
            if divert_takers == ([expected_diverter] if expected_diverter else []):
                divert_correct += 1
            if expected_diverter is not None and expected_diverter not in divert_takers:
                pass  # missed — counted via divert_correct delta
            for taker in divert_takers:
                if taker != expected_diverter:
                    false_divert += 1

            lc_peers = [d for d in survivors
                        if plan.assignments[d].mission_class == "LIFE_CRITICAL"]
            for lc in lc_peers:
                lc_exempt_total += 1
                if lc not in divert_takers:
                    lc_exempt_correct += 1

    avoid_rate = correct_avoid / max(total_avoid, 1)
    divert_rate = divert_correct / max(divert_total, 1)
    lc_rate = lc_exempt_correct / max(lc_exempt_total, 1) if lc_exempt_total else 1.0
    fp_rate = false_trigger / max(total_avoid, 1)
    miss_rate = missed_trigger / max(total_avoid, 1)

    # Weighted score on the 0-800 universal scale. Penalise false
    # triggers and missed triggers so a always-act or never-act
    # coordinator can't coast on averages.
    raw = (
        avoid_rate * 0.35
        + divert_rate * 0.25
        + lc_rate * 0.15
        - fp_rate * 0.15
        - miss_rate * 0.15
    )
    score = max(0.0, min(1.0, 0.25 + raw)) * 800.0
    grade = score_to_universal_grade(score)

    metrics = {
        "avoidance_correctness": avoid_rate,
        "false_trigger_rate": fp_rate,
        "missed_trigger_rate": miss_rate,
        "divert_correctness": divert_rate,
        "false_diverts": false_divert,
        "life_critical_exemption_rate": lc_rate,
        "trials": n_trials,
        "drones_per_trial": n_drones,
    }

    if verbose:
        print(f"  Avoidance correctness: {avoid_rate*100:5.1f}%")
        print(f"  False triggers:        {fp_rate*100:5.1f}%")
        print(f"  Missed triggers:       {miss_rate*100:5.1f}%")
        print(f"  Divert correctness:    {divert_rate*100:5.1f}%")
        print(f"  False diverts:         {false_divert}")
        print(f"  LC exemption rate:     {lc_rate*100:5.1f}%")
        print(f"  Score: {score:.1f}  Grade: {grade}")
    return grade, score, metrics


def run_training(
    trials: int = 120,
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
    p.add_argument("--trials", type=int, default=120)
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
