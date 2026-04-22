"""Storage benchmark — exercise the on-drone Layer 6 logger.

Storage doesn't "train" — it's the field log Layer 7 reads. The
benchmark here verifies the log path round-trips: write each record
type, read it back, confirm the summary matches expectations.

Useful as a UI-launchable sanity check that the field log channel
is healthy on this drone before deploying.

Writes a graded `.pt` placeholder under models/storage/ and appends
a row to models/runs.csv tagged module=storage.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from drone_ai.grading import (
    RunLogger, RunRecord, generate_model_name, next_version,
    score_to_universal_grade,
)
from drone_ai.modules.storage import (
    Storage, UpdateRecord, MissionRecord, MissionOutcome, UpstreamCause,
)


def benchmark(
    n_missions: int = 20,
    seed: int = 42,
    verbose: bool = True,
    storage_root: str = "logs/storage",
    drone_id: str = "bench",
) -> Tuple[str, float, dict]:
    """Write n_missions worth of synthetic field rows, read them back,
    score the round-trip integrity."""
    rng = np.random.default_rng(seed)
    s = Storage(drone_id, root=storage_root)

    expected_missions = 0
    expected_updates = 0
    expected_delivered = 0
    expected_crashed = 0

    outcomes = list(MissionOutcome)
    for i in range(n_missions):
        mid = f"bench-{i}"

        # 1-3 update rows per mission (some accepted, some rejected,
        # occasional rollback) — mirrors what AdaptiveLearner emits.
        for _ in range(int(rng.integers(1, 4))):
            accepted = bool(rng.random() < 0.6)
            rolled_back = (not accepted) and bool(rng.random() < 0.2)
            s.record_update(UpdateRecord(
                mission_id=mid, layer="flycontrol",
                accepted=accepted,
                rejected_reason=None if accepted else "warden_score_drop",
                warden_score_pre=float(rng.uniform(-5, 15)),
                warden_score_post=float(rng.uniform(-5, 15)),
                rollback_triggered=rolled_back,
            ))
            expected_updates += 1

        outcome = outcomes[int(rng.integers(0, len(outcomes)))]
        if outcome == MissionOutcome.DELIVERED:
            expected_delivered += 1
        elif outcome == MissionOutcome.CRASHED:
            expected_crashed += 1

        s.record_mission(MissionRecord(
            mission_id=mid, outcome=outcome,
            deadline_type="HARD" if rng.random() < 0.3 else "SOFT",
            mission_class="LIFE_CRITICAL" if rng.random() < 0.2 else "STANDARD",
            preflight_margin=float(rng.uniform(0.05, 0.3)),
            deliveries_done=int(rng.integers(0, 5)),
            deliveries_total=int(rng.integers(1, 6)),
            upstream_cause=UpstreamCause.UNKNOWN if outcome == MissionOutcome.CRASHED else None,
        ))
        expected_missions += 1

    # Round-trip read.
    summary = s.summary()
    miss_ok = summary["missions_total"] == expected_missions
    upd_ok = summary["updates_total"] == expected_updates
    deliv_ok = summary["missions_by_outcome"]["delivered"] == expected_delivered
    crash_ok = summary["missions_by_outcome"]["crashed"] == expected_crashed

    correct = sum([miss_ok, upd_ok, deliv_ok, crash_ok])
    score = (correct / 4.0) * 800.0
    grade = score_to_universal_grade(score)

    metrics = {
        "round_trip_correct": correct,
        "round_trip_total": 4,
        "missions_written": expected_missions,
        "updates_written": expected_updates,
        "summary": summary,
    }
    if verbose:
        print(f"  Missions written/read: {expected_missions}/{summary['missions_total']}  ok={miss_ok}")
        print(f"  Updates written/read:  {expected_updates}/{summary['updates_total']}  ok={upd_ok}")
        print(f"  Score: {score:.1f}  Grade: {grade}")
    return grade, score, metrics


def run_training(
    n_missions: int = 20,
    save_dir: str = "models/storage",
    seed: int = 42,
    verbose: bool = True,
    log_path: str = "models/runs.csv",
    run_tag: str = "",
    storage_root: str = "logs/storage",
) -> Tuple[str, float]:
    if verbose:
        print(f"\n{'='*60}\n  STORAGE OF LEARNINGS BENCHMARK\n{'='*60}")
    grade, score, metrics = benchmark(
        n_missions, seed, verbose, storage_root=storage_root, drone_id="bench",
    )

    version = next_version(save_dir, "storage")
    fname = generate_model_name(grade, "storage", version)
    path = Path(save_dir) / fname
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"grade": grade, "score": score, "metrics": metrics}, str(path))
    with open(path.with_suffix(".json"), "w") as f:
        json.dump({
            "grade": grade, "score": score, "metrics": metrics,
            "timestamp": datetime.now().isoformat(), "model_file": fname,
        }, f, indent=2, default=str)

    try:
        RunLogger(log_path).append(RunRecord(
            module="storage", stage="round_trip",
            best_score=score, avg_score=score, grade=grade,
            minutes=0.0, episodes=n_missions, run_tag=run_tag,
        ))
    except Exception as e:
        print(f"[storage] run-log append failed: {e}")

    if verbose:
        print(f"  Saved: {path}")
    return grade, score


def main():
    p = argparse.ArgumentParser(description="Benchmark Storage Layer 6")
    p.add_argument("--missions", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", default="models/storage")
    p.add_argument("--storage-root", default="logs/storage")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--run-tag", default="")
    args = p.parse_args()
    run_training(args.missions, args.save_dir, args.seed, not args.quiet,
                 run_tag=args.run_tag, storage_root=args.storage_root)


if __name__ == "__main__":
    main()
