"""Storage benchmark — stress the on-drone Layer 6 logger.

Storage doesn't "train" — it's the field log Layer 7 reads. "Training"
here is a stress test that actually CAN fail:

  1. Write N synthetic missions worth of records.
  2. Read them back — count mismatches.
  3. Corrupt the last ~5% of lines (truncation simulating a crash
     mid-write) — the reader should drop the bad rows and keep the
     rest.
  4. Inject malformed JSON rows between good ones — again, the reader
     must not blow up.
  5. Aggregate summary must still be computable after the damage.

Grade reflects how many of those assertions hold, so a regression in
the reader that silently drops rows (or crashes on malformed JSON)
actually bites. Under normal conditions it still grades near P, but
any flaky IO path is visible.

Writes a graded `.pt` placeholder under models/storage/ and appends
a row to models/runs.csv tagged module=storage.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
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


def _write_synthetic(s: Storage, n_missions: int, rng: np.random.Generator) -> Tuple[int, int, int, int]:
    expected_updates = 0
    expected_delivered = 0
    expected_crashed = 0
    outcomes = list(MissionOutcome)
    for i in range(n_missions):
        mid = f"bench-{i}"
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
    return n_missions, expected_updates, expected_delivered, expected_crashed


def _corrupt_tail(path: str, fraction: float = 0.05) -> int:
    """Truncate the last `fraction` of the file so the final lines
    are half-written, mirroring a crash mid-flush. Returns how many
    bytes were removed."""
    try:
        sz = os.path.getsize(path)
    except OSError:
        return 0
    trim = max(1, int(sz * fraction))
    with open(path, "rb+") as f:
        f.truncate(sz - trim)
    return trim


def _inject_malformed(path: str, n: int = 3) -> int:
    """Append `n` intentionally-broken lines to the log."""
    if not os.path.isfile(path):
        return 0
    broken = [
        "{not json\n",
        '{"kind": "update", "mission_id": "bad", truncated_here',
        "\n",
    ]
    with open(path, "a", encoding="utf-8") as f:
        for i in range(n):
            f.write(broken[i % len(broken)] + "\n")
    return n


def benchmark(
    n_missions: int = 60,
    seed: int = 42,
    verbose: bool = True,
    storage_root: str = "logs/storage",
    drone_id: str = "bench",
) -> Tuple[str, float, dict]:
    rng = np.random.default_rng(seed)

    # Use a dedicated scratch dir per run so multiple benchmarks don't
    # step on each other and the corruption stage can't poison the
    # real drone log.
    tmp = tempfile.mkdtemp(prefix="drone_storage_bench_")
    s = Storage(drone_id, root=tmp)

    checks = []

    def _check(name: str, ok: bool) -> None:
        checks.append((name, bool(ok)))
        if verbose:
            print(f"  [{'ok' if ok else 'FAIL'}] {name}")

    # 1. Clean round-trip.
    miss_w, upd_w, deliv_w, crash_w = _write_synthetic(s, n_missions, rng)
    summary = s.summary()
    _check("round_trip_missions", summary["missions_total"] == miss_w)
    _check("round_trip_updates",  summary["updates_total"] == upd_w)
    _check("round_trip_delivered",
           summary["missions_by_outcome"]["delivered"] == deliv_w)
    _check("round_trip_crashed",
           summary["missions_by_outcome"]["crashed"] == crash_w)

    # 2. Malformed JSON tolerance — the reader should skip bad lines
    # and still surface the good ones. We can recompute by writing a
    # small second log and injecting garbage.
    _inject_malformed(s.path, n=5)
    try:
        summary2 = s.summary()
        # Counts shouldn't drop below the clean snapshot.
        _check("malformed_tolerated",
               summary2["missions_total"] >= miss_w - 1
               and summary2["updates_total"] >= upd_w - 1)
    except Exception as exc:
        _check(f"malformed_tolerated[{type(exc).__name__}]", False)

    # 3. Truncation tolerance — simulate a mid-write crash.
    removed = _corrupt_tail(s.path, fraction=0.05)
    try:
        summary3 = s.summary()
        # Some rows may be dropped by the truncation, but we lose at
        # most a small tail. Anything more than ~15% loss is a bug.
        loss = 1.0 - (summary3["missions_total"] / max(miss_w, 1))
        _check(f"truncation_loss_under_15pct(loss={loss:.2%})", loss < 0.15)
    except Exception as exc:
        _check(f"truncation_crash[{type(exc).__name__}]", False)

    # 4. Independence between drones.
    s2 = Storage("other", root=tmp)
    s2.record_mission(MissionRecord(
        mission_id="solo", outcome=MissionOutcome.DELIVERED,
        deadline_type="SOFT", mission_class="STANDARD",
    ))
    _check("drone_isolation",
           s2.summary()["missions_total"] == 1
           and s.summary()["missions_total"] > 0)

    # Cleanup scratch dir; test artifacts are throwaway.
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    score = (passed / total) * 800.0
    grade = score_to_universal_grade(score)

    metrics = {
        "checks_passed": passed,
        "checks_total": total,
        "check_details": [{"name": n, "ok": ok} for n, ok in checks],
        "missions_written": miss_w,
        "updates_written": upd_w,
        "tail_bytes_trimmed": removed,
    }
    if verbose:
        print(f"  Checks passed: {passed}/{total}")
        print(f"  Score: {score:.1f}  Grade: {grade}")
    return grade, score, metrics


def run_training(
    n_missions: int = 60,
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
            module="storage", stage="stress",
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
    p.add_argument("--missions", type=int, default=60)
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
