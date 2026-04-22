"""Personality benchmark — auto-export from the latest FlyControl checkpoint.

Personalities are deltas. To exercise the export+apply round-trip
without an existing proven drone, we:
  1. Find the latest FlyControl checkpoint on disk (any stage).
  2. Build a "proven" copy by mutating it slightly.
  3. Export the personality (proven - baseline).
  4. Apply it to a fresh baseline copy.
  5. Score the round-trip on weight-difference recovery: how close
     does (baseline + delta) get to the proven weights?

Writes a graded `.pt` placeholder under models/personality/ and
appends a row to models/runs.csv tagged module=personality.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from drone_ai.grading import (
    RunLogger, RunRecord, generate_model_name, next_version,
    score_to_universal_grade, parse_model_name,
)
from drone_ai.modules.flycontrol.agent import PPOAgent, PPOConfig
from drone_ai.modules.flycontrol.environment import OBS_DIM, ACT_DIM
from drone_ai.modules.personality import (
    Personality, export_personality, apply_personality,
)


def _newest_flycontrol_checkpoint(models_root: str = "models") -> Optional[str]:
    """Walk models/flycontrol/<stage>/ and return the newest .pt."""
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
                candidates.append((parsed["version"], parsed["date"],
                                   os.path.join(sd, fname)))
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[1], c[0]), reverse=True)
    return candidates[0][2]


def benchmark(
    seed: int = 42,
    verbose: bool = True,
    out_artifact: Optional[str] = None,
    models_root: str = "models",
) -> Tuple[str, float, dict]:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    baseline_path = _newest_flycontrol_checkpoint(models_root)
    if baseline_path is None:
        # No flycontrol checkpoints yet — build a fresh baseline so we
        # can still exercise the round-trip.
        baseline = PPOAgent(OBS_DIM, ACT_DIM, PPOConfig())
        baseline_name = "fresh_init"
        if verbose:
            print("  No flycontrol checkpoints found — using fresh init as baseline.")
    else:
        baseline = PPOAgent.from_file(baseline_path)
        baseline_name = os.path.basename(baseline_path)
        if verbose:
            print(f"  Baseline: {baseline_path}")

    # "Proven" = baseline + small noise. In real life this would be
    # the post-Layer-5 fine-tuned weights.
    proven = baseline.clone()
    with torch.no_grad():
        for p in proven.policy.parameters():
            p.add_(torch.randn_like(p) * 0.02)

    # Export.
    personality = export_personality(
        proven, baseline,
        source_drone_id="bench",
        baseline_name=baseline_name,
        confidence=0.6,
    )

    # Apply to a fresh copy of baseline.
    target = baseline.clone()
    applied_names = apply_personality(target, personality)

    # Round-trip metric: ‖proven - target‖ / ‖proven - baseline‖.
    # 0 = perfect recovery (target == proven). 1.0 = no recovery
    # (target == baseline). Anything below 1e-3 is essentially exact.
    proven_state = proven.policy.state_dict()
    base_state = baseline.policy.state_dict()
    target_state = target.policy.state_dict()

    num = 0.0
    denom = 0.0
    for k in proven_state:
        num += float(torch.linalg.vector_norm(proven_state[k] - target_state[k]) ** 2)
        denom += float(torch.linalg.vector_norm(proven_state[k] - base_state[k]) ** 2)
    recovery_residual = (num / max(denom, 1e-12)) ** 0.5

    # Score: 0 residual → max score; 1.0 residual → 0 score.
    score = max(0.0, (1.0 - recovery_residual)) * 800.0
    grade = score_to_universal_grade(score)

    metrics = {
        "baseline": baseline_name,
        "applied_tensors": len(applied_names),
        "total_delta_tensors": len(personality.weight_deltas),
        "recovery_residual": recovery_residual,
        "score": score,
        "grade": grade,
    }

    if out_artifact is not None:
        personality.save(out_artifact)
        metrics["artifact_path"] = out_artifact

    if verbose:
        print(f"  Tensors applied: {len(applied_names)}/{len(personality.weight_deltas)}")
        print(f"  Recovery residual: {recovery_residual:.2e} (lower is better)")
        print(f"  Score: {score:.1f}  Grade: {grade}")
        if out_artifact:
            print(f"  Wrote artifact: {out_artifact}")
    return grade, score, metrics


def run_training(
    save_dir: str = "models/personality",
    seed: int = 42,
    verbose: bool = True,
    log_path: str = "models/runs.csv",
    run_tag: str = "",
) -> Tuple[str, float]:
    if verbose:
        print(f"\n{'='*60}\n  PERSONALITY EXPORT BENCHMARK\n{'='*60}")

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = str(out_dir / f"bench_{int(seed)}.personality.pt")
    grade, score, metrics = benchmark(seed, verbose, out_artifact=artifact_path)

    version = next_version(save_dir, "personality")
    fname = generate_model_name(grade, "personality", version)
    path = out_dir / fname
    torch.save({"grade": grade, "score": score, "metrics": metrics}, str(path))
    with open(path.with_suffix(".json"), "w") as f:
        json.dump({
            "grade": grade, "score": score, "metrics": metrics,
            "timestamp": datetime.now().isoformat(), "model_file": fname,
        }, f, indent=2)

    try:
        RunLogger(log_path).append(RunRecord(
            module="personality", stage="export_round_trip",
            best_score=score, avg_score=score, grade=grade,
            minutes=0.0, episodes=1, run_tag=run_tag,
        ))
    except Exception as e:
        print(f"[personality] run-log append failed: {e}")

    if verbose:
        print(f"  Saved: {path}")
    return grade, score


def main():
    p = argparse.ArgumentParser(description="Benchmark Personality Layer 7 export")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", default="models/personality")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--run-tag", default="")
    args = p.parse_args()
    run_training(args.save_dir, args.seed, not args.quiet, run_tag=args.run_tag)


if __name__ == "__main__":
    main()
