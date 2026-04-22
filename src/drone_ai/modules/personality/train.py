"""Personality benchmark — test delta transfer across noisy baselines.

A Personality is only useful if its delta carries the "personality"
(the Layer-5-learned bits) in a way that survives being applied to a
different-but-related drone. The trivial case — apply the exact
delta to the exact source baseline — always reconstructs perfectly
and tells us nothing.

This benchmark:
  1. Pick a baseline (auto-discover the newest FlyControl checkpoint;
     fall back to a fresh PPOAgent).
  2. Build a "proven" drone by mutating the baseline.
  3. Export the delta.
  4. For each of several NOISY sibling baselines (baseline + small
     Gaussian noise), apply the delta and measure how close the
     result is to the proven target.
  5. Score on the average recovery across those noisy siblings.

A correct implementation scores high (~800) when the delta transfer
works well; a broken one (wrong parameter shapes, silently dropped
tensors, etc.) scores much lower because noisy siblings aren't
exactly the baseline and the delta alone can't paper over that.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

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


def _mutate(agent: PPOAgent, noise_std: float, rng: np.random.Generator) -> PPOAgent:
    child = agent.clone()
    g = torch.Generator()
    g.manual_seed(int(rng.integers(0, 2**31)))
    with torch.no_grad():
        for p in child.policy.parameters():
            p.add_(torch.randn(p.shape, generator=g) * noise_std)
    return child


def _recovery_residual(proven: PPOAgent, target: PPOAgent, baseline: PPOAgent) -> float:
    proven_state = proven.policy.state_dict()
    base_state = baseline.policy.state_dict()
    target_state = target.policy.state_dict()
    num = 0.0
    denom = 0.0
    for k in proven_state:
        num += float(torch.linalg.vector_norm(proven_state[k] - target_state[k]) ** 2)
        denom += float(torch.linalg.vector_norm(proven_state[k] - base_state[k]) ** 2)
    return (num / max(denom, 1e-12)) ** 0.5


def benchmark(
    seed: int = 42,
    verbose: bool = True,
    out_artifact: Optional[str] = None,
    models_root: str = "models",
    n_siblings: int = 5,
    sibling_noise: float = 0.02,
    proven_noise: float = 0.04,
) -> Tuple[str, float, dict]:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    baseline_path = _newest_flycontrol_checkpoint(models_root)
    if baseline_path is None:
        baseline = PPOAgent(OBS_DIM, ACT_DIM, PPOConfig())
        baseline_name = "fresh_init"
        if verbose:
            print("  No flycontrol checkpoints found — using fresh init as baseline.")
    else:
        baseline = PPOAgent.from_file(baseline_path)
        baseline_name = os.path.basename(baseline_path)
        if verbose:
            print(f"  Baseline: {baseline_path}")

    # Proven drone: baseline + stronger noise (the "personality").
    proven = _mutate(baseline, proven_noise, rng)

    # Export.
    personality = export_personality(
        proven, baseline,
        source_drone_id="bench",
        baseline_name=baseline_name,
        confidence=0.6,
    )

    # Apply to N noisy siblings. Residual is measured against proven
    # because that's the drone we're trying to replicate.
    residuals: List[float] = []
    for i in range(n_siblings):
        sibling = _mutate(baseline, sibling_noise, rng)
        target = sibling.clone()
        apply_personality(target, personality)
        residuals.append(_recovery_residual(proven, target, baseline))

    # Also record the trivial residual — apply to the exact baseline.
    # A correct delta drives this to ~0 (sanity check, not the grade).
    trivial = baseline.clone()
    apply_personality(trivial, personality)
    trivial_residual = _recovery_residual(proven, trivial, baseline)

    mean_res = float(np.mean(residuals))
    max_res = float(np.max(residuals))

    # Score: residual of 0 → full marks, residual of 1 → 0 marks,
    # blended between mean and worst-case so a single noisy sibling
    # can't sink the grade.
    effective = 0.7 * mean_res + 0.3 * max_res
    score = max(0.0, min(1.0, 1.0 - effective)) * 800.0
    grade = score_to_universal_grade(score)

    metrics = {
        "baseline": baseline_name,
        "n_siblings": n_siblings,
        "sibling_noise": sibling_noise,
        "proven_noise": proven_noise,
        "mean_residual": mean_res,
        "max_residual": max_res,
        "trivial_residual": trivial_residual,
        "applied_tensors": len(personality.weight_deltas),
        "score": score,
        "grade": grade,
    }

    if out_artifact is not None:
        personality.save(out_artifact)
        metrics["artifact_path"] = out_artifact

    if verbose:
        print(f"  Tensors in artifact: {len(personality.weight_deltas)}")
        print(f"  Trivial residual:   {trivial_residual:.3e}  (same baseline)")
        print(f"  Sibling residual:   mean={mean_res:.3f} max={max_res:.3f}")
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
            module="personality", stage="sibling_transfer",
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
