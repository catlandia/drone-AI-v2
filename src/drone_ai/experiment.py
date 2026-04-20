"""Mixed-grade experiment runner.

Lets you combine any grades for the 4 modules and measure drone behavior.
Classic experiments:
  - All P     → perfect drone
  - All F     → disaster
  - P-fly + C-perception → flies well but crashes into missed obstacles
  - P-everything except W-manager → gets confused about what to deliver
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

from drone_ai.drone import DroneAI, GradeConfig
from drone_ai.modules.manager.planner import Priority
from drone_ai.grading import GRADE_ORDER
from drone_ai.simulation.world import World


def run_experiment(
    grades: GradeConfig,
    n_trials: int = 10,
    deliveries: int = 5,
    max_steps: int = 15000,
    obstacle_count: int = 15,
    flycontrol_model: Optional[str] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """Run a single experiment with fixed grade combination."""
    rng = np.random.default_rng(seed)

    trials = []
    for trial_i in range(n_trials):
        drone = DroneAI(
            grades=grades,
            flycontrol_model=flycontrol_model,
            seed=seed + trial_i,
        )
        drone.reset()

        # Generate random world
        world = World()
        world.generate_random_obstacles(obstacle_count, np.random.default_rng(seed + trial_i))
        drone.set_obstacles(world.obstacles)

        # Add random deliveries
        for _ in range(deliveries):
            t = rng.uniform([-80, -80, 0], [80, 80, 0])
            p = rng.choice([Priority.NORMAL, Priority.URGENT, Priority.CRITICAL])
            drone.add_delivery(t, p)

        summary = drone.run(max_steps=max_steps, verbose=False)
        trials.append(summary)

        if verbose:
            c = summary["completed"]
            f = summary.get("crashed", False)
            print(f"  Trial {trial_i+1}: {c}/{deliveries} delivered  crashed={f}  steps={summary['steps']}")

    # Aggregate
    agg = {
        "completion_rate": float(np.mean([t["completed"] / deliveries for t in trials])),
        "crash_rate": float(np.mean([t.get("crashed", False) for t in trials])),
        "avg_steps": float(np.mean([t["steps"] for t in trials])),
        "avg_distance_m": float(np.mean([t.get("total_distance_m", 0) for t in trials])),
        "n_trials": n_trials,
        "grades": {
            "flycontrol": grades.flycontrol,
            "pathfinder": grades.pathfinder,
            "perception": grades.perception,
            "manager": grades.manager,
        },
    }
    return agg


def run_preset(preset: str, verbose: bool = True, **kwargs) -> Dict:
    """Run a named preset experiment."""
    presets = {
        "all-P": GradeConfig("P", "P", "P", "P"),
        "all-A": GradeConfig("A", "A", "A", "A"),
        "all-C": GradeConfig("C", "C", "C", "C"),
        "all-F": GradeConfig("F", "F", "F", "F"),
        "all-W": GradeConfig("W", "W", "W", "W"),
        "blind-ace":      GradeConfig("P", "P", "F", "P"),  # perfect flyer, blind
        "clumsy-seer":    GradeConfig("F", "P", "P", "P"),  # can't fly, sees everything
        "lost-genius":    GradeConfig("P", "F", "P", "P"),  # can't plan routes
        "confused-boss":  GradeConfig("P", "P", "P", "F"),  # bad mission manager
        "legendary":      GradeConfig("P", "P", "P", "P"),
        "nightmare":      GradeConfig("W", "W", "W", "W"),
    }
    if preset not in presets:
        raise ValueError(f"Unknown preset. Options: {list(presets.keys())}")
    if verbose:
        print(f"\n{'='*60}")
        print(f"  EXPERIMENT: {preset}")
        print(f"  {presets[preset].flycontrol} / {presets[preset].pathfinder} / "
              f"{presets[preset].perception} / {presets[preset].manager}")
        print(f"{'='*60}")
    return run_experiment(presets[preset], verbose=verbose, **kwargs)


def run_tier_sweep(
    module: str = "perception",
    n_trials: int = 5,
    verbose: bool = True,
    **kwargs,
) -> List[Dict]:
    """Vary one module's grade while keeping others at P. See how it affects outcomes."""
    results = []
    for grade in GRADE_ORDER:
        grades = GradeConfig("P", "P", "P", "P")
        setattr(grades, module, grade)
        if verbose:
            print(f"\n--- {module}={grade} ---")
        r = run_experiment(grades, n_trials=n_trials, verbose=False, **kwargs)
        r["varied_module"] = module
        r["varied_grade"] = grade
        results.append(r)
        if verbose:
            print(f"  completion={r['completion_rate']*100:.0f}%  crash={r['crash_rate']*100:.0f}%")
    return results


def save_results(results, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "results": results}, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run grade-mixing experiments")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run a preset experiment")
    p_run.add_argument("preset", help="Preset name (e.g. all-P, blind-ace, all-F)")
    p_run.add_argument("--trials", type=int, default=10)
    p_run.add_argument("--deliveries", type=int, default=5)
    p_run.add_argument("--model", default=None, help="Path to flycontrol .pt model")
    p_run.add_argument("--output", default=None)

    p_custom = sub.add_parser("custom", help="Custom grade combination")
    p_custom.add_argument("--flycontrol", default="P", choices=GRADE_ORDER)
    p_custom.add_argument("--pathfinder", default="P", choices=GRADE_ORDER)
    p_custom.add_argument("--perception", default="P", choices=GRADE_ORDER)
    p_custom.add_argument("--manager", default="P", choices=GRADE_ORDER)
    p_custom.add_argument("--trials", type=int, default=10)
    p_custom.add_argument("--deliveries", type=int, default=5)
    p_custom.add_argument("--model", default=None)
    p_custom.add_argument("--output", default=None)

    p_sweep = sub.add_parser("sweep", help="Vary one module's grade")
    p_sweep.add_argument("module", choices=["flycontrol", "pathfinder", "perception", "manager"])
    p_sweep.add_argument("--trials", type=int, default=5)
    p_sweep.add_argument("--deliveries", type=int, default=5)
    p_sweep.add_argument("--model", default=None)
    p_sweep.add_argument("--output", default=None)

    args = parser.parse_args()

    if args.cmd == "run":
        r = run_preset(args.preset, n_trials=args.trials, deliveries=args.deliveries,
                       flycontrol_model=args.model)
        print(f"\n  completion_rate={r['completion_rate']*100:.1f}%")
        print(f"  crash_rate={r['crash_rate']*100:.1f}%")
        if args.output:
            save_results(r, args.output)

    elif args.cmd == "custom":
        g = GradeConfig(args.flycontrol, args.pathfinder, args.perception, args.manager)
        r = run_experiment(g, n_trials=args.trials, deliveries=args.deliveries,
                           flycontrol_model=args.model)
        print(f"\n  completion_rate={r['completion_rate']*100:.1f}%")
        print(f"  crash_rate={r['crash_rate']*100:.1f}%")
        if args.output:
            save_results(r, args.output)

    elif args.cmd == "sweep":
        results = run_tier_sweep(args.module, n_trials=args.trials,
                                 deliveries=args.deliveries, flycontrol_model=args.model)
        if args.output:
            save_results(results, args.output)


if __name__ == "__main__":
    main()
