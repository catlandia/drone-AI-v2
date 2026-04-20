"""Unified CLI entry point for drone-ai."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="drone-ai",
        description="Autonomous drone AI — 4-layer architecture with tier-based grading"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # train commands
    p_train = sub.add_parser("train", help="Train/benchmark an AI module")
    p_train.add_argument("module", choices=["flycontrol", "pathfinder", "perception", "manager", "all"])
    p_train.add_argument("--population", type=int, default=6)
    p_train.add_argument("--ages", type=int, default=10)
    p_train.add_argument("--steps", type=int, default=10000)
    p_train.add_argument("--trials", type=int, default=50)
    p_train.add_argument("--grade", default="P")
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--quiet", action="store_true")

    # experiment commands
    p_exp = sub.add_parser("experiment", help="Run grade-mixing experiments")
    p_exp.add_argument("preset", nargs="?", default=None,
                       help="Preset (all-P, all-F, blind-ace, etc.) or 'list'")
    p_exp.add_argument("--trials", type=int, default=10)
    p_exp.add_argument("--deliveries", type=int, default=5)
    p_exp.add_argument("--model", default=None, help="FlyControl model path")
    p_exp.add_argument("--output", default=None)

    # sweep
    p_sw = sub.add_parser("sweep", help="Vary one module's grade")
    p_sw.add_argument("module", choices=["flycontrol", "pathfinder", "perception", "manager"])
    p_sw.add_argument("--trials", type=int, default=5)
    p_sw.add_argument("--deliveries", type=int, default=5)
    p_sw.add_argument("--model", default=None)
    p_sw.add_argument("--output", default=None)

    # demo
    p_demo = sub.add_parser("demo", help="Run a quick demo mission")
    p_demo.add_argument("--model", default=None)
    p_demo.add_argument("--verbose", action="store_true")

    # curriculum
    p_curr = sub.add_parser("curriculum", help="Run full learning curriculum")
    p_curr.add_argument("--population", type=int, default=6)
    p_curr.add_argument("--ages", type=int, default=10)
    p_curr.add_argument("--steps", type=int, default=10000)
    p_curr.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    if args.command == "train":
        _run_train(args)
    elif args.command == "experiment":
        _run_experiment(args)
    elif args.command == "sweep":
        _run_sweep(args)
    elif args.command == "demo":
        _run_demo(args)
    elif args.command == "curriculum":
        _run_curriculum(args)


def _run_train(args):
    verbose = not args.quiet
    module = args.module
    if module in ("flycontrol", "all"):
        from drone_ai.modules.flycontrol.train import run_training as fc
        fc(args.population, args.ages, args.steps, "models/flycontrol", args.seed, verbose)
    if module in ("pathfinder", "all"):
        from drone_ai.modules.pathfinder.train import run_training as pf
        pf(args.trials, "models/pathfinder", args.seed, verbose)
    if module in ("perception", "all"):
        from drone_ai.modules.perception.train import run_training as pc
        pc(args.grade, args.trials, "models/perception", args.seed, verbose)
    if module in ("manager", "all"):
        from drone_ai.modules.manager.train import run_training as mg
        mg(args.grade, args.trials, "models/manager", args.seed, verbose)


def _run_experiment(args):
    from drone_ai.experiment import run_preset, save_results

    if args.preset is None or args.preset == "list":
        print("Available presets:")
        for name in ("all-P all-A all-C all-F all-W "
                     "blind-ace clumsy-seer lost-genius confused-boss "
                     "legendary nightmare").split():
            print(f"  {name}")
        return

    r = run_preset(
        args.preset,
        n_trials=args.trials,
        deliveries=args.deliveries,
        flycontrol_model=args.model,
    )
    print(f"\n  completion_rate = {r['completion_rate']*100:.1f}%")
    print(f"  crash_rate      = {r['crash_rate']*100:.1f}%")
    print(f"  avg_steps       = {r['avg_steps']:.0f}")
    if args.output:
        save_results(r, args.output)


def _run_sweep(args):
    from drone_ai.experiment import run_tier_sweep, save_results
    results = run_tier_sweep(
        args.module, n_trials=args.trials,
        deliveries=args.deliveries, flycontrol_model=args.model,
    )
    if args.output:
        save_results(results, args.output)


def _run_demo(args):
    from drone_ai import DroneAI
    from drone_ai.modules.manager.planner import Priority
    import numpy as np

    drone = DroneAI(flycontrol_model=args.model)
    drone.reset()
    drone.add_delivery([30, 20, 0], Priority.URGENT)
    drone.add_delivery([-25, 35, 0], Priority.NORMAL)
    drone.add_delivery([40, -30, 0], Priority.CRITICAL)

    # Add some obstacles
    from drone_ai.simulation.world import World, Obstacle
    import numpy as np
    rng = np.random.default_rng(123)
    w = World()
    w.generate_random_obstacles(10, rng)
    drone.set_obstacles(w.obstacles)

    print("Running demo mission...")
    summary = drone.run(max_steps=10000, verbose=args.verbose)
    print(f"\nResult:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


def _run_curriculum(args):
    from drone_ai.curriculum import run_full_curriculum
    run_full_curriculum(
        population_size=args.population,
        ages_per_stage=args.ages,
        steps_per_age=args.steps,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
