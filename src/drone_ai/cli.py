"""Unified CLI entry point for drone-ai."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="drone-ai",
        description="Autonomous drone AI — 8-layer architecture with tier-based grading"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # train commands
    p_train = sub.add_parser("train", help="Train/benchmark an AI module")
    p_train.add_argument("module",
                         choices=["flycontrol", "pathfinder", "perception",
                                  "manager", "adaptive", "all"])
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

    # storage — Layer 6 inspection
    p_st = sub.add_parser("storage", help="Inspect Layer 6 Storage of Learnings")
    p_st.add_argument("drone_id", help="Drone id whose log to summarize")
    p_st.add_argument("--root", default="logs/storage")

    # launch — open the visual launcher (primary entry point for users)
    sub.add_parser(
        "launch",
        help="Open the visual launcher (inspect every module in its own window)",
    )

    # personality — Layer 7 export / inspect
    p_pers = sub.add_parser("personality",
                            help="Export or inspect a Layer 7 personality artifact")
    p_pers_sub = p_pers.add_subparsers(dest="pers_cmd", required=True)
    p_pers_export = p_pers_sub.add_parser("export",
                                          help="Build a personality from proven vs baseline checkpoints")
    p_pers_export.add_argument("--baseline", required=True, help="Baseline FlyControl .pt")
    p_pers_export.add_argument("--proven", required=True, help="Proven FlyControl .pt")
    p_pers_export.add_argument("--drone-id", required=True)
    p_pers_export.add_argument("--out", required=True, help="Path to write the personality .pt")
    p_pers_export.add_argument("--confidence", type=float, default=0.5)
    p_pers_inspect = p_pers_sub.add_parser("inspect",
                                           help="Print a summary of a personality file")
    p_pers_inspect.add_argument("path")

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
    elif args.command == "storage":
        _run_storage(args)
    elif args.command == "personality":
        _run_personality(args)
    elif args.command == "launch":
        from drone_ai.viz.launcher import Launcher
        Launcher().run()


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
    if module == "adaptive":
        # Adaptive requires a trained FlyControl model as its starting point.
        # Defer to the module's own entry point for the richer CLI.
        print("Use: py -m drone_ai.modules.adaptive.train --model <path> [opts]")
        print("(the adaptive module is graded by improvement delta on OOD envs)")


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


def _run_storage(args):
    from drone_ai.modules.storage import Storage
    s = Storage(args.drone_id, root=args.root)
    summary = s.summary()
    if summary["missions_total"] == 0 and summary["updates_total"] == 0:
        print(f"No log entries for drone '{args.drone_id}' in {args.root}")
        return
    print(f"Storage summary for drone '{args.drone_id}':")
    for k, v in summary.items():
        print(f"  {k}: {v}")


def _run_personality(args):
    from drone_ai.modules.personality import Personality, export_personality
    from drone_ai.modules.flycontrol.agent import PPOAgent

    if args.pers_cmd == "export":
        baseline = PPOAgent.from_file(args.baseline)
        proven = PPOAgent.from_file(args.proven)
        import os
        p = export_personality(
            proven, baseline,
            source_drone_id=args.drone_id,
            baseline_name=os.path.basename(args.baseline),
            confidence=args.confidence,
        )
        p.save(args.out)
        print(f"Wrote personality artifact -> {args.out}")
        print(f"  source_drone_id={p.source_drone_id}")
        print(f"  baseline={p.baseline_name}")
        print(f"  delta tensors: {len(p.weight_deltas)}")
    elif args.pers_cmd == "inspect":
        p = Personality.load(args.path)
        print(f"Personality {args.path}")
        print(f"  source_drone_id={p.source_drone_id}")
        print(f"  baseline={p.baseline_name}")
        print(f"  created_at={p.created_at}")
        print(f"  confidence={p.confidence}")
        print(f"  delta tensors: {len(p.weight_deltas)}")
        print(f"  hparams: {p.hparams}")
        print(f"  warden_stats: {p.warden_stats}")
        print(f"  rollback_stats: {p.rollback_stats}")
        print(f"  soft_bound_promotions: {p.soft_bound_promotions}")


if __name__ == "__main__":
    main()
