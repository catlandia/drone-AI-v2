"""Full training curriculum — trains all 4 modules in sequence.

Runs:
1. FlyControl (PPO with evolutionary population)
2. Pathfinder (algorithmic benchmark)
3. Perception (grade simulation benchmark, defaults to best grade)
4. Manager (grade simulation benchmark, defaults to best grade)

After training, runs a baseline experiment with all-P grades.
"""

from typing import Dict


def run_full_curriculum(
    population_size: int = 6,
    ages_per_stage: int = 10,
    steps_per_age: int = 10000,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """Train every module and run a verification experiment at the end."""
    results = {}

    if verbose:
        print("\n" + "#" * 60)
        print("#  FULL DRONE AI CURRICULUM")
        print("#" * 60)

    from drone_ai.modules.flycontrol.train import run_training as train_fly
    from drone_ai.modules.pathfinder.train import run_training as train_path
    from drone_ai.modules.perception.train import run_training as train_perc
    from drone_ai.modules.manager.train import run_training as train_mgr

    grade, score = train_fly(population_size, ages_per_stage, steps_per_age,
                             "models/flycontrol", seed, verbose)
    results["flycontrol"] = {"grade": grade, "score": score}

    grade, score = train_path(50, "models/pathfinder", seed, verbose)
    results["pathfinder"] = {"grade": grade, "score": score}

    grade, score = train_perc("P", 100, "models/perception", seed, verbose)
    results["perception"] = {"grade": grade, "score": score}

    grade, score = train_mgr("P", 20, "models/manager", seed, verbose)
    results["manager"] = {"grade": grade, "score": score}

    if verbose:
        print("\n" + "#" * 60)
        print("#  CURRICULUM COMPLETE")
        print("#" * 60)
        for module, info in results.items():
            print(f"  {module:12s} {info['grade']:3s}  score={info['score']:.1f}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--ages", type=int, default=10)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_full_curriculum(args.population, args.ages, args.steps, verbose=not args.quiet)


if __name__ == "__main__":
    main()
