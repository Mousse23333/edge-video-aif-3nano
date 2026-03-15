#!/usr/bin/env python3
"""Run all scenarios with the heuristic controller."""

import argparse
from engine.episode import EpisodeRunner
from controllers.heuristic import HeuristicController


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=None,
                        help="Single scenario name, or run all if omitted")
    parser.add_argument("--config-dir", default="/app/config")
    parser.add_argument("--output-dir", default="/data")
    args = parser.parse_args()

    scenarios = [args.scenario] if args.scenario else [
        "scenario_ramp_up",
        "scenario_burst",
        "scenario_steady_overload",
    ]

    for scenario_name in scenarios:
        controller = HeuristicController(config_dir=args.config_dir)
        runner = EpisodeRunner(
            scenario_name=scenario_name,
            config_dir=args.config_dir,
            controller=controller,
        )
        runner.run(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
