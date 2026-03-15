#!/usr/bin/env python3
"""Run a workload episode with the no-op controller (baseline: no adaptation)."""

import argparse
from engine.episode import EpisodeRunner, ControllerInterface


class NoOpController(ControllerInterface):
    """Does nothing. All streams stay in their initial mode (FULL)."""
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="scenario_ramp_up")
    parser.add_argument("--config-dir", default="/app/config")
    parser.add_argument("--output-dir", default="/data")
    args = parser.parse_args()

    controller = NoOpController()
    runner = EpisodeRunner(
        scenario_name=args.scenario,
        config_dir=args.config_dir,
        controller=controller,
    )
    runner.run(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
