#!/usr/bin/env python3
"""
Run all controllers × all scenarios and save comparison results.
RL controllers get a training phase before evaluation.
"""

import json
import time
import argparse
from engine.episode import EpisodeRunner, ControllerInterface
from controllers.heuristic import HeuristicController
from controllers.myopic_greedy import MyopicGreedyController
from controllers.rl_tabular import TabularQLearningController
from controllers.rl_reinforce import REINFORCEController
from controllers.rl_dqn import DQNController
from controllers.aif import AIFController


CONTROLLERS = {
    "noop":        ControllerInterface,
    "heuristic":   HeuristicController,
    "myopic":      MyopicGreedyController,
    "tabular_ql":  TabularQLearningController,
    "reinforce":   REINFORCEController,
    "dqn":         DQNController,
    "aif":         AIFController,
}

# Controllers that need a training phase
RL_CONTROLLERS = {"tabular_ql", "reinforce", "dqn"}

SCENARIOS = [
    "scenario_ramp_up",
    "scenario_burst",
    "scenario_steady_overload",
    "scenario_oscillating",
]


def extract_summary(controller_name, scenario_name, history):
    """Extract summary metrics from episode history."""
    total_windows = 0
    total_violations = 0
    n_switches = sum(1 for r in history if r.get("action") is not None)
    skip_windows = 0
    offload_windows = 0

    for rec in history:
        for sid, ps in rec["observation"]["per_stream"].items():
            mode = ps.get("current_mode", "FULL")
            if mode == "SKIP":
                skip_windows += 1
                continue
            if mode == "OFFLOAD":
                offload_windows += 1
                # OFFLOAD streams: check SLO against remote inference metrics
                total_windows += 1
                lat = ps.get("infer_latency_p95_ms", 0) or ps.get("latency_p95_ms", 0)
                fps = ps.get("infer_fps", 0) or ps.get("fps_avg", 0)
                if lat > 150 or fps < 10:
                    total_violations += 1
                continue
            if mode not in ("FULL", "LITE"):
                continue
            total_windows += 1
            lat = ps.get("infer_latency_p95_ms", 0) or ps.get("latency_p95_ms", 0)
            fps = ps.get("infer_fps", 0) or ps.get("fps_avg", 0)
            if lat > 150 or fps < 10:
                total_violations += 1

    slo_rate = 1.0 - (total_violations / total_windows) if total_windows > 0 else 1.0
    total_stream_windows = total_windows + skip_windows
    skip_ratio = skip_windows / total_stream_windows if total_stream_windows > 0 else 0
    offload_ratio = offload_windows / total_stream_windows if total_stream_windows > 0 else 0

    return {
        "controller": controller_name,
        "scenario": scenario_name,
        "slo_satisfaction_rate": round(slo_rate, 4),
        "slo_violations": total_violations,
        "total_windows": total_windows,
        "n_mode_switches": n_switches,
        "skip_ratio": round(skip_ratio, 4),
        "offload_ratio": round(offload_ratio, 4),
        "steps": len(history),
    }


def train_rl_controller(controller, scenario_name, config_dir, n_episodes):
    """Run training episodes for an RL controller (quiet, no save)."""
    for ep in range(n_episodes):
        t0 = time.time()
        runner = EpisodeRunner(
            scenario_name=scenario_name,
            config_dir=config_dir,
            controller=controller,
        )
        runner.run(output_dir=None, quiet=True)
        elapsed = time.time() - t0
        print(f"    train ep {ep+1}/{n_episodes} done ({elapsed:.0f}s)")


def run_eval(controller_name, scenario_name, config_dir, output_dir,
             controller=None):
    """Run one evaluation episode."""
    print(f"\n{'#'*60}")
    print(f"# {controller_name.upper()} x {scenario_name}")
    print(f"{'#'*60}")

    if controller is None:
        ctrl_cls = CONTROLLERS[controller_name]
        if controller_name == "noop":
            controller = ctrl_cls()
        else:
            controller = ctrl_cls(config_dir=config_dir)

    runner = EpisodeRunner(
        scenario_name=scenario_name,
        config_dir=config_dir,
        controller=controller,
    )
    history = runner.run(output_dir=output_dir)

    summary = extract_summary(controller_name, scenario_name, history)

    print(f"\n  -> SLO: {summary['slo_satisfaction_rate']:.1%}, "
          f"violations: {summary['slo_violations']}/{summary['total_windows']}, "
          f"switches: {summary['n_mode_switches']}, "
          f"skip: {summary['skip_ratio']:.1%}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="/app/config")
    parser.add_argument("--output-dir", default="/data")
    parser.add_argument("--controllers", nargs="+",
                        default=["noop", "heuristic", "myopic", "dqn", "aif"])
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--train-episodes", type=int, default=5,
                        help="Training episodes for RL controllers (0 to skip)")
    args = parser.parse_args()

    all_summaries = []

    for scenario in args.scenarios:
        for ctrl_name in args.controllers:
            # Create controller instance
            ctrl_cls = CONTROLLERS[ctrl_name]
            if ctrl_name == "noop":
                controller = ctrl_cls()
            else:
                controller = ctrl_cls(config_dir=args.config_dir)

            # Training phase for RL controllers
            if ctrl_name in RL_CONTROLLERS and args.train_episodes > 0:
                print(f"\n{'='*60}")
                print(f"  TRAINING {ctrl_name.upper()} on {scenario} "
                      f"({args.train_episodes} episodes)")
                print(f"{'='*60}")
                train_rl_controller(controller, scenario,
                                    args.config_dir, args.train_episodes)
                # Switch to evaluation mode (low epsilon / low entropy)
                controller.set_eval_mode(True)

            # Evaluation run
            s = run_eval(ctrl_name, scenario, args.config_dir,
                         args.output_dir, controller=controller)
            all_summaries.append(s)

    # Comparison table
    print(f"\n{'='*80}")
    print("CONTROLLER COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Controller':<12} {'Scenario':<28} {'SLO%':>6} {'Viol':>6} "
          f"{'Switch':>7} {'Skip%':>6} {'Offl%':>6}")
    print("-" * 76)
    for s in all_summaries:
        print(f"{s['controller']:<12} {s['scenario']:<28} "
              f"{s['slo_satisfaction_rate']:>6.1%} {s['slo_violations']:>6} "
              f"{s['n_mode_switches']:>7} {s['skip_ratio']:>6.1%} "
              f"{s.get('offload_ratio', 0):>6.1%}")

    out_path = f"{args.output_dir}/controller_comparison.json"
    with open(out_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
