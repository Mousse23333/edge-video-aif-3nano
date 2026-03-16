#!/usr/bin/env python3
"""
Two supplementary experiments:
  1. DQN extra seeds (3 more runs) — verify high-variance pattern persists
  2. AIF SKIP likelihood tuning — reduce skip ratio to improve burst/ramp_up SLO

Usage (inside container):
  python3 run_extra_experiments.py
  python3 run_extra_experiments.py --only dqn
  python3 run_extra_experiments.py --only skip
"""

import json
import csv
import time
import os
import argparse
import numpy as np
from datetime import datetime
from collections import defaultdict

from engine.episode import EpisodeRunner, ControllerInterface
from controllers.rl_dqn import DQNController
from controllers.aif import AIFController
from run_multi_experiment import (
    extract_summary, extract_per_step_csv, compute_aggregate, NumpyEncoder,
)
from run_ablation import run_aif_variant, save_ablation_results

SCENARIOS = [
    "scenario_ramp_up",
    "scenario_burst",
    "scenario_steady_overload",
    "scenario_oscillating",
]

# SKIP likelihood variants:
#   baseline: [0.50, 0.50, 0.00]  — uninformative (current default)
#   mild:     [0.30, 0.50, 0.20]  — SKIP carries mild negative signal
#   moderate: [0.20, 0.45, 0.35]  — SKIP carries moderate negative signal
SKIP_VARIANTS = {
    "skip_baseline": np.array([
        [0.50, 0.50, 0.00],
        [0.50, 0.50, 0.00],
        [0.50, 0.50, 0.00],
    ]),
    "skip_mild": np.array([
        [0.30, 0.50, 0.20],
        [0.30, 0.50, 0.20],
        [0.30, 0.50, 0.20],
    ]),
    "skip_moderate": np.array([
        [0.20, 0.45, 0.35],
        [0.20, 0.45, 0.35],
        [0.20, 0.45, 0.35],
    ]),
}


def run_dqn_extra(config_dir, output_dir, n_runs=3, train_episodes=5):
    """Run DQN with extra seeds to confirm variance pattern."""
    os.makedirs(output_dir, exist_ok=True)
    all_summaries = []

    print(f"\n{'='*70}")
    print(f"EXPERIMENT 1: DQN Extra Seeds ({n_runs} runs)")
    print(f"{'='*70}")

    for run_id in range(1, n_runs + 1):
        run_dir = f"{output_dir}/histories/run{run_id}"
        os.makedirs(run_dir, exist_ok=True)

        for scenario in SCENARIOS:
            t0 = time.time()
            controller = DQNController(config_dir=config_dir)

            # Training phase
            print(f"  Training DQN on {scenario} ({train_episodes} ep)...",
                  end=" ", flush=True)
            for _ in range(train_episodes):
                runner = EpisodeRunner(
                    scenario_name=scenario,
                    config_dir=config_dir,
                    controller=controller,
                )
                runner.run(output_dir=None, quiet=True)
            controller.set_eval_mode(True)
            t_train = time.time() - t0
            print(f"done ({t_train:.0f}s)")

            # Evaluation
            t0 = time.time()
            runner = EpisodeRunner(
                scenario_name=scenario,
                config_dir=config_dir,
                controller=controller,
            )
            history = runner.run(output_dir=None, quiet=True)

            # Save history
            hist_path = f"{run_dir}/dqn_{scenario}.json"
            with open(hist_path, "w") as f:
                json.dump(history, f, cls=NumpyEncoder)
            csv_path = f"{run_dir}/dqn_{scenario}_steps.csv"
            extract_per_step_csv(history, csv_path)

            summary = extract_summary("dqn", scenario, history, run_id)
            all_summaries.append(summary)

            elapsed = time.time() - t0
            print(f"  Run {run_id} | dqn x {scenario:<28} "
                  f"SLO={summary['slo_satisfaction_rate']:.1%} "
                  f"switches={summary['n_mode_switches']} ({elapsed:.0f}s)")

    # Save results
    csv_path = f"{output_dir}/summaries.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
        writer.writeheader()
        writer.writerows(all_summaries)

    json_path = f"{output_dir}/summaries.json"
    with open(json_path, "w") as f:
        json.dump({
            "config": {
                "n_runs": n_runs,
                "controllers": ["dqn"],
                "scenarios": SCENARIOS,
                "train_episodes": train_episodes,
                "timestamp": datetime.now().isoformat(),
            },
            "summaries": all_summaries,
        }, f, indent=2, cls=NumpyEncoder)

    aggregate = compute_aggregate(all_summaries)
    agg_path = f"{output_dir}/aggregate.json"
    with open(agg_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"DQN EXTRA SEEDS RESULTS ({n_runs} runs)")
    print(f"{'='*70}")
    print(f"{'Scenario':<28} {'SLO%':>12} {'Switch':>12}")
    print("-" * 55)
    for scen in SCENARIOS:
        m = aggregate["dqn"][scen]
        slo = m["slo_satisfaction_rate"]
        sw = m["n_mode_switches"]
        print(f"{scen:<28} {slo['mean']:>5.1%}±{slo['std']:.1%} "
              f"{sw['mean']:>5.0f}±{sw['std']:.0f}")

    means = [aggregate["dqn"][s]["slo_satisfaction_rate"]["mean"]
             for s in SCENARIOS]
    print(f"\n  Average SLO: {np.mean(means):.1%}")
    print(f"  Saved to: {output_dir}/")

    return all_summaries


def run_skip_tuning(config_dir, output_dir, n_runs=3):
    """Run AIF with different SKIP likelihood variants."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT 2: AIF SKIP Likelihood Tuning ({n_runs} runs)")
    print(f"{'='*70}")

    all_results = []
    for variant_name, skip_mat in SKIP_VARIANTS.items():
        print(f"\n  {variant_name}: {skip_mat[0].tolist()}")
        results = run_aif_variant(
            variant_name, config_dir, SCENARIOS, n_runs,
            aif_kwargs={"skip_likelihood": skip_mat})
        all_results.extend(results)

    save_ablation_results(all_results, output_dir, "ablation_skip_likelihood")

    # Print comparison
    print(f"\n{'='*70}")
    print(f"SKIP LIKELIHOOD RESULTS ({n_runs} runs, mean)")
    print(f"{'='*70}")
    print(f"{'Variant':<20} {'Scenario':<28} {'SLO%':>8} {'Skip%':>8} {'Switch':>8}")
    print("-" * 75)

    grouped = defaultdict(list)
    for r in all_results:
        grouped[(r["variant"], r["scenario"])].append(r)

    for variant_name in SKIP_VARIANTS:
        for scen in SCENARIOS:
            runs = grouped[(variant_name, scen)]
            slo = np.mean([r["slo_satisfaction_rate"] for r in runs])
            skip = np.mean([r["skip_ratio"] for r in runs])
            sw = np.mean([r["n_mode_switches"] for r in runs])
            print(f"{variant_name:<20} {scen:<28} {slo:>7.1%} {skip:>7.1%} {sw:>7.0f}")
        # Per-variant average
        all_runs = [r for r in all_results if r["variant"] == variant_name]
        avg_slo = np.mean([r["slo_satisfaction_rate"] for r in all_runs])
        avg_skip = np.mean([r["skip_ratio"] for r in all_runs])
        print(f"  -> avg SLO={avg_slo:.1%}, avg skip={avg_skip:.1%}")

    print(f"\n  Saved to: {output_dir}/")
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="/app/config")
    parser.add_argument("--dqn-output", default="/data/experiments/dqn_extra")
    parser.add_argument("--skip-output", default="/data/experiments/ablation")
    parser.add_argument("--dqn-runs", type=int, default=3)
    parser.add_argument("--skip-runs", type=int, default=3)
    parser.add_argument("--only", choices=["dqn", "skip"], default=None)
    args = parser.parse_args()

    total_start = time.time()
    print(f"{'='*70}")
    print(f"SUPPLEMENTARY EXPERIMENTS")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    if args.only is None or args.only == "dqn":
        run_dqn_extra(args.config_dir, args.dqn_output, args.dqn_runs)

    if args.only is None or args.only == "skip":
        run_skip_tuning(args.config_dir, args.skip_output, args.skip_runs)

    total = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"ALL SUPPLEMENTARY EXPERIMENTS COMPLETE")
    print(f"  Total time: {total/60:.1f} min")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
