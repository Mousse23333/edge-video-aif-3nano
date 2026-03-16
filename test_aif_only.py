#!/usr/bin/env python3
"""
Quick test runner for AIF controller only.
Runs all 4 scenarios and prints results. Use for fast iteration on AIF tuning.

Usage:
  python3 test_aif_only.py
  python3 test_aif_only.py --scenarios scenario_ramp_up scenario_burst
"""

import json
import argparse
from engine.episode import EpisodeRunner, ControllerInterface
from controllers.aif import AIFController
from run_multi_experiment import extract_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="/app/config")
    parser.add_argument("--output-dir", default="/data")
    parser.add_argument("--scenarios", nargs="+", default=[
        "scenario_ramp_up",
        "scenario_burst",
        "scenario_steady_overload",
        "scenario_oscillating",
    ])
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-step episode output")
    args = parser.parse_args()

    summaries = []

    for scenario in args.scenarios:
        print(f"\n{'#'*60}")
        print(f"# AIF x {scenario}")
        print(f"{'#'*60}")

        controller = AIFController(config_dir=args.config_dir)
        runner = EpisodeRunner(
            scenario_name=scenario,
            config_dir=args.config_dir,
            controller=controller,
        )
        history = runner.run(output_dir=args.output_dir, quiet=args.quiet)
        summary = extract_summary("aif", scenario, history, run_id=1)

        print(f"\n  -> SLO: {summary['slo_satisfaction_rate']:.1%}, "
              f"violations: {summary['slo_violations']}/{summary['total_windows']}, "
              f"switches: {summary['n_mode_switches']}, "
              f"skip: {summary['skip_ratio']:.1%}, "
              f"offload: {summary['offload_ratio']:.1%}")

        summaries.append(summary)

    # Summary table
    print(f"\n{'='*70}")
    print("AIF RESULTS")
    print(f"{'='*70}")
    print(f"{'Scenario':<28} {'SLO%':>6} {'Viol':>6} {'Switch':>7} "
          f"{'Skip%':>6} {'Offl%':>6}")
    print("-" * 65)
    for s in summaries:
        print(f"{s['scenario']:<28} "
              f"{s['slo_satisfaction_rate']:>6.1%} {s['slo_violations']:>6} "
              f"{s['n_mode_switches']:>7} {s['skip_ratio']:>6.1%} "
              f"{s.get('offload_ratio', 0):>6.1%}")

    avg_slo = sum(s['slo_satisfaction_rate'] for s in summaries) / len(summaries)
    print(f"\n  Average SLO: {avg_slo:.1%}")

    out_path = f"{args.output_dir}/aif_test_results.json"
    with open(out_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
