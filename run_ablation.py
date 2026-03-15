#!/usr/bin/env python3
"""
Ablation studies for AIF controller and OFFLOAD mechanism.

Experiments:
  1. AIF OFFLOAD likelihood sensitivity: v0 (BAD) vs v1 (system-level)
  2. AIF precision sensitivity: [2.0, 4.0, 6.0, 8.0]
  3. AIF epistemic weight sensitivity: [0.0, 0.1, 0.3, 0.5]
  4. AIF cooldown sensitivity: [1, 3, 5]
  5. OFFLOAD enabled vs disabled (all controllers)

Usage:
  python3 run_ablation.py                    # run all ablations
  python3 run_ablation.py --only likelihood  # run only likelihood ablation
  python3 run_ablation.py --n-runs 3         # 3 runs per config
"""

import json
import csv
import time
import os
import shutil
import argparse
import numpy as np
from datetime import datetime
from collections import defaultdict

import yaml
from engine.episode import EpisodeRunner, ControllerInterface
from controllers.heuristic import HeuristicController
from controllers.myopic_greedy import MyopicGreedyController
from controllers.rl_dqn import DQNController
from controllers.aif import AIFController, V0_OFFLOAD_LIKELIHOOD


ALL_CONTROLLERS = {
    "noop":      ControllerInterface,
    "heuristic": HeuristicController,
    "myopic":    MyopicGreedyController,
    "dqn":       DQNController,
    "aif":       AIFController,
}
RL_CONTROLLERS = {"dqn"}

SCENARIOS = [
    "scenario_ramp_up",
    "scenario_burst",
    "scenario_steady_overload",
    "scenario_oscillating",
]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def extract_summary(controller_name, scenario_name, history, run_id,
                    variant=""):
    total_windows = 0
    total_violations = 0
    skip_windows = 0
    offload_windows = 0
    n_switches = sum(1 for r in history if r.get("action") is not None)
    all_fps = []
    all_lat = []

    for rec in history:
        for sid, ps in rec["observation"]["per_stream"].items():
            mode = ps.get("current_mode", "FULL")
            if mode == "SKIP":
                skip_windows += 1
                continue
            if mode == "OFFLOAD":
                offload_windows += 1
            if mode not in ("FULL", "LITE", "OFFLOAD"):
                continue
            total_windows += 1
            lat = ps.get("infer_latency_p95_ms", 0) or ps.get("latency_p95_ms", 0)
            fps = ps.get("infer_fps", 0) or ps.get("fps_avg", 0)
            if lat > 150 or fps < 10:
                total_violations += 1
            all_fps.append(fps)
            all_lat.append(lat)

    slo_rate = 1.0 - (total_violations / total_windows) if total_windows > 0 else 1.0
    total_sw = total_windows + skip_windows
    skip_ratio = skip_windows / total_sw if total_sw > 0 else 0
    offload_ratio = offload_windows / total_sw if total_sw > 0 else 0

    return {
        "run_id": run_id,
        "controller": controller_name,
        "variant": variant,
        "scenario": scenario_name,
        "slo_satisfaction_rate": round(slo_rate, 4),
        "slo_violations": total_violations,
        "total_windows": total_windows,
        "n_mode_switches": n_switches,
        "skip_ratio": round(skip_ratio, 4),
        "offload_ratio": round(offload_ratio, 4),
        "avg_fps": round(float(np.mean(all_fps)), 2) if all_fps else 0,
        "avg_latency_ms": round(float(np.mean(all_lat)), 2) if all_lat else 0,
        "steps": len(history),
    }


def run_aif_variant(variant_name, config_dir, scenarios, n_runs, aif_kwargs):
    """Run AIF with specific kwargs, return summaries."""
    results = []
    for run_id in range(1, n_runs + 1):
        for scenario in scenarios:
            t0 = time.time()
            controller = AIFController(config_dir=config_dir, **aif_kwargs)
            runner = EpisodeRunner(
                scenario_name=scenario,
                config_dir=config_dir,
                controller=controller,
            )
            history = runner.run(output_dir=None, quiet=True)
            summary = extract_summary("aif", scenario, history, run_id,
                                      variant=variant_name)
            results.append(summary)
            elapsed = time.time() - t0
            print(f"    {variant_name} | run {run_id} | {scenario:<28} "
                  f"SLO={summary['slo_satisfaction_rate']:.1%} ({elapsed:.0f}s)")
    return results


def run_all_controllers_variant(variant_name, config_dir, scenarios, n_runs,
                                train_episodes=5,
                                disable_offload=False):
    """Run all controllers (optionally with OFFLOAD disabled)."""
    # If disabling OFFLOAD, create temp config
    if disable_offload:
        tmp_config = "/tmp/config_no_offload"
        if os.path.exists(tmp_config):
            shutil.rmtree(tmp_config)
        shutil.copytree(config_dir, tmp_config)
        # Modify action_space.yaml
        as_path = f"{tmp_config}/action_space.yaml"
        with open(as_path) as f:
            ac = yaml.safe_load(f)
        ac["modes"]["OFFLOAD"]["enabled"] = False
        with open(as_path, "w") as f:
            yaml.dump(ac, f, default_flow_style=False)
        config_dir = tmp_config

    results = []
    for run_id in range(1, n_runs + 1):
        for scenario in scenarios:
            for ctrl_name, ctrl_cls in ALL_CONTROLLERS.items():
                t0 = time.time()
                if ctrl_name == "noop":
                    controller = ctrl_cls()
                else:
                    controller = ctrl_cls(config_dir=config_dir)

                # RL training
                if ctrl_name in RL_CONTROLLERS and train_episodes > 0:
                    for _ in range(train_episodes):
                        r = EpisodeRunner(scenario_name=scenario,
                                          config_dir=config_dir,
                                          controller=controller)
                        r.run(output_dir=None, quiet=True)
                    controller.set_eval_mode(True)

                runner = EpisodeRunner(
                    scenario_name=scenario,
                    config_dir=config_dir,
                    controller=controller,
                )
                history = runner.run(output_dir=None, quiet=True)
                summary = extract_summary(ctrl_name, scenario, history,
                                          run_id, variant=variant_name)
                results.append(summary)
                elapsed = time.time() - t0
                print(f"    {variant_name} | run {run_id} | "
                      f"{ctrl_name:<12} x {scenario:<28} "
                      f"SLO={summary['slo_satisfaction_rate']:.1%} "
                      f"({elapsed:.0f}s)")
    return results


def save_ablation_results(all_results, output_dir, name):
    """Save ablation results to CSV and JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # CSV
    csv_path = f"{output_dir}/{name}.csv"
    if all_results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

    # JSON
    json_path = f"{output_dir}/{name}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)

    # Aggregate: mean±std per variant×controller×scenario
    grouped = defaultdict(list)
    for s in all_results:
        key = (s["variant"], s["controller"], s["scenario"])
        grouped[key].append(s)

    aggregate = {}
    for (variant, ctrl, scenario), runs in grouped.items():
        if variant not in aggregate:
            aggregate[variant] = {}
        if ctrl not in aggregate[variant]:
            aggregate[variant][ctrl] = {}
        metrics = {}
        for metric in ["slo_satisfaction_rate", "slo_violations",
                       "n_mode_switches", "skip_ratio", "offload_ratio"]:
            vals = [r[metric] for r in runs]
            metrics[metric] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
            }
        aggregate[variant][ctrl][scenario] = metrics

    agg_path = f"{output_dir}/{name}_aggregate.json"
    with open(agg_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"  Saved: {csv_path}, {json_path}, {agg_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="/app/config")
    parser.add_argument("--output-dir", default="/data/experiments/ablation")
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Runs per ablation config")
    parser.add_argument("--only", type=str, default=None,
                        choices=["likelihood", "precision", "epistemic",
                                 "cooldown", "offload_onoff"],
                        help="Run only one ablation type")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    total_start = time.time()

    print(f"{'='*70}")
    print(f"ABLATION STUDIES")
    print(f"  N runs per config: {args.n_runs}")
    print(f"  Output: {args.output_dir}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    # ── 1. AIF OFFLOAD Likelihood Sensitivity ─────────────────────────────

    if args.only is None or args.only == "likelihood":
        print(f"\n{'─'*60}")
        print("ABLATION 1: AIF OFFLOAD Likelihood (v0 vs v1)")
        print(f"{'─'*60}")

        all_results = []

        # v0: per-stream BAD
        print("\n  v0 (OFFLOAD = per-stream BAD):")
        results_v0 = run_aif_variant(
            "v0_offload_bad", args.config_dir, SCENARIOS, args.n_runs,
            aif_kwargs={"offload_likelihood": V0_OFFLOAD_LIKELIHOOD})
        all_results.extend(results_v0)

        # v1: system-level (default)
        print("\n  v1 (OFFLOAD = system-level):")
        results_v1 = run_aif_variant(
            "v1_offload_system", args.config_dir, SCENARIOS, args.n_runs,
            aif_kwargs={})
        all_results.extend(results_v1)

        save_ablation_results(all_results, args.output_dir,
                              "ablation_likelihood")

    # ── 2. AIF Precision Sensitivity ──────────────────────────────────────

    if args.only is None or args.only == "precision":
        print(f"\n{'─'*60}")
        print("ABLATION 2: AIF Precision β ∈ {2.0, 4.0, 6.0, 8.0}")
        print(f"{'─'*60}")

        all_results = []
        for beta in [2.0, 4.0, 6.0, 8.0]:
            print(f"\n  precision={beta}:")
            results = run_aif_variant(
                f"precision_{beta}", args.config_dir, SCENARIOS, args.n_runs,
                aif_kwargs={"precision": beta})
            all_results.extend(results)

        save_ablation_results(all_results, args.output_dir,
                              "ablation_precision")

    # ── 3. AIF Epistemic Weight Sensitivity ───────────────────────────────

    if args.only is None or args.only == "epistemic":
        print(f"\n{'─'*60}")
        print("ABLATION 3: AIF Epistemic Weight ∈ {0.0, 0.1, 0.3, 0.5}")
        print(f"{'─'*60}")

        all_results = []
        for w in [0.0, 0.1, 0.3, 0.5]:
            print(f"\n  epistemic_weight={w}:")
            results = run_aif_variant(
                f"epistemic_{w}", args.config_dir, SCENARIOS, args.n_runs,
                aif_kwargs={"epistemic_weight": w})
            all_results.extend(results)

        save_ablation_results(all_results, args.output_dir,
                              "ablation_epistemic")

    # ── 4. AIF Cooldown Sensitivity ───────────────────────────────────────

    if args.only is None or args.only == "cooldown":
        print(f"\n{'─'*60}")
        print("ABLATION 4: AIF Cooldown ∈ {1, 3, 5}")
        print(f"{'─'*60}")

        all_results = []
        for cd in [1, 3, 5]:
            print(f"\n  cooldown={cd}:")
            results = run_aif_variant(
                f"cooldown_{cd}", args.config_dir, SCENARIOS, args.n_runs,
                aif_kwargs={"cooldown": cd})
            all_results.extend(results)

        save_ablation_results(all_results, args.output_dir,
                              "ablation_cooldown")

    # ── 5. OFFLOAD Enabled vs Disabled (All Controllers) ──────────────────

    if args.only is None or args.only == "offload_onoff":
        print(f"\n{'─'*60}")
        print("ABLATION 5: OFFLOAD Enabled vs Disabled (all controllers)")
        print(f"{'─'*60}")

        all_results = []

        print("\n  OFFLOAD DISABLED:")
        results_off = run_all_controllers_variant(
            "offload_disabled", args.config_dir, SCENARIOS, args.n_runs,
            disable_offload=True)
        all_results.extend(results_off)

        print("\n  OFFLOAD ENABLED (baseline):")
        results_on = run_all_controllers_variant(
            "offload_enabled", args.config_dir, SCENARIOS, args.n_runs,
            disable_offload=False)
        all_results.extend(results_on)

        save_ablation_results(all_results, args.output_dir,
                              "ablation_offload_onoff")

    # ── Summary ───────────────────────────────────────────────────────────

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"ALL ABLATIONS COMPLETE")
    print(f"  Total time: {total_elapsed/3600:.1f} hours")
    print(f"  Output: {args.output_dir}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
