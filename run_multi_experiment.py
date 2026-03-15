#!/usr/bin/env python3
"""
Multi-run experiment for controller comparison.
Runs N independent repetitions and saves structured data for plotting.

Output structure:
  <output_dir>/
    summaries.csv         # flat CSV: run,controller,scenario,metrics...
    summaries.json        # same in JSON
    aggregate.json        # mean/std per controller×scenario
    histories/run1/       # per-step history JSONs
    histories/run1/       # per-step metric CSVs
    ...

Usage:
  python3 run_multi_experiment.py --n-runs 5
  python3 run_multi_experiment.py --n-runs 3 --controllers heuristic aif
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
from controllers.heuristic import HeuristicController
from controllers.myopic_greedy import MyopicGreedyController
from controllers.rl_dqn import DQNController
from controllers.aif import AIFController


CONTROLLERS = {
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


def extract_summary(controller_name, scenario_name, history, run_id):
    """Extract comprehensive summary metrics from episode history."""
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
                total_windows += 1
                lat = ps.get("infer_latency_p95_ms", 0) or ps.get("latency_p95_ms", 0)
                fps = ps.get("infer_fps", 0) or ps.get("fps_avg", 0)
                if lat > 150 or fps < 10:
                    total_violations += 1
                all_fps.append(fps)
                all_lat.append(lat)
                continue
            if mode not in ("FULL", "LITE"):
                continue
            total_windows += 1
            lat = ps.get("infer_latency_p95_ms", 0) or ps.get("latency_p95_ms", 0)
            fps = ps.get("infer_fps", 0) or ps.get("fps_avg", 0)
            if lat > 150 or fps < 10:
                total_violations += 1
            all_fps.append(fps)
            all_lat.append(lat)

    slo_rate = 1.0 - (total_violations / total_windows) if total_windows > 0 else 1.0
    total_stream_windows = total_windows + skip_windows
    skip_ratio = skip_windows / total_stream_windows if total_stream_windows > 0 else 0
    offload_ratio = offload_windows / total_stream_windows if total_stream_windows > 0 else 0

    return {
        "run_id": run_id,
        "controller": controller_name,
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


def extract_per_step_csv(history, path):
    """Save per-step metrics as CSV for time-series plotting."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step", "n_active", "gpu_util",
            "n_full", "n_lite", "n_skip", "n_offload",
            "avg_fps", "avg_lat_ms", "action",
        ])
        for i, rec in enumerate(history):
            obs = rec["observation"]
            ps_dict = obs["per_stream"]
            gl = obs["global"]

            n_full = sum(1 for ps in ps_dict.values()
                         if ps.get("current_mode") == "FULL")
            n_lite = sum(1 for ps in ps_dict.values()
                         if ps.get("current_mode") == "LITE")
            n_skip = sum(1 for ps in ps_dict.values()
                         if ps.get("current_mode") == "SKIP")
            n_offload = sum(1 for ps in ps_dict.values()
                            if ps.get("current_mode") == "OFFLOAD")

            fps_vals = [ps.get("infer_fps", ps.get("fps_avg", 0))
                        for ps in ps_dict.values()
                        if ps.get("current_mode") not in ("SKIP",)]
            lat_vals = [ps.get("infer_latency_p95_ms",
                               ps.get("latency_p95_ms", 0))
                        for ps in ps_dict.values()
                        if ps.get("current_mode") not in ("SKIP",)]

            action = rec.get("action")
            if action and isinstance(action, dict):
                action_str = f"{action['stream_id']}_{action['to_mode']}"
            elif action:
                action_str = str(action)
            else:
                action_str = "noop"

            writer.writerow([
                i,
                gl.get("n_active_streams", 0),
                round(gl.get("gpu_util_avg", 0), 1),
                n_full, n_lite, n_skip, n_offload,
                round(float(np.mean(fps_vals)), 2) if fps_vals else 0,
                round(float(np.mean(lat_vals)), 2) if lat_vals else 0,
                action_str,
            ])


def compute_aggregate(all_summaries):
    """Compute mean±std per controller×scenario across runs."""
    grouped = defaultdict(list)
    for s in all_summaries:
        key = (s["controller"], s["scenario"])
        grouped[key].append(s)

    aggregate = {}
    for (ctrl, scenario), runs in grouped.items():
        if ctrl not in aggregate:
            aggregate[ctrl] = {}
        metrics = {}
        for metric in ["slo_satisfaction_rate", "slo_violations",
                       "n_mode_switches", "skip_ratio", "offload_ratio",
                       "avg_fps", "avg_latency_ms"]:
            vals = [r[metric] for r in runs]
            metrics[metric] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
                "values": [round(float(v), 4) for v in vals],
            }
        aggregate[ctrl][scenario] = metrics
    return aggregate


def train_rl_controller(controller, scenario_name, config_dir, n_episodes):
    """Run training episodes for an RL controller."""
    for ep in range(n_episodes):
        runner = EpisodeRunner(
            scenario_name=scenario_name,
            config_dir=config_dir,
            controller=controller,
        )
        runner.run(output_dir=None, quiet=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--config-dir", default="/app/config")
    parser.add_argument("--output-dir", default="/data/experiments/main")
    parser.add_argument("--controllers", nargs="+",
                        default=["noop", "heuristic", "myopic", "dqn", "aif"])
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--train-episodes", type=int, default=5)
    parser.add_argument("--save-histories", action="store_true", default=True,
                        help="Save full per-step history JSONs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    experiment_start = time.time()
    all_summaries = []

    print(f"{'='*70}")
    print(f"MULTI-RUN EXPERIMENT")
    print(f"  Runs: {args.n_runs}")
    print(f"  Controllers: {args.controllers}")
    print(f"  Scenarios: {args.scenarios}")
    print(f"  Train episodes (RL): {args.train_episodes}")
    print(f"  Output: {args.output_dir}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    for run_id in range(1, args.n_runs + 1):
        run_start = time.time()
        run_dir = f"{args.output_dir}/histories/run{run_id}"
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"  RUN {run_id}/{args.n_runs}")
        print(f"{'='*70}")

        for scenario in args.scenarios:
            for ctrl_name in args.controllers:
                t0 = time.time()

                # Create fresh controller instance (independent per run)
                ctrl_cls = CONTROLLERS[ctrl_name]
                if ctrl_name == "noop":
                    controller = ctrl_cls()
                else:
                    controller = ctrl_cls(config_dir=args.config_dir)

                # Training phase for RL controllers
                if ctrl_name in RL_CONTROLLERS and args.train_episodes > 0:
                    print(f"  Training {ctrl_name} on {scenario} "
                          f"({args.train_episodes} ep)...", end=" ", flush=True)
                    train_rl_controller(controller, scenario,
                                        args.config_dir, args.train_episodes)
                    controller.set_eval_mode(True)
                    print(f"done ({time.time()-t0:.0f}s)")
                    t0 = time.time()

                # Evaluation
                runner = EpisodeRunner(
                    scenario_name=scenario,
                    config_dir=args.config_dir,
                    controller=controller,
                )
                history = runner.run(output_dir=None, quiet=True)

                # Save per-step data
                if args.save_histories:
                    hist_path = f"{run_dir}/{ctrl_name}_{scenario}.json"
                    with open(hist_path, "w") as f:
                        json.dump(history, f, cls=NumpyEncoder)

                    csv_path = f"{run_dir}/{ctrl_name}_{scenario}_steps.csv"
                    extract_per_step_csv(history, csv_path)

                # Extract summary
                summary = extract_summary(ctrl_name, scenario, history, run_id)
                all_summaries.append(summary)

                elapsed = time.time() - t0
                print(f"  Run {run_id} | {ctrl_name:<12} x {scenario:<28} "
                      f"SLO={summary['slo_satisfaction_rate']:.1%}  "
                      f"({elapsed:.0f}s)")

        run_elapsed = time.time() - run_start
        print(f"\n  Run {run_id} complete ({run_elapsed/60:.1f} min)")

    # ── Save summaries ────────────────────────────────────────────────────

    # CSV
    csv_path = f"{args.output_dir}/summaries.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
        writer.writeheader()
        writer.writerows(all_summaries)

    # JSON
    json_path = f"{args.output_dir}/summaries.json"
    with open(json_path, "w") as f:
        json.dump({
            "config": {
                "n_runs": args.n_runs,
                "controllers": args.controllers,
                "scenarios": args.scenarios,
                "train_episodes": args.train_episodes,
                "timestamp": datetime.now().isoformat(),
            },
            "summaries": all_summaries,
        }, f, indent=2, cls=NumpyEncoder)

    # Aggregate
    aggregate = compute_aggregate(all_summaries)
    agg_path = f"{args.output_dir}/aggregate.json"
    with open(agg_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    # ── Print final table ─────────────────────────────────────────────────

    total_elapsed = time.time() - experiment_start
    print(f"\n{'='*80}")
    print(f"AGGREGATE RESULTS ({args.n_runs} runs, mean ± std)")
    print(f"{'='*80}")
    print(f"{'Controller':<12} {'Scenario':<28} {'SLO%':>12} "
          f"{'Viol':>10} {'Switch':>10} {'Skip%':>10} {'Offl%':>10}")
    print("-" * 92)
    for ctrl in args.controllers:
        for scen in args.scenarios:
            m = aggregate[ctrl][scen]
            slo = m["slo_satisfaction_rate"]
            viol = m["slo_violations"]
            sw = m["n_mode_switches"]
            sk = m["skip_ratio"]
            of = m["offload_ratio"]
            print(f"{ctrl:<12} {scen:<28} "
                  f"{slo['mean']:>5.1%}±{slo['std']:.1%} "
                  f"{viol['mean']:>5.0f}±{viol['std']:.0f} "
                  f"{sw['mean']:>5.0f}±{sw['std']:.0f} "
                  f"{sk['mean']:>5.1%}±{sk['std']:.1%} "
                  f"{of['mean']:>5.1%}±{of['std']:.1%}")

    # Per-controller average
    print(f"\n{'Controller':<12} {'Avg SLO%':>12}")
    print("-" * 26)
    for ctrl in args.controllers:
        means = [aggregate[ctrl][s]["slo_satisfaction_rate"]["mean"]
                 for s in args.scenarios]
        print(f"{ctrl:<12} {np.mean(means):>10.1%}")

    print(f"\nTotal time: {total_elapsed/60:.1f} min")
    print(f"Saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
