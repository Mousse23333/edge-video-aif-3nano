"""Episode runner: ties workload, observation, and controller together."""

import time
import threading
import json
import csv
import yaml

from .inference import BatchInferenceEngine
from .stream_manager import StreamManager
from .observer import ObservationCollector
from .workload import WorkloadEngine


class ControllerInterface:
    """
    Base class for all controllers (heuristic, AIF, RL).
    Override select_action() to implement your policy.
    """

    def select_action(self, observation, stream_manager):
        """
        Given an observation dict, return an action or None (NO-OP).

        Returns:
            None for NO-OP, or (stream_id, new_mode) tuple.
        """
        return None  # NO-OP by default

    def on_episode_start(self, config):
        """Called before the episode begins."""
        pass

    def on_episode_end(self, history):
        """Called after the episode ends."""
        pass


class EpisodeRunner:
    """
    Runs one complete scenario episode:
    1. Starts the workload engine (event scheduler)
    2. Each control interval:
       a. Workload engine ticks (may add/remove streams)
       b. ObservationCollector collects window stats
       c. Controller selects an action
       d. Action is applied via StreamManager
    3. All metrics are logged for analysis.
    """

    def __init__(self, scenario_name, config_dir="/app/config",
                 controller=None):
        self.scenario_name = scenario_name
        self.config_dir = config_dir
        self.controller = controller or ControllerInterface()

        # Load configs
        with open(f"{config_dir}/action_space.yaml") as f:
            self.action_cfg = yaml.safe_load(f)
        with open(f"{config_dir}/slo.yaml") as f:
            self.slo_cfg = yaml.safe_load(f)
        with open(f"{config_dir}/switch_cost.yaml") as f:
            self.switch_cfg = yaml.safe_load(f)

        self.control_interval = self.action_cfg.get("control_interval_s", 1.0)
        self.lite_k = self.action_cfg.get("lite_k", 3)

    def run(self, output_dir="/data", quiet=False):
        # Load scenario
        scenario, video_sources = WorkloadEngine.load_scenario(
            f"{self.config_dir}/workload_scenarios.yaml",
            self.scenario_name)

        if not quiet:
            print(f"\n{'='*60}")
            print(f"Episode: {self.scenario_name}")
            print(f"Duration: {scenario['duration_s']}s, "
                  f"Control interval: {self.control_interval}s")
            print(f"Controller: {self.controller.__class__.__name__}")
            print(f"{'='*60}\n")

        # Initialize components — two engines: FULL (imgsz=640) and LITE (imgsz=320)
        engine = BatchInferenceEngine(n_expected=4, imgsz=640)
        lite_engine = BatchInferenceEngine(n_expected=4, imgsz=320)
        # Offload URLs from action_space config (optional)
        offload_cfg = self.action_cfg.get("modes", {}).get("OFFLOAD", {})
        offload_enabled = offload_cfg.get("enabled", False)
        offload_urls = offload_cfg.get("urls", []) if offload_enabled else []
        sm = StreamManager(engine, lite_engine=lite_engine,
                           offload_urls=offload_urls)
        observer = ObservationCollector(sm)
        wl = WorkloadEngine(scenario, video_sources, sm, quiet=quiet)

        # Background GPU sampler
        stop_sampler = threading.Event()
        def gpu_sampler():
            while not stop_sampler.is_set():
                observer.sample_system()
                time.sleep(0.25)
        sampler_thread = threading.Thread(target=gpu_sampler, daemon=True)
        sampler_thread.start()

        # Notify controller
        self.controller.on_episode_start({
            "action_space": self.action_cfg,
            "slo": self.slo_cfg,
            "switch_cost": self.switch_cfg,
        })

        # Run episode
        history = []
        wl.start()
        step = 0

        while not wl.is_done:
            t_step_start = time.time()

            # 1. Workload tick
            fired_events = wl.tick()

            # 2. Wait for control interval
            time.sleep(self.control_interval)

            # 3. Collect observations
            obs = observer.collect()

            # 4. Controller decides
            action = self.controller.select_action(obs, sm)

            # 5. Apply action
            action_log = None
            if action is not None:
                sid, new_mode = action
                old_mode = sm.get_mode(sid)

                # Check min_dwell
                dwell = sm.get_mode_dwell(sid)
                min_dwell = self.switch_cfg.get(
                    "min_dwell_before_exit", {}).get(old_mode, 1)

                if dwell >= min_dwell and old_mode != new_mode:
                    sm.set_mode(sid, new_mode)
                    action_log = {
                        "stream_id": sid,
                        "from_mode": old_mode,
                        "to_mode": new_mode,
                        "dwell_was": dwell,
                    }
                    if not quiet:
                        print(f"  [CTRL {wl.elapsed:6.1f}s] "
                              f"stream {sid}: {old_mode} -> {new_mode}")

            # 6. Update dwell counters
            sm.increment_dwell()

            # 7. Log step
            step_record = {
                "step": step,
                "t_elapsed": round(wl.elapsed, 2),
                "events": [e["action"] for e in fired_events],
                "observation": obs,
                "action": action_log,
                "n_active": obs["global"]["n_active_streams"],
                "mode_counts": obs["global"]["n_streams_by_mode"],
            }
            # Log AIF belief state if available (for interpretability plots)
            if hasattr(self.controller, 'belief'):
                step_record["belief"] = self.controller.belief.tolist()
            history.append(step_record)

            # 8. Print status
            if not quiet and step % 5 == 0:
                gc = obs["global"]
                n = gc["n_active_streams"]
                mc = gc["n_streams_by_mode"]
                gpu = gc["gpu_util_avg"]
                fps_list = [
                    f"{sid}:{ps.get('infer_fps', ps['fps_avg']):.0f}"
                    for sid, ps in obs["per_stream"].items()
                ]
                print(f"  [STEP {step:3d} | {wl.elapsed:5.1f}s] "
                      f"streams={n} modes={mc} GPU={gpu:.0f}% "
                      f"FPS=[{', '.join(fps_list)}]")

            step += 1

        # Cleanup
        stop_sampler.set()
        sampler_thread.join()
        sm.stop_all()
        engine.stop()
        lite_engine.stop()

        self.controller.on_episode_end(history)

        # Save history (skip if no output_dir or quiet training)
        if output_dir:
            prefix = f"{output_dir}/episode_{self.scenario_name}"
            with open(f"{prefix}_history.json", "w") as f:
                json.dump(history, f, indent=2, default=str)

        # Summary
        total_violations = 0
        total_windows = 0
        for rec in history:
            for sid, ps in rec["observation"]["per_stream"].items():
                if ps["current_mode"] in ("FULL", "LITE"):
                    total_windows += 1
                    lat = ps.get("infer_latency_p95_ms", 0) or ps.get("latency_p95_ms", 0)
                    fps = ps.get("infer_fps", 0) or ps.get("fps_avg", 0)
                    slo_lat = self.slo_cfg["hard_constraints"]["p95_latency_ms"]["threshold"]
                    slo_fps = self.slo_cfg["hard_constraints"]["min_fps"]["threshold"]
                    if lat > slo_lat or fps < slo_fps:
                        total_violations += 1

        slo_rate = 1 - (total_violations / total_windows) if total_windows > 0 else 1
        n_switches = sum(1 for r in history if r["action"] is not None)

        if not quiet:
            print(f"\n{'='*60}")
            print(f"Episode complete: {self.scenario_name}")
            print(f"  Steps: {step}")
            print(f"  SLO satisfaction: {slo_rate:.1%} ({total_violations}/{total_windows} violations)")
            print(f"  Mode switches: {n_switches}")
            if output_dir:
                print(f"  Saved: {prefix}_history.json")
            print(f"{'='*60}\n")

        return history
