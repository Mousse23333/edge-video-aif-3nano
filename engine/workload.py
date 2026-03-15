"""Workload engine: reads scenario YAML and executes events at scheduled times."""

import time
import yaml


class WorkloadEngine:
    """
    Reads a scenario from workload_scenarios.yaml and executes events
    against a StreamManager at the scheduled wall-clock offsets.

    Completely decoupled from any controller logic.
    """

    def __init__(self, scenario_config, video_sources, stream_manager, quiet=False):
        """
        Args:
            scenario_config: dict with 'duration_s' and 'events' list
            video_sources: dict mapping source name -> { path, ... }
            stream_manager: StreamManager instance
            quiet: if True, suppress print output
        """
        self.duration_s = scenario_config["duration_s"]
        self.events = sorted(scenario_config["events"], key=lambda e: e["t"])
        self.video_sources = video_sources
        self.sm = stream_manager
        self.quiet = quiet

        self._pending = list(self.events)
        self._executed = []
        self._t_start = None

    def start(self):
        self._t_start = time.time()
        self._pending = list(self.events)
        self._executed = []

    @property
    def elapsed(self):
        if self._t_start is None:
            return 0
        return time.time() - self._t_start

    @property
    def is_done(self):
        return self.elapsed >= self.duration_s

    def tick(self):
        """
        Check and execute any events whose time has arrived.
        Call this once per control interval (or more often).

        Returns list of events that fired this tick.
        """
        if self._t_start is None:
            return []

        t = self.elapsed
        fired = []

        while self._pending and self._pending[0]["t"] <= t:
            event = self._pending.pop(0)
            self._execute(event)
            self._executed.append(event)
            fired.append(event)

        return fired

    def _execute(self, event):
        action = event["action"]
        sid = event.get("stream_id")

        if action == "add_stream":
            source_name = event.get("source", "traffic_720p")
            source = self.video_sources.get(source_name, {})
            video_path = source.get("path", "/data/traffic_720p.mp4")
            initial_mode = event.get("initial_mode", "FULL")
            self.sm.add_stream(sid, video_path, initial_mode)
            if not self.quiet:
                print(f"  [WL {self.elapsed:6.1f}s] add_stream {sid} "
                      f"({source_name}, {initial_mode})")

        elif action == "remove_stream":
            self.sm.remove_stream(sid)
            if not self.quiet:
                print(f"  [WL {self.elapsed:6.1f}s] remove_stream {sid}")

        elif action == "switch_video":
            source_name = event.get("source", "traffic_720p")
            source = self.video_sources.get(source_name, {})
            video_path = source.get("path", "/data/traffic_720p.mp4")
            self.sm.switch_video(sid, video_path)
            if not self.quiet:
                print(f"  [WL {self.elapsed:6.1f}s] switch_video {sid} -> {source_name}")

        else:
            if not self.quiet:
                print(f"  [WL {self.elapsed:6.1f}s] unknown event: {event}")

    @staticmethod
    def load_scenario(yaml_path, scenario_name):
        """Load a named scenario from the workload YAML file."""
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        video_sources = config.get("video_sources", {})
        scenario = config.get(scenario_name)
        if scenario is None:
            available = [k for k in config.keys()
                         if k.startswith("scenario_")]
            raise ValueError(
                f"Scenario '{scenario_name}' not found. "
                f"Available: {available}")
        return scenario, video_sources
