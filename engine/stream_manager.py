"""Stream manager: add/remove/switch streams dynamically."""

import threading
from .stream import StreamWorker


class StreamManager:
    """
    Manages the lifecycle of StreamWorkers.
    Provides the event interface for the workload engine:
      add_stream, remove_stream, switch_video.
    Provides the control interface for controllers:
      set_mode(stream_id, mode).
    """

    def __init__(self, engine, lite_engine=None, target_size=(1280, 720),
                 offload_urls=None):
        self.engine = engine
        self.lite_engine = lite_engine or engine
        self.target_size = target_size
        # offload_urls: list of URLs for round-robin offload
        # e.g. ["http://192.168.1.x:8765/infer", "http://192.168.1.y:8765/infer"]
        self.offload_urls = offload_urls or []
        self._offload_idx = 0  # round-robin index
        self._workers = {}   # stream_id -> StreamWorker
        self._lock = threading.Lock()
        # Track mode dwell time (in control windows)
        self._mode_dwell = {}  # stream_id -> int

    # ── Workload events ───────────────────────────────────────────────────

    def add_stream(self, stream_id, video_path, initial_mode="FULL"):
        with self._lock:
            if stream_id in self._workers:
                return  # already exists
            # Assign offload URL (round-robin across available Nano nodes)
            offload_url = None
            if self.offload_urls:
                offload_url = self.offload_urls[self._offload_idx % len(self.offload_urls)]
                self._offload_idx += 1

            worker = StreamWorker(
                stream_id=stream_id,
                video_path=video_path,
                target_size=self.target_size,
                engine=self.engine,
                lite_engine=self.lite_engine,
                initial_mode=initial_mode,
                offload_url=offload_url,
            )
            self._workers[stream_id] = worker
            self._mode_dwell[stream_id] = 0
            worker.start()
            self._update_engine_expected()

    def remove_stream(self, stream_id):
        with self._lock:
            worker = self._workers.pop(stream_id, None)
            self._mode_dwell.pop(stream_id, None)
            if worker:
                worker.stop()
            self._update_engine_expected()

    def switch_video(self, stream_id, new_video_path):
        """Stop the stream and restart with a different video source."""
        with self._lock:
            worker = self._workers.get(stream_id)
            if not worker:
                return
            old_mode = worker.mode
            worker.stop()
            new_worker = StreamWorker(
                stream_id=stream_id,
                video_path=new_video_path,
                target_size=self.target_size,
                engine=self.engine,
                lite_engine=self.lite_engine,
                initial_mode=old_mode,
            )
            self._workers[stream_id] = new_worker
            new_worker.start()

    # ── Controller interface ──────────────────────────────────────────────

    def set_mode(self, stream_id, mode):
        with self._lock:
            worker = self._workers.get(stream_id)
            if worker:
                worker.set_mode(mode)
                self._mode_dwell[stream_id] = 0
                self._update_engine_expected()

    def get_mode(self, stream_id):
        with self._lock:
            worker = self._workers.get(stream_id)
            return worker.mode if worker else None

    def get_active_workers(self):
        with self._lock:
            return dict(self._workers)

    def get_active_stream_ids(self):
        with self._lock:
            return list(self._workers.keys())

    def get_mode_dwell(self, stream_id):
        with self._lock:
            return self._mode_dwell.get(stream_id, 0)

    def increment_dwell(self):
        """Called once per control window to track dwell time."""
        with self._lock:
            for sid in self._mode_dwell:
                self._mode_dwell[sid] += 1

    # ── Internal ──────────────────────────────────────────────────────────

    def _update_engine_expected(self):
        """Update both engines' expected counts based on active stream modes."""
        n_full = sum(1 for w in self._workers.values() if w.mode == "FULL")
        n_lite = sum(1 for w in self._workers.values() if w.mode == "LITE")
        self.engine.update_expected(max(1, n_full))
        self.lite_engine.update_expected(max(1, n_lite))

    def stop_all(self):
        with self._lock:
            for worker in self._workers.values():
                worker.stop()
            self._workers.clear()
            self._mode_dwell.clear()
