"""Observation collector: aggregates per-frame metrics into per-window observations."""

import time
import os
import numpy as np


GPU_LOAD_PATH = "/sys/devices/platform/bus@0/17000000.gpu/load"


def _read_gpu_util():
    try:
        with open(GPU_LOAD_PATH) as f:
            return int(f.read().strip()) / 10.0
    except:
        return -1.0


def _read_mem_used_mb():
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                p = line.split()
                if p[0] in ("MemTotal:", "MemAvailable:"):
                    info[p[0]] = int(p[1]) / 1024
        return info["MemTotal:"] - info["MemAvailable:"]
    except:
        return -1.0


class ObservationCollector:
    """
    Drains metrics from all active StreamWorkers and produces
    a structured observation dict matching observation_space.yaml.

    Called once per control window by the episode runner.
    """

    def __init__(self, stream_manager):
        self.stream_manager = stream_manager
        self._gpu_samples = []
        self._sample_interval = 0.25  # 4Hz

    def sample_system(self):
        """Call this periodically (e.g., 4x per second) from a background thread."""
        self._gpu_samples.append({
            "t": time.time(),
            "gpu_util": _read_gpu_util(),
            "mem_used_mb": _read_mem_used_mb(),
        })

    def collect(self):
        """
        Collect observation for the current control window.

        Returns:
            {
              "per_stream": { stream_id: { fps_avg, latency_p95_ms, ... }, ... },
              "global": { gpu_util_avg, n_active_streams, ... },
              "t_collect": float
            }
        """
        t_now = time.time()

        # --- Per-stream observations ---
        per_stream = {}
        active_workers = self.stream_manager.get_active_workers()

        for sid, worker in active_workers.items():
            frames = worker.drain_metrics()

            if len(frames) < 2:
                per_stream[sid] = {
                    "fps_avg": 0.0,
                    "latency_p95_ms": 0.0,
                    "current_mode": worker.mode,
                    "detection_count_avg": 0.0,
                    "frames_in_current_mode": 0,
                    "frame_count": len(frames),
                }
                continue

            totals = [f["t_total_ms"] for f in frames]
            dets = [f["detections"] for f in frames if f["detections"] > 0]
            duration = frames[-1]["t_wall"] - frames[0]["t_wall"]
            fps = len(frames) / duration if duration > 0 else 0

            # Inference-only metrics (frames that actually ran YOLO)
            infer_frames = [f for f in frames if f["t_infer_ms"] > 0]
            infer_fps = len(infer_frames) / duration if duration > 0 else 0
            infer_latencies = [f["t_total_ms"] for f in infer_frames]

            per_stream[sid] = {
                "fps_avg": round(fps, 2),
                "infer_fps": round(infer_fps, 2),
                "latency_p95_ms": round(np.percentile(totals, 95), 2),
                "infer_latency_p95_ms": round(
                    np.percentile(infer_latencies, 95), 2
                ) if infer_latencies else 0.0,
                "current_mode": worker.mode,
                "detection_count_avg": round(np.mean(dets), 2) if dets else 0.0,
                "frames_in_current_mode": len(frames),
                "frame_count": len(frames),
            }

        # --- Global observations ---
        gpu_samples = self._gpu_samples
        self._gpu_samples = []

        gpu_utils = [s["gpu_util"] for s in gpu_samples if s["gpu_util"] >= 0]
        mem_vals = [s["mem_used_mb"] for s in gpu_samples if s["mem_used_mb"] > 0]

        mode_counts = {"FULL": 0, "LITE": 0, "SKIP": 0, "OFFLOAD": 0}
        for sid, worker in active_workers.items():
            m = worker.mode
            if m in mode_counts:
                mode_counts[m] += 1

        global_obs = {
            "gpu_util_avg": round(np.mean(gpu_utils), 1) if gpu_utils else -1.0,
            "n_active_streams": len(active_workers),
            "mem_used_mb": round(np.mean(mem_vals), 0) if mem_vals else -1.0,
            "n_streams_by_mode": mode_counts,
        }

        return {
            "per_stream": per_stream,
            "global": global_obs,
            "t_collect": t_now,
        }
