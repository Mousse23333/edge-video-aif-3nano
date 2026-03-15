#!/usr/bin/env python3
"""
Multi-stream YOLO inference benchmark on Jetson.

Runs N concurrent streams sharing a single YOLOv8n model.
Each stream is a thread that decodes and resizes independently (CPU),
then submits to a shared inference queue (GPU, serialized).

Usage:
  python3 benchmark_multi.py --video /data/traffic_720p.mp4 --streams 1
  python3 benchmark_multi.py --video /data/traffic_720p.mp4 --streams 2
  python3 benchmark_multi.py --video /data/traffic_720p.mp4 --streams 3
  python3 benchmark_multi.py --video /data/traffic.webm    --streams 1 --scale 720p
"""

import argparse
import time
import csv
import json
import threading
import queue
import os
import numpy as np
import cv2
import torch
from ultralytics import YOLO

GPU_LOAD_PATH = "/sys/devices/platform/bus@0/17000000.gpu/load"
THERMAL_BASE  = "/sys/class/thermal"


def get_gpu_util():
    try:
        with open(GPU_LOAD_PATH) as f:
            return int(f.read().strip()) / 10.0
    except:
        return -1.0


def get_mem_used_mb():
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


def get_thermals():
    temps = {}
    try:
        for tz in sorted(os.listdir(THERMAL_BASE)):
            path = os.path.join(THERMAL_BASE, tz)
            if not os.path.isdir(path):
                continue
            try:
                name = open(os.path.join(path, "type")).read().strip()
                temp = int(open(os.path.join(path, "temp")).read().strip()) / 1000.0
                temps[name] = temp
            except:
                continue
    except:
        pass
    return temps


def system_monitor(samples, stop_event, interval=0.25):
    while not stop_event.is_set():
        samples.append({
            "t": time.time(),
            "gpu_util": get_gpu_util(),
            "mem_used_mb": get_mem_used_mb(),
        })
        time.sleep(interval)


# ── Shared inference engine ───────────────────────────────────────────────────

class InferenceEngine:
    """Single model shared across all streams. Inference is serialized via lock."""

    def __init__(self, model_path="yolov8n.pt", device="cuda"):
        print(f"Loading model on {device}...")
        self.model = YOLO(model_path)
        self.model.to(device)
        self.device = device
        self.lock = threading.Lock()
        # Warmup
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(10):
            self._infer(dummy)
        print(f"Model ready. Device: {next(self.model.model.parameters()).device}")

    def _infer(self, frame):
        results = self.model(frame, verbose=False)
        return len(results[0].boxes) if results else 0

    def infer(self, frame):
        """Thread-safe inference call."""
        with self.lock:
            t0 = time.time()
            n = self._infer(frame)
            t1 = time.time()
        return n, (t1 - t0) * 1000


# ── Per-stream worker ─────────────────────────────────────────────────────────

def stream_worker(stream_id, video_path, target_size, engine,
                  max_frames, metrics_list, ready_event, start_event):
    """
    One thread per stream.
    Decodes + resizes on CPU, then calls engine.infer() (serialized GPU).
    """
    target_w, target_h = target_size

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Stream {stream_id}] ERROR: cannot open {video_path}")
        return

    metrics = []
    ready_event.set()        # signal: ready to start
    start_event.wait()       # wait for all streams to be ready

    frame_idx = 0
    while frame_idx < max_frames:
        t0 = time.time()

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        t_read = time.time()

        if target_size:
            frame = cv2.resize(frame, (target_w, target_h))
        t_resize = time.time()

        n_det, infer_ms = engine.infer(frame)
        t_done = time.time()

        metrics.append({
            "stream": stream_id,
            "frame": frame_idx,
            "t_read_ms":   (t_read   - t0)     * 1000,
            "t_resize_ms": (t_resize - t_read)  * 1000,
            "t_infer_ms":  infer_ms,
            "t_total_ms":  (t_done   - t0)      * 1000,
            "t_wait_ms":   (t_done   - t_resize) * 1000 - infer_ms,
            "detections":  n_det,
            "wall_ts":     t_done,
        })
        frame_idx += 1

    cap.release()
    metrics_list.extend(metrics)


# ── Main benchmark ────────────────────────────────────────────────────────────

def run_benchmark(video_path, n_streams, target_size, max_frames_per_stream):
    tw, th = target_size
    print(f"\n{'='*60}")
    print(f"Streams: {n_streams}  |  Resolution: {tw}x{th}  |  Frames/stream: {max_frames_per_stream}")
    print(f"{'='*60}")

    engine = InferenceEngine()

    # System monitor
    sys_samples = []
    stop_mon = threading.Event()
    mon_thread = threading.Thread(target=system_monitor, args=(sys_samples, stop_mon))
    mon_thread.start()

    # Launch stream threads
    all_metrics = []
    ready_events = [threading.Event() for _ in range(n_streams)]
    start_event  = threading.Event()

    threads = []
    for i in range(n_streams):
        t = threading.Thread(
            target=stream_worker,
            args=(i, video_path, target_size, engine,
                  max_frames_per_stream, all_metrics,
                  ready_events[i], start_event),
            daemon=True,
        )
        threads.append(t)
        t.start()

    # Wait for all streams to be ready, then fire
    for e in ready_events:
        e.wait()
    t_start = time.time()
    start_event.set()

    for t in threads:
        t.join()
    t_end = time.time()

    stop_mon.set()
    mon_thread.join()

    # ── Analysis ──────────────────────────────────────────────────────────────
    total_frames = len(all_metrics)
    wall_time    = t_end - t_start
    total_fps    = total_frames / wall_time

    # Per-stream FPS
    stream_fps = []
    for sid in range(n_streams):
        sm = [m for m in all_metrics if m["stream"] == sid]
        if len(sm) >= 2:
            dur = sm[-1]["wall_ts"] - sm[0]["wall_ts"]
            stream_fps.append(len(sm) / dur if dur > 0 else 0)
        else:
            stream_fps.append(0)

    infer_times  = [m["t_infer_ms"]  for m in all_metrics]
    total_times  = [m["t_total_ms"]  for m in all_metrics]
    wait_times   = [m["t_wait_ms"]   for m in all_metrics]
    read_times   = [m["t_read_ms"]   for m in all_metrics]
    resize_times = [m["t_resize_ms"] for m in all_metrics]
    gpu_utils    = [s["gpu_util"]    for s in sys_samples if s["gpu_util"] >= 0]
    mem_used     = [s["mem_used_mb"] for s in sys_samples]

    summary = {
        "n_streams":          n_streams,
        "resolution":         f"{tw}x{th}",
        "total_frames":       total_frames,
        "wall_time_s":        round(wall_time, 2),
        "total_fps":          round(total_fps, 2),
        "per_stream_fps_avg": round(np.mean(stream_fps), 2),
        "per_stream_fps_min": round(np.min(stream_fps), 2),
        "per_stream_fps":     [round(f, 2) for f in stream_fps],
        "read_mean_ms":       round(np.mean(read_times), 2),
        "resize_mean_ms":     round(np.mean(resize_times), 2),
        "infer_mean_ms":      round(np.mean(infer_times), 2),
        "infer_p50_ms":       round(np.percentile(infer_times, 50), 2),
        "infer_p95_ms":       round(np.percentile(infer_times, 95), 2),
        "infer_p99_ms":       round(np.percentile(infer_times, 99), 2),
        "wait_mean_ms":       round(np.mean(wait_times), 2),
        "total_mean_ms":      round(np.mean(total_times), 2),
        "total_p95_ms":       round(np.percentile(total_times, 95), 2),
        "gpu_util_mean":      round(np.mean(gpu_utils), 1) if gpu_utils else -1,
        "gpu_util_p95":       round(np.percentile(gpu_utils, 95), 1) if gpu_utils else -1,
        "mem_used_mean_mb":   round(np.mean(mem_used), 0) if mem_used else -1,
        "mem_used_max_mb":    round(np.max(mem_used), 0) if mem_used else -1,
        "thermals_final_C":   {k: round(v, 1) for k, v in get_thermals().items()},
    }

    print(f"\n  Total throughput:  {summary['total_fps']:.1f} FPS ({total_frames} frames in {wall_time:.1f}s)")
    print(f"  Per-stream FPS:    avg={summary['per_stream_fps_avg']:.1f}, "
          f"min={summary['per_stream_fps_min']:.1f}, each={summary['per_stream_fps']}")
    print(f"  Latency breakdown (mean per frame):")
    print(f"    decode:  {summary['read_mean_ms']:.1f}ms")
    print(f"    resize:  {summary['resize_mean_ms']:.1f}ms")
    print(f"    infer:   {summary['infer_mean_ms']:.1f}ms  (p95={summary['infer_p95_ms']:.1f}ms)")
    print(f"    wait:    {summary['wait_mean_ms']:.1f}ms  (queued for GPU lock)")
    print(f"    total:   {summary['total_mean_ms']:.1f}ms  (p95={summary['total_p95_ms']:.1f}ms)")
    print(f"  GPU util:  mean={summary['gpu_util_mean']}%, p95={summary['gpu_util_p95']}%")
    print(f"  Memory:    mean={summary['mem_used_mean_mb']:.0f}MB, max={summary['mem_used_max_mb']:.0f}MB")
    print(f"  Thermals:  {summary['thermals_final_C']}")

    # Save CSVs
    prefix = f"/data/multi_{n_streams}s_{tw}x{th}"
    with open(f"{prefix}_frames.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
        writer.writeheader()
        writer.writerows(all_metrics)

    flat_sys = [{"t": s["t"], "gpu_util": s["gpu_util"], "mem_used_mb": s["mem_used_mb"]}
                for s in sys_samples]
    with open(f"{prefix}_sysmon.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat_sys[0].keys())
        writer.writeheader()
        writer.writerows(flat_sys)

    print(f"  Saved: {prefix}_frames.csv, {prefix}_sysmon.csv")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",   default="/data/traffic_720p.mp4")
    parser.add_argument("--streams", type=int, default=None,
                        help="Number of streams (default: run 1,2,3,4 sweep)")
    parser.add_argument("--frames",  type=int, default=500,
                        help="Frames per stream")
    parser.add_argument("--width",   type=int, default=1280)
    parser.add_argument("--height",  type=int, default=720)
    parser.add_argument("--scale",   default=None,
                        help="Force resize e.g. 720p (only needed for 4K source)")
    args = parser.parse_args()

    # If input is 4K and --scale given, keep resize; if 720p source, no resize needed
    if args.scale:
        scale_map = {"360p": (640,360), "480p": (854,480), "720p": (1280,720)}
        target_size = scale_map.get(args.scale, (args.width, args.height))
    else:
        cap = cv2.VideoCapture(args.video)
        target_size = (int(cap.get(3)), int(cap.get(4)))
        cap.release()
        print(f"Source resolution detected: {target_size[0]}x{target_size[1]} (no resize)")

    stream_counts = [args.streams] if args.streams else [1, 2, 3, 4]

    all_summaries = []
    for n in stream_counts:
        s = run_benchmark(args.video, n, target_size, args.frames)
        all_summaries.append(s)

    # Final comparison table
    print(f"\n{'='*72}")
    print("MULTI-STREAM SUMMARY")
    print(f"{'='*72}")
    print(f"{'Streams':>8} {'TotalFPS':>9} {'Strm_FPS':>9} {'Infer_ms':>9} "
          f"{'InfP95':>7} {'Wait_ms':>8} {'GPU%':>5} {'Mem_MB':>7}")
    print("-" * 72)
    for s in all_summaries:
        print(f"{s['n_streams']:>8} {s['total_fps']:>9.1f} {s['per_stream_fps_avg']:>9.1f} "
              f"{s['infer_mean_ms']:>9.1f} {s['infer_p95_ms']:>7.1f} "
              f"{s['wait_mean_ms']:>8.1f} {s['gpu_util_mean']:>5.1f} "
              f"{s['mem_used_mean_mb']:>7.0f}")

    out_path = f"/data/multi_stream_summary.json"
    with open(out_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
