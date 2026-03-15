#!/usr/bin/env python3
"""
Batched multi-stream YOLO inference benchmark.

Architecture:
  N stream threads  →  shared frame queue  →  1 inference thread (batch GPU)
                                           →  result back to each stream thread

Each stream thread decodes + resizes (CPU), submits frame to queue,
then blocks waiting for its result. The inference thread collects
up to batch_size frames and fires a single model([f0,f1,...]) call.
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
        for tz in sorted(os.listdir("/sys/class/thermal")):
            path = os.path.join("/sys/class/thermal", tz)
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


# ── Batch inference engine ────────────────────────────────────────────────────

class BatchInferenceEngine:
    """
    Runs a background thread that collects frames from all streams
    into a batch and fires a single GPU inference call.

    Frame submission protocol:
      frame_queue.put( (stream_id, frame_idx, frame_ndarray, result_dict, done_event) )
    After done_event is set, result_dict contains:
      { 'n_det': int, 'infer_ms': float, 'batch_size': int, 't_queued': float, 't_done': float }
    """

    def __init__(self, model_path, n_streams, device="cuda", max_wait_ms=8.0):
        self.n_streams  = n_streams
        self.max_wait   = max_wait_ms / 1000.0   # max time to wait for a full batch
        self.frame_q    = queue.Queue()
        self.device     = device
        self._stop      = threading.Event()

        print(f"Loading model on {device}...")
        self.model = YOLO(model_path)
        self.model.to(device)

        # Warmup
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        for _ in range(10):
            self.model([dummy] * n_streams, verbose=False)
        print(f"Model ready. Batch size={n_streams}, device={next(self.model.model.parameters()).device}")

        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()

    def _inference_loop(self):
        while not self._stop.is_set():
            batch_items = []  # list of (stream_id, frame_idx, frame, result_dict, done_event)

            # Block until first frame arrives
            try:
                item = self.frame_q.get(timeout=0.1)
                batch_items.append(item)
            except queue.Empty:
                continue

            # Collect remaining frames up to n_streams, with a short wait
            deadline = time.time() + self.max_wait
            while len(batch_items) < self.n_streams:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    item = self.frame_q.get(timeout=remaining)
                    batch_items.append(item)
                except queue.Empty:
                    break

            # Build batch and infer
            frames = [item[2] for item in batch_items]
            t_infer_start = time.time()
            results = self.model(frames, verbose=False)
            t_infer_end = time.time()

            infer_ms   = (t_infer_end - t_infer_start) * 1000
            batch_size = len(frames)

            # Distribute results back to each waiting stream thread
            for i, (sid, fidx, frame, result_dict, done_event) in enumerate(batch_items):
                n_det = len(results[i].boxes) if results else 0
                result_dict.update({
                    "n_det":      n_det,
                    "infer_ms":   infer_ms,
                    "batch_size": batch_size,
                    "t_done":     t_infer_end,
                })
                done_event.set()

    def submit(self, stream_id, frame_idx, frame):
        """Submit a frame for inference. Blocks until result is ready."""
        result_dict = {}
        done_event  = threading.Event()
        t_queued    = time.time()
        self.frame_q.put((stream_id, frame_idx, frame, result_dict, done_event))
        done_event.wait()
        result_dict["t_queued"] = t_queued
        return result_dict

    def stop(self):
        self._stop.set()
        self._thread.join()


# ── Per-stream worker ─────────────────────────────────────────────────────────

def stream_worker(stream_id, video_path, target_size, engine,
                  max_frames, metrics_list, ready_event, start_event):
    target_w, target_h = target_size

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Stream {stream_id}] ERROR: cannot open {video_path}")
        ready_event.set()
        return

    metrics = []
    ready_event.set()
    start_event.wait()

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

        frame = cv2.resize(frame, (target_w, target_h))
        t_resize = time.time()

        # Submit to batch engine (blocks until GPU inference completes)
        result = engine.submit(stream_id, frame_idx, frame)
        t_done = time.time()

        wait_ms = (result["t_done"] - t_resize) * 1000 - result["infer_ms"]

        metrics.append({
            "stream":      stream_id,
            "frame":       frame_idx,
            "t_read_ms":   (t_read   - t0)    * 1000,
            "t_resize_ms": (t_resize - t_read) * 1000,
            "t_queue_ms":  wait_ms,
            "t_infer_ms":  result["infer_ms"],
            "batch_size":  result["batch_size"],
            "t_total_ms":  (t_done   - t0)    * 1000,
            "detections":  result["n_det"],
            "wall_ts":     t_done,
        })
        frame_idx += 1

    cap.release()
    metrics_list.extend(metrics)


# ── Main benchmark ────────────────────────────────────────────────────────────

def run_benchmark(video_path, n_streams, target_size, max_frames_per_stream):
    tw, th = target_size
    print(f"\n{'='*60}")
    print(f"[BATCH] Streams: {n_streams}  |  {tw}x{th}  |  {max_frames_per_stream} frames/stream")
    print(f"{'='*60}")

    engine = BatchInferenceEngine("yolov8n.pt", n_streams=n_streams)

    sys_samples = []
    stop_mon    = threading.Event()
    mon_thread  = threading.Thread(target=system_monitor, args=(sys_samples, stop_mon))
    mon_thread.start()

    all_metrics  = []
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

    for e in ready_events:
        e.wait()
    t_start = time.time()
    start_event.set()

    for t in threads:
        t.join()
    t_end = time.time()

    engine.stop()
    stop_mon.set()
    mon_thread.join()

    # ── Analysis ──────────────────────────────────────────────────────────────
    wall_time   = t_end - t_start
    total_frames = len(all_metrics)
    total_fps   = total_frames / wall_time

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
    queue_times  = [m["t_queue_ms"]  for m in all_metrics]
    batch_sizes  = [m["batch_size"]  for m in all_metrics]
    gpu_utils    = [s["gpu_util"]    for s in sys_samples if s["gpu_util"] >= 0]
    mem_used     = [s["mem_used_mb"] for s in sys_samples]

    summary = {
        "mode":               "batch",
        "n_streams":          n_streams,
        "resolution":         f"{tw}x{th}",
        "total_frames":       total_frames,
        "wall_time_s":        round(wall_time, 2),
        "total_fps":          round(total_fps, 2),
        "per_stream_fps_avg": round(np.mean(stream_fps), 2),
        "per_stream_fps_min": round(np.min(stream_fps), 2),
        "per_stream_fps":     [round(f, 2) for f in stream_fps],
        "infer_mean_ms":      round(np.mean(infer_times), 2),
        "infer_p95_ms":       round(np.percentile(infer_times, 95), 2),
        "queue_mean_ms":      round(np.mean(queue_times), 2),
        "total_mean_ms":      round(np.mean(total_times), 2),
        "total_p95_ms":       round(np.percentile(total_times, 95), 2),
        "avg_batch_size":     round(np.mean(batch_sizes), 2),
        "gpu_util_mean":      round(np.mean(gpu_utils), 1) if gpu_utils else -1,
        "gpu_util_p95":       round(np.percentile(gpu_utils, 95), 1) if gpu_utils else -1,
        "mem_used_mean_mb":   round(np.mean(mem_used), 0) if mem_used else -1,
        "thermals_final_C":   {k: round(v, 1) for k, v in get_thermals().items()},
    }

    print(f"\n  Total throughput:  {summary['total_fps']:.1f} FPS")
    print(f"  Per-stream FPS:    avg={summary['per_stream_fps_avg']:.1f}, "
          f"min={summary['per_stream_fps_min']:.1f}, each={summary['per_stream_fps']}")
    print(f"  Avg actual batch:  {summary['avg_batch_size']:.2f} / {n_streams}")
    print(f"  Latency (mean):")
    print(f"    infer:  {summary['infer_mean_ms']:.1f}ms  (p95={summary['infer_p95_ms']:.1f}ms)")
    print(f"    queue:  {summary['queue_mean_ms']:.1f}ms")
    print(f"    total:  {summary['total_mean_ms']:.1f}ms  (p95={summary['total_p95_ms']:.1f}ms)")
    print(f"  GPU util:  mean={summary['gpu_util_mean']}%, p95={summary['gpu_util_p95']}%")
    print(f"  Memory:    mean={summary['mem_used_mean_mb']:.0f}MB, max={summary['mem_used_mean_mb']:.0f}MB")
    print(f"  Thermals:  {summary['thermals_final_C']}")

    prefix = f"/data/batch_{n_streams}s_{tw}x{th}"
    with open(f"{prefix}_frames.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
        writer.writeheader()
        writer.writerows(all_metrics)

    sys_flat = [{"t": s["t"], "gpu_util": s["gpu_util"], "mem_used_mb": s["mem_used_mb"]}
                for s in sys_samples]
    with open(f"{prefix}_sysmon.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sys_flat[0].keys())
        writer.writeheader()
        writer.writerows(sys_flat)

    print(f"  Saved: {prefix}_frames.csv")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",   default="/data/traffic_720p.mp4")
    parser.add_argument("--streams", type=int, default=None,
                        help="Single stream count (default: sweep 1-4)")
    parser.add_argument("--frames",  type=int, default=500)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    target_size = (int(cap.get(3)), int(cap.get(4)))
    cap.release()
    print(f"Source: {target_size[0]}x{target_size[1]}")

    stream_counts = [args.streams] if args.streams else [1, 2, 3, 4]

    summaries = []
    for n in stream_counts:
        s = run_benchmark(args.video, n, target_size, args.frames)
        summaries.append(s)

    # Comparison: batch vs serial (reference values from previous run)
    serial_ref = {1: (30.4, 0.0, 35.6), 2: (11.1, 42.6, 95.5),
                  3: (7.1, 92.0, 146.6), 4: (5.3, 140.4, 194.9)}

    print(f"\n{'='*80}")
    print("BATCH vs SERIAL COMPARISON")
    print(f"{'='*80}")
    print(f"{'Streams':>8} | {'Batch FPS':>10} {'Serial FPS':>11} | "
          f"{'Batch P95':>10} {'Serial P95':>11} | {'GPU%':>5}")
    print("-" * 80)
    for s in summaries:
        n   = s["n_streams"]
        ref = serial_ref.get(n, (0, 0, 0))
        print(f"{n:>8} | {s['per_stream_fps_avg']:>10.1f} {ref[0]:>11.1f} | "
              f"{s['total_p95_ms']:>10.1f} {ref[2]:>11.1f} | "
              f"{s['gpu_util_mean']:>5.1f}")

    out = "/data/batch_summary.json"
    with open(out, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
