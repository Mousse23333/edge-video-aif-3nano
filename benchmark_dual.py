#!/usr/bin/env python3
"""
Dual-model batched inference benchmark.

Two YOLOv8n instances run on separate CUDA streams, each handling
half the streams. Goal: push GPU utilization above single-model ceiling.

Topology:
  Streams 0..N/2-1  →  BatchEngine-0 (cuda stream 0)  ─┐
  Streams N/2..N-1  →  BatchEngine-1 (cuda stream 1)  ─┘→ GPU (concurrent)
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


# ── Single batch engine (one model, one CUDA stream) ─────────────────────────

class BatchEngine:
    def __init__(self, engine_id, n_streams, max_wait_ms=8.0):
        self.engine_id = engine_id
        self.n_streams = n_streams
        self.max_wait  = max_wait_ms / 1000.0
        self.frame_q   = queue.Queue()
        self._stop     = threading.Event()

        self.model = YOLO("yolov8n.pt")
        self.model.to("cuda")

        # Dedicated CUDA stream for this engine
        self.cuda_stream = torch.cuda.Stream()

        # Warmup on this stream
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        with torch.cuda.stream(self.cuda_stream):
            for _ in range(10):
                self.model([dummy] * max(1, n_streams), verbose=False)
        torch.cuda.synchronize()

        print(f"  Engine-{engine_id}: ready, {n_streams} streams, "
              f"CUDA stream id={self.cuda_stream.cuda_stream}")

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                item = self.frame_q.get(timeout=0.1)
            except queue.Empty:
                continue

            batch_items = [item]
            deadline = time.time() + self.max_wait
            while len(batch_items) < self.n_streams:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    batch_items.append(self.frame_q.get(timeout=remaining))
                except queue.Empty:
                    break

            frames = [it[2] for it in batch_items]

            # Run inference on dedicated CUDA stream
            with torch.cuda.stream(self.cuda_stream):
                t0 = time.time()
                results = self.model(frames, verbose=False)
                torch.cuda.synchronize()
                t1 = time.time()

            infer_ms   = (t1 - t0) * 1000
            batch_size = len(frames)

            for i, (sid, fidx, frame, result_dict, done_event) in enumerate(batch_items):
                n_det = len(results[i].boxes) if results else 0
                result_dict.update({
                    "n_det":      n_det,
                    "infer_ms":   infer_ms,
                    "batch_size": batch_size,
                    "engine_id":  self.engine_id,
                    "t_done":     t1,
                })
                done_event.set()

    def submit(self, stream_id, frame_idx, frame):
        result_dict = {}
        done_event  = threading.Event()
        self.frame_q.put((stream_id, frame_idx, frame, result_dict, done_event))
        done_event.wait()
        return result_dict

    def stop(self):
        self._stop.set()
        self._thread.join()


# ── Stream worker ─────────────────────────────────────────────────────────────

def stream_worker(stream_id, video_path, target_size, engine,
                  max_frames, metrics_list, ready_event, start_event):
    target_w, target_h = target_size
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
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

        result = engine.submit(stream_id, frame_idx, frame)
        t_done = time.time()

        metrics.append({
            "stream":      stream_id,
            "engine":      result.get("engine_id", -1),
            "frame":       frame_idx,
            "t_read_ms":   (t_read   - t0)    * 1000,
            "t_resize_ms": (t_resize - t_read) * 1000,
            "t_infer_ms":  result["infer_ms"],
            "batch_size":  result["batch_size"],
            "t_total_ms":  (t_done   - t0)    * 1000,
            "detections":  result["n_det"],
            "wall_ts":     t_done,
        })
        frame_idx += 1

    cap.release()
    metrics_list.extend(metrics)


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_dual_benchmark(video_path, n_streams, target_size, max_frames):
    tw, th = target_size
    print(f"\n{'='*60}")
    print(f"[DUAL] Streams: {n_streams}  |  {tw}x{th}  |  {max_frames} frames/stream")
    print(f"{'='*60}")

    # Split streams evenly between two engines
    n0 = n_streams // 2
    n1 = n_streams - n0

    print(f"Engine split: Engine-0 handles {n0} streams, Engine-1 handles {n1} streams")
    engine0 = BatchEngine(engine_id=0, n_streams=max(1, n0))
    engine1 = BatchEngine(engine_id=1, n_streams=max(1, n1))

    mem_after_load = get_mem_used_mb()
    print(f"Memory after loading 2 models: {mem_after_load:.0f}MB")

    sys_samples = []
    stop_mon    = threading.Event()
    mon_thread  = threading.Thread(target=system_monitor, args=(sys_samples, stop_mon))
    mon_thread.start()

    all_metrics  = []
    ready_events = [threading.Event() for _ in range(n_streams)]
    start_event  = threading.Event()

    threads = []
    for i in range(n_streams):
        # Assign first half to engine0, second half to engine1
        engine = engine0 if i < n0 else engine1
        t = threading.Thread(
            target=stream_worker,
            args=(i, video_path, target_size, engine,
                  max_frames, all_metrics,
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

    engine0.stop()
    engine1.stop()
    stop_mon.set()
    mon_thread.join()

    wall_time    = t_end - t_start
    total_frames = len(all_metrics)
    total_fps    = total_frames / wall_time

    stream_fps = []
    for sid in range(n_streams):
        sm = sorted([m for m in all_metrics if m["stream"] == sid], key=lambda x: x["wall_ts"])
        if len(sm) >= 2:
            dur = sm[-1]["wall_ts"] - sm[0]["wall_ts"]
            stream_fps.append(len(sm) / dur if dur > 0 else 0)

    infer_times = [m["t_infer_ms"] for m in all_metrics]
    total_times = [m["t_total_ms"] for m in all_metrics]
    batch_sizes = [m["batch_size"]  for m in all_metrics]
    gpu_utils   = [s["gpu_util"]    for s in sys_samples if s["gpu_util"] >= 0]
    mem_used    = [s["mem_used_mb"] for s in sys_samples]

    summary = {
        "mode":               "dual",
        "n_streams":          n_streams,
        "resolution":         f"{tw}x{th}",
        "total_fps":          round(total_fps, 2),
        "per_stream_fps_avg": round(np.mean(stream_fps), 2),
        "per_stream_fps_min": round(np.min(stream_fps), 2),
        "per_stream_fps":     [round(f, 2) for f in stream_fps],
        "infer_mean_ms":      round(np.mean(infer_times), 2),
        "infer_p95_ms":       round(np.percentile(infer_times, 95), 2),
        "total_mean_ms":      round(np.mean(total_times), 2),
        "total_p95_ms":       round(np.percentile(total_times, 95), 2),
        "avg_batch_size":     round(np.mean(batch_sizes), 2),
        "gpu_util_mean":      round(np.mean(gpu_utils), 1) if gpu_utils else -1,
        "gpu_util_p95":       round(np.percentile(gpu_utils, 95), 1) if gpu_utils else -1,
        "mem_used_mean_mb":   round(np.mean(mem_used), 0) if mem_used else -1,
        "thermals_final_C":   {k: round(v, 1) for k, v in get_thermals().items()},
    }

    print(f"\n  Total FPS:         {summary['total_fps']:.1f}")
    print(f"  Per-stream FPS:    avg={summary['per_stream_fps_avg']:.1f}, "
          f"min={summary['per_stream_fps_min']:.1f}")
    print(f"  Avg actual batch:  {summary['avg_batch_size']:.2f}")
    print(f"  Infer: mean={summary['infer_mean_ms']:.1f}ms, p95={summary['infer_p95_ms']:.1f}ms")
    print(f"  Total: mean={summary['total_mean_ms']:.1f}ms, p95={summary['total_p95_ms']:.1f}ms")
    print(f"  GPU util: mean={summary['gpu_util_mean']}%, p95={summary['gpu_util_p95']}%")
    print(f"  Memory:   mean={summary['mem_used_mean_mb']:.0f}MB")
    print(f"  Thermals: {summary['thermals_final_C']}")

    out = f"/data/dual_{n_streams}s_{tw}x{th}.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",   default="/data/traffic_720p.mp4")
    parser.add_argument("--streams", type=int, default=None)
    parser.add_argument("--frames",  type=int, default=300)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    target_size = (int(cap.get(3)), int(cap.get(4)))
    cap.release()
    print(f"Source: {target_size[0]}x{target_size[1]}")

    # Reference from single-model batch
    single_ref = {
        4: (13.2, 88),
        5: (11.1, 127),
        6: (9.0,  161),
        8: (7.2,  195),
    }

    stream_counts = [args.streams] if args.streams else [4, 6, 8, 10]
    summaries = []
    for n in stream_counts:
        s = run_dual_benchmark(args.video, n, target_size, args.frames)
        summaries.append(s)

    print(f"\n{'='*72}")
    print("DUAL-MODEL vs SINGLE-MODEL COMPARISON")
    print(f"{'='*72}")
    print(f"{'Streams':>8} | {'Dual FPS':>9} {'Single FPS':>11} | "
          f"{'Dual P95':>9} {'Single P95':>11} | {'GPU%':>5}")
    print("-" * 72)
    for s in summaries:
        n   = s["n_streams"]
        ref = single_ref.get(n, (0, 0))
        print(f"{n:>8} | {s['per_stream_fps_avg']:>9.1f} {ref[0]:>11.1f} | "
              f"{s['total_p95_ms']:>9.1f} {ref[1]:>11.0f} | "
              f"{s['gpu_util_mean']:>5.1f}")

    with open("/data/dual_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)


if __name__ == "__main__":
    main()
