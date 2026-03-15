#!/usr/bin/env python3
"""
SKIP mode benchmark: graceful degradation.
Decode + downsample to 240p, no inference, passthrough only.
Measures CPU cost of decode+resize per stream at various stream counts.
"""

import argparse
import time
import csv
import json
import threading
import os
import numpy as np
import cv2

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


def system_monitor(samples, stop_event, interval=0.25):
    while not stop_event.is_set():
        samples.append({
            "t": time.time(),
            "gpu_util": get_gpu_util(),
            "mem_used_mb": get_mem_used_mb(),
        })
        time.sleep(interval)


def stream_worker(stream_id, video_path, target_size,
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

        # Downsample to 240p - this is the only operation in SKIP mode
        _ = cv2.resize(frame, (target_w, target_h))
        t_done = time.time()

        metrics.append({
            "stream":      stream_id,
            "frame":       frame_idx,
            "t_read_ms":   (t_read - t0)   * 1000,
            "t_resize_ms": (t_done - t_read) * 1000,
            "t_total_ms":  (t_done - t0)   * 1000,
            "wall_ts":     t_done,
        })
        frame_idx += 1

    cap.release()
    metrics_list.extend(metrics)


def run_benchmark(video_path, n_streams, skip_size, max_frames):
    tw, th = skip_size
    print(f"\n{'='*60}")
    print(f"[SKIP] Streams: {n_streams}  |  downsample to {tw}x{th}  |  {max_frames} frames/stream")
    print(f"{'='*60}")

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
            args=(i, video_path, skip_size,
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

    stop_mon.set()
    mon_thread.join()

    wall_time    = t_end - t_start
    total_frames = len(all_metrics)
    total_fps    = total_frames / wall_time

    stream_fps = []
    for sid in range(n_streams):
        sm = sorted([m for m in all_metrics if m["stream"] == sid],
                    key=lambda x: x["wall_ts"])
        if len(sm) >= 2:
            dur = sm[-1]["wall_ts"] - sm[0]["wall_ts"]
            stream_fps.append(len(sm) / dur if dur > 0 else 0)

    total_times = [m["t_total_ms"] for m in all_metrics]
    read_times  = [m["t_read_ms"]  for m in all_metrics]
    gpu_utils   = [s["gpu_util"]   for s in sys_samples if s["gpu_util"] >= 0]
    mem_used    = [s["mem_used_mb"] for s in sys_samples]

    summary = {
        "mode":               "skip",
        "n_streams":          n_streams,
        "skip_resolution":    f"{tw}x{th}",
        "total_fps":          round(total_fps, 2),
        "per_stream_fps_avg": round(np.mean(stream_fps), 2),
        "per_stream_fps_min": round(np.min(stream_fps), 2),
        "read_mean_ms":       round(np.mean(read_times), 2),
        "total_mean_ms":      round(np.mean(total_times), 2),
        "total_p95_ms":       round(np.percentile(total_times, 95), 2),
        "gpu_util_mean":      round(np.mean(gpu_utils), 1) if gpu_utils else -1,
        "mem_used_mean_mb":   round(np.mean(mem_used), 0) if mem_used else -1,
    }

    print(f"  Total FPS:       {summary['total_fps']:.1f}")
    print(f"  Per-stream FPS:  avg={summary['per_stream_fps_avg']:.1f}, "
          f"min={summary['per_stream_fps_min']:.1f}")
    print(f"  Decode mean:     {summary['read_mean_ms']:.1f}ms")
    print(f"  Total mean:      {summary['total_mean_ms']:.1f}ms  "
          f"(p95={summary['total_p95_ms']:.1f}ms)")
    print(f"  GPU util:        mean={summary['gpu_util_mean']}%  (should be ~0)")
    print(f"  Memory:          {summary['mem_used_mean_mb']:.0f}MB")

    prefix = f"/data/skip_{n_streams}s_{tw}x{th}"
    with open(f"{prefix}_frames.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"  Saved: {prefix}_frames.csv")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  default="/data/traffic_720p.mp4")
    parser.add_argument("--frames", type=int, default=300)
    args = parser.parse_args()

    skip_size    = (426, 240)   # 240p
    stream_counts = [1, 2, 4, 6, 8, 10, 12]

    summaries = []
    for n in stream_counts:
        s = run_benchmark(args.video, n, skip_size, args.frames)
        summaries.append(s)

    print(f"\n{'='*60}")
    print("SKIP MODE SUMMARY (240p passthrough, no inference)")
    print(f"{'='*60}")
    print(f"{'Streams':>8}  {'Strm_FPS':>9}  {'Total_ms':>9}  {'P95_ms':>7}  {'GPU%':>5}")
    print("-" * 46)
    for s in summaries:
        print(f"{s['n_streams']:>8}  {s['per_stream_fps_avg']:>9.1f}  "
              f"{s['total_mean_ms']:>9.1f}  {s['total_p95_ms']:>7.1f}  "
              f"{s['gpu_util_mean']:>5.1f}")

    with open("/data/skip_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print("\nSaved to /data/skip_summary.json")


if __name__ == "__main__":
    main()
