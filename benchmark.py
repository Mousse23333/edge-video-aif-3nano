#!/usr/bin/env python3
"""Single-stream YOLO inference benchmark on Jetson.

Profiles per-frame latency breakdown, throughput, GPU/memory utilization
over a longer run for stable performance characterization.
"""

import time
import csv
import sys
import os
import threading
import json
import cv2
import numpy as np
from ultralytics import YOLO

GPU_LOAD_PATHS = [
    "/sys/devices/platform/bus@0/17000000.gpu/load",
    "/sys/devices/gpu.0/load",
]

MEMINFO_PATH = "/proc/meminfo"


def get_gpu_util():
    for p in GPU_LOAD_PATHS:
        try:
            with open(p, "r") as f:
                return int(f.read().strip()) / 10.0
        except:
            continue
    return -1.0


def get_mem_info_mb():
    """Return (total, available, used) in MB from /proc/meminfo."""
    try:
        info = {}
        with open(MEMINFO_PATH, "r") as f:
            for line in f:
                parts = line.split()
                if parts[0] in ("MemTotal:", "MemAvailable:", "MemFree:", "Buffers:", "Cached:"):
                    info[parts[0].rstrip(":")] = int(parts[1]) / 1024  # kB -> MB
        total = info.get("MemTotal", 0)
        avail = info.get("MemAvailable", 0)
        return total, avail, total - avail
    except:
        return -1, -1, -1


def get_thermal():
    """Read Jetson thermal zones."""
    temps = {}
    base = "/sys/class/thermal"
    try:
        for tz in sorted(os.listdir(base)):
            tz_path = os.path.join(base, tz)
            if not os.path.isdir(tz_path):
                continue
            try:
                with open(os.path.join(tz_path, "type"), "r") as f:
                    name = f.read().strip()
                with open(os.path.join(tz_path, "temp"), "r") as f:
                    temp_mc = int(f.read().strip())
                temps[name] = temp_mc / 1000.0
            except:
                continue
    except:
        pass
    return temps


def system_monitor(samples, stop_event, interval=0.25):
    """Background thread sampling GPU, memory, and thermals."""
    while not stop_event.is_set():
        mem_total, mem_avail, mem_used = get_mem_info_mb()
        samples.append({
            "t": time.time(),
            "gpu_util": get_gpu_util(),
            "mem_used_mb": mem_used,
            "mem_avail_mb": mem_avail,
            "thermals": get_thermal(),
        })
        time.sleep(interval)


def benchmark_yolo(video_path, resolution, max_frames, model, loop_video=True):
    target_w, target_h = resolution

    print(f"\n{'='*60}")
    print(f"Benchmark: {target_w}x{target_h}, max_frames={max_frames}")
    print(f"{'='*60}")

    # Warmup
    dummy = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    for _ in range(10):
        model(dummy, verbose=False)
    print("Warmup done (10 frames).")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return None

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Source: {int(cap.get(3))}x{int(cap.get(4))} @ {src_fps:.2f} FPS, {total_src_frames} frames")

    # Start system monitor
    sys_samples = []
    stop_event = threading.Event()
    mon_thread = threading.Thread(target=system_monitor, args=(sys_samples, stop_event))
    mon_thread.start()

    frame_metrics = []
    frame_idx = 0
    t_start = time.time()

    while frame_idx < max_frames:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            if loop_video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                break
        t_read = time.time()

        frame_resized = cv2.resize(frame, (target_w, target_h))
        t_resize = time.time()

        results = model(frame_resized, verbose=False)
        t_infer = time.time()

        num_det = len(results[0].boxes) if results else 0

        frame_metrics.append({
            "frame": frame_idx,
            "t_read_ms": (t_read - t0) * 1000,
            "t_resize_ms": (t_resize - t_read) * 1000,
            "t_infer_ms": (t_infer - t_resize) * 1000,
            "t_total_ms": (t_infer - t0) * 1000,
            "detections": num_det,
            "wall_ts": t_infer,
        })
        frame_idx += 1

        # Progress every 200 frames
        if frame_idx % 200 == 0:
            elapsed = time.time() - t_start
            print(f"  [{frame_idx}/{max_frames}] {elapsed:.1f}s elapsed, "
                  f"avg FPS: {frame_idx/elapsed:.1f}, "
                  f"GPU: {get_gpu_util():.0f}%")

    t_end = time.time()
    cap.release()
    stop_event.set()
    mon_thread.join()

    # === Analysis ===
    total_time = t_end - t_start
    infer_times = [m["t_infer_ms"] for m in frame_metrics]
    total_times = [m["t_total_ms"] for m in frame_metrics]
    read_times = [m["t_read_ms"] for m in frame_metrics]
    resize_times = [m["t_resize_ms"] for m in frame_metrics]
    gpu_utils = [s["gpu_util"] for s in sys_samples if s["gpu_util"] >= 0]
    mem_used = [s["mem_used_mb"] for s in sys_samples]

    # Windowed FPS (1-second windows)
    fps_windows = []
    if frame_metrics:
        win_start = frame_metrics[0]["wall_ts"]
        win_count = 0
        for m in frame_metrics:
            if m["wall_ts"] - win_start >= 1.0:
                fps_windows.append(win_count)
                win_start = m["wall_ts"]
                win_count = 0
            win_count += 1
        if win_count > 0:
            fps_windows.append(win_count)

    # Thermal snapshot
    final_thermals = get_thermal()

    summary = {
        "resolution": f"{target_w}x{target_h}",
        "frames_processed": frame_idx,
        "wall_time_s": round(total_time, 2),
        "avg_fps": round(frame_idx / total_time, 2) if total_time > 0 else 0,
        "fps_p5": round(np.percentile(fps_windows, 5), 1) if fps_windows else 0,
        "fps_p50": round(np.percentile(fps_windows, 50), 1) if fps_windows else 0,
        "fps_min": min(fps_windows) if fps_windows else 0,
        "read_mean_ms": round(np.mean(read_times), 2),
        "resize_mean_ms": round(np.mean(resize_times), 2),
        "infer_mean_ms": round(np.mean(infer_times), 2),
        "infer_p50_ms": round(np.percentile(infer_times, 50), 2),
        "infer_p95_ms": round(np.percentile(infer_times, 95), 2),
        "infer_p99_ms": round(np.percentile(infer_times, 99), 2),
        "infer_max_ms": round(np.max(infer_times), 2),
        "total_mean_ms": round(np.mean(total_times), 2),
        "total_p95_ms": round(np.percentile(total_times, 95), 2),
        "gpu_util_mean": round(np.mean(gpu_utils), 1) if gpu_utils else -1,
        "gpu_util_p95": round(np.percentile(gpu_utils, 95), 1) if gpu_utils else -1,
        "gpu_util_max": round(np.max(gpu_utils), 1) if gpu_utils else -1,
        "mem_used_mean_mb": round(np.mean(mem_used), 0) if mem_used else -1,
        "mem_used_max_mb": round(np.max(mem_used), 0) if mem_used else -1,
        "thermals_final_C": {k: round(v, 1) for k, v in final_thermals.items()},
    }

    print(f"\n--- Results: {target_w}x{target_h} ({frame_idx} frames) ---")
    print(f"  Wall time:       {summary['wall_time_s']}s")
    print(f"  FPS:             avg={summary['avg_fps']}, p50={summary['fps_p50']}, "
          f"p5={summary['fps_p5']}, min={summary['fps_min']}")
    print(f"  Latency breakdown (mean):")
    print(f"    decode:  {summary['read_mean_ms']:.1f}ms")
    print(f"    resize:  {summary['resize_mean_ms']:.1f}ms")
    print(f"    infer:   {summary['infer_mean_ms']:.1f}ms")
    print(f"    total:   {summary['total_mean_ms']:.1f}ms")
    print(f"  Inference latency: p50={summary['infer_p50_ms']:.1f}, "
          f"p95={summary['infer_p95_ms']:.1f}, p99={summary['infer_p99_ms']:.1f}, "
          f"max={summary['infer_max_ms']:.1f}ms")
    print(f"  GPU util:  mean={summary['gpu_util_mean']}%, "
          f"p95={summary['gpu_util_p95']}%, max={summary['gpu_util_max']}%")
    print(f"  Memory:    mean={summary['mem_used_mean_mb']:.0f}MB, "
          f"max={summary['mem_used_max_mb']:.0f}MB")
    print(f"  Thermals:  {summary['thermals_final_C']}")

    # Save per-frame CSV
    csv_path = f"/data/bench_{target_w}x{target_h}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=frame_metrics[0].keys())
        writer.writeheader()
        writer.writerows(frame_metrics)
    print(f"  Per-frame CSV: {csv_path}")

    # Save system monitor CSV
    sys_csv_path = f"/data/sysmon_{target_w}x{target_h}.csv"
    with open(sys_csv_path, "w", newline="") as f:
        flat = []
        for s in sys_samples:
            row = {"t": s["t"], "gpu_util": s["gpu_util"],
                   "mem_used_mb": s["mem_used_mb"], "mem_avail_mb": s["mem_avail_mb"]}
            for k, v in s["thermals"].items():
                row[f"temp_{k}"] = v
            flat.append(row)
        if flat:
            writer = csv.DictWriter(f, fieldnames=flat[0].keys())
            writer.writeheader()
            writer.writerows(flat)
    print(f"  System monitor CSV: {sys_csv_path}")

    return summary


def main():
    video_path = "/data/traffic.webm"
    max_frames = 1000  # ~33s at 30fps, loops video if needed

    print("Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    model.to("cuda")
    print(f"Model loaded. Device: {next(model.model.parameters()).device}")

    resolutions = [
        (640, 360),
        (854, 480),
        (1280, 720),
    ]

    all_summaries = []
    for res in resolutions:
        s = benchmark_yolo(video_path, res, max_frames, model)
        if s:
            all_summaries.append(s)

    # Comparison table
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON (1000 frames per resolution)")
    print(f"{'='*70}")
    print(f"{'Res':<10} {'FPS':>5} {'FPS_p5':>7} {'Decode':>7} {'Resize':>7} "
          f"{'Infer':>7} {'Inf_p95':>8} {'Total':>7} {'GPU%':>5} {'Mem_MB':>7}")
    print("-" * 70)
    for s in all_summaries:
        print(f"{s['resolution']:<10} {s['avg_fps']:>5.1f} {s['fps_p5']:>7.1f} "
              f"{s['read_mean_ms']:>7.1f} {s['resize_mean_ms']:>7.1f} "
              f"{s['infer_mean_ms']:>7.1f} {s['infer_p95_ms']:>8.1f} "
              f"{s['total_mean_ms']:>7.1f} {s['gpu_util_mean']:>5.1f} "
              f"{s['mem_used_mean_mb']:>7.0f}")

    # Save summary JSON
    summary_path = "/data/benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nFull summary saved to {summary_path}")


if __name__ == "__main__":
    main()
