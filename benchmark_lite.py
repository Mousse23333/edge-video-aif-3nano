#!/usr/bin/env python3
"""
LITE K=3 batch benchmark: measures actual inference FPS and total FPS
when each stream only submits every K-th frame for inference.

Comparison with FULL (K=1) to determine real capacity difference.
"""

import argparse
import time
import json
import threading
import queue
import numpy as np
import cv2
from ultralytics import YOLO

GPU_LOAD_PATH = "/sys/devices/platform/bus@0/17000000.gpu/load"


def get_gpu_util():
    try:
        with open(GPU_LOAD_PATH) as f:
            return int(f.read().strip()) / 10.0
    except:
        return -1.0


def system_monitor(samples, stop_event, interval=0.25):
    while not stop_event.is_set():
        samples.append({"t": time.time(), "gpu_util": get_gpu_util()})
        time.sleep(interval)


class BatchEngine:
    def __init__(self, n_expected, max_wait_ms=8.0):
        self.n_expected = max(1, n_expected)
        self.max_wait = max_wait_ms / 1000.0
        self.frame_q = queue.Queue()
        self._stop = threading.Event()
        self.model = YOLO("yolov8n.pt")
        self.model.to("cuda")
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        for _ in range(10):
            self.model(dummy, verbose=False)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                item = self.frame_q.get(timeout=0.1)
            except queue.Empty:
                continue
            batch = [item]
            deadline = time.time() + self.max_wait
            while len(batch) < self.n_expected:
                rem = deadline - time.time()
                if rem <= 0:
                    break
                try:
                    batch.append(self.frame_q.get(timeout=rem))
                except queue.Empty:
                    break
            frames = [b[2] for b in batch]
            t0 = time.time()
            results = self.model(frames, verbose=False)
            t1 = time.time()
            infer_ms = (t1 - t0) * 1000
            for i, (sid, fidx, fr, rd, ev) in enumerate(batch):
                rd.update({"n_det": len(results[i].boxes), "infer_ms": infer_ms,
                           "batch_size": len(frames), "t_done": t1})
                ev.set()

    def submit(self, sid, fidx, frame):
        rd = {}
        ev = threading.Event()
        self.frame_q.put((sid, fidx, frame, rd, ev))
        ev.wait()
        return rd

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)


def stream_worker(stream_id, video_path, target_size, engine, K,
                  max_frames, metrics, ready_event, start_event):
    tw, th = target_size
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        ready_event.set()
        return

    local_metrics = []
    ready_event.set()
    start_event.wait()

    for fidx in range(max_frames):
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        t_read = time.time()

        frame_resized = cv2.resize(frame, (tw, th))
        t_resize = time.time()

        is_infer = (fidx % K == 0)
        if is_infer:
            result = engine.submit(stream_id, fidx, frame_resized)
            t_done = time.time()
            local_metrics.append({
                "stream": stream_id, "frame": fidx, "is_infer": True,
                "t_total_ms": (t_done - t0) * 1000,
                "t_infer_ms": result["infer_ms"],
                "batch_size": result["batch_size"],
                "wall_ts": t_done,
            })
        else:
            t_done = time.time()
            local_metrics.append({
                "stream": stream_id, "frame": fidx, "is_infer": False,
                "t_total_ms": (t_done - t0) * 1000,
                "t_infer_ms": 0, "batch_size": 0,
                "wall_ts": t_done,
            })

    cap.release()
    metrics.extend(local_metrics)


def run_benchmark(video_path, n_streams, target_size, K, max_frames):
    tw, th = target_size
    print(f"\n{'='*60}")
    print(f"K={K}  Streams={n_streams}  Res={tw}x{th}  Frames/stream={max_frames}")
    print(f"{'='*60}")

    # For LITE K=3, each stream submits 1/K frames, so expected batch size
    # is the number of streams that have an inference frame in the same cycle.
    # In practice with asynchronous threads, it's unpredictable. Just use n_streams.
    engine = BatchEngine(n_expected=n_streams)

    sys_samples = []
    stop_mon = threading.Event()
    mon_thread = threading.Thread(target=system_monitor, args=(sys_samples, stop_mon))
    mon_thread.start()

    all_metrics = []
    ready_events = [threading.Event() for _ in range(n_streams)]
    start_event = threading.Event()
    threads = []

    for i in range(n_streams):
        t = threading.Thread(
            target=stream_worker,
            args=(i, video_path, target_size, engine, K,
                  max_frames, all_metrics, ready_events[i], start_event),
            daemon=True)
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

    wall_time = t_end - t_start
    total_frames = len(all_metrics)
    infer_frames = [m for m in all_metrics if m["is_infer"]]

    total_fps = total_frames / wall_time
    infer_total_fps = len(infer_frames) / wall_time

    # Per-stream
    stream_fps = []
    stream_infer_fps = []
    for sid in range(n_streams):
        sm = sorted([m for m in all_metrics if m["stream"] == sid],
                    key=lambda x: x["wall_ts"])
        if len(sm) >= 2:
            dur = sm[-1]["wall_ts"] - sm[0]["wall_ts"]
            stream_fps.append(len(sm) / dur if dur > 0 else 0)
            si = [m for m in sm if m["is_infer"]]
            stream_infer_fps.append(len(si) / dur if dur > 0 else 0)

    infer_latencies = [m["t_infer_ms"] for m in infer_frames]
    gpu_utils = [s["gpu_util"] for s in sys_samples if s["gpu_util"] >= 0]

    summary = {
        "K": K,
        "n_streams": n_streams,
        "wall_time_s": round(wall_time, 2),
        "total_fps": round(total_fps, 1),
        "infer_total_fps": round(infer_total_fps, 1),
        "per_stream_fps_avg": round(np.mean(stream_fps), 1),
        "per_stream_infer_fps_avg": round(np.mean(stream_infer_fps), 1),
        "per_stream_infer_fps_min": round(np.min(stream_infer_fps), 1),
        "infer_mean_ms": round(np.mean(infer_latencies), 1) if infer_latencies else 0,
        "infer_p95_ms": round(np.percentile(infer_latencies, 95), 1) if infer_latencies else 0,
        "gpu_util_mean": round(np.mean(gpu_utils), 1) if gpu_utils else -1,
    }

    print(f"  Total FPS:        {summary['total_fps']} (all frames)")
    print(f"  Infer total FPS:  {summary['infer_total_fps']} (inference frames only)")
    print(f"  Per-stream:       total={summary['per_stream_fps_avg']}, "
          f"infer={summary['per_stream_infer_fps_avg']} "
          f"(min={summary['per_stream_infer_fps_min']})")
    print(f"  Infer latency:    mean={summary['infer_mean_ms']}ms, "
          f"p95={summary['infer_p95_ms']}ms")
    print(f"  GPU util:         {summary['gpu_util_mean']}%")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="/data/traffic_720p.mp4")
    parser.add_argument("--frames", type=int, default=300)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    target_size = (int(cap.get(3)), int(cap.get(4)))
    cap.release()

    results = []

    # FULL (K=1) sweep
    for n in [1, 2, 3, 4, 5, 6, 7, 8]:
        s = run_benchmark(args.video, n, target_size, K=1, max_frames=args.frames)
        results.append(s)

    # LITE K=3 sweep
    for n in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]:
        s = run_benchmark(args.video, n, target_size, K=3, max_frames=args.frames)
        results.append(s)

    # Comparison table
    print(f"\n{'='*80}")
    print("FULL (K=1) vs LITE (K=3) COMPARISON")
    print(f"{'='*80}")
    print(f"{'K':>3} {'Strm':>5} {'TotalFPS':>9} {'InferFPS':>9} "
          f"{'Strm_FPS':>9} {'Strm_InfFPS':>12} {'InfP95':>7} {'GPU%':>5}")
    print("-" * 80)
    for s in results:
        print(f"{s['K']:>3} {s['n_streams']:>5} {s['total_fps']:>9.1f} "
              f"{s['infer_total_fps']:>9.1f} {s['per_stream_fps_avg']:>9.1f} "
              f"{s['per_stream_infer_fps_avg']:>12.1f} "
              f"{s['infer_p95_ms']:>7.1f} {s['gpu_util_mean']:>5.1f}")

    with open("/data/lite_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to /data/lite_comparison.json")


if __name__ == "__main__":
    main()
