#!/usr/bin/env python3
"""
Diagnose WHY LITE K=3 has lower throughput than FULL K=1.

Hypothesis: passthrough frames break the natural thread synchronization
that FULL mode gets for free from blocking on inference.

This test measures:
1. GPU idle gap between consecutive batch inferences
2. Per-stream CPU time during passthrough vs blocked-on-inference
3. Thread desynchronization: arrival spread when submitting to batch
"""

import time
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


class DiagnosticBatchEngine:
    """Batch engine that also records timing of batch lifecycle."""

    def __init__(self, n_expected, max_wait_ms=8.0):
        self.n_expected = n_expected
        self.max_wait = max_wait_ms / 1000.0
        self.frame_q = queue.Queue()
        self._stop = threading.Event()
        self.model = YOLO("yolov8n.pt")
        self.model.to("cuda")
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        for _ in range(10):
            self.model(dummy, verbose=False)

        # Diagnostic: record each batch's timing
        self.batch_log = []  # (t_first_arrive, t_batch_fire, t_infer_done, batch_size)
        self._batch_log_lock = threading.Lock()

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                item = self.frame_q.get(timeout=0.1)
            except queue.Empty:
                continue

            t_first = time.time()
            batch = [item]
            deadline = t_first + self.max_wait

            while len(batch) < self.n_expected:
                rem = deadline - time.time()
                if rem <= 0:
                    break
                try:
                    batch.append(self.frame_q.get(timeout=rem))
                except queue.Empty:
                    break

            t_fire = time.time()
            frames = [b[2] for b in batch]
            results = self.model(frames, verbose=False)
            t_done = time.time()

            with self._batch_log_lock:
                self.batch_log.append({
                    "t_first_arrive": t_first,
                    "t_batch_fire": t_fire,
                    "t_infer_done": t_done,
                    "batch_size": len(batch),
                    "collect_ms": (t_fire - t_first) * 1000,
                    "infer_ms": (t_done - t_fire) * 1000,
                })

            for i, (sid, fidx, fr, rd, ev) in enumerate(batch):
                rd.update({"n_det": len(results[i].boxes),
                           "infer_ms": (t_done - t_fire) * 1000,
                           "batch_size": len(batch), "t_done": t_done})
                ev.set()

    def submit(self, sid, fidx, frame):
        rd = {}
        ev = threading.Event()
        t_submit = time.time()
        self.frame_q.put((sid, fidx, frame, rd, ev))
        ev.wait()
        rd["t_submit"] = t_submit
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

    local = []
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
        t_decode = time.time()

        frame_resized = cv2.resize(frame, (tw, th))
        t_resize = time.time()

        is_infer = (fidx % K == 0)
        if is_infer:
            result = engine.submit(stream_id, fidx, frame_resized)
            t_done = time.time()
            local.append({
                "stream": stream_id, "frame": fidx, "is_infer": True,
                "t_decode_ms": (t_decode - t0) * 1000,
                "t_resize_ms": (t_resize - t_decode) * 1000,
                "t_blocked_ms": (t_done - t_resize) * 1000,
                "t_total_ms": (t_done - t0) * 1000,
                "wall_ts": t_done,
                "t_submit": result.get("t_submit", 0),
            })
        else:
            t_done = time.time()
            local.append({
                "stream": stream_id, "frame": fidx, "is_infer": False,
                "t_decode_ms": (t_decode - t0) * 1000,
                "t_resize_ms": (t_resize - t_decode) * 1000,
                "t_blocked_ms": 0,
                "t_total_ms": (t_done - t0) * 1000,
                "wall_ts": t_done,
                "t_submit": 0,
            })

    cap.release()
    metrics.extend(local)


def run_diagnostic(video_path, n_streams, target_size, K, max_frames):
    label = f"K={K}"
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC: {label}, {n_streams} streams, {max_frames} frames/stream")
    print(f"{'='*60}")

    engine = DiagnosticBatchEngine(n_expected=n_streams, max_wait_ms=8.0)

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

    # Sample GPU util during run
    gpu_samples = []
    stop_gpu = threading.Event()
    def gpu_sampler():
        while not stop_gpu.is_set():
            gpu_samples.append(get_gpu_util())
            time.sleep(0.05)  # 20Hz sampling
    gpu_thread = threading.Thread(target=gpu_sampler, daemon=True)
    gpu_thread.start()

    t_start = time.time()
    start_event.set()

    for t in threads:
        t.join()
    t_end = time.time()

    stop_gpu.set()
    gpu_thread.join()
    engine.stop()

    wall_time = t_end - t_start

    # ── Analysis ──────────────────────────────────────────────────────────

    # 1. Batch engine analysis
    blog = engine.batch_log
    if len(blog) >= 2:
        gaps = []
        for i in range(1, len(blog)):
            gap = (blog[i]["t_first_arrive"] - blog[i-1]["t_infer_done"]) * 1000
            gaps.append(gap)

        collect_times = [b["collect_ms"] for b in blog]
        infer_times = [b["infer_ms"] for b in blog]
        batch_sizes = [b["batch_size"] for b in blog]

        print(f"\n  Batch Engine ({len(blog)} batches):")
        print(f"    Avg batch size:    {np.mean(batch_sizes):.2f}")
        print(f"    Collect time:      mean={np.mean(collect_times):.1f}ms, "
              f"p95={np.percentile(collect_times, 95):.1f}ms")
        print(f"    Inference time:    mean={np.mean(infer_times):.1f}ms, "
              f"p95={np.percentile(infer_times, 95):.1f}ms")
        print(f"    GPU idle gap:      mean={np.mean(gaps):.1f}ms, "
              f"p50={np.percentile(gaps, 50):.1f}ms, "
              f"p95={np.percentile(gaps, 95):.1f}ms, "
              f"max={np.max(gaps):.1f}ms")
        total_infer = sum(infer_times)
        total_gap = sum(gaps)
        total_collect = sum(collect_times)
        print(f"    Time budget:       infer={total_infer/1000:.1f}s, "
              f"gap={total_gap/1000:.1f}s, "
              f"collect={total_collect/1000:.1f}s, "
              f"wall={wall_time:.1f}s")
        print(f"    GPU busy ratio:    {total_infer/(total_infer+total_gap+total_collect)*100:.1f}%")

    # 2. Per-stream thread analysis
    infer_frames = [m for m in all_metrics if m["is_infer"]]
    pass_frames = [m for m in all_metrics if not m["is_infer"]]

    if infer_frames:
        print(f"\n  Inference frames ({len(infer_frames)}):")
        print(f"    Decode:  mean={np.mean([f['t_decode_ms'] for f in infer_frames]):.1f}ms")
        print(f"    Resize:  mean={np.mean([f['t_resize_ms'] for f in infer_frames]):.1f}ms")
        print(f"    Blocked: mean={np.mean([f['t_blocked_ms'] for f in infer_frames]):.1f}ms")
        print(f"    Total:   mean={np.mean([f['t_total_ms'] for f in infer_frames]):.1f}ms")

    if pass_frames:
        print(f"\n  Passthrough frames ({len(pass_frames)}):")
        print(f"    Decode:  mean={np.mean([f['t_decode_ms'] for f in pass_frames]):.1f}ms")
        print(f"    Resize:  mean={np.mean([f['t_resize_ms'] for f in pass_frames]):.1f}ms")
        print(f"    Total:   mean={np.mean([f['t_total_ms'] for f in pass_frames]):.1f}ms, "
              f"p95={np.percentile([f['t_total_ms'] for f in pass_frames], 95):.1f}ms")

    # 3. Thread synchronization analysis
    # Group inference submissions by which batch they ended up in
    if infer_frames and len(blog) > 5:
        # Look at submission time spread within batches
        # For each batch, find the spread of t_submit times
        print(f"\n  Submission spread (how synchronized are threads?):")
        # Use batch log timestamps to match submissions
        spreads = []
        for b in blog:
            t_start_b = b["t_first_arrive"]
            t_end_b = b["t_batch_fire"]
            batch_submits = [
                f["t_submit"] for f in infer_frames
                if f["t_submit"] > 0
                and t_start_b - 0.001 <= f["t_submit"] <= t_end_b + 0.001
            ]
            if len(batch_submits) >= 2:
                spread = (max(batch_submits) - min(batch_submits)) * 1000
                spreads.append(spread)
        if spreads:
            print(f"    Arrival spread:    mean={np.mean(spreads):.1f}ms, "
                  f"p95={np.percentile(spreads, 95):.1f}ms, "
                  f"max={np.max(spreads):.1f}ms")

    # 4. GPU utilization
    valid_gpu = [g for g in gpu_samples if g >= 0]
    if valid_gpu:
        print(f"\n  GPU utilization:     mean={np.mean(valid_gpu):.1f}%, "
              f"p95={np.percentile(valid_gpu, 95):.1f}%")

    # 5. Total throughput
    total_infer_fps = len(infer_frames) / wall_time
    total_fps = len(all_metrics) / wall_time
    print(f"\n  Throughput:")
    print(f"    Total FPS:         {total_fps:.1f}")
    print(f"    Infer total FPS:   {total_infer_fps:.1f}")
    print(f"    Per-stream infer:  {total_infer_fps/n_streams:.1f}")


def main():
    video_path = "/data/traffic_720p.mp4"
    cap = cv2.VideoCapture(video_path)
    target_size = (int(cap.get(3)), int(cap.get(4)))
    cap.release()

    n_streams = 6
    max_frames = 300

    # Run FULL as reference
    run_diagnostic(video_path, n_streams, target_size, K=1, max_frames=max_frames)

    # Run LITE K=3
    run_diagnostic(video_path, n_streams, target_size, K=3, max_frames=max_frames)

    # Run LITE K=2 for comparison
    run_diagnostic(video_path, n_streams, target_size, K=2, max_frames=max_frames)


if __name__ == "__main__":
    main()
