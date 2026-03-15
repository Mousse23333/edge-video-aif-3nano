#!/usr/bin/env python3
"""
LITE K=3 batch tuning benchmark.

Tests two optimization axes:
  1. max_wait: how long batch engine waits before firing (8, 16, 24, 32 ms)
  2. sync mode: whether all LITE streams infer on the same frame indices

Also measures the cost of dynamically changing n_expected.
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


class TunableBatchEngine:
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
                  max_frames, metrics, sync_barrier,
                  ready_event, start_event):
    """
    sync_barrier: if not None, all streams wait at this barrier before
    each inference frame, ensuring synchronized batch submission.
    """
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

        is_infer = (fidx % K == 0)
        if is_infer:
            # Sync point: wait for other streams to reach their inference frame
            if sync_barrier is not None:
                sync_barrier.wait()

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


def run_test(video_path, n_streams, target_size, K, max_wait_ms,
             use_sync, max_frames):
    sync_label = "sync" if use_sync else "async"
    print(f"\n  K={K} streams={n_streams} max_wait={max_wait_ms}ms "
          f"mode={sync_label}", end="", flush=True)

    engine = TunableBatchEngine(n_expected=n_streams, max_wait_ms=max_wait_ms)

    sync_barrier = threading.Barrier(n_streams) if use_sync else None

    all_metrics = []
    ready_events = [threading.Event() for _ in range(n_streams)]
    start_event = threading.Event()
    threads = []

    for i in range(n_streams):
        t = threading.Thread(
            target=stream_worker,
            args=(i, video_path, target_size, engine, K,
                  max_frames, all_metrics, sync_barrier,
                  ready_events[i], start_event),
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

    wall_time = t_end - t_start
    infer_frames = [m for m in all_metrics if m["is_infer"]]
    infer_total_fps = len(infer_frames) / wall_time if wall_time > 0 else 0

    # Per-stream infer FPS
    stream_infer_fps = []
    for sid in range(n_streams):
        sm = sorted([m for m in all_metrics if m["stream"] == sid],
                    key=lambda x: x["wall_ts"])
        if len(sm) >= 2:
            dur = sm[-1]["wall_ts"] - sm[0]["wall_ts"]
            si = [m for m in sm if m["is_infer"]]
            stream_infer_fps.append(len(si) / dur if dur > 0 else 0)

    infer_latencies = [m["t_infer_ms"] for m in infer_frames]
    batch_sizes = [m["batch_size"] for m in infer_frames]

    summary = {
        "K": K,
        "n_streams": n_streams,
        "max_wait_ms": max_wait_ms,
        "sync": use_sync,
        "infer_total_fps": round(infer_total_fps, 1),
        "per_stream_infer_fps": round(np.mean(stream_infer_fps), 1),
        "infer_p95_ms": round(np.percentile(infer_latencies, 95), 1) if infer_latencies else 0,
        "avg_batch_size": round(np.mean(batch_sizes), 2) if batch_sizes else 0,
    }

    print(f"  → infer_fps={summary['infer_total_fps']}, "
          f"strm={summary['per_stream_infer_fps']}, "
          f"batch={summary['avg_batch_size']:.1f}, "
          f"p95={summary['infer_p95_ms']}ms")

    return summary


def test_batch_change_cost():
    """Measure the cost of dynamically changing n_expected."""
    print("\n" + "=" * 60)
    print("BATCH SIZE CHANGE COST TEST")
    print("=" * 60)

    engine = TunableBatchEngine(n_expected=4, max_wait_ms=8.0)
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Warmup at batch=4
    for _ in range(20):
        engine.submit(0, 0, dummy)

    # Measure normal inference
    times_before = []
    for _ in range(50):
        t0 = time.time()
        engine.submit(0, 0, dummy)
        times_before.append((time.time() - t0) * 1000)

    # Change batch size
    t_change = time.time()
    engine.n_expected = 8
    change_cost = (time.time() - t_change) * 1000

    # Measure first few inferences at new batch size
    times_after = []
    for _ in range(50):
        t0 = time.time()
        engine.submit(0, 0, dummy)
        times_after.append((time.time() - t0) * 1000)

    engine.stop()

    print(f"  Change n_expected 4→8: {change_cost:.3f}ms")
    print(f"  Before: mean={np.mean(times_before):.1f}ms, "
          f"p95={np.percentile(times_before, 95):.1f}ms")
    print(f"  After (first 5): {[f'{t:.1f}' for t in times_after[:5]]}ms")
    print(f"  After (mean): {np.mean(times_after):.1f}ms, "
          f"p95={np.percentile(times_after, 95):.1f}ms")

    return {
        "change_cost_ms": round(change_cost, 3),
        "before_mean_ms": round(np.mean(times_before), 1),
        "after_first5_ms": [round(t, 1) for t in times_after[:5]],
        "after_mean_ms": round(np.mean(times_after), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="/data/traffic_720p.mp4")
    parser.add_argument("--frames", type=int, default=300)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    target_size = (int(cap.get(3)), int(cap.get(4)))
    cap.release()

    # 1. Test batch change cost
    change_cost = test_batch_change_cost()

    # 2. LITE K=3 tuning: vary max_wait and sync mode
    print("\n" + "=" * 60)
    print("LITE K=3 TUNING (6 streams)")
    print("=" * 60)

    results = []
    n = 6  # 6 streams, beyond FULL's SLO boundary

    # Baseline: FULL K=1 at 6 streams
    r = run_test(args.video, n, target_size, K=1, max_wait_ms=8,
                 use_sync=False, max_frames=args.frames)
    results.append(r)

    # LITE K=3, vary max_wait, async
    for mw in [8, 16, 24, 32, 48]:
        r = run_test(args.video, n, target_size, K=3, max_wait_ms=mw,
                     use_sync=False, max_frames=args.frames)
        results.append(r)

    # LITE K=3, vary max_wait, sync
    for mw in [8, 16, 24, 32]:
        r = run_test(args.video, n, target_size, K=3, max_wait_ms=mw,
                     use_sync=True, max_frames=args.frames)
        results.append(r)

    # 3. Test across different stream counts with best settings
    print("\n" + "=" * 60)
    print("LITE K=3 SYNC SWEEP (best max_wait)")
    print("=" * 60)

    # Find best max_wait from sync results
    sync_results = [r for r in results if r["sync"] and r["K"] == 3]
    best_mw = max(sync_results, key=lambda x: x["per_stream_infer_fps"])["max_wait_ms"]
    print(f"Best max_wait for sync LITE: {best_mw}ms")

    sweep = []
    for n in [3, 4, 5, 6, 7, 8, 10]:
        r = run_test(args.video, n, target_size, K=3, max_wait_ms=best_mw,
                     use_sync=True, max_frames=args.frames)
        sweep.append(r)

    # Summary
    print(f"\n{'='*72}")
    print("SUMMARY: LITE K=3 TUNING")
    print(f"{'='*72}")
    print(f"{'K':>2} {'Strm':>5} {'Wait':>5} {'Sync':>5} "
          f"{'InfFPS':>7} {'StrmInf':>8} {'AvgBatch':>9} {'P95':>6}")
    print("-" * 72)
    for r in results:
        print(f"{r['K']:>2} {r['n_streams']:>5} {r['max_wait_ms']:>5} "
              f"{'Y' if r['sync'] else 'N':>5} "
              f"{r['infer_total_fps']:>7.1f} {r['per_stream_infer_fps']:>8.1f} "
              f"{r['avg_batch_size']:>9.1f} {r['infer_p95_ms']:>6.1f}")

    print(f"\n{'='*72}")
    print(f"SYNC SWEEP (K=3, max_wait={best_mw}ms)")
    print(f"{'='*72}")
    print(f"{'Strm':>5} {'InfFPS':>7} {'StrmInf':>8} {'AvgBatch':>9} "
          f"{'P95':>6} {'SLO≥10':>7}")
    print("-" * 50)
    for r in sweep:
        slo = "✅" if r["per_stream_infer_fps"] >= 10 else "❌"
        print(f"{r['n_streams']:>5} {r['infer_total_fps']:>7.1f} "
              f"{r['per_stream_infer_fps']:>8.1f} {r['avg_batch_size']:>9.1f} "
              f"{r['infer_p95_ms']:>6.1f} {slo:>7}")

    all_data = {
        "batch_change_cost": change_cost,
        "tuning_results": results,
        "sync_sweep": sweep,
    }
    with open("/data/lite_tuning.json", "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nSaved to /data/lite_tuning.json")


if __name__ == "__main__":
    main()
