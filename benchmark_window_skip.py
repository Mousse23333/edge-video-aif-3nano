#!/usr/bin/env python3
"""
Window-skip LITE and batch size scaling benchmark.

Test 1: Window skip — instead of per-frame K, skip entire batch cycles.
  duty=1/3: infer 1 batch, skip 2 batches (equivalent to K=3 effect)
  duty=1/2: infer 1 batch, skip 1 batch (equivalent to K=2 effect)
  All inference batches maintain full batch_size = n_streams.

Test 2: Batch accumulation — each stream buffers M frames before submitting.
  batch_size = n_streams * M. Tests if bigger batches improve GPU efficiency.
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
        self.batch_log = []
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
            self.batch_log.append({"size": len(frames), "infer_ms": (t1-t0)*1000})
            for i, (sid, fidx, fr, rd, ev) in enumerate(batch):
                rd.update({"n_det": len(results[i].boxes),
                           "infer_ms": (t1-t0)*1000,
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


# ── Test 1: Window Skip ──────────────────────────────────────────────────────

def window_skip_worker(stream_id, video_path, target_size, engine,
                       infer_every, skip_count, max_frames,
                       metrics, gate, ready_event, start_event):
    """
    gate: shared threading.Event. When set, streams submit for inference.
    When cleared, streams do passthrough only.
    Controlled externally by the window scheduler.
    """
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
        t_read = time.time()
        frame_resized = cv2.resize(frame, (tw, th))
        t_resize = time.time()

        if gate.is_set():
            result = engine.submit(stream_id, fidx, frame_resized)
            t_done = time.time()
            local.append({
                "stream": stream_id, "frame": fidx, "is_infer": True,
                "t_total_ms": (t_done - t0) * 1000,
                "t_infer_ms": result["infer_ms"],
                "batch_size": result["batch_size"],
                "wall_ts": t_done,
            })
        else:
            t_done = time.time()
            local.append({
                "stream": stream_id, "frame": fidx, "is_infer": False,
                "t_total_ms": (t_done - t0) * 1000,
                "t_infer_ms": 0, "batch_size": 0,
                "wall_ts": t_done,
            })

    cap.release()
    metrics.extend(local)


def run_window_skip(video_path, n_streams, target_size, duty_ratio,
                    max_frames, window_ms=100):
    """
    duty_ratio: fraction of time doing inference (e.g., 0.33 = 1/3 duty)
    window_ms: base window duration in ms
    """
    infer_ms = int(window_ms * duty_ratio)
    skip_ms = window_ms - infer_ms

    print(f"\n  Window skip: duty={duty_ratio:.2f} "
          f"(infer {infer_ms}ms / skip {skip_ms}ms per {window_ms}ms window), "
          f"{n_streams} streams", end="", flush=True)

    engine = BatchEngine(n_expected=n_streams, max_wait_ms=8.0)
    gate = threading.Event()
    gate.set()  # start with inference enabled

    all_metrics = []
    ready_events = [threading.Event() for _ in range(n_streams)]
    start_event = threading.Event()
    threads = []

    for i in range(n_streams):
        t = threading.Thread(
            target=window_skip_worker,
            args=(i, video_path, target_size, engine, None, None,
                  max_frames, all_metrics, gate,
                  ready_events[i], start_event),
            daemon=True)
        threads.append(t)
        t.start()

    for e in ready_events:
        e.wait()

    # Window scheduler: toggle gate on/off
    stop_sched = threading.Event()
    def scheduler():
        while not stop_sched.is_set():
            gate.set()
            time.sleep(infer_ms / 1000.0)
            gate.clear()
            time.sleep(skip_ms / 1000.0)
    sched_thread = threading.Thread(target=scheduler, daemon=True)
    sched_thread.start()

    t_start = time.time()
    start_event.set()

    for t in threads:
        t.join()
    t_end = time.time()

    stop_sched.set()
    engine.stop()

    wall_time = t_end - t_start
    infer_frames = [m for m in all_metrics if m["is_infer"]]
    pass_frames = [m for m in all_metrics if not m["is_infer"]]

    total_fps = len(all_metrics) / wall_time
    infer_total_fps = len(infer_frames) / wall_time
    per_stream_infer_fps = infer_total_fps / n_streams

    batch_sizes = [m["batch_size"] for m in infer_frames]
    infer_latencies = [m["t_infer_ms"] for m in infer_frames]

    summary = {
        "mode": "window_skip",
        "duty_ratio": duty_ratio,
        "window_ms": window_ms,
        "n_streams": n_streams,
        "total_fps": round(total_fps, 1),
        "infer_total_fps": round(infer_total_fps, 1),
        "per_stream_infer_fps": round(per_stream_infer_fps, 1),
        "avg_batch_size": round(np.mean(batch_sizes), 2) if batch_sizes else 0,
        "infer_p95_ms": round(np.percentile(infer_latencies, 95), 1) if infer_latencies else 0,
        "infer_ratio": round(len(infer_frames) / len(all_metrics), 3) if all_metrics else 0,
        "batch_count": len(engine.batch_log),
    }

    print(f"  → infer_fps={summary['infer_total_fps']}, "
          f"strm={summary['per_stream_infer_fps']}, "
          f"batch={summary['avg_batch_size']:.1f}, "
          f"p95={summary['infer_p95_ms']}ms, "
          f"infer_ratio={summary['infer_ratio']:.2f}")

    return summary


# ── Test 2: Batch Accumulation ───────────────────────────────────────────────

def accum_worker(stream_id, video_path, target_size, accum_q,
                 max_frames, metrics, ready_event, start_event):
    """Each stream decodes frames and puts them in a shared accumulation queue."""
    tw, th = target_size
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        ready_event.set()
        return

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
        frame_resized = cv2.resize(frame, (tw, th))
        t_prep = time.time()

        result_holder = {}
        done_event = threading.Event()
        accum_q.put((stream_id, fidx, frame_resized,
                     result_holder, done_event, t0))
        done_event.wait()
        t_done = time.time()

        metrics.append({
            "stream": stream_id, "frame": fidx,
            "t_prep_ms": (t_prep - t0) * 1000,
            "t_total_ms": (t_done - t0) * 1000,
            "t_infer_ms": result_holder.get("infer_ms", 0),
            "batch_size": result_holder.get("batch_size", 0),
            "wall_ts": t_done,
        })

    cap.release()


def run_batch_accum(video_path, n_streams, target_size,
                    target_batch_size, max_frames):
    """
    Accumulate frames from multiple streams until target_batch_size is reached,
    then fire one big batch inference.
    """
    print(f"\n  Batch accum: batch={target_batch_size}, "
          f"{n_streams} streams", end="", flush=True)

    model = YOLO("yolov8n.pt")
    model.to("cuda")
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    for _ in range(10):
        model(dummy, verbose=False)

    accum_q = queue.Queue()
    stop_engine = threading.Event()
    batch_log = []

    def inference_loop():
        while not stop_engine.is_set():
            batch = []
            # Wait for first item
            try:
                item = accum_q.get(timeout=0.1)
                batch.append(item)
            except queue.Empty:
                continue

            # Collect up to target_batch_size with short timeout
            deadline = time.time() + 0.020  # 20ms max wait
            while len(batch) < target_batch_size:
                rem = deadline - time.time()
                if rem <= 0:
                    break
                try:
                    batch.append(accum_q.get(timeout=rem))
                except queue.Empty:
                    break

            frames = [b[2] for b in batch]
            t0 = time.time()
            results = model(frames, verbose=False)
            t1 = time.time()
            infer_ms = (t1 - t0) * 1000
            batch_log.append({"size": len(frames), "infer_ms": infer_ms})

            for i, (sid, fidx, fr, rh, ev, t_submit) in enumerate(batch):
                rh.update({"n_det": len(results[i].boxes),
                           "infer_ms": infer_ms,
                           "batch_size": len(frames)})
                ev.set()

    engine_thread = threading.Thread(target=inference_loop, daemon=True)
    engine_thread.start()

    all_metrics = []
    ready_events = [threading.Event() for _ in range(n_streams)]
    start_event = threading.Event()
    threads = []

    for i in range(n_streams):
        local_metrics = []
        all_metrics.append(local_metrics)
        t = threading.Thread(
            target=accum_worker,
            args=(i, video_path, target_size, accum_q,
                  max_frames, local_metrics,
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

    stop_engine.set()
    engine_thread.join(timeout=5)

    wall_time = t_end - t_start
    flat_metrics = [m for sublist in all_metrics for m in sublist]

    total_fps = len(flat_metrics) / wall_time
    infer_latencies = [m["t_infer_ms"] for m in flat_metrics]
    batch_sizes = [b["size"] for b in batch_log]

    # Per-stream FPS
    stream_fps = []
    for sid in range(n_streams):
        sm = sorted([m for sl in all_metrics for m in sl if m["stream"] == sid],
                    key=lambda x: x["wall_ts"])
        if len(sm) >= 2:
            dur = sm[-1]["wall_ts"] - sm[0]["wall_ts"]
            stream_fps.append(len(sm) / dur if dur > 0 else 0)

    summary = {
        "mode": "batch_accum",
        "target_batch": target_batch_size,
        "n_streams": n_streams,
        "total_fps": round(total_fps, 1),
        "per_stream_fps": round(np.mean(stream_fps), 1) if stream_fps else 0,
        "avg_batch_size": round(np.mean(batch_sizes), 1) if batch_sizes else 0,
        "infer_mean_ms": round(np.mean(infer_latencies), 1),
        "infer_p95_ms": round(np.percentile(infer_latencies, 95), 1),
        "total_p95_ms": round(np.percentile([m["t_total_ms"] for m in flat_metrics], 95), 1),
        "batch_count": len(batch_log),
    }

    print(f"  → total_fps={summary['total_fps']}, "
          f"strm={summary['per_stream_fps']}, "
          f"batch={summary['avg_batch_size']:.0f}, "
          f"infer={summary['infer_mean_ms']}ms, "
          f"total_p95={summary['total_p95_ms']}ms")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="/data/traffic_720p.mp4")
    parser.add_argument("--frames", type=int, default=300)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    target_size = (int(cap.get(3)), int(cap.get(4)))
    cap.release()

    n_streams = 6
    results = []

    # ── Test 1: Window Skip ──────────────────────────────────────────────
    print("=" * 60)
    print("TEST 1: WINDOW SKIP (6 streams)")
    print("=" * 60)

    # FULL baseline
    r = run_window_skip(args.video, n_streams, target_size,
                        duty_ratio=1.0, max_frames=args.frames)
    results.append(r)

    # Window skip at various duty ratios
    for duty in [0.5, 0.33, 0.25]:
        r = run_window_skip(args.video, n_streams, target_size,
                            duty_ratio=duty, max_frames=args.frames)
        results.append(r)

    # Also test with different stream counts at duty=0.33
    print("\n" + "=" * 60)
    print("TEST 1b: WINDOW SKIP duty=0.33 SWEEP")
    print("=" * 60)
    for n in [4, 6, 8, 10, 12]:
        r = run_window_skip(args.video, n, target_size,
                            duty_ratio=0.33, max_frames=args.frames)
        results.append(r)

    # ── Test 2: Batch Accumulation ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 2: BATCH ACCUMULATION (6 streams)")
    print("=" * 60)

    accum_results = []
    for batch_target in [6, 12, 18, 24]:
        r = run_batch_accum(args.video, n_streams, target_size,
                            target_batch_size=batch_target,
                            max_frames=args.frames)
        accum_results.append(r)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("WINDOW SKIP SUMMARY")
    print(f"{'='*72}")
    print(f"{'Duty':>5} {'Strm':>5} {'InfFPS':>7} {'StrmInf':>8} "
          f"{'Batch':>6} {'P95':>6} {'InfRatio':>9}")
    print("-" * 52)
    ws_results = [r for r in results if r["mode"] == "window_skip"]
    for r in ws_results:
        print(f"{r['duty_ratio']:>5.2f} {r['n_streams']:>5} "
              f"{r['infer_total_fps']:>7.1f} {r['per_stream_infer_fps']:>8.1f} "
              f"{r['avg_batch_size']:>6.1f} {r['infer_p95_ms']:>6.1f} "
              f"{r['infer_ratio']:>9.2f}")

    print(f"\n{'='*72}")
    print("BATCH ACCUMULATION SUMMARY")
    print(f"{'='*72}")
    print(f"{'Target':>7} {'TotalFPS':>9} {'StrmFPS':>8} {'AvgBatch':>9} "
          f"{'Infer_ms':>9} {'Total_P95':>10}")
    print("-" * 56)
    for r in accum_results:
        print(f"{r['target_batch']:>7} {r['total_fps']:>9.1f} "
              f"{r['per_stream_fps']:>8.1f} {r['avg_batch_size']:>9.0f} "
              f"{r['infer_mean_ms']:>9.1f} {r['total_p95_ms']:>10.1f}")

    all_data = {"window_skip": ws_results, "batch_accum": accum_results}
    with open("/data/window_skip_results.json", "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nSaved to /data/window_skip_results.json")


if __name__ == "__main__":
    main()
