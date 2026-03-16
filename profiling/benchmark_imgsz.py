#!/usr/bin/env python3
"""
Test YOLO imgsz as the LITE knob.

Hypothesis: reducing imgsz (640→320) reduces per-inference GPU compute,
increasing total throughput ceiling beyond the ~55 FPS barrier.
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


class ImgszBatchEngine:
    def __init__(self, n_expected, imgsz=640, max_wait_ms=8.0):
        self.n_expected = max(1, n_expected)
        self.max_wait = max_wait_ms / 1000.0
        self.imgsz = imgsz
        self.frame_q = queue.Queue()
        self._stop = threading.Event()
        self.model = YOLO("yolov8n.pt")
        self.model.to("cuda")
        # Warmup at target imgsz
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        for _ in range(10):
            self.model(dummy, imgsz=imgsz, verbose=False)
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
            results = self.model(frames, imgsz=self.imgsz, verbose=False)
            t1 = time.time()
            infer_ms = (t1 - t0) * 1000
            for i, (sid, fidx, fr, rd, ev) in enumerate(batch):
                rd.update({"n_det": len(results[i].boxes),
                           "infer_ms": infer_ms,
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


def stream_worker(stream_id, video_path, target_size, engine,
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
        t_read = time.time()
        frame_resized = cv2.resize(frame, (tw, th))
        result = engine.submit(stream_id, fidx, frame_resized)
        t_done = time.time()
        local.append({
            "stream": stream_id, "frame": fidx,
            "t_read_ms": (t_read - t0) * 1000,
            "t_infer_ms": result["infer_ms"],
            "t_total_ms": (t_done - t0) * 1000,
            "batch_size": result["batch_size"],
            "detections": result["n_det"],
            "wall_ts": t_done,
        })
    cap.release()
    metrics.extend(local)


def run_test(video_path, n_streams, target_size, imgsz, max_frames):
    print(f"\n  imgsz={imgsz} streams={n_streams}", end="", flush=True)

    engine = ImgszBatchEngine(n_expected=n_streams, imgsz=imgsz)

    sys_samples = []
    stop_mon = threading.Event()
    def monitor():
        while not stop_mon.is_set():
            sys_samples.append(get_gpu_util())
            time.sleep(0.25)
    mon = threading.Thread(target=monitor, daemon=True)
    mon.start()

    all_metrics = []
    ready_events = [threading.Event() for _ in range(n_streams)]
    start_event = threading.Event()
    threads = []

    for i in range(n_streams):
        t = threading.Thread(
            target=stream_worker,
            args=(i, video_path, target_size, engine,
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
    mon.join()

    wall = t_end - t_start
    total_fps = len(all_metrics) / wall
    infer_times = [m["t_infer_ms"] for m in all_metrics]
    det_counts = [m["detections"] for m in all_metrics]
    batch_sizes = [m["batch_size"] for m in all_metrics]
    gpu_vals = [g for g in sys_samples if g >= 0]

    stream_fps = []
    for sid in range(n_streams):
        sm = sorted([m for m in all_metrics if m["stream"] == sid],
                    key=lambda x: x["wall_ts"])
        if len(sm) >= 2:
            dur = sm[-1]["wall_ts"] - sm[0]["wall_ts"]
            stream_fps.append(len(sm) / dur if dur > 0 else 0)

    s = {
        "imgsz": imgsz,
        "n_streams": n_streams,
        "total_fps": round(total_fps, 1),
        "per_stream_fps": round(np.mean(stream_fps), 1),
        "per_stream_fps_min": round(np.min(stream_fps), 1),
        "infer_mean_ms": round(np.mean(infer_times), 1),
        "infer_p95_ms": round(np.percentile(infer_times, 95), 1),
        "avg_batch": round(np.mean(batch_sizes), 1),
        "avg_detections": round(np.mean(det_counts), 1),
        "gpu_util_mean": round(np.mean(gpu_vals), 1) if gpu_vals else -1,
    }

    print(f"  → fps={s['total_fps']}, strm={s['per_stream_fps']}, "
          f"infer={s['infer_mean_ms']}ms, p95={s['infer_p95_ms']}ms, "
          f"det={s['avg_detections']:.1f}, GPU={s['gpu_util_mean']}%")
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="/data/traffic_720p.mp4")
    parser.add_argument("--frames", type=int, default=300)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    target_size = (int(cap.get(3)), int(cap.get(4)))
    cap.release()

    results = []

    # Single-stream: measure raw inference speed at different imgsz
    print("=" * 60)
    print("SINGLE STREAM: imgsz comparison")
    print("=" * 60)
    for imgsz in [640, 480, 416, 320, 256]:
        r = run_test(args.video, 1, target_size, imgsz, args.frames)
        results.append(r)

    # Multi-stream sweep at imgsz=640 (baseline) and imgsz=320 (LITE candidate)
    print("\n" + "=" * 60)
    print("MULTI-STREAM SWEEP: imgsz=640 vs 320")
    print("=" * 60)
    for imgsz in [640, 320]:
        for n in [1, 2, 3, 4, 5, 6, 7, 8, 10]:
            r = run_test(args.video, n, target_size, imgsz, args.frames)
            results.append(r)

    # Summary
    print(f"\n{'='*80}")
    print("SINGLE STREAM COMPARISON")
    print(f"{'='*80}")
    print(f"{'imgsz':>6} {'FPS':>6} {'Infer_ms':>9} {'P95_ms':>7} "
          f"{'Detections':>11} {'GPU%':>5}")
    print("-" * 50)
    single = [r for r in results if r["n_streams"] == 1]
    # deduplicate by imgsz
    seen = set()
    for r in single:
        if r["imgsz"] not in seen:
            seen.add(r["imgsz"])
            print(f"{r['imgsz']:>6} {r['per_stream_fps']:>6.1f} "
                  f"{r['infer_mean_ms']:>9.1f} {r['infer_p95_ms']:>7.1f} "
                  f"{r['avg_detections']:>11.1f} {r['gpu_util_mean']:>5.1f}")

    print(f"\n{'='*80}")
    print("MULTI-STREAM: imgsz=640 vs 320")
    print(f"{'='*80}")
    print(f"{'imgsz':>6} {'Strm':>5} {'TotalFPS':>9} {'StrmFPS':>8} "
          f"{'Infer_ms':>9} {'P95':>6} {'Det':>5} {'GPU%':>5} {'SLO':>4}")
    print("-" * 65)
    multi = [r for r in results
             if r["imgsz"] in (640, 320) and r not in single]
    for r in multi:
        slo = "✅" if r["per_stream_fps"] >= 10 else "❌"
        print(f"{r['imgsz']:>6} {r['n_streams']:>5} {r['total_fps']:>9.1f} "
              f"{r['per_stream_fps']:>8.1f} {r['infer_mean_ms']:>9.1f} "
              f"{r['infer_p95_ms']:>6.1f} {r['avg_detections']:>5.1f} "
              f"{r['gpu_util_mean']:>5.1f} {slo:>4}")

    with open("/data/imgsz_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to /data/imgsz_results.json")


if __name__ == "__main__":
    main()
