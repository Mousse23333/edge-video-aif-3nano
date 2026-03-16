#!/usr/bin/env python3
"""
Mode switching cost benchmark.

Runs N streams. The target stream switches mode at a preset frame index.
Measures transient response on switched stream + bystander impact.
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
SKIP_SIZE = (426, 240)
LITE_K = 3


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


def system_monitor(samples, stop_event, interval=0.1):
    while not stop_event.is_set():
        samples.append({"t": time.time(), "gpu_util": get_gpu_util(),
                        "mem_used_mb": get_mem_used_mb()})
        time.sleep(interval)


class BatchEngine:
    def __init__(self, n_expected):
        self.n_expected = max(1, n_expected)
        self.frame_q = queue.Queue()
        self._stop = threading.Event()
        self.model = YOLO("yolov8n.pt")
        self.model.to("cuda")
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        for _ in range(10):
            self.model(dummy, verbose=False)
        self.max_wait = 0.008
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def update_expected(self, n):
        self.n_expected = max(1, n)

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
            for i, (sid, fidx, frame, rd, ev) in enumerate(batch):
                nd = len(results[i].boxes) if results else 0
                rd.update({"n_det": nd, "infer_ms": infer_ms,
                           "batch_size": len(frames), "t_done": t1})
                ev.set()

    def submit(self, stream_id, frame_idx, frame):
        rd = {}
        ev = threading.Event()
        self.frame_q.put((stream_id, frame_idx, frame, rd, ev))
        ev.wait()
        return rd

    def stop(self):
        self._stop.set()
        self._thread.join()


def stream_worker(stream_id, video_path, target_size, engine,
                  total_frames, metrics_list,
                  switch_at_frame, from_mode, to_mode,
                  is_switch_stream, ready_event, start_event):
    tw, th = target_size
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        ready_event.set()
        return

    metrics = []
    last_det = 0

    ready_event.set()
    start_event.wait()

    for frame_idx in range(total_frames):
        # Determine current mode
        if is_switch_stream and frame_idx >= switch_at_frame:
            mode = to_mode
        elif is_switch_stream:
            mode = from_mode
        else:
            mode = "FULL"  # bystanders always FULL

        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        t_read = time.time()

        if mode == "SKIP":
            _ = cv2.resize(frame, SKIP_SIZE)
            t_done = time.time()
            metrics.append({
                "stream": stream_id, "frame": frame_idx, "mode": mode,
                "t_read_ms": (t_read - t0) * 1000,
                "t_infer_ms": 0.0, "batch_size": 0,
                "t_total_ms": (t_done - t0) * 1000,
                "detections": 0, "wall_ts": t_done,
            })

        elif mode == "LITE" and frame_idx % LITE_K != 0:
            frame_resized = cv2.resize(frame, (tw, th))
            t_done = time.time()
            metrics.append({
                "stream": stream_id, "frame": frame_idx, "mode": mode,
                "t_read_ms": (t_read - t0) * 1000,
                "t_infer_ms": 0.0, "batch_size": 0,
                "t_total_ms": (t_done - t0) * 1000,
                "detections": last_det, "wall_ts": t_done,
            })

        else:  # FULL, or LITE inference frame
            frame_resized = cv2.resize(frame, (tw, th))
            result = engine.submit(stream_id, frame_idx, frame_resized)
            t_done = time.time()
            last_det = result["n_det"]
            metrics.append({
                "stream": stream_id, "frame": frame_idx, "mode": mode,
                "t_read_ms": (t_read - t0) * 1000,
                "t_infer_ms": result["infer_ms"],
                "batch_size": result["batch_size"],
                "t_total_ms": (t_done - t0) * 1000,
                "detections": last_det, "wall_ts": t_done,
            })

    cap.release()
    metrics_list.extend(metrics)


def analyze_switch(all_metrics, switch_stream, switch_at_frame, n_streams,
                   from_mode, to_mode):
    sw = sorted([m for m in all_metrics if m["stream"] == switch_stream],
                key=lambda x: x["frame"])
    pre  = [m for m in sw if m["frame"] < switch_at_frame]
    post = [m for m in sw if m["frame"] >= switch_at_frame]

    # Find switch wall time
    t_switch = post[0]["wall_ts"] if post else 0

    by_all = sorted([m for m in all_metrics if m["stream"] != switch_stream],
                    key=lambda x: x["wall_ts"])
    by_pre  = [m for m in by_all if m["wall_ts"] < t_switch]
    by_post = [m for m in by_all if m["wall_ts"] >= t_switch]

    def calc_stats(frames, label):
        if len(frames) < 2:
            return {"label": label, "count": len(frames),
                    "total_mean_ms": 0, "total_p95_ms": 0,
                    "infer_mean_ms": 0, "fps": 0}
        totals = [f["t_total_ms"] for f in frames]
        infers = [f["t_infer_ms"] for f in frames if f["t_infer_ms"] > 0]
        dur = frames[-1]["wall_ts"] - frames[0]["wall_ts"]
        return {
            "label": label,
            "count": len(frames),
            "total_mean_ms": round(np.mean(totals), 2),
            "total_p95_ms": round(np.percentile(totals, 95), 2),
            "infer_mean_ms": round(np.mean(infers), 2) if infers else 0,
            "fps": round(len(frames) / dur, 1) if dur > 0 else 0,
        }

    transient_window = 15  # frames to consider as transient
    transient = post[:transient_window]
    steady    = post[transient_window:]

    s_pre  = calc_stats(pre, f"pre({from_mode})")
    s_tran = calc_stats(transient, f"transient({to_mode})")
    s_sted = calc_stats(steady, f"steady({to_mode})")
    b_pre  = calc_stats(by_pre, "bystander_pre")
    b_post = calc_stats(by_post, "bystander_post")

    # Detect stabilization: first frame where total_ms is within 15% of steady mean
    stab_frame = -1
    if steady and s_sted["total_mean_ms"] > 0:
        target = s_sted["total_mean_ms"]
        for idx, m in enumerate(post):
            if abs(m["t_total_ms"] - target) / max(target, 1) < 0.15:
                stab_frame = idx
                break

    # Peak latency in transient
    peak_ms = max([m["t_total_ms"] for m in transient]) if transient else 0

    result = {
        "transition":         f"{from_mode} -> {to_mode}",
        "n_streams":          n_streams,
        "pre_fps":            s_pre["fps"],
        "pre_total_ms":       s_pre["total_mean_ms"],
        "transient_fps":      s_tran["fps"],
        "transient_total_ms": s_tran["total_mean_ms"],
        "transient_peak_ms":  round(peak_ms, 2),
        "steady_fps":         s_sted["fps"],
        "steady_total_ms":    s_sted["total_mean_ms"],
        "stabilization_frames": stab_frame,
        "bystander_pre_fps":  b_pre["fps"],
        "bystander_post_fps": b_post["fps"],
        "bystander_fps_delta": round(b_post["fps"] - b_pre["fps"], 1),
    }

    print(f"\n  Switched stream #{switch_stream}:")
    print(f"    Pre  ({from_mode}):    FPS={s_pre['fps']}, latency={s_pre['total_mean_ms']:.1f}ms")
    print(f"    Transient ({to_mode}): FPS={s_tran['fps']}, latency={s_tran['total_mean_ms']:.1f}ms, "
          f"peak={peak_ms:.1f}ms")
    print(f"    Steady ({to_mode}):    FPS={s_sted['fps']}, latency={s_sted['total_mean_ms']:.1f}ms")
    print(f"    Stabilization:         {stab_frame} frames")
    print(f"  Bystanders:")
    print(f"    Pre: {b_pre['fps']} FPS → Post: {b_post['fps']} FPS  "
          f"(delta={result['bystander_fps_delta']:+.1f})")

    return result


def run_switch_test(video_path, n_streams, target_size,
                    switch_stream, from_mode, to_mode,
                    warmup_frames, post_frames):
    total = warmup_frames + post_frames

    print(f"\n{'='*60}")
    print(f"Switch: {from_mode} -> {to_mode}  (stream {switch_stream}, "
          f"{n_streams} total, switch@frame {warmup_frames})")
    print(f"{'='*60}")

    # Count expected inference streams for batch engine
    n_infer = n_streams  # initially all FULL
    engine = BatchEngine(n_expected=n_infer)

    sys_samples = []
    stop_mon = threading.Event()
    mon_thread = threading.Thread(target=system_monitor, args=(sys_samples, stop_mon))
    mon_thread.start()

    all_metrics = []
    ready_events = [threading.Event() for _ in range(n_streams)]
    start_event = threading.Event()

    threads = []
    for i in range(n_streams):
        is_sw = (i == switch_stream)
        t = threading.Thread(
            target=stream_worker,
            args=(i, video_path, target_size, engine, total, all_metrics,
                  warmup_frames, from_mode, to_mode, is_sw,
                  ready_events[i], start_event),
            daemon=True,
        )
        threads.append(t)
        t.start()

    for e in ready_events:
        e.wait()
    start_event.set()

    for t in threads:
        t.join()

    engine.stop()
    stop_mon.set()
    mon_thread.join()

    result = analyze_switch(all_metrics, switch_stream, warmup_frames,
                           n_streams, from_mode, to_mode)

    # Save CSV
    prefix = f"/data/switch_{from_mode}_{to_mode}_{n_streams}s"
    with open(f"{prefix}_frames.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
        writer.writeheader()
        writer.writerows(sorted(all_metrics, key=lambda x: x["wall_ts"]))
    print(f"  Saved: {prefix}_frames.csv")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",   default="/data/traffic_720p.mp4")
    parser.add_argument("--streams", type=int, default=4)
    parser.add_argument("--warmup",  type=int, default=200)
    parser.add_argument("--post",    type=int, default=200)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    target_size = (int(cap.get(3)), int(cap.get(4)))
    cap.release()

    transitions = [
        ("FULL", "LITE"),
        ("LITE", "FULL"),
        ("FULL", "SKIP"),
        ("SKIP", "FULL"),
        ("LITE", "SKIP"),
        ("SKIP", "LITE"),
    ]

    all_results = []
    for from_m, to_m in transitions:
        r = run_switch_test(
            args.video, args.streams, target_size,
            switch_stream=0, from_mode=from_m, to_mode=to_m,
            warmup_frames=args.warmup, post_frames=args.post,
        )
        all_results.append(r)

    # Summary
    print(f"\n{'='*80}")
    print("SWITCH COST SUMMARY (4 streams, stream #0 switches)")
    print(f"{'='*80}")
    print(f"{'Transition':<16} {'Pre_FPS':>8} {'Trans_FPS':>10} {'Steady_FPS':>11} "
          f"{'Peak_ms':>8} {'Stab_frm':>9} {'By_delta':>9}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['transition']:<16} {r['pre_fps']:>8.1f} {r['transient_fps']:>10.1f} "
              f"{r['steady_fps']:>11.1f} {r['transient_peak_ms']:>8.1f} "
              f"{r['stabilization_frames']:>9} "
              f"{r['bystander_fps_delta']:>+9.1f}")

    with open("/data/switch_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to /data/switch_summary.json")


if __name__ == "__main__":
    main()
