"""Batch inference engine shared across all streams."""

import threading
import queue
import time
import numpy as np
from ultralytics import YOLO


class BatchInferenceEngine:
    """
    Single YOLO model, single inference thread.
    Streams submit frames via submit(). The engine collects up to
    n_expected frames (with a short timeout) and fires one batch call.
    """

    def __init__(self, model_path="yolov8n.pt", device="cuda",
                 n_expected=4, max_wait_ms=8.0, imgsz=640):
        self.n_expected = max(1, n_expected)
        self.max_wait = max_wait_ms / 1000.0
        self.imgsz = imgsz
        self.frame_q = queue.Queue()
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self.model = YOLO(model_path)
        self.model.to(device)

        # Warmup
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        for _ in range(10):
            self.model(dummy, imgsz=self.imgsz, verbose=False)

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def update_expected(self, n):
        with self._lock:
            self.n_expected = max(1, n)

    def _loop(self):
        while not self._stop.is_set():
            try:
                item = self.frame_q.get(timeout=0.1)
            except queue.Empty:
                continue

            with self._lock:
                target = self.n_expected

            batch = [item]
            deadline = time.time() + self.max_wait
            while len(batch) < target:
                rem = deadline - time.time()
                if rem <= 0:
                    break
                try:
                    batch.append(self.frame_q.get(timeout=rem))
                except queue.Empty:
                    break

            frames = [b["frame"] for b in batch]
            t0 = time.time()
            results = self.model(frames, imgsz=self.imgsz, verbose=False)
            t1 = time.time()
            infer_ms = (t1 - t0) * 1000
            bs = len(frames)

            for i, item in enumerate(batch):
                n_det = len(results[i].boxes) if results else 0
                item["result"].update({
                    "n_det": n_det, "infer_ms": infer_ms,
                    "batch_size": bs, "t_infer_done": t1,
                })
                item["event"].set()

    def submit(self, stream_id, frame_idx, frame_array):
        """Submit a frame for inference. Blocks until result ready."""
        result = {}
        event = threading.Event()
        t_submit = time.time()
        self.frame_q.put({
            "stream_id": stream_id, "frame_idx": frame_idx,
            "frame": frame_array, "result": result, "event": event,
        })
        event.wait()
        result["t_submit"] = t_submit
        return result

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)
