"""Per-stream video processing worker."""

import threading
import time
import cv2
import numpy as np

try:
    import urllib.request
    import json as _json
    HAS_HTTP = True
except ImportError:
    HAS_HTTP = False

SKIP_SIZE = (426, 240)


class StreamWorker:
    """
    One thread per video stream. Decodes frames, applies the current mode
    (FULL / LITE / SKIP), and submits to the inference engine as needed.

    Mode is controlled externally via set_mode().
    Can be started and stopped dynamically (add_stream / remove_stream).
    """

    def __init__(self, stream_id, video_path, target_size, engine,
                 lite_engine=None, initial_mode="FULL",
                 offload_url=None):
        self.stream_id = stream_id
        self.video_path = video_path
        self.target_size = target_size
        self.engine = engine          # FULL engine (imgsz=640)
        self.lite_engine = lite_engine or engine  # LITE engine (imgsz=320)
        self.offload_url = offload_url  # e.g. "http://192.168.1.x:8765/infer"

        self._mode = initial_mode
        self._mode_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

        # Per-frame metrics buffer (consumed by ObservationCollector)
        self.metrics_buffer = []
        self._metrics_lock = threading.Lock()

        self.frame_idx = 0
        self.last_det = 0

    @property
    def mode(self):
        with self._mode_lock:
            return self._mode

    def set_mode(self, mode):
        with self._mode_lock:
            self._mode = mode

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def drain_metrics(self):
        """Return and clear buffered metrics since last drain."""
        with self._metrics_lock:
            out = list(self.metrics_buffer)
            self.metrics_buffer.clear()
        return out

    def _run(self):
        tw, th = self.target_size
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return

        while not self._stop.is_set():
            mode = self.mode
            t0 = time.time()

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            t_read = time.time()

            metric = {
                "stream_id": self.stream_id,
                "frame_idx": self.frame_idx,
                "mode": mode,
                "t_wall": 0,
                "t_read_ms": (t_read - t0) * 1000,
                "t_infer_ms": 0.0,
                "t_total_ms": 0.0,
                "batch_size": 0,
                "detections": 0,
            }

            if mode == "SKIP":
                _ = cv2.resize(frame, SKIP_SIZE)
                t_done = time.time()
                metric["t_total_ms"] = (t_done - t0) * 1000
                metric["t_wall"] = t_done

            elif mode == "OFFLOAD" and self.offload_url:
                # Send frame to remote Nano inference server
                frame_resized = cv2.resize(frame, (tw, th))
                _, jpg = cv2.imencode(".jpg", frame_resized,
                                      [cv2.IMWRITE_JPEG_QUALITY, 85])
                jpg_bytes = jpg.tobytes()
                t_send = time.time()
                try:
                    req = urllib.request.Request(
                        self.offload_url,
                        data=jpg_bytes,
                        headers={"Content-Type": "image/jpeg"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=2.0) as resp:
                        result_data = _json.loads(resp.read())
                    t_done = time.time()
                    self.last_det = result_data.get("n_det", 0)
                    metric["t_infer_ms"] = result_data.get("infer_ms", 0)
                    metric["t_network_ms"] = (t_done - t_send) * 1000
                    metric["batch_size"] = 1
                except Exception:
                    # Network failure: treat as SKIP
                    t_done = time.time()
                    metric["t_infer_ms"] = 0
                    metric["t_network_ms"] = (t_done - t_send) * 1000
                    metric["batch_size"] = 0
                metric["t_total_ms"] = (t_done - t0) * 1000
                metric["detections"] = self.last_det
                metric["t_wall"] = t_done

            else:
                # FULL (imgsz=640) or LITE (imgsz=320) - infer every frame
                frame_resized = cv2.resize(frame, (tw, th))
                infer_engine = self.lite_engine if mode == "LITE" else self.engine
                result = infer_engine.submit(
                    self.stream_id, self.frame_idx, frame_resized)
                t_done = time.time()
                self.last_det = result["n_det"]
                metric["t_infer_ms"] = result["infer_ms"]
                metric["batch_size"] = result["batch_size"]
                metric["t_total_ms"] = (t_done - t0) * 1000
                metric["detections"] = self.last_det
                metric["t_wall"] = t_done

            with self._metrics_lock:
                self.metrics_buffer.append(metric)

            self.frame_idx += 1

        cap.release()

    @property
    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()
