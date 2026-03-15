#!/usr/bin/env python3
"""
Lightweight inference server for Nano offload nodes.

Receives frames via HTTP POST, runs YOLOv8n inference, returns detections.
Deploy this on nano2/nano3 and run with:
  PYTHONPATH=~/gpu-bench-nano2/lib python3 nano_inference_server.py --port 8765
"""

import argparse
import time
import io
import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

MODEL = None
IMGSZ = 640


def load_model(model_path="yolov8n.pt", imgsz=640, device="cuda"):
    global MODEL, IMGSZ
    from ultralytics import YOLO
    import torch

    if torch.cuda.is_available():
        dev = "cuda"
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = "cpu"
        logging.warning("CUDA not available, using CPU")

    MODEL = YOLO(model_path)
    MODEL.to(dev)
    IMGSZ = imgsz

    # Warmup
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(5):
        MODEL(dummy, imgsz=IMGSZ, verbose=False)
    logging.info(f"Model ready: {model_path}, imgsz={imgsz}, device={dev}")


class InferenceHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress default per-request HTTP logs (use explicit logging)

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/infer":
            self.send_response(404)
            self.end_headers()
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(length)

            # Decode frame: JPEG bytes
            t0 = time.time()
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Failed to decode frame")

            # Inference
            t_infer = time.time()
            results = MODEL(frame, imgsz=IMGSZ, verbose=False)
            t_done = time.time()

            boxes = []
            if results and len(results) > 0:
                for box in results[0].boxes:
                    boxes.append({
                        "cls": int(box.cls[0]),
                        "conf": float(box.conf[0]),
                        "xyxy": box.xyxy[0].tolist(),
                    })

            response = {
                "n_det": len(boxes),
                "infer_ms": (t_done - t_infer) * 1000,
                "total_ms": (t_done - t0) * 1000,
                "boxes": boxes,
            }

            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
            logging.info(f"infer: {len(boxes)} det, {(t_done-t_infer)*1000:.0f}ms")

        except Exception as e:
            logging.error(f"Inference error: {e}")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model", default="yolov8n.pt")
    args = parser.parse_args()

    load_model(args.model, args.imgsz)

    server = HTTPServer(("0.0.0.0", args.port), InferenceHandler)
    logging.info(f"Inference server listening on port {args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
