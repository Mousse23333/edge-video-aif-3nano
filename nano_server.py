"""Nano inference server: receives JPEG frames via HTTP, returns YOLO detections."""

import time
import numpy as np
import cv2
from http.server import HTTPServer, BaseHTTPRequestHandler
from ultralytics import YOLO

MODEL = None


class InferHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        jpg_bytes = self.rfile.read(length)

        # Decode JPEG
        arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error":"bad image"}')
            return

        # Inference
        t0 = time.time()
        results = MODEL(frame, imgsz=640, verbose=False)
        t1 = time.time()
        n_det = len(results[0].boxes)
        infer_ms = (t1 - t0) * 1000

        # Respond
        import json
        body = json.dumps({"n_det": n_det, "infer_ms": round(infer_ms, 1)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # Quiet logging: only print every 100th request
        pass


def main():
    global MODEL
    print("Loading YOLOv8n...")
    MODEL = YOLO("yolov8n.pt")
    MODEL.to("cuda")

    # Warmup
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    for _ in range(5):
        MODEL(dummy, imgsz=640, verbose=False)
    print("Warmup done.")

    server = HTTPServer(("0.0.0.0", 8765), InferHandler)
    print("Serving on 0.0.0.0:8765")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
