#!/bin/bash
export LD_LIBRARY_PATH=/home/user/gpu-bench-nano3/libcusparse_lt-linux-aarch64-0.6.3.2-archive/lib
export PYTHONPATH=/home/user/gpu-bench-nano3/lib

python3 << 'PYEOF'
import sys, time
import numpy as np
print("=== Nano3 Verification ===")
print(f"Python: {sys.version}")

# 1. torch + CUDA
import torch
print(f"\n[1] torch: {torch.__version__}")
print(f"    CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("    ERROR: CUDA not available!")
    sys.exit(1)
print(f"    Device: {torch.cuda.get_device_name(0)}")
print(f"    Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
a = torch.randn(1000, 1000, device="cuda")
b = torch.randn(1000, 1000, device="cuda")
c = torch.mm(a, b)
print(f"    GPU matmul: OK ({c.shape})")

# 2. OpenCV
import cv2
print(f"\n[2] cv2: {cv2.__version__}")
img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
resized = cv2.resize(img, (640, 640))
print(f"    resize 1280x720 -> {resized.shape}: OK")

# 3. YOLO single inference
from ultralytics import YOLO
print("\n[3] YOLO loading yolov8n.pt...")
model = YOLO("yolov8n.pt")
model.to("cuda")
print("    Model loaded on CUDA")
dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
model(dummy, imgsz=640, verbose=False)
print("    Warmup: OK")
times = []
for i in range(10):
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    t0 = time.time()
    r = model(frame, imgsz=640, verbose=False)
    t1 = time.time()
    times.append((t1 - t0) * 1000)
avg = sum(times) / len(times)
print(f"    10x inference avg: {avg:.1f} ms/frame")
print(f"    Detections last frame: {len(r[0].boxes)}")

# 4. batch inference
frames = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(4)]
t0 = time.time()
results = model(frames, imgsz=640, verbose=False)
t1 = time.time()
print(f"\n[4] Batch(4): {(t1-t0)*1000:.1f} ms, {len(results)} results")

print("\n=== Nano3 ALL PASSED ===")
PYEOF
