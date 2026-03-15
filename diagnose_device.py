#!/usr/bin/env python3
"""Diagnose whether YOLO inference actually runs on CPU or GPU."""

import time
import numpy as np
import torch

print("=" * 60)
print("DEVICE DIAGNOSTIC")
print("=" * 60)

# 1. PyTorch CUDA availability
print(f"\ntorch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
    print(f"torch.cuda.get_device_properties(0): {torch.cuda.get_device_properties(0)}")

# 2. Check if torch was compiled with CUDA
print(f"\ntorch.version.cuda: {torch.version.cuda}")
print(f"torch.backends.cudnn.enabled: {torch.backends.cudnn.enabled}")
print(f"torch.backends.cudnn.version(): {torch.backends.cudnn.version()}")

# 3. Raw tensor operation benchmark: CPU vs CUDA
print("\n" + "=" * 60)
print("RAW TENSOR BENCHMARK (matmul 1024x1024, 50 iterations)")
print("=" * 60)

for device_name in ["cpu", "cuda"]:
    if device_name == "cuda" and not torch.cuda.is_available():
        print(f"  {device_name}: SKIPPED (not available)")
        continue

    a = torch.randn(1024, 1024, device=device_name)
    b = torch.randn(1024, 1024, device=device_name)

    # Warmup
    for _ in range(10):
        c = torch.matmul(a, b)
    if device_name == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    t0 = time.time()
    for _ in range(50):
        c = torch.matmul(a, b)
    if device_name == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    print(f"  {device_name}: {elapsed*1000:.1f}ms total, {elapsed/50*1000:.2f}ms per op")

# 4. YOLO inference on explicit CPU vs CUDA
print("\n" + "=" * 60)
print("YOLO INFERENCE: CPU vs CUDA (100 frames each)")
print("=" * 60)

from ultralytics import YOLO

dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

for device_name in ["cpu", "cuda"]:
    if device_name == "cuda" and not torch.cuda.is_available():
        continue

    model = YOLO("yolov8n.pt")
    model.to(device_name)

    actual_device = next(model.model.parameters()).device
    print(f"\n  [{device_name}] Model parameters on: {actual_device}")

    # Warmup
    for _ in range(10):
        model(dummy, verbose=False)
    if device_name == "cuda":
        torch.cuda.synchronize()

    # Benchmark with GPU activity monitoring
    gpu_load_path = "/sys/devices/platform/bus@0/17000000.gpu/load"
    def read_gpu():
        try:
            with open(gpu_load_path) as f:
                return int(f.read().strip()) / 10.0
        except:
            return -1

    gpu_before = read_gpu()

    t0 = time.time()
    for i in range(100):
        model(dummy, verbose=False)
    if device_name == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    gpu_after = read_gpu()

    print(f"  [{device_name}] 100 frames: {elapsed*1000:.0f}ms total, "
          f"{elapsed/100*1000:.1f}ms/frame, {100/elapsed:.1f} FPS")
    print(f"  [{device_name}] GPU load: before={gpu_before:.0f}%, after={gpu_after:.0f}%")

    # Check CUDA memory usage
    if device_name == "cuda":
        print(f"  [{device_name}] CUDA memory allocated: "
              f"{torch.cuda.memory_allocated()/1024/1024:.1f}MB")
        print(f"  [{device_name}] CUDA memory reserved: "
              f"{torch.cuda.memory_reserved()/1024/1024:.1f}MB")

    del model
    if device_name == "cuda":
        torch.cuda.empty_cache()

# 5. CUDA profiling - check if kernels actually launch
print("\n" + "=" * 60)
print("CUDA KERNEL ACTIVITY CHECK")
print("=" * 60)

if torch.cuda.is_available():
    model = YOLO("yolov8n.pt")

    # Test on CPU
    model.to("cpu")
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    model(dummy, verbose=False)
    mem_after = torch.cuda.memory_allocated()
    print(f"  CPU model: CUDA mem change = {(mem_after-mem_before)/1024:.1f}KB")

    # Test on CUDA
    model.to("cuda")
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    model(dummy, verbose=False)
    mem_after = torch.cuda.memory_allocated()
    print(f"  CUDA model: CUDA mem change = {(mem_after-mem_before)/1024:.1f}KB")
    print(f"  CUDA model: peak CUDA mem = {torch.cuda.max_memory_allocated()/1024/1024:.1f}MB")
