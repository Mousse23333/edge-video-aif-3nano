# Nano Device Setup Reference

## Device Inventory

| Property | Nano2 | Nano3 |
|---|---|---|
| Hostname | nano2 | nano3 |
| IP | 192.168.1.52 | 192.168.1.53 |
| User | user | user |
| GPU | Orin (Jetson Orin Nano) | Orin (Jetson Orin Nano) |
| JetPack | R36.4.7 | R36.4.7 |
| CUDA | 12.6 | 12.6 |
| torch | 2.5.0a0+872d972e41.nv24.08 | 2.5.0a0+872d972e41.nv24.08 |
| torchvision | 0.20.0 (SAM import disabled) | 0.20.0 (SAM import disabled) |
| ultralytics | latest (pip) | latest (pip) |
| Python | 3.10 | 3.10 |
| Single-frame infer | ~31ms (imgsz=640) | ~31ms (imgsz=640) |

## Orin (Controller Node)

| Property | Value |
|---|---|
| Hostname | ubuntu |
| IP | 192.168.1.x (same LAN) |
| User | jet1 |
| GPU | Orin (2048 CUDA cores, 7.6GB) |
| JetPack | R36.5.0 |
| CUDA | 12.6 |
| Project dir | /home/jet1/gpu-bench/ |

## Network Topology

```
Laptop → school network → dragonfire (router) → 192.168.1.x LAN
                                                   ├── Jetson Orin (jet1@ubuntu)
                                                   ├── Nano2 (user@192.168.1.52)
                                                   └── Nano3 (user@192.168.1.53)
```

## SSH Access

- **Orin → Nano**: SSH key-based auth (jet1's pubkey added to both Nanos' authorized_keys)
  ```bash
  ssh user@192.168.1.52   # nano2
  ssh user@192.168.1.53   # nano3
  ```
- **Laptop → Nano**: Via ProxyJump through dragonfire, key-based auth
- **No password access** to Nanos (key-only)

## File Layout on Nanos

Both Nanos use the same structure (substitute nano2/nano3):

```
~/gpu-bench-nano{2,3}/
├── lib/                          # Python packages (pip --target)
│   ├── torch/                    # NVIDIA CUDA torch 2.5.0a0
│   ├── torchvision/              # 0.20.0 (SAM import patched)
│   ├── ultralytics/              # YOLO framework
│   ├── cv2/                      # opencv-python-headless
│   └── ...
├── libcusparse_lt-linux-aarch64-0.6.3.2-archive/
│   └── lib/                      # libcusparseLt.so.0
├── nano_server.py                # HTTP inference server
└── yolov8n.pt                    # YOLO model (auto-downloaded)
```

## Environment Variables (required before any Python command)

nano2:
```bash
export LD_LIBRARY_PATH=/home/user/gpu-bench-nano2/libcusparse_lt-linux-aarch64-0.6.3.2-archive/lib
export PYTHONPATH=/home/user/gpu-bench-nano2/lib
```

nano3:
```bash
export LD_LIBRARY_PATH=/home/user/gpu-bench-nano3/libcusparse_lt-linux-aarch64-0.6.3.2-archive/lib
export PYTHONPATH=/home/user/gpu-bench-nano3/lib
```

## HTTP Inference Server (nano_server.py)

- **Port**: 8765
- **Endpoint**: POST /infer
- **Input**: JPEG image bytes (Content-Type: image/jpeg)
- **Output**: JSON `{"n_det": int, "infer_ms": float}`
- **Model**: YOLOv8n, imgsz=640, CUDA

Start command (example for nano2):
```bash
export LD_LIBRARY_PATH=/home/user/gpu-bench-nano2/libcusparse_lt-linux-aarch64-0.6.3.2-archive/lib
export PYTHONPATH=/home/user/gpu-bench-nano2/lib
cd ~/gpu-bench-nano2 && python3 nano_server.py
```

Ready when output shows: `Serving on 0.0.0.0:8765`

## OFFLOAD Integration with Orin

In `config/action_space.yaml`:
```yaml
OFFLOAD:
  urls:
    - "http://192.168.1.52:8765/infer"   # nano2
    - "http://192.168.1.53:8765/infer"   # nano3
  imgsz: 640
```

StreamManager assigns offload URLs round-robin to streams in OFFLOAD mode.
StreamWorker sends JPEG-encoded frames via HTTP POST, receives detection count + inference time.

## Verification Status (2026-03-15)

- [x] Nano2: torch CUDA ✅, YOLO inference ✅, HTTP server ✅
- [x] Nano3: torch CUDA ✅, YOLO inference ✅, HTTP server ✅
- [x] Orin → Nano2 HTTP: connected ✅
- [x] Orin → Nano3 HTTP: connected ✅
- [x] Orin → Nano2 SSH: key auth ✅
- [x] Orin → Nano3 SSH: key auth ✅

## Known Issues / Workarounds

1. **torchvision incompatibility**: PyPI torchvision 0.20.0 C extensions incompatible with NVIDIA custom torch. Fix: disabled SAM import in ultralytics (`sed -i 's/from .sam import SAM/# from .sam import SAM/' lib/ultralytics/models/__init__.py`). Does not affect YOLO.

2. **No sudo on Nanos**: All packages installed with `pip3 install --target=` to local directory. No system-level changes.

3. **libcusparseLt not in JetPack**: Standalone library downloaded from NVIDIA CDN, loaded via LD_LIBRARY_PATH.

4. **Cleanup on return**: `rm -rf ~/gpu-bench-nano{2,3}` removes everything. Also remove `~/.config/Ultralytics/` (auto-created settings).
