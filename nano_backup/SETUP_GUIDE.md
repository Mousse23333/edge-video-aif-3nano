# Nano Offload Node Setup Guide

Two Jetson Nano devices serve as offload inference nodes for the gpu-bench system.

## Network Topology

| Device | Hostname | IP | User | Role |
|--------|----------|----|------|------|
| Jetson (main) | jet1 | — | jet1 | Controller + main inference |
| Nano 2 | nano2 | 192.168.1.52 | user | Offload inference server |
| Nano 3 | nano3 | 192.168.1.53 | user | Offload inference server |

SSH from jet1: `ssh user@192.168.1.52` / `ssh user@192.168.1.53`
(jet1's `~/.ssh/id_rsa.pub` must be in each Nano's `~user/.ssh/authorized_keys`)

## System Info (both Nanos identical)

- JetPack: R36.4.7 (L4T)
- Python: 3.10.12
- PyTorch: 2.5.0a0+872d972e41.nv24.08 (CUDA 12.6)
- YOLO: ultralytics with yolov8n.pt

## Directory Structure (on each Nano)

```
~/gpu-bench-nano{2,3}/
├── nano_server.py          # Inference server script (the only custom code)
├── yolov8n.pt              # YOLOv8n model weights
├── lib/                    # Python packages (torch, ultralytics, cv2, numpy, etc.)
│   └── (97 packages installed via pip --target)
├── libcusparse_lt-.../     # libcusparseLt shared library (required by torch)
│   └── lib/
│       ├── libcusparseLt.so
│       ├── libcusparseLt.so.0
│       └── libcusparseLt.so.0.6.3.2
└── torch-2.5.0a0+...nv24.08...-aarch64.whl  # PyTorch wheel (keep for reinstall)
```

Note: nano2 also has a `Dockerfile`, `build.log`, and `venv/` from an earlier Docker attempt — these are unused.

## Setup Steps (from scratch)

### 1. Install system dependencies

```bash
sudo apt-get update
sudo apt-get install -y python3-pip libgl1 libglib2.0-0
```

### 2. Create working directory

```bash
mkdir -p ~/gpu-bench-nano2   # or nano3
cd ~/gpu-bench-nano2
```

### 3. Install PyTorch (Jetson-specific wheel)

Download the JetPack-compatible PyTorch wheel from NVIDIA:

```bash
# The wheel filename for JetPack R36 / CUDA 12.6:
# torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
pip3 install --target=./lib torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
```

### 4. Install libcusparseLt

```bash
# Download and extract
wget https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/libcusparse_lt-linux-aarch64-0.6.3.2-archive.tar.xz
tar xf libcusparse_lt-linux-aarch64-0.6.3.2-archive.tar.xz
```

### 5. Install Python packages

```bash
pip3 install --target=./lib ultralytics pyyaml opencv-python-headless "numpy==1.26.4"
```

### 6. Download YOLOv8n model

```bash
PYTHONPATH=./lib LD_LIBRARY_PATH=./libcusparse_lt-linux-aarch64-0.6.3.2-archive/lib \
  python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### 7. Copy nano_server.py

From jet1:
```bash
scp ~/gpu-bench/nano_backup/nano2/nano_server.py user@192.168.1.52:~/gpu-bench-nano2/
scp ~/gpu-bench/nano_backup/nano3/nano_server.py user@192.168.1.53:~/gpu-bench-nano3/
```

Or copy from this backup directory — the file is identical for both Nanos.

## Running the Inference Server

```bash
cd ~/gpu-bench-nano2  # or nano3

LD_LIBRARY_PATH=~/gpu-bench-nano2/libcusparse_lt-linux-aarch64-0.6.3.2-archive/lib \
PYTHONPATH=~/gpu-bench-nano2/lib \
  python3 nano_server.py
```

Output:
```
Loading YOLOv8n...
Warmup done.
Serving on 0.0.0.0:8765
```

### Run in background (persistent)

```bash
cd ~/gpu-bench-nano2

nohup bash -c 'LD_LIBRARY_PATH=~/gpu-bench-nano2/libcusparse_lt-linux-aarch64-0.6.3.2-archive/lib \
PYTHONPATH=~/gpu-bench-nano2/lib \
python3 nano_server.py' > server.log 2>&1 &
```

## Verify from jet1

```bash
# Health check (if implemented) or quick inference test:
curl -s http://192.168.1.52:8765/infer --data-binary @test.jpg -H "Content-Type: application/octet-stream"
curl -s http://192.168.1.53:8765/infer --data-binary @test.jpg -H "Content-Type: application/octet-stream"
```

Expected response: `{"n_det": <N>, "infer_ms": <T>}`

## SSH Key Setup (for new Nanos)

From jet1:
```bash
ssh-copy-id user@<nano-ip>
```

## Backup Contents

```
nano_backup/
├── SETUP_GUIDE.md          # This file
├── nano2/
│   ├── nano_server.py      # Inference server (identical to nano3)
│   ├── Dockerfile          # Unused Docker attempt
│   └── build.log           # Empty
└── nano3/
    └── nano_server.py      # Inference server (identical to nano2)
```

The large files (torch wheel, lib/, libcusparse, yolov8n.pt) are NOT backed up here — they can be reinstalled following steps 3-6 above.
