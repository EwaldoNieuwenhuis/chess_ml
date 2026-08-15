# 🛠️ Operational Guide: Environment Setup & GPU Configuration

This guide provides instructions for configuring your development environment using **`uv`** with full **NVIDIA CUDA 12.x** GPU acceleration.

---

## 1. Quick Setup via `uv` (Recommended)

### Step 1: Install `uv`
```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 2: Create Environment and Synchronize Dependencies
```bash
# Create Python 3.11 virtual environment
uv venv --python 3.11 .venv

# Activate environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install all dependencies including dev tools and PyTorch CUDA 12.x
uv sync --extra dev --extra gui
```

---

## 2. Verify GPU Acceleration & Package Health
Run the diagnostic script:
```bash
python scripts/verify_env.py
```
Expected output on NVIDIA RTX 50-series (Blackwell `sm_120`) or RTX 40/30-series:
```text
==============================================================================
♟️  CHESS ML ENVIRONMENT & HARDWARE DIAGNOSTICS REPORT
==============================================================================
[✅ PASS] Python Version                   : Python 3.11.15 (Windows-10)
[✅ PASS] PyTorch & CUDA Runtime           : PyTorch 2.11.0+cu128 | CUDA 12.8 | cuDNN 91900
[✅ PASS] GPU Accelerator                  : NVIDIA GeForce RTX 5060 Ti (Devices: 1, Total VRAM: 15.93 GB)
[✅ PASS] GPU Tensor Compute Test          : CUDA Matrix Multiplication passed (Tensor shape: [1000, 1000])
[✅ PASS] Ultralytics (YOLO26/YOLOv12)     : v8.4.120
[✅ PASS] OpenCV (Perspective & Geometry)  : v4.11.0
[✅ PASS] python-chess (Rules & FEN Engine): v1.11.2
[✅ PASS] Albumentations (Augmentations)   : v2.0.8
[✅ PASS] ONNX Runtime (Model Execution)   : v1.28.0
[✅ PASS] Pydantic (Typed Data Contracts)  : v2.13.4
[✅ PASS] NumPy (Array Mathematics)        : v2.4.6
[✅ PASS] Pillow (Image IO)                : v12.3.0
[✅ PASS] PyYAML (Configuration Management): v6.0.3
------------------------------------------------------------------------------
🎉 All environment dependencies and hardware checks passed successfully!
==============================================================================
```

---

## 3. Stockfish Binary Configuration
The engine manager automatically checks the following paths:
1. Custom environment variable: `STOCKFISH_PATH`
2. Local repository directory: `bin/stockfish.exe` (Windows) or `bin/stockfish` (Linux)
3. System `PATH` (`where stockfish` / `which stockfish`)

If no binary is present, `StockfishManager.download_binary()` can download the official Stockfish 16.1+ binary automatically into `bin/`. Offline CI environments will automatically fallback to the internal `python-chess` deterministic evaluator.
