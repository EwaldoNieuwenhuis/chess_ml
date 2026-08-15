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
Expected output:
```text
==============================================================================
♟️  CHESS ML ENVIRONMENT & HARDWARE DIAGNOSTICS REPORT
==============================================================================
[✅ PASS] Python Version                   : Python 3.11.x (Windows-10)
[✅ PASS] PyTorch & CUDA Runtime           : PyTorch 2.x | CUDA 12.x | cuDNN 8.x/9.x
[✅ PASS] GPU Accelerator                  : NVIDIA GeForce RTX ... (VRAM: ... GB)
[✅ PASS] GPU Tensor Compute Test          : CUDA Matrix Multiplication passed
[✅ PASS] Ultralytics (YOLO26/v12)         : loaded
[✅ PASS] OpenCV (Perspective & Geometry)  : loaded
[✅ PASS] python-chess (Rules & FEN Engine): loaded
[✅ PASS] Albumentations (Augmentations)   : loaded
[✅ PASS] ONNX Runtime (Model Execution)   : loaded
[✅ PASS] Pydantic (Typed Data Contracts)  : loaded
------------------------------------------------------------------------------
🎉 All environment dependencies and hardware checks passed successfully!
==============================================================================
```
