#!/usr/bin/env python3
"""
Environment & Hardware Verification Script for Chess ML Pipeline.

Checks Python runtime, PyTorch CUDA 12.x capabilities, GPU VRAM,
and imports for all required computer vision and ML packages.
"""

from __future__ import annotations

import platform
import sys
from typing import NamedTuple


class CheckResult(NamedTuple):
    name: str
    status: bool
    details: str


def check_python_version() -> CheckResult:
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    passed = (v.major == 3) and (v.minor in (11, 12))
    details = f"Python {version_str} ({platform.platform()})"
    if not passed:
        details += " [WARNING: Recommended Python version is 3.11.x or 3.12.x]"
    return CheckResult("Python Version", passed, details)


def check_pytorch_cuda() -> list[CheckResult]:
    results: list[CheckResult] = []
    try:
        import torch  # type: ignore[import-not-found]

        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            results.append(
                CheckResult(
                    "PyTorch & CUDA Runtime",
                    True,
                    f"PyTorch {torch_version} | CUDA {cuda_version} | cuDNN {torch.backends.cudnn.version()}",
                )
            )
            results.append(
                CheckResult(
                    "GPU Accelerator",
                    True,
                    f"{device_name} (Devices: {device_count}, Total VRAM: {total_vram_gb:.2f} GB)",
                )
            )

            # Quick GPU Tensor compute test
            try:
                a = torch.randn(1000, 1000, device="cuda")
                b = torch.randn(1000, 1000, device="cuda")
                c = torch.matmul(a, b)
                torch.cuda.synchronize()
                results.append(
                    CheckResult(
                        "GPU Tensor Compute Test",
                        True,
                        f"CUDA Matrix Multiplication passed (Tensor shape: {list(c.shape)})",
                    )
                )
            except Exception as e:
                results.append(CheckResult("GPU Tensor Compute Test", False, f"Compute failed: {e}"))
        else:
            results.append(
                CheckResult(
                    "PyTorch (CPU Only)",
                    True,
                    f"PyTorch {torch_version} loaded without CUDA acceleration.",
                )
            )
            results.append(
                CheckResult(
                    "GPU Accelerator",
                    False,
                    "No CUDA-enabled GPU detected. Training/Inference will run in CPU fallback mode.",
                )
            )
    except ImportError as e:
        results.append(CheckResult("PyTorch", False, f"Failed to import torch: {e}"))

    return results


def check_required_packages() -> list[CheckResult]:
    packages = [
        ("ultralytics", "Ultralytics (YOLO26/YOLOv12 Framework)"),
        ("cv2", "OpenCV (Perspective & Image Geometry)"),
        ("chess", "python-chess (Rules & FEN Engine)"),
        ("albumentations", "Albumentations (Image Augmentations)"),
        ("onnxruntime", "ONNX Runtime (Accelerated Model Execution)"),
        ("pydantic", "Pydantic (Typed Data Contracts)"),
        ("numpy", "NumPy (Array Mathematics)"),
        ("PIL", "Pillow (Image IO)"),
        ("yaml", "PyYAML (Configuration Management)"),
    ]

    results: list[CheckResult] = []
    for module_name, label in packages:
        try:
            mod = __import__(module_name)
            ver = getattr(mod, "__version__", "loaded")
            results.append(CheckResult(label, True, f"v{ver}"))
        except ImportError as e:
            results.append(CheckResult(label, False, f"Import Error: {e}"))
    return results


def print_report(results: list[CheckResult]) -> bool:
    print("\n" + "=" * 78)
    print("♟️  CHESS ML ENVIRONMENT & HARDWARE DIAGNOSTICS REPORT")
    print("=" * 78)

    all_passed = True
    for item in results:
        status_icon = "✅ PASS" if item.status else "❌ FAIL"
        if not item.status and "GPU" in item.name and "CPU" in item.details:
            status_icon = "⚠️ WARN"
        if not item.status:
            all_passed = False
        print(f"[{status_icon}] {item.name:<32} : {item.details}")

    print("-" * 78)
    if all_passed:
        print("🎉 All environment dependencies and hardware checks passed successfully!")
    else:
        print("⚠️ One or more checks reported issues or missing dependencies.")
    print("=" * 78 + "\n")
    return all_passed


def main() -> int:
    results: list[CheckResult] = [check_python_version()]
    results.extend(check_pytorch_cuda())
    results.extend(check_required_packages())
    success = print_report(results)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
