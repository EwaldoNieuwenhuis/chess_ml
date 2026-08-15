# ⚡ Model Export, Quantization & Latency Benchmarking

## 1. Exporting to ONNX & TensorRT

To achieve $<10\text{ ms}$ inference latency on GPU:

```bash
# Export trained model to ONNX with FP16 precision
yolo export \
  model=experiments/chess_yolo/yolo26s_baseline/weights/best.pt \
  format=onnx \
  half=True \
  dynamic=False \
  imgsz=640

# Export directly to TensorRT engine (NVIDIA GPUs)
yolo export \
  model=experiments/chess_yolo/yolo26s_baseline/weights/best.pt \
  format=engine \
  half=True \
  device=0
```

---

## 2. Latency Benchmarking Script
Run latency profiling across batch sizes:
```bash
python scripts/benchmark_inference.py \
  --weights experiments/chess_yolo/yolo26s_baseline/weights/best.onnx \
  --device cuda:0 \
  --iterations 500
```
