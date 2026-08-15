# 🧠 Model Card: Chess Piece & Board Localization Detectors

## 1. Model Overview
* **Model Families:** Ultralytics YOLO26n / YOLO26s (NMS-free), YOLOv12s (Attention-Centric), YOLO-Pose (Keypoint Base Localization).
* **Target Classes (13 Total):**
  1. `white-pawn` (`P`)
  2. `white-knight` (`N`)
  3. `white-bishop` (`B`)
  4. `white-rook` (`R`)
  5. `white-queen` (`Q`)
  6. `white-king` (`K`)
  7. `black-pawn` (`p`)
  8. `black-knight` (`n`)
  9. `black-bishop` (`b`)
  10. `black-rook` (`r`)
  11. `black-queen` (`q`)
  12. `black-king` (`k`)
  13. `board-corner` (Corner Keypoint)

---

## 2. Evaluation Metrics & Benchmark Targets

| Metric | Minimum Target | Optimal Target |
| :--- | :---: | :---: |
| **mAP@50 (All Classes)** | $\ge 0.95$ | $\ge 0.985$ |
| **mAP@50-95 (All Classes)**| $\ge 0.78$ | $\ge 0.860$ |
| **Square-Level FEN Accuracy**| $\ge 98.0\%$ | $\ge 99.5\%$ |
| **Full-Board FEN Accuracy** | $\ge 88.0\%$ | $\ge 96.0\%$ |
| **Inference Latency (GPU)**| $< 20\text{ ms}$ (50 FPS) | $< 10\text{ ms}$ (100 FPS) |

---

## 3. Quantization & Export Formats
* **FP16 ONNX**: Standard cross-platform deployment.
* **INT8 / FP16 TensorRT**: NVIDIA GPU edge/server deployment.
