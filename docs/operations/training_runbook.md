# 🏋️ Training Runbook: Fine-Tuning YOLO for Chess Pieces

## 1. Directory Structure for Datasets
```text
data/
├── raw/                      # Downloaded source datasets (ChessReD, Roboflow)
├── processed/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
└── chess_data.yaml           # Dataset configuration file
```

---

## 2. Launching Training

Run the Ultralytics CLI or Python API to train the detector:

```bash
yolo detect train \
  data=data/chess_data.yaml \
  model=yolo26s.pt \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=8 \
  project=experiments/chess_yolo \
  name=yolo26s_baseline
```

---

## 3. Recommended Hyperparameters

| Parameter | Recommended Value | Rationale |
| :--- | :--- | :--- |
| `imgsz` | `640` or `800` | Higher resolution helps distinguish distant or pawn-sized pieces. |
| `batch` | `16` | Optimal gradient stability on 8GB+ VRAM GPUs. |
| `lr0` | `0.01` (SGD) / `0.001` (AdamW) | Initial learning rate. |
| `mosaic` | `0.5` | Reduced mosaic to prevent unnatural piece cuts. |
| `degrees` | `10.0` | Small angle rotation for board orientation shifts. |
| `perspective` | `0.0005` | Mild perspective warping to simulate camera tilts. |
