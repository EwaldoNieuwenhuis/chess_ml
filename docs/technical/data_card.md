# 📊 Data Card: Chess Datasets & Provenance

## 1. Dataset Breakdown & Composition

The training and evaluation splits aggregate open-source benchmark datasets:

| Dataset | Modality | Sample Count | Primary Use |
| :--- | :--- | :--- | :--- |
| **ChessReD (Chess Recognition Dataset)** | 3D Smartphone photos | 10,800 images | Real-world perspective & lighting robustness |
| **Roboflow Universe Chess Collections** | 3D Webcams & Photos | ~6,500 images | Piece variation & corner detection |
| **Synthetic 2D Digital Generator** | 2D Screenshots | 25,000 images | Chess.com & Lichess theme coverage |

---

## 2. Augmentation Strategy (Albumentations Pipeline)

To ensure generalization across boards (wood, plastic, marble, digital themes), the training pipeline applies:
* **Spatial Transforms:** Perspective warp ($\pm 15^\circ$), random rotation ($\pm 10^\circ$), random scaling ($0.8\times - 1.2\times$).
* **Photometric Transforms:** Random glare/lighting shadows, Gaussian blur ($3\times3$ to $5\times5$), ColorJitter (brightness $\pm 0.2$, contrast $\pm 0.2$, saturation $\pm 0.3$), ISO noise.
