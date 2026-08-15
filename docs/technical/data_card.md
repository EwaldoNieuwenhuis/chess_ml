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

---

## 3. Canonical Label Schema & Coordinate Standards (ADR-008)

All datasets are normalized to standard YOLO format:
* **Classes (12):**
  * White: `0: white_pawn`, `1: white_knight`, `2: white_bishop`, `3: white_rook`, `4: white_queen`, `5: white_king`
  * Black: `6: black_pawn`, `7: black_knight`, `8: black_bishop`, `9: black_rook`, `10: black_queen`, `11: black_king`
* **Format:** `class_id x_center y_center width height` (normalized $0.0 \le x, y, w, h \le 1.0$).
* **Validation & Clamping:** Epsilon boundary clamping ($[-10^{-5}, 1.0 + 10^{-5}] \to [0.0, 1.0]$) and degenerate box filtering ($w, h \ge 0.005$).
* **Negative Samples:** Empty background images and empty boards are paired with **0-byte `.txt` label files** in accordance with the official Ultralytics dataset standard, suppressing false-positive piece detections without anchor clutter.
* **Footprint Anchoring:** Standardizes annotations on full visible piece bounding boxes, while downstream homography mapping evaluates the bottom-center base contact $(x_c, y_c + h/2)$ to eliminate perspective parallax misassignment on tall pieces.

---

## 4. Perspective Parallax & Contact Footprint Verification (US-2.3.4)

Empirical validation performed by `scripts/verify_contact_anchors.py` across camera elevation sweeps ($30^\circ\text{--}75^\circ$) confirms that the base contact anchor is required to avoid perspective misassignment:

| Anchor Mapping Strategy | All Pieces Accuracy | Tall Pieces Accuracy (K, Q, R, B, N) | Avg Rank Displacement | Status / Reliability |
| :--- | :---: | :---: | :---: | :---: |
| **Base Contact Anchor $(x_c, y_c + h/2)$** | **100.0%** | **100.0%** | **0.00 tiles** | **EXACT FOOTPRINT (ADR-008)** |
| **Naive Bounding Box Centroid $(x_c, y_c)$** | 88.4% | 76.5% | 0.12 tiles | UNRELIABLE (Perspective Tilt Fails) |



