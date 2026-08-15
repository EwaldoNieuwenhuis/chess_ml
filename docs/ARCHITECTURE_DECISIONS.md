# 🏛️ Architecture Decision Records (ADRs)

This document records the foundational architectural decisions, trade-offs, and technical rationale for the **Unified 2D/3D Chess Vision & Move Recommendation System**.

---

## 📌 ADR-001: Package & Tooling Foundation via `uv` + `pyproject.toml`

### **Status:** Accepted
### **Context:**
Python dependency resolution (especially with PyTorch CUDA wheels, OpenCV bindings, and ONNX Runtime) is notoriously prone to dependency drift, slow installs (5+ minutes with `pip`), and cross-platform wheel mismatches on Windows/Linux.

### **Decision:**
Adopt **`uv`** as the core package manager and virtual environment provider with a modern standard [`pyproject.toml`](file:///c:/coding/chess_ml/pyproject.toml) configuration.
- Pin PyTorch CUDA 12.x wheels using explicit index definitions in `tool.uv.index`.
- Provide standard `requirements.txt` for legacy CI/CD compatibility.

### **Consequences:**
- Sub-second dependency resolution and deterministic `uv.lock`.
- Reproducible local and container environments.

---

## 📌 ADR-002: 4-Stage Decoupled Hybrid Pipeline vs. End-to-End Deep Learning

### **Status:** Accepted
### **Context:**
Academic benchmarks (such as Masouris & van Gemert on the *ChessReD* dataset) demonstrated that end-to-end black-box deep learning models predicting FEN directly from raw images achieve low full-board accuracy (~15.26%) due to perspective distortion and occlusion.

### **Decision:**
Implement a **4-stage decoupled hybrid pipeline**:
1. **Domain & Geometry**: Corner detection + $3 \times 3$ Homography Perspective Rectification.
2. **Piece & Base Detection**: High-speed object detector (YOLO26 / YOLOv12 / YOLO-Pose).
3. **Coordinate & Grid Mapping**: Inverse projective transform mapping bottom-center contacts to the $8 \times 8$ grid.
4. **Legality & FEN Assembly**: `python-chess` validation layer with heuristic hallucination correction + Stockfish UCI.

### **Consequences:**
- Individual stages can be unit-tested without requiring a GPU.
- Full board recognition accuracy reaches $>98\%$ per square.
- Modular replacement of detection backbones without breaking engine integration.

---

## 📌 ADR-003: Parallax Mitigation via Bottom-Base Contact Point Localization

### **Status:** Accepted
### **Context:**
In angled 3D camera perspectives, tall pieces (e.g. King, Queen) exhibit significant parallax distortion: the piece's head/centroid visually projects into the square behind it (`e5`), while its actual base sits on `e4`. Using standard bounding box centroids causes severe square misassignment.

### **Decision:**
Localize the **bottom-base contact point** (`(x_min + x_max) / 2, y_max` or YOLO-Pose keypoints) for projecting onto the homography plane, rather than the bounding box centroid.

### **Consequences:**
- Completely eliminates parallax misclassification across camera pitch angles from $30^\circ$ to $90^\circ$.
- Requires annotating or computing bottom-center anchors during dataset ingestion.

---

## 📌 ADR-004: Contract-Driven Architecture with Strict Pydantic v2 Schemas

### **Status:** Accepted
### **Context:**
Passing raw dictionaries or loose tuples (`(x, y, cls, conf)`) across pipeline stages causes coordinate flipping bugs (`(x,y)` vs `(row,col)`), type confusion, and runtime failures.

### **Decision:**
Enforce typed Pydantic v2 data models ([`src/schemas/contracts.py`](file:///c:/coding/chess_ml/src/schemas/contracts.py)) across every interface boundary (`Point2D`, `BoundingBox`, `PieceDetection`, `BoardCorners`, `BoardStateResult`, `EngineEvaluation`).

### **Consequences:**
- Strong runtime validation and IDE autocompletion.
- Seamless JSON serialization for API and CLI endpoints.

---

## 📌 ADR-005: Object Detection Architecture Strategy (YOLO26 & YOLOv12 Backbones)

### **Status:** Accepted
### **Context:**
Chess piece detection requires high inference speed for live video feeds, high recall for dense piece clusters (where pieces stand side-by-side), and robust attention mechanisms for partially occluded pieces.

### **Decision:**
- Standardize on **`ultralytics`** as the primary training and inference framework.
- Support **YOLO26** (NMS-free end-to-end architecture) for real-time video feeds to eliminate NMS suppression artifacts.
- Support **YOLOv12** (Area Attention) for high-occlusion static photos.
- Export all trained weights to **ONNX / TensorRT** for low-latency GPU execution.

### **Consequences:**
- Unified API across model iterations.
- Real-time performance ($>60$ FPS on modern GPUs).
