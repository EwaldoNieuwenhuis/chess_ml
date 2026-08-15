# 🎯 Functional Specification: Use Cases & Workflows

## 1. Overview
The **Unified 2D/3D Chess Vision & Move Recommendation System** supports two primary operational modes:
1. **Digital 2D Screenshots** (Chess.com, Lichess, PDF diagrams, chess streams).
2. **Physical 3D Photos & Video Streams** (Smartphone captures, webcam over-the-board recordings, tournaments).

---

## 2. Core User Journeys

### ♟️ Use Case 1: Digital Screenshot Analysis (2D)
```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Pipeline
    participant Classifier as Domain Classifier
    participant Extractor as Digital Grid Extractor
    participant Detector as 2D Piece Classifier
    participant Engine as Stockfish Manager

    User->>Pipeline: Upload Screenshot (PNG/JPEG)
    Pipeline->>Classifier: Detect Domain
    Classifier-->>Pipeline: Domain = DIGITAL_2D
    Pipeline->>Extractor: Segment 8x8 Grid Squares
    Pipeline->>Detector: Classify Pieces per Cell
    Pipeline->>Engine: Send Synthesized FEN
    Engine-->>Pipeline: Top Move (e.g. "e2e4", Eval: +0.45)
    Pipeline-->>User: Annotated Image with Move Arrow + Evaluation Panel
```

### 📷 Use Case 2: Physical Over-the-Board Camera Capture (3D)
```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Pipeline
    participant Geometry as 4-Corner Homography
    participant YOLO as YOLO26/v12 Detector
    participant Mapper as Bottom-Anchor Mapper
    participant Validator as python-chess Layer
    participant Engine as Stockfish Manager

    User->>Pipeline: Live Frame or Photo
    Pipeline->>Geometry: Detect 4 Board Corners & Warp Perspective
    Pipeline->>YOLO: Detect Pieces & Base Points
    Pipeline->>Mapper: Project Base Points to Rectified 8x8 Grid
    Pipeline->>Validator: Validate King Counts, Pawn Rows & Legality
    Validator-->>Pipeline: Legal FEN String
    Pipeline->>Engine: Compute Best Move
    Engine-->>Pipeline: Top Move & PV Line
    Pipeline-->>User: Move Overlay + FEN Clipboard Export
```

---

## 3. Supported Input Modalities
* **Static Images:** `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`.
* **Video & Live Streams:** RTSP, USB Webcam, MP4/AVI video files (processed at target framerate e.g., 10-30 FPS).
* **CLI / API:** Base64-encoded image payloads or local file paths.
