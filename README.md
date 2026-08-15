# ♟️ Unified 2D/3D Chess Vision & Move Recommendation System

An end-to-end computer vision and machine learning pipeline that ingests either a digital chessboard screenshot (Chess.com / Lichess / other chess apps) or a real-world 3D chessboard photo, automatically classifies the domain, identifies the board state, converts it into a valid FEN string, and queries the Stockfish chess engine to output the best tactical move with visual overlays.

---

## 🏛️ Pipeline Architecture

```
[Input Image (2D Screenshot or 3D Photo)]
                   │
                   ▼
     ┌───────────────────────────┐
     │ 1. Domain Classifier      │ (Laplacian/Palette Heuristics or MobileNet)
     └─────────────┬─────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
 ┌───────────────┐   ┌───────────────────────────────┐
 │ Digital Branch│   │ Physical Branch               │
 │ (Contour/Grid)│   │ (4-Corner Homography / Warp)  │
 └───────┬───────┘   └───────────────┬───────────────┘
         │                           │
         └─────────────┬─────────────┘
                       ▼
       ┌───────────────────────────────┐
       │ 2. Piece Detection (YOLO/ONNX)│ (Trained on Open-Source Curated Hybrid Dataset)
       └───────────────┬───────────────┘
                       ▼
       ┌───────────────────────────────┐
       │ 3. Coordinate & FEN Mapping   │ (Bottom-Center Anchor + Orientation)
       └───────────────┬───────────────┘
                       ▼
       ┌───────────────────────────────┐
       │ 4. Legality Check & Engine    │ (python-chess + Stockfish UCI)
       └───────────────┬───────────────┘
                       ▼
       ┌───────────────────────────────┐
       │ 5. Annotated Move Visualizer  │ (Best Move Arrow + Eval Score)
       └───────────────────────────────┘
```

---

## 📋 Product Backlog & Tracking

Project work is organized and tracked inside [BACKLOG.md](file:///c:/coding/chess_ml/BACKLOG.md).
Every Epic includes:
1. **Visual Architecture Pipeline Diagram** (Mermaid).
2. **User Stories & Acceptance Criteria** with auto-updating checkboxes (`[ ]` $\to$ `[-]` $\to$ `[x]`).
3. **💡 Architectural Notes & Alternative Considerations** (documenting trade-offs, alternative models, and evaluation rationales).

### Core Epics:
- **EPIC-01:** Core Infrastructure, Tooling & Engine Evaluation
- **EPIC-02:** Open-Source Dataset Discovery, Ingestion & Curation (Roboflow Universe, Kaggle, ChessReD)
- **EPIC-03:** Domain Classification & Geometric Pre-Processing
- **EPIC-04:** Chess Piece Detection & Recognition Model Architecture
- **EPIC-05:** Coordinate Mapping, Orientation & FEN Synthesis
- **EPIC-06:** End-to-End Pipeline, Move Recommendation & Visualization
