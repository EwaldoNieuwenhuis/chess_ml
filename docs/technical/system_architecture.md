# 🏗️ Technical Architecture & Pipeline Design

## 1. System Overview (C4 Container Diagram)

```mermaid
graph TD
    User["Client / Camera Stream"] --> API["FastAPI / CLI Entrypoint"]
    
    subgraph Core Pipeline ["Unified Chess ML Pipeline (src/pipeline)"]
        API --> DC["1. Domain Classifier (src/domain_classifier)"]
        DC -->|Digital 2D| DProc["2D Contour & Grid Extractor"]
        DC -->|Physical 3D| Geo["2. Homography & Corner Detector (src/geometry)"]
        
        DProc --> Det["3. YOLO26/v12 Piece Detector (src/detection)"]
        Geo --> Det
        
        Det --> Mapper["4. Coordinate & Base-Point Mapper (src/fen_mapper)"]
        Mapper --> Val["5. Rule Validator & FEN Generator (src/fen_mapper)"]
    end
    
    subgraph Engine Layer ["Stockfish Engine Manager (src/engine)"]
        Val --> UCI["Stockfish UCI Manager"]
        UCI -->|Subprocess Pipe| Binary["Stockfish 16+ Binary"]
        UCI -->|Fallback| PyChess["python-chess Internal Engine"]
    end
    
    UCI --> Viz["6. Visualization & Overlay Renderer (src/utils)"]
    Viz --> API
```

---

### 3. Stockfish Engine Layer Architecture (`src/engine`)

The Engine Manager abstracts the UCI communication into a high-performance, non-blocking evaluation service:

```mermaid
flowchart TD
    A[FEN String] --> B{Valid FEN & Legal?}
    B -->|Invalid| C[Raise InvalidFENError]
    B -->|Checkmate / Stalemate| D[Immediate Terminal Evaluation: score_mate=0 / score_cp=0]
    B -->|Active Game State| E{Stockfish Binary Available?}
    E -->|Yes: Local bin / PATH| F[Persistent UCI Subprocess Session]
    E -->|No / CI Test Mode| G[python-chess Internal Fallback Evaluator]
    F --> H[UCI Protocol: setoption, position fen, go depth/time]
    G --> I[Minimax Search / Heuristic Score]
    H --> J[Parse InfoDict: PV, PovScore, Centipawns/Mate]
    I --> J
    J --> K[EngineEvaluation Pydantic Contract]
```

#### Key Engineering Features:
* **Persistent Session Pool**: Avoids $150\text{ ms}$ process re-spawning penalty; keeps the engine initialized for $<2\text{ ms}$ queries.
* **Windows GUI Isolation**: Sets `subprocess.STARTUPINFO.dwFlags |= STARTF_USESHOWWINDOW` to prevent command prompt windows from popping up.
* **Auto-Downloader**: Automatically fetches the official Stockfish release for the local host platform if missing.
* **Async & Sync Dual API**: Supports both `async def evaluate_async()` (for FastAPI / video loops) and `def evaluate_sync()` (for CLI / tests).

---

## 2. Component Directory Architecture

| Package | Responsibility | Primary Classes / Functions |
| :--- | :--- | :--- |
| **`src/schemas`** | Pydantic v2 data contracts | `PieceDetection`, `BoardStateResult`, `EngineEvaluation` |
| **`src/domain_classifier`** | Detects digital vs. physical images | `DomainClassifier`, `classify_domain()` |
| **`src/geometry`** | Corner detection & Homography | `CornerDetector`, `compute_homography()`, `warp_board()` |
| **`src/detection`** | YOLO piece & keypoint inference | `YOLOPieceDetector`, `ONNXInferenceEngine` |
| **`src/fen_mapper`** | Base-to-grid mapping & FEN rules | `FENMapper`, `validate_legal_state()` |
| **`src/engine`** | UCI Stockfish wrapper & fallback | `StockfishManager`, `query_evaluation()` |
| **`src/pipeline`** | End-to-end orchestration | `ChessVisionPipeline`, `process_image()` |
| **`src/utils`** | Image plotting, arrows, metrics | `draw_board_overlay()`, `setup_logger()` |
