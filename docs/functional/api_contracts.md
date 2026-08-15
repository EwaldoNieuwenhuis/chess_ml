# 📡 API & Data Contract Specifications

## 1. Request Schema (Inference Endpoint)

```json
{
  "image": "data:image/jpeg;base64,...",
  "domain_override": null,
  "perspective_side": "white",
  "engine_depth": 18,
  "engine_time_limit_sec": 1.5,
  "return_annotated_image": true
}
```

### Parameter Reference
| Field | Type | Required | Description |
| :--- | :--- | :---: | :--- |
| `image` | `string` | **Yes** | Base64-encoded image string or multipart form data. |
| `domain_override` | `string` | No | `"digital_2d"`, `"physical_3d"`, or `null` for auto-classification. |
| `perspective_side` | `string` | No | Player perspective (`"white"`, `"black"`, or `"auto"`). |
| `engine_depth` | `int` | No | Stockfish depth limit (Default: 18, Range: 1–30). |
| `engine_time_limit_sec`| `float` | No | Maximum engine computation time in seconds. |
| `return_annotated_image`| `bool` | No | Whether to return base64-encoded annotated image. |

---

## 2. Response Schema

```json
{
  "status": "success",
  "domain": "physical_3d",
  "confidence": 0.964,
  "board_state": {
    "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "is_legal": true,
    "validation_warnings": []
  },
  "evaluation": {
    "best_move_uci": "d2d4",
    "best_move_san": "d4",
    "score_cp": 45,
    "score_mate": null,
    "formatted_score": "+0.45",
    "depth": 18,
    "pv": ["d2d4", "e5d4", "f3d4", "g8f6"]
  },
  "annotated_image": "data:image/jpeg;base64,..."
}
```
