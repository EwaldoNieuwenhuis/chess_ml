# ⚠️ Error Handling & Edge Case Specification

This document details how the vision pipeline detects, reports, and auto-corrects failure modes and edge cases.

---

## 1. Edge Case Taxonomy & Mitigation Strategies

| Error Scenario | Detection Mechanism | Automated Mitigation / System Response |
| :--- | :--- | :--- |
| **No Board Detected** | Corner detector confidence $< 0.40$ or contour ratio invalid. | Return `E1001_BOARD_NOT_FOUND` with debug bounding boxes; prompt user to center the board. |
| **Duplicate Kings** | $>1$ White or Black King detected. | Keep King with highest classification confidence; demote secondary detection to Queen or Pawn. |
| **Missing King** | $0$ White or Black King detected. | Flag `is_legal = false`; infer King location based on highest unassigned square confidence. |
| **Pawns on 1st / 8th Rank** | Pawn assigned to rank 1 or 8. | Automatically promote to Queen or demote to nearest legal rank based on bounding box. |
| **Severe Occlusion** | Multiple piece bottom-anchors fall within the same square threshold. | Apply Non-Maximum Suppression on square assignment; query move history if in live video mode. |
| **Ambiguous Orientation** | Files/ranks inverted (Black on ranks 1-2). | Auto-detect piece baseline clusters (White pieces usually start on ranks 1-2); flip orientation if inverted. |
| **Stockfish Binary Unavailable** | Subprocess cannot find `stockfish.exe`. | Seamlessly fallback to `python-chess` internal engine / mock evaluator with a warning. |

---

## 2. Standardized Error Codes

```text
E1000 - UNKNOWN_PIPELINE_ERROR
E1001 - BOARD_NOT_FOUND (Cannot locate 4 board corners)
E1002 - PERSPECTIVE_WARP_FAILED (Homography matrix singular)
E1003 - PIECE_DETECTION_TIMEOUT
E1004 - ILLEGAL_BOARD_STATE (Cannot construct legal chess position)
E2001 - ENGINE_BINARY_NOT_FOUND
E2002 - ENGINE_TIMEOUT
```
