# 📐 Coordinate Geometry & Homography Specification

## 1. Homography Transformation (Perspective Rectification)

A physical chessboard photographed from an angle undergoes projective distortion. We compute the $3 \times 3$ Homography matrix $H$ using the 4 detected board corners:

$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} \sim H \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

### Source & Destination Points
* **Source ($P_s$):** Detected 4 corners in original image plane: $[(x_{TL}, y_{TL}), (x_{TR}, y_{TR}), (x_{BR}, y_{BR}), (x_{BL}, y_{BL})]$.
* **Destination ($P_d$):** Canonical square board grid of size $S \times S$ (e.g. $800 \times 800$ px): $[(0, 0), (S, 0), (S, S), (0, S)]$.

```python
import cv2
import numpy as np

def compute_homography_matrix(src_corners: np.ndarray, board_size: int = 800) -> np.ndarray:
    """
    Computes 3x3 projective transformation matrix H.
    
    Args:
        src_corners: shape (4, 2) in order [TL, TR, BR, BL]
        board_size: Target square pixel dimension (default: 800)
    Returns:
        H: shape (3, 3), dtype float64
    """
    dst_corners = np.array([
        [0, 0],
        [board_size, 0],
        [board_size, board_size],
        [0, board_size]
    ], dtype=np.float32)
    
    H, _ = cv2.findHomography(src_corners.astype(np.float32), dst_corners)
    return H
```

---

## 2. Parallax-Free Square Assignment

Instead of using the center of the bounding box $(\frac{x_{min}+x_{max}}{2}, \frac{y_{min}+y_{max}}{2})$, we project the **bottom contact point** $(\frac{x_{min}+x_{max}}{2}, y_{max})$:

1. Transform the bottom-contact point using $H$:
   $$\begin{bmatrix} \tilde{x}' \\ \tilde{y}' \\ w \end{bmatrix} = H \begin{bmatrix} x_{base} \\ y_{base} \\ 1 \end{bmatrix}, \quad x'_{rect} = \frac{\tilde{x}'}{w}, \quad y'_{rect} = \frac{\tilde{y}'}{w}$$

2. Convert canonical pixel coordinates to file and rank indices ($0 \dots 7$):
   $$\text{file\_idx} = \left\lfloor \frac{x'_{rect}}{S / 8} \right\rfloor, \quad \text{rank\_idx} = 7 - \left\lfloor \frac{y'_{rect}}{S / 8} \right\rfloor$$
