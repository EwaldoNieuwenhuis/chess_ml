"""
Perspective Parallax & Contact Footprint Diagnostic Engine (US-2.3.4 / ADR-008).

Analyzes and visualizes why bottom-center base contact anchors (x_c, y_c + h/2) eliminate
perspective tilt errors for tall chess pieces across 30°–75° angled photographs compared to
naive bounding box centroids (x_c, y_c).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from src.dataset.normalizer import (
    DEFAULT_CANONICAL_CONFIG_PATH,
    CanonicalClassMapper,
    NormalizedBBox,
    StandardizedAnnotation,
)
from src.schemas.contracts import PieceType

logger = logging.getLogger(__name__)

# BGR Color definitions
COLOR_CENTROID_BGR = (0, 0, 255)         # Red
COLOR_CONTACT_BGR = (0, 255, 0)          # Green / Lime
COLOR_VECTOR_BGR = (0, 255, 255)         # Yellow
COLOR_MISASSIGNED_TILE = (40, 40, 180)   # Reddish wash
COLOR_CORRECT_TILE = (40, 180, 40)       # Greenish wash

FILE_NAMES = ["a", "b", "c", "d", "e", "f", "g", "h"]
RANK_NAMES = ["1", "2", "3", "4", "5", "6", "7", "8"]


@dataclass
class ContactAnchorPoint:
    """Represents a projected point on the board plane."""

    x: float
    y: float
    anchor_type: str  # "CENTROID" or "BOTTOM_CENTER"
    file_idx: int
    rank_idx: int
    square_name: str


@dataclass
class PieceParallaxMetric:
    """Individual piece parallax displacement and square assignment comparison."""

    class_id: int
    piece_name: str
    piece_type: str
    is_tall: bool
    bbox_height: float
    centroid_pt: tuple[float, float]
    contact_pt: tuple[float, float]
    centroid_square: str
    contact_square: str
    is_misassigned: bool
    rank_error: int
    pixel_displacement: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "piece_name": self.piece_name,
            "piece_type": self.piece_type,
            "is_tall": self.is_tall,
            "bbox_height": round(self.bbox_height, 4),
            "centroid_pt": [round(c, 4) for c in self.centroid_pt],
            "contact_pt": [round(c, 4) for c in self.contact_pt],
            "centroid_square": self.centroid_square,
            "contact_square": self.contact_square,
            "is_misassigned": self.is_misassigned,
            "rank_error": self.rank_error,
            "pixel_displacement": round(self.pixel_displacement, 2),
        }


@dataclass
class ParallaxDiagnosticResult:
    """Aggregated diagnostic metrics across an entire board image."""

    total_pieces: int = 0
    tall_pieces_count: int = 0
    centroid_correct_count: int = 0
    centroid_misassigned_count: int = 0
    contact_correct_count: int = 0
    
    centroid_accuracy_pct: float = 0.0
    contact_accuracy_pct: float = 100.0
    tall_piece_centroid_accuracy_pct: float = 0.0
    average_rank_error: float = 0.0
    max_rank_error: int = 0

    piece_metrics: list[PieceParallaxMetric] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_pieces": self.total_pieces,
            "tall_pieces_count": self.tall_pieces_count,
            "centroid_correct_count": self.centroid_correct_count,
            "centroid_misassigned_count": self.centroid_misassigned_count,
            "contact_correct_count": self.contact_correct_count,
            "centroid_accuracy_pct": round(self.centroid_accuracy_pct, 2),
            "contact_accuracy_pct": round(self.contact_accuracy_pct, 2),
            "tall_piece_centroid_accuracy_pct": round(self.tall_piece_centroid_accuracy_pct, 2),
            "average_rank_error": round(self.average_rank_error, 2),
            "max_rank_error": self.max_rank_error,
            "piece_metrics": [m.to_dict() for m in self.piece_metrics],
        }


class ParallaxContactAnalyzer:
    """
    Mathematical and visual analyzer for perspective parallax footprint contact anchoring.
    """

    TALL_PIECE_TYPES: set[PieceType] = {
        PieceType.KING,
        PieceType.QUEEN,
        PieceType.ROOK,
        PieceType.BISHOP,
        PieceType.KNIGHT,
    }

    def __init__(self, mapper: CanonicalClassMapper | None = None) -> None:
        self.mapper = mapper or CanonicalClassMapper(DEFAULT_CANONICAL_CONFIG_PATH)

    @staticmethod
    def compute_centroid(bbox: NormalizedBBox) -> tuple[float, float]:
        """Calculates normalized bounding box centroid (x_c, y_c)."""
        return (bbox.x_center, bbox.y_center)

    @staticmethod
    def compute_contact_anchor(bbox: NormalizedBBox) -> tuple[float, float]:
        """
        Calculates normalized bottom-center base contact anchor (x_c, y_max) = (x_c, y_c + h/2).
        This point lies on the planar ground surface of the chessboard (Z=0).
        """
        return (bbox.x_center, bbox.y_center + bbox.height / 2.0)

    @staticmethod
    def map_point_to_grid(
        pt: tuple[float, float],
        H_inv: np.ndarray | None = None,
    ) -> tuple[int, int, str]:
        """
        Maps a 2D normalized point (or homography-warped point) to an 8x8 chessboard tile (file_idx, rank_idx).
        
        Returns:
            (file_idx: 0..7, rank_idx: 0..7, square_name: e.g. 'e4')
        """
        x, y = pt
        if H_inv is not None:
            # Transform point via planar homography to rectified orthogonal space
            p_arr = np.array([[[x, y]]], dtype=np.float32)
            warped = cv2.perspectiveTransform(p_arr, H_inv)[0][0]
            x, y = float(warped[0]), float(warped[1])

        # Clamp normalized coordinates to [0.0, 0.999999]
        x_c = max(0.0, min(0.999999, x))
        y_c = max(0.0, min(0.999999, y))

        file_idx = int(x_c * 8.0)
        # Rank 0 is top (rank 8 in chess notation), Rank 7 is bottom (rank 1 in chess notation)
        rank_idx_top_down = int(y_c * 8.0)
        chess_rank = 8 - rank_idx_top_down
        chess_file = FILE_NAMES[file_idx]

        square_name = f"{chess_file}{chess_rank}"
        return file_idx, rank_idx_top_down, square_name

    def analyze_image_annotations(
        self,
        annotations: Sequence[StandardizedAnnotation],
        img_width: int = 640,
        img_height: int = 640,
        H_inv: np.ndarray | None = None,
    ) -> ParallaxDiagnosticResult:
        """
        Computes piece-by-piece rank displacement and square assignment comparisons
        between Centroid and Base Contact Anchor projections.
        """
        result = ParallaxDiagnosticResult()
        if not annotations:
            return result

        rank_errors: list[int] = []
        tall_total = 0
        tall_centroid_correct = 0

        for ann in annotations:
            info = self.mapper.get_class_info(ann.class_id)
            is_tall = info.piece_type in self.TALL_PIECE_TYPES

            # Compute normalized points
            centroid_pt = self.compute_centroid(ann.bbox)
            contact_pt = self.compute_contact_anchor(ann.bbox)

            # Map to grid
            _, c_rank_idx, c_sq = self.map_point_to_grid(centroid_pt, H_inv)
            _, b_rank_idx, b_sq = self.map_point_to_grid(contact_pt, H_inv)

            # Rank displacement error: difference in rank indices
            rank_err = abs(c_rank_idx - b_rank_idx)
            rank_errors.append(rank_err)

            is_misassigned = (c_sq != b_sq)
            pixel_disp = math.hypot(
                (contact_pt[0] - centroid_pt[0]) * img_width,
                (contact_pt[1] - centroid_pt[1]) * img_height,
            )

            metric = PieceParallaxMetric(
                class_id=ann.class_id,
                piece_name=info.name,
                piece_type=info.piece_type.value,
                is_tall=is_tall,
                bbox_height=ann.bbox.height,
                centroid_pt=centroid_pt,
                contact_pt=contact_pt,
                centroid_square=c_sq,
                contact_square=b_sq,
                is_misassigned=is_misassigned,
                rank_error=rank_err,
                pixel_displacement=pixel_disp,
            )
            result.piece_metrics.append(metric)

            result.total_pieces += 1
            if is_tall:
                tall_total += 1
                if not is_misassigned:
                    tall_centroid_correct += 1

            if not is_misassigned:
                result.centroid_correct_count += 1
            else:
                result.centroid_misassigned_count += 1

            result.contact_correct_count += 1  # Ground-truth reference

        result.tall_pieces_count = tall_total
        if result.total_pieces > 0:
            result.centroid_accuracy_pct = (result.centroid_correct_count / result.total_pieces) * 100.0
            result.average_rank_error = sum(rank_errors) / result.total_pieces
            result.max_rank_error = max(rank_errors) if rank_errors else 0

        if tall_total > 0:
            result.tall_piece_centroid_accuracy_pct = (tall_centroid_correct / tall_total) * 100.0

        return result

    def render_composite_diagnostic(
        self,
        img: np.ndarray,
        annotations: Sequence[StandardizedAnnotation],
        H_inv: np.ndarray | None = None,
        title: str = "Perspective Parallax Footprint Verification",
    ) -> np.ndarray:
        """
        Renders a high-resolution 3-panel comparative diagnostic canvas:
          Panel 1 (Left):   Camera Perspective View with Bounding Boxes, Centroids (Red), Contact Anchors (Green), & Vectors.
          Panel 2 (Middle): Rectified 8x8 Top-Down Board with Projected Centroids vs Contact Anchors.
          Panel 3 (Right):  Statistical HUD with Piece-by-Piece Rank Displacements and Accuracy Comparison.
        """
        img_h, img_w = img.shape[:2]
        panel_w, panel_h = 640, 640

        # 1. Prepare Left Panel: Perspective Camera View
        left_panel = cv2.resize(img.copy(), (panel_w, panel_h))
        for ann in annotations:
            info = self.mapper.get_class_info(ann.class_id)
            c_norm = self.compute_centroid(ann.bbox)
            b_norm = self.compute_contact_anchor(ann.bbox)

            cx, cy = int(round(c_norm[0] * panel_w)), int(round(c_norm[1] * panel_h))
            bx, by = int(round(b_norm[0] * panel_w)), int(round(b_norm[1] * panel_h))

            # Bounding box coordinates
            xmin = int(round(ann.bbox.x_min * panel_w))
            ymin = int(round(ann.bbox.y_min * panel_h))
            xmax = int(round(ann.bbox.x_max * panel_w))
            ymax = int(round(ann.bbox.y_max * panel_h))

            # Draw thin bounding box
            cv2.rectangle(left_panel, (xmin, ymin), (xmax, ymax), (180, 180, 180), 1)

            # Draw parallax displacement vector (Yellow arrow)
            cv2.arrowedLine(left_panel, (cx, cy), (bx, by), COLOR_VECTOR_BGR, 2, tipLength=0.25)

            # Draw Centroid (Red 'X' and circle)
            cv2.circle(left_panel, (cx, cy), 4, COLOR_CENTROID_BGR, -1)
            cv2.drawMarker(left_panel, (cx, cy), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 6, 1)

            # Draw Base Contact Anchor (Green glowing dot)
            cv2.circle(left_panel, (bx, by), 6, (0, 0, 0), -1)
            cv2.circle(left_panel, (bx, by), 4, COLOR_CONTACT_BGR, -1)

        # Left panel header banner
        cv2.rectangle(left_panel, (0, 0), (panel_w, 36), (20, 20, 20), -1)
        cv2.putText(
            left_panel,
            "1. Perspective Camera View (Parallax Shift)",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # 2. Prepare Middle Panel: Rectified 8x8 Top-Down Board
        middle_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        tile_sz = panel_w // 8

        # Draw alternating board tiles
        light_color = (240, 217, 181)  # BGR
        dark_color = (181, 136, 99)    # BGR
        for r in range(8):
            for c in range(8):
                t_col = light_color if (r + c) % 2 == 0 else dark_color
                y1, y2 = r * tile_sz, (r + 1) * tile_sz
                x1, x2 = c * tile_sz, (c + 1) * tile_sz
                middle_panel[y1:y2, x1:x2] = t_col

                # Subtle grid coordinate labels
                sq_name = f"{FILE_NAMES[c]}{8 - r}"
                cv2.putText(
                    middle_panel,
                    sq_name,
                    (x1 + 4, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (100, 100, 100),
                    1,
                    cv2.LINE_AA,
                )

        # Analyze piece positions
        diag = self.analyze_image_annotations(annotations, panel_w, panel_h, H_inv)

        # Draw piece points onto rectified 8x8 board
        for metric in diag.piece_metrics:
            # Contact anchor position
            bx_norm, by_norm = metric.contact_pt
            if H_inv is not None:
                p_arr = np.array([[[bx_norm, by_norm]]], dtype=np.float32)
                warped = cv2.perspectiveTransform(p_arr, H_inv)[0][0]
                bx_norm, by_norm = float(warped[0]), float(warped[1])

            # Centroid position
            cx_norm, cy_norm = metric.centroid_pt
            if H_inv is not None:
                p_arr = np.array([[[cx_norm, cy_norm]]], dtype=np.float32)
                warped = cv2.perspectiveTransform(p_arr, H_inv)[0][0]
                cx_norm, cy_norm = float(warped[0]), float(warped[1])

            bx_pix = int(round(bx_norm * panel_w))
            by_pix = int(round(by_norm * panel_h))
            cx_pix = int(round(cx_norm * panel_w))
            cy_pix = int(round(cy_norm * panel_h))

            # Connect centroid to contact point
            cv2.line(middle_panel, (cx_pix, cy_pix), (bx_pix, by_pix), (0, 0, 0), 2)

            # Draw Centroid on Top-Down Board (Red X)
            cv2.circle(middle_panel, (cx_pix, cy_pix), 5, COLOR_CENTROID_BGR, -1)
            cv2.drawMarker(middle_panel, (cx_pix, cy_pix), (255, 255, 255), cv2.MARKER_CROSS, 8, 1)

            # Draw Contact Anchor on Top-Down Board (Green Dot)
            cv2.circle(middle_panel, (bx_pix, by_pix), 7, (0, 0, 0), -1)
            cv2.circle(middle_panel, (bx_pix, by_pix), 5, COLOR_CONTACT_BGR, -1)

            # If misassigned square, draw red alert ring
            if metric.is_misassigned:
                cv2.circle(middle_panel, (cx_pix, cy_pix), 12, (0, 0, 255), 2)

        # Middle panel header banner
        cv2.rectangle(middle_panel, (0, 0), (panel_w, 36), (20, 20, 20), -1)
        cv2.putText(
            middle_panel,
            "2. Rectified Top-Down Plane (Grid Projection)",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # 3. Prepare Right Panel: Diagnostic HUD & Statistical Breakdown
        right_panel = np.full((panel_h, 480, 3), 28, dtype=np.uint8)

        # Header
        cv2.rectangle(right_panel, (0, 0), (480, 36), (45, 45, 45), -1)
        cv2.putText(
            right_panel,
            "3. Parallax Diagnostic Metrics",
            (15, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 215, 255),
            2,
            cv2.LINE_AA,
        )

        # Metrics overview
        y_pos = 65
        cv2.putText(
            right_panel,
            f"Total Pieces Detected: {diag.total_pieces}",
            (15, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y_pos += 24

        # Accuracy comparison boxes
        acc_text = f"Contact Anchor Accuracy: {diag.contact_accuracy_pct:.1f}% [100% Ground Truth]"
        cv2.putText(
            right_panel,
            acc_text,
            (15, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOR_CONTACT_BGR,
            1,
            cv2.LINE_AA,
        )
        y_pos += 24

        cent_color = (0, 255, 255) if diag.centroid_accuracy_pct > 80 else (0, 100, 255)
        cv2.putText(
            right_panel,
            f"Naive Centroid Accuracy: {diag.centroid_accuracy_pct:.1f}%",
            (15, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            cent_color,
            1,
            cv2.LINE_AA,
        )
        y_pos += 24

        cv2.putText(
            right_panel,
            f"Tall Piece Centroid Acc: {diag.tall_piece_centroid_accuracy_pct:.1f}%",
            (15, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 165, 255),
            1,
            cv2.LINE_AA,
        )
        y_pos += 24

        cv2.putText(
            right_panel,
            f"Avg Rank Error: {diag.average_rank_error:.2f} tiles | Max: {diag.max_rank_error} tiles",
            (15, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        y_pos += 30

        # Divider
        cv2.line(right_panel, (15, y_pos), (465, y_pos), (70, 70, 70), 1)
        y_pos += 20

        # Legend
        cv2.putText(right_panel, "LEGEND:", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
        y_pos += 18
        cv2.circle(right_panel, (22, y_pos - 4), 4, COLOR_CONTACT_BGR, -1)
        cv2.putText(right_panel, "Base Contact Anchor (xc, yc + h/2)", (35, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
        y_pos += 18
        cv2.circle(right_panel, (22, y_pos - 4), 4, COLOR_CENTROID_BGR, -1)
        cv2.putText(right_panel, "Bounding Box Centroid (xc, yc)", (35, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
        y_pos += 18
        cv2.arrowedLine(right_panel, (16, y_pos - 4), (28, y_pos - 4), COLOR_VECTOR_BGR, 2, tipLength=0.3)
        cv2.putText(right_panel, "Parallax Displacement Vector (0, h/2)", (35, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
        y_pos += 26

        # Divider
        cv2.line(right_panel, (15, y_pos), (465, y_pos), (70, 70, 70), 1)
        y_pos += 20

        # Piece Breakdown Table Header
        cv2.putText(right_panel, "PIECE-LEVEL PARALLAX DISPLACEMENTS:", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 215, 255), 1)
        y_pos += 18
        header_line = "Piece Type     Height   Centroid  Anchor   Rank Diff"
        cv2.putText(right_panel, header_line, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 150, 150), 1)
        y_pos += 16

        # Sample top pieces
        for metric in diag.piece_metrics[:12]:
            p_name = f"{metric.piece_name[:12]:<14}"
            h_str = f"{metric.bbox_height:.3f}"
            c_sq = f"{metric.centroid_square:<8}"
            b_sq = f"{metric.contact_square:<8}"
            r_err = f"{metric.rank_error}"
            row_str = f"{p_name} {h_str}   {c_sq}  {b_sq}  {r_err}"

            row_color = (0, 255, 0) if not metric.is_misassigned else (0, 120, 255)
            cv2.putText(
                right_panel,
                row_str,
                (15, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                row_color,
                1,
                cv2.LINE_AA,
            )
            y_pos += 16
            if y_pos > panel_h - 20:
                break

        # Assemble side-by-side composite canvas
        composite = np.hstack([left_panel, middle_panel, right_panel])

        # Overall Title Banner across top
        full_w = composite.shape[1]
        top_banner = np.zeros((45, full_w, 3), dtype=np.uint8)
        top_banner[:] = (15, 15, 15)
        cv2.putText(
            top_banner,
            f"Chess ML - {title} (US-2.3.4 / ADR-008)",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        final_composite = np.vstack([top_banner, composite])
        return final_composite
