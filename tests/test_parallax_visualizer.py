"""
Unit, Property, and Integration Tests for Perspective Parallax & Contact Footprint Visualizer (US-2.3.4 / ADR-008).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.dataset.normalizer import (
    CanonicalClassMapper,
    NormalizedBBox,
    StandardizedAnnotation,
)
from src.dataset.parallax import (
    ParallaxContactAnalyzer,
    ParallaxDiagnosticResult,
    PieceParallaxMetric,
)


@pytest.fixture
def temp_workspace() -> Path:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def analyzer() -> ParallaxContactAnalyzer:
    return ParallaxContactAnalyzer()


def create_dummy_board_image(path: Path, size: int = 640) -> Path:
    """Creates a simple 8x8 checkerboard image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tile_sz = size // 8
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for r in range(8):
        for c in range(8):
            col = (240, 217, 181) if (r + c) % 2 == 0 else (181, 136, 99)
            img[r * tile_sz : (r + 1) * tile_sz, c * tile_sz : (c + 1) * tile_sz] = col
    cv2.imwrite(str(path), img)
    return path


def create_dummy_label(path: Path, annotations: list[tuple[int, float, float, float, float]]) -> Path:
    """Writes YOLO format annotations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for cid, xc, yc, w, h in annotations:
            f.write(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
    return path


class TestParallaxContactAnalyzer:
    """Mathematical and geometric unit tests for ParallaxContactAnalyzer."""

    def test_contact_anchor_vs_centroid_calculation(self, analyzer: ParallaxContactAnalyzer) -> None:
        """Verify mathematical formulas: centroid = (xc, yc), contact anchor = (xc, yc + h/2)."""
        bbox = NormalizedBBox(x_center=0.50, y_center=0.45, width=0.10, height=0.16)

        centroid = analyzer.compute_centroid(bbox)
        contact = analyzer.compute_contact_anchor(bbox)

        assert centroid == (0.50, 0.45)
        assert contact == (0.50, 0.53)  # 0.45 + 0.16 / 2 = 0.53

    def test_grid_mapping_standard_squares(self, analyzer: ParallaxContactAnalyzer) -> None:
        """Verify normalized 2D coordinates map to correct chess square algebraic names."""
        # a8 (top-left tile: x in [0, 0.125], y in [0, 0.125])
        file_idx, rank_idx, sq = analyzer.map_point_to_grid((0.0625, 0.0625))
        assert file_idx == 0
        assert rank_idx == 0
        assert sq == "a8"

        # e4 (file e = index 4 [0.50..0.625], rank 4 = index 4 [0.50..0.625] from top)
        file_idx, rank_idx, sq = analyzer.map_point_to_grid((0.5625, 0.5625))
        assert file_idx == 4
        assert rank_idx == 4
        assert sq == "e4"

        # h1 (bottom-right tile: x in [0.875, 1.0], y in [0.875, 1.0])
        file_idx, rank_idx, sq = analyzer.map_point_to_grid((0.95, 0.95))
        assert file_idx == 7
        assert rank_idx == 7
        assert sq == "h1"

    def test_tall_piece_centroid_suffers_perspective_rank_error(self, analyzer: ParallaxContactAnalyzer) -> None:
        """
        Verify that for a tall piece (e.g. King on e4 with height = 0.14),
        the base contact anchor correctly locates e4, whereas the naive centroid
        displaces upward into e5 (rank error >= 1).
        """
        # King standing on square e4 (center of e4 square is at normalized y = 0.5625).
        # Base contact point should be at bottom of tile = 0.600.
        # Height of tall King = 0.14.
        # Centroid is at yc = 0.600 - 0.14/2 = 0.530 (or higher on tilted pieces, e.g. yc = 0.48).
        tall_king_bbox = NormalizedBBox(x_center=0.5625, y_center=0.48, width=0.08, height=0.16)
        # Contact anchor = (0.5625, 0.48 + 0.08 = 0.56) -> square e4
        # Centroid = (0.5625, 0.48) -> square e5 (rank 5 instead of rank 4!)

        _, _, contact_sq = analyzer.map_point_to_grid(analyzer.compute_contact_anchor(tall_king_bbox))
        _, _, centroid_sq = analyzer.map_point_to_grid(analyzer.compute_centroid(tall_king_bbox))

        assert contact_sq == "e4"
        assert centroid_sq == "e5"
        assert centroid_sq != contact_sq  # Centroid suffers from perspective parallax error!

    def test_short_pawn_piece_smaller_displacement(self, analyzer: ParallaxContactAnalyzer) -> None:
        """Verify that short pawns (h < 0.05) suffer smaller displacement than tall pieces."""
        pawn_bbox = NormalizedBBox(x_center=0.5625, y_center=0.55, width=0.06, height=0.04)
        contact = analyzer.compute_contact_anchor(pawn_bbox)
        centroid = analyzer.compute_centroid(pawn_bbox)

        _, _, contact_sq = analyzer.map_point_to_grid(contact)
        _, _, centroid_sq = analyzer.map_point_to_grid(centroid)

        # Both remain in e4 due to low piece height
        assert contact_sq == "e4"
        assert centroid_sq == "e4"

    def test_analyze_image_annotations_aggregate_metrics(self, analyzer: ParallaxContactAnalyzer) -> None:
        """Verify aggregate accuracy and rank error calculation across multiple pieces."""
        annotations = [
            # 1. White King (Class 5 - Tall piece, misassigned centroid)
            StandardizedAnnotation(
                class_id=5,
                bbox=NormalizedBBox(x_center=0.5625, y_center=0.48, width=0.08, height=0.16),
                class_name="white_king",
            ),
            # 2. Black Queen (Class 10 - Tall piece, misassigned centroid)
            StandardizedAnnotation(
                class_id=10,
                bbox=NormalizedBBox(x_center=0.4375, y_center=0.48, width=0.08, height=0.16),
                class_name="black_queen",
            ),
            # 3. White Pawn (Class 0 - Short piece, correct centroid)
            StandardizedAnnotation(
                class_id=0,
                bbox=NormalizedBBox(x_center=0.1875, y_center=0.55, width=0.06, height=0.04),
                class_name="white_pawn",
            ),
        ]

        result = analyzer.analyze_image_annotations(annotations, img_width=640, img_height=640)

        assert result.total_pieces == 3
        assert result.tall_pieces_count == 2
        assert result.contact_correct_count == 3
        assert result.contact_accuracy_pct == 100.0
        assert result.centroid_misassigned_count == 2
        assert result.centroid_correct_count == 1
        assert abs(result.centroid_accuracy_pct - 33.33) < 0.1
        assert result.tall_piece_centroid_accuracy_pct == 0.0

    def test_render_composite_diagnostic_output_shape(self, analyzer: ParallaxContactAnalyzer) -> None:
        """Verify that render_composite_diagnostic produces a valid multi-panel image."""
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        annotations = [
            StandardizedAnnotation(
                class_id=5,
                bbox=NormalizedBBox(x_center=0.5, y_center=0.5, width=0.1, height=0.15),
                class_name="white_king",
            )
        ]

        composite = analyzer.render_composite_diagnostic(img, annotations, title="Test Diagnostic")

        # Expect height = 640 + 45 (top banner) = 685
        # Expect width = 640 (left) + 640 (middle) + 480 (right HUD) = 1760
        assert composite.ndim == 3
        assert composite.shape[0] == 685
        assert composite.shape[1] == 1760
        assert composite.shape[2] == 3
        assert np.any(composite > 0)  # Non-empty canvas


class TestVerifyContactAnchorsCLI:
    """Integration test suite for scripts/verify_contact_anchors.py CLI."""

    def test_cli_execution_generates_diagnostics(self, temp_workspace: Path) -> None:
        """Verify CLI runs successfully, generates diagnostic images, and exports reports."""
        dataset_dir = temp_workspace / "hybrid_chess"
        output_dir = temp_workspace / "diagnostics" / "parallax_verification"

        img_p = dataset_dir / "images" / "train" / "sample_01.jpg"
        lbl_p = dataset_dir / "labels" / "train" / "sample_01.txt"

        create_dummy_board_image(img_p)
        create_dummy_label(
            lbl_p,
            [
                (5, 0.5625, 0.48, 0.08, 0.16),  # White King
                (4, 0.4375, 0.48, 0.08, 0.16),  # White Queen
                (0, 0.1875, 0.55, 0.06, 0.04),  # White Pawn
            ],
        )

        cmd = [
            sys.executable,
            "scripts/verify_contact_anchors.py",
            "--dataset-dir",
            str(dataset_dir),
            "--output-dir",
            str(output_dir),
            "--samples",
            "1",
            "--sweep-angles",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

        assert result.returncode == 0
        assert "Parallax diagnostic verification complete" in result.stdout
        assert output_dir.exists()

        # Check exported report files
        summary_json = output_dir / "summary.json"
        summary_md = output_dir / "summary.md"
        assert summary_json.exists()
        assert summary_md.exists()

        # Check angle sweep output images
        for deg in (30, 45, 60, 75):
            angle_img = output_dir / f"parallax_sweep_angle_{deg}deg.png"
            assert angle_img.exists()
            assert angle_img.stat().st_size > 0
