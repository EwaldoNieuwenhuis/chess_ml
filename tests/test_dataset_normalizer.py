"""
Unit and Property Tests for Canonical Class Name & Coordinate Normalizer (US-2.3.1 / ADR-008).
"""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from src.dataset.normalizer import (
    AnnotationStandardizer,
    CanonicalClassInfo,
    CanonicalClassMapper,
    DatasetLabelNormalizer,
    InvalidCoordinateError,
    NormalizedBBox,
    SanitizationStats,
    StandardizedAnnotation,
    UnknownClassError,
)
from src.schemas.contracts import PieceColor, PieceType


class TestCanonicalClassMapper:
    """Tests for CanonicalClassMapper class ontology and aliasing."""

    @pytest.fixture
    def mapper(self) -> CanonicalClassMapper:
        return CanonicalClassMapper()

    def test_canonical_12_classes_exist(self, mapper: CanonicalClassMapper) -> None:
        """Verify all 12 canonical piece classes are properly defined and indexed 0..11."""
        names = mapper.get_canonical_names()
        assert len(names) == 12
        assert sorted(names.keys()) == list(range(12))

        # Check White pieces 0..5
        assert names[0] == "white_pawn"
        assert names[1] == "white_knight"
        assert names[2] == "white_bishop"
        assert names[3] == "white_rook"
        assert names[4] == "white_queen"
        assert names[5] == "white_king"

        # Check Black pieces 6..11
        assert names[6] == "black_pawn"
        assert names[7] == "black_knight"
        assert names[8] == "black_bishop"
        assert names[9] == "black_rook"
        assert names[10] == "black_queen"
        assert names[11] == "black_king"

    def test_grouped_indexing_properties(self, mapper: CanonicalClassMapper) -> None:
        """Verify mathematical color and piece type properties (ADR-008)."""
        for cid in range(12):
            info = mapper.get_class_info(cid)

            # Color property
            if cid < 6:
                assert info.color == PieceColor.WHITE
                assert info.is_white
                assert not info.is_black
            else:
                assert info.color == PieceColor.BLACK
                assert info.is_black
                assert not info.is_white

            # Modulo piece type property
            expected_types = [
                PieceType.PAWN,
                PieceType.KNIGHT,
                PieceType.BISHOP,
                PieceType.ROOK,
                PieceType.QUEEN,
                PieceType.KING,
            ]
            assert info.piece_type == expected_types[cid % 6]

            # Bidirectional conversion roundtrip
            pt, col = mapper.to_piece_type_and_color(cid)
            assert pt == info.piece_type
            assert col == info.color
            reconstructed_cid = mapper.from_piece_type_and_color(pt, col)
            assert reconstructed_cid == cid

    def test_fen_char_bidirectional_mapping(self, mapper: CanonicalClassMapper) -> None:
        """Verify FEN piece character conversion."""
        fen_expectations = {
            0: "P", 1: "N", 2: "B", 3: "R", 4: "Q", 5: "K",
            6: "p", 7: "n", 8: "b", 9: "r", 10: "q", 11: "k",
        }
        for cid, char in fen_expectations.items():
            assert mapper.to_fen_char(cid) == char
            info = mapper.from_fen_char(char)
            assert info.class_id == cid

    @pytest.mark.parametrize(
        ("raw_label", "expected_class_id", "expected_name"),
        [
            # FEN chars
            ("P", 0, "white_pawn"),
            ("p", 6, "black_pawn"),
            ("Q", 4, "white_queen"),
            ("q", 10, "black_queen"),
            ("K", 5, "white_king"),
            ("k", 11, "black_king"),
            # Short notation
            ("wP", 0, "white_pawn"),
            ("bP", 6, "black_pawn"),
            ("wN", 1, "white_knight"),
            ("bN", 7, "black_knight"),
            ("W_B", 2, "white_bishop"),
            ("B_B", 8, "black_bishop"),
            ("W_R", 3, "white_rook"),
            ("B_R", 9, "black_rook"),
            ("wQ", 4, "white_queen"),
            ("bQ", 10, "black_queen"),
            ("wK", 5, "white_king"),
            ("bK", 11, "black_king"),
            ("wp", 0, "white_pawn"),
            ("bp", 6, "black_pawn"),
            ("wn", 1, "white_knight"),
            ("bn", 7, "black_knight"),
            # Hyphenated & snake_case
            ("white-pawn", 0, "white_pawn"),
            ("black-pawn", 6, "black_pawn"),
            ("white_knight", 1, "white_knight"),
            ("black_knight", 7, "black_knight"),
            ("White-Queen", 4, "white_queen"),
            ("Black_Queen", 10, "black_queen"),
            ("WhiteKing", 5, "white_king"),
            ("black king", 11, "black_king"),
        ],
    )
    def test_alias_mapping(
        self,
        mapper: CanonicalClassMapper,
        raw_label: str,
        expected_class_id: int,
        expected_name: str,
    ) -> None:
        """Verify robust string alias translation."""
        info = mapper.map_class(raw_label)
        assert info.class_id == expected_class_id
        assert info.name == expected_name

    def test_dataset_specific_mapping_chessred(self, mapper: CanonicalClassMapper) -> None:
        """Verify 1-indexed ChessReD category ID translation."""
        for cat_id in range(1, 13):
            info = mapper.map_class(cat_id, dataset_source="chessred")
            assert info.class_id == cat_id - 1

    def test_dataset_specific_mapping_roboflow_staunton(self, mapper: CanonicalClassMapper) -> None:
        """Verify alphabetical 12-class Roboflow translation."""
        # In Roboflow alphabetical: 0 is black-bishop (canonical 8), 6 is white-bishop (canonical 2)
        assert mapper.map_class(0, dataset_source="roboflow_staunton").class_id == 8  # black_bishop
        assert mapper.map_class(1, dataset_source="roboflow_staunton").class_id == 11  # black_king
        assert mapper.map_class(6, dataset_source="roboflow_staunton").class_id == 2  # white_bishop
        assert mapper.map_class(9, dataset_source="roboflow_staunton").class_id == 0  # white_pawn

    def test_unknown_class_raises_error(self, mapper: CanonicalClassMapper) -> None:
        """Verify invalid or non-piece classes raise UnknownClassError."""
        with pytest.raises(UnknownClassError):
            mapper.map_class("board_corner")

        with pytest.raises(UnknownClassError):
            mapper.map_class("chessboard")

        with pytest.raises(UnknownClassError):
            mapper.map_class(99)


class TestNormalizedBBox:
    """Tests for NormalizedBBox coordinate conversion and sanitization."""

    def test_normalized_yolo_constructor(self) -> None:
        bbox = NormalizedBBox.from_normalized_yolo(0.5, 0.5, 0.2, 0.4)
        assert bbox.x_center == pytest.approx(0.5)
        assert bbox.y_center == pytest.approx(0.5)
        assert bbox.width == pytest.approx(0.2)
        assert bbox.height == pytest.approx(0.4)
        assert bbox.x_min == pytest.approx(0.4)
        assert bbox.x_max == pytest.approx(0.6)
        assert bbox.y_min == pytest.approx(0.3)
        assert bbox.y_max == pytest.approx(0.7)
        assert bbox.area == pytest.approx(0.08)
        assert bbox.bottom_center == pytest.approx((0.5, 0.7))

    def test_from_xyxy_pixels(self) -> None:
        # 800x600 image, box at [100, 150, 300, 450]
        bbox = NormalizedBBox.from_xyxy_pixels(100, 150, 300, 450, img_w=800, img_h=600)
        assert bbox.x_min == pytest.approx(100 / 800)
        assert bbox.y_min == pytest.approx(150 / 600)
        assert bbox.x_max == pytest.approx(300 / 800)
        assert bbox.y_max == pytest.approx(450 / 600)

    def test_from_coco_pixels(self) -> None:
        # 1000x1000 image, COCO box [x_min=100, y_min=200, width=50, height=80]
        bbox = NormalizedBBox.from_coco_pixels(100, 200, 50, 80, img_w=1000, img_h=1000)
        assert bbox.x_min == pytest.approx(0.10)
        assert bbox.y_min == pytest.approx(0.20)
        assert bbox.width == pytest.approx(0.05)
        assert bbox.height == pytest.approx(0.08)
        assert bbox.x_center == pytest.approx(0.125)
        assert bbox.y_center == pytest.approx(0.24)

    def test_invalid_image_dimension_raises_error(self) -> None:
        with pytest.raises(InvalidCoordinateError):
            NormalizedBBox.from_xyxy_pixels(0, 0, 10, 10, img_w=0, img_h=100)

    def test_epsilon_clamping_slight_drift(self) -> None:
        """Verify coordinates slightly outside [0, 1] due to float precision are clamped."""
        # Box slightly past left and right borders: x_min = -0.000005, x_max = 1.000004
        bbox = NormalizedBBox.from_normalized_xyxy(-0.000005, 0.1, 1.000004, 0.9)
        clamped, was_clamped = bbox.clamp(epsilon=1e-5)
        assert clamped is not None
        assert was_clamped is True
        assert clamped.x_min == pytest.approx(0.0)
        assert clamped.x_max == pytest.approx(1.0)
        assert clamped.is_valid()

    def test_severe_out_of_bounds_discarded(self) -> None:
        """Verify box with < 40% area inside frame is dropped."""
        # Box mostly outside left edge: x_min = -0.3, x_max = 0.05 (only ~14% inside)
        bbox = NormalizedBBox.from_normalized_xyxy(-0.3, 0.2, 0.05, 0.6)
        clamped, _ = bbox.clamp(min_visibility_ratio=0.40)
        assert clamped is None

        # Box completely outside
        outside_box = NormalizedBBox.from_normalized_xyxy(1.2, 0.2, 1.5, 0.6)
        clamped_outside, _ = outside_box.clamp()
        assert clamped_outside is None

    def test_degenerate_box_rejection(self) -> None:
        """Verify boxes smaller than minimum dimension or zero area are rejected."""
        # Zero width
        zero_w = NormalizedBBox.from_normalized_yolo(0.5, 0.5, 0.0, 0.2)
        assert not zero_w.is_valid()

        # Tiny noise box (w < 0.005)
        tiny_box = NormalizedBBox.from_normalized_yolo(0.5, 0.5, 0.001, 0.001)
        assert not tiny_box.is_valid(min_dim=0.005)


class TestAnnotationStandardizer:
    """Tests for AnnotationStandardizer parsing and batch sanitization."""

    @pytest.fixture
    def standardizer(self) -> AnnotationStandardizer:
        return AnnotationStandardizer()

    def test_parse_yolo_line_valid(self, standardizer: AnnotationStandardizer) -> None:
        stats = SanitizationStats()
        line = "0 0.500000 0.500000 0.100000 0.200000\n"
        ann = standardizer.parse_yolo_line(line, stats=stats)
        assert ann is not None
        assert ann.class_id == 0
        assert ann.class_name == "white_pawn"
        assert ann.bbox.x_center == pytest.approx(0.5)
        assert ann.to_yolo_line() == "0 0.500000 0.500000 0.100000 0.200000"
        assert stats.valid_annotations == 1

    def test_parse_yolo_line_alias_and_clamping(self, standardizer: AnnotationStandardizer) -> None:
        stats = SanitizationStats()
        # Uses string alias 'bQ' and slight negative coordinate
        line = "bQ -0.000002 0.400000 0.150000 0.300000\n"
        ann = standardizer.parse_yolo_line(line, stats=stats)
        assert ann is not None
        assert ann.class_id == 10  # black_queen
        assert ann.class_name == "black_queen"
        assert ann.bbox.x_min >= 0.0
        assert stats.clamped_annotations == 1

    def test_parse_yolo_file(self, standardizer: AnnotationStandardizer, tmp_path: Path) -> None:
        test_file = tmp_path / "sample.txt"
        test_file.write_text(
            "white-king 0.5 0.5 0.1 0.2\n"
            "black-king 0.5 0.2 0.1 0.2\n"
            "# comment line\n"
            "unknown_piece 0.1 0.1 0.1 0.1\n"
            "0 0.5 0.5 0.0001 0.0001\n",  # degenerate
            encoding="utf-8",
        )
        stats = SanitizationStats()
        annotations = standardizer.parse_yolo_file(test_file, stats=stats)
        assert len(annotations) == 2
        assert annotations[0].class_id == 5  # white_king
        assert annotations[1].class_id == 11  # black_king
        assert stats.discarded_unknown_class == 1
        assert stats.discarded_degenerate == 1

    def test_parse_coco_annotations(self, standardizer: AnnotationStandardizer) -> None:
        raw_coco = [
            {"category_id": 1, "bbox": [100, 200, 60, 120]},  # ChessReD 1 -> white_pawn (0)
            {"category_id": 12, "bbox": [500, 500, 80, 160]},  # ChessReD 12 -> black_king (11)
        ]
        stats = SanitizationStats()
        anns = standardizer.parse_coco_annotations(raw_coco, img_w=1000, img_h=1000, dataset_source="chessred", stats=stats)
        assert len(anns) == 2
        assert anns[0].class_id == 0
        assert anns[1].class_id == 11
        assert stats.valid_annotations == 2

    def test_parse_pascal_voc_xml(self, standardizer: AnnotationStandardizer, tmp_path: Path) -> None:
        xml_file = tmp_path / "sample.xml"
        xml_content = """<annotation>
            <size>
                <width>800</width>
                <height>800</height>
            </size>
            <object>
                <name>white_rook</name>
                <bndbox>
                    <xmin>100</xmin>
                    <ymin>100</ymin>
                    <xmax>200</xmax>
                    <ymax>300</ymax>
                </bndbox>
            </object>
            <object>
                <name>black_bishop</name>
                <bndbox>
                    <xmin>400</xmin>
                    <ymin>400</ymin>
                    <xmax>500</xmax>
                    <ymax>600</ymax>
                </bndbox>
            </object>
        </annotation>"""
        xml_file.write_text(xml_content, encoding="utf-8")

        stats = SanitizationStats()
        anns = standardizer.parse_pascal_voc_xml(xml_file, stats=stats)
        assert len(anns) == 2
        assert anns[0].class_id == 3  # white_rook
        assert anns[1].class_id == 8  # black_bishop
        assert stats.valid_annotations == 2


class TestDatasetLabelNormalizer:
    """Tests for DatasetLabelNormalizer directory batch processing."""

    def test_normalize_yolo_directory(self, tmp_path: Path) -> None:
        src_dir = tmp_path / "raw_labels"
        dst_dir = tmp_path / "std_labels"
        src_dir.mkdir()

        # Create file with Roboflow alphabetical class names
        (src_dir / "img1.txt").write_text(
            "white-pawn 0.5 0.5 0.1 0.2\n"
            "black-queen 0.3 0.3 0.1 0.2\n",
            encoding="utf-8",
        )
        # Create negative background image (empty label file)
        (src_dir / "img_empty.txt").write_text("", encoding="utf-8")

        normalizer = DatasetLabelNormalizer()
        stats = normalizer.normalize_yolo_directory(src_dir, dst_dir)

        assert stats.valid_annotations == 2
        assert (dst_dir / "img1.txt").exists()
        assert (dst_dir / "img_empty.txt").exists()

        out_lines = (dst_dir / "img1.txt").read_text(encoding="utf-8").strip().splitlines()
        assert len(out_lines) == 2
        assert out_lines[0].startswith("0 ")  # white_pawn
        assert out_lines[1].startswith("10 ")  # black_queen
