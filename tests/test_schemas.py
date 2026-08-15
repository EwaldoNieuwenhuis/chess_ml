"""
Unit tests for Pydantic v2 data contracts in src.schemas.contracts.
"""

import pytest
from pydantic import ValidationError

from src.schemas.contracts import (
    BoardCorners,
    BoardSquare,
    BoardStateResult,
    BoundingBox,
    DomainType,
    EngineEvaluation,
    PieceColor,
    PieceDetection,
    PieceType,
    Point2D,
)


def test_point2d_creation_and_tuples() -> None:
    pt = Point2D(x=120.45, y=340.89)
    assert pt.x == 120.45
    assert pt.y == 340.89
    assert pt.as_tuple() == (120.45, 340.89)
    assert pt.as_int_tuple() == (120, 341)


def test_bounding_box_properties_and_anchors() -> None:
    box = BoundingBox(x_min=100.0, y_min=200.0, x_max=150.0, y_max=300.0, confidence=0.95)
    assert box.width == 50.0
    assert box.height == 100.0
    assert box.center == Point2D(x=125.0, y=250.0)
    assert box.bottom_center == Point2D(x=125.0, y=300.0)


def test_bounding_box_validation_error() -> None:
    with pytest.raises(ValidationError):
        # x_max < x_min is invalid
        BoundingBox(x_min=200.0, y_min=100.0, x_max=150.0, y_max=300.0)

    with pytest.raises(ValidationError):
        # negative coordinate
        BoundingBox(x_min=-10.0, y_min=0.0, x_max=100.0, y_max=100.0)


def test_piece_detection_fen_symbols() -> None:
    box = BoundingBox(x_min=10.0, y_min=10.0, x_max=50.0, y_max=50.0)
    base = Point2D(x=30.0, y=50.0)

    white_queen = PieceDetection(
        piece_type=PieceType.QUEEN,
        color=PieceColor.WHITE,
        confidence=0.98,
        bbox=box,
        base_point=base,
    )
    assert white_queen.fen_symbol == "Q"

    black_knight = PieceDetection(
        piece_type=PieceType.KNIGHT,
        color=PieceColor.BLACK,
        confidence=0.92,
        bbox=box,
        base_point=base,
    )
    assert black_knight.fen_symbol == "n"


def test_board_square_validation() -> None:
    sq = BoardSquare(name="e4", file_idx=4, rank_idx=3)
    assert sq.name == "e4"

    with pytest.raises(ValidationError):
        BoardSquare(name="z9", file_idx=0, rank_idx=0)


def test_board_corners_conversion() -> None:
    corners = BoardCorners(
        top_left=Point2D(x=10.0, y=10.0),
        top_right=Point2D(x=790.0, y=10.0),
        bottom_right=Point2D(x=790.0, y=790.0),
        bottom_left=Point2D(x=10.0, y=790.0),
    )
    corner_list = corners.as_list()
    assert len(corner_list) == 4
    assert corner_list[0] == (10.0, 10.0)
    assert corner_list[2] == (790.0, 790.0)


def test_board_state_result_and_engine_eval() -> None:
    result = BoardStateResult(
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        is_legal=True,
        confidence=0.99,
        domain=DomainType.PHYSICAL_3D,
    )
    assert result.is_legal is True
    assert result.domain == DomainType.PHYSICAL_3D

    eval_result = EngineEvaluation(
        best_move_uci="e2e4",
        best_move_san="e4",
        score_cp=35,
        depth=20,
    )
    assert eval_result.formatted_score == "+0.35"

    eval_mate = EngineEvaluation(
        best_move_uci="d8h4",
        score_mate=2,
        depth=15,
    )
    assert eval_mate.formatted_score == "M2"
