"""
Typed Data Contracts for the Unified 2D/3D Chess Vision & Recommendation Pipeline.

Enforces strict Pydantic v2 schemas across all modular pipeline boundaries:
Vision Detection -> Geometric Homography -> FEN Synthesis -> Engine Evaluation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PieceType(str, Enum):
    PAWN = "pawn"
    KNIGHT = "knight"
    BISHOP = "bishop"
    ROOK = "rook"
    QUEEN = "queen"
    KING = "king"


class PieceColor(str, Enum):
    WHITE = "white"
    BLACK = "black"


class DomainType(str, Enum):
    DIGITAL_2D = "digital_2d"
    PHYSICAL_3D = "physical_3d"


class Point2D(BaseModel):
    """Represents a 2D subpixel coordinate (x, y) on the image plane."""

    model_config = ConfigDict(frozen=True)

    x: float = Field(..., description="Horizontal coordinate on the image plane (pixels)")
    y: float = Field(..., description="Vertical coordinate on the image plane (pixels)")

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def as_int_tuple(self) -> tuple[int, int]:
        return (int(round(self.x)), int(round(self.y)))


class BoundingBox(BaseModel):
    """Axis-aligned bounding box with confidence score."""

    model_config = ConfigDict(frozen=True)

    x_min: float = Field(..., ge=0.0, description="Top-left x coordinate")
    y_min: float = Field(..., ge=0.0, description="Top-left y coordinate")
    x_max: float = Field(..., ge=0.0, description="Bottom-right x coordinate")
    y_max: float = Field(..., ge=0.0, description="Bottom-right y coordinate")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Detection confidence")

    @model_validator(mode="after")
    def validate_box_dimensions(self) -> BoundingBox:
        if self.x_max < self.x_min:
            raise ValueError(f"x_max ({self.x_max}) cannot be less than x_min ({self.x_min})")
        if self.y_max < self.y_min:
            raise ValueError(f"y_max ({self.y_max}) cannot be less than y_min ({self.y_min})")
        return self

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def center(self) -> Point2D:
        return Point2D(x=(self.x_min + self.x_max) / 2.0, y=(self.y_min + self.y_max) / 2.0)

    @property
    def bottom_center(self) -> Point2D:
        """Bottom-center anchor point used for parallax-free board square mapping."""
        return Point2D(x=(self.x_min + self.x_max) / 2.0, y=self.y_max)


class PieceDetection(BaseModel):
    """Represents a localized and classified chess piece on the board."""

    model_config = ConfigDict(frozen=True)

    piece_type: PieceType
    color: PieceColor
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox
    base_point: Point2D = Field(
        ...,
        description="Bottom contact point on the board plane for square assignment",
    )

    @property
    def fen_symbol(self) -> str:
        """Single-character FEN piece representation (uppercase for White, lowercase for Black)."""
        mapping = {
            PieceType.PAWN: "p",
            PieceType.KNIGHT: "n",
            PieceType.BISHOP: "b",
            PieceType.ROOK: "r",
            PieceType.QUEEN: "q",
            PieceType.KING: "k",
        }
        sym = mapping[self.piece_type]
        return sym.upper() if self.color == PieceColor.WHITE else sym


class BoardCorners(BaseModel):
    """Four ordered corner coordinates of the physical or digital chessboard."""

    model_config = ConfigDict(frozen=True)

    top_left: Point2D
    top_right: Point2D
    bottom_right: Point2D
    bottom_left: Point2D
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    def as_list(self) -> list[tuple[float, float]]:
        return [
            self.top_left.as_tuple(),
            self.top_right.as_tuple(),
            self.bottom_right.as_tuple(),
            self.bottom_left.as_tuple(),
        ]


class BoardSquare(BaseModel):
    """Represents one of the 64 squares on an 8x8 chessboard."""

    name: str = Field(..., description="Square algebraic notation e.g. 'e4', 'a1'")
    file_idx: int = Field(..., ge=0, le=7, description="File index 0 (a) to 7 (h)")
    rank_idx: int = Field(..., ge=0, le=7, description="Rank index 0 (1) to 7 (8)")
    piece: PieceDetection | None = None

    @field_validator("name")
    @classmethod
    def validate_algebraic_name(cls, v: str) -> str:
        v_clean = v.strip().lower()
        if len(v_clean) != 2 or v_clean[0] not in "abcdefgh" or v_clean[1] not in "12345678":
            raise ValueError(f"Invalid algebraic square notation: '{v}'")
        return v_clean


class BoardStateResult(BaseModel):
    """Complete synthesized board state produced by the vision-to-FEN pipeline."""

    fen: str = Field(..., description="Complete Forsyth-Edwards Notation string (placement part or full)")
    is_legal: bool = Field(default=True, description="Whether the synthesized position is legal under chess rules")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall pipeline confidence")
    domain: DomainType = Field(..., description="Source image domain (2D screenshot or 3D camera)")
    raw_detections: list[PieceDetection] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)
    grid_occupancy: list[list[str | None]] = Field(
        default_factory=list,
        description="8x8 board matrix representation [rank 8 -> rank 1][file a -> file h]",
    )


class EngineEvaluation(BaseModel):
    """Evaluation result generated by the Stockfish UCI engine."""

    best_move_uci: str = Field(..., description="Best move in UCI format e.g. 'e2e4'")
    best_move_san: str = Field(default="", description="Best move in Standard Algebraic Notation e.g. 'e4'")
    score_cp: int | None = Field(default=None, description="Score in centipawns from current player perspective")
    score_mate: int | None = Field(default=None, description="Mate in N moves (positive=winning, negative=losing)")
    depth: int = Field(..., ge=1, description="Stockfish search depth reached")
    ponder_move_uci: str | None = Field(default=None, description="Ponder move if available")
    pv: list[str] = Field(default_factory=list, description="Principal variation move sequence")

    @property
    def formatted_score(self) -> str:
        if self.score_mate is not None:
            return f"M{self.score_mate}"
        if self.score_cp is not None:
            return f"{self.score_cp / 100.0:+.2f}"
        return "0.00"
