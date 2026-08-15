"""
Canonical Class Name & Bounding Box Coordinate Standardizer.

Standardizes heterogeneous class naming schemes (e.g. ['wP', 'bK'], ['white-queen', 'black-rook'],
['W_P', 'B_K'], [0..11]) and coordinate formats (COCO, Pascal VOC, YOLO) into the project's
canonical 12-class schema in normalized YOLO format with epsilon boundary sanitization (ADR-008).
"""

from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import yaml

from src.schemas.contracts import PieceColor, PieceType

logger = logging.getLogger(__name__)

DEFAULT_CANONICAL_CONFIG_PATH = Path("configs/dataset/canonical_classes.yaml")


class NormalizerError(Exception):
    """Base exception for dataset normalization errors."""


class UnknownClassError(NormalizerError):
    """Raised when an unrecognized class label is encountered."""


class InvalidCoordinateError(NormalizerError):
    """Raised when bounding box coordinates cannot be validated or sanitized."""


@dataclass(frozen=True)
class CanonicalClassInfo:
    """Metadata for a canonical chess piece class."""

    class_id: int
    name: str
    piece_type: PieceType
    color: PieceColor
    fen_char: str

    @property
    def is_white(self) -> bool:
        return self.color == PieceColor.WHITE

    @property
    def is_black(self) -> bool:
        return self.color == PieceColor.BLACK


class CanonicalClassMapper:
    """
    Translates heterogeneous class naming conventions into project-standard 12 classes.

    Canonical Schema (Grouped Indexing):
      0: white_pawn    1: white_knight  2: white_bishop
      3: white_rook    4: white_queen   5: white_king
      6: black_pawn    7: black_knight  8: black_bishop
      9: black_rook    10: black_queen  11: black_king

    Mathematical Properties:
      - color = WHITE if class_id < 6 else BLACK
      - piece_type = class_id % 6 (0: Pawn, 1: Knight, 2: Bishop, 3: Rook, 4: Queen, 5: King)
    """

    _PIECE_ORDER: tuple[PieceType, ...] = (
        PieceType.PAWN,
        PieceType.KNIGHT,
        PieceType.BISHOP,
        PieceType.ROOK,
        PieceType.QUEEN,
        PieceType.KING,
    )

    def __init__(self, config_path: Path | str | None = None) -> None:
        self.config_path = Path(config_path) if config_path else DEFAULT_CANONICAL_CONFIG_PATH
        self._classes: dict[int, CanonicalClassInfo] = {}
        self._name_to_info: dict[str, CanonicalClassInfo] = {}
        self._fen_to_info: dict[str, CanonicalClassInfo] = {}
        self._aliases: dict[str, str] = {}
        self._dataset_mappings: dict[str, dict[Any, str]] = {}
        self._sanitization_config: dict[str, float] = {
            "epsilon": 1e-5,
            "min_dimension": 0.005,
            "min_area": 0.000025,
            "min_visibility_ratio": 0.40,
        }

        self._load_config()

    def _load_config(self) -> None:
        """Loads canonical classes, aliases, and dataset-specific mappings from YAML."""
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                raw_data = yaml.safe_load(f) or {}
        else:
            raw_data = {}

        # 1. Populate canonical 12 classes
        canonical_raw = raw_data.get("canonical_classes", {})
        if canonical_raw:
            for cid_key, cdata in canonical_raw.items():
                cid = int(cid_key)
                info = CanonicalClassInfo(
                    class_id=cid,
                    name=cdata["name"],
                    piece_type=PieceType(cdata["piece_type"]),
                    color=PieceColor(cdata["color"]),
                    fen_char=cdata["fen_char"],
                )
                self._classes[cid] = info
                self._name_to_info[info.name] = info
                self._fen_to_info[info.fen_char] = info
        else:
            # Hardcoded standard fallback
            self._init_default_classes()

        # 2. Populate aliases
        self._aliases = {str(k): str(v) for k, v in raw_data.get("aliases", {}).items()}

        # 3. Populate dataset mappings
        for ds_name, mapping in raw_data.get("dataset_mappings", {}).items():
            self._dataset_mappings[ds_name] = {k: str(v) for k, v in mapping.items()}

        # 4. Sanitization thresholds
        if "sanitization" in raw_data:
            self._sanitization_config.update(raw_data["sanitization"])

    def _init_default_classes(self) -> None:
        """Initialize standard 12 classes programmatically if config file is absent."""
        fen_chars_white = ["P", "N", "B", "R", "Q", "K"]
        fen_chars_black = ["p", "n", "b", "r", "q", "k"]

        for i, ptype in enumerate(self._PIECE_ORDER):
            # White (0..5)
            w_info = CanonicalClassInfo(
                class_id=i,
                name=f"white_{ptype.value}",
                piece_type=ptype,
                color=PieceColor.WHITE,
                fen_char=fen_chars_white[i],
            )
            self._classes[i] = w_info
            self._name_to_info[w_info.name] = w_info
            self._fen_to_info[w_info.fen_char] = w_info

            # Black (6..11)
            b_info = CanonicalClassInfo(
                class_id=i + 6,
                name=f"black_{ptype.value}",
                piece_type=ptype,
                color=PieceColor.BLACK,
                fen_char=fen_chars_black[i],
            )
            self._classes[i + 6] = b_info
            self._name_to_info[b_info.name] = b_info
            self._fen_to_info[b_info.fen_char] = b_info

    @property
    def sanitization_config(self) -> dict[str, float]:
        return self._sanitization_config.copy()

    def get_class_info(self, class_id: int) -> CanonicalClassInfo:
        """Retrieve canonical class metadata by class ID (0..11)."""
        if class_id not in self._classes:
            raise UnknownClassError(f"Invalid canonical class ID: {class_id}. Expected integer in 0..11.")
        return self._classes[class_id]

    def get_canonical_names(self) -> dict[int, str]:
        """Returns dict of {class_id: class_name} for YOLO data.yaml generation."""
        return {cid: info.name for cid, info in sorted(self._classes.items())}

    def map_class(
        self,
        raw_label: str | int,
        dataset_source: str | None = None,
    ) -> CanonicalClassInfo:
        """
        Maps an arbitrary raw label (string, abbreviation, FEN char, or dataset-specific index)
        to a canonical class info object.

        Args:
            raw_label: Input label (e.g. 'wP', 'black-rook', 'W_Q', 0, 1)
            dataset_source: Optional dataset key (e.g. 'chessred', 'roboflow_staunton')
        Returns:
            CanonicalClassInfo
        Raises:
            UnknownClassError if label cannot be mapped.
        """
        # 1. Dataset-specific mapping lookup if provided
        if dataset_source and dataset_source in self._dataset_mappings:
            ds_map = self._dataset_mappings[dataset_source]
            if raw_label in ds_map:
                target_name = ds_map[raw_label]
                if target_name in self._name_to_info:
                    return self._name_to_info[target_name]
            elif isinstance(raw_label, int) and raw_label in ds_map:
                target_name = ds_map[raw_label]
                if target_name in self._name_to_info:
                    return self._name_to_info[target_name]
            elif str(raw_label) in ds_map:
                target_name = ds_map[str(raw_label)]
                if target_name in self._name_to_info:
                    return self._name_to_info[target_name]

        # 2. Direct integer check (assumed already canonical 0..11 if no dataset source specified)
        if isinstance(raw_label, int):
            if raw_label in self._classes:
                return self._classes[raw_label]
            raise UnknownClassError(f"Integer class index {raw_label} is outside canonical range 0..11.")

        # 3. String lookup
        label_str = str(raw_label).strip()

        # Case-sensitive FEN character match (e.g. 'P' vs 'p')
        if label_str in self._fen_to_info:
            return self._fen_to_info[label_str]

        # Check aliases dictionary (exact match first)
        if label_str in self._aliases:
            canon_name = self._aliases[label_str]
            if canon_name in self._name_to_info:
                return self._name_to_info[canon_name]

        # Check canonical names directly
        if label_str in self._name_to_info:
            return self._name_to_info[label_str]

        # Normalized string variations
        norm_key = label_str.lower().replace("-", "_").replace(" ", "_")
        if norm_key in self._name_to_info:
            return self._name_to_info[norm_key]
        if norm_key in self._aliases:
            canon_name = self._aliases[norm_key]
            if canon_name in self._name_to_info:
                return self._name_to_info[canon_name]

        # Short notation handling: e.g. 'wp' -> 'white_pawn', 'bq' -> 'black_queen'
        if len(norm_key) == 2:
            color_prefix = norm_key[0]
            piece_code = norm_key[1]
            if color_prefix in ("w", "b"):
                color_name = "white" if color_prefix == "w" else "black"
                piece_map = {"p": "pawn", "n": "knight", "b": "bishop", "r": "rook", "q": "queen", "k": "king"}
                if piece_code in piece_map:
                    candidate = f"{color_name}_{piece_map[piece_code]}"
                    if candidate in self._name_to_info:
                        return self._name_to_info[candidate]

        raise UnknownClassError(
            f"Unable to map raw label '{raw_label}' (source='{dataset_source}') to a canonical chess piece class."
        )

    def to_fen_char(self, class_id: int) -> str:
        """Convert canonical class ID to single FEN character."""
        return self.get_class_info(class_id).fen_char

    def from_fen_char(self, fen_char: str) -> CanonicalClassInfo:
        """Convert single FEN character (e.g. 'N', 'q') to canonical class info."""
        if fen_char not in self._fen_to_info:
            raise UnknownClassError(f"Invalid FEN piece character: '{fen_char}'")
        return self._fen_to_info[fen_char]

    def to_piece_type_and_color(self, class_id: int) -> tuple[PieceType, PieceColor]:
        """
        Decompose class ID into (PieceType, PieceColor) using grouped indexing properties.
        """
        info = self.get_class_info(class_id)
        return (info.piece_type, info.color)

    def from_piece_type_and_color(self, piece_type: PieceType | str, color: PieceColor | str) -> int:
        """
        Calculate canonical class ID from PieceType and PieceColor.
        """
        pt = PieceType(piece_type) if isinstance(piece_type, str) else piece_type
        col = PieceColor(color) if isinstance(color, str) else color

        piece_idx = self._PIECE_ORDER.index(pt)
        return piece_idx if col == PieceColor.WHITE else piece_idx + 6


@dataclass(frozen=True)
class NormalizedBBox:
    """
    Standardized bounding box in normalized YOLO format: (x_center, y_center, width, height) in [0.0, 1.0].
    """

    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float = 1.0

    @property
    def x_min(self) -> float:
        return self.x_center - self.width / 2.0

    @property
    def y_min(self) -> float:
        return self.y_center - self.height / 2.0

    @property
    def x_max(self) -> float:
        return self.x_center + self.width / 2.0

    @property
    def y_max(self) -> float:
        return self.y_center + self.height / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Parallax-free bottom-center footprint contact point."""
        return (self.x_center, self.y_max)

    @classmethod
    def from_normalized_yolo(
        cls,
        x_center: float,
        y_center: float,
        width: float,
        height: float,
        confidence: float = 1.0,
    ) -> NormalizedBBox:
        """Construct from normalized YOLO coordinates."""
        return cls(
            x_center=float(x_center),
            y_center=float(y_center),
            width=float(width),
            height=float(height),
            confidence=float(confidence),
        )

    @classmethod
    def from_normalized_xyxy(
        cls,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        confidence: float = 1.0,
    ) -> NormalizedBBox:
        """Construct from normalized [x_min, y_min, x_max, y_max] in [0, 1]."""
        w = max(0.0, float(x_max) - float(x_min))
        h = max(0.0, float(y_max) - float(y_min))
        xc = float(x_min) + w / 2.0
        yc = float(y_min) + h / 2.0
        return cls(x_center=xc, y_center=yc, width=w, height=h, confidence=float(confidence))

    @classmethod
    def from_xyxy_pixels(
        cls,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        img_w: int,
        img_h: int,
        confidence: float = 1.0,
    ) -> NormalizedBBox:
        """Construct from Pascal VOC / pixel [x_min, y_min, x_max, y_max]."""
        if img_w <= 0 or img_h <= 0:
            raise InvalidCoordinateError(f"Image dimensions must be positive, got {img_w}x{img_h}.")

        x_min_n = float(x_min) / img_w
        y_min_n = float(y_min) / img_h
        x_max_n = float(x_max) / img_w
        y_max_n = float(y_max) / img_h
        return cls.from_normalized_xyxy(x_min_n, y_min_n, x_max_n, y_max_n, confidence=confidence)

    @classmethod
    def from_coco_pixels(
        cls,
        x_min: float,
        y_min: float,
        width: float,
        height: float,
        img_w: int,
        img_h: int,
        confidence: float = 1.0,
    ) -> NormalizedBBox:
        """Construct from COCO [x_min, y_min, width, height] in pixel units."""
        if img_w <= 0 or img_h <= 0:
            raise InvalidCoordinateError(f"Image dimensions must be positive, got {img_w}x{img_h}.")

        x_min_n = float(x_min) / img_w
        y_min_n = float(y_min) / img_h
        w_n = float(width) / img_w
        h_n = float(height) / img_h
        xc = x_min_n + w_n / 2.0
        yc = y_min_n + h_n / 2.0
        return cls(x_center=xc, y_center=yc, width=w_n, height=h_n, confidence=float(confidence))

    def clamp(
        self,
        epsilon: float = 1e-5,
        min_visibility_ratio: float = 0.40,
    ) -> tuple[NormalizedBBox | None, bool]:
        """
        Enforces epsilon boundary clamping (ADR-008).
        Clamps coordinates to [0.0, 1.0] if within epsilon tolerance or slightly out of bounds.

        If the box intersects the frame boundary:
          - If visible area >= min_visibility_ratio, clamps coordinates to [0.0, 1.0].
          - If visible area < min_visibility_ratio, returns (None, False) (discard box).

        Returns:
            (sanitized_bbox, was_clamped)
        """
        orig_x_min = self.x_min
        orig_y_min = self.y_min
        orig_x_max = self.x_max
        orig_y_max = self.y_max
        orig_area = max(1e-12, (orig_x_max - orig_x_min) * (orig_y_max - orig_y_min))

        # Check if box is completely outside the frame
        if orig_x_max <= -epsilon or orig_x_min >= 1.0 + epsilon:
            return (None, False)
        if orig_y_max <= -epsilon or orig_y_min >= 1.0 + epsilon:
            return (None, False)

        # Clamped coordinates
        clamped_x_min = max(0.0, min(1.0, orig_x_min))
        clamped_y_min = max(0.0, min(1.0, orig_y_min))
        clamped_x_max = max(0.0, min(1.0, orig_x_max))
        clamped_y_max = max(0.0, min(1.0, orig_y_max))

        clamped_w = clamped_x_max - clamped_x_min
        clamped_h = clamped_y_max - clamped_y_min
        clamped_area = clamped_w * clamped_h

        # Visibility ratio test
        visibility = clamped_area / orig_area
        if visibility < min_visibility_ratio:
            return (None, False)

        was_clamped = (
            orig_x_min < 0.0
            or orig_y_min < 0.0
            or orig_x_max > 1.0
            or orig_y_max > 1.0
            or abs(clamped_x_min - orig_x_min) > 1e-9
            or abs(clamped_y_min - orig_y_min) > 1e-9
            or abs(clamped_x_max - orig_x_max) > 1e-9
            or abs(clamped_y_max - orig_y_max) > 1e-9
        )

        clamped_box = NormalizedBBox.from_normalized_xyxy(
            clamped_x_min,
            clamped_y_min,
            clamped_x_max,
            clamped_y_max,
            confidence=self.confidence,
        )
        return (clamped_box, was_clamped)

    def is_valid(
        self,
        min_dim: float = 0.005,
        min_area: float = 0.000025,
    ) -> bool:
        """
        Validates that coordinates are within [0.0, 1.0] and not degenerate.
        """
        if math.isnan(self.x_center) or math.isnan(self.y_center) or math.isnan(self.width) or math.isnan(self.height):
            return False
        if self.width < min_dim or self.height < min_dim:
            return False
        if self.area < min_area:
            return False
        if not (0.0 <= self.x_min <= 1.0 and 0.0 <= self.x_max <= 1.0):
            return False
        if not (0.0 <= self.y_min <= 1.0 and 0.0 <= self.y_max <= 1.0):
            return False
        return True

    def to_yolo_tuple(self) -> tuple[float, float, float, float]:
        """Returns (x_center, y_center, width, height)."""
        return (self.x_center, self.y_center, self.width, self.height)

    def to_xyxy_pixels(self, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        """Converts back to absolute pixel coordinates [x_min, y_min, x_max, y_max]."""
        return (
            self.x_min * img_w,
            self.y_min * img_h,
            self.x_max * img_w,
            self.y_max * img_h,
        )


@dataclass(frozen=True)
class StandardizedAnnotation:
    """A standardized piece annotation ready for YOLO export."""

    class_id: int
    class_name: str
    bbox: NormalizedBBox
    confidence: float = 1.0

    def to_yolo_line(self, precision: int = 6) -> str:
        """Generates standard YOLO txt line: 'class_id x_c y_c w h'."""
        xc = f"{self.bbox.x_center:.{precision}f}"
        yc = f"{self.bbox.y_center:.{precision}f}"
        w = f"{self.bbox.width:.{precision}f}"
        h = f"{self.bbox.height:.{precision}f}"
        return f"{self.class_id} {xc} {yc} {w} {h}"


@dataclass
class SanitizationStats:
    """Metrics recorded during annotation standardization and sanitization."""

    total_annotations: int = 0
    valid_annotations: int = 0
    clamped_annotations: int = 0
    discarded_degenerate: int = 0
    discarded_out_of_bounds: int = 0
    discarded_unknown_class: int = 0
    class_counts: dict[int, int] = field(default_factory=dict)

    def register_valid(self, class_id: int, was_clamped: bool) -> None:
        self.total_annotations += 1
        self.valid_annotations += 1
        if was_clamped:
            self.clamped_annotations += 1
        self.class_counts[class_id] = self.class_counts.get(class_id, 0) + 1

    def register_degenerate(self) -> None:
        self.total_annotations += 1
        self.discarded_degenerate += 1

    def register_out_of_bounds(self) -> None:
        self.total_annotations += 1
        self.discarded_out_of_bounds += 1

    def register_unknown_class(self) -> None:
        self.total_annotations += 1
        self.discarded_unknown_class += 1

    def merge(self, other: SanitizationStats) -> None:
        self.total_annotations += other.total_annotations
        self.valid_annotations += other.valid_annotations
        self.clamped_annotations += other.clamped_annotations
        self.discarded_degenerate += other.discarded_degenerate
        self.discarded_out_of_bounds += other.discarded_out_of_bounds
        self.discarded_unknown_class += other.discarded_unknown_class
        for cid, count in other.class_counts.items():
            self.class_counts[cid] = self.class_counts.get(cid, 0) + count


class AnnotationStandardizer:
    """
    Standardizes single annotations and label files from various source formats
    (YOLO txt, COCO json, Pascal VOC XML) into validated canonical YOLO format.
    """

    def __init__(self, mapper: CanonicalClassMapper | None = None) -> None:
        self.mapper = mapper or CanonicalClassMapper()
        self.sanitization = self.mapper.sanitization_config

    def standardize_box(
        self,
        raw_label: str | int,
        bbox: NormalizedBBox,
        dataset_source: str | None = None,
        stats: SanitizationStats | None = None,
    ) -> StandardizedAnnotation | None:
        """
        Sanitizes bounding box and maps class label to canonical schema.
        """
        # 1. Map class
        try:
            class_info = self.mapper.map_class(raw_label, dataset_source=dataset_source)
        except UnknownClassError as e:
            logger.debug("Skipping non-canonical class: %s", e)
            if stats:
                stats.register_unknown_class()
            return None

        # 2. Clamp bounding box
        clamped_box, was_clamped = bbox.clamp(
            epsilon=self.sanitization.get("epsilon", 1e-5),
            min_visibility_ratio=self.sanitization.get("min_visibility_ratio", 0.40),
        )

        if clamped_box is None:
            if stats:
                stats.register_out_of_bounds()
            return None

        # 3. Check for degenerate dimensions
        if not clamped_box.is_valid(
            min_dim=self.sanitization.get("min_dimension", 0.005),
            min_area=self.sanitization.get("min_area", 0.000025),
        ):
            if stats:
                stats.register_degenerate()
            return None

        if stats:
            stats.register_valid(class_info.class_id, was_clamped)

        return StandardizedAnnotation(
            class_id=class_info.class_id,
            class_name=class_info.name,
            bbox=clamped_box,
            confidence=bbox.confidence,
        )

    def parse_yolo_line(
        self,
        line: str,
        dataset_source: str | None = None,
        stats: SanitizationStats | None = None,
    ) -> StandardizedAnnotation | None:
        """
        Parses a single YOLO text line: 'class_id x_c y_c w h [conf]'.
        """
        parts = line.strip().split()
        if len(parts) < 5:
            return None

        raw_cls = parts[0]
        try:
            # Check if integer ID
            if raw_cls.isdigit() or (raw_cls.startswith("-") and raw_cls[1:].isdigit()):
                cls_val: str | int = int(raw_cls)
            else:
                cls_val = raw_cls
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            conf = float(parts[5]) if len(parts) > 5 else 1.0
        except ValueError as e:
            logger.warning("Failed to parse YOLO line '%s': %s", line, e)
            return None

        raw_bbox = NormalizedBBox.from_normalized_yolo(xc, yc, w, h, confidence=conf)
        return self.standardize_box(cls_val, raw_bbox, dataset_source=dataset_source, stats=stats)

    def parse_yolo_file(
        self,
        file_path: Path | str,
        dataset_source: str | None = None,
        stats: SanitizationStats | None = None,
    ) -> list[StandardizedAnnotation]:
        """
        Parses a complete YOLO format annotation file (.txt).
        """
        path = Path(file_path)
        if not path.exists():
            return []

        results: list[StandardizedAnnotation] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith("#"):
                    continue
                ann = self.parse_yolo_line(line_str, dataset_source=dataset_source, stats=stats)
                if ann is not None:
                    results.append(ann)
        return results

    def parse_coco_annotations(
        self,
        annotations: Sequence[dict[str, Any]],
        img_w: int,
        img_h: int,
        dataset_source: str | None = "chessred",
        stats: SanitizationStats | None = None,
    ) -> list[StandardizedAnnotation]:
        """
        Parses COCO bounding box annotations for an image.
        Each annotation dict must contain 'category_id' and 'bbox': [x_min, y_min, width, height] in pixels.
        """
        results: list[StandardizedAnnotation] = []
        for ann in annotations:
            cat_id = ann.get("category_id")
            bbox_raw = ann.get("bbox")
            if cat_id is None or not bbox_raw or len(bbox_raw) < 4:
                continue

            x_min, y_min, w_pix, h_pix = bbox_raw[:4]
            conf = float(ann.get("score", 1.0))
            raw_bbox = NormalizedBBox.from_coco_pixels(x_min, y_min, w_pix, h_pix, img_w, img_h, confidence=conf)
            std_ann = self.standardize_box(cat_id, raw_bbox, dataset_source=dataset_source, stats=stats)
            if std_ann is not None:
                results.append(std_ann)
        return results

    def parse_pascal_voc_xml(
        self,
        xml_path: Path | str,
        dataset_source: str | None = "kaggle_tripod",
        stats: SanitizationStats | None = None,
    ) -> list[StandardizedAnnotation]:
        """
        Parses a Pascal VOC format XML file (<annotation><size>...<object>...).
        """
        path = Path(xml_path)
        if not path.exists():
            return []

        tree = ET.parse(path)
        root = tree.getroot()

        size_elem = root.find("size")
        if size_elem is None:
            raise InvalidCoordinateError(f"Pascal VOC XML {path} lacks <size> element.")

        img_w = int(size_elem.findtext("width", "0"))
        img_h = int(size_elem.findtext("height", "0"))
        if img_w <= 0 or img_h <= 0:
            raise InvalidCoordinateError(f"Invalid image dimensions in {path}: {img_w}x{img_h}")

        results: list[StandardizedAnnotation] = []
        for obj in root.findall("object"):
            name = obj.findtext("name", "").strip()
            bndbox = obj.find("bndbox")
            if not name or bndbox is None:
                continue

            x_min = float(bndbox.findtext("xmin", "0"))
            y_min = float(bndbox.findtext("ymin", "0"))
            x_max = float(bndbox.findtext("xmax", "0"))
            y_max = float(bndbox.findtext("ymax", "0"))

            raw_bbox = NormalizedBBox.from_xyxy_pixels(x_min, y_min, x_max, y_max, img_w, img_h)
            std_ann = self.standardize_box(name, raw_bbox, dataset_source=dataset_source, stats=stats)
            if std_ann is not None:
                results.append(std_ann)

        return results


class DatasetLabelNormalizer:
    """
    Batch processor that normalizes an entire raw dataset directory into canonical YOLO label files.
    """

    def __init__(self, standardizer: AnnotationStandardizer | None = None) -> None:
        self.standardizer = standardizer or AnnotationStandardizer()

    def normalize_yolo_directory(
        self,
        source_labels_dir: Path | str,
        target_labels_dir: Path | str,
        dataset_source: str | None = None,
    ) -> SanitizationStats:
        """
        Batch standardizes a directory of raw YOLO .txt files.
        """
        src = Path(source_labels_dir)
        dst = Path(target_labels_dir)
        dst.mkdir(parents=True, exist_ok=True)

        stats = SanitizationStats()
        if not src.exists():
            logger.warning("Source labels directory not found: %s", src)
            return stats

        label_files = list(src.glob("*.txt"))
        for txt_file in label_files:
            standardized = self.standardizer.parse_yolo_file(txt_file, dataset_source=dataset_source, stats=stats)
            out_file = dst / txt_file.name
            with open(out_file, "w", encoding="utf-8") as f:
                for ann in standardized:
                    f.write(f"{ann.to_yolo_line()}\n")

        return stats
