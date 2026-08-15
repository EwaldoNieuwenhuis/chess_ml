"""
Automated Dataset Integrity & Corruption Audit Tool (US-2.3.3 / ADR-008).

Provides comprehensive static and statistical validation for standardized and hybrid chess datasets:
1. Asserts 100% of bounding box annotations conform to YOLO contracts:
   - 0 <= class_id <= 11 (integer canonical piece index)
   - Coordinates (x_center, y_center, width, height) in [0.0, 1.0]
   - No NaN or infinite values
   - Strict positive dimensions (w > 0, h > 0)
2. Verifies parallel existence of image and label pairs and flags orphaned files.
3. Asserts empty negative sample images possess 0-byte label files.
4. Validates image file integrity and decodability.
5. Computes class balance distribution histograms and degenerate box rejection rates.
6. Generates structured JSON/Markdown audit reports and supports automated CI exit codes.
"""

from __future__ import annotations

import enum
import json
import logging
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from src.dataset.normalizer import (
    DEFAULT_CANONICAL_CONFIG_PATH,
    CanonicalClassMapper,
)
from src.schemas.contracts import PieceColor, PieceType

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tiff",
)


class ViolationSeverity(enum.Enum):
    """Severity level of an audit violation."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class ViolationType(enum.Enum):
    """Categorized dataset corruption and integrity violation types."""

    INVALID_CLASS_ID = "invalid_class_id"
    OUT_OF_BOUNDS_COORD = "out_of_bounds_coord"
    NAN_OR_INF_VALUE = "nan_or_inf_value"
    NON_POSITIVE_DIMENSION = "non_positive_dimension"
    DEGENERATE_BOX = "degenerate_box"
    MALFORMED_LINE = "malformed_line"
    ORPHAN_IMAGE = "orphan_image"
    ORPHAN_LABEL = "orphan_label"
    CORRUPTED_NEGATIVE = "corrupted_negative"
    CORRUPTED_IMAGE = "corrupted_image"
    MISSING_FILE = "missing_file"


@dataclass
class AnnotationViolation:
    """Represents a specific audit error or warning discovered during dataset inspection."""

    file_path: Path
    line_number: int | None
    violation_type: ViolationType
    severity: ViolationSeverity
    description: str
    raw_content: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": str(self.file_path.as_posix()),
            "line_number": self.line_number,
            "violation_type": self.violation_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "raw_content": self.raw_content,
        }


@dataclass
class AuditConfig:
    """Configuration parameters for dataset validation rules."""

    num_classes: int = 12
    epsilon: float = 1e-5
    min_dimension: float = 0.005
    min_area: float = 0.000025
    supported_image_extensions: tuple[str, ...] = SUPPORTED_IMAGE_EXTENSIONS
    strict_degenerate_as_error: bool = False
    strict_bounds: bool = False
    validate_image_decoding: bool = True
    config_path: Path | str = DEFAULT_CANONICAL_CONFIG_PATH


@dataclass
class FilePairInfo:
    """Represents an image and corresponding label pair discovered during scanning."""

    image_path: Path | None = None
    label_path: Path | None = None
    is_negative: bool = False
    split_name: str = ""


@dataclass
class ClassDistributionSummary:
    """Class frequency distribution metrics."""

    class_id: int
    class_name: str
    piece_type: str
    color: str
    count: int = 0
    percentage: float = 0.0


@dataclass
class DatasetAuditReport:
    """Comprehensive statistical and validation report for an audited dataset."""

    dataset_path: Path
    total_images_scanned: int = 0
    total_labels_scanned: int = 0
    matched_pairs_count: int = 0
    orphaned_images_count: int = 0
    orphaned_labels_count: int = 0
    negative_samples_count: int = 0
    corrupted_images_count: int = 0

    total_boxes_scanned: int = 0
    valid_boxes_count: int = 0
    corrupted_boxes_count: int = 0
    degenerate_boxes_count: int = 0
    clamped_boxes_count: int = 0

    # Class balance distributions
    class_counts: dict[int, int] = field(default_factory=lambda: {i: 0 for i in range(12)})
    white_piece_count: int = 0
    black_piece_count: int = 0
    piece_type_counts: dict[str, int] = field(default_factory=dict)

    # Detailed list of violations
    violations: list[AnnotationViolation] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        """Count of severe errors that fail validation."""
        return sum(1 for v in self.violations if v.severity == ViolationSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        """Count of warnings (e.g. degenerate boxes or non-fatal boundary clamps)."""
        return sum(1 for v in self.violations if v.severity == ViolationSeverity.WARNING)

    @property
    def passed(self) -> bool:
        """Indicates whether the dataset passed validation with 0 fatal errors."""
        return self.error_count == 0

    @property
    def degenerate_box_rejection_rate(self) -> float:
        """Rejection rate of degenerate boxes relative to total scanned boxes."""
        if self.total_boxes_scanned == 0:
            return 0.0
        return (self.degenerate_boxes_count / self.total_boxes_scanned) * 100.0

    @property
    def negative_sample_ratio(self) -> float:
        """Ratio of negative sample images relative to total matched images."""
        if self.matched_pairs_count == 0:
            return 0.0
        return (self.negative_samples_count / self.matched_pairs_count) * 100.0

    def to_dict(self) -> dict[str, Any]:
        """Converts report to dictionary representation for serialization."""
        return {
            "dataset_path": str(self.dataset_path.as_posix()),
            "passed": self.passed,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "pairing_summary": {
                "total_images_scanned": self.total_images_scanned,
                "total_labels_scanned": self.total_labels_scanned,
                "matched_pairs_count": self.matched_pairs_count,
                "orphaned_images_count": self.orphaned_images_count,
                "orphaned_labels_count": self.orphaned_labels_count,
                "negative_samples_count": self.negative_samples_count,
                "negative_sample_ratio_pct": round(self.negative_sample_ratio, 2),
                "corrupted_images_count": self.corrupted_images_count,
            },
            "annotation_summary": {
                "total_boxes_scanned": self.total_boxes_scanned,
                "valid_boxes_count": self.valid_boxes_count,
                "corrupted_boxes_count": self.corrupted_boxes_count,
                "degenerate_boxes_count": self.degenerate_boxes_count,
                "degenerate_box_rejection_rate_pct": round(self.degenerate_box_rejection_rate, 4),
                "clamped_boxes_count": self.clamped_boxes_count,
            },
            "class_distribution": {
                "class_counts": {str(k): v for k, v in self.class_counts.items()},
                "white_piece_count": self.white_piece_count,
                "black_piece_count": self.black_piece_count,
                "piece_type_counts": self.piece_type_counts,
            },
            "violations": [v.to_dict() for v in self.violations],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serializes the report to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self, mapper: CanonicalClassMapper | None = None) -> str:
        """Generates a Markdown summary report."""
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"# Dataset Integrity Audit Report: {self.dataset_path.name}",
            "",
            f"**Overall Status:** `{status}` | **Errors:** `{self.error_count}` | **Warnings:** `{self.warning_count}`",
            "",
            "## 1. Image & Label Pairing Summary",
            "",
            "| Metric | Count | Share / Percentage |",
            "| :--- | :---: | :---: |",
            f"| Total Images Scanned | {self.total_images_scanned} | 100.0% |",
            f"| Total Labels Scanned | {self.total_labels_scanned} | - |",
            f"| Matched Image-Label Pairs | {self.matched_pairs_count} | - |",
            f"| Orphaned Images (Missing Label) | {self.orphaned_images_count} | - |",
            f"| Orphaned Labels (Missing Image) | {self.orphaned_labels_count} | - |",
            f"| 0-Byte Negative Samples | {self.negative_samples_count} | {self.negative_sample_ratio:.2f}% |",
            f"| Corrupted Images | {self.corrupted_images_count} | - |",
            "",
            "## 2. Bounding Box Annotation Summary",
            "",
            "| Metric | Count | Percentage |",
            "| :--- | :---: | :---: |",
            f"| Total Annotations Scanned | {self.total_boxes_scanned} | 100.0% |",
            f"| Valid Bounding Boxes | {self.valid_boxes_count} | {(self.valid_boxes_count / max(1, self.total_boxes_scanned)) * 100:.2f}% |",
            f"| Corrupted Bounding Boxes | {self.corrupted_boxes_count} | {(self.corrupted_boxes_count / max(1, self.total_boxes_scanned)) * 100:.2f}% |",
            f"| Degenerate Boxes (Rejected) | {self.degenerate_boxes_count} | {self.degenerate_box_rejection_rate:.4f}% |",
            f"| Epsilon Clamped Boxes | {self.clamped_boxes_count} | {(self.clamped_boxes_count / max(1, self.total_boxes_scanned)) * 100:.2f}% |",
            "",
            "## 3. Canonical Class Balance Distribution",
            "",
            "| Class ID | Canonical Name | Color | Piece Type | Count | Share (%) |",
            "| :---: | :--- | :---: | :---: | :---: | :---: |",
        ]

        total_valid = sum(self.class_counts.values()) or 1
        for cid in range(12):
            cnt = self.class_counts.get(cid, 0)
            pct = (cnt / total_valid) * 100.0
            name = mapper.get_class_info(cid).name if mapper else f"class_{cid}"
            color = "White" if cid < 6 else "Black"
            ptype = (
                ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"][cid % 6]
            )
            lines.append(f"| {cid} | `{name}` | {color} | {ptype} | {cnt} | {pct:.2f}% |")

        lines.extend([
            "",
            f"- **Color Balance:** White pieces: `{self.white_piece_count}` ({(self.white_piece_count / total_valid) * 100:.1f}%) | Black pieces: `{self.black_piece_count}` ({(self.black_piece_count / total_valid) * 100:.1f}%)",
            "",
        ])

        if self.violations:
            lines.extend([
                "## 4. Violation Details",
                "",
                "| Severity | Type | File | Line | Description |",
                "| :---: | :--- | :--- | :---: | :--- |",
            ])
            for v in self.violations[:50]:  # Cap output at 50 for readability
                rel_path = v.file_path.name
                line_str = str(v.line_number) if v.line_number is not None else "-"
                lines.append(
                    f"| `{v.severity.value}` | `{v.violation_type.value}` | `{rel_path}` | {line_str} | {v.description} |"
                )
            if len(self.violations) > 50:
                lines.append(f"\n*...and {len(self.violations) - 50} additional violations omitted from preview.*")

        return "\n".join(lines)


class DatasetIntegrityAuditor:
    """
    Automated Dataset Integrity & Corruption Auditor.
    
    Verifies 100% of annotation files, image pairs, 0-byte negative samples,
    and calculates statistical class balance and degenerate box rejection rates.
    """

    def __init__(
        self,
        config: AuditConfig | None = None,
        class_mapper: CanonicalClassMapper | None = None,
    ) -> None:
        self.config = config or AuditConfig()
        self.class_mapper = class_mapper or CanonicalClassMapper(self.config.config_path)

    def audit_dataset(self, target_dir: Path | str) -> DatasetAuditReport:
        """
        Executes a complete integrity audit over a standardized or hybrid dataset directory.

        Discovers all images and labels (supporting flat directories or partitioned
        YOLO folder layouts like images/{train,val,test} and labels/{train,val,test}).
        """
        root_path = Path(target_dir)
        report = DatasetAuditReport(dataset_path=root_path)

        if not root_path.exists():
            report.violations.append(
                AnnotationViolation(
                    file_path=root_path,
                    line_number=None,
                    violation_type=ViolationType.MISSING_FILE,
                    severity=ViolationSeverity.ERROR,
                    description=f"Target dataset directory does not exist: {root_path}",
                )
            )
            return report

        # 1. Discover all image and label pairs
        pairs = self._discover_file_pairs(root_path)

        # 2. Audit image & label pairing synchronicity
        for key, pair in pairs.items():
            # Check for orphaned labels
            if pair.label_path and not pair.image_path:
                report.total_labels_scanned += 1
                report.orphaned_labels_count += 1
                report.violations.append(
                    AnnotationViolation(
                        file_path=pair.label_path,
                        line_number=None,
                        violation_type=ViolationType.ORPHAN_LABEL,
                        severity=ViolationSeverity.ERROR,
                        description=f"Orphaned label file has no corresponding image in dataset: {pair.label_path.name}",
                    )
                )
                continue

            # Check for orphaned images
            if pair.image_path and not pair.label_path:
                report.total_images_scanned += 1
                report.orphaned_images_count += 1
                report.violations.append(
                    AnnotationViolation(
                        file_path=pair.image_path,
                        line_number=None,
                        violation_type=ViolationType.ORPHAN_IMAGE,
                        severity=ViolationSeverity.ERROR,
                        description=f"Orphaned image file has no corresponding label (.txt) file: {pair.image_path.name}",
                    )
                )
                continue

            # Matched pair
            report.total_images_scanned += 1
            report.total_labels_scanned += 1
            report.matched_pairs_count += 1

            assert pair.image_path is not None
            assert pair.label_path is not None

            # 3. Verify Image File Integrity
            if self.config.validate_image_decoding:
                is_valid_img, img_err = self.verify_image_integrity(pair.image_path)
                if not is_valid_img:
                    report.corrupted_images_count += 1
                    report.violations.append(
                        AnnotationViolation(
                            file_path=pair.image_path,
                            line_number=None,
                            violation_type=ViolationType.CORRUPTED_IMAGE,
                            severity=ViolationSeverity.ERROR,
                            description=f"Image decoding failed or corrupted image header: {img_err}",
                        )
                    )

            # 4. Audit Label File Annotations
            self._audit_label_file(pair.label_path, report)

        # 5. Compute class distribution totals and piece color balances
        self._compute_aggregate_metrics(report)

        return report

    def _discover_file_pairs(self, root_path: Path) -> dict[str, FilePairInfo]:
        """
        Discovers all image and label files under root_path, handling:
        - YOLO standard: images/{train,val,test} + labels/{train,val,test}
        - Partition standard: {train,val,test}/images + {train,val,test}/labels
        - Flat directory: images/ + labels/ or all files in a single folder
        """
        pairs: dict[str, FilePairInfo] = {}

        # 1. Scan for all image files
        for ext in self.config.supported_image_extensions:
            for img_path in root_path.rglob(f"*{ext}"):
                rel_stem = self._extract_logical_stem(img_path, root_path, is_label=False)
                if rel_stem not in pairs:
                    pairs[rel_stem] = FilePairInfo()
                pairs[rel_stem].image_path = img_path

        # Also search uppercase extensions (e.g. .JPG, .PNG)
        for ext in self.config.supported_image_extensions:
            for img_path in root_path.rglob(f"*{ext.upper()}"):
                rel_stem = self._extract_logical_stem(img_path, root_path, is_label=False)
                if rel_stem not in pairs:
                    pairs[rel_stem] = FilePairInfo()
                pairs[rel_stem].image_path = img_path

        # 2. Scan for all label files (.txt)
        for lbl_path in root_path.rglob("*.txt"):
            # Exclude metadata files like classes.txt, data.yaml, manifest.json, etc.
            if lbl_path.name in ("classes.txt", "README.txt", "urls.txt"):
                continue
            rel_stem = self._extract_logical_stem(lbl_path, root_path, is_label=True)
            if rel_stem not in pairs:
                pairs[rel_stem] = FilePairInfo()
            pairs[rel_stem].label_path = lbl_path

        return pairs

    def _extract_logical_stem(self, file_path: Path, root_path: Path, is_label: bool) -> str:
        """
        Normalizes relative file paths so that 'images/train/sample_01.jpg' and
        'labels/train/sample_01.txt' map to the identical key 'train/sample_01'.
        """
        rel = file_path.relative_to(root_path)
        parts = list(rel.parts)

        # Strip image extension or .txt
        stem = file_path.stem

        # Remove leading 'images' or 'labels' if present
        if parts and parts[0] in ("images", "labels"):
            parts = parts[1:]

        # If subdirectories exist (e.g. ['train', 'sample.txt']), normalize
        if len(parts) > 1:
            # Check if second part is 'images' or 'labels'
            if parts[1] in ("images", "labels"):
                parts = [parts[0]] + parts[2:]

            dir_prefix = "/".join(parts[:-1])
            return f"{dir_prefix}/{stem}"
        return stem

    def verify_image_integrity(self, image_path: Path) -> tuple[bool, str | None]:
        """
        Verifies that an image file exists, has non-zero size, and can be verified by PIL.
        """
        if not image_path.exists():
            return False, "File does not exist"

        if image_path.stat().st_size == 0:
            return False, "Image file is 0 bytes (empty file)"

        try:
            with Image.open(image_path) as img:
                img.verify()
            # Re-open for dimension check (verify closes file descriptor in PIL)
            with Image.open(image_path) as img:
                w, h = img.size
                if w <= 0 or h <= 0:
                    return False, f"Invalid image dimensions: {w}x{h}"
            return True, None
        except Exception as exc:
            return False, str(exc)

    def _audit_label_file(self, label_path: Path, report: DatasetAuditReport) -> None:
        """
        Audits a single label file line by line, validating YOLO format and mathematical bounds.
        """
        file_size = label_path.stat().st_size
        content = label_path.read_text(encoding="utf-8", errors="replace")

        # 1. Negative Sample (0-byte file check)
        if file_size == 0 or not content.strip():
            report.negative_samples_count += 1
            # Valid negative sample!
            return

        lines = content.splitlines()
        has_annotations = False

        for line_idx, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            has_annotations = True
            report.total_boxes_scanned += 1

            # Parse line tokens
            tokens = line.split()
            if len(tokens) < 5:
                report.corrupted_boxes_count += 1
                report.violations.append(
                    AnnotationViolation(
                        file_path=label_path,
                        line_number=line_idx,
                        violation_type=ViolationType.MALFORMED_LINE,
                        severity=ViolationSeverity.ERROR,
                        description=f"Expected at least 5 tokens (class_id xc yc w h), got {len(tokens)}: '{line}'",
                        raw_content=raw_line,
                    )
                )
                continue

            # 2. Parse Class ID
            class_id_str = tokens[0]
            try:
                # Check for float in class_id (e.g. 1.0)
                if "." in class_id_str:
                    raise ValueError(f"Class ID must be an integer, got float '{class_id_str}'")
                class_id = int(class_id_str)
            except ValueError as exc:
                report.corrupted_boxes_count += 1
                report.violations.append(
                    AnnotationViolation(
                        file_path=label_path,
                        line_number=line_idx,
                        violation_type=ViolationType.INVALID_CLASS_ID,
                        severity=ViolationSeverity.ERROR,
                        description=f"Invalid class ID '{class_id_str}': {exc}",
                        raw_content=raw_line,
                    )
                )
                continue

            # Assert class ID in 0..11
            if not (0 <= class_id < self.config.num_classes):
                report.corrupted_boxes_count += 1
                report.violations.append(
                    AnnotationViolation(
                        file_path=label_path,
                        line_number=line_idx,
                        violation_type=ViolationType.INVALID_CLASS_ID,
                        severity=ViolationSeverity.ERROR,
                        description=(
                            f"Class ID {class_id} out of canonical range [0, {self.config.num_classes - 1}]"
                        ),
                        raw_content=raw_line,
                    )
                )
                continue

            # 3. Parse Normalized Coordinates (xc, yc, w, h)
            coord_strs = tokens[1:5]
            coords: list[float] = []
            has_nan_or_inf = False

            for c_name, c_str in zip(["xc", "yc", "w", "h"], coord_strs):
                try:
                    c_val = float(c_str)
                    if math.isnan(c_val) or math.isinf(c_val):
                        has_nan_or_inf = True
                        break
                    coords.append(c_val)
                except ValueError:
                    has_nan_or_inf = True
                    break

            if has_nan_or_inf or len(coords) < 4:
                report.corrupted_boxes_count += 1
                report.violations.append(
                    AnnotationViolation(
                        file_path=label_path,
                        line_number=line_idx,
                        violation_type=ViolationType.NAN_OR_INF_VALUE,
                        severity=ViolationSeverity.ERROR,
                        description=f"Coordinate tokens contain NaN, Inf, or unparseable numeric values: '{line}'",
                        raw_content=raw_line,
                    )
                )
                continue

            xc, yc, w, h = coords[:4]

            # 4. Assert Positive Dimensions (w > 0, h > 0)
            if w <= 0.0 or h <= 0.0:
                report.corrupted_boxes_count += 1
                report.violations.append(
                    AnnotationViolation(
                        file_path=label_path,
                        line_number=line_idx,
                        violation_type=ViolationType.NON_POSITIVE_DIMENSION,
                        severity=ViolationSeverity.ERROR,
                        description=f"Bounding box dimensions must be strictly positive: w={w}, h={h}",
                        raw_content=raw_line,
                    )
                )
                continue

            # 5. Assert Coordinates within [0.0, 1.0] (with epsilon tolerance)
            eps = self.config.epsilon
            x_min = xc - w / 2.0
            y_min = yc - h / 2.0
            x_max = xc + w / 2.0
            y_max = yc + h / 2.0

            out_of_bounds = False
            # Check center coordinates
            if not (-eps <= xc <= 1.0 + eps) or not (-eps <= yc <= 1.0 + eps):
                out_of_bounds = True
            # Check dimensions <= 1.0 + eps
            if w > 1.0 + eps or h > 1.0 + eps:
                out_of_bounds = True

            # Check boundary limits
            if x_min < -eps or y_min < -eps or x_max > 1.0 + eps or y_max > 1.0 + eps:
                out_of_bounds = True

            if out_of_bounds:
                report.corrupted_boxes_count += 1
                report.violations.append(
                    AnnotationViolation(
                        file_path=label_path,
                        line_number=line_idx,
                        violation_type=ViolationType.OUT_OF_BOUNDS_COORD,
                        severity=ViolationSeverity.ERROR,
                        description=(
                            f"Coordinates out of normalized domain [0.0, 1.0]: "
                            f"xc={xc:.5f}, yc={yc:.5f}, w={w:.5f}, h={h:.5f} (bbox: [{x_min:.4f}, {y_min:.4f}, {x_max:.4f}, {y_max:.4f}])"
                        ),
                        raw_content=raw_line,
                    )
                )
                continue

            # Check if minor epsilon clamping is applicable
            if (
                x_min < 0.0
                or y_min < 0.0
                or x_max > 1.0
                or y_max > 1.0
                or xc < 0.0
                or yc < 0.0
                or xc > 1.0
                or yc > 1.0
            ):
                report.clamped_boxes_count += 1

            # 6. Check for Degenerate Boxes (w < min_dim or h < min_dim or area < min_area)
            area = w * h
            if w < self.config.min_dimension or h < self.config.min_dimension or area < self.config.min_area:
                report.degenerate_boxes_count += 1
                sev = (
                    ViolationSeverity.ERROR
                    if self.config.strict_degenerate_as_error
                    else ViolationSeverity.WARNING
                )
                report.violations.append(
                    AnnotationViolation(
                        file_path=label_path,
                        line_number=line_idx,
                        violation_type=ViolationType.DEGENERATE_BOX,
                        severity=sev,
                        description=(
                            f"Degenerate box below minimum threshold (w={w:.5f}, h={h:.5f}, area={area:.6f} < {self.config.min_area})"
                        ),
                        raw_content=raw_line,
                    )
                )
                # If strict, count as corrupted
                if self.config.strict_degenerate_as_error:
                    report.corrupted_boxes_count += 1
                    continue

            # Annotation is valid!
            report.valid_boxes_count += 1
            report.class_counts[class_id] += 1

    def _compute_aggregate_metrics(self, report: DatasetAuditReport) -> None:
        """Calculates color balances, piece type counts, and aggregate stats."""
        white_count = 0
        black_count = 0
        piece_type_counter: Counter[str] = Counter()

        piece_names = ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"]

        for cid, count in report.class_counts.items():
            if cid < 6:
                white_count += count
            else:
                black_count += count

            ptype = piece_names[cid % 6]
            piece_type_counter[ptype] += count

        report.white_piece_count = white_count
        report.black_piece_count = black_count
        report.piece_type_counts = dict(piece_type_counter)
