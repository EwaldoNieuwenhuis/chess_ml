"""
Unit, Integration, and CLI Tests for Automated Dataset Integrity & Corruption Auditor (US-2.3.3 / ADR-008).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from src.dataset.auditor import (
    AuditConfig,
    DatasetAuditReport,
    DatasetIntegrityAuditor,
    ViolationSeverity,
    ViolationType,
)
from src.dataset.normalizer import CanonicalClassMapper


@pytest.fixture
def temp_dataset_dir() -> Path:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def create_dummy_image(path: Path, width: int = 100, height: int = 100, color: tuple[int, int, int] = (200, 200, 200)) -> Path:
    """Creates a valid PIL image file on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), color=color)
    img.save(path)
    return path


def create_dummy_label(path: Path, lines: list[str]) -> Path:
    """Creates a YOLO label file on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")
    return path


class TestDatasetIntegrityAuditor:
    """Tests for core DatasetIntegrityAuditor validation rules."""

    @pytest.fixture
    def auditor(self) -> DatasetIntegrityAuditor:
        return DatasetIntegrityAuditor()

    def test_clean_dataset_passes_audit(self, auditor: DatasetIntegrityAuditor, temp_dataset_dir: Path) -> None:
        """Verify that a clean dataset with valid bounding boxes and 0-byte negative samples passes with 0 errors."""
        img_dir = temp_dataset_dir / "images"
        lbl_dir = temp_dataset_dir / "labels"

        # 1. Image with valid White and Black pieces
        create_dummy_image(img_dir / "board_01.jpg")
        create_dummy_label(
            lbl_dir / "board_01.txt",
            [
                "0 0.500000 0.500000 0.100000 0.150000",  # white_pawn
                "5 0.200000 0.300000 0.080000 0.120000",  # white_king
                "6 0.800000 0.800000 0.090000 0.140000",  # black_pawn
                "11 0.700000 0.200000 0.080000 0.120000", # black_king
            ],
        )

        # 2. Negative sample image with 0-byte label
        create_dummy_image(img_dir / "empty_board.jpg")
        create_dummy_label(lbl_dir / "empty_board.txt", [])

        report = auditor.audit_dataset(temp_dataset_dir)

        assert report.passed is True
        assert report.error_count == 0
        assert report.total_images_scanned == 2
        assert report.total_labels_scanned == 2
        assert report.matched_pairs_count == 2
        assert report.orphaned_images_count == 0
        assert report.orphaned_labels_count == 0
        assert report.negative_samples_count == 1
        assert report.total_boxes_scanned == 4
        assert report.valid_boxes_count == 4
        assert report.corrupted_boxes_count == 0
        assert report.white_piece_count == 2
        assert report.black_piece_count == 2
        assert report.class_counts[0] == 1
        assert report.class_counts[5] == 1
        assert report.class_counts[6] == 1
        assert report.class_counts[11] == 1

    @pytest.mark.parametrize(
        "invalid_class_line, expected_reason",
        [
            ("12 0.5 0.5 0.1 0.1", "out of canonical range"),
            ("-1 0.5 0.5 0.1 0.1", "out of canonical range"),
            ("100 0.5 0.5 0.1 0.1", "out of canonical range"),
            ("1.0 0.5 0.5 0.1 0.1", "must be an integer"),
            ("white_pawn 0.5 0.5 0.1 0.1", "Invalid class ID"),
        ],
    )
    def test_invalid_class_id_fails_audit(
        self, auditor: DatasetIntegrityAuditor, temp_dataset_dir: Path, invalid_class_line: str, expected_reason: str
    ) -> None:
        """Verify invalid class IDs (< 0, >= 12, floats, strings) fail the audit."""
        img_path = temp_dataset_dir / "images" / "test.jpg"
        lbl_path = temp_dataset_dir / "labels" / "test.txt"
        create_dummy_image(img_path)
        create_dummy_label(lbl_path, [invalid_class_line])

        report = auditor.audit_dataset(temp_dataset_dir)

        assert report.passed is False
        assert report.error_count >= 1
        assert report.corrupted_boxes_count == 1
        violations = [v for v in report.violations if v.violation_type == ViolationType.INVALID_CLASS_ID]
        assert len(violations) == 1
        assert violations[0].severity == ViolationSeverity.ERROR

    @pytest.mark.parametrize(
        "out_of_bounds_line",
        [
            "0 -0.1 0.5 0.1 0.1",    # Negative xc
            "0 0.5 -0.1 0.1 0.1",    # Negative yc
            "0 1.2 0.5 0.1 0.1",     # xc > 1.0
            "0 0.5 1.5 0.1 0.1",     # yc > 1.0
            "0 0.95 0.5 0.2 0.1",    # x_max = 0.95 + 0.1 = 1.05 > 1.0
            "0 0.5 0.95 0.1 0.2",    # y_max = 0.95 + 0.1 = 1.05 > 1.0
            "0 0.02 0.5 0.1 0.1",    # x_min = 0.02 - 0.05 = -0.03 < 0.0
            "0 0.5 0.02 0.1 0.1",    # y_min = 0.02 - 0.05 = -0.03 < 0.0
        ],
    )
    def test_out_of_bounds_coordinates_fail_audit(
        self, auditor: DatasetIntegrityAuditor, temp_dataset_dir: Path, out_of_bounds_line: str
    ) -> None:
        """Verify out-of-bounds normalized coordinates fail the audit."""
        img_path = temp_dataset_dir / "images" / "test.jpg"
        lbl_path = temp_dataset_dir / "labels" / "test.txt"
        create_dummy_image(img_path)
        create_dummy_label(lbl_path, [out_of_bounds_line])

        report = auditor.audit_dataset(temp_dataset_dir)

        assert report.passed is False
        assert report.error_count >= 1
        assert report.corrupted_boxes_count == 1
        violations = [v for v in report.violations if v.violation_type == ViolationType.OUT_OF_BOUNDS_COORD]
        assert len(violations) == 1

    @pytest.mark.parametrize(
        "nan_inf_line",
        [
            "0 nan 0.5 0.1 0.1",
            "0 0.5 inf 0.1 0.1",
            "0 0.5 0.5 -inf 0.1",
            "0 0.5 0.5 0.1 NaN",
            "0 text 0.5 0.1 0.1",
        ],
    )
    def test_nan_inf_values_fail_audit(
        self, auditor: DatasetIntegrityAuditor, temp_dataset_dir: Path, nan_inf_line: str
    ) -> None:
        """Verify NaN, Inf, and non-numeric coordinate values trigger fatal errors."""
        img_path = temp_dataset_dir / "images" / "test.jpg"
        lbl_path = temp_dataset_dir / "labels" / "test.txt"
        create_dummy_image(img_path)
        create_dummy_label(lbl_path, [nan_inf_line])

        report = auditor.audit_dataset(temp_dataset_dir)

        assert report.passed is False
        assert report.error_count >= 1
        assert report.corrupted_boxes_count == 1
        violations = [v for v in report.violations if v.violation_type == ViolationType.NAN_OR_INF_VALUE]
        assert len(violations) == 1

    @pytest.mark.parametrize(
        "non_positive_line",
        [
            "0 0.5 0.5 0.0 0.1",   # w == 0
            "0 0.5 0.5 0.1 0.0",   # h == 0
            "0 0.5 0.5 -0.05 0.1", # w < 0
            "0 0.5 0.5 0.1 -0.05", # h < 0
        ],
    )
    def test_non_positive_dimensions_fail_audit(
        self, auditor: DatasetIntegrityAuditor, temp_dataset_dir: Path, non_positive_line: str
    ) -> None:
        """Verify zero or negative width/height trigger fatal errors."""
        img_path = temp_dataset_dir / "images" / "test.jpg"
        lbl_path = temp_dataset_dir / "labels" / "test.txt"
        create_dummy_image(img_path)
        create_dummy_label(lbl_path, [non_positive_line])

        report = auditor.audit_dataset(temp_dataset_dir)

        assert report.passed is False
        assert report.error_count >= 1
        assert report.corrupted_boxes_count == 1
        violations = [v for v in report.violations if v.violation_type == ViolationType.NON_POSITIVE_DIMENSION]
        assert len(violations) == 1

    def test_degenerate_box_tracking_and_rejection_rate(self, temp_dataset_dir: Path) -> None:
        """Verify degenerate box detection (w < 0.005 or h < 0.005) and rejection rate metric."""
        img_path = temp_dataset_dir / "images" / "test.jpg"
        lbl_path = temp_dataset_dir / "labels" / "test.txt"
        create_dummy_image(img_path)
        create_dummy_label(
            lbl_path,
            [
                "0 0.5 0.5 0.100 0.100",   # Valid normal box
                "1 0.2 0.2 0.002 0.050",   # Degenerate (w < 0.005)
                "2 0.3 0.3 0.050 0.003",   # Degenerate (h < 0.005)
                "3 0.7 0.7 0.100 0.100",   # Valid normal box
            ],
        )

        # 1. Standard mode: warnings recorded, rejection rate = 50.0%
        auditor_std = DatasetIntegrityAuditor()
        report_std = auditor_std.audit_dataset(temp_dataset_dir)

        assert report_std.passed is True  # Degenerate boxes are warnings in standard mode
        assert report_std.total_boxes_scanned == 4
        assert report_std.degenerate_boxes_count == 2
        assert report_std.degenerate_box_rejection_rate == 50.0
        assert report_std.warning_count == 2

        # 2. Strict mode: degenerate boxes are fatal errors
        config_strict = AuditConfig(strict_degenerate_as_error=True)
        auditor_strict = DatasetIntegrityAuditor(config=config_strict)
        report_strict = auditor_strict.audit_dataset(temp_dataset_dir)

        assert report_strict.passed is False
        assert report_strict.error_count == 2
        assert report_strict.corrupted_boxes_count == 2

    def test_orphaned_image_detection(self, auditor: DatasetIntegrityAuditor, temp_dataset_dir: Path) -> None:
        """Verify that an image without a matching label file is flagged as an orphan error."""
        img_path = temp_dataset_dir / "images" / "orphan_board.jpg"
        create_dummy_image(img_path)

        report = auditor.audit_dataset(temp_dataset_dir)

        assert report.passed is False
        assert report.orphaned_images_count == 1
        assert report.matched_pairs_count == 0
        violations = [v for v in report.violations if v.violation_type == ViolationType.ORPHAN_IMAGE]
        assert len(violations) == 1

    def test_orphaned_label_detection(self, auditor: DatasetIntegrityAuditor, temp_dataset_dir: Path) -> None:
        """Verify that a label without a matching image file is flagged as an orphan error."""
        lbl_path = temp_dataset_dir / "labels" / "orphan_label.txt"
        create_dummy_label(lbl_path, ["0 0.5 0.5 0.1 0.1"])

        report = auditor.audit_dataset(temp_dataset_dir)

        assert report.passed is False
        assert report.orphaned_labels_count == 1
        assert report.matched_pairs_count == 0
        violations = [v for v in report.violations if v.violation_type == ViolationType.ORPHAN_LABEL]
        assert len(violations) == 1

    def test_corrupted_image_detection(self, auditor: DatasetIntegrityAuditor, temp_dataset_dir: Path) -> None:
        """Verify that a corrupted / 0-byte image file is flagged with a decoding error."""
        img_path = temp_dataset_dir / "images" / "corrupted.jpg"
        lbl_path = temp_dataset_dir / "labels" / "corrupted.txt"

        # Create 0-byte corrupt image
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(b"")
        create_dummy_label(lbl_path, ["0 0.5 0.5 0.1 0.1"])

        report = auditor.audit_dataset(temp_dataset_dir)

        assert report.passed is False
        assert report.corrupted_images_count == 1
        violations = [v for v in report.violations if v.violation_type == ViolationType.CORRUPTED_IMAGE]
        assert len(violations) == 1

    def test_partitioned_yolo_directory_structure(self, auditor: DatasetIntegrityAuditor, temp_dataset_dir: Path) -> None:
        """Verify auditor correctly pairs images and labels in images/{train,val,test} and labels/{train,val,test}."""
        for split in ("train", "val", "test"):
            img_p = temp_dataset_dir / "images" / split / f"{split}_01.png"
            lbl_p = temp_dataset_dir / "labels" / split / f"{split}_01.txt"
            create_dummy_image(img_p)
            create_dummy_label(lbl_p, ["0 0.5 0.5 0.2 0.2", "6 0.3 0.3 0.1 0.1"])

        report = auditor.audit_dataset(temp_dataset_dir)

        assert report.passed is True
        assert report.total_images_scanned == 3
        assert report.total_labels_scanned == 3
        assert report.matched_pairs_count == 3
        assert report.total_boxes_scanned == 6
        assert report.valid_boxes_count == 6
        assert report.white_piece_count == 3
        assert report.black_piece_count == 3

    def test_report_serialization_and_markdown(self, auditor: DatasetIntegrityAuditor, temp_dataset_dir: Path) -> None:
        """Verify JSON and Markdown export formats."""
        img_p = temp_dataset_dir / "images" / "sample.jpg"
        lbl_p = temp_dataset_dir / "labels" / "sample.txt"
        create_dummy_image(img_p)
        create_dummy_label(lbl_p, ["0 0.5 0.5 0.1 0.1"])

        report = auditor.audit_dataset(temp_dataset_dir)
        d = report.to_dict()
        assert d["passed"] is True
        assert "pairing_summary" in d
        assert "class_distribution" in d

        json_str = report.to_json()
        parsed_json = json.loads(json_str)
        assert parsed_json["annotation_summary"]["total_boxes_scanned"] == 1

        md_str = report.to_markdown()
        assert "# Dataset Integrity Audit Report" in md_str
        assert "PASSED" in md_str
        assert "| 0 | `white_pawn` | White | Pawn | 1 |" in md_str or "| 0 |" in md_str


class TestAuditCLI:
    """Integration tests for scripts/audit_standardized_dataset.py CLI."""

    def test_cli_exit_code_zero_on_clean_dataset(self, temp_dataset_dir: Path) -> None:
        """Verify CLI returns exit code 0 on clean dataset."""
        img_p = temp_dataset_dir / "images" / "test.jpg"
        lbl_p = temp_dataset_dir / "labels" / "test.txt"
        create_dummy_image(img_p)
        create_dummy_label(lbl_p, ["0 0.5 0.5 0.1 0.1", "7 0.2 0.2 0.08 0.08"])

        json_report = temp_dataset_dir / "report.json"
        md_report = temp_dataset_dir / "report.md"

        cmd = [
            sys.executable,
            "scripts/audit_standardized_dataset.py",
            "--target-dir",
            str(temp_dataset_dir),
            "--report-json",
            str(json_report),
            "--report-md",
            str(md_report),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

        assert result.returncode == 0
        assert "AUDIT PASSED" in result.stdout
        assert json_report.exists()
        assert md_report.exists()

    def test_cli_exit_code_one_on_corrupt_dataset(self, temp_dataset_dir: Path) -> None:
        """Verify CLI returns non-zero exit code on corrupted dataset."""
        img_p = temp_dataset_dir / "images" / "corrupted.jpg"
        lbl_p = temp_dataset_dir / "labels" / "corrupted.txt"
        create_dummy_image(img_p)
        # Inject corrupted coordinate and invalid class ID
        create_dummy_label(lbl_p, ["99 1.5 0.5 0.1 0.1"])

        cmd = [
            sys.executable,
            "scripts/audit_standardized_dataset.py",
            "--target-dir",
            str(temp_dataset_dir),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

        assert result.returncode == 1
        assert "AUDIT FAILED" in result.stdout
        assert "invalid_class_id" in result.stdout or "out_of_bounds_coord" in result.stdout
