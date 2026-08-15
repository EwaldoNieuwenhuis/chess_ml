"""
Unit and Property Tests for Hybrid Dataset Builder, Deduplicator & Stratified Splitter (US-2.3.2 / ADR-008).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from src.dataset.builder import (
    DatasetBuilderError,
    DatasetDeduplicator,
    DatasetEmptyError,
    DatasetManifest,
    DatasetSplitRatio,
    DeduplicationReport,
    HybridDatasetBuilder,
    ImageSample,
    InvalidSplitRatioError,
    StratifiedDatasetSplitter,
)
from src.dataset.normalizer import CanonicalClassMapper
from src.schemas.contracts import DomainType


@pytest.fixture
def temp_workspace() -> Path:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def create_dummy_image(path: Path, width: int = 100, height: int = 100, color: tuple[int, int, int] = (255, 0, 0)) -> Path:
    """Helper to write a dummy RGB image to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), color=color)
    img.save(path)
    return path


def create_dummy_label(path: Path, annotations: list[tuple[int, float, float, float, float]]) -> Path:
    """Helper to write normalized YOLO bounding boxes to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for cid, xc, yc, w, h in annotations:
            f.write(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
    return path


class TestDatasetSplitRatio:
    """Tests for DatasetSplitRatio validation rules."""

    def test_default_ratios(self) -> None:
        r = DatasetSplitRatio()
        assert r.train == 0.70
        assert r.val == 0.15
        assert r.test == 0.15

    def test_custom_valid_ratios(self) -> None:
        r = DatasetSplitRatio(train=0.80, val=0.10, test=0.10)
        assert r.train == 0.80
        assert r.val == 0.10
        assert r.test == 0.10

    @pytest.mark.parametrize(
        "train, val, test",
        [
            (0.70, 0.20, 0.20),   # Sum 1.10 != 1.0
            (0.50, 0.20, 0.20),   # Sum 0.90 != 1.0
            (-0.10, 0.50, 0.60),  # Negative train
            (0.70, -0.10, 0.40),  # Negative val
            (1.20, 0.00, 0.00),   # Overshoot
        ],
    )
    def test_invalid_ratios_raise_error(self, train: float, val: float, test: float) -> None:
        with pytest.raises(InvalidSplitRatioError):
            DatasetSplitRatio(train=train, val=val, test=test)


class TestDatasetDeduplicator:
    """Tests for cryptographic image deduplication."""

    @pytest.fixture
    def deduplicator(self) -> DatasetDeduplicator:
        return DatasetDeduplicator()

    def test_deduplicate_identical_images(self, deduplicator: DatasetDeduplicator, temp_workspace: Path) -> None:
        """Identical image bytes with different paths must be filtered down to 1 unique sample."""
        img1 = create_dummy_image(temp_workspace / "source_a" / "board_01.jpg", color=(100, 150, 200))
        img2 = create_dummy_image(temp_workspace / "source_b" / "board_copy.jpg", color=(100, 150, 200))
        img3 = create_dummy_image(temp_workspace / "source_c" / "distinct_board.jpg", color=(50, 60, 70))

        samples = [
            ImageSample(image_path=img1, label_path=None, domain=DomainType.PHYSICAL_3D, source_name="a"),
            ImageSample(image_path=img2, label_path=None, domain=DomainType.PHYSICAL_3D, source_name="b"),
            ImageSample(image_path=img3, label_path=None, domain=DomainType.DIGITAL_2D, source_name="c"),
        ]

        unique, report = deduplicator.deduplicate_samples(samples)

        assert len(unique) == 2
        assert report.total_scanned == 3
        assert report.unique_retained == 2
        assert report.duplicates_removed == 1
        assert len(report.duplicate_mappings) == 1

    def test_deduplicate_all_unique(self, deduplicator: DatasetDeduplicator, temp_workspace: Path) -> None:
        samples = []
        for i in range(5):
            p = create_dummy_image(temp_workspace / f"img_{i}.jpg", color=(i * 20, i * 30, i * 40))
            samples.append(ImageSample(image_path=p, label_path=None, domain=DomainType.DIGITAL_2D))

        unique, report = deduplicator.deduplicate_samples(samples)
        assert len(unique) == 5
        assert report.duplicates_removed == 0


class TestStratifiedDatasetSplitter:
    """Tests for multi-domain & negative sample stratified splitting."""

    def test_split_empty_raises_error(self) -> None:
        splitter = StratifiedDatasetSplitter()
        with pytest.raises(DatasetEmptyError):
            splitter.split_samples([])

    def test_stratified_domain_ratios_preserved(self, temp_workspace: Path) -> None:
        """Physical vs Digital ratios should be balanced across train (70%), val (15%), test (15%)."""
        samples: list[ImageSample] = []
        # 100 Physical samples
        for i in range(100):
            p = create_dummy_image(temp_workspace / "phys" / f"p_{i}.jpg", color=(i % 255, 0, 0))
            samples.append(ImageSample(image_path=p, label_path=None, domain=DomainType.PHYSICAL_3D))

        # 100 Digital samples
        for i in range(100):
            p = create_dummy_image(temp_workspace / "dig" / f"d_{i}.jpg", color=(0, i % 255, 0))
            samples.append(ImageSample(image_path=p, label_path=None, domain=DomainType.DIGITAL_2D))

        splitter = StratifiedDatasetSplitter(ratios=DatasetSplitRatio(train=0.70, val=0.15, test=0.15), seed=42)
        splits = splitter.split_samples(samples)

        # Check total split sizes
        assert len(splits["train"]) == 140
        assert len(splits["val"]) == 30
        assert len(splits["test"]) == 30

        # Check domain breakdown in each split
        train_p = sum(1 for s in splits["train"] if s.domain == DomainType.PHYSICAL_3D)
        train_d = sum(1 for s in splits["train"] if s.domain == DomainType.DIGITAL_2D)
        assert train_p == 70
        assert train_d == 70

        val_p = sum(1 for s in splits["val"] if s.domain == DomainType.PHYSICAL_3D)
        val_d = sum(1 for s in splits["val"] if s.domain == DomainType.DIGITAL_2D)
        assert val_p == 15
        assert val_d == 15

        test_p = sum(1 for s in splits["test"] if s.domain == DomainType.PHYSICAL_3D)
        test_d = sum(1 for s in splits["test"] if s.domain == DomainType.DIGITAL_2D)
        assert test_p == 15
        assert test_d == 15

    def test_stratified_negative_samples_balanced(self, temp_workspace: Path) -> None:
        """Negative samples (empty boards) must be proportionally partitioned across splits."""
        samples: list[ImageSample] = []
        # 80 Positive physical boards
        for i in range(80):
            p = create_dummy_image(temp_workspace / "pos" / f"pos_{i}.jpg")
            samples.append(ImageSample(image_path=p, label_path=None, domain=DomainType.PHYSICAL_3D, is_negative=False))

        # 20 Negative empty boards
        for i in range(20):
            p = create_dummy_image(temp_workspace / "neg" / f"neg_{i}.jpg")
            samples.append(ImageSample(image_path=p, label_path=None, domain=DomainType.PHYSICAL_3D, is_negative=True))

        splitter = StratifiedDatasetSplitter(ratios=DatasetSplitRatio(train=0.70, val=0.15, test=0.15), seed=42)
        splits = splitter.split_samples(samples)

        neg_train = sum(1 for s in splits["train"] if s.is_negative)
        neg_val = sum(1 for s in splits["val"] if s.is_negative)
        neg_test = sum(1 for s in splits["test"] if s.is_negative)

        assert neg_train == 14  # 70% of 20
        assert neg_val == 3     # 15% of 20
        assert neg_test == 3    # 15% of 20

    def test_deterministic_splitting(self, temp_workspace: Path) -> None:
        samples = [
            ImageSample(
                image_path=create_dummy_image(temp_workspace / f"img_{i}.jpg", color=(i, i, i)),
                label_path=None,
                domain=DomainType.PHYSICAL_3D,
            )
            for i in range(20)
        ]

        s1 = StratifiedDatasetSplitter(seed=1234).split_samples(samples)
        s2 = StratifiedDatasetSplitter(seed=1234).split_samples(samples)

        assert [s.image_path for s in s1["train"]] == [s.image_path for s in s2["train"]]
        assert [s.image_path for s in s1["val"]] == [s.image_path for s in s2["val"]]
        assert [s.image_path for s in s1["test"]] == [s.image_path for s in s2["test"]]


class TestHybridDatasetBuilder:
    """Integration and filesystem tests for HybridDatasetBuilder."""

    def test_scan_directory_and_pair_labels(self, temp_workspace: Path) -> None:
        img_dir = temp_workspace / "images"
        lbl_dir = temp_workspace / "labels"

        # Create 3 images with labels, 1 image without label (negative)
        for i in range(3):
            img_p = create_dummy_image(img_dir / f"board_{i}.jpg")
            create_dummy_label(lbl_dir / f"board_{i}.txt", [(0, 0.5, 0.5, 0.1, 0.1), (6, 0.6, 0.6, 0.1, 0.1)])

        # Negative sample (no label file)
        create_dummy_image(img_dir / "board_empty.jpg")

        builder = HybridDatasetBuilder(output_dir=temp_workspace / "out")
        samples = builder.scan_directory(img_dir, lbl_dir, domain=DomainType.PHYSICAL_3D, source_name="test")

        assert len(samples) == 4
        annotated = [s for s in samples if not s.is_negative]
        negatives = [s for s in samples if s.is_negative]
        assert len(annotated) == 3
        assert len(negatives) == 1
        assert annotated[0].class_counts == {0: 1, 6: 1}

    def test_generate_data_yaml(self, temp_workspace: Path) -> None:
        out_dir = temp_workspace / "hybrid_out"
        builder = HybridDatasetBuilder(output_dir=out_dir)
        yaml_path = builder.generate_data_yaml()

        assert yaml_path.exists()
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["nc"] == 12
        assert data["train"] == "images/train"
        assert data["val"] == "images/val"
        assert data["test"] == "images/test"
        assert len(data["names"]) == 12
        assert data["names"][0] == "white_pawn"
        assert data["names"][5] == "white_king"
        assert data["names"][6] == "black_pawn"
        assert data["names"][11] == "black_king"

    def test_full_dataset_build_pipeline(self, temp_workspace: Path) -> None:
        """End-to-end integration test creating train/val/test splits, 0-byte negative files, data.yaml, and manifest."""
        src_phys = temp_workspace / "raw_phys"
        src_dig = temp_workspace / "raw_dig"
        out_dir = temp_workspace / "hybrid_dataset"

        # Generate 10 physical samples (8 positive, 2 negative)
        for i in range(8):
            img = create_dummy_image(src_phys / "images" / f"phys_{i}.jpg", color=(i * 10 + 5, 0, 0))
            create_dummy_label(src_phys / "labels" / f"phys_{i}.txt", [(0, 0.5, 0.5, 0.1, 0.1)])
        for i in range(8, 10):
            create_dummy_image(src_phys / "images" / f"phys_{i}.jpg", color=(i * 10 + 100, 50, 0))
            create_dummy_label(src_phys / "labels" / f"phys_{i}.txt", [])  # Explicit 0-byte label

        # Generate 10 digital samples
        for i in range(10):
            img = create_dummy_image(src_dig / "images" / f"dig_{i}.jpg", color=(0, i * 10 + 5, 0))
            create_dummy_label(src_dig / "labels" / f"dig_{i}.txt", [(6, 0.2, 0.2, 0.05, 0.05)])

        builder = HybridDatasetBuilder(
            output_dir=out_dir,
            split_ratios=DatasetSplitRatio(train=0.70, val=0.15, test=0.15),
            seed=42,
            copy_mode="copy",
        )

        phys_samples = builder.scan_directory(src_phys / "images", src_phys / "labels", domain=DomainType.PHYSICAL_3D, source_name="phys")
        dig_samples = builder.scan_directory(src_dig / "images", src_dig / "labels", domain=DomainType.DIGITAL_2D, source_name="dig")
        all_samples = phys_samples + dig_samples

        manifest = builder.build_dataset(all_samples, dry_run=False)

        # Assert Manifest
        assert manifest.total_unique_images == 20
        assert manifest.total_pieces == 18  # 8 phys * 1 + 10 dig * 1
        assert (out_dir / "data.yaml").exists()
        assert (out_dir / "manifest.json").exists()

        # Assert Directory Layout
        for split in ("train", "val", "test"):
            img_dir = out_dir / "images" / split
            lbl_dir = out_dir / "labels" / split
            assert img_dir.exists()
            assert lbl_dir.exists()
            assert len(list(img_dir.glob("*.jpg"))) > 0
            assert len(list(lbl_dir.glob("*.txt"))) > 0
            assert len(list(img_dir.glob("*.jpg"))) == len(list(lbl_dir.glob("*.txt")))

        # Assert Negative Samples produce 0-byte .txt files
        all_labels = list((out_dir / "labels").rglob("*.txt"))
        zero_byte_labels = [p for p in all_labels if p.stat().st_size == 0]
        assert len(zero_byte_labels) == 2  # The 2 negative physical samples

    def test_dry_run_leaves_filesystem_clean(self, temp_workspace: Path) -> None:
        src = temp_workspace / "src"
        out_dir = temp_workspace / "dry_out"

        for i in range(5):
            create_dummy_image(src / f"img_{i}.jpg", color=(i * 30, i * 20, i * 10))
            create_dummy_label(src / f"img_{i}.txt", [(0, 0.5, 0.5, 0.1, 0.1)])

        builder = HybridDatasetBuilder(output_dir=out_dir)
        samples = builder.scan_directory(src, src, domain=DomainType.DIGITAL_2D)
        manifest = builder.build_dataset(samples, dry_run=True)

        assert manifest.total_unique_images == 5
        assert not (out_dir / "images").exists()
        assert not (out_dir / "labels").exists()
        assert not (out_dir / "data.yaml").exists()

