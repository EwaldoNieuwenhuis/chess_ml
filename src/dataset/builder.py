"""
Unified Hybrid Dataset Builder, Deduplicator & Stratified YOLO Splitter.

Merges physical and digital chess datasets into a balanced hybrid dataset (data/hybrid_chess/)
with cryptographic deduplication, stratified train/validation/test splits, 0-byte negative samples,
and Ultralytics-compliant data.yaml generation (US-2.3.2 / ADR-008).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import yaml

from src.dataset.normalizer import (
    DEFAULT_CANONICAL_CONFIG_PATH,
    CanonicalClassMapper,
)
from src.schemas.contracts import DomainType

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tiff",
)


class DatasetBuilderError(Exception):
    """Base exception for hybrid dataset builder operations."""


class InvalidSplitRatioError(DatasetBuilderError):
    """Raised when train/val/test split ratios do not sum to 1.0 or are invalid."""


class DatasetEmptyError(DatasetBuilderError):
    """Raised when no valid images or labels are discovered in source paths."""


@dataclass(frozen=True)
class DatasetSplitRatio:
    """Configures the proportion of samples allocated across partitions."""

    train: float = 0.70
    val: float = 0.15
    test: float = 0.15

    def __post_init__(self) -> None:
        if self.train < 0.0 or self.val < 0.0 or self.test < 0.0:
            raise InvalidSplitRatioError(
                f"Split ratios must be non-negative: train={self.train}, val={self.val}, test={self.test}"
            )
        total = self.train + self.val + self.test
        if not (0.999 <= total <= 1.001):
            raise InvalidSplitRatioError(
                f"Split ratios must sum to 1.0 (got {total:.4f}: train={self.train}, val={self.val}, test={self.test})"
            )


@dataclass
class ImageSample:
    """Represents an image and optional annotation pair with metadata."""

    image_path: Path
    label_path: Path | None
    domain: DomainType
    sha256_hash: str = ""
    class_counts: dict[int, int] = field(default_factory=dict)
    is_negative: bool = False
    source_name: str = "generic"

    @property
    def total_pieces(self) -> int:
        return sum(self.class_counts.values())

    @property
    def has_annotation(self) -> bool:
        return self.label_path is not None and self.label_path.exists()


@dataclass
class DeduplicationReport:
    """Summary of image hash deduplication."""

    total_scanned: int = 0
    unique_retained: int = 0
    duplicates_removed: int = 0
    duplicate_mappings: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class SplitPartitionSummary:
    """Summary metrics for a specific split partition (train, val, or test)."""

    split_name: str
    total_images: int = 0
    physical_count: int = 0
    digital_count: int = 0
    negative_count: int = 0
    total_pieces: int = 0
    class_distribution: dict[int, int] = field(default_factory=dict)


@dataclass
class DatasetManifest:
    """Comprehensive metadata manifest exported alongside the generated dataset."""

    dataset_name: str = "hybrid_chess"
    version: str = "1.0.0"
    split_ratios: dict[str, float] = field(default_factory=dict)
    total_images: int = 0
    total_unique_images: int = 0
    total_duplicates_removed: int = 0
    total_pieces: int = 0
    domain_totals: dict[str, int] = field(default_factory=dict)
    splits: dict[str, SplitPartitionSummary] = field(default_factory=dict)
    class_names: dict[int, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


class DatasetDeduplicator:
    """
    Computes cryptographic SHA-256 hashes of image binary contents to detect
    exact duplicate files across disparate datasets, preventing train/test data leakage.
    """

    @staticmethod
    def compute_file_sha256(file_path: Path | str, chunk_size: int = 65536) -> str:
        """Computes SHA-256 digest of a file in streaming chunks."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()

    def deduplicate_samples(self, samples: Sequence[ImageSample]) -> tuple[list[ImageSample], DeduplicationReport]:
        """
        Filters duplicate images by SHA-256 hash. Preserves the first instance encountered.
        """
        seen_hashes: dict[str, ImageSample] = {}
        unique_samples: list[ImageSample] = []
        report = DeduplicationReport(total_scanned=len(samples))

        for sample in samples:
            if not sample.sha256_hash:
                sample.sha256_hash = self.compute_file_sha256(sample.image_path)

            img_hash = sample.sha256_hash
            if img_hash not in seen_hashes:
                seen_hashes[img_hash] = sample
                unique_samples.append(sample)
            else:
                report.duplicates_removed += 1
                if img_hash not in report.duplicate_mappings:
                    report.duplicate_mappings[img_hash] = [str(seen_hashes[img_hash].image_path)]
                report.duplicate_mappings[img_hash].append(str(sample.image_path))

        report.unique_retained = len(unique_samples)
        logger.info(
            "Deduplication complete: %d scanned, %d retained, %d duplicates removed",
            report.total_scanned,
            report.unique_retained,
            report.duplicates_removed,
        )
        return unique_samples, report


class StratifiedDatasetSplitter:
    """
    Splits samples into train, validation, and test partitions while balancing:
    1. Modality ratio (Physical 3D photos vs Digital 2D screenshots).
    2. Negative sample distribution (empty background boards with 0-byte labels).
    3. Piece class distributions across partitions.
    """

    def __init__(self, ratios: DatasetSplitRatio | None = None, seed: int = 42) -> None:
        self.ratios = ratios or DatasetSplitRatio()
        self.seed = seed

    def split_samples(
        self, samples: Sequence[ImageSample]
    ) -> dict[str, list[ImageSample]]:
        """
        Partitions samples into {'train': [...], 'val': [...], 'test': [...]}
        with domain and negative-sample stratification.
        """
        if not samples:
            raise DatasetEmptyError("Cannot split an empty sample list.")

        rng = random.Random(self.seed)

        # 1. Group samples by (domain, is_negative) strata
        strata: dict[tuple[DomainType, bool], list[ImageSample]] = defaultdict(list)
        for s in samples:
            strata[(s.domain, s.is_negative)].append(s)

        splits: dict[str, list[ImageSample]] = {
            "train": [],
            "val": [],
            "test": [],
        }

        # 2. Split each stratum deterministically by configured ratio
        for (domain, is_neg), stratum_samples in strata.items():
            shuffled = list(stratum_samples)
            rng.shuffle(shuffled)

            n_total = len(shuffled)
            n_train = int(round(n_total * self.ratios.train))
            n_val = int(round(n_total * self.ratios.val))
            # Allocate remainder to test to prevent rounding drift
            n_test = n_total - n_train - n_val

            # Adjust if rounding causes negative or overshoot
            if n_train + n_val > n_total:
                n_train = min(n_train, n_total)
                n_val = max(0, n_total - n_train)
                n_test = 0

            train_part = shuffled[:n_train]
            val_part = shuffled[n_train : n_train + n_val]
            test_part = shuffled[n_train + n_val :]

            splits["train"].extend(train_part)
            splits["val"].extend(val_part)
            splits["test"].extend(test_part)

        # 3. Final shuffle within each split for randomized batch loading
        for split_name in ("train", "val", "test"):
            rng.shuffle(splits[split_name])

        logger.info(
            "Stratified split: train=%d, val=%d, test=%d (total=%d)",
            len(splits["train"]),
            len(splits["val"]),
            len(splits["test"]),
            len(samples),
        )
        return splits


class HybridDatasetBuilder:
    """
    Orchestrates scanning, deduplicating, splitting, assembling YOLO directory
    structures, writing 0-byte negative labels, and generating data.yaml + manifest.json.
    """

    def __init__(
        self,
        output_dir: Path | str = "data/hybrid_chess",
        split_ratios: DatasetSplitRatio | None = None,
        class_mapper: CanonicalClassMapper | None = None,
        seed: int = 42,
        copy_mode: str = "copy",
    ) -> None:
        """
        Args:
            output_dir: Destination path for the unified hybrid dataset.
            split_ratios: Proportions for train/val/test splits.
            class_mapper: Canonical 12-class schema mapper.
            seed: Random seed for deterministic reproducibility.
            copy_mode: One of 'copy', 'symlink', or 'hardlink'.
        """
        self.output_dir = Path(output_dir)
        self.split_ratios = split_ratios or DatasetSplitRatio()
        self.class_mapper = class_mapper or CanonicalClassMapper()
        self.seed = seed
        self.copy_mode = copy_mode.lower()
        if self.copy_mode not in ("copy", "symlink", "hardlink"):
            raise ValueError(f"Invalid copy_mode '{copy_mode}'. Must be 'copy', 'symlink', or 'hardlink'.")

        self.deduplicator = DatasetDeduplicator()
        self.splitter = StratifiedDatasetSplitter(ratios=self.split_ratios, seed=self.seed)

    @staticmethod
    def _parse_yolo_class_counts(label_path: Path | None) -> dict[int, int]:
        """Parses a normalized YOLO .txt file and counts occurrences of each class_id."""
        counts: dict[int, int] = defaultdict(int)
        if label_path is None or not label_path.exists():
            return counts

        with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if parts and parts[0].isdigit():
                    counts[int(parts[0])] += 1
        return dict(counts)

    def scan_directory(
        self,
        images_dir: Path | str,
        labels_dir: Path | str | None = None,
        domain: DomainType = DomainType.PHYSICAL_3D,
        source_name: str = "generic",
        is_negative: bool = False,
    ) -> list[ImageSample]:
        """
        Scans an image directory and pairs each image with its corresponding YOLO .txt label.
        If labels_dir is None, looks for matching .txt files in a parallel 'labels' directory or adjacent.
        """
        img_path = Path(images_dir)
        if not img_path.exists():
            logger.warning("Images directory does not exist: %s", img_path)
            return []

        lbl_path = Path(labels_dir) if labels_dir else None

        samples: list[ImageSample] = []
        for file in img_path.rglob("*"):
            if file.is_file() and file.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                # Find matching label
                target_label: Path | None = None
                if lbl_path and lbl_path.exists():
                    candidate = lbl_path / f"{file.stem}.txt"
                    if candidate.exists():
                        target_label = candidate
                else:
                    # Check parallel 'labels' folder or adjacent file
                    candidate1 = file.parent.parent / "labels" / f"{file.stem}.txt"
                    candidate2 = file.parent / f"{file.stem}.txt"
                    if candidate1.exists():
                        target_label = candidate1
                    elif candidate2.exists():
                        target_label = candidate2

                # Check if this sample is a negative sample (no pieces or explicit empty file)
                sample_is_neg = is_negative
                class_counts: dict[int, int] = {}
                if target_label and target_label.exists():
                    class_counts = self._parse_yolo_class_counts(target_label)
                    if len(class_counts) == 0:
                        sample_is_neg = True
                elif target_label is None:
                    sample_is_neg = True

                samples.append(
                    ImageSample(
                        image_path=file,
                        label_path=target_label,
                        domain=domain,
                        class_counts=class_counts,
                        is_negative=sample_is_neg,
                        source_name=source_name,
                    )
                )

        logger.info(
            "Scanned %d images from %s (domain=%s, source=%s, negatives=%d)",
            len(samples),
            img_path,
            domain.value,
            source_name,
            sum(1 for s in samples if s.is_negative),
        )
        return samples

    def _transfer_file(self, src: Path, dst: Path) -> None:
        """Copies, symlinks, or hardlinks a file from src to dst."""
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()

        if self.copy_mode == "symlink":
            try:
                os.symlink(src.resolve(), dst)
                return
            except OSError:
                # Fallback to copy if symlink privileges are missing on Windows
                shutil.copy2(src, dst)
        elif self.copy_mode == "hardlink":
            try:
                os.link(src, dst)
                return
            except OSError:
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)

    def generate_data_yaml(self, custom_names: dict[int, str] | None = None) -> Path:
        """
        Generates an Ultralytics-compliant data.yaml with canonical 12 piece classes.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = self.output_dir / "data.yaml"
        names = custom_names or {cid: self.class_mapper.get_class_info(cid).name for cid in range(12)}

        # Format relative paths for portable execution
        yaml_content = {
            "path": str(self.output_dir.as_posix()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(names),
            "names": names,
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, sort_keys=False, default_flow_style=False)

        logger.info("Generated YOLO data.yaml at: %s", yaml_path)
        return yaml_path

    def build_dataset(
        self,
        samples: Sequence[ImageSample],
        dry_run: bool = False,
    ) -> DatasetManifest:
        """
        Executes full dataset compilation: deduplication, stratified splitting,
        YOLO directory writing, 0-byte negative label generation, and manifest export.
        """
        if not samples:
            raise DatasetEmptyError("Cannot build hybrid dataset from empty sample pool.")

        # 1. Cryptographic Deduplication
        unique_samples, dedup_report = self.deduplicator.deduplicate_samples(samples)
        if not unique_samples:
            raise DatasetEmptyError("All scanned samples were filtered out during deduplication.")

        # 2. Stratified Partitioning
        splits = self.splitter.split_samples(unique_samples)

        # 3. Compute Manifest Summaries
        manifest = DatasetManifest(
            dataset_name="hybrid_chess",
            version="1.0.0",
            split_ratios={
                "train": self.split_ratios.train,
                "val": self.split_ratios.val,
                "test": self.split_ratios.test,
            },
            total_images=len(samples),
            total_unique_images=len(unique_samples),
            total_duplicates_removed=dedup_report.duplicates_removed,
            class_names={cid: self.class_mapper.get_class_info(cid).name for cid in range(12)},
        )

        domain_counts: Counter[str] = Counter()
        total_pieces = 0

        for split_name, split_samples in splits.items():
            p_count = sum(1 for s in split_samples if s.domain == DomainType.PHYSICAL_3D)
            d_count = sum(1 for s in split_samples if s.domain == DomainType.DIGITAL_2D)
            n_count = sum(1 for s in split_samples if s.is_negative)

            class_dist: Counter[int] = Counter()
            split_pieces = 0
            for s in split_samples:
                domain_counts[s.domain.value] += 1
                for cid, count in s.class_counts.items():
                    class_dist[cid] += count
                    split_pieces += count

            total_pieces += split_pieces

            summary = SplitPartitionSummary(
                split_name=split_name,
                total_images=len(split_samples),
                physical_count=p_count,
                digital_count=d_count,
                negative_count=n_count,
                total_pieces=split_pieces,
                class_distribution=dict(class_dist),
            )
            manifest.splits[split_name] = summary

        manifest.domain_totals = dict(domain_counts)
        manifest.total_pieces = total_pieces

        if dry_run:
            logger.info("Dry-run requested: skipping file system write operations.")
            return manifest

        # 4. Write Files to Disk (YOLO Layout)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_samples in splits.items():
            img_dest_dir = self.output_dir / "images" / split_name
            lbl_dest_dir = self.output_dir / "labels" / split_name
            img_dest_dir.mkdir(parents=True, exist_ok=True)
            lbl_dest_dir.mkdir(parents=True, exist_ok=True)

            for idx, sample in enumerate(split_samples):
                # Ensure unique destination filenames across disparate source datasets
                prefix = f"{sample.domain.value[:3]}_{sample.source_name}_"
                dst_img_name = f"{prefix}{sample.image_path.name}"
                dst_lbl_name = f"{prefix}{sample.image_path.stem}.txt"

                dst_img_path = img_dest_dir / dst_img_name
                dst_lbl_path = lbl_dest_dir / dst_lbl_name

                # Copy/link image
                self._transfer_file(sample.image_path, dst_img_path)

                # Write label (or 0-byte negative file)
                if sample.is_negative or not sample.label_path or not sample.label_path.exists():
                    # Create 0-byte empty file for negative samples
                    with open(dst_lbl_path, "w", encoding="utf-8") as f:
                        pass
                else:
                    self._transfer_file(sample.label_path, dst_lbl_path)

        # 5. Generate data.yaml and manifest.json
        self.generate_data_yaml()

        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        logger.info("Hybrid dataset compilation complete at: %s", self.output_dir)
        return manifest
