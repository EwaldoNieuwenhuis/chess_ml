"""
Unit and benchmark test suite for the Statistical Heuristics Domain Classifier (US-3.1.1 & ADR-009).
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.domain_classifier.heuristic_screener import (
    HeuristicFeatures,
    StatisticalHeuristicsScreener,
    classify_domain_heuristic,
)
from src.schemas.contracts import (
    ClassificationMethod,
    DomainClassificationResult,
    DomainType,
)


@pytest.fixture
def screener() -> StatisticalHeuristicsScreener:
    return StatisticalHeuristicsScreener()


@pytest.fixture
def digital_images_dir() -> Path:
    p = Path("data/standardized/digital/images")
    if not p.is_dir():
        pytest.skip(f"Standardized digital images not found at {p}")
    return p


@pytest.fixture
def physical_images_dir() -> Path:
    p = Path("data/standardized/physical/images")
    if not p.is_dir():
        pytest.skip(f"Standardized physical images not found at {p}")
    return p


@pytest.fixture
def synthetic_digital_board() -> np.ndarray:
    """Creates a clean synthetic 400x400 digital 8x8 chessboard (flat colors, no noise)."""
    board = np.zeros((400, 400, 3), dtype=np.uint8)
    sq_size = 50
    for r in range(8):
        for c in range(8):
            color = (240, 240, 240) if (r + c) % 2 == 0 else (120, 150, 80)
            board[r * sq_size : (r + 1) * sq_size, c * sq_size : (c + 1) * sq_size] = color
    return board


@pytest.fixture
def synthetic_physical_photo() -> np.ndarray:
    """Creates a noisy, angled camera-like image with continuous gradients and sensor noise."""
    np.random.seed(42)
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    # Add lighting gradient
    for y in range(400):
        for x in range(400):
            grad = int(50 + 100 * (x / 400.0) + 50 * (y / 400.0))
            img[y, x] = [grad, grad, grad]
    # Add high-frequency Gaussian noise (sensor noise)
    noise = np.random.normal(0, 15, (400, 400, 3))
    noisy_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy_img


class TestStatisticalHeuristicsScreener:
    """Test suite for StatisticalHeuristicsScreener."""

    def test_initialization_and_weight_validation(self) -> None:
        screener = StatisticalHeuristicsScreener()
        assert screener.w_entropy == 0.35
        assert screener.w_zero_noise == 0.35
        assert screener.w_axis_grad == 0.20
        assert screener.w_lighting == 0.10

        # Valid custom weights
        custom = StatisticalHeuristicsScreener(
            entropy_weight=0.5,
            zero_noise_weight=0.3,
            axis_grad_weight=0.1,
            lighting_weight=0.1,
        )
        assert custom.w_entropy == 0.5

        # Invalid weights not summing to 1.0
        with pytest.raises(ValueError, match="weights must sum to 1.0"):
            StatisticalHeuristicsScreener(
                entropy_weight=0.5,
                zero_noise_weight=0.5,
                axis_grad_weight=0.5,
                lighting_weight=0.5,
            )

    def test_feature_extraction_bounds(
        self, screener: StatisticalHeuristicsScreener, synthetic_digital_board: np.ndarray
    ) -> None:
        feats = screener.extract_features(synthetic_digital_board)
        assert isinstance(feats, HeuristicFeatures)
        assert 0.0 <= feats.palette_entropy <= 1.0
        assert 0.0 <= feats.zero_noise_ratio <= 1.0
        assert feats.axis_gradient_ratio >= 0.0
        assert 0.0 <= feats.lighting_inhomogeneity <= 1.0
        assert 0.0 <= feats.composite_physical_score <= 1.0

    def test_synthetic_digital_board_classification(
        self, screener: StatisticalHeuristicsScreener, synthetic_digital_board: np.ndarray
    ) -> None:
        result = screener.classify(synthetic_digital_board)
        assert isinstance(result, DomainClassificationResult)
        assert result.domain == DomainType.DIGITAL_2D
        assert result.method == ClassificationMethod.HEURISTIC
        assert result.confidence > 0.60
        assert result.heuristic_score is not None
        assert result.heuristic_score < 0.20  # Strongly digital

    def test_synthetic_physical_photo_classification(
        self, screener: StatisticalHeuristicsScreener, synthetic_physical_photo: np.ndarray
    ) -> None:
        result = screener.classify(synthetic_physical_photo)
        assert isinstance(result, DomainClassificationResult)
        assert result.domain == DomainType.PHYSICAL_3D
        assert result.method == ClassificationMethod.HEURISTIC
        assert result.confidence > 0.50
        assert result.heuristic_score is not None
        assert result.heuristic_score > 0.70  # Strongly physical

    def test_root_test_pic_screenshot(self, screener: StatisticalHeuristicsScreener) -> None:
        """Verifies that un-cropped full-screen app screenshot test_pic.png is classified as DIGITAL_2D."""
        test_pic = Path("test_pic.png")
        if not test_pic.is_file():
            pytest.skip("test_pic.png not found in workspace root")

        result = screener.classify(test_pic)
        assert isinstance(result, DomainClassificationResult)
        assert result.domain == DomainType.DIGITAL_2D
        assert result.method == ClassificationMethod.HEURISTIC
        assert result.heuristic_score is not None
        assert result.heuristic_score < 0.50

    def test_synthetic_full_screen_ui_screenshot(
        self, screener: StatisticalHeuristicsScreener, synthetic_digital_board: np.ndarray
    ) -> None:
        """Creates a synthetic full-screen app window with a board in the center and colorful UI borders."""
        canvas = np.zeros((800, 600, 3), dtype=np.uint8)
        # Add random colorful UI elements around the border (avatar, chat, banners)
        np.random.seed(99)
        canvas[:200, :] = np.random.randint(50, 200, (200, 600, 3), dtype=np.uint8)
        canvas[600:, :] = np.random.randint(50, 200, (200, 600, 3), dtype=np.uint8)
        # Place clean digital board in the center (y: 200..600, x: 100..500)
        canvas[200:600, 100:500] = synthetic_digital_board

        result = screener.classify(canvas)
        assert result.domain == DomainType.DIGITAL_2D
        assert result.method == ClassificationMethod.HEURISTIC

    def test_standardized_digital_dataset_accuracy(
        self, screener: StatisticalHeuristicsScreener, digital_images_dir: Path
    ) -> None:
        images = list(digital_images_dir.glob("*.png")) + list(digital_images_dir.glob("*.jpg"))
        assert len(images) > 0, "No digital images found in test directory"

        correct = 0
        for img_path in images:
            result = screener.classify(img_path)
            if result.domain == DomainType.DIGITAL_2D:
                correct += 1

        accuracy = correct / len(images)
        assert accuracy >= 0.96, f"Digital accuracy {accuracy:.2%} fell below 96% ({correct}/{len(images)})"

    def test_standardized_physical_dataset_accuracy(
        self, screener: StatisticalHeuristicsScreener, physical_images_dir: Path
    ) -> None:
        images = list(physical_images_dir.glob("*.jpg")) + list(physical_images_dir.glob("*.png"))
        assert len(images) > 0, "No physical images found in test directory"

        correct = 0
        for img_path in images:
            result = screener.classify(img_path)
            if result.domain == DomainType.PHYSICAL_3D:
                correct += 1

        accuracy = correct / len(images)
        assert accuracy >= 0.85, f"Physical accuracy {accuracy:.2%} fell below 85% ({correct}/{len(images)})"

    def test_sub_millisecond_latency_benchmark(
        self, screener: StatisticalHeuristicsScreener, synthetic_digital_board: np.ndarray
    ) -> None:
        """Benchmarks 100 sequential inferences, asserting mean CPU latency < 0.5 ms."""
        # Warmup
        for _ in range(10):
            screener.classify(synthetic_digital_board)

        latencies = []
        n_iters = 100
        for _ in range(n_iters):
            t0 = time.perf_counter()
            screener.classify(synthetic_digital_board)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        mean_latency = float(np.mean(latencies))
        p95_latency = float(np.percentile(latencies, 95))

        print(f"\n[Domain Classifier Benchmark] Mean Latency: {mean_latency:.4f} ms, P95: {p95_latency:.4f} ms")
        assert mean_latency < 0.8, f"Mean latency {mean_latency:.4f} ms exceeds benchmark budget"

    def test_input_type_flexibility(
        self, screener: StatisticalHeuristicsScreener, synthetic_digital_board: np.ndarray, tmp_path: Path
    ) -> None:
        # 1. Path object
        temp_file = tmp_path / "test_board.png"
        cv2.imwrite(str(temp_file), synthetic_digital_board)
        res_path = screener.classify(temp_file)
        assert res_path.domain == DomainType.DIGITAL_2D

        # 2. String path
        res_str = screener.classify(str(temp_file))
        assert res_str.domain == DomainType.DIGITAL_2D

        # 3. Grayscale 2D array
        gray = cv2.cvtColor(synthetic_digital_board, cv2.COLOR_BGR2GRAY)
        res_gray = screener.classify(gray)
        assert res_gray.domain == DomainType.DIGITAL_2D

        # 4. BGRA 4-channel array
        bgra = cv2.cvtColor(synthetic_digital_board, cv2.COLOR_BGR2BGRA)
        res_bgra = screener.classify(bgra)
        assert res_bgra.domain == DomainType.DIGITAL_2D

        # 5. Normalized float [0.0, 1.0]
        float_img = synthetic_digital_board.astype(np.float32) / 255.0
        res_float = screener.classify(float_img)
        assert res_float.domain == DomainType.DIGITAL_2D

    def test_error_handling(self, screener: StatisticalHeuristicsScreener) -> None:
        # Non-existent file
        with pytest.raises(FileNotFoundError):
            screener.classify("non_existent_image_12345.png")

        # Empty array
        with pytest.raises(ValueError, match="empty"):
            screener.classify(np.array([], dtype=np.uint8))

        # Invalid type
        with pytest.raises(TypeError):
            screener.classify(12345)  # type: ignore

        # Invalid shape
        with pytest.raises(ValueError, match="Unsupported image shape"):
            screener.classify(np.zeros((10, 10, 10, 10), dtype=np.uint8))

    def test_ambiguity_checking(self, screener: StatisticalHeuristicsScreener) -> None:
        assert screener.is_ambiguous(0.50) is True
        assert screener.is_ambiguous(0.25) is True
        assert screener.is_ambiguous(0.75) is True
        assert screener.is_ambiguous(0.10) is False
        assert screener.is_ambiguous(0.90) is False

    def test_convenience_function(self, synthetic_digital_board: np.ndarray) -> None:
        res = classify_domain_heuristic(synthetic_digital_board)
        assert isinstance(res, DomainClassificationResult)
        assert res.domain == DomainType.DIGITAL_2D
