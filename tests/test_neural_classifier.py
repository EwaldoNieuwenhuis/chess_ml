"""
Unit and benchmark test suite for the Lightweight ONNX MicroCNN Domain Classifier (US-3.1.2 & ADR-009).
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.domain_classifier.micro_cnn import (
    MicroCNN,
    build_domain_classifier_model,
)
from src.domain_classifier.neural_classifier import (
    NeuralDomainClassifier,
    classify_domain_neural,
)
from src.domain_classifier.train_micro_cnn import (
    generate_synthetic_digital_board,
    generate_synthetic_physical_photo,
    generate_synthetic_screen_recapture,
    train_and_export_micro_cnn,
)
from src.schemas.contracts import (
    ClassificationMethod,
    DomainClassificationResult,
    DomainType,
)


@pytest.fixture(scope="session")
def onnx_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Provides a valid ONNX model path, training a fast checkpoint if none exists."""
    default_path = Path("src/domain_classifier/weights/domain_classifier_microcnn.onnx")
    if default_path.is_file():
        return default_path

    # Train and export a fast model for the test session
    temp_dir = tmp_path_factory.mktemp("models")
    export_path = temp_dir / "test_microcnn.onnx"
    train_and_export_micro_cnn(
        output_onnx_path=export_path,
        epochs=5,
        batch_size=32,
        num_train_samples=400,
        num_val_samples=100,
    )
    return export_path


@pytest.fixture
def neural_classifier(onnx_model_path: Path) -> NeuralDomainClassifier:
    return NeuralDomainClassifier(model_path=onnx_model_path, device="cpu")


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


# ---------------------------------------------------------------------------
# Architecture Tests
# ---------------------------------------------------------------------------


class TestMicroCNNArchitecture:
    """Validates MicroCNN PyTorch module specifications and parameter budgets."""

    def test_forward_pass_shapes(self) -> None:
        model = MicroCNN(num_classes=2)
        model.eval()

        dummy_input = torch.randn(4, 3, 128, 128, dtype=torch.float32)
        output = model(dummy_input)

        assert output.shape == (4, 2), f"Expected shape (4, 2), got {output.shape}"

    def test_parameter_count_budget(self) -> None:
        model = MicroCNN(num_classes=2)
        total_params = sum(p.numel() for p in model.parameters())

        print(f"[MicroCNN Param Count]: {total_params:,} parameters")
        assert total_params < 250_000, f"Parameter count {total_params} exceeds 250k budget"
        assert total_params > 50_000, "Parameter count is unexpectedly low"

    def test_model_builder_factory(self) -> None:
        model = build_domain_classifier_model(arch="micro_cnn", num_classes=2)
        assert isinstance(model, MicroCNN)

        with pytest.raises(ValueError, match="Unknown architecture"):
            build_domain_classifier_model(arch="invalid_arch")  # type: ignore


# ---------------------------------------------------------------------------
# Neural Classifier & ONNX Inference Tests
# ---------------------------------------------------------------------------


class TestNeuralDomainClassifier:
    """Validates ONNX Runtime inference, accuracy, and latency bounds."""

    def test_onnx_model_size_constraint(self, onnx_model_path: Path) -> None:
        """Verifies the exported ONNX model file size is strictly < 1.5 MB."""
        assert onnx_model_path.is_file(), f"ONNX file not found at {onnx_model_path}"
        size_mb = onnx_model_path.stat().st_size / (1024 * 1024)
        print(f"[ONNX File Size]: {size_mb:.3f} MB")
        assert size_mb < 1.5, f"ONNX model size {size_mb:.2f} MB exceeds 1.5 MB limit"

    def test_synthetic_digital_board_classification(
        self, neural_classifier: NeuralDomainClassifier
    ) -> None:
        board = generate_synthetic_digital_board(size=128)
        result = neural_classifier.classify(board)

        assert isinstance(result, DomainClassificationResult)
        assert result.domain == DomainType.DIGITAL_2D
        assert result.method == ClassificationMethod.NEURAL
        assert result.confidence >= 0.50
        assert result.latency_ms > 0.0

    def test_synthetic_physical_photo_classification(
        self, neural_classifier: NeuralDomainClassifier
    ) -> None:
        photo = generate_synthetic_physical_photo(size=128)
        result = neural_classifier.classify(photo)

        assert isinstance(result, DomainClassificationResult)
        assert result.domain == DomainType.PHYSICAL_3D
        assert result.method == ClassificationMethod.NEURAL
        assert result.confidence >= 0.50

    def test_screen_recapture_moire_classification(
        self, neural_classifier: NeuralDomainClassifier
    ) -> None:
        """
        Critical Acceptance Criteria:
        Recaptured monitor screen photos (with Moiré fringes) must be classified
        to DomainType.PHYSICAL_3D to enforce homography rectification.
        """
        recaptured = generate_synthetic_screen_recapture(size=128)
        result = neural_classifier.classify(recaptured)

        assert isinstance(result, DomainClassificationResult)
        assert result.domain == DomainType.PHYSICAL_3D, (
            f"Screen recapture incorrectly classified as {result.domain}. "
            f"Recaptured monitor displays must route to PHYSICAL_3D for homography."
        )
        assert result.method == ClassificationMethod.NEURAL

    def test_sub_2_5_ms_cpu_latency_benchmark(
        self, neural_classifier: NeuralDomainClassifier
    ) -> None:
        """Benchmarks 100 sequential inferences on CPU, asserting mean latency < 2.5 ms."""
        dummy = generate_synthetic_digital_board(size=128)

        # Warmup passes
        for _ in range(10):
            neural_classifier.classify(dummy)

        latencies: list[float] = []
        n_iters = 100
        for _ in range(n_iters):
            t0 = time.perf_counter()
            neural_classifier.classify(dummy)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        mean_latency = float(np.mean(latencies))
        p95_latency = float(np.percentile(latencies, 95))

        print(f"\n[MicroCNN Benchmark] Mean Latency: {mean_latency:.4f} ms, P95: {p95_latency:.4f} ms")
        assert mean_latency < 2.5, f"Mean latency {mean_latency:.4f} ms exceeds 2.5 ms acceptance threshold"

    def test_input_type_flexibility(
        self, neural_classifier: NeuralDomainClassifier, tmp_path: Path
    ) -> None:
        board = generate_synthetic_digital_board(size=128)

        # 1. Path object
        temp_file = tmp_path / "board_test.png"
        cv2.imwrite(str(temp_file), board)
        res_path = neural_classifier.classify(temp_file)
        assert isinstance(res_path, DomainClassificationResult)

        # 2. String path
        res_str = neural_classifier.classify(str(temp_file))
        assert isinstance(res_str, DomainClassificationResult)

        # 3. Grayscale 2D
        gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        res_gray = neural_classifier.classify(gray)
        assert isinstance(res_gray, DomainClassificationResult)

        # 4. BGRA 4-channel
        bgra = cv2.cvtColor(board, cv2.COLOR_BGR2BGRA)
        res_bgra = neural_classifier.classify(bgra)
        assert isinstance(res_bgra, DomainClassificationResult)

        # 5. Float [0.0, 1.0]
        float_img = board.astype(np.float32) / 255.0
        res_float = neural_classifier.classify(float_img)
        assert isinstance(res_float, DomainClassificationResult)

    def test_error_handling(self, neural_classifier: NeuralDomainClassifier) -> None:
        # Non-existent file
        with pytest.raises(FileNotFoundError):
            neural_classifier.classify("non_existent_file_99999.png")

        # Empty array
        with pytest.raises(ValueError, match="empty"):
            neural_classifier.classify(np.array([], dtype=np.uint8))

        # Invalid type
        with pytest.raises(TypeError):
            neural_classifier.classify(12345)  # type: ignore

        # Invalid shape
        with pytest.raises(ValueError, match="Unsupported image shape"):
            neural_classifier.classify(np.zeros((10, 10, 10, 10), dtype=np.uint8))

    def test_convenience_function(self, onnx_model_path: Path) -> None:
        board = generate_synthetic_digital_board(size=128)
        res = classify_domain_neural(board, model_path=onnx_model_path)
        assert isinstance(res, DomainClassificationResult)
        assert res.method == ClassificationMethod.NEURAL
