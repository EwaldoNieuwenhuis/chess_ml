"""
ONNX Runtime Neural Domain Classifier (Tier-2 Fallback).

Fulfills US-3.1.2 and ADR-009:
Provides an ultra-low latency, zero-PyTorch dependency neural classifier
running the exported MicroCNN ONNX model via ONNX Runtime.

Triggered when Tier-1 heuristic screener confidence is ambiguous (0.20 <= S <= 0.80).
Guarantees sub-2.5 ms CPU inference (< 0.4 ms GPU) and correct routing of screen
recaptures to Domain.PHYSICAL_3D.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import onnxruntime as ort

from src.schemas.contracts import (
    ClassificationMethod,
    DomainClassificationResult,
    DomainType,
)


class NeuralDomainClassifier:
    """
    Tier-2 Neural Fallback Domain Classifier.
    
    Executes the trained and exported MicroCNN ONNX model via ONNX Runtime.
    Accepts arbitrary input image representations (filepath, BGR/RGB array, float)
    and returns a typed DomainClassificationResult.
    """

    DEFAULT_MODEL_PATH: Path = (
        Path(__file__).parent / "weights" / "domain_classifier_microcnn.onnx"
    )
    INPUT_SIZE: int = 128

    # ImageNet normalization parameters
    MEAN: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    STD: np.ndarray = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    def __init__(
        self,
        model_path: Union[str, Path, None] = None,
        device: str = "cpu",
        num_threads: int = 1,
    ) -> None:
        """
        Initializes the ONNX Runtime Inference Session.
        
        Args:
            model_path: Path to the .onnx model weights. If None, uses DEFAULT_MODEL_PATH.
            device: Execution target ('cpu' or 'cuda').
            num_threads: Intra-op thread count for CPU execution.
        """
        self.model_path = Path(model_path) if model_path is not None else self.DEFAULT_MODEL_PATH
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Domain classifier ONNX model not found at {self.model_path}. "
                f"Please run the training script to export weights: "
                f"uv run python -m src.domain_classifier.train_micro_cnn"
            )

        # Configure session options for minimal latency
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.intra_op_num_threads = num_threads

        # Select execution providers
        providers: list[str] = []
        if device.lower() in ("cuda", "gpu") and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    @staticmethod
    def _prepare_image(image: Union[np.ndarray, str, Path]) -> np.ndarray:
        """Loads and normalizes the input image to uint8 BGR format."""
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.is_file():
                raise FileNotFoundError(f"Image file not found: {path}")
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to decode image from path: {path}")
            return img

        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected image as np.ndarray, str, or Path, got {type(image)}")

        if image.size == 0:
            raise ValueError("Input image array is empty")

        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating):
                if image.max() <= 1.0:
                    image = (image * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    image = image.clip(0, 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3:
            if image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            elif image.shape[2] == 1:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 3:
                return image
            else:
                raise ValueError(f"Unsupported number of channels: {image.shape[2]}")
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

    def preprocess(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Preprocesses a BGR uint8 image into an ONNX NCHW float32 normalized tensor.
        
        Args:
            bgr_image: uint8 numpy array (H, W, 3) in BGR space.
            
        Returns:
            Normalized float32 tensor of shape (1, 3, 128, 128).
        """
        # Resize to 128x128 thumbnail
        h, w = bgr_image.shape[:2]
        if h != self.INPUT_SIZE or w != self.INPUT_SIZE:
            resized = cv2.resize(
                bgr_image,
                (self.INPUT_SIZE, self.INPUT_SIZE),
                interpolation=cv2.INTER_AREA if (h > self.INPUT_SIZE or w > self.INPUT_SIZE) else cv2.INTER_LINEAR,
            )
        else:
            resized = bgr_image

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Convert HWC uint8 -> CHW float32 [0.0, 1.0]
        chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
        nchw = np.expand_dims(chw, axis=0)

        # Standard ImageNet normalization
        normalized = (nchw - self.MEAN) / self.STD
        return np.ascontiguousarray(normalized, dtype=np.float32)

    def classify(self, image: Union[np.ndarray, str, Path]) -> DomainClassificationResult:
        """
        Classifies the input image into DIGITAL_2D or PHYSICAL_3D using ONNX Runtime.
        
        Args:
            image: Image file path or numpy array (BGR / RGB / grayscale).
            
        Returns:
            DomainClassificationResult contract with domain, confidence, and latency.
        """
        t0 = time.perf_counter()

        # 1. Prepare and preprocess image
        bgr = self._prepare_image(image)
        input_tensor = self.preprocess(bgr)

        # 2. Run ONNX Runtime forward pass
        raw_outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor},
        )
        logits = raw_outputs[0][0]  # Shape: (2,)

        # 3. Compute Softmax Probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        # Class 0: DIGITAL_2D, Class 1: PHYSICAL_3D
        digital_prob = float(probs[0])
        physical_prob = float(probs[1])

        if physical_prob >= 0.50:
            domain = DomainType.PHYSICAL_3D
            confidence = physical_prob
        else:
            domain = DomainType.DIGITAL_2D
            confidence = digital_prob

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return DomainClassificationResult(
            domain=domain,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            method=ClassificationMethod.NEURAL,
            latency_ms=round(latency_ms, 4),
            heuristic_score=None,
        )


def classify_domain_neural(
    image: Union[np.ndarray, str, Path],
    model_path: Union[str, Path, None] = None,
) -> DomainClassificationResult:
    """
    Convenience function for direct neural domain classification.
    
    Args:
        image: Image file path or numpy array.
        model_path: Optional path to custom ONNX weights.
        
    Returns:
        DomainClassificationResult
    """
    classifier = NeuralDomainClassifier(model_path=model_path)
    return classifier.classify(image)
