"""
Multi-Feature Statistical Heuristics Screener (Tier-1 Fast Path).

Fulfills US-3.1.1 and ADR-009:
Executes a sub-millisecond, zero-weight statistical classifier combining 4 normalized
metrics across a dual-window (Global Frame + Central Board ROI) on downscaled 128x128
thumbnails to classify images as DIGITAL_2D or PHYSICAL_3D.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import numpy as np

from src.schemas.contracts import (
    ClassificationMethod,
    DomainClassificationResult,
    DomainType,
)


@dataclass(frozen=True)
class HeuristicFeatures:
    """Extracted statistical features for input domain screening."""

    palette_entropy: float  # H_norm in [0.0, 1.0] (HSV 64-bin Shannon entropy)
    zero_noise_ratio: float  # ZNR in [0.0, 1.0] (fraction of flat zero-noise patches)
    axis_gradient_ratio: float  # AGE in [0.0, 1.0] (patch texture / gradient energy indicator)
    lighting_inhomogeneity: float  # LH in [0.0, 1.0] (quadrant luminance variance)
    composite_physical_score: float  # S in [0.0, 1.0] (physical probability)


class StatisticalHeuristicsScreener:
    """
    Sub-millisecond Tier-1 statistical domain classifier.

    Extracts 4 complementary visual heuristics on downscaled 128x128 images:
    1. HSV Palette Shannon Entropy (quantized into 64 bins)
    2. Zero-Noise Flat Patch Ratio (sensor photon noise absence detection)
    3. Patch Texture / Gradient Variation (3D piece and shadow complexity)
    4. Lighting Inhomogeneity (ambient illumination falloff & shadow variance)

    Applies the Dual-Window Strategy (ADR-009) to robustly classify both cropped
    boards and full-screen application screenshots with surrounding UI/avatars.
    """

    THUMBNAIL_SIZE: int = 128
    NOISE_THRESHOLD: float = 1.0  # Max std-dev to qualify as a zero-noise patch
    
    # Feature weights summing to 1.0
    WEIGHT_ENTROPY: float = 0.35
    WEIGHT_ZERO_NOISE: float = 0.35
    WEIGHT_AXIS_GRADIENT: float = 0.20
    WEIGHT_LIGHTING: float = 0.10

    # Ambiguity bounds for Tier-2 fallback routing
    AMBIGUOUS_LOWER: float = 0.25
    AMBIGUOUS_UPPER: float = 0.75

    def __init__(
        self,
        entropy_weight: float = WEIGHT_ENTROPY,
        zero_noise_weight: float = WEIGHT_ZERO_NOISE,
        axis_grad_weight: float = WEIGHT_AXIS_GRADIENT,
        lighting_weight: float = WEIGHT_LIGHTING,
        noise_threshold: float = NOISE_THRESHOLD,
    ) -> None:
        total_w = entropy_weight + zero_noise_weight + axis_grad_weight + lighting_weight
        if abs(total_w - 1.0) > 1e-4:
            raise ValueError(f"Feature weights must sum to 1.0, got {total_w}")
        self.w_entropy = entropy_weight
        self.w_zero_noise = zero_noise_weight
        self.w_axis_grad = axis_grad_weight
        self.w_lighting = lighting_weight
        self.noise_threshold = noise_threshold

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
                img = np.clip(image * 255.0 if image.max() <= 1.0 else image, 0, 255).astype(np.uint8)
            else:
                img = np.clip(image, 0, 255).astype(np.uint8)
        else:
            img = image

        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.ndim == 3 and img.shape[2] == 1:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 3:
            return img
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

    def _evaluate_single_thumbnail(self, thumb_bgr: np.ndarray) -> tuple[float, HeuristicFeatures]:
        """Extracts and normalizes features on a 128x128 BGR thumbnail."""
        thumb_gray = cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2GRAY)
        thumb_hsv = cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2HSV)

        # 1. Normalized HSV Palette Shannon Entropy (64 bins)
        h_bin = np.clip(thumb_hsv[:, :, 0] // 23, 0, 7)
        s_bin = np.clip(thumb_hsv[:, :, 1] // 64, 0, 3)
        v_bin = np.clip(thumb_hsv[:, :, 2] // 128, 0, 1)
        bin_idx = (h_bin * 8 + s_bin * 2 + v_bin).astype(np.int32)

        counts = np.bincount(bin_idx.ravel(), minlength=64)
        non_zero_counts = counts[counts > 0]
        p = non_zero_counts / float(self.THUMBNAIL_SIZE * self.THUMBNAIL_SIZE)
        entropy = -np.sum(p * np.log2(p))
        h_norm = float(np.clip(entropy / np.log2(64.0), 0.0, 1.0))
        s_entropy = float(np.clip((h_norm - 0.22) / 0.10, 0.0, 1.0))

        # 2. Zero-Noise Flat Patch Ratio (256 non-overlapping 8x8 patches)
        patches = thumb_gray.reshape(16, 8, 16, 8).swapaxes(1, 2).reshape(256, 8, 8).astype(np.float32)
        patch_stds = np.std(patches, axis=(1, 2))
        zero_noise_count = np.count_nonzero(patch_stds < self.noise_threshold)
        znr = float(zero_noise_count / 256.0)
        s_noise = float(np.clip((0.35 - znr) / 0.20, 0.0, 1.0))

        # 3. Patch Texture & Gradient Variation
        med_std = float(np.median(patch_stds))
        s_texture = float(np.clip((med_std - 25.0) / 20.0, 0.0, 1.0))
        texture_norm = float(np.clip(med_std / 60.0, 0.0, 1.0))

        # 4. Lighting Inhomogeneity (Quadrant Luminance Variance)
        q_tl = float(np.mean(thumb_gray[:64, :64]))
        q_tr = float(np.mean(thumb_gray[:64, 64:]))
        q_bl = float(np.mean(thumb_gray[64:, :64]))
        q_br = float(np.mean(thumb_gray[64:, 64:]))
        quad_std = float(np.std([q_tl, q_tr, q_bl, q_br]))
        lh = float(np.clip(quad_std / 35.0, 0.0, 1.0))
        s_lighting = lh

        score = float(
            np.clip(
                self.w_entropy * s_entropy
                + self.w_zero_noise * s_noise
                + self.w_axis_grad * s_texture
                + self.w_lighting * s_lighting,
                0.0,
                1.0,
            )
        )

        feats = HeuristicFeatures(
            palette_entropy=h_norm,
            zero_noise_ratio=znr,
            axis_gradient_ratio=texture_norm,
            lighting_inhomogeneity=lh,
            composite_physical_score=score,
        )
        return score, feats

    def extract_features(self, image: Union[np.ndarray, str, Path]) -> HeuristicFeatures:
        """
        Extracts the 4-dimensional normalized heuristic feature vector.

        Applies the Dual-Window Strategy (ADR-009) to robustly classify both cropped
        boards and full-screen application screenshots with surrounding UI/avatars.
        """
        img_bgr = self._prepare_image(image)
        h, w, _ = img_bgr.shape

        if h == self.THUMBNAIL_SIZE and w == self.THUMBNAIL_SIZE:
            _, feats = self._evaluate_single_thumbnail(img_bgr)
            return feats

        # Full-size image: evaluate global thumbnail
        thumb_global = cv2.resize(
            img_bgr,
            (self.THUMBNAIL_SIZE, self.THUMBNAIL_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )
        score_global, feats_global = self._evaluate_single_thumbnail(thumb_global)

        # Dual-Window Strategy (ADR-009): Evaluate central 70% ROI for un-cropped app screenshots
        ch0, ch1 = int(h * 0.15), int(h * 0.85)
        cw0, cw1 = int(w * 0.15), int(w * 0.85)
        crop_center = img_bgr[ch0:ch1, cw0:cw1]

        if crop_center.shape[0] >= 32 and crop_center.shape[1] >= 32:
            thumb_center = cv2.resize(
                crop_center,
                (self.THUMBNAIL_SIZE, self.THUMBNAIL_SIZE),
                interpolation=cv2.INTER_LINEAR,
            )
            score_center, feats_center = self._evaluate_single_thumbnail(thumb_center)

            if score_center < score_global and score_center < 0.40:
                return feats_center

        return feats_global

    def is_ambiguous(
        self,
        score: float,
        lower: float = AMBIGUOUS_LOWER,
        upper: float = AMBIGUOUS_UPPER,
    ) -> bool:
        """Returns True if the heuristic score falls in the ambiguous zone for Tier-2 routing."""
        return lower <= score <= upper

    def classify(self, image: Union[np.ndarray, str, Path]) -> DomainClassificationResult:
        """
        Classifies the input image into DIGITAL_2D or PHYSICAL_3D.

        Returns a typed DomainClassificationResult with calibrated confidence score and latency.
        """
        t0 = time.perf_counter()
        features = self.extract_features(image)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        score = features.composite_physical_score
        
        # Decision boundary: S >= 0.50 -> PHYSICAL_3D, else DIGITAL_2D
        if score >= 0.50:
            domain = DomainType.PHYSICAL_3D
            confidence = float(np.clip(2.0 * (score - 0.50), 0.0, 1.0))
        else:
            domain = DomainType.DIGITAL_2D
            confidence = float(np.clip(2.0 * (0.50 - score), 0.0, 1.0))

        return DomainClassificationResult(
            domain=domain,
            confidence=confidence,
            method=ClassificationMethod.HEURISTIC,
            latency_ms=latency_ms,
            heuristic_score=score,
        )


# Module-level convenience function
def classify_domain_heuristic(image: Union[np.ndarray, str, Path]) -> DomainClassificationResult:
    """Convenience helper to classify domain using the default statistical heuristics screener."""
    screener = StatisticalHeuristicsScreener()
    return screener.classify(image)
