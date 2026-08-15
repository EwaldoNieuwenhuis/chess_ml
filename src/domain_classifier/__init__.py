"""
Domain Classifier Module.

Classifies whether an input image is a 2D digital screenshot or a 3D physical camera image.
Implements the Two-Tier Cascaded Classifier Architecture (ADR-009):
- Tier-1: Multi-Feature Statistical Screener (< 0.4 ms, zero-weight fast path)
- Tier-2: Lightweight ONNX MicroCNN Fallback (< 0.6 MB, sub-2.5 ms ambiguous router)
"""

from src.domain_classifier.heuristic_screener import (
    HeuristicFeatures,
    StatisticalHeuristicsScreener,
    classify_domain_heuristic,
)
from src.domain_classifier.micro_cnn import (
    MicroCNN,
    build_domain_classifier_model,
)
from src.domain_classifier.neural_classifier import (
    NeuralDomainClassifier,
    classify_domain_neural,
)

# Default domain classifier alias
DomainClassifier = StatisticalHeuristicsScreener

__all__ = [
    "DomainClassifier",
    "HeuristicFeatures",
    "MicroCNN",
    "NeuralDomainClassifier",
    "StatisticalHeuristicsScreener",
    "build_domain_classifier_model",
    "classify_domain_heuristic",
    "classify_domain_neural",
]
