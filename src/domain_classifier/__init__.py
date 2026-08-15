"""
Domain Classifier Module.

Classifies whether an input image is a 2D digital screenshot or a 3D physical camera image.
Implements the Two-Tier Cascaded Classifier Architecture (ADR-009).
"""

from src.domain_classifier.heuristic_screener import (
    HeuristicFeatures,
    StatisticalHeuristicsScreener,
    classify_domain_heuristic,
)

# Default domain classifier alias
DomainClassifier = StatisticalHeuristicsScreener

__all__ = [
    "DomainClassifier",
    "HeuristicFeatures",
    "StatisticalHeuristicsScreener",
    "classify_domain_heuristic",
]
