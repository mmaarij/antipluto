"""
antipluto.classifier.heads
========================
Modular fusion classification heads for Anti-PLUTO:
- XGBoost (Gradient Boosted Trees)
- Random Forest (Bagged Ensembles)
- MLP (Deep Multi-Layer Perceptron)
- 1D-CNN (Convolutional Neural Network)
"""

from __future__ import annotations

from typing import Any, Dict, Type

from antipluto.classifier.heads.base import BaseFusionHead, concatenate_features
from antipluto.classifier.heads.xgboost_head import XGBoostFusionHead
from antipluto.classifier.heads.rf_head import RandomForestFusionHead
from antipluto.classifier.heads.mlp_head import MLPFusionHead
from antipluto.classifier.heads.cnn_head import CNNFusionHead

REGISTRY: Dict[str, Type[BaseFusionHead]] = {
    "xgboost": XGBoostFusionHead,
    "rf": RandomForestFusionHead,
    "random_forest": RandomForestFusionHead,
    "mlp": MLPFusionHead,
    "cnn": CNNFusionHead,
    "1d_cnn": CNNFusionHead,
}

AVAILABLE_HEADS = ["xgboost", "rf", "mlp", "cnn"]


def get_classifier_head(name: str, **kwargs: Any) -> BaseFusionHead:
    """Factory function to instantiate a fusion classifier head by name."""
    canonical_name = name.strip().lower()
    if canonical_name not in REGISTRY:
        raise ValueError(
            f"Unknown classifier head '{name}'. Available heads: {list(REGISTRY.keys())}"
        )
    cls = REGISTRY[canonical_name]
    return cls(**kwargs)


__all__ = [
    "BaseFusionHead",
    "concatenate_features",
    "XGBoostFusionHead",
    "RandomForestFusionHead",
    "MLPFusionHead",
    "CNNFusionHead",
    "REGISTRY",
    "AVAILABLE_HEADS",
    "get_classifier_head",
]
