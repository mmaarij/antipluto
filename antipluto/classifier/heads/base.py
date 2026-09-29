"""
antipluto.classifier.heads.base
=============================
Abstract base class and contract for Anti-PLUTO fusion classifier heads.
"""

from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)


def concatenate_features(
    stylometric_features: sparse.csr_matrix,
    semantic_embeddings: np.ndarray,
) -> sparse.csr_matrix:
    """Concatenate sparse TF-IDF matrix and dense embedding matrix into a unified CSR matrix."""
    dense_sparse = sparse.csr_matrix(semantic_embeddings)
    return sparse.hstack([stylometric_features, dense_sparse], format="csr")


class BaseFusionHead(abc.ABC):
    """Abstract base class defining the contract for all fusion classification heads."""

    name: str = "base"

    @abc.abstractmethod
    def fit(
        self,
        X_train_stylo: sparse.csr_matrix,
        X_train_sem: np.ndarray,
        y_train: Union[List[int], np.ndarray],
        X_val_stylo: Optional[sparse.csr_matrix] = None,
        X_val_sem: Optional[np.ndarray] = None,
        y_val: Optional[Union[List[int], np.ndarray]] = None,
        verbose: bool = True,
    ) -> BaseFusionHead:
        """Fit classifier on concatenated stylometric and semantic features."""
        pass

    @abc.abstractmethod
    def predict(
        self,
        X_stylo: sparse.csr_matrix,
        X_sem: np.ndarray,
    ) -> np.ndarray:
        """Predict 4-class labels (0..3) for input features."""
        pass

    @abc.abstractmethod
    def predict_proba(
        self,
        X_stylo: sparse.csr_matrix,
        X_sem: np.ndarray,
    ) -> np.ndarray:
        """Predict probability distribution shape (N, 4) for input features."""
        pass

    @abc.abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Save model checkpoints and configuration."""
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, path: Union[str, Path]) -> BaseFusionHead:
        """Load trained model from disk."""
        pass

    @abc.abstractmethod
    def export_onnx(
        self,
        output_path: Union[str, Path],
        num_features: int = 100768,
    ) -> Path:
        """Export trained model to portable ONNX format."""
        pass
