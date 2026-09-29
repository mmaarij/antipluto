"""
antipluto.classifier.heads.rf_head
================================
Random Forest bagged decision tree fusion head.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier

from antipluto.classifier.heads.base import BaseFusionHead, concatenate_features

logger = logging.getLogger(__name__)


class RandomForestFusionHead(BaseFusionHead):
    """Random Forest bagged decision tree classifier for fused stylometric + semantic features."""

    name: str = "rf"

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: Optional[int] = 30,
        max_features: Union[str, float] = "sqrt",
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_features=self.max_features,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self._is_fitted = False

    def fit(
        self,
        X_train_stylo: sparse.csr_matrix,
        X_train_sem: np.ndarray,
        y_train: Union[List[int], np.ndarray],
        X_val_stylo: Optional[sparse.csr_matrix] = None,
        X_val_sem: Optional[np.ndarray] = None,
        y_val: Optional[Union[List[int], np.ndarray]] = None,
        verbose: bool = True,
    ) -> RandomForestFusionHead:
        logger.info("Assembling concatenated feature matrices for Random Forest...")
        X_train = concatenate_features(X_train_stylo, X_train_sem)
        y_train = np.asarray(y_train)

        logger.info(
            "Fitting Random Forest (%d samples, %d features, trees=%d, max_depth=%s, n_jobs=%d)...",
            X_train.shape[0],
            X_train.shape[1],
            self.n_estimators,
            str(self.max_depth),
            self.n_jobs,
        )

        self.model.fit(X_train, y_train)
        self._is_fitted = True
        logger.info("Random Forest training finished successfully.")
        return self

    def predict(
        self,
        X_stylo: sparse.csr_matrix,
        X_sem: np.ndarray,
    ) -> np.ndarray:
        X = concatenate_features(X_stylo, X_sem)
        return self.model.predict(X)

    def predict_proba(
        self,
        X_stylo: sparse.csr_matrix,
        X_sem: np.ndarray,
    ) -> np.ndarray:
        X = concatenate_features(X_stylo, X_sem)
        return self.model.predict_proba(X)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model_file = path.with_suffix(".joblib") if path.suffix != ".joblib" else path
        joblib.dump(
            {
                "model": self.model,
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "max_features": self.max_features,
                "is_fitted": self._is_fitted,
            },
            model_file,
            compress=3,
        )
        logger.info("Saved RandomForestFusionHead model to %s", model_file)

    @classmethod
    def load(cls, path: Union[str, Path]) -> RandomForestFusionHead:
        model_file = Path(path)
        if model_file.suffix != ".joblib":
            model_file = model_file.with_suffix(".joblib")

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        data = joblib.load(model_file)
        inst = cls(
            n_estimators=data.get("n_estimators", 150),
            max_depth=data.get("max_depth", 30),
            max_features=data.get("max_features", "sqrt"),
        )
        inst.model = data["model"]
        inst._is_fitted = data.get("is_fitted", True)
        return inst

    def export_onnx(
        self,
        output_path: Union[str, Path],
        num_features: int = 100768,
    ) -> Path:
        import skl2onnx
        from skl2onnx.common.data_types import FloatTensorType

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Exporting Random Forest to %s via skl2onnx...", output_path)
        initial_type = [("float_input", FloatTensorType([None, num_features]))]
        onx = skl2onnx.convert_sklearn(
            self.model,
            initial_types=initial_type,
            target_opset=14,
            options={type(self.model): {"zipmap": False}},
        )
        with open(output_path, "wb") as f:
            f.write(onx.SerializeToString())
        logger.info("Saved Random Forest ONNX model to %s", output_path)
        return output_path
