"""
antipluto.classifier.heads.xgboost_head
=====================================
XGBoost gradient-boosted tree fusion head.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from scipy import sparse
import xgboost as xgb

from antipluto.classifier.heads.base import BaseFusionHead, concatenate_features

logger = logging.getLogger(__name__)


class XGBoostFusionHead(BaseFusionHead):
    """XGBoost gradient-boosted decision tree classifier."""

    name: str = "xgboost"

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.08,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        device: Optional[str] = None,
        early_stopping_rounds: int = 30,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state

        if device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            objective="multi:softprob",
            num_class=4,
            tree_method="hist",
            device=self.device,
            eval_metric=["mlogloss", "merror"],
            early_stopping_rounds=self.early_stopping_rounds,
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
    ) -> XGBoostFusionHead:
        logger.info("Assembling concatenated feature matrices for XGBoost...")
        X_train = concatenate_features(X_train_stylo, X_train_sem)
        y_train = np.asarray(y_train)

        eval_set = None
        if X_val_stylo is not None and X_val_sem is not None and y_val is not None:
            X_val = concatenate_features(X_val_stylo, X_val_sem)
            y_val = np.asarray(y_val)
            eval_set = [(X_train, y_train), (X_val, y_val)]

        logger.info(
            "Fitting XGBoost (%d samples, %d features, device=%s)...",
            X_train.shape[0],
            X_train.shape[1],
            self.device,
        )

        try:
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=50 if verbose else False,
            )
        except Exception as exc:
            err_msg = str(exc).lower()
            if "out of memory" in err_msg or "cudaerrormemoryallocation" in err_msg:
                logger.warning(
                    "CUDA out of memory in XGBoost. Falling back to multi-threaded CPU..."
                )
                self.device = "cpu"
                self.model = xgb.XGBClassifier(
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree,
                    objective="multi:softprob",
                    num_class=4,
                    tree_method="hist",
                    device="cpu",
                    n_jobs=-1,
                    eval_metric=["mlogloss", "merror"],
                    early_stopping_rounds=self.early_stopping_rounds,
                    random_state=self.random_state,
                )
                self.model.fit(
                    X_train,
                    y_train,
                    eval_set=eval_set,
                    verbose=50 if verbose else False,
                )
            else:
                raise exc

        self._is_fitted = True
        best_iteration = getattr(self.model, "best_iteration", self.n_estimators)
        logger.info("XGBoost training finished. Best iteration: %d", best_iteration)
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
        model_file = path.with_suffix(".json") if path.suffix != ".json" else path
        self.model.save_model(str(model_file))

        meta_file = model_file.with_name(f"{model_file.stem}_meta.joblib")
        joblib.dump(
            {
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "device": self.device,
                "is_fitted": self._is_fitted,
            },
            meta_file,
        )
        logger.info("Saved XGBoostFusionHead model to %s", model_file)

    @classmethod
    def load(cls, path: Union[str, Path]) -> XGBoostFusionHead:
        model_file = Path(path)
        if model_file.suffix != ".json":
            model_file = model_file.with_suffix(".json")

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        meta_file = model_file.with_name(f"{model_file.stem}_meta.joblib")
        if meta_file.exists():
            meta = joblib.load(meta_file)
            inst = cls(
                n_estimators=meta.get("n_estimators", 500),
                learning_rate=meta.get("learning_rate", 0.08),
                max_depth=meta.get("max_depth", 6),
                device=meta.get("device", "cpu"),
            )
        else:
            inst = cls()

        inst.model.load_model(str(model_file))
        inst._is_fitted = True
        return inst

    def export_onnx(
        self,
        output_path: Union[str, Path],
        num_features: int = 100768,
    ) -> Path:
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        initial_type = [("float_input", FloatTensorType([None, num_features]))]
        booster = self.model.get_booster()

        orig_get_dump = booster.get_dump

        def sanitized_get_dump(*args, **kwargs):
            dumps = orig_get_dump(*args, **kwargs)
            cleaned = []
            for d in dumps:
                d_clean = re.sub(r':\s*-inf\b', ': -1e38', d)
                d_clean = re.sub(r':\s*inf\b', ': 1e38', d_clean)
                cleaned.append(d_clean)
            return cleaned

        booster.get_dump = sanitized_get_dump

        logger.info("Exporting XGBoost to %s via onnxmltools...", output_path)
        onnx_model = onnxmltools.convert_xgboost(
            booster,
            initial_types=initial_type,
            target_opset=14,
        )
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        logger.info("Saved XGBoost ONNX model to %s", output_path)
        return output_path
