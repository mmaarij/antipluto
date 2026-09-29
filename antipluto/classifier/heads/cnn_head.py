"""
antipluto.classifier.heads.cnn_head
=================================
1D-Convolutional Neural Network (CNN) fusion head.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import sparse
from sklearn.metrics import f1_score
import torch
import torch.nn as nn

from antipluto.classifier.heads.base import BaseFusionHead, concatenate_features

logger = logging.getLogger(__name__)


class PyTorch1DCNN(nn.Module):
    """1D-CNN feature extractor and classifier over projected fused features."""

    def __init__(
        self,
        in_features: int = 100768,
        proj_dim: int = 512,
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_features = in_features
        self.proj_dim = proj_dim
        self.channels = 32
        self.seq_len = proj_dim // self.channels  # 512 // 32 = 16

        self.projection = nn.Sequential(
            nn.Linear(in_features, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
        )

        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=self.channels, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # length 16 -> 8
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # length -> 1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, in_features)
        proj = self.projection(x)  # (batch_size, 512)
        feat_map = proj.view(-1, self.channels, self.seq_len)  # (batch_size, 32, 16)
        conv_out = self.conv_net(feat_map)  # (batch_size, 128, 1)
        logits = self.classifier(conv_out)  # (batch_size, 4)
        return logits


class CNNFusionHead(BaseFusionHead):
    """PyTorch 1D-CNN classifier for fused stylometric + semantic features."""

    name: str = "cnn"

    def __init__(
        self,
        in_features: int = 100768,
        proj_dim: int = 512,
        num_classes: int = 4,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        epochs: int = 15,
        early_stopping_patience: int = 3,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        self.in_features = in_features
        self.proj_dim = proj_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model: Optional[PyTorch1DCNN] = None
        self._is_fitted = False

    def _init_model(self, num_features: int) -> None:
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        self.in_features = num_features
        self.model = PyTorch1DCNN(
            in_features=num_features,
            proj_dim=self.proj_dim,
            num_classes=self.num_classes,
            dropout=self.dropout,
        ).to(self.device)

    def fit(
        self,
        X_train_stylo: sparse.csr_matrix,
        X_train_sem: np.ndarray,
        y_train: Union[List[int], np.ndarray],
        X_val_stylo: Optional[sparse.csr_matrix] = None,
        X_val_sem: Optional[np.ndarray] = None,
        y_val: Optional[Union[List[int], np.ndarray]] = None,
        verbose: bool = True,
    ) -> CNNFusionHead:
        logger.info("Assembling concatenated feature matrices for 1D-CNN...")
        X_train = concatenate_features(X_train_stylo, X_train_sem)
        y_train = np.asarray(y_train, dtype=np.int64)

        has_val = X_val_stylo is not None and X_val_sem is not None and y_val is not None
        if has_val:
            X_val = concatenate_features(X_val_stylo, X_val_sem)
            y_val = np.asarray(y_val, dtype=np.int64)

        num_samples, num_features = X_train.shape
        self._init_model(num_features)

        logger.info(
            "Fitting PyTorch 1D-CNN (%d samples, %d features, device=%s, batch_size=%d)...",
            num_samples,
            num_features,
            self.device,
            self.batch_size,
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        indices = np.arange(num_samples)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            np.random.shuffle(indices)
            total_train_loss = 0.0
            num_batches = int(np.ceil(num_samples / self.batch_size))

            for b in range(num_batches):
                b_start = b * self.batch_size
                b_end = min(b_start + self.batch_size, num_samples)
                batch_idx = indices[b_start:b_end]

                batch_x_dense = X_train[batch_idx].toarray().astype(np.float32)
                batch_x = torch.from_numpy(batch_x_dense).to(self.device)
                batch_y = torch.from_numpy(y_train[batch_idx]).to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * len(batch_idx)

            scheduler.step()
            avg_train_loss = total_train_loss / num_samples

            val_info = ""
            if has_val:
                val_loss, val_f1 = self._eval_dataset(X_val, y_val, criterion)
                val_info = f" | Val Loss: {val_loss:.4f} | Val Macro F1: {val_f1:.4f}"

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        if verbose:
                            logger.info(
                                "Epoch %2d/%2d: Train Loss: %.4f%s [Early Stopping Triggered]",
                                epoch,
                                self.epochs,
                                avg_train_loss,
                                val_info,
                            )
                        break

            if verbose:
                logger.info(
                    "Epoch %2d/%2d: Train Loss: %.4f%s",
                    epoch,
                    self.epochs,
                    avg_train_loss,
                    val_info,
                )

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        self._is_fitted = True
        logger.info("1D-CNN training finished successfully.")
        return self

    def _eval_dataset(
        self,
        X: sparse.csr_matrix,
        y: np.ndarray,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        self.model.eval()
        n = X.shape[0]
        bs = self.batch_size * 2
        total_loss = 0.0
        all_preds = []

        with torch.no_grad():
            for b_start in range(0, n, bs):
                b_end = min(b_start + bs, n)
                batch_x = torch.from_numpy(X[b_start:b_end].toarray().astype(np.float32)).to(self.device)
                batch_y = torch.from_numpy(y[b_start:b_end]).to(self.device)

                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                total_loss += loss.item() * (b_end - b_start)

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)

        avg_loss = total_loss / n
        macro_f1 = float(f1_score(y, all_preds, average="macro"))
        return avg_loss, macro_f1

    def predict(
        self,
        X_stylo: sparse.csr_matrix,
        X_sem: np.ndarray,
    ) -> np.ndarray:
        probs = self.predict_proba(X_stylo, X_sem)
        return np.argmax(probs, axis=1)

    def predict_proba(
        self,
        X_stylo: sparse.csr_matrix,
        X_sem: np.ndarray,
    ) -> np.ndarray:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("CNNFusionHead is not fitted yet.")

        X = concatenate_features(X_stylo, X_sem)
        n = X.shape[0]
        bs = self.batch_size * 2
        self.model.eval()

        probs_list = []
        with torch.no_grad():
            for b_start in range(0, n, bs):
                b_end = min(b_start + bs, n)
                batch_x = torch.from_numpy(X[b_start:b_end].toarray().astype(np.float32)).to(self.device)
                logits = self.model(batch_x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probs_list.append(probs)

        return np.vstack(probs_list)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model_file = path.with_suffix(".pt") if path.suffix != ".pt" else path

        payload = {
            "state_dict": self.model.state_dict() if self.model else None,
            "in_features": self.in_features,
            "proj_dim": self.proj_dim,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
            "is_fitted": self._is_fitted,
        }
        torch.save(payload, model_file)
        logger.info("Saved CNNFusionHead model to %s", model_file)

    @classmethod
    def load(cls, path: Union[str, Path]) -> CNNFusionHead:
        model_file = Path(path)
        if model_file.suffix != ".pt":
            model_file = model_file.with_suffix(".pt")

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        payload = torch.load(model_file, map_location="cpu", weights_only=True)
        inst = cls(
            in_features=payload.get("in_features", 100768),
            proj_dim=payload.get("proj_dim", 512),
            num_classes=payload.get("num_classes", 4),
            dropout=payload.get("dropout", 0.3),
        )
        inst._init_model(inst.in_features)
        inst.model.load_state_dict(payload["state_dict"])
        inst.model.to(inst.device)
        inst._is_fitted = payload.get("is_fitted", True)
        return inst

    def export_onnx(
        self,
        output_path: Union[str, Path],
        num_features: int = 100768,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        class SoftmaxWrapper(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model

            def forward(self, x):
                logits = self.base_model(x)
                return torch.softmax(logits, dim=1)

        logger.info("Exporting 1D-CNN to %s via torch.onnx.export...", output_path)
        model_cpu = self.model.to("cpu")
        model_cpu.eval()
        wrapper = SoftmaxWrapper(model_cpu)
        wrapper.eval()

        dummy_input = torch.zeros((1, num_features), dtype=torch.float32)

        torch.onnx.export(
            wrapper,
            dummy_input,
            str(output_path),
            input_names=["float_input"],
            output_names=["probabilities"],
            dynamic_axes={
                "float_input": {0: "batch_size"},
                "probabilities": {0: "batch_size"},
            },
            opset_version=14,
            do_constant_folding=True,
        )

        self.model.to(self.device)
        logger.info("Saved 1D-CNN ONNX model to %s", output_path)
        return output_path
