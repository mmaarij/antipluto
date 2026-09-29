"""
antipluto.api.service
===================
Singleton inference service managing the lifecycle and concurrent prediction
of the dual-branch feature extractors and fusion classifier heads.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Dict, Optional

import numpy as np
import torch

from antipluto.api.pipeline import EmailIngestionPipeline, PreprocessedEmail
from antipluto.api.schemas import PredictResponse, ProcessingMetadata
from antipluto.classifier.data import CLASS_NAMES
from antipluto.classifier.semantic import SemanticEncoder
from antipluto.classifier.heads import (
    BaseFusionHead,
    XGBoostFusionHead,
    RandomForestFusionHead,
    MLPFusionHead,
    CNNFusionHead,
)
from antipluto.classifier.stylometric import StylometricExtractor

logger = logging.getLogger(__name__)

CLASSIFIER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "xgboost": {
        "class": XGBoostFusionHead,
        "rel_path": "checkpoints/xgboost/xgb_model.json",
    },
    "rf": {
        "class": RandomForestFusionHead,
        "rel_path": "checkpoints/rf/rf_model.joblib",
    },
    "mlp": {
        "class": MLPFusionHead,
        "rel_path": "checkpoints/mlp/mlp_model.pt",
    },
    "cnn": {
        "class": CNNFusionHead,
        "rel_path": "checkpoints/cnn/cnn_model.pt",
    },
}


class InferenceService:
    """Manages pre-loaded model artifacts and executes thread-safe predictions across multiple modes."""

    def __init__(
        self,
        model_dir: Path | str = "models",
        classifier_name: str = "xgboost",
        default_mode: str = "fusion",
        device: Optional[str] = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.classifier_name = classifier_name.lower().strip()
        self.default_mode = default_mode.lower().strip()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if self.classifier_name not in CLASSIFIER_REGISTRY:
            raise ValueError(
                f"Unknown classifier head '{self.classifier_name}'. "
                f"Available: {list(CLASSIFIER_REGISTRY.keys())}"
            )

        if self.default_mode not in ("fusion", "semantic", "stylometric"):
            raise ValueError(
                f"Unknown default mode '{self.default_mode}'. "
                f"Available: ['fusion', 'semantic', 'stylometric']"
            )

        self.pipeline = EmailIngestionPipeline()
        self.stylometric_extractor: Optional[StylometricExtractor] = None
        self.stylometric_classifier: Optional[Any] = None
        self.semantic_encoder: Optional[SemanticEncoder] = None
        self.classifier_head: Optional[BaseFusionHead] = None
        self.is_ready = False

    def load_artifacts(self) -> None:
        """Load TF-IDF, DeBERTa-v3, and classification head into memory."""
        logger.info(
            "Initializing Anti-PLUTO InferenceService (classifier: %s, default_mode: %s, device: %s)...",
            self.classifier_name,
            self.default_mode,
            self.device,
        )
        t0 = time.time()

        tfidf_path = self.model_dir / "checkpoints" / "tfidf_pipeline.joblib"
        deberta_dir = self.model_dir / "checkpoints" / "deberta" / "final_model"
        clf_cfg = CLASSIFIER_REGISTRY[self.classifier_name]
        clf_path = self.model_dir / clf_cfg["rel_path"]
        stylo_clf_path = self.model_dir / "checkpoints" / "stylometric" / "stylo_model.joblib"

        # Validate core existence
        for p, desc in [(tfidf_path, "TF-IDF pipeline"), (deberta_dir, "DeBERTa model"), (clf_path, f"{self.classifier_name.upper()} head")]:
            if not p.exists():
                raise FileNotFoundError(f"Required model artifact not found: {desc} ({p})")

        logger.info("Loading TF-IDF stylometric pipeline from %s...", tfidf_path)
        self.stylometric_extractor = StylometricExtractor.load(tfidf_path)

        if stylo_clf_path.exists():
            import joblib
            logger.info("Loading standalone stylometric classifier from %s...", stylo_clf_path)
            self.stylometric_classifier = joblib.load(stylo_clf_path)

        logger.info("Loading DeBERTa-v3 semantic encoder from %s...", deberta_dir)
        self.semantic_encoder = SemanticEncoder.load(deberta_dir, device=self.device)

        logger.info("Loading %s classifier head from %s...", self.classifier_name.upper(), clf_path)
        self.classifier_head = clf_cfg["class"].load(clf_path)

        self.is_ready = True
        logger.info("Anti-PLUTO InferenceService ready in %.2fs.", time.time() - t0)

    def predict(
        self,
        subject: str,
        body: str,
        mode: Optional[str] = None,
    ) -> PredictResponse:
        """Execute email cleaning and prediction in fusion, semantic, or stylometric mode.

        Parameters
        ----------
        subject : str
            Raw subject line.
        body : str
            Raw body text or HTML payload.
        mode : Optional[str]
            Execution mode: 'fusion' (dual-branch, default), 'semantic' (DeBERTa-only),
            or 'stylometric' (TF-IDF only). Defaults to self.default_mode if None.

        Returns
        -------
        PredictResponse
            Prediction outcome, class confidences, and ingestion telemetry.
        """
        if not self.is_ready or self.stylometric_extractor is None or self.semantic_encoder is None or self.classifier_head is None:
            raise RuntimeError("InferenceService has not loaded model artifacts yet. Call load_artifacts() first.")

        active_mode = (mode or self.default_mode).lower().strip()
        if active_mode not in ("fusion", "semantic", "stylometric"):
            raise ValueError(
                f"Unknown inference mode '{active_mode}'. Supported modes: 'fusion', 'semantic', 'stylometric'"
            )

        t_start = time.perf_counter()

        # 1. Clean, normalize, and mask payload
        prep: PreprocessedEmail = self.pipeline.process(subject, body)

        # 2. Execute feature extraction and model prediction based on mode
        if active_mode == "semantic":
            # Direct transformer prediction: bypasses stylometric extraction completely (saving CPU/memory)
            probs = self.semantic_encoder.predict_proba([prep.formatted_text])[0]
            active_clf = f"{self.classifier_name}_semantic_deberta"

        elif active_mode == "stylometric":
            # Pure stylometric prediction: bypasses transformer forward pass (runs on CPU without GPU)
            if self.stylometric_classifier is None:
                raise RuntimeError(
                    "Stylometric classifier checkpoint not loaded. Check models/checkpoints/stylometric/stylo_model.joblib"
                )
            X_stylo = self.stylometric_extractor.transform([prep.formatted_text])
            probs = self.stylometric_classifier.predict_proba(X_stylo)[0]
            active_clf = f"{self.classifier_name}_stylometric_linear"

        else:
            # Full dual-branch fusion mode (default): combines TF-IDF and DeBERTa embeddings
            X_stylo = self.stylometric_extractor.transform([prep.formatted_text])
            X_sem = self.semantic_encoder.extract_embeddings([prep.formatted_text], show_progress=False)
            probs = self.classifier_head.predict_proba(X_stylo, X_sem)[0]
            active_clf = self.classifier_name

        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]

        # 3. Binary determination: Human Phish (1) or LLM Phish (3)
        p_phish = float(probs[1] + probs[3])
        is_phishing = pred_idx in (1, 3) or p_phish >= 0.50
        is_llm_generated = pred_idx in (2, 3)

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        confidence_scores = {
            CLASS_NAMES[i]: float(probs[i])
            for i in range(len(CLASS_NAMES))
        }

        # Truncate processed preview for readable response payload
        preview = prep.formatted_text if len(prep.formatted_text) <= 300 else prep.formatted_text[:300] + "..."

        return PredictResponse(
            prediction=pred_class,
            is_phishing=is_phishing,
            is_llm_generated=is_llm_generated,
            phishing_probability=p_phish,
            confidence_scores=confidence_scores,
            processed_preview=preview,
            metadata=ProcessingMetadata(
                classifier=active_clf,
                mode=active_mode,
                latency_ms=latency_ms,
                detected_urls=len(prep.detected_urls),
                html_stripped=prep.html_stripped,
                thread_history_sliced=prep.thread_history_sliced,
            ),
        )
