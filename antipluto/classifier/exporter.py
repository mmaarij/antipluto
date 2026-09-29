"""
antipluto.classifier.exporter
==========================
ONNX model export and portable inference runtime for Anti-PLUTO.
Exports TF-IDF, DeBERTa, and all classifier fusion heads (XGBoost, RF, MLP, 1D-CNN)
to ONNX format, and provides a unified ONNXInferencePipeline for single-email predictions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np

logger = logging.getLogger(__name__)


def export_deberta_onnx(
    model_dir: Union[str, Path],
    output_path: Union[str, Path],
    max_seq_len: int = 512,
) -> Path:
    """Export fine-tuned DeBERTa model backbone to ONNX."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_dir = Path(model_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading DeBERTa model from %s for ONNX export...", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModel.from_pretrained(str(model_dir))
    model.eval()

    dummy_text = "Subject: Test subject. Body: Test email body."
    dummy_inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        max_length=max_seq_len,
        padding="max_length",
        truncation=True,
    )

    input_ids = dummy_inputs["input_ids"]
    attention_mask = dummy_inputs["attention_mask"]

    class ClsExtractorWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, input_ids, attention_mask):
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
            # Return [CLS] representation (index 0)
            return outputs.last_hidden_state[:, 0, :]

    wrapper = ClsExtractorWrapper(model)
    wrapper.eval()

    logger.info("Exporting DeBERTa backbone to %s via torch.onnx.export...", output_path)
    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["cls_embedding"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "cls_embedding": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    logger.info("Successfully exported DeBERTa ONNX model to %s", output_path)
    return output_path


def export_xgboost_onnx(
    xgb_classifier,
    output_path: Union[str, Path],
    num_features: int = 100768,
) -> Path:
    """Export XGBoost classifier to ONNX format via onnxmltools."""
    return xgb_classifier.export_onnx(output_path, num_features)


def export_rf_onnx(
    rf_classifier,
    output_path: Union[str, Path],
    num_features: int = 100768,
) -> Path:
    """Export Random Forest classifier to ONNX format via skl2onnx."""
    return rf_classifier.export_onnx(output_path, num_features)


def export_mlp_onnx(
    mlp_classifier,
    output_path: Union[str, Path],
    num_features: int = 100768,
) -> Path:
    """Export PyTorch MLP classifier to ONNX format via torch.onnx.export."""
    return mlp_classifier.export_onnx(output_path, num_features)


def export_cnn_onnx(
    cnn_classifier,
    output_path: Union[str, Path],
    num_features: int = 100768,
) -> Path:
    """Export PyTorch 1D-CNN classifier to ONNX format via torch.onnx.export."""
    return cnn_classifier.export_onnx(output_path, num_features)


def export_fusion_onnx(
    classifier,
    output_path: Union[str, Path],
    num_features: int = 100768,
) -> Path:
    """Export any BaseFusionHead instance to ONNX format."""
    return classifier.export_onnx(output_path, num_features)


class ONNXInferencePipeline:
    """Inference runner using exported ONNX sessions for any classifier head."""

    def __init__(
        self,
        deberta_onnx_path: Union[str, Path],
        fusion_onnx_path: Union[str, Path],
        tfidf_path: Union[str, Path],
        tokenizer_dir: Union[str, Path],
        providers: Optional[List[str]] = None,
    ):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        from antipluto.classifier.stylometric import StylometricExtractor

        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        logger.info("Loading ONNX sessions with providers: %s", providers)
        self.deberta_session = ort.InferenceSession(str(deberta_onnx_path), providers=providers)
        self.fusion_session = ort.InferenceSession(str(fusion_onnx_path), providers=["CPUExecutionProvider"])
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        self.stylo_extractor = StylometricExtractor.load(tfidf_path)

    def predict_one(self, subject: str, body: str) -> Dict[str, Any]:
        """Predict class and probabilities for a single email."""
        from antipluto.classifier.data import format_email_text, CLASS_NAMES

        text = format_email_text(subject, body)

        # 1. Stylometric sparse features
        stylo_feat = self.stylo_extractor.transform([text]).toarray().astype(np.float32)

        # 2. DeBERTa semantic embeddings via ONNX
        tokens = self.tokenizer(
            text,
            return_tensors="np",
            max_length=512,
            truncation=True,
            padding=True,
        )
        deberta_inputs = {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }
        deberta_outputs = self.deberta_session.run(None, deberta_inputs)
        sem_emb = deberta_outputs[0].astype(np.float32)

        # 3. Concatenate
        combined = np.hstack([stylo_feat, sem_emb])

        # 4. Fusion ONNX prediction
        input_name = self.fusion_session.get_inputs()[0].name
        fusion_preds = self.fusion_session.run(None, {input_name: combined})

        # Check outputs format:
        # Format A: [labels, probabilities] (e.g., TreeEnsemble / XGBoost / skl2onnx)
        # Format B: [probabilities] (e.g., PyTorch SoftmaxWrapper (1, 4))
        if len(fusion_preds) >= 2:
            predicted_label = int(fusion_preds[0][0])
            probabilities = fusion_preds[1][0]
        else:
            probs = fusion_preds[0][0]
            predicted_label = int(np.argmax(probs))
            probabilities = probs

        prob_dict = {}
        if probabilities is not None:
            if isinstance(probabilities, dict):
                prob_dict = {CLASS_NAMES[k]: float(v) for k, v in probabilities.items()}
            elif isinstance(probabilities, (list, np.ndarray)):
                prob_dict = {CLASS_NAMES[i]: float(p) for i, p in enumerate(probabilities)}

        return {
            "text": text,
            "predicted_class_index": predicted_label,
            "predicted_class_name": CLASS_NAMES[predicted_label],
            "probabilities": prob_dict,
            "is_phishing": predicted_label in [1, 3],
            "is_llm_generated": predicted_label in [2, 3],
        }
