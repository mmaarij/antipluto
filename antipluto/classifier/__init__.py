"""
antipluto.classifier
==================
Dual-branch phishing / AI-origin detection classifier framework.

Architecture
------------
- Stylometric Branch: Word-level (1-2) and character-level (3-5) TF-IDF n-grams
  preserving casing, punctuation, and functional words.
- Semantic Branch: Fine-tuned DeBERTa-v3-small transformer embeddings ([CLS] pooling).
- Fusion Heads: Modular classification heads (XGBoost, Random Forest, MLP, 1D-CNN)
  evaluating concatenated sparse + dense feature vectors.
- Output: 4-class prediction (Human Benign, Human Phishing, LLM Benign, LLM Phishing).
"""

from antipluto.classifier.data import (
    CLASS_NAMES,
    COHORT_FILES,
    DatasetSplits,
    EmailRecord,
    format_email_text,
    load_cohort_records,
    load_dataset_splits,
)
from antipluto.classifier.evaluator import evaluate_predictions
from antipluto.classifier.heads import (
    BaseFusionHead,
    XGBoostFusionHead,
    RandomForestFusionHead,
    MLPFusionHead,
    CNNFusionHead,
    concatenate_features,
    get_classifier_head,
)
from antipluto.classifier.semantic import SemanticEncoder
from antipluto.classifier.stylometric import StylometricExtractor
from antipluto.classifier.trainer import ClassifierTrainer
from antipluto.classifier.benchmark import run_benchmark
from antipluto.classifier.exporter import (
    ONNXInferencePipeline,
    export_deberta_onnx,
    export_xgboost_onnx,
    export_rf_onnx,
    export_mlp_onnx,
    export_cnn_onnx,
    export_fusion_onnx,
)

__all__ = [
    "CLASS_NAMES",
    "COHORT_FILES",
    "DatasetSplits",
    "EmailRecord",
    "format_email_text",
    "load_cohort_records",
    "load_dataset_splits",
    "evaluate_predictions",
    "BaseFusionHead",
    "XGBoostFusionHead",
    "RandomForestFusionHead",
    "MLPFusionHead",
    "CNNFusionHead",
    "concatenate_features",
    "get_classifier_head",
    "SemanticEncoder",
    "StylometricExtractor",
    "ClassifierTrainer",
    "run_benchmark",
    "ONNXInferencePipeline",
    "export_deberta_onnx",
    "export_xgboost_onnx",
    "export_rf_onnx",
    "export_mlp_onnx",
    "export_cnn_onnx",
    "export_fusion_onnx",
]
