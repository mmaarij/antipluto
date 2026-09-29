"""
antipluto.classifier.benchmark
============================
Comparative benchmark suite for Anti-PLUTO classifier heads:
Evaluates XGBoost, Random Forest, MLP, and 1D-CNN on the exact same
concatenated feature representation (100,000-d TF-IDF + 768-d DeBERTa-v3).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from antipluto.classifier.data import DatasetSplits, CLASS_NAMES
from antipluto.classifier.heads import (
    AVAILABLE_HEADS,
    BaseFusionHead,
    XGBoostFusionHead,
    RandomForestFusionHead,
    MLPFusionHead,
    CNNFusionHead,
    get_classifier_head,
)
from antipluto.classifier.stylometric import StylometricExtractor

logger = logging.getLogger(__name__)

CHECKPOINT_CONFIGS = {
    "xgboost": {
        "class": XGBoostFusionHead,
        "rel_path": Path("checkpoints/xgboost/xgb_model.json"),
        "ext": ".json",
    },
    "rf": {
        "class": RandomForestFusionHead,
        "rel_path": Path("checkpoints/rf/rf_model.joblib"),
        "ext": ".joblib",
    },
    "mlp": {
        "class": MLPFusionHead,
        "rel_path": Path("checkpoints/mlp/mlp_model.pt"),
        "ext": ".pt",
    },
    "cnn": {
        "class": CNNFusionHead,
        "rel_path": Path("checkpoints/cnn/cnn_model.pt"),
        "ext": ".pt",
    },
}


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Any]:
    """Compute comprehensive 4-class, phishing, and provenance evaluation metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        roc_auc = None

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    # Phishing detection evaluation (Classes 1 & 3 vs Classes 0 & 2)
    y_true_phish = np.isin(y_true, [1, 3]).astype(int)
    y_pred_phish = np.isin(y_pred, [1, 3]).astype(int)

    phish_acc = float(accuracy_score(y_true_phish, y_pred_phish))
    phish_prec = float(precision_score(y_true_phish, y_pred_phish, zero_division=0))
    phish_rec = float(recall_score(y_true_phish, y_pred_phish, zero_division=0))
    phish_f1 = float(f1_score(y_true_phish, y_pred_phish, zero_division=0))

    # Cohort-specific phishing breakdown
    hp_mask = y_true == 1
    hp_recall = float(np.mean(y_pred_phish[hp_mask])) if np.sum(hp_mask) > 0 else 0.0

    lp_mask = y_true == 3
    lp_recall = float(np.mean(y_pred_phish[lp_mask])) if np.sum(lp_mask) > 0 else 0.0

    # Benign False Positive Rates (FPR)
    hb_mask = y_true == 0
    hb_fpr = float(np.mean(y_pred_phish[hb_mask])) if np.sum(hb_mask) > 0 else 0.0

    lb_mask = y_true == 2
    lb_fpr = float(np.mean(y_pred_phish[lb_mask])) if np.sum(lb_mask) > 0 else 0.0

    overall_benign_mask = y_true_phish == 0
    overall_fpr = float(np.mean(y_pred_phish[overall_benign_mask])) if np.sum(overall_benign_mask) > 0 else 0.0

    # Provenance Attribution evaluation (Classes 2 & 3 vs Classes 0 & 1)
    y_true_ai = np.isin(y_true, [2, 3]).astype(int)
    y_pred_ai = np.isin(y_pred, [2, 3]).astype(int)

    ai_acc = float(accuracy_score(y_true_ai, y_pred_ai))
    ai_prec = float(precision_score(y_true_ai, y_pred_ai, zero_division=0))
    ai_rec = float(recall_score(y_true_ai, y_pred_ai, zero_division=0))
    ai_f1 = float(f1_score(y_true_ai, y_pred_ai, zero_division=0))

    return {
        "four_class": {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "roc_auc_ovr": roc_auc,
            "classification_report": report,
            "confusion_matrix": cm,
        },
        "phishing_detection": {
            "accuracy": phish_acc,
            "overall_recall": phish_rec,
            "human_phish_recall": hp_recall,
            "llm_phish_recall": lp_recall,
            "precision": phish_prec,
            "f1_score": phish_f1,
            "overall_fpr": overall_fpr,
            "human_benign_fpr": hb_fpr,
            "llm_benign_fpr": lb_fpr,
        },
        "provenance_attribution": {
            "accuracy": ai_acc,
            "precision": ai_prec,
            "recall": ai_rec,
            "f1_score": ai_f1,
        },
    }


def train_single_head(
    head_name: str,
    output_dir: Path,
    splits: DatasetSplits,
    X_train_stylo: sparse.csr_matrix,
    X_train_sem: np.ndarray,
    X_val_stylo: sparse.csr_matrix,
    X_val_sem: np.ndarray,
    device: Optional[str] = None,
    seed: int = 42,
) -> BaseFusionHead:
    """Train an individual classifier head and save its checkpoint."""
    cfg = CHECKPOINT_CONFIGS[head_name]
    save_path = output_dir / cfg["rel_path"]
    save_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing %s classifier head...", head_name.upper())
    kwargs: Dict[str, Any] = {"random_state": seed}
    if head_name == "xgboost":
        kwargs["device"] = device or "cpu"
    elif head_name in ["mlp", "cnn"]:
        if device is not None:
            kwargs["device"] = device

    clf = get_classifier_head(head_name, **kwargs)
    clf.fit(
        X_train_stylo=X_train_stylo,
        X_train_sem=X_train_sem,
        y_train=splits.train_labels,
        X_val_stylo=X_val_stylo,
        X_val_sem=X_val_sem,
        y_val=splits.val_labels,
        verbose=True,
    )
    clf.save(save_path)
    return clf


def run_benchmark(
    model_dir: Union[str, Path] = "models",
    heads: Optional[List[str]] = None,
    force_retrain: bool = False,
    device: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run full benchmark comparison across all specified classifier heads.
    Loads cached embeddings and splits, trains missing heads, evaluates
    on unseen test set, and saves results to models/evaluation/classifier_comparison.json.
    """
    model_dir = Path(model_dir)
    cache_dir = model_dir / "cache"
    checkpoints_dir = model_dir / "checkpoints"
    eval_dir = model_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    selected_heads = [h.strip().lower() for h in (heads or AVAILABLE_HEADS)]
    for h in selected_heads:
        if h not in CHECKPOINT_CONFIGS:
            raise ValueError(f"Unknown classifier head: {h}. Choose from {list(CHECKPOINT_CONFIGS.keys())}")

    logger.info("=========================================================")
    logger.info("Starting Anti-PLUTO Multi-Classifier Comparison Benchmark")
    logger.info("Selected Heads: %s", ", ".join(h.upper() for h in selected_heads))
    logger.info("Model Directory: %s", model_dir)
    logger.info("=========================================================")

    # 1. Verify and load cached splits and TF-IDF pipeline
    splits_path = cache_dir / "dataset_splits.pkl"
    tfidf_path = checkpoints_dir / "tfidf_pipeline.joblib"
    test_emb_path = cache_dir / "test_embeddings.npy"

    for p, desc in [(splits_path, "Dataset splits"), (tfidf_path, "TF-IDF pipeline"), (test_emb_path, "Test embeddings")]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact: {desc} at {p}. Run 'python -m antipluto train' first.")

    logger.info("Loading cached dataset splits from %s...", splits_path)
    with open(splits_path, "rb") as f:
        splits = pickle.load(f)

    logger.info("Loading TF-IDF vectorizer from %s...", tfidf_path)
    stylo_extractor = StylometricExtractor.load(tfidf_path)

    logger.info("Extracting stylometric features for test set (%d samples)...", len(splits.test_records))
    X_test_stylo = stylo_extractor.transform(splits.test_texts)
    X_test_sem = np.load(test_emb_path)
    y_test = np.asarray(splits.test_labels, dtype=np.int64)

    # Lazily loaded training features (only loaded if training is needed)
    X_train_stylo = None
    X_train_sem = None
    X_val_stylo = None
    X_val_sem = None

    def ensure_train_features():
        nonlocal X_train_stylo, X_train_sem, X_val_stylo, X_val_sem
        if X_train_stylo is None:
            train_emb_path = cache_dir / "train_embeddings.npy"
            val_emb_path = cache_dir / "val_embeddings.npy"
            if not train_emb_path.exists() or not val_emb_path.exists():
                raise FileNotFoundError(f"Missing training embeddings in {cache_dir}. Run training pipeline first.")

            logger.info("Loading cached semantic embeddings for training...")
            X_train_sem = np.load(train_emb_path)
            X_val_sem = np.load(val_emb_path)

            logger.info("Transforming stylometric features for train (%d) and val (%d)...", len(splits.train_records), len(splits.val_records))
            X_train_stylo = stylo_extractor.transform(splits.train_texts)
            X_val_stylo = stylo_extractor.transform(splits.val_texts)

    results: Dict[str, Any] = {}

    for head_name in selected_heads:
        cfg = CHECKPOINT_CONFIGS[head_name]
        ckpt_path = model_dir / cfg["rel_path"]
        head_cls = cfg["class"]

        needs_train = force_retrain or not ckpt_path.exists()
        if needs_train:
            logger.info("\n>>> Training Head: %s (checkpoint %s missing or retrain=True)", head_name.upper(), ckpt_path)
            ensure_train_features()
            t_train_start = time.time()
            clf = train_single_head(
                head_name=head_name,
                output_dir=model_dir,
                splits=splits,
                X_train_stylo=X_train_stylo,
                X_train_sem=X_train_sem,
                X_val_stylo=X_val_stylo,
                X_val_sem=X_val_sem,
                device=device,
                seed=seed,
            )
            train_duration_sec = time.time() - t_train_start
            logger.info("Trained %s in %.2f seconds (%.2f mins)", head_name.upper(), train_duration_sec, train_duration_sec / 60.0)
        else:
            logger.info("\n>>> Loading Existing Checkpoint for Head: %s from %s", head_name.upper(), ckpt_path)
            clf = head_cls.load(ckpt_path)
            train_duration_sec = None

        # Measure test inference latency
        logger.info("Evaluating %s on %d unseen test samples...", head_name.upper(), len(y_test))
        t_infer_start = time.time()
        y_test_pred = clf.predict(X_test_stylo, X_test_sem)
        y_test_prob = clf.predict_proba(X_test_stylo, X_test_sem)
        infer_duration_sec = time.time() - t_infer_start
        latency_ms = (infer_duration_sec / len(y_test)) * 1000.0
        throughput = len(y_test) / infer_duration_sec if infer_duration_sec > 0 else 0.0

        # Calculate model size on disk
        model_size_mb = 0.0
        if ckpt_path.exists():
            model_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)

        metrics = compute_metrics(y_true=y_test, y_pred=y_test_pred, y_prob=y_test_prob)
        metrics["computational"] = {
            "test_latency_ms_per_sample": round(latency_ms, 3),
            "throughput_samples_per_sec": round(throughput, 1),
            "model_size_mb": round(model_size_mb, 2),
            "train_duration_sec": round(train_duration_sec, 2) if train_duration_sec else None,
        }

        results[head_name] = metrics

        acc = metrics["four_class"]["accuracy"] * 100.0
        f1 = metrics["four_class"]["macro_f1"] * 100.0
        phish_rec = metrics["phishing_detection"]["overall_recall"] * 100.0
        fpr = metrics["phishing_detection"]["overall_fpr"] * 100.0
        logger.info(
            "%s Results: 4-Class Acc=%.2f%%, Macro F1=%.2f%%, Phish Recall=%.2f%%, FPR=%.2f%%, Latency=%.3f ms/sample",
            head_name.upper(),
            acc,
            f1,
            phish_rec,
            fpr,
            latency_ms,
        )

    # Save summary JSON
    out_json_path = eval_dir / "classifier_comparison.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("\nSaved full comparison metrics to %s", out_json_path)

    # Print comparative table to console
    print_comparison_table(results)
    return results


def print_comparison_table(results: Dict[str, Any]) -> None:
    """Print an ASCII comparison table across all evaluated classifier heads."""
    header = (
        f"\n{'Classifier':<14} | {'Accuracy':<9} | {'Macro F1':<9} | {'ROC-AUC':<8} | "
        f"{'Phish Rec':<10} | {'LLM P-Rec':<10} | {'Benign FPR':<10} | {'AI Attr F1':<10} | {'Latency':<10}"
    )
    sep = "-" * len(header)
    rows = [header, sep]

    for head_name, m in results.items():
        fc = m["four_class"]
        ph = m["phishing_detection"]
        ai = m["provenance_attribution"]
        comp = m["computational"]

        acc_str = f"{fc['accuracy'] * 100:.2f}%"
        f1_str = f"{fc['macro_f1'] * 100:.2f}%"
        auc_str = f"{fc['roc_auc_ovr']:.4f}" if fc.get("roc_auc_ovr") else "N/A"
        prec_str = f"{ph['overall_recall'] * 100:.2f}%"
        lp_rec_str = f"{ph['llm_phish_recall'] * 100:.2f}%"
        fpr_str = f"{ph['overall_fpr'] * 100:.2f}%"
        ai_f1_str = f"{ai['f1_score'] * 100:.2f}%"
        lat_str = f"{comp['test_latency_ms_per_sample']:.2f} ms"

        row = (
            f"{head_name.upper():<14} | {acc_str:<9} | {f1_str:<9} | {auc_str:<8} | "
            f"{prec_str:<10} | {lp_rec_str:<10} | {fpr_str:<10} | {ai_f1_str:<10} | {lat_str:<10}"
        )
        rows.append(row)

    rows.append(sep)
    output_table = "\n".join(rows)
    print(output_table)
    logger.info(output_table)
