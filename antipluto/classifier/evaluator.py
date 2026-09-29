"""
antipluto.classifier.evaluator
===========================
Comprehensive 4-class and derived thesis evaluation metrics.
Computes multi-class performance, confusion matrices, and mapped binary
evaluations (phishing detection, AI-origin detection, and LLM-benign FPR).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from antipluto.classifier.data import CLASS_NAMES

logger = logging.getLogger(__name__)


def evaluate_predictions(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    y_prob: Optional[np.ndarray] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Compute complete evaluation suite for the 4-class classifier and thesis hypotheses."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # 1. 4-Class Metrics
    acc_4class = accuracy_score(y_true, y_pred)
    macro_f1_4class = f1_score(y_true, y_pred, average="macro")
    weighted_f1_4class = f1_score(y_true, y_pred, average="weighted")
    cm_4class = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    clf_report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
    )

    roc_auc_4class = None
    if y_prob is not None:
        try:
            roc_auc_4class = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except Exception as e:
            logger.warning("Could not calculate 4-class ROC-AUC: %s", e)

    # 2. Phishing Detection Metrics (Binary Mapping: Benign={0, 2} -> 0, Phishing={1, 3} -> 1)
    y_true_phish = np.isin(y_true, [1, 3]).astype(int)
    y_pred_phish = np.isin(y_pred, [1, 3]).astype(int)

    phish_acc = accuracy_score(y_true_phish, y_pred_phish)
    phish_prec = precision_score(y_true_phish, y_pred_phish, zero_division=0)
    phish_rec = recall_score(y_true_phish, y_pred_phish, zero_division=0)
    phish_f1 = f1_score(y_true_phish, y_pred_phish, zero_division=0)

    # False Positive Rates on Benign Cohorts
    # FPR = FP / (FP + TN)
    # Human benign records (label 0): how many predicted as phishing (1 or 3)?
    human_benign_mask = (y_true == 0)
    human_benign_count = int(np.sum(human_benign_mask))
    human_benign_fp = int(np.sum(y_pred_phish[human_benign_mask]))
    human_benign_fpr = float(human_benign_fp / human_benign_count) if human_benign_count > 0 else 0.0

    # LLM benign records (label 2): how many falsely flagged as phishing?
    llm_benign_mask = (y_true == 2)
    llm_benign_count = int(np.sum(llm_benign_mask))
    llm_benign_fp = int(np.sum(y_pred_phish[llm_benign_mask]))
    llm_benign_fpr = float(llm_benign_fp / llm_benign_count) if llm_benign_count > 0 else 0.0

    # Phishing recall by origin
    human_phish_mask = (y_true == 1)
    human_phish_count = int(np.sum(human_phish_mask))
    human_phish_detected = int(np.sum(y_pred_phish[human_phish_mask]))
    human_phish_recall = float(human_phish_detected / human_phish_count) if human_phish_count > 0 else 0.0

    llm_phish_mask = (y_true == 3)
    llm_phish_count = int(np.sum(llm_phish_mask))
    llm_phish_detected = int(np.sum(y_pred_phish[llm_phish_mask]))
    llm_phish_recall = float(llm_phish_detected / llm_phish_count) if llm_phish_count > 0 else 0.0

    # 3. AI-Origin Detection Metrics (Binary Mapping: Human={0, 1} -> 0, AI={2, 3} -> 1)
    y_true_ai = np.isin(y_true, [2, 3]).astype(int)
    y_pred_ai = np.isin(y_pred, [2, 3]).astype(int)

    ai_acc = accuracy_score(y_true_ai, y_pred_ai)
    ai_prec = precision_score(y_true_ai, y_pred_ai, zero_division=0)
    ai_rec = recall_score(y_true_ai, y_pred_ai, zero_division=0)
    ai_f1 = f1_score(y_true_ai, y_pred_ai, zero_division=0)

    results: Dict[str, Any] = {
        "four_class": {
            "accuracy": float(acc_4class),
            "macro_f1": float(macro_f1_4class),
            "weighted_f1": float(weighted_f1_4class),
            "roc_auc_ovr": roc_auc_4class,
            "confusion_matrix": cm_4class.tolist(),
            "classification_report": clf_report,
        },
        "phishing_detection": {
            "accuracy": float(phish_acc),
            "precision": float(phish_prec),
            "recall": float(phish_rec),
            "f1": float(phish_f1),
            "human_benign_fpr": human_benign_fpr,
            "llm_benign_fpr": llm_benign_fpr,
            "human_phishing_recall": human_phish_recall,
            "llm_phishing_recall": llm_phish_recall,
        },
        "ai_detection": {
            "accuracy": float(ai_acc),
            "precision": float(ai_prec),
            "recall": float(ai_rec),
            "f1": float(ai_f1),
        },
    }

    # Format human-readable text report
    report_lines = [
        "=================================================================",
        "                 Anti-PLUTO CLASSIFIER EVALUATION REPORT            ",
        "=================================================================",
        f"Overall 4-Class Accuracy : {acc_4class:.4f}",
        f"4-Class Macro F1         : {macro_f1_4class:.4f}",
        f"4-Class Weighted F1      : {weighted_f1_4class:.4f}",
    ]
    if roc_auc_4class is not None:
        report_lines.append(f"4-Class ROC-AUC (OvR)    : {roc_auc_4class:.4f}")

    report_lines.extend([
        "",
        "--- 4-Class Confusion Matrix ---",
        f"                 Pred HB    Pred HP    Pred LB    Pred LP",
        f"True HB (0):    {cm_4class[0][0]:8d}   {cm_4class[0][1]:8d}   {cm_4class[0][2]:8d}   {cm_4class[0][3]:8d}",
        f"True HP (1):    {cm_4class[1][0]:8d}   {cm_4class[1][1]:8d}   {cm_4class[1][2]:8d}   {cm_4class[1][3]:8d}",
        f"True LB (2):    {cm_4class[2][0]:8d}   {cm_4class[2][1]:8d}   {cm_4class[2][2]:8d}   {cm_4class[2][3]:8d}",
        f"True LP (3):    {cm_4class[3][0]:8d}   {cm_4class[3][1]:8d}   {cm_4class[3][2]:8d}   {cm_4class[3][3]:8d}",
        "",
        "--- Thesis Phishing Detection Hypotheses ---",
        f"Phishing Accuracy        : {phish_acc:.4f}",
        f"Phishing Recall (Total)  : {phish_rec:.4f}",
        f"  - Human Phish Recall   : {human_phish_recall:.4f}",
        f"  - LLM Phish Recall     : {llm_phish_recall:.4f}",
        f"Phishing False Positives :",
        f"  - Human Benign FPR     : {human_benign_fpr:.4f} ({human_benign_fp}/{human_benign_count})",
        f"  - LLM Benign FPR       : {llm_benign_fpr:.4f} ({llm_benign_fp}/{llm_benign_count})",
        "",
        "--- AI-Origin Discrimination ---",
        f"AI vs Human Accuracy     : {ai_acc:.4f}",
        f"AI vs Human F1           : {ai_f1:.4f}",
        "=================================================================",
    ])
    full_report_str = "\n".join(report_lines)
    logger.info("\n%s", full_report_str)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "eval_report.txt", "w", encoding="utf-8") as f:
            f.write(full_report_str)
        with open(out / "eval_metrics.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved evaluation outputs to %s", out)

    return results
