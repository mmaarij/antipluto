"""
antipluto.baseline.evaluator
==========================
Comparative evaluation harness benchmarking Anti-PLUTO against industry-standard
email security baselines (Apache SpamAssassin and Rspamd) on the exact 6,400
unseen test emails under standardized RFC 5322 MIME envelopes.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import numpy as np

from antipluto.baseline.clients import SpamAssassinClient, RspamdClient, BaselineFilterResult
from antipluto.baseline.unmasked_loader import load_unmasked_test_records, UnmaskedEmailRecord

logger = logging.getLogger(__name__)


def compute_binary_metrics(
    y_true: np.ndarray,  # 1 for Phish, 0 for Benign
    y_pred: np.ndarray,  # 1 for Phish, 0 for Benign
    cohort_labels: np.ndarray,  # 0: HB, 1: HP, 2: LB, 3: LP
    latencies: List[float],
) -> Dict[str, Any]:
    """Compute comprehensive phishing detection, false alarm, and latency metrics."""
    total_samples = len(y_true)

    # Phishing metrics
    phish_mask = (y_true == 1)
    benign_mask = (y_true == 0)

    total_phish = int(np.sum(phish_mask))
    total_benign = int(np.sum(benign_mask))

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    recall = tp / total_phish if total_phish > 0 else 0.0
    fpr = fp / total_benign if total_benign > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Cohort breakdowns
    hp_mask = (cohort_labels == 1)
    lp_mask = (cohort_labels == 3)
    hb_mask = (cohort_labels == 0)
    lb_mask = (cohort_labels == 2)

    hp_total = int(np.sum(hp_mask))
    lp_total = int(np.sum(lp_mask))
    hb_total = int(np.sum(hb_mask))
    lb_total = int(np.sum(lb_mask))

    hp_caught = int(np.sum(y_pred[hp_mask] == 1))
    lp_caught = int(np.sum(y_pred[lp_mask] == 1))
    hb_fp = int(np.sum(y_pred[hb_mask] == 1))
    lb_fp = int(np.sum(y_pred[lb_mask] == 1))

    hp_recall = hp_caught / hp_total if hp_total > 0 else 0.0
    lp_recall = lp_caught / lp_total if lp_total > 0 else 0.0
    lp_bypass = 1.0 - lp_recall

    hb_fpr = hb_fp / hb_total if hb_total > 0 else 0.0
    lb_fpr = lb_fp / lb_total if lb_total > 0 else 0.0

    # Latency stats
    valid_latencies = [l for l in latencies if l > 0]
    avg_latency = float(np.mean(valid_latencies)) if valid_latencies else 0.0
    median_latency = float(np.median(valid_latencies)) if valid_latencies else 0.0
    p95_latency = float(np.percentile(valid_latencies, 95)) if valid_latencies else 0.0
    throughput = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "cohorts": {
            "human_phish_recall": hp_recall,
            "human_phish_caught": hp_caught,
            "human_phish_total": hp_total,
            "llm_phish_recall": lp_recall,
            "llm_phish_bypass_rate": lp_bypass,
            "llm_phish_caught": lp_caught,
            "llm_phish_total": lp_total,
            "human_benign_fpr": hb_fpr,
            "human_benign_fp": hb_fp,
            "human_benign_total": hb_total,
            "llm_benign_fpr": lb_fpr,
            "llm_benign_fp": lb_fp,
            "llm_benign_total": lb_total,
        },
        "latency": {
            "mean_ms": avg_latency,
            "median_ms": median_latency,
            "p95_ms": p95_latency,
            "throughput_msgs_sec": throughput,
        },
    }


def evaluate_external_baselines(
    project_root: Optional[Path | str] = None,
    engines: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    output_json: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Execute comparative benchmark of Anti-PLUTO vs. SpamAssassin and Rspamd.

    Parameters
    ----------
    project_root : Optional[Path | str]
        Workspace root path.
    engines : Optional[List[str]]
        Engines to test: 'spamassassin', 'rspamd', 'antipluto', or all.
    max_samples : Optional[int]
        Limit evaluation to first N samples (useful for rapid dry-runs).
    output_json : Optional[Path | str]
        Path to save report. Defaults to ``models/evaluation/external_baselines_comparison.json``.

    Returns
    -------
    Dict[str, Any]
        Structured benchmark dictionary with full comparative metrics.
    """
    if project_root is None:
        project_root = Path.cwd()
    else:
        project_root = Path(project_root)

    if output_json is None:
        output_json = project_root / "models" / "evaluation" / "external_baselines_comparison.json"
    else:
        output_json = Path(output_json)

    if engines is None:
        engines = ["spamassassin", "rspamd", "antipluto"]
    else:
        engines = [e.lower() for e in engines]

    print("=" * 78)
    print("  Anti-PLUTO vs. EXTERNAL BASELINES COMPARATIVE BENCHMARK")
    print("=" * 78)

    # 1. Load unmasked test records
    print("\n1. Loading unmasked test records (RFC 5322 MIME envelopes)...")
    records = load_unmasked_test_records(project_root=project_root)
    if max_samples is not None and max_samples < len(records):
        print(f"   [DRY-RUN] Subsetting to first {max_samples} samples.")
        records = records[:max_samples]

    y_true = np.array([1 if r.is_phishing else 0 for r in records], dtype=int)
    cohort_labels = np.array([r.label for r in records], dtype=int)

    benchmark_report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_samples": len(records),
        "engines": {},
    }

    # 2. Check engine connectivity
    sa_client = SpamAssassinClient()
    rspamd_client = RspamdClient()

    # -----------------------------------------------------------------------
    # Evaluate Apache SpamAssassin
    # -----------------------------------------------------------------------
    if "spamassassin" in engines or "all" in engines:
        print("\n2. Evaluating Apache SpamAssassin (spamd on port 783)...")
        if not sa_client.ping():
            print("   [WARNING] SpamAssassin daemon (spamd) is not responding on localhost:783.")
            print("   Please start spamd inside WSL2: `spamd -d -p 783 -L`.")
            benchmark_report["engines"]["spamassassin"] = {"status": "offline", "error": "Connection refused"}
        else:
            print("   Connected to spamd. Processing emails...")
            sa_preds = []
            sa_scores = []
            sa_latencies = []

            t_start = time.time()
            for idx, rec in enumerate(records, 1):
                res = sa_client.check(rec.mime_bytes, record_id=rec.id)
                sa_preds.append(1 if res.is_phish_or_spam else 0)
                sa_scores.append(res.score)
                sa_latencies.append(res.latency_ms)

                if idx % 500 == 0 or idx == len(records):
                    print(f"   [{idx}/{len(records)}] Processed in {time.time() - t_start:.1f}s...")

            sa_metrics = compute_binary_metrics(
                y_true=y_true,
                y_pred=np.array(sa_preds, dtype=int),
                cohort_labels=cohort_labels,
                latencies=sa_latencies,
            )
            sa_metrics["status"] = "completed"
            benchmark_report["engines"]["spamassassin"] = sa_metrics
            print(f"   SpamAssassin Complete: Catch Rate={sa_metrics['recall']*100:.2f}%, FPR={sa_metrics['fpr']*100:.2f}%")

    # -----------------------------------------------------------------------
    # Evaluate Rspamd
    # -----------------------------------------------------------------------
    if "rspamd" in engines or "all" in engines:
        print("\n3. Evaluating Rspamd (HTTP REST on port 11333)...")
        if not rspamd_client.ping():
            print("   [WARNING] Rspamd daemon is not responding on http://localhost:11333/ping.")
            print("   Please start rspamd inside WSL2: `rspamd`.")
            benchmark_report["engines"]["rspamd"] = {"status": "offline", "error": "Connection refused"}
        else:
            print("   Connected to Rspamd. Processing emails...")
            rspamd_preds = []
            rspamd_scores = []
            rspamd_latencies = []

            t_start = time.time()
            for idx, rec in enumerate(records, 1):
                res = rspamd_client.check(rec.mime_bytes, record_id=rec.id)
                rspamd_preds.append(1 if res.is_phish_or_spam else 0)
                rspamd_scores.append(res.score)
                rspamd_latencies.append(res.latency_ms)

                if idx % 500 == 0 or idx == len(records):
                    print(f"   [{idx}/{len(records)}] Processed in {time.time() - t_start:.1f}s...")

            rspamd_metrics = compute_binary_metrics(
                y_true=y_true,
                y_pred=np.array(rspamd_preds, dtype=int),
                cohort_labels=cohort_labels,
                latencies=rspamd_latencies,
            )
            rspamd_metrics["status"] = "completed"
            benchmark_report["engines"]["rspamd"] = rspamd_metrics
            print(f"   Rspamd Complete: Catch Rate={rspamd_metrics['recall']*100:.2f}%, FPR={rspamd_metrics['fpr']*100:.2f}%")

    # -----------------------------------------------------------------------
    # Evaluate Anti-PLUTO (XGBoost Fusion Head)
    # -----------------------------------------------------------------------
    if "antipluto" in engines or "all" in engines:
        print("\n4. Evaluating Anti-PLUTO (XGBoost Fusion Head)...")
        from antipluto.classifier.heads import XGBoostFusionHead
        from antipluto.classifier.stylometric import StylometricExtractor

        checkpoints_dir = project_root / "models" / "checkpoints"
        cache_dir = project_root / "models" / "cache"

        stylo = StylometricExtractor.load(checkpoints_dir / "tfidf_pipeline.joblib")
        xgb_clf = XGBoostFusionHead.load(checkpoints_dir / "xgboost" / "xgb_model.json")

        # Load cached semantic embeddings and stylometric features
        # Note: rec.masked_subject and rec.masked_body are used for Anti-PLUTO inference
        antipluto_texts = [f"Subject: {r.masked_subject}. Body: {r.masked_body}" for r in records]
        X_stylo = stylo.transform(antipluto_texts)
        X_sem = np.load(cache_dir / "test_embeddings.npy")
        if max_samples is not None and max_samples < len(X_sem):
            X_sem = X_sem[:max_samples]

        # Time inference latency on XGBoost head
        latencies = []
        n_warmup = 50
        # Warmup
        for _ in range(n_warmup):
            _ = xgb_clf.predict_proba(X_stylo[:1], X_sem[:1])

        t0 = time.perf_counter()
        probs = xgb_clf.predict_proba(X_stylo, X_sem)
        total_time_ms = (time.perf_counter() - t0) * 1000.0
        per_sample_ms = total_time_ms / len(records)
        latencies = [per_sample_ms] * len(records)

        # Phishing prediction: p(Human Phish) + p(LLM Phish) >= 0.50
        p_phish = probs[:, 1] + probs[:, 3]
        antipluto_preds = (p_phish >= 0.50).astype(int)

        antipluto_metrics = compute_binary_metrics(
            y_true=y_true,
            y_pred=antipluto_preds,
            cohort_labels=cohort_labels,
            latencies=latencies,
        )
        antipluto_metrics["status"] = "completed"
        benchmark_report["engines"]["antipluto"] = antipluto_metrics
        print(f"   Anti-PLUTO Complete: Catch Rate={antipluto_metrics['recall']*100:.2f}%, FPR={antipluto_metrics['fpr']*100:.2f}%")

    # 3. Save report
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)
    print(f"\nSaved benchmark metrics to: {output_json}")

    # 4. Render comparative ASCII Table
    _render_comparison_table(benchmark_report)

    return benchmark_report


def _render_comparison_table(report: Dict[str, Any]) -> None:
    """Print an ASCII comparison table of all completed engines."""
    engines = report.get("engines", {})
    active_engines = [k for k, v in engines.items() if v.get("status") == "completed"]

    if not active_engines:
        print("\n[NOTE] No external engines were active/completed during this run.")
        return

    col_names = [e.upper() for e in active_engines]
    headers = ["Metric / Dimension"] + col_names

    rows = [
        ("Overall Phishing Catch Rate (Recall)", [f"{engines[e]['recall']*100:.2f}%" for e in active_engines]),
        ("Human Phishing Catch Rate", [f"{engines[e]['cohorts']['human_phish_recall']*100:.2f}%" for e in active_engines]),
        ("LLM Phishing Catch Rate", [f"{engines[e]['cohorts']['llm_phish_recall']*100:.2f}%" for e in active_engines]),
        ("LLM Phishing Bypass Rate", [f"{engines[e]['cohorts']['llm_phish_bypass_rate']*100:.2f}%" for e in active_engines]),
        ("Benign False Positive Rate (FPR)", [f"{engines[e]['fpr']*100:.2f}%" for e in active_engines]),
        ("Human Benign False Alarms", [f"{engines[e]['cohorts']['human_benign_fp']}/{engines[e]['cohorts']['human_benign_total']}" for e in active_engines]),
        ("LLM Benign False Alarms", [f"{engines[e]['cohorts']['llm_benign_fp']}/{engines[e]['cohorts']['llm_benign_total']}" for e in active_engines]),
        ("Macro F1-Score", [f"{engines[e]['f1']*100:.2f}%" for e in active_engines]),
        ("Overall Accuracy", [f"{engines[e]['accuracy']*100:.2f}%" for e in active_engines]),
        ("Inference Latency (Mean)", [f"{engines[e]['latency']['mean_ms']:.3f} ms" for e in active_engines]),
        ("Throughput", [f"{engines[e]['latency']['throughput_msgs_sec']:.0f} msgs/s" for e in active_engines]),
    ]

    print("\n" + "=" * 90)
    print("                   EMPIRICAL BENCHMARK SUMMARY TABLE")
    print("=" * 90)

    # Format table
    col_widths = [38] + [max(16, len(h) + 2) for h in col_names]
    row_fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)

    print(row_fmt.format(*headers))
    print("-" * (sum(col_widths) + 2 * (len(col_widths) - 1)))

    for label, vals in rows:
        print(row_fmt.format(label, *vals))

    print("=" * 90 + "\n")
