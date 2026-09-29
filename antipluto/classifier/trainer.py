"""
antipluto.classifier.trainer
=========================
Master training orchestrator for the Anti-PLUTO dual-branch classifier.
Coordinates data preparation, TF-IDF fitting, DeBERTa fine-tuning,
embedding extraction, fusion classification head training (XGBoost, RF, MLP, 1D-CNN),
and thesis evaluation.
Includes phase-level state management and pause/resume support.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np
from scipy import sparse

from antipluto.classifier.data import DatasetSplits, load_dataset_splits
from antipluto.classifier.stylometric import StylometricExtractor
from antipluto.classifier.semantic import SemanticEncoder
from antipluto.classifier.heads import (
    AVAILABLE_HEADS,
    BaseFusionHead,
    get_classifier_head,
)
from antipluto.classifier.benchmark import CHECKPOINT_CONFIGS
from antipluto.classifier.evaluator import evaluate_predictions

logger = logging.getLogger(__name__)


class ClassifierTrainer:
    """End-to-end training pipeline orchestrator with pause/resume support."""

    def __init__(
        self,
        output_dir: Union[str, Path] = "models",
        data_dir: Union[str, Path] = "datasets/datasets_masked",
        samples_per_cohort: Optional[int] = 16000,
        model_name: str = "microsoft/deberta-v3-small",
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        classifier: str = "xgboost",
        xgb_n_estimators: int = 500,
        xgb_learning_rate: float = 0.08,
        xgb_max_depth: int = 6,
        xgb_device: Optional[str] = "cpu",
        seed: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)
        self.samples_per_cohort = samples_per_cohort
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.classifier = classifier.lower().strip()
        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_learning_rate = xgb_learning_rate
        self.xgb_max_depth = xgb_max_depth
        self.xgb_device = xgb_device
        self.seed = seed

        # Subdirectories for phase artifacts
        self.cache_dir = self.output_dir / "cache"
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.deberta_dir = self.checkpoints_dir / "deberta"
        self.eval_dir = self.output_dir / "evaluation"

        for d in [self.cache_dir, self.checkpoints_dir, self.deberta_dir, self.eval_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def run(self, resume: bool = False, force_deberta_retrain: bool = False) -> Dict[str, Any]:
        """Execute the training pipeline with resumption of completed phases."""
        start_time = time.time()
        logger.info("=========================================================")
        logger.info("Starting Anti-PLUTO Classifier Training Pipeline")
        logger.info("Output Directory: %s", self.output_dir)
        logger.info("Target Classifier Head: %s", self.classifier.upper())
        logger.info("Resume Mode: %s", resume)
        logger.info("=========================================================")

        # -----------------------------------------------------------------
        # Phase 1: Data Preparation & Stratified Split
        # -----------------------------------------------------------------
        logger.info("\n>>> Phase 1: Data Loading and Cohort Sampling")
        splits_cache = self.cache_dir / "dataset_splits.pkl"
        splits = load_dataset_splits(
            data_dir=self.data_dir,
            samples_per_cohort=self.samples_per_cohort,
            seed=self.seed,
            cache_path=splits_cache,
        )

        # -----------------------------------------------------------------
        # Phase 2: Stylometric Feature Extraction (TF-IDF)
        # -----------------------------------------------------------------
        logger.info("\n>>> Phase 2: Stylometric Feature Extraction (TF-IDF)")
        tfidf_path = self.checkpoints_dir / "tfidf_pipeline.joblib"
        if resume and tfidf_path.exists():
            logger.info("Loading existing TF-IDF pipeline from %s", tfidf_path)
            stylo_extractor = StylometricExtractor.load(tfidf_path)
        else:
            stylo_extractor = StylometricExtractor()
            stylo_extractor.fit(splits.train_texts)
            stylo_extractor.save(tfidf_path)

        logger.info("Transforming stylometric features...")
        X_train_stylo = stylo_extractor.transform(splits.train_texts)
        X_val_stylo = stylo_extractor.transform(splits.val_texts)
        X_test_stylo = stylo_extractor.transform(splits.test_texts)

        # -----------------------------------------------------------------
        # Phase 3: Semantic Encoder Fine-Tuning (DeBERTa)
        # -----------------------------------------------------------------
        logger.info("\n>>> Phase 3: Semantic Encoder Fine-Tuning (DeBERTa-v3)")
        final_deberta_path = self.deberta_dir / "final_model"
        deberta_checkpoint_exists = final_deberta_path.exists() and any(final_deberta_path.iterdir())

        if resume and deberta_checkpoint_exists and not force_deberta_retrain:
            logger.info("Resuming: Found existing DeBERTa model at %s. Skipping fine-tuning.", final_deberta_path)
            semantic_encoder = SemanticEncoder.load(final_deberta_path)
        else:
            semantic_encoder = SemanticEncoder(
                model_name_or_path=self.model_name,
                num_labels=4,
            )
            semantic_encoder.fine_tune(
                train_texts=splits.train_texts,
                train_labels=splits.train_labels,
                val_texts=splits.val_texts,
                val_labels=splits.val_labels,
                output_dir=self.deberta_dir,
                epochs=self.epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                resume=resume,
            )

        # -----------------------------------------------------------------
        # Phase 4: Dense Semantic Embedding Extraction
        # -----------------------------------------------------------------
        logger.info("\n>>> Phase 4: Dense Semantic Embedding Extraction")
        train_emb_path = self.cache_dir / "train_embeddings.npy"
        val_emb_path = self.cache_dir / "val_embeddings.npy"
        test_emb_path = self.cache_dir / "test_embeddings.npy"

        if resume and train_emb_path.exists():
            logger.info("Loading cached train embeddings from %s", train_emb_path)
            X_train_sem = np.load(train_emb_path)
        else:
            logger.info("Extracting train embeddings (%d samples)...", len(splits.train_records))
            X_train_sem = semantic_encoder.extract_embeddings(splits.train_texts)
            np.save(train_emb_path, X_train_sem)

        if resume and val_emb_path.exists():
            logger.info("Loading cached val embeddings from %s", val_emb_path)
            X_val_sem = np.load(val_emb_path)
        else:
            logger.info("Extracting val embeddings (%d samples)...", len(splits.val_records))
            X_val_sem = semantic_encoder.extract_embeddings(splits.val_texts)
            np.save(val_emb_path, X_val_sem)

        if resume and test_emb_path.exists():
            logger.info("Loading cached test embeddings from %s", test_emb_path)
            X_test_sem = np.load(test_emb_path)
        else:
            logger.info("Extracting test embeddings (%d samples)...", len(splits.test_records))
            X_test_sem = semantic_encoder.extract_embeddings(splits.test_texts)
            np.save(test_emb_path, X_test_sem)

        # -----------------------------------------------------------------
        # Phase 5: Classifier Fusion Head Training
        # -----------------------------------------------------------------
        logger.info("\n>>> Phase 5: Classifier Fusion Head Training")
        # Free GPU memory from neural encoder
        del semantic_encoder
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        heads_to_train = AVAILABLE_HEADS if self.classifier == "all" else [self.classifier]
        trained_heads: Dict[str, BaseFusionHead] = {}

        for h in heads_to_train:
            cfg = CHECKPOINT_CONFIGS[h]
            ckpt_path = self.output_dir / cfg["rel_path"]
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)

            if resume and ckpt_path.exists():
                logger.info("Loading existing %s checkpoint from %s", h.upper(), ckpt_path)
                clf = cfg["class"].load(ckpt_path)
            else:
                logger.info("Training %s fusion head...", h.upper())
                kwargs: Dict[str, Any] = {"random_state": self.seed}
                if h == "xgboost":
                    kwargs["n_estimators"] = self.xgb_n_estimators
                    kwargs["learning_rate"] = self.xgb_learning_rate
                    kwargs["max_depth"] = self.xgb_max_depth
                    kwargs["device"] = self.xgb_device
                elif h in ["mlp", "cnn"]:
                    kwargs["device"] = None  # Auto-select CUDA

                clf = get_classifier_head(h, **kwargs)
                clf.fit(
                    X_train_stylo=X_train_stylo,
                    X_train_sem=X_train_sem,
                    y_train=splits.train_labels,
                    X_val_stylo=X_val_stylo,
                    X_val_sem=X_val_sem,
                    y_val=splits.val_labels,
                )
                clf.save(ckpt_path)
            trained_heads[h] = clf

        # -----------------------------------------------------------------
        # Phase 6: Final Evaluation on Test Set
        # -----------------------------------------------------------------
        logger.info("\n>>> Phase 6: Model Evaluation on Unseen Test Cohorts")
        last_results = None
        for h, clf in trained_heads.items():
            logger.info("Evaluating %s on test set...", h.upper())
            y_test_pred = clf.predict(X_test_stylo, X_test_sem)
            y_test_prob = clf.predict_proba(X_test_stylo, X_test_sem)

            res = evaluate_predictions(
                y_true=splits.test_labels,
                y_pred=y_test_pred,
                y_prob=y_test_prob,
                output_dir=self.eval_dir / h if self.classifier == "all" else self.eval_dir,
            )
            last_results = res

        total_elapsed = time.time() - start_time
        logger.info("All training phases completed in %.1f minutes.", total_elapsed / 60.0)
        return last_results
