"""
antipluto.classifier.semantic
==========================
Semantic feature branch using DeBERTa-v3-small.
Provides fine-tuning on 4-class email data with pause/resume support,
and embedding extraction ([CLS] representation) for fusion.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EvalPrediction,
)
from datasets import Dataset

logger = logging.getLogger(__name__)


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute accuracy and macro F1 for evaluation."""
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": float(acc), "f1": float(f1)}


class SemanticEncoder:
    """Manages DeBERTa-v3-small tokenization, fine-tuning, checkpointing, and embedding extraction."""

    def __init__(
        self,
        model_name_or_path: str = "microsoft/deberta-v3-small",
        num_labels: int = 4,
        max_seq_len: int = 512,
        device: Optional[str] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels
        self.max_seq_len = max_seq_len

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info("Initializing SemanticEncoder with %s on %s", self.model_name_or_path, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model: Optional[AutoModelForSequenceClassification] = None

    def _ensure_model_loaded(self) -> AutoModelForSequenceClassification:
        if self.model is None:
            logger.info("Loading model weights for %s...", self.model_name_or_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name_or_path,
                num_labels=self.num_labels,
            )
            self.model.to(self.device)
        return self.model

    def _prepare_hf_dataset(self, texts: List[str], labels: Optional[List[int]] = None) -> Dataset:
        """Tokenize text records into a Hugging Face Dataset."""
        data = {"text": texts}
        if labels is not None:
            data["label"] = labels

        raw_dataset = Dataset.from_dict(data)

        def tokenize_batch(batch):
            return self.tokenizer(
                batch["text"],
                truncation=True,
                max_length=self.max_seq_len,
            )

        tokenized = raw_dataset.map(
            tokenize_batch,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing dataset",
        )
        return tokenized

    def fine_tune(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int],
        output_dir: Union[str, Path] = "models/checkpoints/deberta",
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        fp16: Optional[bool] = None,
        resume: bool = False,
    ) -> Path:
        """Fine-tune DeBERTa-v3 on 4-class email data with epoch checkpointing."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if fp16 is None:
            fp16 = torch.cuda.is_available()

        logger.info("Preparing tokenized train and validation datasets...")
        train_ds = self._prepare_hf_dataset(train_texts, train_labels)
        val_ds = self._prepare_hf_dataset(val_texts, val_labels)

        self._ensure_model_loaded()

        # Check for existing checkpoint to resume from
        resume_checkpoint = None
        if resume:
            checkpoints = [
                d for d in output_dir.glob("checkpoint-*") if d.is_dir()
            ]
            if checkpoints:
                # Sort by step number
                checkpoints.sort(key=lambda p: int(p.name.split("-")[-1]))
                resume_checkpoint = str(checkpoints[-1])
                logger.info("Resuming fine-tuning from checkpoint: %s", resume_checkpoint)
            else:
                logger.info("No checkpoint found in %s; starting training from scratch", output_dir)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            gradient_accumulation_steps=1 if batch_size >= 16 else (16 // batch_size),
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            fp16=fp16,
            dataloader_num_workers=0,  # Windows multi-processing stability
            report_to="none",
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        logger.info("Starting DeBERTa fine-tuning (%d epochs, batch_size=%d, fp16=%s)...", epochs, batch_size, fp16)
        train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        logger.info("Training completed: %s", train_result)

        # Save best model to final checkpoint directory
        final_dir = output_dir / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(final_dir))
        self.tokenizer.save_pretrained(str(final_dir))
        logger.info("Saved fine-tuned DeBERTa model and tokenizer to %s", final_dir)

        # Update reference path to loaded model
        self.model_name_or_path = str(final_dir)
        return final_dir

    def extract_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Extract [CLS] token dense vectors (768-d) for all input texts."""
        self._ensure_model_loaded()
        self.model.eval()

        embeddings: List[np.ndarray] = []
        n_samples = len(texts)

        # Determine backbone to avoid classification head computation
        backbone = getattr(self.model, "deberta", None)
        if backbone is None:
            backbone = getattr(self.model, "base_model", self.model)

        logger.info("Extracting %d embeddings using %s...", n_samples, self.device)

        with torch.no_grad():
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_texts = texts[start_idx:end_idx]

                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_len,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)

                outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)

                # Extract [CLS] token representation (index 0)
                # DeBERTa-v3 outputs last_hidden_state: (batch, seq_len, hidden_dim)
                last_hidden = outputs.last_hidden_state
                cls_embeddings = last_hidden[:, 0, :].detach().cpu().numpy()
                embeddings.append(cls_embeddings)

                if show_progress and (start_idx // batch_size) % 50 == 0 and start_idx > 0:
                    logger.info("Extracted %d / %d embeddings (%.1f%%)", start_idx, n_samples, (start_idx / n_samples) * 100)

        all_embeddings = np.vstack(embeddings)
        logger.info("Extracted embeddings array shape: %s", all_embeddings.shape)
        return all_embeddings

    def predict_proba(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Predict 4-class probabilities directly using fine-tuned DeBERTa classification head."""
        self._ensure_model_loaded()
        self.model.eval()

        probs_list: List[np.ndarray] = []
        n_samples = len(texts)

        with torch.no_grad():
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_texts = texts[start_idx:end_idx]

                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_len,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                batch_probs = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()
                probs_list.append(batch_probs)

        return np.vstack(probs_list)

    def save(self, path: Union[str, Path]) -> None:
        """Save model and tokenizer."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))
        logger.info("Saved SemanticEncoder to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[str] = None) -> SemanticEncoder:
        """Load fine-tuned model and tokenizer from disk."""
        path = Path(path)
        encoder = cls(model_name_or_path=str(path), device=device)
        encoder._ensure_model_loaded()
        return encoder
