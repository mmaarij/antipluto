"""
antipluto.classifier.data
======================
Dataset loading, balanced cohort sampling, formatting, and train/val/test
splitting for the Anti-PLUTO dual-branch classifier.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

CLASS_NAMES: List[str] = [
    "Human Benign",
    "Human Phishing",
    "LLM Benign",
    "LLM Phishing",
]

COHORT_FILES: Dict[int, str] = {
    0: "human_written_benign.jsonl",
    1: "human_written_phishing.jsonl",
    2: "llm_generated_benign.jsonl",
    3: "llm_generated_phishing.jsonl",
}


@dataclass
class EmailRecord:
    id: str
    subject: str
    body: str
    label: int  # 0: Human Benign, 1: Human Phishing, 2: LLM Benign, 3: LLM Phishing
    text: str   # "Subject: {subject}. Body: {body}"
    metadata: Dict[str, Any]


@dataclass
class DatasetSplits:
    train_records: List[EmailRecord]
    val_records: List[EmailRecord]
    test_records: List[EmailRecord]

    @property
    def train_texts(self) -> List[str]:
        return [r.text for r in self.train_records]

    @property
    def train_labels(self) -> List[int]:
        return [r.label for r in self.train_records]

    @property
    def val_texts(self) -> List[str]:
        return [r.text for r in self.val_records]

    @property
    def val_labels(self) -> List[int]:
        return [r.label for r in self.val_records]

    @property
    def test_texts(self) -> List[str]:
        return [r.text for r in self.test_records]

    @property
    def test_labels(self) -> List[int]:
        return [r.label for r in self.test_records]


def format_email_text(subject: Optional[str], body: Optional[str]) -> str:
    """Format subject and body with explicit delimiter tokens."""
    subj = (subject or "").strip()
    b = (body or "").strip()
    return f"Subject: {subj}. Body: {b}"


def load_cohort_records(
    file_path: Path,
    label: int,
    sample_size: Optional[int] = None,
    seed: int = 42,
) -> List[EmailRecord]:
    """Load records from a single masked JSONL file with optional reservoir/subsampling."""
    if not file_path.exists():
        raise FileNotFoundError(f"Cohort file not found: {file_path}")

    records: List[EmailRecord] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            record_id = str(data.get("id", f"{file_path.stem}_{idx}"))
            subj = data.get("subject", "")
            body = data.get("body", "")
            text = format_email_text(subj, body)

            # Keep remaining fields in metadata
            meta = {k: v for k, v in data.items() if k not in ("id", "subject", "body")}

            records.append(
                EmailRecord(
                    id=record_id,
                    subject=subj,
                    body=body,
                    label=label,
                    text=text,
                    metadata=meta,
                )
            )

    logger.info("Loaded %d raw records from %s", len(records), file_path.name)

    if sample_size is not None and sample_size < len(records):
        rng = random.Random(seed)
        records = rng.sample(records, sample_size)
        logger.info(
            "Subsampled %d records from %s (seed=%d)",
            len(records),
            file_path.name,
            seed,
        )

    return records


def load_dataset_splits(
    data_dir: Path | str = "datasets/datasets_masked",
    samples_per_cohort: Optional[int] = 16000,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    cache_path: Optional[Path | str] = "models/cache/dataset_splits.pkl",
    force_reload: bool = False,
) -> DatasetSplits:
    """Load, sample, split, and cache all four cohorts with stratification."""
    if cache_path:
        cache_path = Path(cache_path)
        if not force_reload and cache_path.exists():
            logger.info("Loading cached dataset splits from %s", cache_path)
            with open(cache_path, "rb") as f:
                splits: DatasetSplits = pickle.load(f)
            logger.info(
                "Loaded splits: Train=%d, Val=%d, Test=%d",
                len(splits.train_records),
                len(splits.val_records),
                len(splits.test_records),
            )
            return splits

    data_dir = Path(data_dir)
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"

    train_records: List[EmailRecord] = []
    val_records: List[EmailRecord] = []
    test_records: List[EmailRecord] = []

    rng = random.Random(seed)

    for label, filename in COHORT_FILES.items():
        file_path = data_dir / filename
        cohort = load_cohort_records(
            file_path=file_path,
            label=label,
            sample_size=samples_per_cohort,
            seed=seed,
        )

        # Shuffle deterministically
        rng.shuffle(cohort)

        n_total = len(cohort)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_records.extend(cohort[:n_train])
        val_records.extend(cohort[n_train : n_train + n_val])
        test_records.extend(cohort[n_train + n_val :])

    # Shuffle the combined splits so batches are not ordered by class
    rng.shuffle(train_records)
    rng.shuffle(val_records)
    rng.shuffle(test_records)

    splits = DatasetSplits(
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
    )

    logger.info(
        "Prepared dataset splits: Train=%d, Val=%d, Test=%d (Total=%d)",
        len(splits.train_records),
        len(splits.val_records),
        len(splits.test_records),
        len(splits.train_records) + len(splits.val_records) + len(splits.test_records),
    )

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(splits, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved dataset splits cache to %s", cache_path)

    return splits
