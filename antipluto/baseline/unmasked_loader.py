"""
antipluto.baseline.unmasked_loader
================================
Loads the exact 6,400 unseen test samples from ``dataset_splits.pkl`` and
efficiently maps each record back to its original, unmasked source text
in ``datasets_preprocessed/`` and ``datasets_synthetic/``.

Provides 100% data parity between external engines (which receive unmasked
content with raw URLs/text in RFC 5322 MIME envelopes) and Anti-PLUTO (which
receives the same raw email and applies its internal tokenizer masking).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import pickle
from typing import Dict, List, Optional, Set

from antipluto.baseline.mime_envelope import build_mime_message

logger = logging.getLogger(__name__)

COHORT_NAMES: Dict[int, str] = {
    0: "Human Benign",
    1: "Human Phishing",
    2: "LLM Benign",
    3: "LLM Phishing",
}

# Mapping of file stems to raw unmasked source paths
SOURCE_PATHS: Dict[str, str] = {
    "human_written_benign": "datasets/datasets_preprocessed/human_written_benign.jsonl",
    "human_written_phishing": "datasets/datasets_preprocessed/human_written_phishing.jsonl",
    "llm_generated_benign": "datasets/datasets_synthetic/llm_generated_benign.jsonl",
    "llm_generated_phishing": "datasets/datasets_synthetic/llm_generated_phishing.jsonl",
}


@dataclass
class UnmaskedEmailRecord:
    id: str
    label: int
    cohort_name: str
    unmasked_subject: str
    unmasked_body: str
    masked_subject: str
    masked_body: str
    mime_bytes: bytes

    @property
    def is_phishing(self) -> bool:
        """Binary ground truth: True for Human Phishing (1) and LLM Phishing (3)."""
        return self.label in (1, 3)

    @property
    def is_synthetic(self) -> bool:
        """Generative provenance: True for LLM Benign (2) and LLM Phishing (3)."""
        return self.label in (2, 3)


def load_unmasked_test_records(
    project_root: Optional[Path | str] = None,
    cache_path: Optional[Path | str] = None,
) -> List[UnmaskedEmailRecord]:
    """Load the exact 6,400 test records and resolve unmasked source texts.

    Performs a single sequential streaming pass over each source file to index
    the needed line numbers without loading entire multi-hundred-megabyte
    files into memory.

    Parameters
    ----------
    project_root : Optional[Path | str]
        Root directory of the thesis project. Defaults to current working directory.
    cache_path : Optional[Path | str]
        Path to ``dataset_splits.pkl``. Defaults to ``models/cache/dataset_splits.pkl``.

    Returns
    -------
    List[UnmaskedEmailRecord]
        List of 6,400 test records with both unmasked and masked representations
        and pre-built RFC 5322 MIME byte payloads.
    """
    if project_root is None:
        project_root = Path.cwd()
    else:
        project_root = Path(project_root)

    if cache_path is None:
        cache_path = project_root / "models" / "cache" / "dataset_splits.pkl"
    else:
        cache_path = Path(cache_path)

    if not cache_path.exists():
        raise FileNotFoundError(f"Dataset splits cache not found at: {cache_path}")

    logger.info("Loading test split IDs from %s...", cache_path)
    with open(cache_path, "rb") as f:
        splits = pickle.load(f)

    test_records = splits.test_records
    logger.info("Found %d test records in cache.", len(test_records))

    # Group needed line indices by file stem
    indices_by_stem: Dict[str, Set[int]] = {}
    for r in test_records:
        stem, idx_str = r.id.rsplit("_", 1)
        line_idx = int(idx_str)
        indices_by_stem.setdefault(stem, set()).add(line_idx)

    # Stream source files and extract unmasked subject & body
    unmasked_lookup: Dict[str, tuple[str, str]] = {}

    for stem, target_indices in indices_by_stem.items():
        if stem not in SOURCE_PATHS:
            raise KeyError(f"Unknown cohort file stem: {stem}")

        source_file = project_root / SOURCE_PATHS[stem]
        if not source_file.exists():
            raise FileNotFoundError(f"Unmasked source file not found: {source_file}")

        logger.info(
            "Extracting %d unmasked records from %s...",
            len(target_indices),
            source_file.name,
        )

        max_target_idx = max(target_indices)
        found_for_stem = 0

        with open(source_file, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if line_idx in target_indices:
                    data = json.loads(line)
                    record_id = f"{stem}_{line_idx}"
                    unmasked_lookup[record_id] = (
                        data.get("subject", "") or "",
                        data.get("body", "") or "",
                    )
                    found_for_stem += 1

                if line_idx >= max_target_idx and found_for_stem == len(target_indices):
                    break

        logger.info("Successfully extracted %d records for %s.", found_for_stem, stem)

    # Reassemble in exact test split sequence
    unmasked_records: List[UnmaskedEmailRecord] = []
    for r in test_records:
        unmasked_subj, unmasked_body = unmasked_lookup[r.id]
        mime_bytes = build_mime_message(
            record_id=r.id,
            subject=unmasked_subj,
            body=unmasked_body,
        )

        unmasked_records.append(
            UnmaskedEmailRecord(
                id=r.id,
                label=r.label,
                cohort_name=COHORT_NAMES.get(r.label, f"Class {r.label}"),
                unmasked_subject=unmasked_subj,
                unmasked_body=unmasked_body,
                masked_subject=r.subject,
                masked_body=r.body,
                mime_bytes=mime_bytes,
            )
        )

    logger.info("Completed assembling %d unmasked test records.", len(unmasked_records))
    return unmasked_records
