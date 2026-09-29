"""
preprocessing/pipeline.py
=========================
Top-level orchestrator for the Anti-PLUTO preprocessing pipeline.

The ``PreprocessingPipeline`` class loads the configuration file, instantiates
the enabled parsers, streams records through the cleaning and masking layers,
validates each record against the Pydantic schema, and writes valid records
directly to the output JSONL files without accumulating them in memory.

This streaming design is critical for the Enron corpus, which produces
~500,000+ records.  Accumulating all records before writing would require
several gigabytes of RAM.

Architecture
------------
::

    Config (YAML)
        │
        ▼
    [Parser A] ─── yields raw dicts ───┐
    [Parser B] ─── yields raw dicts ───┤
    [Parser C] ─── yields raw dicts ───┘
                                       │
                                       ▼
                              PIIMasker (optional)
                                       │
                                       ▼
                            Pydantic EmailRecord validator
                                       │
                              ┌────────┴────────┐
                              ▼                 ▼
                   benign.jsonl          phishing.jsonl
                   (label=0)             (label=1)

Windows Multiprocessing Guard
------------------------------
All parsers use ``ProcessPoolExecutor`` internally with module-level worker
functions.  The ``if __name__ == \"__main__\":`` guard in ``__main__.py`` ensures
the pipeline is never launched as a side-effect of a worker process importing
the module.  Calling ``pipeline.run()`` directly (without the guard) in user
scripts is safe for single-threaded use, but **must** be protected by the
guard before spawning workers on Windows.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Iterator

import yaml
from pydantic import ValidationError

from antipluto.masking.masker import PIIMasker
from antipluto.preprocessing.validators.schema import EmailRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output routing helpers
# ---------------------------------------------------------------------------


def _route_label(label: int) -> str:
    """Return a human-readable category name for logging."""
    return "benign" if label == 0 else "phishing"


# ---------------------------------------------------------------------------
# PreprocessingPipeline
# ---------------------------------------------------------------------------


class PreprocessingPipeline:
    """
    Orchestrates the full ingestion → cleaning → masking → validation → JSONL
    export pipeline.

    Parameters
    ----------
    config_path : Path
        Absolute path to the YAML configuration file (typically
        ``antipluto/configs/default.yaml``).
    """

    def __init__(self, config_path: Path) -> None:
        logger.info("Pipeline › Loading configuration from %s", config_path)
        self.config = self._load_config(config_path)
        self.project_root = Path(self.config["project_root"])
        self.raw_dir = self.project_root / self.config["datasets"]["raw_dir"]
        self.output_dir = self.project_root / self.config["datasets"]["output_dir"]
        self.output_dir.mkdir(parents=True, exist_ok=True)

        output_files = self.config["datasets"]["output_files"]
        self.benign_path = self.output_dir / output_files["benign"]
        self.phishing_path = self.output_dir / output_files["phishing"]

        self.processing_cfg = self.config.get("processing", {})
        self.sources_cfg = self.config.get("sources", {})
        self.masking_cfg = self.config.get("masking", {})
        self.min_body_length: int = self.processing_cfg.get("min_body_length", 50)
        self.mode: str = self.processing_cfg.get("mode", "full")
        self.custom_fields: list[str] = self.processing_cfg.get("custom_fields", [])

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        source_filter: str | None = None,
        mode_override: str | None = None,
    ) -> None:
        """
        Execute the full preprocessing pipeline.

        Parameters
        ----------
        source_filter : str or None
            If provided, only the named source is processed
            (e.g. ``"enron"``).
        mode_override : str or None
            If provided, overrides the export mode in configuration
            (``"full"``, ``"minimal"``, or ``"custom"``).
        """
        active_mode = (mode_override or self.mode).lower()
        start_time = time.monotonic()
        logger.info("=" * 60)
        logger.info("Pipeline › Starting preprocessing run (mode: %s).", active_mode)
        if source_filter:
            logger.info("Pipeline › Source filter active: %s", source_filter)
        logger.info("=" * 60)

        masker = self._build_masker()

        # Open both output files for appending so that incremental /
        # per-source runs do not overwrite existing data.
        with (
            open(self.benign_path, "a", encoding="utf-8") as benign_fh,
            open(self.phishing_path, "a", encoding="utf-8") as phishing_fh,
        ):
            output_handles = {0: benign_fh, 1: phishing_fh}
            total_stats: dict[str, int] = {
                "kept": 0, "invalid": 0, "skipped_sources": 0
            }

            for source_name, source_cfg in self.sources_cfg.items():
                if not source_cfg.get("enabled", False):
                    logger.info(
                        "Pipeline › Source '%s' is disabled — skipping.", source_name
                    )
                    total_stats["skipped_sources"] += 1
                    continue

                if source_filter and source_name != source_filter:
                    continue

                stats = self._run_source(
                    source_name=source_name,
                    source_cfg=source_cfg,
                    masker=masker,
                    output_handles=output_handles,
                    active_mode=active_mode,
                )
                total_stats["kept"] += stats["kept"]
                total_stats["invalid"] += stats["invalid"]

        elapsed = time.monotonic() - start_time
        logger.info("=" * 60)
        logger.info(
            "Pipeline › Run complete in %.1fs. "
            "total_kept=%d total_invalid=%d",
            elapsed,
            total_stats["kept"],
            total_stats["invalid"],
        )
        logger.info("Benign output  -> %s", self.benign_path)
        logger.info("Phishing output -> %s", self.phishing_path)
        logger.info("=" * 60)

    def validate_file(self, jsonl_path: Path) -> dict[str, int]:
        """
        Perform a dry-run schema validation pass on an existing JSONL file.

        Parameters
        ----------
        jsonl_path : Path
            Path to the ``.jsonl`` file to validate.

        Returns
        -------
        dict[str, int]
            Dictionary with keys ``"valid"``, ``"invalid"``, ``"total"``.
        """
        valid = 0
        invalid = 0
        logger.info("Validation › Scanning %s ...", jsonl_path)

        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    EmailRecord(**data)
                    valid += 1
                except (json.JSONDecodeError, ValidationError) as exc:
                    logger.warning(
                        "Validation › Line %d failed: %s", line_num, exc
                    )
                    invalid += 1

        total = valid + invalid
        logger.info(
            "Validation › Complete. total=%d valid=%d invalid=%d",
            total, valid, invalid,
        )
        return {"valid": valid, "invalid": invalid, "total": total}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_source(
        self,
        source_name: str,
        source_cfg: dict,
        masker: PIIMasker,
        output_handles: dict[int, object],
        active_mode: str = "full",
    ) -> dict[str, int]:
        """
        Process a single data source end-to-end.
        """
        source_path = self.raw_dir / source_cfg["path"]
        logger.info(
            "Pipeline › [%s] Starting. source_path=%s",
            source_name.upper(),
            source_path,
        )

        if not source_path.exists():
            logger.error(
                "Pipeline › [%s] Source path does not exist: %s — skipping.",
                source_name.upper(),
                source_path,
            )
            return {"kept": 0, "invalid": 0}

        parser = self._build_parser(source_name)
        kept = 0
        invalid = 0
        dupes = 0

        # Per-source content-hash deduplication set.
        seen_hashes: set[str] = set()

        for raw_record in parser.parse(source_path):
            # Content-based deduplication.
            fingerprint = hashlib.md5(
                (raw_record.get("subject", "") + raw_record.get("body", "")).encode(
                    "utf-8", errors="ignore"
                )
            ).hexdigest()
            if fingerprint in seen_hashes:
                dupes += 1
                continue
            seen_hashes.add(fingerprint)

            # Apply PII masking to subject and body independently.
            if masker.enabled:
                raw_record["subject"] = masker.mask(raw_record.get("subject", ""))
                raw_record["body"] = masker.mask(raw_record.get("body", ""))

            # Enforce min_body_length before Pydantic (cheap early reject).
            if len(raw_record.get("body", "")) < self.min_body_length:
                invalid += 1
                continue

            # Pydantic schema validation.
            try:
                record = EmailRecord(**raw_record)
            except ValidationError as exc:
                logger.debug(
                    "Pipeline › [%s] Validation failed: %s", source_name, exc
                )
                invalid += 1
                continue

            # Export filtering according to active_mode
            record_dict = {
                k: v for k, v in record.model_dump().items()
                if k != "seed_idx" or v is not None
            }
            if active_mode == "minimal":
                minimal_keys = {"subject", "body", "label", "source"}
                record_dict = {k: v for k, v in record_dict.items() if k in minimal_keys}
            elif active_mode == "custom" and self.custom_fields:
                custom_keys = set(self.custom_fields) | {"label"}
                record_dict = {k: v for k, v in record_dict.items() if k in custom_keys}

            # Stream directly to the appropriate output file.
            fh = output_handles[record.label]
            fh.write(  # type: ignore[attr-defined]
                json.dumps(record_dict, ensure_ascii=False) + "\n"
            )
            kept += 1

        logger.info(
            "Pipeline › [%s] Source complete. kept=%d dupes_removed=%d invalid=%d",
            source_name.upper(),
            kept,
            dupes,
            invalid,
        )
        return {"kept": kept, "invalid": invalid}

    def _build_parser(self, source_name: str) -> object:
        """
        Instantiate and return the appropriate parser for ``source_name``.
        """
        source_cfg = self.sources_cfg[source_name]

        if source_name == "enron":
            from antipluto.preprocessing.parsers.enron import EnronParser
            return EnronParser(source_cfg, self.processing_cfg, self.project_root)

        if source_name == "nazario":
            from antipluto.preprocessing.parsers.nazario import NazarioParser
            return NazarioParser(source_cfg, self.processing_cfg, self.project_root)

        if source_name == "spamassassin":
            from antipluto.preprocessing.parsers.spamassassin import SpamAssassinParser
            return SpamAssassinParser(source_cfg, self.processing_cfg, self.project_root)

        if source_name == "trec07":
            from antipluto.preprocessing.parsers.trec import TrecParser
            return TrecParser(source_cfg, self.processing_cfg, self.project_root)

        if source_name == "nigerian_fraud":
            from antipluto.preprocessing.parsers.nigerian_fraud import NigerianFraudParser
            return NigerianFraudParser(source_cfg, self.processing_cfg, self.project_root)

        raise ValueError(
            f"Unknown source '{source_name}'. "
            f"Valid sources: enron, nazario, spamassassin, trec07, nigerian_fraud."
        )

    def _build_masker(self) -> PIIMasker:
        """
        Instantiate the PII masker from the masking configuration block.
        """
        enabled = self.masking_cfg.get("enabled", False)
        spacy_model = self.masking_cfg.get("spacy_model", "en_core_web_sm")
        masker = PIIMasker(spacy_model=spacy_model, enabled=enabled)
        if enabled:
            logger.info("Pipeline › PII masking ENABLED (model: %s).", spacy_model)
        else:
            logger.info("Pipeline › PII masking DISABLED (raw extraction mode).")
        return masker

    @staticmethod
    def _load_config(config_path: Path) -> dict:
        """
        Load and parse the YAML configuration file.
        """
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Did you forget to pass --config <path>?"
            )
        with open(config_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
