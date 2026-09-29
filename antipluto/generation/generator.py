"""
generation/generator.py
=======================
Batch orchestrator for LLM email cohort generation.

The ``CohortGenerator`` reads seed emails from the human-written JSONL
files, samples them deterministically according to the configuration,
dispatches batches to the configured LLM providers via provider-agnostic
``BaseGenerationClient`` instances, enriches each response with computed
fields (URLs, language) and provenance (``seed_idx``), validates against
``EmailRecord``, and streams valid records directly to the output JSONL file.

Provider Routing
----------------
Each model in the configuration specifies its own ``provider`` key, which
maps to a provider entry in the ``providers`` section of the config. The
generator creates one client per unique provider and routes each model's
API calls to the correct client automatically.

Pipeline & Provenance
---------------------
::

    human_written_*.jsonl
           │
           ▼
    filter(language == "en")
           │
           ▼
    deterministic pool per model: n_target × (1 + buffer_ratio)
           │
           ├─── Model A pool ──▶ batch_size slices ──▶ Provider API ──┐
           ├─── Model B pool ──▶ batch_size slices ──▶ Provider API ──┤
           └─── Model C pool ──▶ batch_size slices ──▶ Provider API ──┘
                                                                       │
                                                            enrich + seed_idx + validate
                                                                       │
                                                          llm_generated_*.jsonl

Self-Describing & Batch-Atomic Resume
-------------------------------------
Every generated record contains a ``seed_idx`` field indicating its exact
position in the model's deterministic seed pool.

- **Batch Atomicity:** If a batch succeeds and produces the expected number
  of valid records, all records in the batch are written with their corresponding
  ``seed_idx`` values. If an API call fails or the model returns an incomplete
  batch, the entire batch is skipped cleanly and replacement candidates are
  drawn sequentially from the deterministic buffer pool.
- **Traceability:** Any record in the output JSONL can be mapped directly to
  the originating human seed email via ``model_pool[record["seed_idx"]]``.
  Records with ``seed_idx < target_quota`` came from the primary sample;
  records with ``seed_idx >= target_quota`` were drawn from the buffer.
- **Self-Describing Resume:** Resume state is derived directly from existing
  records in the output JSONL file. No external sidecars or checkpoint files
  are needed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from antipluto.generation.client import BaseGenerationClient, create_client
from antipluto.generation.prompts import (
    format_benign_prompt,
    format_benign_prompt_v3,
    format_phishing_prompt,
    format_phishing_prompt_v3,
)
from antipluto.preprocessing.cleaners.feature_extractors import (
    compute_url_metrics,
    detect_language,
    extract_urls_from_payloads,
)
from antipluto.preprocessing.validators.schema import EmailRecord
from antipluto.utils.io import read_jsonl

logger = logging.getLogger(__name__)

CohortType = Literal["benign", "phishing"]


# ---------------------------------------------------------------------------
# CohortGenerator
# ---------------------------------------------------------------------------


class CohortGenerator:
    """
    Generates a synthetic LLM email cohort from human-written seed emails.

    Parameters
    ----------
    config : dict
        The ``generation`` sub-dict from ``default.yaml``.
    project_root : Path
        Absolute path to the thesis project root.
    api_key : str or None
        Optional global API key override (from CLI ``--api-key``).
        Passed to all providers; each provider's own config takes
        precedence over this only if ``api_key_override`` is ``None``.
    """

    def __init__(
        self,
        config: dict,
        project_root: Path,
        api_key: str | None = None,
    ) -> None:
        self.cfg = config
        self.project_root = project_root
        self.api_key_override = api_key or None

        # ── Parse providers ─────────────────────────────────────────────
        self.providers: dict[str, dict] = config.get("providers", {})
        if not self.providers:
            raise ValueError(
                "No providers configured. Add a 'providers' section to the "
                "generation config, e.g.:\n"
                "  providers:\n"
                "    openai:\n"
                "      endpoint: https://...\n"
                "      api_key_env: FOUNDRY_API_KEY"
            )

        # ── Parse models (list of {name, provider} dicts) ──────────────
        raw_models = config.get("models", [])
        if not raw_models:
            raise ValueError("No models configured in generation config.")

        self.models: list[dict[str, str]] = []
        for entry in raw_models:
            if not isinstance(entry, dict) or "name" not in entry:
                raise ValueError(
                    f"Each model entry must be a dict with 'name' and 'provider' "
                    f"keys. Got: {entry!r}"
                )
            provider = entry.get("provider", "openai")
            if provider not in self.providers:
                raise ValueError(
                    f"Model '{entry['name']}' references unknown provider "
                    f"'{provider}'. Available providers: "
                    f"{sorted(self.providers.keys())}"
                )
            self.models.append({"name": entry["name"], "provider": provider})

        # Model names for convenience.
        self.model_names: list[str] = [m["name"] for m in self.models]

        # ── Generation parameters ──────────────────────────────────────
        self.n_samples: int = config["n_samples"]
        self.batch_size: int = config.get("batch_size", 10)
        self.language_filter: str | None = config.get("language_filter", "en") or None
        self.random_seed: int = config.get("random_seed", 42)
        self.buffer_ratio: float = float(config.get("buffer_ratio", 0.10))
        self.request_timeout: float = float(config.get("request_timeout", 120))
        self.max_retries: int = config.get("max_retries", 3)
        self.retry_delay: float = float(config.get("retry_delay", 5))
        self.max_tokens: int = config.get("max_tokens", 4096)
        self.temperature: float = float(config.get("temperature", 0.9))

        output_dir = project_root / config["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        output_files = config["output_files"]
        self.output_paths: dict[CohortType, Path] = {
            "benign": output_dir / output_files["benign"],
            "phishing": output_dir / output_files["phishing"],
        }

        seed_files = config["seed_files"]
        self.seed_paths: dict[CohortType, Path] = {
            "benign": project_root / seed_files["benign"],
            "phishing": project_root / seed_files["phishing"],
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        cohort: CohortType,
        model_filter: str | None = None,
        n_samples_override: int | None = None,
        overwrite: bool = False,
    ) -> None:
        """
        Generate the specified cohort and write records to the output JSONL.

        Parameters
        ----------
        cohort : "benign" or "phishing"
            Which cohort to generate.
        model_filter : str or None
            If provided, only use this model (ignores the models list).
        n_samples_override : int or None
            Override the configured ``n_samples`` for this run.
        overwrite : bool
            If ``True``, truncate the output file before generating.
            If ``False`` (default), append and skip already-generated records.
        """
        n_total = n_samples_override or self.n_samples
        output_path = self.output_paths[cohort]
        seed_path = self.seed_paths[cohort]
        label = 0 if cohort == "benign" else 1
        prompt_version = str(self.cfg.get("prompt_version", "v2")).lower()
        if prompt_version == "v3":
            prompt_fn = format_benign_prompt_v3 if cohort == "benign" else format_phishing_prompt_v3
        else:
            prompt_fn = format_benign_prompt if cohort == "benign" else format_phishing_prompt

        # ── Resolve active models ──────────────────────────────────────
        if model_filter:
            active_models = [m for m in self.models if m["name"] == model_filter]
            if not active_models:
                logger.error(
                    "Generator › Model filter '%s' matched no configured models.",
                    model_filter,
                )
                return
        else:
            active_models = self.models

        active_model_names = [m["name"] for m in active_models]

        logger.info("=" * 60)
        logger.info(
            "Generator › Starting '%s' cohort. n_samples=%d models=%s (buffer_ratio=%.2f)",
            cohort, n_total, active_model_names, self.buffer_ratio,
        )
        logger.info("=" * 60)

        # ── Handle resume / overwrite ──────────────────────────────────────
        target_quotas = self._compute_quotas(n_total, active_model_names)
        existing_indices: dict[str, set[int]] = {m: set() for m in active_model_names}

        if overwrite:
            if output_path.exists():
                output_path.unlink()
                logger.info("Generator › Overwrite mode — cleared %s.", output_path.name)
            already_done = 0
        elif output_path.exists():
            existing_indices = self._scan_existing_indices(output_path, active_model_names)
            already_done = sum(len(indices) for indices in existing_indices.values())

            if already_done >= n_total:
                logger.info(
                    "Generator › Output already contains %d records (target=%d). "
                    "Nothing to do. Pass --overwrite to regenerate.",
                    already_done, n_total,
                )
                return

            logger.info(
                "Generator › Resuming: %d/%d records already written.",
                already_done, n_total,
            )
            logger.info(
                "Generator › Per-model progress: %s",
                {m: f"{len(existing_indices[m])}/{target_quotas[m]}" for m in active_model_names},
            )
        else:
            already_done = 0

        # ── Load and filter seed emails ────────────────────────────────────
        seeds = self._load_seeds(seed_path)
        if not seeds:
            logger.error(
                "Generator › No seeds found at %s — aborting.", seed_path
            )
            return

        # ── Create clients per unique provider ─────────────────────────────
        used_providers = {m["provider"] for m in active_models}
        total_written = already_done
        start_time = time.monotonic()

        with ExitStack() as stack:
            clients: dict[str, BaseGenerationClient] = {}
            for prov_name in sorted(used_providers):
                prov_cfg = self.providers[prov_name]
                client = create_client(
                    provider=prov_name,
                    provider_config=prov_cfg,
                    timeout=self.request_timeout,
                    max_retries=self.max_retries,
                    retry_delay=self.retry_delay,
                    api_key_override=self.api_key_override,
                )
                stack.enter_context(client)
                clients[prov_name] = client
                logger.info(
                    "Generator › Initialized '%s' provider client (endpoint: %s).",
                    prov_name, prov_cfg.get("endpoint", "?"),
                )

            # ── Run generation per model ───────────────────────────────────
            for model_entry in active_models:
                model = model_entry["name"]
                provider = model_entry["provider"]
                client = clients[provider]

                target_quota = target_quotas[model]
                done_indices = existing_indices.get(model, set())
                model_written = len(done_indices)

                if model_written >= target_quota:
                    logger.info(
                        "Generator › [%s] Already complete (%d/%d) — skipping.",
                        model, model_written, target_quota,
                    )
                    continue

                # Sample deterministic pool with buffer for this model.
                pool_size = int(target_quota * (1.0 + self.buffer_ratio))
                model_hash = int(hashlib.md5(model.encode("utf-8")).hexdigest(), 16) % 10000
                rng = random.Random(self.random_seed + model_hash)
                model_pool = rng.choices(seeds, k=pool_size)

                # Determine starting pool index on resume.
                if done_indices:
                    pool_idx = ((max(done_indices) // self.batch_size) + 1) * self.batch_size
                else:
                    pool_idx = 0

                logger.info(
                    "Generator › [%s] Target: %d records (written: %d, starting pool_idx: %d, pool_size: %d).",
                    model, target_quota, model_written, pool_idx, pool_size,
                )

                failed_batches = 0

                while model_written < target_quota and pool_idx < pool_size:
                    batch_needed = min(self.batch_size, target_quota - model_written)
                    batch_end = min(pool_idx + batch_needed, pool_size)
                    batch_indices = list(range(pool_idx, batch_end))
                    batch_seeds = [model_pool[i] for i in batch_indices]

                    system_prompt, user_prompt = prompt_fn(batch_seeds)

                    raw_records = client.generate_batch(
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )

                    if raw_records is None:
                        failed_batches += 1
                        logger.warning(
                            "Generator › [%s] Batch at pool_idx %d (%d seeds) API error — "
                            "skipping batch, drawing next from buffer.",
                            model, pool_idx, len(batch_indices),
                        )
                        pool_idx += len(batch_indices)
                        continue

                    # Enrich and validate records with deterministic seed provenance.
                    validated_records: list[dict] = []
                    for offset, raw in enumerate(raw_records):
                        if offset >= len(batch_indices):
                            break
                        rec_seed_idx = batch_indices[offset]
                        record = self._enrich_and_validate(
                            raw, model, label, seed_idx=rec_seed_idx
                        )
                        if record is not None:
                            validated_records.append(record)

                    # Batch-atomic rule: require all requested seeds in the batch to be valid.
                    if len(validated_records) == len(batch_indices):
                        with open(output_path, "a", encoding="utf-8") as fh:
                            for record in validated_records:
                                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                        model_written += len(validated_records)
                        total_written += len(validated_records)
                        logger.info(
                            "Generator › [%s] Batch at pool_idx %d OK — wrote %d records "
                            "(model: %d/%d, overall: %d/%d).",
                            model, pool_idx, len(validated_records),
                            model_written, target_quota, total_written, n_total,
                        )
                    else:
                        failed_batches += 1
                        logger.warning(
                            "Generator › [%s] Batch at pool_idx %d incomplete (%d/%d valid) — "
                            "skipping batch, drawing next from buffer.",
                            model, pool_idx, len(validated_records), len(batch_indices),
                        )

                    pool_idx += len(batch_indices)

                if model_written < target_quota:
                    logger.warning(
                        "Generator › [%s] Pool exhausted (%d seeds) before reaching target "
                        "(%d/%d written, %d failed batches). Increase buffer_ratio in config.",
                        model, pool_size, model_written, target_quota, failed_batches,
                    )
                else:
                    logger.info(
                        "Generator › [%s] Target reached: %d/%d records written (highest seed_idx: %d, failed_batches: %d).",
                        model, model_written, target_quota, pool_idx - 1, failed_batches,
                    )

        elapsed = time.monotonic() - start_time
        logger.info("=" * 60)
        logger.info(
            "Generator › '%s' cohort done in %.1fs. "
            "total_written=%d target=%d output=%s",
            cohort, elapsed, total_written, n_total, output_path,
        )
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scan_existing_indices(
        self,
        output_path: Path,
        models: list[str],
    ) -> dict[str, set[int]]:
        """
        Scan output JSONL file to extract already generated seed_idx values per model.
        """
        existing: dict[str, set[int]] = {m: set() for m in models}
        legacy_counts: dict[str, int] = {m: 0 for m in models}

        for rec in read_jsonl(output_path):
            src = rec.get("source", "")
            if src in existing:
                s_idx = rec.get("seed_idx")
                if s_idx is not None and isinstance(s_idx, int):
                    existing[src].add(s_idx)
                else:
                    legacy_counts[src] += 1

        # Fallback for legacy records without seed_idx: assume sequential 0..N-1
        for m in models:
            if not existing[m] and legacy_counts[m] > 0:
                logger.info(
                    "Generator › [%s] Found %d legacy records without seed_idx — "
                    "estimating sequential indices [0..%d].",
                    m, legacy_counts[m], legacy_counts[m] - 1,
                )
                existing[m] = set(range(legacy_counts[m]))

        return existing

    def _load_seeds(self, seed_path: Path) -> list[dict]:
        """
        Load and optionally language-filter seed records from a JSONL file.
        """
        if not seed_path.exists():
            logger.error("Generator › Seed file not found: %s", seed_path)
            return []

        seeds: list[dict] = []
        skipped_lang = 0

        for record in read_jsonl(seed_path):
            if self.language_filter:
                if record.get("language", "") != self.language_filter:
                    skipped_lang += 1
                    continue
            seeds.append(record)

        logger.info(
            "Generator › Loaded %d seeds from %s "
            "(language_filter=%r, skipped_lang=%d).",
            len(seeds), seed_path.name,
            self.language_filter, skipped_lang,
        )
        return seeds

    def _compute_quotas(self, n_total: int, models: list[str]) -> dict[str, int]:
        """
        Divide ``n_total`` samples evenly across ``models``.

        Each model gets ``floor(n_total / n_models)`` samples. The remainder
        is distributed one-extra to the first ``remainder`` models so the
        total always sums to exactly ``n_total``.
        """
        n_models = len(models)
        base = n_total // n_models
        remainder = n_total % n_models
        quotas: dict[str, int] = {
            model: base + (1 if i < remainder else 0)
            for i, model in enumerate(models)
        }
        logger.info(
            "Generator › Target quotas: %s",
            {m: q for m, q in quotas.items()},
        )
        return quotas

    def _enrich_and_validate(
        self,
        raw: dict,
        model: str,
        label: int,
        seed_idx: int | None = None,
    ) -> dict | None:
        """
        Enrich an LLM-generated ``{"subject", "body"}`` record with
        computed fields, seed provenance, and validate against ``EmailRecord``.

        Parameters
        ----------
        raw : dict
            Raw LLM output dict with at least ``subject`` and ``body``.
        model : str
            Model deployment name (used as the ``source`` field).
        label : int
            Ground-truth label (0 = benign, 1 = phishing).
        seed_idx : int or None
            Deterministic index into the model's seed pool.

        Returns
        -------
        dict or None
            Full validated record dict, or ``None`` if validation fails.
        """
        if not isinstance(raw, dict):
            logger.debug(
                "Generator › Skipping non-dict element in parsed array (got %s).",
                type(raw).__name__,
            )
            return None

        subject = str(raw.get("subject") or "").strip()
        body = str(raw.get("body") or "").strip()

        if not body:
            logger.debug("Generator › Skipping record with empty body.")
            return None

        if not subject:
            logger.debug("Generator › Skipping record with empty subject.")
            return None

        # Compute URL features from generated body
        urls = extract_urls_from_payloads("", body)
        url_metrics = compute_url_metrics(urls)
        language = detect_language(body)

        record_data = {
            "source": model,
            "seed_idx": seed_idx,
            "subject": subject,
            "body": body,
            "label": label,
            "language": language,
            "urls": urls,
            "url_count": url_metrics["url_count"],
            "url_length_max": url_metrics["url_length_max"],
            "url_length_avg": url_metrics["url_length_avg"],
            "url_subdom_max": url_metrics["url_subdom_max"],
            "url_subdom_avg": url_metrics["url_subdom_avg"],
            "content_types": ["text/plain"],
            "sender": "",
            "sender_domain": "",
            "receiver": "",
            "receiver_domain": "",
            "date": "",
            "attachment_count": 0,
            "has_attachments": 0,
            "attachment_types": [],
        }

        try:
            validated = EmailRecord(**record_data)
            return validated.model_dump()
        except ValidationError as exc:
            logger.debug(
                "Generator › Validation failed for generated record: %s", exc
            )
            return None
