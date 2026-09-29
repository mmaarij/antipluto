"""
generation/provenance.py
========================
Provenance and audit utilities for synthetic LLM-generated emails.

Given any synthetic email record (dict or JSON string), ``ProvenanceTracer``
reconstructs the exact deterministic candidate pool and retrieves the
original human seed email that was provided to the LLM.

Example
-------
::

    from antipluto.generation.provenance import trace_seed

    synthetic_record = {
        "source": "gpt-5-mini",
        "seed_idx": 42,
        "label": 1,
        "subject": "Immediate Action Required: Account Verification",
        "body": "...",
    }

    original_seed = trace_seed(synthetic_record)
    print("Original human seed subject:", original_seed["subject"])
    print("Original human seed body:", original_seed["body"])
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

import yaml

from antipluto.utils.io import read_jsonl

logger = logging.getLogger(__name__)


class ProvenanceTracer:
    """
    Reconstructs the original human seed email for any synthetic email record.

    Caches loaded seed files in memory to allow fast repeated queries across
    large cohorts.

    Parameters
    ----------
    config : dict or None
        The configuration dict (or ``generation`` sub-dict). If ``None``,
        loads from ``config_path``.
    config_path : Path or str
        Path to ``configs/default.yaml``.
    project_root : Path or str or None
        Project root directory. Defaults to the current working directory.
    """

    def __init__(
        self,
        config: dict | None = None,
        config_path: Path | str = "antipluto/configs/default.yaml",
        project_root: Path | str | None = None,
    ) -> None:
        if config is None:
            config_file = Path(config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_file}")
            with open(config_file, "r", encoding="utf-8") as fh:
                full_config = yaml.safe_load(fh)
            self.cfg = full_config.get("generation", full_config)
            resolved_root = full_config.get("project_root")
            self.project_root = Path(project_root or resolved_root or ".")
        else:
            self.cfg = config.get("generation", config)
            self.project_root = Path(project_root or ".")

        # Models may be dicts ({name, provider}) or bare strings.
        raw_models = self.cfg.get("models", [])
        self.models: list[str] = [
            m["name"] if isinstance(m, dict) else m
            for m in raw_models
        ]
        self.n_samples: int = self.cfg.get("n_samples", 14000)
        self.random_seed: int = self.cfg.get("random_seed", 42)
        self.buffer_ratio: float = float(self.cfg.get("buffer_ratio", 0.10))
        self.language_filter: str | None = self.cfg.get("language_filter", "en")

        seed_files = self.cfg.get("seed_files", {})
        self.seed_paths = {
            "benign": self.project_root / seed_files.get("benign", "datasets/datasets_preprocessed/human_written_benign.jsonl"),
            "phishing": self.project_root / seed_files.get("phishing", "datasets/datasets_preprocessed/human_written_phishing.jsonl"),
        }

        # In-memory cache for loaded seed datasets
        self._seed_cache: dict[str, list[dict]] = {}
        # In-memory cache for sampled model pools: (cohort, model) -> pool
        self._pool_cache: dict[tuple[str, str], list[dict]] = {}

    def _get_seeds(self, cohort: str) -> list[dict]:
        """Load and language-filter seed dataset with caching."""
        if cohort in self._seed_cache:
            return self._seed_cache[cohort]

        seed_path = self.seed_paths[cohort]
        if not seed_path.exists():
            raise FileNotFoundError(
                f"Seed file for '{cohort}' cohort not found at {seed_path}"
            )

        seeds: list[dict] = []
        for rec in read_jsonl(seed_path):
            if self.language_filter:
                if rec.get("language", "") != self.language_filter:
                    continue
            seeds.append(rec)

        self._seed_cache[cohort] = seeds
        return seeds

    def _get_model_pool(self, cohort: str, model: str) -> list[dict]:
        """Deterministically reconstruct the candidate seed pool for a model."""
        cache_key = (cohort, model)
        if cache_key in self._pool_cache:
            return self._pool_cache[cache_key]

        seeds = self._get_seeds(cohort)

        # Quota per model
        model_names = self.models if model in self.models else [model]
        n_models = len(model_names) if model_names else 1
        target_quota = self.n_samples // n_models
        pool_size = int(target_quota * (1.0 + self.buffer_ratio))

        model_hash = int(hashlib.md5(model.encode("utf-8")).hexdigest(), 16) % 10000
        rng = random.Random(self.random_seed + model_hash)
        pool = rng.choices(seeds, k=pool_size)

        self._pool_cache[cache_key] = pool
        return pool

    def trace(self, synthetic_record: dict[str, Any] | str) -> dict[str, Any]:
        """
        Given a synthetic email record (dict or JSON string), retrieve the
        original human seed email.

        Parameters
        ----------
        synthetic_record : dict or str
            The synthetic email record. Must contain ``source`` (model deployment name),
            ``label`` (0 for benign, 1 for phishing), and ``seed_idx`` (pool index).

        Returns
        -------
        dict
            The complete original human seed email dictionary.
        """
        if isinstance(synthetic_record, str):
            try:
                rec = json.loads(synthetic_record)
            except Exception:
                try:
                    rec = yaml.safe_load(synthetic_record)
                except Exception:
                    import ast
                    try:
                        rec = ast.literal_eval(synthetic_record)
                    except Exception as exc:
                        raise ValueError(f"Could not parse synthetic record string: {exc}") from exc
        elif isinstance(synthetic_record, dict):
            rec = synthetic_record
        else:
            raise TypeError(
                f"Expected dict or JSON string, received {type(synthetic_record).__name__}"
            )

        source = rec.get("source")
        if not source:
            raise ValueError(
                "Synthetic record is missing the 'source' field (model identifier)."
            )

        seed_idx = rec.get("seed_idx")
        if seed_idx is None or not isinstance(seed_idx, int):
            raise ValueError(
                "Synthetic record is missing an integer 'seed_idx' field. "
                "Ensure this record was generated with provenance tracking enabled."
            )

        label = rec.get("label", 0)
        cohort = "benign" if label == 0 else "phishing"

        pool = self._get_model_pool(cohort=cohort, model=source)

        if seed_idx < 0 or seed_idx >= len(pool):
            raise IndexError(
                f"seed_idx {seed_idx} out of range for pool size {len(pool)} "
                f"(model={source!r}, cohort={cohort!r})."
            )

        return pool[seed_idx]


# ---------------------------------------------------------------------------
# Convenience top-level function
# ---------------------------------------------------------------------------


def trace_seed(
    synthetic_record: dict[str, Any] | str,
    config: dict | None = None,
    config_path: Path | str = "antipluto/configs/default.yaml",
    project_root: Path | str | None = None,
) -> dict[str, Any]:
    """
    Convenience function: Trace a synthetic email record back to its human seed.

    Parameters
    ----------
    synthetic_record : dict or str
        Full synthetic JSON item (as a Python dict or JSON string).
    config : dict or None
        Optional config dict override.
    config_path : Path or str
        Path to default.yaml configuration file.
    project_root : Path or str or None
        Path to project root.

    Returns
    -------
    dict
        The full original human seed email record.

    Example
    -------
    ::

        original = trace_seed(synthetic_json_dict)
    """
    tracer = ProvenanceTracer(
        config=config,
        config_path=config_path,
        project_root=project_root,
    )
    return tracer.trace(synthetic_record)
