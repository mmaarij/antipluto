"""
antipluto
=======
Anti-PLUTO — Anti-Phishing Lexical Utilities and Threat Observation.

A modular, end-to-end Python framework for:

  - Preprocessing raw email corpora into clean, schema-validated JSONL datasets.
  - Anonymising PII via a two-stage MeAJOR-compatible masking pipeline.
  - Generating LLM-assisted and LLM-generated email cohorts via OpenRouter.
  - Training a dual-branch stylometric + semantic phishing classifier.
  - Evaluating classifier performance against SpamAssassin / Rspamd baselines.
  - Serving real-time phishing predictions via a FastAPI endpoint.

Academic Context
----------------
MSc Computing Thesis — Dual-branch detection of LLM-generated phishing email.
(Stylometric TF-IDF + Semantic DeBERTaV3 → XGBoost)

Subpackages
-----------
antipluto.preprocessing   Corpus ingestion, cleaning, and schema validation.
antipluto.masking         Two-stage PII anonymisation (spaCy NER + regex).
antipluto.generation      LLM cohort generation via OpenRouter API.
antipluto.classifier      Dual-branch stylometric + semantic classifier.
antipluto.baseline        SpamAssassin / Rspamd comparison harness.
antipluto.api             FastAPI real-world prediction endpoint.
antipluto.utils           Shared JSONL I/O utilities.

License: MIT
"""

__version__ = "2.0.0"
__author__ = "Maarij"
