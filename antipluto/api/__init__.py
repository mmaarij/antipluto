"""
antipluto.api
===========
FastAPI REST API package for real-world Anti-PLUTO email classification.
"""

from __future__ import annotations

from antipluto.api.app import create_app
from antipluto.api.parser import parse_rfc822_email
from antipluto.api.pipeline import EmailIngestionPipeline, PreprocessedEmail
from antipluto.api.schemas import (
    ConfidenceScores,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ProcessingMetadata,
)
from antipluto.api.service import InferenceService

__all__ = [
    "create_app",
    "InferenceService",
    "EmailIngestionPipeline",
    "PreprocessedEmail",
    "parse_rfc822_email",
    "PredictRequest",
    "PredictResponse",
    "ConfidenceScores",
    "ProcessingMetadata",
    "HealthResponse",
]
