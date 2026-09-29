"""
antipluto.api.schemas
===================
Pydantic data contracts for the Anti-PLUTO REST API.
"""

from __future__ import annotations

from typing import Dict, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Structured email prediction request payload."""

    subject: str = Field(
        default="",
        description="The subject line of the email.",
        example="Urgent: Verify your account immediately",
    )
    body: str = Field(
        default="",
        description="The body of the email (plain text or HTML markup).",
        example="Dear customer, please click http://security-update-bank.com/login to confirm your identity.",
    )
    mode: Optional[str] = Field(
        default="fusion",
        description="Inference representation mode: 'fusion' (dual-branch stylometric + semantic), 'semantic' (DeBERTa-v3 only), or 'stylometric' (TF-IDF only).",
        example="fusion",
    )


class ConfidenceScores(BaseModel):
    """Percentage confidences across the 4 evaluation cohorts."""

    human_benign: float = Field(..., description="Probability of Human Benign class (0.0 to 1.0)")
    human_phishing: float = Field(..., description="Probability of Human Phishing class (0.0 to 1.0)")
    llm_benign: float = Field(..., description="Probability of LLM Benign class (0.0 to 1.0)")
    llm_phishing: float = Field(..., description="Probability of LLM Phishing class (0.0 to 1.0)")


class ProcessingMetadata(BaseModel):
    """Diagnostic and runtime telemetry about the ingestion and inference pipeline."""

    classifier: str = Field(..., description="Active classification head (xgboost, rf, mlp, cnn, deberta_direct, stylo_linear)")
    mode: str = Field(default="fusion", description="Inference representation mode executed ('fusion', 'semantic', 'stylometric')")
    latency_ms: float = Field(..., description="Total end-to-end inference latency in milliseconds")
    detected_urls: int = Field(default=0, description="Count of distinct URLs detected in the payload")
    html_stripped: bool = Field(default=False, description="Whether HTML tags were detected and stripped")
    thread_history_sliced: bool = Field(default=False, description="Whether forwarded or reply thread history was sliced off")


class PredictResponse(BaseModel):
    """Complete prediction output schema."""

    prediction: str = Field(..., description="Predicted class label (Human Benign, Human Phishing, LLM Benign, LLM Phishing)")
    is_phishing: bool = Field(..., description="Binary determination: True if classified as Human or LLM Phishing")
    is_llm_generated: bool = Field(..., description="Provenance determination: True if classified as LLM Benign or LLM Phishing")
    phishing_probability: float = Field(..., description="Cumulative probability of phishing (P(Human Phish) + P(LLM Phish))")
    confidence_scores: Dict[str, float] = Field(..., description="Human-readable dictionary of class probabilities")
    processed_preview: str = Field(..., description="Preview of normalized, stripped, and masked text fed to the model")
    metadata: ProcessingMetadata


class HealthResponse(BaseModel):
    """System health and model readiness status."""

    status: str = Field(default="healthy")
    version: str = Field(default="1.0.0")
    classifier: str
    device: str
    available_modes: list[str] = Field(default_factory=lambda: ["fusion", "semantic", "stylometric"])
    models_loaded: Dict[str, bool]
