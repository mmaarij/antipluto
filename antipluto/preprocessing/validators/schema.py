"""
validators/schema.py
====================
Pydantic v2 schema definition for a processed email record.

Every record produced by the preprocessing pipeline is validated against
``EmailRecord`` before being written to the output JSONL stream.  Invalid
records are counted and logged rather than propagated, ensuring the output
files contain only well-formed data.

Schema Fields
-------------
subject : str
    The decoded, stripped RFC-2047 subject header.  May be an empty string
    for emails with no subject header.
body : str
    The cleaned, thread-sliced plain-text body.  Minimum length is enforced
    via a configurable threshold (default: 50 characters).
label : int
    Binary classification ground-truth.
    ``0`` = benign (ham), ``1`` = phishing / spam.
source : str
    Provenance identifier for the originating dataset.
    One of: ``"enron"``, ``"nazario"``, ``"spamassassin"``,
    ``"trec07"``, ``"nigerian_fraud"``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Human corpus provenance identifiers.
HUMAN_SOURCES: frozenset[str] = frozenset({
    "enron",
    "nazario",
    "spamassassin",
    "trec07",
    "nigerian_fraud",
})
# Kept for backwards-compatibility.
VALID_SOURCES = HUMAN_SOURCES
VALID_LABELS: frozenset[int] = frozenset({0, 1})

# Minimum body length in characters.  Records below this threshold are
# rejected during validation.  Can be overridden by the pipeline config.
DEFAULT_MIN_BODY_LENGTH: int = 50


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class EmailRecord(BaseModel):
    """
    Validated output record for a single preprocessed email.

    This model is the canonical contract between the parsing/cleaning layer
    and the downstream JSONL serialisation layer.  All fields are mandatory
    and subject to strict validation rules documented below.
    """

    # --- Core Fields ---
    subject: str = Field(
        default="",
        description="Decoded and whitespace-stripped email subject line.",
    )
    body: str = Field(
        description=(
            "Cleaned plain-text body with HTML stripped, thread history "
            "sliced off, and consecutive blank lines collapsed."
        ),
    )
    label: int = Field(
        description="Binary label: 0 = benign (ham), 1 = phishing / spam.",
    )
    source: str = Field(
        description=(
            "Provenance tag identifying the originating corpus. "
            "Human corpora: 'enron', 'nazario', 'spamassassin', 'trec07', 'nigerian_fraud'. "
            "LLM cohorts: OpenRouter model identifier, e.g. 'openai/gpt-4o-mini'."
        ),
    )
    seed_idx: int | None = Field(
        default=None,
        description=(
            "Deterministic index into the sampled seed pool for synthetic records. "
            "Enables exact one-to-one provenance tracing back to the human seed email."
        ),
    )

    # --- MeAJOR Table II Metadata & Network Features ---
    sender: str = Field(default="", description="SHA-256 hash of sender email address.")
    sender_domain: str = Field(default="", description="Sender email domain.")
    receiver: str = Field(default="", description="SHA-256 hash of primary receiver email address.")
    receiver_domain: str = Field(default="", description="Receiver email domain.")
    date: str = Field(default="", description="RFC 2822 or ISO date string.")
    content_types: list[str] = Field(default_factory=list, description="MIME content types present.")
    urls: list[str] = Field(default_factory=list, description="Extracted URLs.")
    url_count: int = Field(default=0, description="Count of extracted URLs.")
    url_length_max: int = Field(default=0, description="Max URL length.")
    url_length_avg: float = Field(default=0.0, description="Avg URL length.")
    url_subdom_max: int = Field(default=0, description="Max URL subdomains.")
    url_subdom_avg: float = Field(default=0.0, description="Avg URL subdomains.")
    attachment_count: int = Field(default=0, description="Count of attachments.")
    has_attachments: int = Field(default=0, description="1 if has attachments else 0.")
    attachment_types: list[str] = Field(default_factory=list, description="Attachment content types/extensions.")
    language: str = Field(default="en", description="Detected language code.")

    # ------------------------------------------------------------------
    # Field validators
    # ------------------------------------------------------------------

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: int) -> int:
        """Enforce binary label constraint."""
        if v not in VALID_LABELS:
            raise ValueError(
                f"label must be 0 (benign) or 1 (phishing); received {v!r}."
            )
        return v

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        """Enforce non-empty provenance identifier.

        Accepts human corpus names (enron, nazario, etc.) and arbitrary
        LLM model identifiers from OpenRouter (e.g. 'openai/gpt-4o-mini').
        """
        if not v or not v.strip():
            raise ValueError(
                "source must be a non-empty string (corpus name or LLM model identifier)."
            )
        return v

    @field_validator("body")
    @classmethod
    def validate_body_length(cls, v: str) -> str:
        """
        Reject trivially short bodies.

        Note: The pipeline enforces its own configurable ``min_body_length``
        threshold before Pydantic validation.  This validator provides an
        absolute floor of 1 character as a final safety net.
        """
        if not v or not v.strip():
            raise ValueError("body must not be empty or whitespace-only.")
        return v

    @field_validator("subject", "body", mode="before")
    @classmethod
    def coerce_to_str(cls, v: object) -> str:
        """Coerce None or non-string subject/body to empty/string safely."""
        if v is None:
            return ""
        return str(v)

    class Config:
        # Emit clean JSON without extra whitespace when serialising.
        json_encoders = {}
        str_strip_whitespace = False  # Stripping handled upstream by parsers.
