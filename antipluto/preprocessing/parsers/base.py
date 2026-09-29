"""
parsers/base.py
===============
Abstract base class and shared field-extraction utilities for all email
dataset parsers in the Anti-PLUTO framework.

Every concrete parser (Enron, Nazario, SpamAssassin) inherits from
``BaseEmailParser`` and implements the ``parse()`` generator method.
The shared ``extract_email_fields()`` module-level function is intentionally
defined at module scope rather than as an instance method so that it can be
safely imported and called inside ``ProcessPoolExecutor`` worker functions on
Windows, where the "spawn" start method requires all callable objects to be
importable (i.e., non-lambda, non-closure, module-level).

Architecture Note
-----------------
The ``parse()`` method is declared to yield ``dict`` objects (pre-validation
raw records).  Schema validation via Pydantic is performed *outside* the
parser, in ``preprocessing/pipeline.py``, so that parsers remain independent of
the validation layer and can be tested in isolation.
"""

from __future__ import annotations

import email
import logging
import re
from abc import ABC, abstractmethod
from email.message import Message
from pathlib import Path
from typing import Iterator

from antipluto.preprocessing.cleaners.html_stripper import HTMLStripper
from antipluto.preprocessing.cleaners.header_slicer import HeaderSlicer
from antipluto.preprocessing.cleaners.unicode_normalizer import normalize_unicode_text
from antipluto.preprocessing.cleaners.feature_extractors import (
    compute_url_metrics,
    detect_language,
    extract_urls_from_payloads,
    hash_email_address,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System-message subject filter
# ---------------------------------------------------------------------------
#
# Bounce / NDR / auto-reply messages are generated entirely by mail servers
# or helpdesk systems.  They contain no primary human-authored content and
# must not be included in the stylometric training corpus.
#
# The pattern is compiled once at module import time and reused across all
# worker processes (each spawned process re-imports the module).

_NDR_SUBJECT_RE = re.compile(
    r"""(?x)          # verbose mode
    (?:
        Returned\s+mail\s*:               # "Returned mail: User unknown"
      | Delivery\s+(?:                    # "Delivery Status Notification..."
            Status\s+Notification
          | Failure
          | failed
          | Error
        )
      | Undeliverable\s*:                # "Undeliverable: Your message"
      | Undelivered\s+Mail               # "Undelivered Mail Returned"
      | Message\s+status\s*[-\u2013]     # "Message status - undeliverable"
      | DELIVERY\s+FAILURE               # Lotus Notes NDR
      | Mail\s+delivery\s+failed         # Postfix / Sendmail NDR
      | NDR\s*:                          # explicit NDR label
      | Auto(?:matic)?\s*Reply\s*:       # out-of-office / auto-reply
      | Out\s+of\s+(?:the\s+)?Office     # OOO messages
    )
    """,
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# System-generated body boilerplate filter
# ---------------------------------------------------------------------------
#
# Some emails pass the subject filter but contain only auto-generated body
# text with zero human-authored content.  These are detected and dropped
# after the cleaning/slicing stages.

_SYSTEM_BODY_RE = re.compile(
    r"""(?x)
    (?:
        # Lotus Notes encryption migration error
        The\s+original\s+message\s+is\s+encrypted\s+using\s+Lotus\s+Notes

        # Generic auto-generation markers
      | This\s+(?:message|email)\s+(?:was|is)\s+automatically\s+generated
      | This\s+is\s+an\s+automatically\s+generated\s+(?:Delivery|email|message)

        # Lotus Notes / Exchange migration error variants
      | [Ee]ncrypted\s+messages?\s+cannot\s+be\s+migrated

        # Calendar / meeting-request noise
      | This\s+(?:message|email)\s+was\s+sent\s+by\s+the\s+Microsoft\s+Exchange

        # SpamAssassin corpus mv-script artifact.
        # When SA moves files between training folders it emits a shell script
        # of the form "mv NNNNN.hexhash NNNNN.hexhash" — one line per file.
        # These are dataset management artefacts, not emails.
      | ^mv\s+\d{5}\.[a-f0-9]{32}\s+\d{5}\.[a-f0-9]{32}
    )
    """,
    re.IGNORECASE | re.MULTILINE,
)


def decode_subject(msg: Message) -> str:
    """
    Decode and normalise the ``Subject`` header of an email message.

    Handles RFC 2047 encoded-word encoding (e.g. ``=?UTF-8?B?...?=``),
    which is automatically decoded by the ``email.policy.default`` policy
    when ``email.message_from_file()`` is used with that policy.  For the
    legacy ``compat32`` policy (used by SpamAssassin files), we apply
    ``email.header.decode_header`` manually.

    Parameters
    ----------
    msg : email.message.Message
        Parsed email message object.

    Returns
    -------
    str
        Decoded, stripped subject string.  Returns an empty string if no
        ``Subject`` header is present.
    """
    raw_subject = msg.get("Subject", "")
    if not raw_subject:
        return ""

    try:
        decoded_parts = email.header.decode_header(raw_subject)
        parts: list[str] = []
        for part, charset in decoded_parts:
            if isinstance(part, bytes):
                parts.append(part.decode(charset or "utf-8", errors="replace"))
            else:
                parts.append(str(part))
        subj_text = " ".join(parts).strip()
        return normalize_unicode_text(subj_text)
    except Exception:
        return normalize_unicode_text(str(raw_subject).strip())


def extract_email_payloads_and_attachments(
    msg: Message,
) -> tuple[str, str, list[str], int, list[str]]:
    """
    Walk message parts to extract:
    (raw_plain_text, raw_html_text, content_types, attachment_count, attachment_types)
    """
    plain_parts: list[str] = []
    html_parts: list[str] = []
    content_types: set[str] = set()
    attachment_types: list[str] = []
    attachment_count: int = 0

    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype and ctype != "multipart/mixed" and not ctype.startswith("multipart/"):
                content_types.add(ctype)

            cdisp = str(part.get("Content-Disposition", "") or "")
            filename = part.get_filename()

            if "attachment" in cdisp.lower() or filename:
                attachment_count += 1
                if ctype:
                    attachment_types.append(ctype)
                continue

            payload = part.get_payload(decode=True)
            if not payload:
                continue

            charset = part.get_content_charset() or "utf-8"
            decoded = payload.decode(charset, errors="ignore")

            if ctype == "text/plain":
                plain_parts.append(decoded)
            elif ctype == "text/html":
                html_parts.append(decoded)
    else:
        ctype = msg.get_content_type()
        if ctype:
            content_types.add(ctype)

        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            text = payload.decode(charset, errors="ignore")
            if ctype == "text/html":
                html_parts.append(text)
            else:
                plain_parts.append(text)

    raw_plain = "".join(plain_parts)
    raw_html = "".join(html_parts)
    return raw_plain, raw_html, sorted(content_types), attachment_count, attachment_types


def extract_body_text(msg: Message) -> str:
    """
    Extract plain-text body from an email.message.Message object.
    """
    raw_plain, raw_html, _, _, _ = extract_email_payloads_and_attachments(msg)
    return raw_plain if raw_plain else raw_html


def extract_email_fields(
    msg: Message,
    stripper: HTMLStripper,
    slicer: HeaderSlicer,
    min_body_length: int,
    label: int,
    source: str,
) -> dict | None:
    """
    Apply the full cleaning and feature extraction pipeline to a parsed email.

    Extracts all 20 MeAJOR Table II features (Mendes et al., 2025).
    """
    # Stage 0: Drop NDR / bounce messages by subject
    raw_subject = msg.get("Subject", "") or ""
    if _NDR_SUBJECT_RE.search(raw_subject):
        return None

    # Extract payloads, MIME content types, attachments
    raw_plain, raw_html, content_types, attachment_count, attachment_types = (
        extract_email_payloads_and_attachments(msg)
    )
    raw_body = raw_plain if raw_plain else raw_html
    if not raw_body:
        return None

    # Stage 1: Strip HTML markup
    clean_body = stripper.strip(raw_body)
    if not clean_body:
        return None

    # Stage 2: Slice off thread history
    sliced_body = slicer.slice(clean_body)
    if not sliced_body:
        return None

    # Stage 3: Drop system-generated body boilerplate
    if _SYSTEM_BODY_RE.search(sliced_body):
        return None

    # Stage 4: Enforce minimum length
    if len(sliced_body) < min_body_length:
        return None

    # Metadata & Network extraction
    sender_hash, sender_domain = hash_email_address(msg.get("From"))
    receiver_hash, receiver_domain = hash_email_address(msg.get("To"))
    date_str = str(msg.get("Date", "") or "").strip()

    # URL metrics
    urls = extract_urls_from_payloads(raw_html, raw_plain)
    url_metrics = compute_url_metrics(urls)

    # Language detection
    language = detect_language(sliced_body)

    subject = decode_subject(msg)

    return {
        "source": source,
        "sender": sender_hash,
        "sender_domain": sender_domain,
        "receiver": receiver_hash,
        "receiver_domain": receiver_domain,
        "date": date_str,
        "subject": subject,
        "content_types": content_types,
        "body": sliced_body,
        "urls": urls,
        "url_count": url_metrics["url_count"],
        "url_length_max": url_metrics["url_length_max"],
        "url_length_avg": url_metrics["url_length_avg"],
        "url_subdom_max": url_metrics["url_subdom_max"],
        "url_subdom_avg": url_metrics["url_subdom_avg"],
        "attachment_count": attachment_count,
        "has_attachments": 1 if attachment_count > 0 else 0,
        "attachment_types": attachment_types,
        "language": language,
        "label": label,
    }


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class BaseEmailParser(ABC):
    """
    Abstract base class enforcing the parser interface for all dataset sources.

    Concrete implementations must override ``parse()``.  The ``__init__``
    constructor accepts a configuration sub-dictionary for the specific
    source (e.g. ``config["sources"]["enron"]``) and the global processing
    settings (e.g. ``config["processing"]``).

    Parameters
    ----------
    source_config : dict
        Source-specific configuration block from ``default.yaml``.
    processing_config : dict
        Global processing settings (``chunk_size``, ``min_body_length``, etc.).
    project_root : Path
        Absolute path to the thesis project root, used to resolve relative
        dataset paths from the configuration file.
    """

    def __init__(
        self,
        source_config: dict,
        processing_config: dict,
        project_root: Path,
    ) -> None:
        self.source_config = source_config
        self.processing_config = processing_config
        self.project_root = project_root
        self.min_body_length: int = processing_config.get(
            "min_body_length", 50
        )
        self.chunk_size: int = processing_config.get("chunk_size", 1000)
        self.max_workers: int | None = processing_config.get("max_workers")

    @abstractmethod
    def parse(self, source_path: Path) -> Iterator[dict]:
        """
        Ingest the dataset at ``source_path`` and yield raw record dicts.

        Each yielded dict must contain at minimum:
        ``{subject: str, body: str, label: int, source: str}``.

        Schema validation is performed by the pipeline, not the parser.

        Parameters
        ----------
        source_path : Path
            Absolute path to the root of the source dataset.

        Yields
        ------
        dict
            Raw pre-validation record.
        """
        ...
