"""
antipluto.baseline.mime_envelope
==============================
Generates standardized, RFC 5322 compliant MIME email envelopes with
neutral dummy transmission headers.

Academic Rationale
------------------
Standardizing clean, valid headers across all four evaluation cohorts
eliminates confounding transmission artifacts (e.g. outdated MTA hops in
legacy phishing sets vs. zero hops in synthetic LLM emails). This isolates
the benchmark strictly to linguistic content, stylometry, semantic intent,
and payload URLs.
"""

from __future__ import annotations

from email.message import EmailMessage
from typing import Optional


DEFAULT_SENDER = "user@example.com"
DEFAULT_RECIPIENT = "recipient@example.com"
DEFAULT_DATE = "Thu, 10 Sep 2026 12:00:00 +0000"


def build_mime_message(
    record_id: str,
    subject: str,
    body: str,
    sender: str = DEFAULT_SENDER,
    recipient: str = DEFAULT_RECIPIENT,
    date_str: str = DEFAULT_DATE,
    content_type: str = "text/plain",
    charset: str = "utf-8",
) -> bytes:
    """Build a standard RFC 5322 MIME message byte string.

    Parameters
    ----------
    record_id : str
        Unique identifier for the sample (used in Message-ID header).
    subject : str
        Email subject line.
    body : str
        Email body text.
    sender : str
        Envelope From header.
    recipient : str
        Envelope To header.
    date_str : str
        Fixed RFC 2822 date header to prevent timestamp-based rule penalties.
    content_type : str
        MIME content type (default 'text/plain').
    charset : str
        Character encoding (default 'utf-8').

    Returns
    -------
    bytes
        RFC 5322 compliant raw MIME message bytes ready for transmission to
        SpamAssassin (spamd) or Rspamd.
    """
    # RFC 5322 header fields must not contain unencoded raw linebreaks
    clean_subj = " ".join((subject or "").replace("\r", " ").replace("\n", " ").split())
    clean_sender = " ".join((sender or DEFAULT_SENDER).replace("\r", " ").replace("\n", " ").split())
    clean_recipient = " ".join((recipient or DEFAULT_RECIPIENT).replace("\r", " ").replace("\n", " ").split())
    clean_date = " ".join((date_str or DEFAULT_DATE).replace("\r", " ").replace("\n", " ").split())

    msg = EmailMessage()
    msg["From"] = clean_sender
    msg["To"] = clean_recipient
    msg["Date"] = clean_date
    msg["Message-ID"] = f"<test-{record_id}@example.com>"
    msg["Subject"] = clean_subj
    msg.set_content(body or "", subtype="plain", charset=charset)

    return msg.as_bytes()
