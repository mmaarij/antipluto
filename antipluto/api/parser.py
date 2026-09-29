"""
antipluto.api.parser
==================
RFC 5322 MIME message (.eml) parser extracting subject and body payloads.
"""

from __future__ import annotations

import email
from email import policy
from email.message import EmailMessage, Message
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def parse_rfc822_email(raw_content: bytes | str) -> Tuple[str, str]:
    """Parse raw RFC 5322 MIME email bytes or text and extract subject and body.

    Handles multipart MIME structures (preferring text/plain, falling back to
    text/html), decoded character sets, and encoded header words.

    Parameters
    ----------
    raw_content : bytes | str
        Raw .eml email file bytes or string.

    Returns
    -------
    Tuple[str, str]
        Extracted (subject, body).
    """
    if isinstance(raw_content, str):
        msg: Message = email.message_from_string(raw_content, policy=policy.default)
    else:
        msg = email.message_from_bytes(raw_content, policy=policy.default)

    # 1. Extract subject line
    subject = msg.get("Subject", "") or ""

    # 2. Extract body payload
    body = ""
    if msg.is_multipart():
        plain_parts = []
        html_parts = []

        for part in msg.walk():
            # Skip container parts and non-text attachments
            if part.is_multipart():
                continue

            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))

            # Skip attachments
            if "attachment" in content_disposition.lower():
                continue

            try:
                payload = part.get_content()
                if isinstance(payload, str):
                    if content_type == "text/plain":
                        plain_parts.append(payload)
                    elif content_type == "text/html":
                        html_parts.append(payload)
            except Exception as exc:
                logger.debug("Failed to extract part content (%s): %s", content_type, exc)
                continue

        # Prefer plain text if available; otherwise use HTML
        if plain_parts:
            body = "\n\n".join(plain_parts)
        elif html_parts:
            body = "\n\n".join(html_parts)
    else:
        # Single-part message
        try:
            content = msg.get_content()
            body = content if isinstance(content, str) else str(content)
        except Exception as exc:
            logger.debug("Failed to extract single-part content: %s", exc)
            payload = msg.get_payload(decode=True)
            if payload and isinstance(payload, bytes):
                body = payload.decode("utf-8", errors="replace")
            else:
                body = str(payload or "")

    return subject.strip(), body.strip()
