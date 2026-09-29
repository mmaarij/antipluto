"""
antipluto.api.pipeline
====================
Real-world email ingestion and preprocessing pipeline chaining:
1. HTML stripping (HTMLStripper)
2. Thread boundary slicing (HeaderSlicer)
3. Unicode normalization, zero-width stripping, and homoglyph mapping (normalize_unicode_text)
4. URL and entity token masking (PIIMasker)
5. Model text formatting (format_email_text)
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import List, Tuple

from antipluto.classifier.data import format_email_text
from antipluto.masking.masker import PIIMasker
from antipluto.preprocessing.cleaners.html_stripper import HTMLStripper
from antipluto.preprocessing.cleaners.header_slicer import HeaderSlicer
from antipluto.preprocessing.cleaners.unicode_normalizer import normalize_unicode_text
from antipluto.preprocessing.cleaners.feature_extractors import extract_urls_from_payloads

logger = logging.getLogger(__name__)


@dataclass
class PreprocessedEmail:
    raw_subject: str
    raw_body: str
    clean_subject: str
    clean_body: str
    masked_subject: str
    masked_body: str
    formatted_text: str
    detected_urls: List[str]
    html_stripped: bool
    thread_history_sliced: bool


class EmailIngestionPipeline:
    """Stateless or cached preprocessor chaining all cleaning stages."""

    def __init__(self, masker: PIIMasker | None = None) -> None:
        self.stripper = HTMLStripper()
        self.slicer = HeaderSlicer()
        self.masker = masker if masker is not None else PIIMasker(enabled=True)

    def process(self, subject: str, body: str) -> PreprocessedEmail:
        """Run raw email subject and body through the complete 5-stage cleaning pipeline.

        Parameters
        ----------
        subject : str
            Raw email subject line.
        body : str
            Raw email body (plain text, HTML, or mixed).

        Returns
        -------
        PreprocessedEmail
            Container with intermediate and final masked/formatted representations
            and extraction telemetry.
        """
        raw_subj = subject or ""
        raw_bdy = body or ""

        # Stage 1: HTML Detection and Stripping
        is_html = self.stripper._is_html(raw_bdy)
        clean_bdy = self.stripper.strip(raw_bdy) if is_html else raw_bdy

        # Extract URLs before slicing/masking so metadata reflects all payload links
        urls = extract_urls_from_payloads(raw_html=raw_bdy if is_html else "", raw_text=clean_bdy)

        # Stage 2: Thread Boundary Slicing (strip embedded reply/forward history)
        sliced_bdy = self.slicer.slice(clean_bdy)
        thread_sliced = len(sliced_bdy) < len(clean_bdy)

        # Stage 3: Unicode Normalization (NFKC, zero-width strip, homoglyph mapping)
        norm_subj = normalize_unicode_text(raw_subj)
        norm_bdy = normalize_unicode_text(sliced_bdy)

        # Stage 4: PII and URL Token Masking
        masked_subj = self.masker.mask(norm_subj) if norm_subj else ""
        masked_bdy = self.masker.mask(norm_bdy) if norm_bdy else ""

        # Stage 5: Unified Formatting
        formatted = format_email_text(masked_subj, masked_bdy)

        return PreprocessedEmail(
            raw_subject=raw_subj,
            raw_body=raw_bdy,
            clean_subject=norm_subj,
            clean_body=norm_bdy,
            masked_subject=masked_subj,
            masked_body=masked_bdy,
            formatted_text=formatted,
            detected_urls=urls,
            html_stripped=is_html,
            thread_history_sliced=thread_sliced,
        )
