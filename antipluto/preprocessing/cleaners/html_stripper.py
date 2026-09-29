"""
cleaners/html_stripper.py
=========================
Utility class for converting raw email body payloads into clean plain text.

``HTMLStripper`` detects HTML envelopes by inspecting the raw string for
common HTML markers and delegates parsing to ``BeautifulSoup4``, which is
both more robust and more memory-efficient than manual regex substitutions
for deeply nested HTML structures (e.g. multi-part phishing emails with
extensive CSS, JavaScript, and base64-encoded image attachments).

Post-processing normalises whitespace and collapses consecutive blank lines,
producing paragraph-separated prose that preserves the structural rhythm of
the original author — a critical requirement for downstream stylometric
feature extraction.

Usage
-----
::

    from antipluto.preprocessing.cleaners.html_stripper import HTMLStripper

    stripper = HTMLStripper()
    clean_text = stripper.strip(raw_body)
"""

from __future__ import annotations

import re
from bs4 import BeautifulSoup
from antipluto.preprocessing.cleaners.unicode_normalizer import normalize_unicode_text

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Markers whose presence strongly implies an HTML envelope.
_HTML_MARKERS: tuple[str, ...] = (
    "<html",
    "<body",
    "<div",
    "<table",
    "<span",
    "<p ",
    "<br",
    "<!doctype",
)

# Collapse 3+ consecutive blank lines into a single blank line.
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


# ---------------------------------------------------------------------------
# HTMLStripper
# ---------------------------------------------------------------------------


class HTMLStripper:
    """
    Converts raw email payload text into normalised plain text.

    The class is stateless; all methods are safe to call from multiple
    threads or processes simultaneously.

    Processing Pipeline
    -------------------
    1. **HTML detection** — Check for HTML markers in the raw string.
    2. **HTML parsing** — If HTML is detected, use ``BeautifulSoup4`` with
       the ``html.parser`` backend (no external binary dependency) to extract
       readable text, using ``\\n`` as the separator to preserve paragraph
       structure.
    3. **Whitespace normalisation** — Strip leading/trailing whitespace from
       each line, then collapse sequences of 3+ blank lines into one.
    4. **Final trim** — Strip leading and trailing whitespace from the result.
    """

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def strip(self, raw: str) -> str:
        """
        Parse and normalise ``raw`` into clean plain text.

        Parameters
        ----------
        raw : str
            The raw email payload string, which may or may not contain HTML.

        Returns
        -------
        str
            Normalised plain text.  Returns an empty string if ``raw`` is
            empty or reduces to whitespace after processing.
        """
        if not raw:
            return ""

        text = self._parse_html(raw) if self._is_html(raw) else raw
        return self._normalise(text)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_html(raw: str) -> bool:
        """
        Return True if the raw string contains discernible HTML markup.

        The check is deliberately permissive — a single recognised marker
        is sufficient to trigger full HTML parsing.  False positives are
        acceptable because BeautifulSoup handles plain text gracefully.
        """
        raw_lower = raw.lower()
        return any(marker in raw_lower for marker in _HTML_MARKERS) or "</" in raw

    @staticmethod
    def _parse_html(raw: str) -> str:
        """
        Extract readable text from an HTML string using BeautifulSoup4.

        Parameters
        ----------
        raw : str
            Raw HTML content.

        Returns
        -------
        str
            Plain text extracted from the HTML tree, with newline separators
            to maintain structural paragraph breaks.
        """
        soup = BeautifulSoup(raw, "html.parser")

        # Remove script and style nodes entirely — their text content is
        # never meaningful prose and frequently introduces noise tokens.
        for tag in soup(["script", "style", "head", "meta", "link"]):
            tag.decompose()

        return soup.get_text(separator="\n")

    @staticmethod
    def _normalise(text: str) -> str:
        """
        Normalise whitespace and Unicode characters in a plain-text string.

        Steps:
          1. Apply Unicode NFKC normalization, strip invisible control chars, map homoglyphs.
          2. Strip trailing whitespace from every line.
          3. Collapse 3+ consecutive blank lines into a single blank line.
          4. Strip leading and trailing whitespace from the result.

        Parameters
        ----------
        text : str
            Plain text (may include excessive blank lines or homoglyphs).

        Returns
        -------
        str
            Normalised text.
        """
        text = normalize_unicode_text(text)
        lines = [line.rstrip() for line in text.splitlines()]
        normalised = "\n".join(lines)
        normalised = _MULTI_BLANK_RE.sub("\n\n", normalised)
        return normalised.strip()
