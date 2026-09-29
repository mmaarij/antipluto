"""
cleaners/feature_extractors.py
===============================
Feature extraction utilities for MeAJOR Table II parity.

Extracts structured metadata, network tokens, URL metrics, attachment stats,
and language identification from parsed email messages.
"""

from __future__ import annotations

import email.utils
import hashlib
import re
from urllib.parse import urlparse
from typing import Sequence

from bs4 import BeautifulSoup

try:
    from langdetect import detect as _lang_detect, DetectorFactory
    DetectorFactory.seed = 42
    _HAS_LANGDETECT = True
except ImportError:
    _HAS_LANGDETECT = False

# Regex for extracting URLs from plain text
_URL_REGEX = re.compile(
    r'https?://[^\s<>"]+|www\.[^\s<>"]+',
    re.IGNORECASE,
)


def hash_email_address(raw_header: str | None) -> tuple[str, str]:
    """
    Extract email address and domain from an email header (e.g. From/To),
    returning (sha256_hash, domain).

    Parameters
    ----------
    raw_header : str or None
        Raw header string (e.g., 'Phillip Allen <pallen@enron.com>').

    Returns
    -------
    tuple[str, str]
        (sha256_hex_hash, domain)
    """
    if not raw_header:
        return "", ""

    _, addr = email.utils.parseaddr(str(raw_header))
    addr = addr.strip().lower()
    if not addr or "@" not in addr:
        return "", ""

    domain = addr.split("@")[-1].strip()
    hashed_addr = hashlib.sha256(addr.encode("utf-8")).hexdigest()
    return hashed_addr, domain


def extract_urls_from_payloads(raw_html: str, raw_text: str) -> list[str]:
    """
    Extract unique URLs from both HTML markup (href attributes) and plain text.

    Parameters
    ----------
    raw_html : str
        Raw HTML payload text.
    raw_text : str
        Raw plain-text payload.

    Returns
    -------
    list[str]
        Deduplicated list of extracted URL strings.
    """
    found_urls: set[str] = set()

    if raw_html:
        try:
            soup = BeautifulSoup(raw_html, "html.parser")
            for a_tag in soup.find_all("a", href=True):
                href = str(a_tag["href"]).strip()
                if href.startswith(("http://", "https://", "www.")):
                    found_urls.add(href)
        except Exception:
            pass

    # Extract via regex from HTML + plain text
    combined = (raw_html or "") + "\n" + (raw_text or "")
    for match in _URL_REGEX.findall(combined):
        url_str = match.strip().rstrip(".,;)>'\"]")
        if url_str:
            found_urls.add(url_str)

    return sorted(found_urls)


def count_subdomains(url: str) -> int:
    """
    Count the number of subdomain levels in a URL.

    Example:
        'http://sub.domain.co.uk/path' -> hostname 'sub.domain.co.uk' -> 1 subdomain
        'http://example.com' -> hostname 'example.com' -> 0 subdomains

    Parameters
    ----------
    url : str
        URL string.

    Returns
    -------
    int
        Subdomain depth.
    """
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        if not hostname:
            return 0
        # Remove IPv4/IPv6 literals
        if hostname.replace(".", "").isdigit():
            return 0
        parts = hostname.split(".")
        # Standard domain has 2 parts (e.g. example.com). Subdomains are extra parts.
        if len(parts) <= 2:
            return 0
        return len(parts) - 2
    except Exception:
        return 0


def compute_url_metrics(urls: Sequence[str]) -> dict[str, int | float]:
    """
    Compute statistical summary metrics over a collection of URLs.

    Returns keys:
        - url_count (int)
        - url_length_max (int)
        - url_length_avg (float)
        - url_subdom_max (int)
        - url_subdom_avg (float)
    """
    if not urls:
        return {
            "url_count": 0,
            "url_length_max": 0,
            "url_length_avg": 0.0,
            "url_subdom_max": 0,
            "url_subdom_avg": 0.0,
        }

    lengths = [len(u) for u in urls]
    subdoms = [count_subdomains(u) for u in urls]
    n = len(urls)

    return {
        "url_count": n,
        "url_length_max": max(lengths),
        "url_length_avg": round(sum(lengths) / n, 2),
        "url_subdom_max": max(subdoms),
        "url_subdom_avg": round(sum(subdoms) / n, 2),
    }


def detect_language(text: str) -> str:
    """
    Detect the primary language code of the text body.

    Parameters
    ----------
    text : str
        Cleaned body text.

    Returns
    -------
    str
        Language code string (e.g., 'en', 'es', 'unknown').
    """
    if not text or len(text.strip()) < 15:
        return "unknown"

    if _HAS_LANGDETECT:
        try:
            return str(_lang_detect(text[:1000]))
        except Exception:
            return "en"

    return "en"
