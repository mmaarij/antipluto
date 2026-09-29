"""
cleaners/header_slicer.py
=========================
Thread boundary detection and body slicing for multi-hop email threads.

Overview
--------
Raw email bodies frequently contain embedded history: Outlook "Original
Message" blocks, Enron corporate "Forwarded by" banners, Gmail reply
attributions (``On <date>, <person> wrote:``), and manually-quoted lines
prefixed with ``>``.  Retaining this history would contaminate the stylistic
signal of the primary author — a critical data-quality issue for the
downstream stylometric classifier.

Rather than discarding the entire email when a thread marker is detected
(the approach of the original ``raw_dataset_preprocess.py``), ``HeaderSlicer``
locates the *earliest* boundary across a compiled library of regex patterns
and returns only the text *above* that boundary.  This recovers the primary
authored content while surgically removing the embedded history, increasing
the usable sample yield without introducing noise.

Pattern Library
---------------
The slicer targets the following boundary types:

1. **Enron corporate forward banner**
   ``----- Forwarded by <name> on <date> -----``

2. **Standard Outlook original-message divider**
   ``-----Original Message-----``

3. **"Begin forwarded message" block** (macOS Mail / Apple clients)

4. **Reply attribution line** (Gmail, modern Outlook)
   ``On <date>, <name> <email> wrote:``

5. **Quoted reply lines** — lines beginning with ``>`` (RFC 3676)

6. **Outlook underline divider** — 5 or more consecutive underscores

7. **Repeated From/To/Date header block** in body (Outlook inline forward)

8. **"Transcript of session follows"** (automated Enron mailer footers)

Usage
-----
::

    from email_preprocessor.cleaners.header_slicer import HeaderSlicer

    slicer = HeaderSlicer()
    primary_body = slicer.slice(raw_body)
"""

from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Compiled boundary patterns
# ---------------------------------------------------------------------------
#
# Each pattern targets a distinct class of thread-boundary marker.
# All patterns are pre-compiled with appropriate flags for efficiency —
# these objects are created once at module import time and reused across
# all worker processes (each spawned process re-imports the module).
#
# Ordering has no effect on correctness because the slicer finds the
# *minimum* match position across all patterns.

_BOUNDARY_PATTERNS: list[re.Pattern] = [

    # ------------------------------------------------------------------ #
    # 1. Enron corporate forward banner                                   #
    #    Example: "----- Forwarded by John Doe/Enron on 01/01/2001 -----" #
    # ------------------------------------------------------------------ #
    re.compile(r"-{3,}\s*Forwarded by\b", re.IGNORECASE),

    # ------------------------------------------------------------------ #
    # 2. Standard Outlook original-message divider                        #
    #    Example: "-----Original Message-----"                            #
    #    Also catches "--- Original Message ---" and similar variants.    #
    # ------------------------------------------------------------------ #
    re.compile(r"-{3,}\s*Original\s+Message(?:\s+Follows)?\s*-{0,}", re.IGNORECASE),

    # ------------------------------------------------------------------ #
    # 3. "Begin forwarded message" block (Apple Mail / macOS)             #
    #    Example: "---------- Forwarded message ----------"               #
    # ------------------------------------------------------------------ #
    re.compile(r"-{3,}\s*(?:Begin|Start)\s+(?:of\s+)?Forwarded\s+Message\s*-{0,}",
               re.IGNORECASE),
    # Catches "-- Forwarded Message --" (2-dash, Apple Mail / mobile MUAs)
    # and "---------- Forwarded message ----------" (10-dash variant).
    re.compile(r"-{2,}\s*Forwarded\s+(?:Message|mail|by)\b", re.IGNORECASE),

    # ------------------------------------------------------------------ #
    # 4. Reply attribution line (Gmail / modern Outlook)                  #
    #    Example: "On Mon, Jan 1, 2001 at 12:00 PM, John Doe wrote:"     #
    #    Requires at least 10 chars between "On" and "wrote:" to avoid   #
    #    false-positives on short body sentences.                         #
    # ------------------------------------------------------------------ #
    re.compile(r"^On\s+.{10,250}?\s+wrote\s*:\s*$", re.MULTILINE | re.DOTALL),

    # ------------------------------------------------------------------ #
    # 5. Quoted reply lines (RFC 3676)                                    #
    #    Matches the first line in the body that begins with one or more  #
    #    ">" quote markers, as inserted by virtually all MUAs.            #
    # ------------------------------------------------------------------ #
    re.compile(r"^\s*>+\s", re.MULTILINE),

    # ------------------------------------------------------------------ #
    # 6. Outlook underline divider                                        #
    #    Example: "________________________________________"              #
    # ------------------------------------------------------------------ #
    re.compile(r"_{5,}"),

    # ------------------------------------------------------------------ #
    # 7. Repeated From/To/Date header block in body                       #
    #    Outlook inline-forwarded messages repeat email headers as plain  #
    #    text in the body. Detect the "From:" line of such a block.       #
    #    The pattern requires a preceding newline to avoid matching the   #
    #    original message's RFC-2822 "From:" header that may bleed        #
    #    through in plaintext repro.                                      #
    # ------------------------------------------------------------------ #
    re.compile(r"\n\s*From\s*:\s*\S+@\S+", re.IGNORECASE),
    re.compile(r"\n\s*From\s*:\s*[A-Z][a-zA-Z\s,]+\n\s*(?:Sent|To|Subject)\s*:", re.IGNORECASE),

    # ------------------------------------------------------------------ #
    # 8. Enron automated mailer footers                                   #
    #    "Transcript of session follows" appears in automated             #
    #    confirmations and FYI-forward templates.                         #
    # ------------------------------------------------------------------ #
    re.compile(r"transcript of session follows", re.IGNORECASE),

    # ------------------------------------------------------------------ #
    # 9. "To:" block appearing mid-body (Lotus Notes forward style)       #
    # ------------------------------------------------------------------ #
    re.compile(r"\n\s*To\s*:\s*.+\n\s*(?:cc|subject|from)\s*:", re.IGNORECASE),

    # ------------------------------------------------------------------ #
    # 10. "Original Message Excluded:" footer                             #
    #     Produced by Lotus Notes, Ameritrade mailer, and similar         #
    #     systems that strip the quoted original but leave a marker line. #
    #     Example: "Original Message Excluded:\n--------------------------"#
    # ------------------------------------------------------------------ #
    re.compile(r"Original\s+Message\s+Excluded\s*:", re.IGNORECASE),

    # ------------------------------------------------------------------ #
    # 11. NDR / bounce "possibly truncated original message follows"      #
    #     Delivery failure notifications include the original message     #
    #     headers below this line.  The word "Possibly" is optional.      #
    #     Example: "Possibly truncated original message follows:"         #
    # ------------------------------------------------------------------ #
    re.compile(r"(?:Possibly\s+)?[Tt]runcated\s+original\s+message\s+follows", re.IGNORECASE),

    # ------------------------------------------------------------------ #
    # 12. Inline attachment / separator dividers                          #
    #     Example: "--------- Inline attachment follows ---------"        #
    # ------------------------------------------------------------------ #
    re.compile(r"-{3,}\s*Inline\s+attachment\s+follows\s*-{3,}", re.IGNORECASE),

    # ------------------------------------------------------------------ #
    # 13. "Original Message Follows:" — Lotus Notes / helpdesk mailers   #
    #     Sibling of pattern 10 ("Excluded").  Both leave a marker line  #
    #     before the quoted original text.                                #
    #     Example: "Original Message Follows:\n------------------------"  #
    # ------------------------------------------------------------------ #
    re.compile(r"Original\s+Message\s+Follows\s*:", re.IGNORECASE),

    # ------------------------------------------------------------------ #
    # 14. "Forwarded-by:" / "Forwarded by:" routing headers              #
    #     Listserv and email-forwarding systems prepend one or more       #
    #     "Forwarded-by: Name <addr>" lines at the TOP of the payload.   #
    #     These are routing metadata, not primary authored content.       #
    #     Example: "Forwarded-by: Rob Windsor <windsor@warthog.com>"      #
    # ------------------------------------------------------------------ #
    re.compile(r"^Forwarded(?:-by|_by|\s+by)\s*:", re.MULTILINE | re.IGNORECASE),

    # ------------------------------------------------------------------ #
    # 15. Lotus Notes reply attribution "Name|Org wrote on M/D/YY HH:MM" #
    #     The Enron / Lotus Notes MUA generates attribution lines of the  #
    #     form "First Last|DEPT|ECT wrote on 7/24/00 8:34 am:" without   #
    #     the leading "On " prefix used by Gmail/Outlook (pattern 4).    #
    #     Example: "Chip Cox|PDX|ECT wrote on 7/24/00 8:34 am:"          #
    # ------------------------------------------------------------------ #
    re.compile(r"^\S.{0,120}\s+wrote\s+on\s+\d{1,2}/\d{1,2}", re.MULTILINE),

    # ------------------------------------------------------------------ #
    # 16. NDR "The following addresses had delivery problems" banner      #
    #     Mail-server bounce notices include this dashed-section header   #
    #     before the list of failed recipient addresses.                  #
    #     Example: "   ----- The following addresses had delivery         #
    #               problems -----"                                       #
    # ------------------------------------------------------------------ #
    re.compile(
        r"-{3,}\s*The following addresses\s+had\s+(?:delivery|permanent|fatal)",
        re.IGNORECASE,
    ),

    # ------------------------------------------------------------------ #
    # 17. "*** ATTENTION ***" NDR / postmaster separator                  #
    #     Some MTA bounce templates use this attention banner to separate  #
    #     the postmaster explanation from the returned headers.            #
    # ------------------------------------------------------------------ #
    re.compile(r"\*{3,}\s*ATTENTION\s*\*{3,}", re.IGNORECASE),
]



# ---------------------------------------------------------------------------
# HeaderSlicer
# ---------------------------------------------------------------------------


class HeaderSlicer:
    """
    Slices email body text at the first detected thread boundary.

    The slicer is stateless — all methods may be called concurrently
    from multiple threads or processes.

    Algorithm
    ---------
    For a given body string, the slicer:

    1. Tests each compiled boundary pattern against the full body using
       ``re.search``.
    2. Collects the ``match.start()`` position for every hit.
    3. Slices the body at the *minimum* (earliest) hit position, returning
       only the text above that boundary.
    4. Falls back to returning the full body unchanged if no boundary is
       detected (safe default — no data is silently discarded).

    The final result is stripped of leading/trailing whitespace.
    """

    def slice(self, body: str) -> str:
        """
        Return the primary authored content of an email body.

        Parameters
        ----------
        body : str
            Full email body text, potentially containing embedded thread
            history below a boundary marker.

        Returns
        -------
        str
            The text above the first detected boundary, or the full body
            if no boundary is found.
        """
        if not body:
            return ""

        boundary_pos: Optional[int] = self._find_earliest_boundary(body)

        if boundary_pos is None:
            # No thread boundary detected — return the full body as-is.
            return body.strip()

        if boundary_pos == 0:
            # The boundary is at the very start of the body, meaning there is
            # no primary authored content above it — the entire payload is
            # thread history (e.g. a bare "FW:" forward with no added comment).
            # Return an empty string so the caller discards this record rather
            # than writing pure thread noise to the output.
            return ""

        sliced = body[:boundary_pos].strip()
        return sliced

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_earliest_boundary(body: str) -> Optional[int]:
        """
        Scan the body for all boundary patterns and return the earliest hit.

        Parameters
        ----------
        body : str
            Full email body.

        Returns
        -------
        int or None
            The character position of the earliest boundary, or ``None``
            if no boundary pattern matched.
        """
        earliest: Optional[int] = None

        for pattern in _BOUNDARY_PATTERNS:
            match = pattern.search(body)
            if match:
                pos = match.start()
                if earliest is None or pos < earliest:
                    earliest = pos

        return earliest
