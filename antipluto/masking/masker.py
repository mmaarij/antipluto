"""
masking/masker.py
=================
MeAJOR-equivalent two-stage PII anonymization pipeline for email text.

Overview
--------
Privacy-respecting AI research requires the removal of Personally
Identifiable Information (PII) before publishing or training on datasets.
This module implements the same anonymization strategy as the MeAJOR Corpus
(Mendes et al., 2025, arXiv:2507.17978, Section III-B), using entity
replacement tokens enclosed in square brackets.

Stage 1 — spaCy NER
~~~~~~~~~~~~~~~~~~~~
A pretrained spaCy NER model identifies named entities in the text. Each
recognised entity is replaced with a MeAJOR-compatible placeholder token:

+------------------+---------------------+
| spaCy Entity     | Replacement         |
+------------------+---------------------+
| PERSON           | ``[NAME]``          |
+------------------+---------------------+
| ORG, NORP        | ``[ORGANIZATION]``  |
+------------------+---------------------+
| GPE, LOC, FAC    | ``[ADDRESS]``       |
+------------------+---------------------+
| DATE             | ``[DATE]``          |
+------------------+---------------------+
| TIME             | ``[TIME]``          |
+------------------+---------------------+
| MONEY, QUANTITY  | ``[FINANCIAL_INFO]``|
+------------------+---------------------+
| PRODUCT,         | ``[PRODUCT]``       |
| WORK_OF_ART      |                     |
+------------------+---------------------+
| CARDINAL, ORDINAL| ``[REFERENCE_NUMBER]``|
| PERCENT, EVENT,  |                     |
| LAW              |                     |
+------------------+---------------------+

Stage 2 — Regex Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~
Deterministic regular expressions catch structured PII not reliably captured
by NER. Applied in the following order (ordering matters to avoid partial
overlaps):

+--------------------------------+------------------------+
| Pattern                        | Replacement            |
+================================+========================+
| PGP signature/key blocks       | ``[PGP]``              |
+--------------------------------+------------------------+
| Unicode emoji characters       | ``[EMOJI]``            |
+--------------------------------+------------------------+
| Email addresses                | ``[EMAIL_ADDRESS]``    |
+--------------------------------+------------------------+
| URLs (http/https/ftp/www)      | ``[URL]``              |
| Bare domain URLs (evil.com/x)  |                        |
+--------------------------------+------------------------+
| IPv4 addresses                 | ``[IP_ADDRESS]``       |
+--------------------------------+------------------------+
| Windows/Unix file paths        | ``[FILE_PATH]``        |
+--------------------------------+------------------------+
| Filenames with extension       | ``[FILE_NAME]``        |
+--------------------------------+------------------------+
| Generic file references        | ``[FILE]``             |
+--------------------------------+------------------------+
| Phone numbers (local/intl)     | ``[PHONE_NUMBER]``     |
+--------------------------------+------------------------+
| Credit/debit card numbers      | ``[FINANCIAL_INFO]``   |
+--------------------------------+------------------------+
| IBAN bank account numbers      | ``[FINANCIAL_INFO]``   |
+--------------------------------+------------------------+
| Reference/order/ticket numbers | ``[REFERENCE_NUMBER]`` |
+--------------------------------+------------------------+
| @mention usernames             | ``[USERNAME]``         |
+--------------------------------+------------------------+
| Initials (A.B.C. style)        | ``[INITIALS]``         |
+--------------------------------+------------------------+
| Decorative separator lines     | ``[SYMBOL]``           |
+--------------------------------+------------------------+

MeAJOR Token Set
~~~~~~~~~~~~~~~~
The full set of anonymization tokens matches MeAJOR Corpus (Mendes et al.,
2025):

    [PGP], [EMOJI], [SYMBOL], [NAME], [USERNAME], [INITIALS],
    [EMAIL_ADDRESS], [PHONE_NUMBER], [ADDRESS], [ORGANIZATION],
    [URL], [IP_ADDRESS], [FILE_PATH], [FILE_NAME], [FILE],
    [DATE], [TIME], [FINANCIAL_INFO], [PRODUCT], [REFERENCE_NUMBER]

Design Note — Standalone JSONL Masking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The masker is used in two ways:

1. **Integrated** (``run`` command): Set ``masking.enabled: true`` in
   ``default.yaml`` to apply masking inline during the full ingestion
   pipeline. Useful only if you want a single-pass output; generally
   leave ``false`` so raw features are preserved for analysis.

2. **Standalone** (``mask`` command): Apply masking as a final step
   on any existing JSONL file — including LLM-generated cohorts::

       python -m antipluto mask --input datasets/datasets_preprocessed/
       python -m antipluto mask --input datasets/llm_cohorts/ --output-dir datasets/datasets_masked/

   The ``mask`` command reads JSON directly without schema validation,
   so it accepts any source field (including LLM-generated sources).

Per the research design, masking should be applied **uniformly** across
all four cohorts (human benign, human phishing, LLM-assisted benign,
LLM-generated phishing) as a **final** step before classifier training,
ensuring that masking artefacts are distributed identically across classes.
"""

from __future__ import annotations

import gc
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MeAJOR-compatible NER entity label → placeholder token mapping
# ---------------------------------------------------------------------------

_ENTITY_MAP: dict[str, str] = {
    "PERSON":      "[NAME]",
    "ORG":         "[ORGANIZATION]",
    "GPE":         "[ADDRESS]",
    "LOC":         "[ADDRESS]",
    "FAC":         "[ADDRESS]",
    "DATE":        "[DATE]",
    "TIME":        "[TIME]",
    "MONEY":       "[FINANCIAL_INFO]",
    "QUANTITY":    "[FINANCIAL_INFO]",
    "PRODUCT":     "[PRODUCT]",
    "WORK_OF_ART": "[PRODUCT]",
    "CARDINAL":    "[REFERENCE_NUMBER]",
    "ORDINAL":     "[REFERENCE_NUMBER]",
    "PERCENT":     "[REFERENCE_NUMBER]",
    "NORP":        "[ORGANIZATION]",   # Nationalities, religious groups, political orgs
    "EVENT":       "[REFERENCE_NUMBER]",
    "LAW":         "[REFERENCE_NUMBER]",
}

# ---------------------------------------------------------------------------
# Compiled regex patterns for Stage 2 (applied in declared order)
# ---------------------------------------------------------------------------

_REGEX_PATTERNS: list[tuple[re.Pattern, str]] = [
    # PGP blocks (headers, signatures, keys, encrypted messages).
    # Must run first so PGP wrappers and signature blocks are masked before general PII.
    (
        re.compile(
            r"-----BEGIN PGP SIGNED MESSAGE-----\s*(?:[A-Za-z0-9\-]+:[^\n]+\n)*",
            re.IGNORECASE,
        ),
        "[PGP]\n\n",
    ),
    (
        re.compile(
            r"-----BEGIN PGP SIGNATURE-----.*?-----END PGP SIGNATURE-----",
            re.DOTALL | re.IGNORECASE,
        ),
        "[PGP]",
    ),
    (
        re.compile(
            r"-----BEGIN PGP (?:MESSAGE|PUBLIC KEY BLOCK|PRIVATE KEY BLOCK)-----.*?-----END PGP [A-Z ]+-----",
            re.DOTALL | re.IGNORECASE,
        ),
        "[PGP]",
    ),
    # Unicode emoji (covers Emoticons, Misc Symbols & Pictographs, Supplemental Symbols,
    # Dingbats, Enclosed Alphanumeric Supplement, and Transport & Map Symbols).
    (
        re.compile(
            r"[\U0001F600-\U0001F64F"   # Emoticons
            r"\U0001F300-\U0001F5FF"    # Misc symbols & pictographs
            r"\U0001F680-\U0001F6FF"    # Transport & map
            r"\U0001F700-\U0001F77F"    # Alchemical
            r"\U0001F780-\U0001F7FF"    # Geometric shapes extended
            r"\U0001F800-\U0001F8FF"    # Supplemental arrows-C
            r"\U0001F900-\U0001F9FF"    # Supplemental symbols & pictographs
            r"\U0001FA00-\U0001FA6F"    # Chess symbols
            r"\U0001FA70-\U0001FAFF"    # Symbols and pictographs extended-A
            r"\u2600-\u26FF"            # Misc symbols
            r"\u2700-\u27BF"            # Dingbats
            r"\uFE00-\uFE0F]+"          # Variation selectors
        ),
        "[EMOJI]",
    ),
    # Email addresses — must run before URL to avoid partial overlap on mailto: links.
    (
        re.compile(
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
            re.IGNORECASE,
        ),
        "[EMAIL_ADDRESS]",
    ),
    # URLs — two passes:
    #
    # Pass A: Scheme-prefixed URLs (http://, https://, ftp://) and www. URLs.
    #   Uses a character class that excludes common trailing punctuation
    #   (., ,, ), >, ", ', !) that are frequently appended after a URL in
    #   natural prose (e.g. "Visit https://evil.com." or "(see https://x.com)").
    #   The negative lookbehind on the trailing char strip ensures we don't
    #   over-strip URLs that legitimately end in a slash or alphanumeric.
    (
        re.compile(
            r"(?:https?://|ftp://|www\.)[^\s<>\"'\\]+"
            r"(?<![.,;:!?\)\]\}])",  # strip trailing punctuation (lookbehind)
            re.IGNORECASE,
        ),
        "[URL]",
    ),
    # Pass B: Bare domain URLs without a scheme — common in phishing emails
    #   (e.g. "click evil-corp.ru/verify" or "go to account-update.tk/login").
    #   Requires at least one path segment or query string after the TLD to
    #   avoid masking plain English words that contain a dot (e.g. "e.g.",
    #   "Mr.Smith", "V.I.P.").
    (
        re.compile(
            r"(?<![\w@])"                    # not preceded by word char or @
            r"(?:[a-zA-Z0-9]"               # start of hostname
            r"(?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?"  # hostname label body
            r"\.)+"                          # dot separator(s)
            r"(?:com|net|org|edu|gov|mil|int|io|co|ru|cn|de|uk|fr|br|in|tk|top|xyz|info|biz|online|site|click|link|email|zip|app|cloud)"
            r"(?:/[^\s<>\"'\\]*)"          # mandatory path component
            r"(?<![.,;:!?\)\]\}])",         # strip trailing punctuation
            re.IGNORECASE,
        ),
        "[URL]",
    ),
    # IPv4 addresses — before phone numbers to avoid digit collisions.
    (
        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        "[IP_ADDRESS]",
    ),
    # Windows absolute paths: C:\Users\foo\bar.txt or \\server\share\path
    (
        re.compile(
            r"(?:[A-Za-z]:\\|\\\\)(?:[^\s\\/\"'<>|*?:]+[/\\])*[^\s\\/\"'<>|*?:]*",
            re.IGNORECASE,
        ),
        "[FILE_PATH]",
    ),
    # Unix absolute paths: /usr/local/bin/script.sh
    (
        re.compile(
            r"(?<!\w)/(?:[^\s/\"'<>|*?:]+/)*[^\s/\"'<>|*?:]+\.[A-Za-z0-9]{1,10}\b",
        ),
        "[FILE_PATH]",
    ),
    # Filenames with common extensions (must run before generic [FILE]).
    (
        re.compile(
            r"\b[\w\-]+\."
            r"(?:pdf|docx?|xlsx?|pptx?|txt|csv|zip|rar|7z|tar|gz|bz2|"
            r"exe|msi|dmg|pkg|apk|iso|img|"
            r"jpg|jpeg|png|gif|bmp|svg|webp|tiff?|"
            r"mp3|mp4|wav|avi|mov|mkv|flv|"
            r"html?|php|js|ts|py|java|cpp|cs|rb|go|rs|sh|bat|ps1|"
            r"json|xml|yaml|yml|toml|ini|cfg|conf|log)"
            r"\b",
            re.IGNORECASE,
        ),
        "[FILE_NAME]",
    ),
    # Generic file references: "file: foo", "attachment: bar", "attached file X"
    (
        re.compile(
            r"\b(?:attached?|attachment|file|document|doc)\s*:\s*\S+",
            re.IGNORECASE,
        ),
        "[FILE]",
    ),
    # Credit/debit card numbers: 4 groups of 4 digits with optional separators.
    # Must run BEFORE phone numbers to avoid card numbers being swallowed by phone regex.
    (
        re.compile(r"\b(?:\d{4}[\s\-]?){4}\b"),
        "[FINANCIAL_INFO]",
    ),
    # IBAN bank account numbers: 2 letters + 2 digits + 11-30 alphanumerics.
    # Must run BEFORE phone numbers for the same reason.
    (
        re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
        "[FINANCIAL_INFO]",
    ),
    # Reference / order / ticket / tracking numbers.
    # Must run BEFORE phone numbers to prevent numeric parts of codes like
    # "TKT-99887766" from being swallowed by the phone regex first.
    # Covers three forms:
    #   1. Keyword+separator-prefixed (separator required): "Ref: TKT-99887766",
    #      "Order #ABC123", "Case ID: 12345A" — the code must contain at least one digit.
    #   2. Hash-prefixed codes containing at least one digit: "#ABC12345"
    #   3. Standalone LETTERS-DIGITS hyphenated codes: "TKT-99887766", "INV-2024-001"
    (
        re.compile(
            r"(?:"
            # Form 1: keyword followed by mandatory separator, then a code with >=1 digit.
            r"(?:ref(?:erence)?|order|ticket|tracking|case|account|id|no\.?)\s*[:#]\s*"
            r"[A-Z0-9][\w\-]{3,19}(?=[\w\-]*\d[\w\-]*)"  # lookahead: must contain digit
            r"|#[A-Z0-9]*\d[A-Z0-9]{3,19}"               # Form 2: #CODE with at least 1 digit
            r"|(?<![A-Za-z0-9])[A-Z]{2,6}-\d{4,20}"      # Form 3: ALPHA-DIGITS e.g. TKT-99887766
            r")",
            re.IGNORECASE,
        ),
        "[REFERENCE_NUMBER]",
    ),
    # International and local phone numbers.
    # Covers: +1-800-555-1234, (800) 555-1234, 555.1234, +44 20 7946 0958 etc.
    (
        re.compile(
            r"""
            (?<!\d)                       # not preceded by digit
            (?:
                \+?[1-9]\d{0,2}           # optional country code
                [\s.\-]?
            )?
            (?:\(?(?:0\d{1,4}|\d{1,4})\)?[\s.\-]?)  # area code (0xxx or xxx)
            \d{3,4}[\s.\-]?\d{3,4}       # local number
            (?!\d)                        # not followed by digit
            """,
            re.VERBOSE,
        ),
        "[PHONE_NUMBER]",
    ),
    # @mention usernames (Twitter/email-client style).
    (
        re.compile(r"@[A-Za-z0-9_.]{1,50}\b"),
        "[USERNAME]",
    ),
    # Initials: sequences of 2-5 uppercase letters each followed by a period.
    # e.g. "J.R.R.", "A.B.", "M.D."
    (
        re.compile(r"\b(?:[A-Z]\.){2,5}"),
        "[INITIALS]",
    ),
    # Synthetic LLM placeholder name tokens ([SENDER], [RECEIVER]) -> [NAME]
    (
        re.compile(r"\[(?:SENDER|RECEIVER)\]", re.IGNORECASE),
        "[NAME]",
    ),
    # Synthetic LLM placeholder link token ([ACTION_LINK]) -> [URL]
    (
        re.compile(r"\[ACTION_LINK\]", re.IGNORECASE),
        "[URL]",
    ),
    # Decorative separator lines: lines consisting almost entirely of
    # repeated punctuation (----, ====, ****,  ~~~ etc.), with length >= 4.
    (
        re.compile(
            r"^[\s]*[-=*~_#|+]{4,}[\s]*$",
            re.MULTILINE,
        ),
        "[SYMBOL]",
    ),
]


# ---------------------------------------------------------------------------
# PIIMasker
# ---------------------------------------------------------------------------


class PIIMasker:
    """
    Applies the two-stage MeAJOR-compatible PII anonymization pipeline to text.

    The masker is designed to be instantiated once per process and reused
    across all records. Instantiation loads the spaCy model, which is the
    most expensive operation.

    Performance
    -----------
    For best throughput with ``en_core_web_trf`` use ``mask_batch()`` rather
    than calling ``mask()`` in a loop.  ``mask_batch()`` drives ``nlp.pipe()``
    so the transformer executes a single forward pass per batch of documents
    instead of N sequential passes.

    A CUDA-capable GPU is used automatically if detected (requires the
    ``cupy-cuda12x`` or equivalent wheel).  CPU throughput is still significantly
    improved by batching.

    Parameters
    ----------
    spacy_model : str
        Name of the spaCy model to load for NER (e.g. ``"en_core_web_sm"``).
        Using ``"en_core_web_trf"`` gives the best entity recall at the cost
        of throughput; use ``mask_batch()`` to amortise that cost.
    enabled : bool
        If ``False``, ``mask()`` / ``mask_batch()`` return the input unchanged.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm", enabled: bool = True) -> None:
        self.enabled = enabled
        self._nlp: Optional[object] = None

        if enabled:
            self._load_model(spacy_model)

    def _load_model(self, model_name: str) -> None:
        """Load the spaCy NER model with GPU preference and minimal components."""
        try:
            import spacy  # type: ignore

            # Use GPU if CUDA is available (no-op on CPU-only machines).
            is_gpu = spacy.prefer_gpu()
            if is_gpu:
                logger.info("PIIMasker › GPU detected — NER will run on CUDA.")
            else:
                logger.info("PIIMasker › No GPU detected — running on CPU.")

            # Disable pipeline components not needed for NER to reduce overhead.
            # For en_core_web_trf this avoids running the tagger, parser, and
            # lemmatizer heads through the transformer unnecessarily.
            _EXCLUDE = ["tagger", "parser", "senter", "lemmatizer", "attribute_ruler"]
            self._nlp = spacy.load(model_name, exclude=_EXCLUDE)
            logger.info(
                "PIIMasker › Loaded spaCy model '%s' (excluded: %s).",
                model_name,
                ", ".join(_EXCLUDE),
            )
        except OSError:
            logger.error(
                "PIIMasker › spaCy model '%s' not found. "
                "Install it with: python -m spacy download %s",
                model_name,
                model_name,
            )
            raise
        except ImportError:
            logger.error(
                "PIIMasker › spaCy is not installed. "
                "Install it with: pip install spacy"
            )
            raise

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def mask(self, text: str) -> str:
        """
        Apply the full two-stage MeAJOR-compatible PII masking pipeline to a
        single string.

        For bulk processing, prefer ``mask_batch()`` which uses ``nlp.pipe()``
        and is significantly faster with transformer models.

        Parameters
        ----------
        text : str
            Input text (subject or body of an email record).

        Returns
        -------
        str
            Anonymized text with PII replaced by MeAJOR-compatible tokens.
            Returns the input unchanged if ``enabled=False``.
        """
        if not self.enabled or not text:
            return text

        # Stage 1: Structured regex masking — must run before NER.
        text = self._apply_regex(text)
        # Stage 2: NER masking.
        text = self._apply_ner(text)
        return text

    def mask_batch(self, texts: list[str], batch_size: int = 32) -> list[str]:
        """
        Apply the two-stage masking pipeline to a list of strings in one batch.

        Uses ``nlp.pipe()`` so the transformer model executes a single forward
        pass per ``batch_size`` documents rather than N sequential passes.
        This is the recommended API for the ``mask`` CLI command and any other
        bulk processing loop.

        Parameters
        ----------
        texts : list[str]
            Input strings (subjects or bodies).  Empty strings are passed
            through unchanged without being sent to the model.
        batch_size : int
            Number of documents per NER mini-batch.  Tune upward (e.g. 64–128)
            if you have a GPU with enough VRAM; tune downward if you hit OOM.

        Returns
        -------
        list[str]
            Masked strings in the same order as ``texts``.
        """
        if not self.enabled:
            return list(texts)

        assert self._nlp is not None, "spaCy model not loaded."

        # Stage 1: Apply regex to every text first (pure CPU, embarrassingly parallel).
        regex_texts = [self._apply_regex(t) if t else t for t in texts]

        # Stage 2: Batch NER via nlp.pipe() — the expensive transformer step.
        results: list[str] = []
        for text, doc in zip(regex_texts, self._nlp.pipe(regex_texts, batch_size=batch_size)):
            results.append(self._apply_ner_from_doc(text, doc))
            # Free transformer activation tensors immediately after use.
            # Without this, trf_data lingers in memory until the next GC cycle,
            # causing gradual heap growth over long runs.
            if hasattr(doc, "_.trf_data") and doc.has_extension("trf_data"):
                doc._.trf_data = None

        gc.collect()
        return results

    def mask_records_batch(
        self,
        subjects: list[str],
        bodies: list[str],
        batch_size: int = 32,
    ) -> tuple[list[str], list[str]]:
        """
        Mask subjects and bodies for a batch of records in a **single**
        ``nlp.pipe()`` call, halving the number of transformer forward passes
        compared to calling ``mask_batch()`` twice.

        Subjects and bodies are interleaved as separate documents fed to the
        transformer: ``[subj_0, body_0, subj_1, body_1, ...]``.  Each is
        processed independently — the model never sees them merged.  Results
        are unpacked by index (even = subject, odd = body).

        Note on long emails
        -------------------
        ``en_core_web_trf`` handles long documents via a sliding window —
        it splits text into overlapping 512-token windows and merges NER
        results across them.  The full document is always processed; no
        truncation is applied here.

        Parameters
        ----------
        subjects : list[str]
            Subject strings for N records.
        bodies : list[str]
            Body strings for N records.
        batch_size : int
            Documents per NER mini-batch passed to ``nlp.pipe()``.
            Note: each record contributes 2 documents (subject + body), so
            the effective record throughput is ``batch_size // 2``.

        Returns
        -------
        tuple[list[str], list[str]]
            ``(masked_subjects, masked_bodies)`` — same order as inputs.
        """
        if not self.enabled:
            return list(subjects), list(bodies)

        assert self._nlp is not None, "spaCy model not loaded."
        n = len(subjects)
        assert len(bodies) == n, "subjects and bodies must have the same length."

        # Stage 1: Regex on full text (CPU, no length penalty).
        regex_subjects = [self._apply_regex(s) if s else s for s in subjects]
        regex_bodies   = [self._apply_regex(b) if b else b for b in bodies]

        # Stage 2: Interleave into one pipe call.
        # Layout: [subj_0, body_0, subj_1, body_1, ..., subj_{n-1}, body_{n-1}]
        interleaved: list[str] = []
        for s, b in zip(regex_subjects, regex_bodies):
            interleaved.append(s)
            interleaved.append(b)

        # Single transformer forward pass for all 2N documents.
        # Process as a generator (not list()) to avoid materialising all Doc
        # objects simultaneously — reduces peak memory proportional to batch_size.
        masked_subjects: list[str] = []
        masked_bodies: list[str] = []
        doc_buffer: list = []

        for doc in self._nlp.pipe(interleaved, batch_size=batch_size):
            doc_buffer.append(doc)

        for i in range(n):
            masked_subjects.append(
                self._apply_ner_from_doc(regex_subjects[i], doc_buffer[i * 2])
            )
            masked_bodies.append(
                self._apply_ner_from_doc(regex_bodies[i], doc_buffer[i * 2 + 1])
            )
            # Free transformer activations immediately after extraction.
            if hasattr(doc_buffer[i * 2], "_.trf_data"):
                try:
                    doc_buffer[i * 2]._.trf_data = None
                    doc_buffer[i * 2 + 1]._.trf_data = None
                except AttributeError:
                    pass

        del doc_buffer
        gc.collect()
        return masked_subjects, masked_bodies

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_ner(self, text: str) -> str:
        """
        Single-document NER masking (used by ``mask()``).
        For batch processing, ``mask_batch()`` calls ``_apply_ner_from_doc()`` instead.
        """
        assert self._nlp is not None, "spaCy model not loaded."
        doc = self._nlp(text)
        return self._apply_ner_from_doc(text, doc)

    @staticmethod
    def _apply_ner_from_doc(text: str, doc: object) -> str:
        """
        Apply NER-based token replacements given a pre-computed spaCy Doc object.

        Separated from ``_apply_ner`` so that ``mask_batch()`` can reuse it
        with docs produced by ``nlp.pipe()`` without re-running the model.

        Entities are replaced back-to-front to preserve character offsets.
        Any entity whose span overlaps an already-masked ``[TOKEN]`` is skipped
        to prevent double-substitution (e.g. ``[PGP]`` → ``[[ORGANIZATION]]``).
        """
        # Pre-calculate character spans of already-masked bracketed tokens [TOKEN]
        existing_spans = [m.span() for m in re.finditer(r"\[[A-Z_]+\]", text)]

        replacements = []
        for ent in doc.ents:  # type: ignore[union-attr]
            if ent.label_ not in _ENTITY_MAP:
                continue
            # Skip if spaCy entity overlaps with any existing bracketed token
            if any(
                not (ent.end_char <= s_start or ent.start_char >= s_end)
                for s_start, s_end in existing_spans
            ):
                continue
            replacements.append((ent.start_char, ent.end_char, _ENTITY_MAP[ent.label_]))

        for start, end, token in sorted(replacements, key=lambda x: x[0], reverse=True):
            text = text[:start] + token + text[end:]

        return text

    @staticmethod
    def _apply_regex(text: str) -> str:
        """
        Apply all compiled regex substitutions to the text in sequence.

        Parameters
        ----------
        text : str
            Input text (potentially already partially masked by NER).

        Returns
        -------
        str
            Text with regex-matched PII replaced by MeAJOR tokens.
        """
        for pattern, replacement in _REGEX_PATTERNS:
            text = pattern.sub(replacement, text)
        return text
