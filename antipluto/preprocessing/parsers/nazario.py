"""
parsers/nazario.py
==================
Parser for the Monkey.org Nazario Phishing Email Archive (mbox format).

Dataset Overview
----------------
The Nazario archive consists of 13 ``.mbox`` files stored directly under the
``nazario/`` directory:

  - ``phishing0.mbox`` through ``phishing3.mbox``  (classic archive)
  - ``private-phishing4.mbox``                      (extended archive)
  - ``phishing-2015.mbox`` through ``phishing-2022.mbox`` (yearly archives)

All records produced by this parser are labelled ``1`` (phishing).

Per the research design, archives from 2023 onwards are excluded to prevent
contamination from LLM-generated spam that began appearing in phishing
corpora from that period.

Why mailbox.mbox Instead of Manual Splitting
---------------------------------------------
The original ``raw_dataset_preprocess.py`` split mbox files using
``raw_text.split("\\nFrom ")``, which is not RFC-compliant.  That approach:

  1. Drops the first email in every file (no preceding ``\\n``).
  2. Breaks on multi-line ``From_`` continuation lines (rare but present).
  3. Fails silently when ``From `` appears inside a base64-encoded payload.

The Python standard library ``mailbox.mbox`` class implements the correct
RFC-4155 mbox parsing algorithm and handles all these edge cases.

Parallelism
-----------
The Nazario corpus is small enough (~13 files, ~18,000 emails total) that
the overhead of multiprocessing is not warranted.  Each ``.mbox`` file is
parsed sequentially within the main process.  If the corpus grows
substantially, the approach can be upgraded to per-file parallelism.
"""

from __future__ import annotations

import logging
import mailbox
from pathlib import Path
from typing import Iterator

from antipluto.preprocessing.cleaners.html_stripper import HTMLStripper
from antipluto.preprocessing.cleaners.header_slicer import HeaderSlicer
from antipluto.preprocessing.parsers.base import extract_email_fields

logger = logging.getLogger(__name__)

# Source identifier written into every record.
_SOURCE = "nazario"
_LABEL = 1  # All Nazario emails are phishing.

# Exclude yearly archives from 2023 onwards to avoid LLM-generated spam
# contamination, as documented in the research log (07/07/2026).
_MAX_YEAR = 2022


# ---------------------------------------------------------------------------
# NazarioParser
# ---------------------------------------------------------------------------


class NazarioParser:
    """
    Ingests all Nazario phishing ``.mbox`` archives from a directory.

    The parser iterates over every ``.mbox`` file present in the source
    directory.  Yearly archives (``phishing-YYYY.mbox``) are filtered to
    exclude 2023 and later.  Classic archives (``phishing0``–``phishing3``,
    ``private-phishing4``) are always included.

    Parameters
    ----------
    source_config : dict
        Source configuration block (``config["sources"]["nazario"]``).
    processing_config : dict
        Global processing settings.
    project_root : Path
        Absolute path to the thesis project root.
    """

    def __init__(
        self,
        source_config: dict,
        processing_config: dict,
        project_root: Path,
    ) -> None:
        self.min_body_length: int = processing_config.get("min_body_length", 50)

    def parse(self, source_path: Path) -> Iterator[dict]:
        """
        Parse all Nazario ``.mbox`` files and yield cleaned, labelled records.

        Parameters
        ----------
        source_path : Path
            Absolute path to the Nazario archive directory.

        Yields
        ------
        dict
            Raw pre-validation record with keys:
            ``{subject, body, label, source}``.
        """
        stripper = HTMLStripper()
        slicer = HeaderSlicer()

        mbox_files = sorted(
            p for p in source_path.iterdir()
            if p.suffix.lower() == ".mbox"
        )

        if not mbox_files:
            logger.warning(
                "Nazario › No .mbox files found at %s — skipping.", source_path
            )
            return

        # Apply the year exclusion filter.
        mbox_files = [f for f in mbox_files if self._is_included(f)]
        logger.info(
            "Nazario › Found %d .mbox files (after year filter).", len(mbox_files)
        )

        total_kept = 0
        total_dropped = 0
        total_errors = 0

        for mbox_path in mbox_files:
            kept, dropped, errors = yield from self._parse_mbox(
                mbox_path, stripper, slicer
            )
            total_kept += kept
            total_dropped += dropped
            total_errors += errors
            logger.info(
                "Nazario › %s: kept=%d dropped=%d errors=%d",
                mbox_path.name,
                kept,
                dropped,
                errors,
            )

        logger.info(
            "Nazario › Complete. total_kept=%d total_dropped=%d total_errors=%d",
            total_kept,
            total_dropped,
            total_errors,
        )

    def _parse_mbox(
        self,
        mbox_path: Path,
        stripper: HTMLStripper,
        slicer: HeaderSlicer,
    ) -> Iterator[dict]:
        """
        Parse a single ``.mbox`` file using ``mailbox.mbox`` and yield records.

        This method also returns summary counts via a ``(kept, dropped,
        errors)`` tuple yielded as a generator's ``return`` value, which
        the caller receives via ``yield from``.

        Parameters
        ----------
        mbox_path : Path
            Absolute path to the ``.mbox`` file.
        stripper : HTMLStripper
            Shared HTML stripper instance.
        slicer : HeaderSlicer
            Shared thread slicer instance.

        Yields
        ------
        dict
            Cleaned record dict.
        """
        kept = 0
        dropped = 0
        errors = 0

        try:
            mbox = mailbox.mbox(str(mbox_path), create=False)
        except Exception as exc:
            logger.error(
                "Nazario › Failed to open %s: %s", mbox_path.name, exc
            )
            return kept, dropped, errors

        try:
            # Iterate over message keys individually rather than using the
            # default iterator (`for msg in mbox`).  The stdlib mailbox.mbox
            # iterator decodes each message's `From_` separator line using
            # ASCII, which crashes on mbox files that contain non-ASCII bytes
            # in the separator (e.g. accented characters in sender addresses).
            # Iterating by key lets us catch and skip individual bad messages
            # without aborting the entire file.
            for key in mbox.keys():
                try:
                    msg = mbox[key]
                    result = extract_email_fields(
                        msg,
                        stripper=stripper,
                        slicer=slicer,
                        min_body_length=self.min_body_length,
                        label=_LABEL,
                        source=_SOURCE,
                    )
                    if result is not None:
                        kept += 1
                        yield result
                    else:
                        dropped += 1
                except UnicodeDecodeError:
                    # Non-ASCII bytes in the mbox From_ separator line.
                    # Skip this message and continue with the remainder.
                    errors += 1
                except Exception as exc:
                    logger.debug(
                        "Nazario › Error parsing message in %s: %s",
                        mbox_path.name,
                        exc,
                    )
                    errors += 1
        finally:
            mbox.close()


        return kept, dropped, errors

    @staticmethod
    def _is_included(mbox_path: Path) -> bool:
        """
        Return ``True`` if the archive should be included in the corpus.

        Classic archives (``phishing0``–``phishing3``, ``private-phishing4``)
        are always included.  Yearly archives (``phishing-YYYY.mbox``) are
        excluded for years > ``_MAX_YEAR`` to prevent LLM-generated phishing
        contamination.

        Parameters
        ----------
        mbox_path : Path
            Path to the candidate ``.mbox`` file.

        Returns
        -------
        bool
        """
        name = mbox_path.stem  # e.g. "phishing-2023" or "phishing2"

        # Attempt to extract a year from the filename.
        parts = name.split("-")
        if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 4:
            year = int(parts[1])
            if year > _MAX_YEAR:
                logger.info(
                    "Nazario › Excluding %s (year %d > %d).",
                    mbox_path.name,
                    year,
                    _MAX_YEAR,
                )
                return False

        return True
