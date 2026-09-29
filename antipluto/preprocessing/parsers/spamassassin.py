"""
parsers/spamassassin.py
=======================
Parser for the Apache SpamAssassin Public Corpus.

Dataset Overview
----------------
The SpamAssassin corpus is organised as a collection of archive directories
under a single root (``spamassasin/``).  After extraction, the layout is::

    spamassasin/
    ├── 20021010_spam.tar/
    │   └── spam/
    │       ├── 0000.<md5hash>
    │       ├── 0001.<md5hash>
    │       └── ...
    ├── 20021010_easy_ham.tar/
    │   └── easy_ham/
    │       ├── 0001.<md5hash>
    │       └── ...
    ├── 20030228_hard_ham.tar/
    │   └── hard_ham/
    │       └── ...
    └── ... (9 archive directories total)

Each email file is a bare RFC-2822 text file prefixed with a single
mbox-style ``From <address>  <date>`` separator line.  Files have no
extension and are named ``NNNN.<md5hash>``.

Label Assignment
----------------
The parent directory name determines the ground-truth label:

- ``spam``     → ``label = 1`` (phishing / spam)
- ``easy_ham`` → ``label = 0`` (benign)
- ``hard_ham`` → ``label = 0`` (benign)

Deduplication
-------------
Inspection of the extracted files reveals that several archives contain
exact duplicates (files sharing an identical MD5 hash in their filename,
e.g. ``0035.8e582263...`` and ``0036.8e582263...``).  The parser extracts
the MD5 suffix from each filename and skips files whose hash has already
been processed.

Parallelism
-----------
The corpus is small (~6,000 files total).  Files are processed in a single
parallel chunk rather than the multi-chunk strategy used for Enron.  A
``ProcessPoolExecutor`` is still used for consistency.
"""

from __future__ import annotations

import email
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from email.policy import compat32 as _compat32_policy
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

_SOURCE = "spamassassin"

# Leaf directory names that indicate the label for their contents.
_SPAM_DIRS: frozenset[str] = frozenset({"spam"})
_HAM_DIRS: frozenset[str] = frozenset({"easy_ham", "hard_ham"})


# ---------------------------------------------------------------------------
# Module-level worker function (picklable for Windows spawn mode)
# ---------------------------------------------------------------------------


def _spamassassin_worker(
    args: tuple[list[tuple[str, int]], int],
) -> tuple[list[dict], int, int]:
    """
    Process a batch of SpamAssassin email files in an isolated worker process.

    Parameters
    ----------
    args : tuple
        ``(file_label_pairs, min_body_length)`` where ``file_label_pairs``
        is a list of ``(absolute_path_str, label)`` tuples.

    Returns
    -------
    tuple[list[dict], int, int]
        ``(records, dropped_count, error_count)``
    """
    from antipluto.preprocessing.cleaners.html_stripper import HTMLStripper
    from antipluto.preprocessing.cleaners.header_slicer import HeaderSlicer
    from antipluto.preprocessing.parsers.base import extract_email_fields

    file_label_pairs, min_body_length = args
    stripper = HTMLStripper()
    slicer = HeaderSlicer()

    records: list[dict] = []
    dropped: int = 0
    errors: int = 0

    for path_str, label in file_label_pairs:
        try:
            raw_text = Path(path_str).read_text(encoding="utf-8", errors="ignore")

            # Strip the mbox "From " separator line before parsing.
            # SpamAssassin files begin with: "From addr@example.com  Thu Jan 1 ..."
            if raw_text.startswith("From "):
                newline_pos = raw_text.find("\n")
                if newline_pos != -1:
                    raw_text = raw_text[newline_pos + 1 :]

            msg = email.message_from_string(raw_text, policy=_compat32_policy)

            result = extract_email_fields(
                msg,
                stripper=stripper,
                slicer=slicer,
                min_body_length=min_body_length,
                label=label,
                source=_SOURCE,
            )

            if result is not None:
                records.append(result)
            else:
                dropped += 1

        except Exception:
            errors += 1

    return records, dropped, errors


# ---------------------------------------------------------------------------
# SpamAssassinParser
# ---------------------------------------------------------------------------


class SpamAssassinParser:
    """
    Ingests the Apache SpamAssassin Public Corpus from an extracted archive root.

    The parser recursively walks the corpus root, classifies each file by
    its parent directory name (``spam`` → phishing, ``easy_ham`` /
    ``hard_ham`` → benign), deduplicates by MD5 hash, and submits batches
    to a worker pool for parallel processing.

    Parameters
    ----------
    source_config : dict
        Source configuration block (``config["sources"]["spamassassin"]``).
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
        self.chunk_size: int = processing_config.get("chunk_size", 1000)
        self.max_workers: int | None = processing_config.get("max_workers")

    def parse(self, source_path: Path) -> Iterator[dict]:
        """
        Walk the SpamAssassin corpus root, classify files, and yield records.

        Parameters
        ----------
        source_path : Path
            Absolute path to the SpamAssassin root directory
            (e.g. ``datasets_raw/spamassasin/``).

        Yields
        ------
        dict
            Raw pre-validation record with keys:
            ``{subject, body, label, source}``.
        """
        logger.info("SpamAssassin › Scanning corpus root: %s", source_path)

        # Enumerate and classify all email files; deduplicate by MD5 hash.
        file_label_pairs: list[tuple[str, int]] = []
        seen_hashes: set[str] = set()
        skipped_unknown = 0
        skipped_dupes = 0

        for email_file in source_path.rglob("*"):
            if not email_file.is_file():
                continue

            parent_name = email_file.parent.name

            if parent_name in _SPAM_DIRS:
                label = 1
            elif parent_name in _HAM_DIRS:
                label = 0
            else:
                # File is in an unrecognised directory (e.g. a stray
                # nested archive dir) — skip safely.
                skipped_unknown += 1
                continue

            # Deduplicate on MD5 hash extracted from filename.
            # Filename format: "NNNN.<md5hash>" or just "<md5hash>".
            stem = email_file.stem
            md5_hash = stem.split(".")[-1] if "." in stem else stem
            if md5_hash in seen_hashes:
                skipped_dupes += 1
                continue
            seen_hashes.add(md5_hash)

            file_label_pairs.append((str(email_file), label))

        total_files = len(file_label_pairs)
        logger.info(
            "SpamAssassin › Found %d unique files "
            "(skipped_dupes=%d, skipped_unknown_dirs=%d).",
            total_files,
            skipped_dupes,
            skipped_unknown,
        )

        if total_files == 0:
            logger.warning(
                "SpamAssassin › No files found at %s — skipping.", source_path
            )
            return

        # Partition into chunks for parallel processing.
        chunks: list[list[tuple[str, int]]] = [
            file_label_pairs[i : i + self.chunk_size]
            for i in range(0, total_files, self.chunk_size)
        ]
        total_chunks = len(chunks)
        worker_args = [(chunk, self.min_body_length) for chunk in chunks]

        total_kept = 0
        total_dropped = 0
        total_errors = 0
        completed_chunks = 0

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(_spamassassin_worker, arg): idx
                for idx, arg in enumerate(worker_args)
            }

            for future in as_completed(futures):
                completed_chunks += 1
                try:
                    records, dropped, errors = future.result()
                except Exception as exc:
                    logger.error(
                        "SpamAssassin › Chunk %d raised an unhandled exception: %s",
                        futures[future],
                        exc,
                    )
                    total_errors += 1
                    continue

                total_kept += len(records)
                total_dropped += dropped
                total_errors += errors

                for record in records:
                    yield record

                logger.info(
                    "SpamAssassin › Chunk %d/%d complete | "
                    "kept=%d dropped=%d errors=%d",
                    completed_chunks,
                    total_chunks,
                    total_kept,
                    total_dropped,
                    total_errors,
                )

        logger.info(
            "SpamAssassin › Complete. "
            "total_files=%d kept=%d dropped=%d errors=%d",
            total_files,
            total_kept,
            total_dropped,
            total_errors,
        )
