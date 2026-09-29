"""
utils/io.py
===========
Lightweight JSONL read/write utilities.

These helpers are intentionally thin wrappers around the standard ``json``
library — they use ``ensure_ascii=False`` to preserve non-ASCII characters
in email bodies and avoid the escaped-slash artefact (``\\/``) produced by
the ``pandas`` JSON serialiser (documented in RESEARCH_LOG.md, 08/07/2026).

Functions
---------
- ``write_jsonl(records, path, mode)`` — Write an iterable of dicts to JSONL.
- ``read_jsonl(path)`` — Lazily read a JSONL file, yielding one dict per line.
- ``count_jsonl(path)`` — Count lines in a JSONL file without loading it.
- ``filter_jsonl(path, **kwargs)`` — Filter JSONL records by field equality.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


def write_jsonl(
    records: Iterator[dict] | list[dict],
    path: Path,
    mode: str = "w",
) -> int:
    """
    Write an iterable of dictionaries to a JSONL file.

    Each record is serialised as a single line of JSON with
    ``ensure_ascii=False`` to preserve the original character encoding of
    email bodies.

    Parameters
    ----------
    records : iterable of dict
        Records to write.  May be a generator (memory-efficient for large
        corpora) or a list.
    path : Path
        Output file path.  Parent directories are created if absent.
    mode : str
        File open mode.  Use ``"w"`` to overwrite (default) or ``"a"``
        to append to an existing file.

    Returns
    -------
    int
        Number of records written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(path, mode, encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    logger.debug("io › Wrote %d records to %s (mode='%s').", written, path, mode)
    return written


def read_jsonl(path: Path) -> Iterator[dict]:
    """
    Lazily read a JSONL file and yield one dictionary per line.

    Empty lines are silently skipped.  Lines that fail JSON parsing are
    logged at WARNING level and skipped.

    Parameters
    ----------
    path : Path
        Path to the ``.jsonl`` file.

    Yields
    ------
    dict
        Parsed record dictionary.
    """
    with open(path, "r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "io › Skipping malformed JSON on line %d of %s: %s",
                    line_num, path, exc,
                )


def count_jsonl(path: Path) -> int:
    """
    Count the number of non-empty lines in a JSONL file.

    This reads the file line-by-line without JSON-parsing each record,
    making it suitable for quickly inspecting large output files.

    Parameters
    ----------
    path : Path
        Path to the ``.jsonl`` file.

    Returns
    -------
    int
        Number of non-empty lines (i.e. records).
    """
    if not path.exists():
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def filter_jsonl(path: Path, **field_filters: object) -> Iterator[dict]:
    """
    Filter records from a JSONL file by field equality.

    Parameters
    ----------
    path : Path
        Path to the ``.jsonl`` file.
    **field_filters
        Keyword arguments of the form ``field=value``.
        Records where all specified fields match the given values are yielded.

    Examples
    --------
    Retrieve only Enron records::

        from email_preprocessor.utils.io import filter_jsonl
        enron_records = list(filter_jsonl(benign_path, source="enron"))

    Retrieve SpamAssassin ham records::

        sa_ham = list(filter_jsonl(benign_path, source="spamassassin", label=0))

    Yields
    ------
    dict
        Records matching all filter criteria.
    """
    for record in read_jsonl(path):
        if all(record.get(k) == v for k, v in field_filters.items()):
            yield record
