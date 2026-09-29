"""
parsers/trec.py
===============
Parser for the TREC 2007 Public Spam Corpus (trec07p).

Dataset Overview
----------------
The trec07p corpus contains 75,419 chronologically ordered real emails
(50,199 spam + 25,220 ham) collected over ~3 months in 2007 from a
University of Waterloo mail server.  It is the standard public benchmark
for spam filtering research and one of the five source datasets used in the
MeAJOR corpus (Mendes et al., 2025).

Directory Layout
----------------
::

    trec07p/
      data/
        inmail.1        <- individual RFC-2822 files (one per email)
        inmail.2
        ...
        inmail.75419
      full/
        index           <- ground-truth labels, one per line:
                           "spam ../data/inmail.1"
                           "ham  ../data/inmail.2"
      delay/            <- alternative label-timing variants (not used)
      partial/          <- alternative label-timing variants (not used)
      README.txt

Label Index Format
------------------
Each line of ``full/index`` is::

    <label> <relative_path>

where ``<label>`` is either ``spam`` or ``ham`` and ``<relative_path>``
is of the form ``../data/inmail.N``.  We extract the filename stem
(e.g. ``inmail.1``) and map it to an integer label (spam=1, ham=0).

Email File Format
-----------------
Each ``inmail.N`` file starts with an mbox-style ``From `` separator
line, followed by standard RFC-2822 headers and a body.  This is
identical to the SpamAssassin individual-file format.  The separator
line is stripped before feeding the text into ``email.message_from_string``.

Parallelism
-----------
With 75,419 files, sequential processing would take several minutes.
The same ``ProcessPoolExecutor`` + chunked batch pattern used by
``SpamAssassinParser`` is applied here.  Worker functions are defined at
module level to satisfy the Windows ``spawn`` pickling requirement.

Citation
--------
Cormack, G.V. (2007). TREC 2007 Spam Track Overview.
https://trec.nist.gov/pubs/trec16/papers/SPAM.OVERVIEW16.pdf
"""

from __future__ import annotations

import email
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from email.policy import compat32 as _compat32_policy
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

_SOURCE = "trec07"

# Label mapping from index file tokens.
_LABEL_MAP: dict[str, int] = {"spam": 1, "ham": 0}


# ---------------------------------------------------------------------------
# Module-level worker function (picklable for Windows spawn mode)
# ---------------------------------------------------------------------------


def _trec_worker(
    args: tuple[list[tuple[str, int]], int],
) -> tuple[list[dict], int, int]:
    """
    Process a batch of TREC-07p email files in an isolated worker process.

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
            # trec07p files begin with: "From addr@example.com  Sun Apr 8 ..."
            if raw_text.startswith("From "):
                newline_pos = raw_text.find("\n")
                if newline_pos != -1:
                    raw_text = raw_text[newline_pos + 1:]

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
# TrecParser
# ---------------------------------------------------------------------------


class TrecParser:
    """
    Ingests the TREC 2007 Public Spam Corpus (trec07p).

    Reads the ground-truth label index from ``full/index``, walks the
    ``data/`` directory, and submits batches to a parallel worker pool for
    processing.

    Parameters
    ----------
    source_config : dict
        Source configuration block (``config["sources"]["trec07"]``).
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
        Load the label index, enumerate email files, and yield cleaned records.

        Parameters
        ----------
        source_path : Path
            Absolute path to the trec07p root directory.

        Yields
        ------
        dict
            Raw pre-validation record with keys:
            ``{subject, body, label, source}``.
        """
        index_path = source_path / "full" / "index"
        data_dir = source_path / "data"

        if not index_path.exists():
            logger.error(
                "TREC07 > Label index not found at %s -- skipping.", index_path
            )
            return

        if not data_dir.exists():
            logger.error(
                "TREC07 > Data directory not found at %s -- skipping.", data_dir
            )
            return

        # ------------------------------------------------------------------
        # Parse the label index: "spam ../data/inmail.N" or "ham ../data/..."
        # ------------------------------------------------------------------
        label_map: dict[str, int] = {}  # stem -> int label

        with open(index_path, "r", encoding="ascii", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    continue
                label_token, path_token = parts
                label = _LABEL_MAP.get(label_token.lower())
                if label is None:
                    continue
                # path_token is like "../data/inmail.1" -- take just the filename.
                stem = Path(path_token).name  # "inmail.1"
                label_map[stem] = label

        logger.info(
            "TREC07 > Loaded %d label entries from index "
            "(spam=%d, ham=%d).",
            len(label_map),
            sum(1 for v in label_map.values() if v == 1),
            sum(1 for v in label_map.values() if v == 0),
        )

        if not label_map:
            logger.error("TREC07 > Label index is empty -- skipping.")
            return

        # ------------------------------------------------------------------
        # Enumerate data files and cross-reference labels.
        # ------------------------------------------------------------------
        file_label_pairs: list[tuple[str, int]] = []
        skipped_no_label = 0

        for email_file in data_dir.iterdir():
            if not email_file.is_file():
                continue
            stem = email_file.name  # "inmail.N"
            label = label_map.get(stem)
            if label is None:
                skipped_no_label += 1
                continue
            file_label_pairs.append((str(email_file), label))

        total_files = len(file_label_pairs)
        logger.info(
            "TREC07 > Found %d labelled files (skipped_no_label=%d).",
            total_files,
            skipped_no_label,
        )

        if total_files == 0:
            logger.warning("TREC07 > No labelled files found -- skipping.")
            return

        # ------------------------------------------------------------------
        # Parallel processing with chunked batches.
        # ------------------------------------------------------------------
        chunks: list[list[tuple[str, int]]] = [
            file_label_pairs[i: i + self.chunk_size]
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
                executor.submit(_trec_worker, arg): idx
                for idx, arg in enumerate(worker_args)
            }

            for future in as_completed(futures):
                completed_chunks += 1
                try:
                    records, dropped, errors = future.result()
                except Exception as exc:
                    logger.error(
                        "TREC07 > Chunk %d raised an unhandled exception: %s",
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
                    "TREC07 > Chunk %d/%d complete | "
                    "kept=%d dropped=%d errors=%d",
                    completed_chunks,
                    total_chunks,
                    total_kept,
                    total_dropped,
                    total_errors,
                )

        logger.info(
            "TREC07 > Complete. "
            "total_files=%d kept=%d dropped=%d errors=%d",
            total_files,
            total_kept,
            total_dropped,
            total_errors,
        )
