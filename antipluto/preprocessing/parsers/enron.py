"""
parsers/enron.py
================
Parser for the CMU Enron Email Dataset (maildir format).

Dataset Overview
----------------
The Enron maildir contains approximately 517,000 individual email files
spread across 150 user directories (e.g. ``allen-p/``, ``lay-k/``), each
with subdirectories corresponding to mail folders (``inbox/``, ``sent/``,
``all_documents/``, etc.).  Every file is a bare RFC-2822 message with no
file extension.

All records produced by this parser are labelled ``0`` (benign / ham).

Parallelism Strategy
--------------------
Processing 500k+ files sequentially would take an unacceptable amount of
time.  This parser divides the full file list into chunks of configurable
size and submits each chunk to a ``ProcessPoolExecutor``.  The worker
function ``_enron_worker`` is defined at **module level** (not as a lambda
or closure) so it is safely picklable on Windows, where the default
multiprocessing start method is "spawn" rather than "fork".

The ``if __name__ == "__main__":`` guard is enforced at the CLI entry point
(``main.py``) — users must invoke the framework through that entry point to
prevent recursive worker spawning on Windows.
"""

from __future__ import annotations

import email
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from email.policy import default as _email_policy_default
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# Source identifier written into every record.
_SOURCE = "enron"
_LABEL = 0  # All Enron emails are benign.


# ---------------------------------------------------------------------------
# Module-level worker function (must be at module scope for pickling)
# ---------------------------------------------------------------------------


def _enron_worker(
    args: tuple[list[str], int],
) -> tuple[list[dict], int, int]:
    """
    Process a chunk of Enron email file paths in an isolated worker process.

    This function is intentionally self-contained: it imports its
    dependencies internally so that it can be safely spawned as a new
    process on Windows without carrying over the parent process's state.

    Parameters
    ----------
    args : tuple[list[str], int]
        A tuple of ``(file_paths, min_body_length)`` where ``file_paths``
        is a list of absolute path strings and ``min_body_length`` is the
        minimum character threshold for the cleaned body.

    Returns
    -------
    tuple[list[dict], int, int]
        ``(records, dropped_count, error_count)``
    """
    # Local imports — executed inside the spawned worker process.
    from antipluto.preprocessing.cleaners.html_stripper import HTMLStripper
    from antipluto.preprocessing.cleaners.header_slicer import HeaderSlicer
    from antipluto.preprocessing.parsers.base import extract_email_fields

    file_paths, min_body_length = args
    stripper = HTMLStripper()
    slicer = HeaderSlicer()

    records: list[dict] = []
    dropped: int = 0
    errors: int = 0

    for path_str in file_paths:
        try:
            with open(path_str, "r", encoding="utf-8", errors="ignore") as fh:
                msg = email.message_from_file(fh, policy=_email_policy_default)

            result = extract_email_fields(
                msg,
                stripper=stripper,
                slicer=slicer,
                min_body_length=min_body_length,
                label=_LABEL,
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
# EnronParser
# ---------------------------------------------------------------------------


class EnronParser:
    """
    Ingests the CMU Enron Email Dataset from a maildir directory tree.

    This parser is intentionally decoupled from the ``BaseEmailParser`` ABC
    so that the worker function (``_enron_worker``) can remain at module
    scope.  It adheres to the same ``parse(source_path) -> Iterator[dict]``
    interface contract used by the pipeline orchestrator.

    Parameters
    ----------
    source_config : dict
        Source configuration block (``config["sources"]["enron"]``).
    processing_config : dict
        Global processing settings (chunk size, max workers, etc.).
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
        Walk the Enron maildir and yield cleaned, labelled records.

        The walk is parallelised: files are batched into chunks and
        submitted to a ``ProcessPoolExecutor``.  Records are yielded
        as soon as each chunk future completes (not in submission order)
        to maximise throughput.

        Parameters
        ----------
        source_path : Path
            Absolute path to the Enron ``maildir/`` root.

        Yields
        ------
        dict
            Raw pre-validation record with keys:
            ``{subject, body, label, source}``.
        """
        logger.info("Enron › Scanning maildir: %s", source_path)

        all_files: list[str] = [
            os.path.join(root, fname)
            for root, _, fnames in os.walk(source_path)
            for fname in fnames
        ]
        total_files = len(all_files)
        logger.info("Enron › Found %d files across maildir.", total_files)

        if total_files == 0:
            logger.warning("Enron › No files found at %s — skipping.", source_path)
            return

        # Partition into chunks.
        chunks: list[list[str]] = [
            all_files[i : i + self.chunk_size]
            for i in range(0, total_files, self.chunk_size)
        ]
        total_chunks = len(chunks)
        logger.info(
            "Enron › Dispatching %d chunks (~%d files/chunk) to workers.",
            total_chunks,
            self.chunk_size,
        )

        # Package arguments for the worker (must be picklable).
        worker_args = [(chunk, self.min_body_length) for chunk in chunks]

        total_kept = 0
        total_dropped = 0
        total_errors = 0
        completed_chunks = 0

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(_enron_worker, arg): idx
                for idx, arg in enumerate(worker_args)
            }

            for future in as_completed(futures):
                completed_chunks += 1
                try:
                    records, dropped, errors = future.result()
                except Exception as exc:
                    logger.error(
                        "Enron › Chunk %d raised an unhandled exception: %s",
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

                # Log progress at 10% intervals.
                if completed_chunks % max(1, total_chunks // 10) == 0 or \
                        completed_chunks == total_chunks:
                    pct = (completed_chunks / total_chunks) * 100
                    logger.info(
                        "Enron › Progress: %d/%d chunks (%.1f%%) | "
                        "kept=%d dropped=%d errors=%d",
                        completed_chunks,
                        total_chunks,
                        pct,
                        total_kept,
                        total_dropped,
                        total_errors,
                    )

        logger.info(
            "Enron › Complete. total_files=%d kept=%d dropped=%d errors=%d",
            total_files,
            total_kept,
            total_dropped,
            total_errors,
        )
